//! GPU population inference through wgpu (Vulkan, DX12 or Metal).
//!
//! Every organism keeps its *own* weights. Genomes that share a catalog
//! topology live in persistent per-topology GPU slabs, one storage buffer per
//! affine layer. Within a slot a layer is stored transposed, `[in][out]`,
//! followed by its `out` biases, so the threads of a workgroup (one thread per
//! output neuron) read consecutive addresses. A genome is uploaded once, on
//! its first evaluation, and its slot is released after it stops being
//! evaluated.
//!
//! One dispatch per topology evaluates every due organism of that topology:
//! one workgroup per organism runs the whole MLP (tanh hidden layers, linear
//! output) with activations in workgroup memory, and the 36 logits per
//! organism are read back.
//!
//! This backend is outside the Franken stack: FrankenTorch has no CUDA or
//! Vulkan path, and the plan's Metal route needs macOS. Floating-point
//! accumulation order differs from fl-core's scalar reference, so results
//! agree with it to a tolerance, not bitwise; a run that uses this backend is
//! its own determinism class (plan §10.3, §10.7).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::mpsc;

use fl_core::nn::{layer_dims, CATALOG, INPUT_WIDTH, OUTPUT_WIDTH};
use fl_core::field::Field;
use fl_core::sources::Source;
use fl_core::{Config, ControllerBackend, FieldAccounting, FieldBackend, InferenceJob, SenseInferBackend, SenseJob, SenseOutput};

const SHADER: &str = include_str!("mlp.wgsl");
const SENSE_SHADER: &str = include_str!("sense.wgsl");
const PYRAMID_SHADER: &str = include_str!("pyramid.wgsl");
const FIELD_SHADER: &str = include_str!("field.wgsl");
const SOURCE_STRIDE: usize = 44;
/// Floats per slot in the resident traits buffer: 33 motion sensitivities + padding.
const TRAITS_STRIDE: usize = 36;
const JOB_STRIDE: usize = 12;
const SENSE_OUT_STRIDE: usize = 40;
const MAX_LEVELS: usize = 8;
const EVICT_AFTER_CALLS: u64 = 64;
const MAX_LAYERS: usize = 3;

/// What was found when the GPU was opened.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub max_storage_binding_bytes: u64,
    pub max_buffer_bytes: u64,
}

/// Where the time of the most recent `evaluate` call went.
#[derive(Debug, Default, Clone, Copy)]
pub struct EvalTiming {
    /// CPU work before submission: slot lookup, genome uploads, input packing, buffer writes.
    pub host_prep_ms: f64,
    /// Submit, wait for completion and read the logits back (includes the kernel).
    pub gpu_roundtrip_ms: f64,
    /// Compute-pass time from GPU timestamps, when the adapter supports them.
    pub kernel_ms: Option<f64>,
    /// Packing inputs per dispatch, scattering logits back, and slot eviction.
    pub bookkeeping_ms: f64,
    /// The whole `evaluate` call.
    pub total_ms: f64,
    /// Sensor-pyramid build on the GPU (timestamps), fused path only.
    pub pyramid_ms: Option<f64>,
    pub dispatches: u32,
}

/// Where the time of the most recent field step went.
#[derive(Debug, Default, Clone, Copy)]
pub struct FieldTiming {
    /// Uploading the committed field and source parameters, recording commands.
    pub host_ms: f64,
    /// Submit, wait, read the observation field and accounting back (includes kernels).
    pub roundtrip_ms: f64,
    /// Field kernels (diffusion, decay, sources, tiny clearing) from GPU timestamps.
    pub kernel_ms: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GpuStats {
    pub calls: u64,
    pub evaluations: u64,
    pub genome_uploads: u64,
    pub evictions: u64,
    pub dispatches: u64,
    pub slab_growths: u64,
    pub last: EvalTiming,
    pub field_last: FieldTiming,
}

struct Slab {
    dims: Vec<(usize, usize)>,
    /// Floats per slot for each layer: `in*out + out`.
    stride: Vec<usize>,
    /// Weight pages: each page holds up to `page_slots` organisms in one buffer per
    /// layer, so no buffer exceeds the adapter's storage-binding limit (sharding).
    pages: Vec<Vec<wgpu::Buffer>>,
    /// Per page: resident genome traits (motion sensitivities), `TRAITS_STRIDE` floats per slot.
    traits: Vec<wgpu::Buffer>,
    page_slots: usize,
    /// Capacity of the last (growing) page; earlier pages are full-size.
    last_cap: usize,
    owner: Vec<Option<u64>>,
    last_used: Vec<u64>,
    free: Vec<usize>,
}

impl Slab {
    fn cap(&self) -> usize {
        if self.pages.is_empty() { 0 } else { (self.pages.len() - 1) * self.page_slots + self.last_cap }
    }
}

struct SensePipelines {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    pyr_level0: wgpu::ComputePipeline,
    pyr_down: wgpu::ComputePipeline,
    pyr_layout: wgpu::BindGroupLayout,
}

struct FieldPipelines {
    layout: wgpu::BindGroupLayout,
    diffuse: wgpu::ComputePipeline,
    decay: wgpu::ComputePipeline,
    source_norm: wgpu::ComputePipeline,
    inject: wgpu::ComputePipeline,
    tiny: wgpu::ComputePipeline,
}

/// Device copies of the field and its density pyramid, sized for one field geometry.
struct FieldGpu {
    key: (usize, usize, usize),
    fe: wgpu::Buffer,
    fm: wgpu::Buffer,
    // Field-step resources.
    fe2: wgpu::Buffer,
    fm2: wgpu::Buffer,
    funiform: wgpu::Buffer,
    acct: wgpu::Buffer,
    fstaging: wgpu::Buffer,
    src: wgpu::Buffer,
    src_cap: usize,
    bind_ab: wgpu::BindGroup,
    bind_ba: wgpu::BindGroup,
    bind_inplace: wgpu::BindGroup,
    cell_grid: (u32, u32),
    plane_grid: (u32, u32),
    pyr: wgpu::Buffer,
    /// Per level: (gx, gy, offset cells, hx, hy).
    levels: Vec<(u32, u32, u32, f32, f32)>,
    binds: Vec<wgpu::BindGroup>,
    grids: Vec<(u32, u32)>,
    s0: f32,
}

struct Timestamps {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
}

struct Io {
    cap: usize,
    slots: wgpu::Buffer,
    inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    staging: wgpu::Buffer,
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    dummy: wgpu::Buffer,
    slabs: Vec<Slab>,
    slot_of: HashMap<u64, (usize, usize)>,
    io: Option<Io>,
    call: u64,
    timestamps: Option<Timestamps>,
    pyramid_timestamps: Option<Timestamps>,
    input_scratch: Vec<f32>,
    sense: SensePipelines,
    field_pipes: FieldPipelines,
    field_timestamps: Option<Timestamps>,
    /// True while the device copy of the field equals the observation field just
    /// returned by `transport_and_sources` (lets fused sensing skip its upload).
    field_resident: bool,
    field_gpu: Option<FieldGpu>,
    sense_io: Option<Io>,
    pub info: GpuInfo,
    pub stats: GpuStats,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl GpuBackend {
    /// Open the highest-performance adapter (`WGPU_BACKEND=vulkan|dx12|metal` overrides the choice).
    pub fn new() -> Result<GpuBackend, String> {
        pollster::block_on(Self::open())
    }

    async fn open() -> Result<GpuBackend, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None, apply_limit_buckets: false })
            .await
            .map_err(|e| format!("no GPU adapter found: {e}"))?;
        let ai = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        // Weight slabs are large: ask for everything the adapter offers here.
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        let ts_ok = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if ts_ok { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("frankenlife"), required_features, required_limits: limits.clone(), ..Default::default() })
            .await
            .map_err(|e| format!("could not open GPU device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mlp"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mlp"),
            entries: &[uniform_entry, storage_entry(1, true), storage_entry(2, true), storage_entry(3, false), storage_entry(4, true), storage_entry(5, true), storage_entry(6, true)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mlp"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mlp"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dims"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor { label: Some("dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let slabs = (0..CATALOG.len())
            .map(|t| {
                let dims = layer_dims(CATALOG[t]);
                let stride: Vec<usize> = dims.iter().map(|&(i, o)| i * o + o).collect();
                let widest = *stride.iter().max().expect("layers") as u64 * 4;
                let page_slots = ((limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) / widest).clamp(1, 1 << 20) as usize;
                Slab { dims, stride, pages: Vec::new(), traits: Vec::new(), page_slots, last_cap: 0, owner: Vec::new(), last_used: Vec::new(), free: Vec::new() }
            })
            .collect();
        let uentry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let sense_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sense"), source: wgpu::ShaderSource::Wgsl(SENSE_SHADER.into()) });
        let sense_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sense"),
            entries: &[uentry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, true), storage_entry(4, true), storage_entry(5, true), storage_entry(6, true), storage_entry(7, false)],
        });
        let spl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sense"), bind_group_layouts: &[Some(&sense_layout)], immediate_size: 0 });
        let sense_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sense"),
            layout: Some(&spl),
            module: &sense_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pyr_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pyramid"), source: wgpu::ShaderSource::Wgsl(PYRAMID_SHADER.into()) });
        let pyr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pyramid"),
            entries: &[uentry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
        });
        let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pyramid"), bind_group_layouts: &[Some(&pyr_layout)], immediate_size: 0 });
        let mkp = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&ppl),
                module: &pyr_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sense = SensePipelines {
            pipeline: sense_pipeline,
            layout: sense_layout,
            uniform: device.create_buffer(&wgpu::BufferDescriptor { label: Some("sense-dims"), size: 304, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            pyr_level0: mkp("level0"),
            pyr_down: mkp("down"),
            pyr_layout,
        };
        let mk_ts = || Timestamps {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: 2 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts-resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            readback: device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts-read"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            period_ns: queue.get_timestamp_period() as f64,
        };
        let pyramid_timestamps = ts_ok.then(mk_ts);
        let field_timestamps = ts_ok.then(mk_ts);
        let field_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("field"), source: wgpu::ShaderSource::Wgsl(FIELD_SHADER.into()) });
        let field_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field"),
            entries: &[uentry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false), storage_entry(4, false), storage_entry(5, false), storage_entry(6, false)],
        });
        let fpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("field"), bind_group_layouts: &[Some(&field_layout)], immediate_size: 0 });
        let mkf = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&fpl),
                module: &field_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let field_pipes = FieldPipelines { diffuse: mkf("diffuse"), decay: mkf("decay"), source_norm: mkf("source_norm"), inject: mkf("inject"), tiny: mkf("tiny"), layout: field_layout };
        let timestamps = ts_ok.then(|| Timestamps {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("kernel"), ty: wgpu::QueryType::Timestamp, count: 2 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts-resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            readback: device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts-read"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            period_ns: queue.get_timestamp_period() as f64,
        });
        let info = GpuInfo {
            name: ai.name.clone(),
            backend: format!("{:?}", ai.backend),
            device_type: format!("{:?}", ai.device_type),
            driver: format!("{} {}", ai.driver, ai.driver_info).trim().to_string(),
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
            max_buffer_bytes: limits.max_buffer_size,
        };
        Ok(GpuBackend { device, queue, pipeline, layout, uniform, dummy, slabs, slot_of: HashMap::new(), io: None, call: 0, timestamps, pyramid_timestamps, input_scratch: Vec::new(), sense, field_pipes, field_timestamps, field_resident: false, field_gpu: None, sense_io: None, info, stats: GpuStats::default() })
    }

    pub fn capability(&self) -> String {
        format!(
            "wgpu {} on {} ({}, {}); persistent per-topology weight slabs; one fused-MLP dispatch per topology; independent weights per organism; tolerance-equivalent to the fl-core scalar reference, not bitwise",
            self.info.backend, self.info.name, self.info.device_type, self.info.driver
        )
    }

    /// Bytes of GPU weight slabs currently allocated (capacity, including free slots).
    pub fn resident_bytes(&self) -> u64 {
        self.slabs.iter().map(|s| s.stride.iter().map(|&st| (s.cap() * st * 4) as u64).sum::<u64>()).sum()
    }

    /// Weight pages currently allocated across all topologies.
    pub fn pages(&self) -> usize {
        self.slabs.iter().map(|s| s.pages.len()).sum()
    }

    /// Cap organisms per weight page (testing / tuning). Only before any genome is uploaded.
    pub fn set_max_page_slots(&mut self, slots: usize) -> Result<(), String> {
        if self.slabs.iter().any(|s| !s.pages.is_empty()) {
            return Err("page size can only change before the first upload".into());
        }
        for s in &mut self.slabs {
            s.page_slots = s.page_slots.min(slots.max(1));
        }
        Ok(())
    }

    fn grow(&mut self, t: usize) -> Result<(), String> {
        let slab = &mut self.slabs[t];
        let new_page = slab.pages.is_empty() || slab.last_cap >= slab.page_slots;
        let old_cap = if new_page { 0 } else { slab.last_cap };
        let new_cap = if new_page { slab.page_slots.min(64) } else { (old_cap * 2).min(slab.page_slots) };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grow") });
        let mut fresh = Vec::with_capacity(slab.stride.len());
        for (l, &st) in slab.stride.iter().enumerate() {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("weights"),
                size: (new_cap * st * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if old_cap > 0 {
                enc.copy_buffer_to_buffer(&slab.pages.last().expect("page")[l], 0, &b, 0, (old_cap * st * 4) as u64);
            }
            fresh.push(b);
        }
        let traits = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traits"),
            size: (new_cap * TRAITS_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if old_cap > 0 {
            enc.copy_buffer_to_buffer(slab.traits.last().expect("traits page"), 0, &traits, 0, (old_cap * TRAITS_STRIDE * 4) as u64);
        }
        self.queue.submit([enc.finish()]);
        let before = slab.cap();
        if new_page {
            slab.pages.push(fresh);
            slab.traits.push(traits);
        } else {
            *slab.pages.last_mut().expect("page") = fresh;
            *slab.traits.last_mut().expect("traits page") = traits;
        }
        slab.last_cap = new_cap;
        let after = slab.cap();
        slab.free.extend((before..after).rev());
        slab.owner.resize(after, None);
        slab.last_used.resize(after, 0);
        self.stats.slab_growths += 1;
        Ok(())
    }

    fn upload(&mut self, t: usize, slot: usize, params: &[f32], motion: &[f32]) {
        let slab = &self.slabs[t];
        {
            let (page, local) = (slot / slab.page_slots, slot % slab.page_slots);
            let mut tr = [0.0f32; TRAITS_STRIDE];
            tr[..motion.len()].copy_from_slice(motion);
            self.queue.write_buffer(&slab.traits[page], (local * TRAITS_STRIDE * 4) as u64, f32_bytes(&tr));
        }
        let mut off = 0;
        for (l, &(i, o)) in slab.dims.iter().enumerate() {
            let mut tw = vec![0.0f32; i * o + o];
            for oo in 0..o {
                for ii in 0..i {
                    tw[ii * o + oo] = params[off + oo * i + ii];
                }
            }
            tw[i * o..].copy_from_slice(&params[off + i * o..off + i * o + o]);
            off += i * o + o;
            let (page, local) = (slot / slab.page_slots, slot % slab.page_slots);
            self.queue.write_buffer(&slab.pages[page][l], (local * slab.stride[l] * 4) as u64, f32_bytes(&tw));
        }
    }

    fn ensure_io(&mut self, n: usize) {
        if self.io.as_ref().is_some_and(|io| io.cap >= n) {
            return;
        }
        let cap = n.next_power_of_two().max(256);
        let mk = |label, size: usize, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        use wgpu::BufferUsages as U;
        self.io = Some(Io {
            cap,
            slots: mk("slots", cap * 4, U::STORAGE | U::COPY_DST),
            inputs: mk("inputs", cap * INPUT_WIDTH * 4, U::STORAGE | U::COPY_DST),
            outputs: mk("outputs", cap * OUTPUT_WIDTH * 4, U::STORAGE | U::COPY_SRC),
            staging: mk("staging", cap * OUTPUT_WIDTH * 4, U::MAP_READ | U::COPY_DST),
        });
    }

    fn run_topology(&mut self, t: usize, page: usize, slots: &[u32], inputs: &[f32]) -> Result<Vec<f32>, String> {
        let prep = std::time::Instant::now();
        let n = slots.len();
        self.ensure_io(n);
        let io = self.io.as_ref().expect("io allocated");
        let slab = &self.slabs[t];
        let grid_x = (n as u32).min(65_535);
        let grid_y = (n as u32).div_ceil(grid_x);
        let mut d = [0u32; 8];
        d[0] = n as u32;
        d[1] = slab.dims.len() as u32;
        d[2] = slab.dims[0].0 as u32;
        for (l, &(_, o)) in slab.dims.iter().enumerate() {
            d[3 + l] = o as u32;
        }
        d[6] = grid_x;
        let ub: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&self.uniform, 0, &ub);
        self.queue.write_buffer(&io.slots, 0, bytemuck::cast_slice(slots));
        self.queue.write_buffer(&io.inputs, 0, f32_bytes(inputs));
        let w = |l: usize| slab.pages[page].get(l).unwrap_or(&self.dummy).as_entire_binding();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mlp"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: io.slots.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: io.inputs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: io.outputs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: w(0) },
                wgpu::BindGroupEntry { binding: 5, resource: w(1) },
                wgpu::BindGroupEntry { binding: 6, resource: if slab.dims.len() == MAX_LAYERS { w(2) } else { self.dummy.as_entire_binding() } },
            ],
        });
        let bytes = (n * OUTPUT_WIDTH * 4) as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mlp") });
        {
            let tw = self.timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites { query_set: &ts.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mlp"), timestamp_writes: tw });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(&io.outputs, 0, &io.staging, 0, bytes);
        if let Some(ts) = &self.timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
        let round = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.stats.dispatches += 1;
        self.stats.last.dispatches += 1;
        let slice = io.staging.slice(0..bytes);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let ts_rx = self.timestamps.as_ref().map(|ts| {
            let (ttx, trx) = mpsc::channel();
            ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = ttx.send(r);
            });
            trx
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        if let (Some(ts), Some(trx)) = (&self.timestamps, ts_rx) {
            if matches!(trx.recv(), Ok(Ok(()))) {
                let k = {
                    let v = ts.readback.slice(..).get_mapped_range().map_err(|e| format!("timestamp map: {e:?}"))?;
                    let t0 = u64::from_le_bytes(v[0..8].try_into().expect("8 bytes"));
                    let t1 = u64::from_le_bytes(v[8..16].try_into().expect("8 bytes"));
                    t1.saturating_sub(t0) as f64 * ts.period_ns / 1e6
                };
                ts.readback.unmap();
                *self.stats.last.kernel_ms.get_or_insert(0.0) += k;
            }
        }
        let out: Vec<f32> = {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            match bytemuck::try_cast_slice::<u8, f32>(&view) {
                Ok(f) => f.to_vec(),
                Err(_) => view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            }
        };
        io.staging.unmap();
        self.stats.last.gpu_roundtrip_ms += round.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }

    /// Resolve each genome's resident slot, uploading genomes seen for the first time.
    fn place<'g>(&mut self, genomes: impl Iterator<Item = &'g fl_core::genome::Genome>) -> Result<Vec<(usize, usize)>, String> {
        let mut placed = Vec::new();
        for g in genomes {
            let t = g.topology as usize;
            let slot = match self.slot_of.get(&g.id) {
                Some(&(tt, s)) if tt == t => s,
                _ => {
                    if self.slabs[t].free.is_empty() {
                        self.grow(t)?;
                    }
                    let s = self.slabs[t].free.pop().expect("grown slab has a free slot");
                    self.upload(t, s, &g.params, &g.motion);
                    self.slabs[t].owner[s] = Some(g.id);
                    self.slot_of.insert(g.id, (t, s));
                    self.stats.genome_uploads += 1;
                    s
                }
            };
            self.slabs[t].last_used[slot] = self.call;
            placed.push((t, slot));
        }
        Ok(placed)
    }

    fn evict_stale(&mut self) {
        for slab in &mut self.slabs {
            for slot in 0..slab.cap() {
                if let Some(g) = slab.owner[slot] {
                    if self.call - slab.last_used[slot] > EVICT_AFTER_CALLS {
                        slab.owner[slot] = None;
                        slab.free.push(slot);
                        self.slot_of.remove(&g);
                        self.stats.evictions += 1;
                    }
                }
            }
        }
    }
}


fn grid_for(workgroups: u32) -> (u32, u32) {
    let gx = workgroups.clamp(1, 65_535);
    (gx, workgroups.div_ceil(gx).max(1))
}

fn read_timestamps(device: &wgpu::Device, ts: &Timestamps, rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>) -> Option<f64> {
    let _ = device;
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return None;
    }
    let ms = {
        let v = ts.readback.slice(..).get_mapped_range().ok()?;
        let t0 = u64::from_le_bytes(v[0..8].try_into().ok()?);
        let t1 = u64::from_le_bytes(v[8..16].try_into().ok()?);
        t1.saturating_sub(t0) as f64 * ts.period_ns / 1e6
    };
    ts.readback.unmap();
    Some(ms)
}

impl GpuBackend {
    fn ensure_field(&mut self, field: &Field, extra_levels: usize, n_sources: usize) -> Result<(), String> {
        let key = (field.gx, field.gy, extra_levels);
        if self.field_gpu.as_ref().is_some_and(|f| f.key == key && f.src_cap >= n_sources) {
            return Ok(());
        }
        self.field_resident = false;
        let src_cap = n_sources.max(64).next_power_of_two();
        if extra_levels + 1 > MAX_LEVELS {
            return Err(format!("{} pyramid levels exceed the GPU limit of {MAX_LEVELS}", extra_levels + 1));
        }
        let cells = field.cells();
        let mut levels = Vec::new();
        let mut off = 0u32;
        for l in 0..=extra_levels {
            let (gx, gy) = ((field.gx >> l) as u32, (field.gy >> l) as u32);
            levels.push((gx, gy, off, (field.hx * (1u64 << l) as f64) as f32, (field.hy * (1u64 << l) as f64) as f32));
            off += gx * gy;
        }
        let pyr_bytes = off as u64 * 33 * 4;
        if pyr_bytes > self.info.max_storage_binding_bytes {
            return Err(format!("sensor pyramid needs {} MiB, over this GPU's {} MiB binding limit", pyr_bytes >> 20, self.info.max_storage_binding_bytes >> 20));
        }
        use wgpu::BufferUsages as U;
        let mk = |label, size: u64, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let fe = mk("field-e", (cells * 4) as u64, U::STORAGE | U::COPY_DST | U::COPY_SRC);
        let fm = mk("field-m", (cells * 32 * 4) as u64, U::STORAGE | U::COPY_DST | U::COPY_SRC);
        let pyr = mk("pyramid", pyr_bytes, U::STORAGE);
        let inv_area = (1.0 / (field.hx * field.hy)) as f32;
        let mut binds = Vec::new();
        let mut grids = Vec::new();
        for l in 0..levels.len() {
            let (gx, gy, dst_off, _, _) = levels[l];
            let (src_off, src_gx) = if l == 0 { (0, 0) } else { (levels[l - 1].2, levels[l - 1].0) };
            let (grid_x, grid_y) = grid_for((gx * gy).div_ceil(64));
            let words: [u32; 8] = [gx, gy, src_off, dst_off, src_gx, inv_area.to_bits(), grid_x, 0];
            let u = mk("pyr-level", 32, U::UNIFORM | U::COPY_DST);
            self.queue.write_buffer(&u, 0, bytemuck::cast_slice(&words));
            binds.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pyramid"),
                layout: &self.sense.pyr_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fe.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pyr.as_entire_binding() },
                ],
            }));
            grids.push((grid_x, grid_y));
        }
        let fe2 = mk("field-e2", (cells * 4) as u64, U::STORAGE | U::COPY_SRC);
        let fm2 = mk("field-m2", (cells * 32 * 4) as u64, U::STORAGE | U::COPY_SRC);
        let acct = mk("field-acct", (cells * 3 * 4) as u64, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let fstaging = mk("field-staging", (cells * 36 * 4) as u64, U::MAP_READ | U::COPY_DST);
        let funiform = mk("field-uniform", 48, U::UNIFORM | U::COPY_DST);
        let src = mk("sources", (src_cap * SOURCE_STRIDE * 4) as u64, U::STORAGE | U::COPY_DST);
        let fbind = |se: &wgpu::Buffer, sm: &wgpu::Buffer, de: &wgpu::Buffer, dm: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field"),
                layout: &self.field_pipes.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: funiform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: se.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: de.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: dm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: acct.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: src.as_entire_binding() },
                ],
            })
        };
        let bind_ab = fbind(&fe, &fm, &fe2, &fm2);
        let bind_ba = fbind(&fe2, &fm2, &fe, &fm);
        let bind_inplace = fbind(&self.dummy, &self.dummy, &fe, &fm);
        let cell_grid = grid_for((cells as u32).div_ceil(64));
        let plane_grid = grid_for((cells as u32 * 33).div_ceil(64));
        self.field_gpu = Some(FieldGpu {
            key,
            fe,
            fm,
            fe2,
            fm2,
            funiform,
            acct,
            fstaging,
            src,
            src_cap,
            bind_ab,
            bind_ba,
            bind_inplace,
            cell_grid,
            plane_grid,
            pyr,
            levels,
            binds,
            grids,
            s0: field.hx.max(field.hy) as f32,
        });
        Ok(())
    }

    fn ensure_sense_io(&mut self, n: usize) {
        if self.sense_io.as_ref().is_some_and(|io| io.cap >= n) {
            return;
        }
        let cap = n.next_power_of_two().max(256);
        use wgpu::BufferUsages as U;
        let mk = |label, size: usize, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        self.sense_io = Some(Io {
            cap,
            slots: mk("unused", 16, U::STORAGE),
            inputs: mk("jobs", cap * JOB_STRIDE * 4, U::STORAGE | U::COPY_DST),
            outputs: mk("sense-out", cap * SENSE_OUT_STRIDE * 4, U::STORAGE | U::COPY_SRC),
            staging: mk("sense-staging", cap * SENSE_OUT_STRIDE * 4, U::MAP_READ | U::COPY_DST),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sense(&mut self, t: usize, page: usize, slots: &[u32], jobs_f: &[f32], sensors: &fl_core::config::SensorsCfg) -> Result<Vec<f32>, String> {
        let prep = std::time::Instant::now();
        let n = slots.len();
        self.ensure_sense_io(n);
        let io = self.sense_io.as_ref().expect("sense io");
        let fg = self.field_gpu.as_ref().expect("field resources");
        let slab = &self.slabs[t];
        let (grid_x, grid_y) = grid_for(n as u32);
        let mut w = [0u32; 76];
        w[0] = n as u32;
        w[1] = slab.dims.len() as u32;
        w[2] = slab.dims[0].0 as u32;
        w[3] = slab.dims[0].1 as u32;
        w[4] = slab.dims[1].1 as u32;
        w[5] = slab.dims.get(2).map_or(0, |d| d.1 as u32);
        w[6] = grid_x;
        w[7] = fg.levels.len() as u32;
        w[8] = fg.s0.to_bits();
        w[9] = (sensors.composition_energy_floor_density as f32).to_bits();
        w[10] = (sensors.energy_density_scale as f32).to_bits();
        w[11] = (sensors.gradient_length_scale_world_units as f32).to_bits();
        for (l, &(gx, gy, off, hx, hy)) in fg.levels.iter().enumerate() {
            w[12 + 4 * l] = gx;
            w[12 + 4 * l + 1] = gy;
            w[12 + 4 * l + 2] = off;
            w[44 + 4 * l] = hx.to_bits();
            w[44 + 4 * l + 1] = hy.to_bits();
        }
        self.queue.write_buffer(&self.sense.uniform, 0, bytemuck::cast_slice(&w));
        self.queue.write_buffer(&io.inputs, 0, f32_bytes(jobs_f));
        let wb = |l: usize| slab.pages[page].get(l).unwrap_or(&self.dummy).as_entire_binding();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sense"),
            layout: &self.sense.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.sense.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: io.inputs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fg.pyr.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: slab.traits[page].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wb(0) },
                wgpu::BindGroupEntry { binding: 5, resource: wb(1) },
                wgpu::BindGroupEntry { binding: 6, resource: if slab.dims.len() == MAX_LAYERS { wb(2) } else { self.dummy.as_entire_binding() } },
                wgpu::BindGroupEntry { binding: 7, resource: io.outputs.as_entire_binding() },
            ],
        });
        let bytes = (n * SENSE_OUT_STRIDE * 4) as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sense") });
        {
            let tw = self.timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites { query_set: &ts.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sense"), timestamp_writes: tw });
            pass.set_pipeline(&self.sense.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(&io.outputs, 0, &io.staging, 0, bytes);
        if let Some(ts) = &self.timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
        let round = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.stats.dispatches += 1;
        self.stats.last.dispatches += 1;
        let slice = io.staging.slice(0..bytes);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let ts_rx = self.timestamps.as_ref().map(|ts| {
            let (ttx, trx) = mpsc::channel();
            ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = ttx.send(r);
            });
            trx
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        if let (Some(ts), Some(trx)) = (&self.timestamps, ts_rx) {
            if let Some(k) = read_timestamps(&self.device, ts, trx) {
                *self.stats.last.kernel_ms.get_or_insert(0.0) += k;
            }
        }
        let out: Vec<f32> = {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            match bytemuck::try_cast_slice::<u8, f32>(&view) {
                Ok(f) => f.to_vec(),
                Err(_) => view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            }
        };
        io.staging.unmap();
        self.stats.last.gpu_roundtrip_ms += round.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }
}

impl SenseInferBackend for GpuBackend {
    fn name(&self) -> &str {
        "wgpu-fused-sense-mlp-f32"
    }

    fn sense_and_infer(&mut self, field: &Field, sensors: &fl_core::config::SensorsCfg, extra_levels: usize, jobs: &[SenseJob<'_>], out: &mut [SenseOutput]) -> Result<(), String> {
        self.call += 1;
        self.stats.calls += 1;
        self.stats.evaluations += jobs.iter().filter(|j| j.due).count() as u64;
        self.stats.last = EvalTiming::default();
        let total = std::time::Instant::now();
        let prep = std::time::Instant::now();
        let placed = self.place(jobs.iter().map(|j| j.genome))?;
        self.ensure_field(field, extra_levels, 0)?;
        // Upload the observation field (unless the field step left it on the device)
        // and build the density pyramid on the GPU.
        let fg = self.field_gpu.as_ref().expect("field resources");
        if !self.field_resident {
            self.queue.write_buffer(&fg.fe, 0, f32_bytes(&field.e));
            self.queue.write_buffer(&fg.fm, 0, f32_bytes(&field.m));
        }
        self.field_resident = false;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pyramid") });
        let nl = fg.binds.len();
        for l in 0..nl {
            let tw = self.pyramid_timestamps.as_ref().filter(|_| l == 0 || l == nl - 1).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: (l == 0).then_some(0),
                end_of_pass_write_index: (l == nl - 1).then_some(1),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pyramid"), timestamp_writes: tw });
            pass.set_pipeline(if l == 0 { &self.sense.pyr_level0 } else { &self.sense.pyr_down });
            pass.set_bind_group(0, &fg.binds[l], &[]);
            pass.dispatch_workgroups(fg.grids[l].0, fg.grids[l].1, 1);
        }
        if let Some(ts) = &self.pyramid_timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.queue.submit([enc.finish()]);
        let pyr_rx = self.pyramid_timestamps.as_ref().map(|ts| {
            let (tx, rx) = mpsc::channel();
            ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx
        });
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
        // One fused dispatch per (topology, weight page).
        let mut groups: Vec<(usize, usize)> = placed.iter().map(|&(t, slot)| (t, slot / self.slabs[t].page_slots)).collect();
        groups.sort_unstable();
        groups.dedup();
        for (t, page) in groups {
            let bk = std::time::Instant::now();
            let ps = self.slabs[t].page_slots;
            let mine: Vec<usize> = (0..jobs.len()).filter(|&k| placed[k].0 == t && placed[k].1 / ps == page).collect();
            let slots: Vec<u32> = mine.iter().map(|&k| (placed[k].1 % ps) as u32).collect();
            let mut jf = std::mem::take(&mut self.input_scratch);
            jf.clear();
            jf.reserve(mine.len() * JOB_STRIDE);
            for (r, &k) in mine.iter().enumerate() {
                let j = &jobs[k];
                jf.extend_from_slice(&[f32::from_bits(slots[r]), j.x as f32, j.y as f32, j.genome.scale() as f32, if j.due { 1.0 } else { 0.0 }]);
                jf.extend_from_slice(&j.intrinsic);
                jf.extend_from_slice(&[0.0, 0.0]);
            }
            self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
            let res = self.run_sense(t, page, &slots, &jf, sensors);
            self.input_scratch = jf;
            let res = res?;
            let bk = std::time::Instant::now();
            for (r, &k) in mine.iter().enumerate() {
                let row = &res[r * SENSE_OUT_STRIDE..(r + 1) * SENSE_OUT_STRIDE];
                let o = &mut out[k];
                o.logits.copy_from_slice(&row[..OUTPUT_WIDTH]);
                o.motion_drive = [row[36] as f64, row[37] as f64];
                o.density0 = row[38] as f64;
            }
            self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
        }
        if let (Some(ts), Some(rx)) = (&self.pyramid_timestamps, pyr_rx) {
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
            self.stats.last.pyramid_ms = read_timestamps(&self.device, ts, rx);
        }
        let bk = std::time::Instant::now();
        self.evict_stale();
        self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
        self.stats.last.total_ms = total.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }
}

impl FieldBackend for GpuBackend {
    fn name(&self) -> &str {
        "wgpu-field-f32"
    }

    fn transport_and_sources(&mut self, field: &mut Field, cfg: &Config, sources: &[Source], n: u64) -> Result<FieldAccounting, String> {
        self.field_resident = false;
        self.stats.field_last = FieldTiming::default();
        let host = std::time::Instant::now();
        self.ensure_field(field, cfg.pyramid_levels(), sources.len())?;
        let dt = cfg.dt();
        let diffusion = cfg.world.diffusion;
        let rx = diffusion * dt / (field.hx * field.hx);
        let ry = diffusion * dt / (field.hy * field.hy);
        let sub = (((rx + ry) / 0.5).ceil() as u32).max(1);
        let lambda = cfg.world.decay_per_second;
        let fg = self.field_gpu.as_ref().expect("field resources");
        let (w, h) = (field.width(), field.height());
        let words: [u32; 12] = [
            field.gx as u32,
            field.gy as u32,
            sources.len() as u32,
            0, // grid_x: patched per dispatch below
            ((rx / sub as f64) as f32).to_bits(),
            ((ry / sub as f64) as f32).to_bits(),
            ((-(-lambda * dt).exp_m1()) as f32).to_bits(),
            (cfg.world.tiny_energy_threshold as f32).to_bits(),
            (w as f32).to_bits(),
            (h as f32).to_bits(),
            (field.hx as f32).to_bits(),
            (field.hy as f32).to_bits(),
        ];
        // Source parameters (centers at the midpoint time of this advance).
        let mut sp = vec![0.0f32; sources.len().max(1) * SOURCE_STRIDE];
        for (i, s) in sources.iter().enumerate() {
            let c = s.center_mid(n, cfg);
            let sigma = cfg.sources.sigma_world_units;
            let cut = cfg.sources.kernel_cutoff_sigma * sigma;
            let (rxb, ryb) = ((cut / field.hx).ceil() as i64 + 1, (cut / field.hy).ceil() as i64 + 1);
            let (gx, gy) = (field.gx as i64, field.gy as i64);
            let cxi = (c[0] / field.hx - 0.5).floor() as i64;
            let cyi = (c[1] / field.hy - 0.5).floor() as i64;
            let (x0, bw) = if 2 * rxb + 1 >= gx { (0, gx) } else { ((cxi - rxb).rem_euclid(gx), 2 * rxb + 1) };
            let (y0, bh) = if 2 * ryb + 1 >= gy { (0, gy) } else { ((cyi - ryb).rem_euclid(gy), 2 * ryb + 1) };
            let b = i * SOURCE_STRIDE;
            sp[b] = c[0] as f32;
            sp[b + 1] = c[1] as f32;
            sp[b + 2] = sigma as f32;
            sp[b + 3] = cut as f32;
            sp[b + 4] = (cfg.sources.power_per_source * dt) as f32;
            sp[b + 6] = f32::from_bits(x0 as u32);
            sp[b + 7] = f32::from_bits(y0 as u32);
            sp[b + 8] = f32::from_bits(bw as u32);
            sp[b + 9] = f32::from_bits(bh as u32);
            sp[b + 10..b + 42].copy_from_slice(&s.chem);
        }
        self.queue.write_buffer(&fg.src, 0, f32_bytes(&sp));
        self.queue.write_buffer(&fg.fe, 0, f32_bytes(&field.e));
        self.queue.write_buffer(&fg.fm, 0, f32_bytes(&field.m));
        let fp = &self.field_pipes;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("field") });
        enc.clear_buffer(&fg.acct, 0, None);
        let mut passes: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32), u32)> = Vec::new();
        if diffusion > 0.0 {
            for k in 0..sub {
                passes.push((&fp.diffuse, if k % 2 == 0 { &fg.bind_ab } else { &fg.bind_ba }, fg.plane_grid, 0));
            }
        }
        let odd = diffusion > 0.0 && sub % 2 == 1;
        let mut tail: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32), u32)> = Vec::new();
        if lambda > 0.0 {
            tail.push((&fp.decay, &fg.bind_inplace, fg.plane_grid, 0));
        }
        if !sources.is_empty() {
            tail.push((&fp.source_norm, &fg.bind_inplace, (sources.len() as u32, 1), 1));
            tail.push((&fp.inject, &fg.bind_inplace, fg.cell_grid, 0));
        }
        tail.push((&fp.tiny, &fg.bind_inplace, fg.cell_grid, 0));
        let total_passes = passes.len() + tail.len();
        let mut k = 0usize;
        let mut run_pass = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, grid: (u32, u32), ts: Option<&Timestamps>| {
            let tw = ts.filter(|_| k == 0 || k + 1 == total_passes).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: (k == 0).then_some(0),
                end_of_pass_write_index: (k + 1 == total_passes).then_some(1),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("field"), timestamp_writes: tw });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
            k += 1;
        };
        // grid_x differs between the plane-wide and cell-wide kernels, so the uniform is
        // written per phase: diffusion/decay share the plane grid, the rest the cell grid.
        let mut wp = words;
        wp[3] = fg.plane_grid.0;
        self.queue.write_buffer(&fg.funiform, 0, bytemuck::cast_slice(&wp));
        for &(pipe, bind, grid, _) in &passes {
            run_pass(&mut enc, pipe, bind, grid, self.field_timestamps.as_ref());
        }
        if odd {
            // The diffused field ended in the B buffers: copy it back to A.
            let cells = field.cells() as u64;
            enc.copy_buffer_to_buffer(&fg.fe2, 0, &fg.fe, 0, cells * 4);
            enc.copy_buffer_to_buffer(&fg.fm2, 0, &fg.fm, 0, cells * 32 * 4);
        }
        let has_decay = lambda > 0.0;
        if has_decay {
            let (pipe, bind, grid, _) = tail[0];
            run_pass(&mut enc, pipe, bind, grid, self.field_timestamps.as_ref());
        }
        // Switch the uniform's grid_x to the cell grid for the remaining passes: they are in a
        // separate submission so the queue write lands in between.
        self.queue.submit([enc.finish()]);
        let mut wc = words;
        wc[3] = fg.cell_grid.0;
        self.queue.write_buffer(&fg.funiform, 0, bytemuck::cast_slice(&wc));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("field-2") });
        for &(pipe, bind, grid, _) in tail.iter().skip(has_decay as usize) {
            run_pass(&mut enc, pipe, bind, grid, self.field_timestamps.as_ref());
        }
        let cells = field.cells() as u64;
        enc.copy_buffer_to_buffer(&fg.fe, 0, &fg.fstaging, 0, cells * 4);
        enc.copy_buffer_to_buffer(&fg.fm, 0, &fg.fstaging, cells * 4, cells * 32 * 4);
        enc.copy_buffer_to_buffer(&fg.acct, 0, &fg.fstaging, cells * 33 * 4, cells * 3 * 4);
        if let Some(ts) = &self.field_timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.stats.field_last.host_ms = host.elapsed().as_secs_f64() * 1e3;
        let round = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        let slice = fg.fstaging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let ts_rx = self.field_timestamps.as_ref().map(|ts| {
            let (ttx, trx) = mpsc::channel();
            ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = ttx.send(r);
            });
            trx
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU field readback: {e}"))?;
        if let (Some(ts), Some(trx)) = (&self.field_timestamps, ts_rx) {
            self.stats.field_last.kernel_ms = read_timestamps(&self.device, ts, trx);
        }
        let mut acct = FieldAccounting { subcycles: if diffusion > 0.0 { sub } else { 1 }, ..Default::default() };
        {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let all: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("field readback alignment: {e:?}"))?;
            let c = cells as usize;
            field.e.copy_from_slice(&all[..c]);
            field.m.copy_from_slice(&all[c..c * 33]);
            let a = &all[c * 33..c * 36];
            for cell in 0..c {
                acct.decay_removed += a[cell * 3] as f64;
                acct.injected += a[cell * 3 + 1] as f64;
                acct.tiny_removed += a[cell * 3 + 2] as f64;
            }
        }
        fg.fstaging.unmap();
        self.stats.field_last.roundtrip_ms = round.elapsed().as_secs_f64() * 1e3;
        self.field_resident = true;
        Ok(acct)
    }
}

impl ControllerBackend for GpuBackend {
    fn name(&self) -> &str {
        "wgpu-fused-mlp-f32"
    }

    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        self.call += 1;
        self.stats.calls += 1;
        self.stats.evaluations += jobs.len() as u64;
        self.stats.last = EvalTiming::default();
        let total = std::time::Instant::now();
        let prep = std::time::Instant::now();
        let placed = self.place(jobs.iter().map(|j| j.genome))?;
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
        // One dispatch per (topology, weight page) that has work.
        let mut groups: Vec<(usize, usize)> = placed.iter().map(|&(t, slot)| (t, slot / self.slabs[t].page_slots)).collect();
        groups.sort_unstable();
        groups.dedup();
        for (t, page) in groups {
            let ps = self.slabs[t].page_slots;
            let mine: Vec<usize> = (0..jobs.len()).filter(|&k| placed[k].0 == t && placed[k].1 / ps == page).collect();
            let slots: Vec<u32> = mine.iter().map(|&k| (placed[k].1 % ps) as u32).collect();
            let bk = std::time::Instant::now();
            let mut inputs = std::mem::take(&mut self.input_scratch);
            inputs.clear();
            inputs.reserve(mine.len() * INPUT_WIDTH);
            for &k in &mine {
                inputs.extend_from_slice(jobs[k].input);
            }
            self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
            let res = self.run_topology(t, page, &slots, &inputs);
            self.input_scratch = inputs;
            let res = res?;
            let bk = std::time::Instant::now();
            for (r, &k) in mine.iter().enumerate() {
                out[k].copy_from_slice(&res[r * OUTPUT_WIDTH..(r + 1) * OUTPUT_WIDTH]);
            }
            self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
        }
        let bk = std::time::Instant::now();
        self.evict_stale();
        self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
        self.stats.last.total_ms = total.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::genome::founder;
    use fl_core::rng::SemanticRng;
    use fl_core::{Config, ReferenceBackend};

    #[test]
    fn gpu_matches_reference_for_every_topology() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        eprintln!("adapter: {:?}", gpu.info);
        let mut cfg = Config::reference();
        let rng = SemanticRng::new(5);
        let mut genomes = Vec::new();
        for (k, h) in [vec![8u32], vec![32], vec![64, 32], vec![96, 64], vec![128, 96]].into_iter().enumerate() {
            cfg.controller.hidden_widths = h;
            for r in 0..70u64 {
                genomes.push(founder(&cfg, &rng, k as u64 * 1000 + r, k as u64 * 1000 + r));
            }
        }
        let inputs: Vec<[f32; INPUT_WIDTH]> = (0..genomes.len()).map(|g| std::array::from_fn(|i| ((g * 37 + i * 11) % 200) as f32 / 100.0 - 1.0)).collect();
        let jobs: Vec<InferenceJob<'_>> = genomes.iter().zip(&inputs).map(|(g, x)| InferenceJob { organism: g.id, genome: g, input: x }).collect();
        let mut a = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
        let mut b = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
        for _ in 0..2 {
            gpu.evaluate(&jobs, &mut a).unwrap();
        }
        ReferenceBackend.evaluate(&jobs, &mut b).unwrap();
        let d = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0f32, f32::max);
        assert!(d < 1e-4, "max |Δlogit| = {d}");
        assert_eq!(gpu.stats.genome_uploads, jobs.len() as u64, "weights stay resident between calls");
        assert_ne!(a[0], a[1], "independent genomes give independent outputs");
    }

    #[test]
    fn sharded_pages_match_reference() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        gpu.set_max_page_slots(100).unwrap(); // force several pages and partial last pages
        let cfg = Config::reference();
        let rng = SemanticRng::new(11);
        let genomes: Vec<_> = (0..357u64).map(|i| founder(&cfg, &rng, i, i)).collect();
        let inputs: Vec<[f32; INPUT_WIDTH]> = (0..genomes.len()).map(|g| std::array::from_fn(|i| ((g * 13 + i * 7) % 200) as f32 / 100.0 - 1.0)).collect();
        let jobs: Vec<InferenceJob<'_>> = genomes.iter().zip(&inputs).map(|(g, x)| InferenceJob { organism: g.id, genome: g, input: x }).collect();
        let mut a = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
        let mut b = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
        gpu.evaluate(&jobs, &mut a).unwrap();
        ReferenceBackend.evaluate(&jobs, &mut b).unwrap();
        assert_eq!(gpu.pages(), 4, "357 organisms over 100-slot pages");
        let d = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0f32, f32::max);
        assert!(d < 1e-4, "max |Δlogit| = {d}");
    }

    fn vivid_small() -> Config {
        let mut c = Config::from_toml(include_str!("../../../presets/vivid-field-ecology.toml")).unwrap();
        c.population.initial_count = 1500;
        c
    }

    #[test]
    fn fused_sensing_matches_cpu_reference_on_a_live_field() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(vivid_small()).unwrap();
        for _ in 0..60 {
            w.step().unwrap();
        }
        let cfg = w.cfg.clone();
        let o = &w.orgs;
        // Vary sensing scales across the whole admitted range so every pyramid level is exercised.
        let genomes: Vec<fl_core::genome::Genome> = (0..o.len())
            .map(|i| {
                let mut g = (*o.genome[i]).clone();
                let lo = cfg.sensors.minimum_scale_world_units.ln();
                let hi = cfg.sensors.maximum_scale_world_units.ln();
                g.log_scale = (lo + (hi - lo) * (i % 17) as f64 / 16.0) as f32;
                g.id = 1_000_000 + i as u64;
                g
            })
            .collect();
        let jobs: Vec<SenseJob<'_>> = (0..o.len())
            .map(|i| SenseJob { organism: o.id[i], genome: &genomes[i], x: o.x[i], y: o.y[i], intrinsic: [0.3, 0.1, -0.2, 0.05, 0.0], due: true })
            .collect();
        let mut a = vec![SenseOutput::default(); jobs.len()];
        let mut b = vec![SenseOutput::default(); jobs.len()];
        let lv = cfg.pyramid_levels();
        gpu.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut a).unwrap();
        fl_core::ReferenceSenseInfer.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut b).unwrap();
        let (mut dl, mut dm, mut dd) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(&b) {
            for k in 0..OUTPUT_WIDTH {
                dl = dl.max((x.logits[k] - y.logits[k]).abs() as f64);
            }
            for k in 0..2 {
                dm = dm.max((x.motion_drive[k] - y.motion_drive[k]).abs() / (1.0 + y.motion_drive[k].abs()));
            }
            dd = dd.max((x.density0 - y.density0).abs() / (1e-6 + y.density0.abs()));
        }
        eprintln!("fused vs reference: logits {dl:.2e}, motion drive (rel) {dm:.2e}, density (rel) {dd:.2e}; pyramid {:?} ms", gpu.stats.last.pyramid_ms);
        assert!(dl < 2e-3, "logits differ by {dl}");
        assert!(dm < 2e-3, "motion drive differs by {dm}");
        assert!(dd < 1e-4, "density differs by {dd}");
    }

    #[test]
    fn world_runs_on_fused_gpu_path() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(vivid_small()).unwrap();
        // The CPU reference sees its first births around tick 300 in this world.
        for _ in 0..640 {
            w.step_inference(fl_core::Inference::Fused(&mut gpu)).unwrap();
        }
        let l = &w.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        assert!((w.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5);
        assert!(w.totals.births > 0, "births happened on the fused path");
        assert!(w.orgs.len() > 500);
    }

    #[test]
    fn gpu_field_step_matches_cpu() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(vivid_small()).unwrap();
        for _ in 0..120 {
            w.step().unwrap();
        }
        for (label, sub_diffusion) in [("one subcycle", None), ("odd subcycles", Some(40.0))] {
            let mut cfg = w.cfg.clone();
            if let Some(d) = sub_diffusion {
                cfg.world.diffusion = d; // forces several (odd) subcycles: exercises the copy-back
            }
            let mut srcs = w.sources.clone();
            for s in &mut srcs {
                s.advance_to(w.tick, &cfg, &w.rng);
            }
            let mut a = w.field.clone();
            let mut b = w.field.clone();
            let ga = gpu.transport_and_sources(&mut a, &cfg, &srcs, w.tick).unwrap();
            let gb = fl_core::world::cpu_transport_and_sources(&mut b, &cfg, &srcs, w.tick).unwrap();
            let emax = b.e.iter().fold(0.0f32, |m, &v| m.max(v.abs())) as f64;
            let de = a.e.iter().zip(&b.e).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / emax;
            let dm = a.m.iter().zip(&b.m).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / emax;
            let rel = |x: f64, y: f64| (x - y).abs() / y.abs().max(1e-9);
            eprintln!(
                "{label}: subcycles {}/{}, field e {de:.2e}, m {dm:.2e}, decay {:.2e}, injected {:.2e}, tiny {} vs {}; kernels {:?} ms",
                ga.subcycles, gb.subcycles, rel(ga.decay_removed, gb.decay_removed), rel(ga.injected, gb.injected), ga.tiny_removed, gb.tiny_removed, gpu.stats.field_last.kernel_ms
            );
            assert_eq!(ga.subcycles, gb.subcycles);
            assert!(de < 1e-5 && dm < 1e-5, "{label}: field differs (e {de}, m {dm})");
            assert!(rel(ga.decay_removed, gb.decay_removed) < 1e-4, "{label}: decay accounting");
            assert!(rel(ga.injected, gb.injected) < 1e-5, "{label}: injection accounting");
            assert!((ga.tiny_removed - gb.tiny_removed).abs() < 1e-9);
        }
    }

    #[test]
    fn world_runs_on_gpu_device_path() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(vivid_small()).unwrap();
        for _ in 0..640 {
            w.step_inference(fl_core::Inference::Device(&mut gpu)).unwrap();
        }
        let l = &w.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        let rel = (w.total_energy() - expected).abs() / l.transferred.max(1.0);
        eprintln!("device path: births {}, population {}, cumulative residual {:.2e} (relative to transfers {rel:.2e})", w.totals.births, w.orgs.len(), l.residual);
        assert!(rel < 1e-5);
        assert!(w.totals.births > 0);
        assert!(w.orgs.len() > 500);
    }
}
