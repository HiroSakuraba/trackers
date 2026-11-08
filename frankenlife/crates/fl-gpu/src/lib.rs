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
use fl_core::{ControllerBackend, InferenceJob};

const SHADER: &str = include_str!("mlp.wgsl");
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
    pub dispatches: u32,
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
}

struct Slab {
    dims: Vec<(usize, usize)>,
    /// Floats per slot for each layer: `in*out + out`.
    stride: Vec<usize>,
    /// Weight pages: each page holds up to `page_slots` organisms in one buffer per
    /// layer, so no buffer exceeds the adapter's storage-binding limit (sharding).
    pages: Vec<Vec<wgpu::Buffer>>,
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
    pub info: GpuInfo,
    pub stats: GpuStats,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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
                Slab { dims, stride, pages: Vec::new(), page_slots, last_cap: 0, owner: Vec::new(), last_used: Vec::new(), free: Vec::new() }
            })
            .collect();
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
        Ok(GpuBackend { device, queue, pipeline, layout, uniform, dummy, slabs, slot_of: HashMap::new(), io: None, call: 0, timestamps, info, stats: GpuStats::default() })
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
        self.queue.submit([enc.finish()]);
        let before = slab.cap();
        if new_page {
            slab.pages.push(fresh);
        } else {
            *slab.pages.last_mut().expect("page") = fresh;
        }
        slab.last_cap = new_cap;
        let after = slab.cap();
        slab.free.extend((before..after).rev());
        slab.owner.resize(after, None);
        slab.last_used.resize(after, 0);
        self.stats.slab_growths += 1;
        Ok(())
    }

    fn upload(&mut self, t: usize, slot: usize, params: &[f32]) {
        let slab = &self.slabs[t];
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
            self.queue.write_buffer(&slab.pages[page][l], (local * slab.stride[l] * 4) as u64, &f32_bytes(&tw));
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
        self.queue.write_buffer(&io.slots, 0, &slots.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        self.queue.write_buffer(&io.inputs, 0, &f32_bytes(inputs));
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
            view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        io.staging.unmap();
        self.stats.last.gpu_roundtrip_ms += round.elapsed().as_secs_f64() * 1e3;
        Ok(out)
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

impl ControllerBackend for GpuBackend {
    fn name(&self) -> &str {
        "wgpu-fused-mlp-f32"
    }

    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        self.call += 1;
        self.stats.calls += 1;
        self.stats.evaluations += jobs.len() as u64;
        self.stats.last = EvalTiming::default();
        let prep = std::time::Instant::now();
        let mut placed = Vec::with_capacity(jobs.len());
        for j in jobs {
            let g = j.genome;
            let t = g.topology as usize;
            let slot = match self.slot_of.get(&g.id) {
                Some(&(tt, s)) if tt == t => s,
                _ => {
                    if self.slabs[t].free.is_empty() {
                        self.grow(t)?;
                    }
                    let s = self.slabs[t].free.pop().expect("grown slab has a free slot");
                    self.upload(t, s, &g.params);
                    self.slabs[t].owner[s] = Some(g.id);
                    self.slot_of.insert(g.id, (t, s));
                    self.stats.genome_uploads += 1;
                    s
                }
            };
            self.slabs[t].last_used[slot] = self.call;
            placed.push((t, slot));
        }
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
        // One dispatch per (topology, weight page) that has work.
        let mut groups: Vec<(usize, usize)> = placed.iter().map(|&(t, slot)| (t, slot / self.slabs[t].page_slots)).collect();
        groups.sort_unstable();
        groups.dedup();
        for (t, page) in groups {
            let ps = self.slabs[t].page_slots;
            let mine: Vec<usize> = (0..jobs.len()).filter(|&k| placed[k].0 == t && placed[k].1 / ps == page).collect();
            let slots: Vec<u32> = mine.iter().map(|&k| (placed[k].1 % ps) as u32).collect();
            let mut inputs = Vec::with_capacity(mine.len() * INPUT_WIDTH);
            for &k in &mine {
                inputs.extend_from_slice(jobs[k].input);
            }
            let res = self.run_topology(t, page, &slots, &inputs)?;
            for (r, &k) in mine.iter().enumerate() {
                out[k].copy_from_slice(&res[r * OUTPUT_WIDTH..(r + 1) * OUTPUT_WIDTH]);
            }
        }
        self.evict_stale();
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
}
