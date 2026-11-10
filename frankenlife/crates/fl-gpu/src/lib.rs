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

use fl_core::nn::{is_recurrent, layer_dims, CATALOG, INPUT_WIDTH, MEMORY_WIDTH, OUTPUT_WIDTH};
use fl_core::genome::Weights;
use fl_core::field::Field;
use fl_core::sources::Source;
use fl_core::{Config, ControllerBackend, FieldAccounting, FieldBackend, FieldDeposit, FinishReport, InferenceJob, ResidentBackend, SenseInferBackend, SenseJob, SenseOutput, UptakeJob, UptakeOut};

const SHADER: &str = include_str!("mlp.wgsl");
const SENSE_SHADER: &str = include_str!("sense.wgsl");
const PYRAMID_SHADER: &str = include_str!("pyramid.wgsl");
const FIELD_SHADER: &str = include_str!("field.wgsl");
const RESIDENT_SHADER: &str = include_str!("resident.wgsl");
const EXPAND_SHADER: &str = include_str!("expand.wgsl");
const JOB_EXPAND_SHADER: &str = include_str!("job_expand.wgsl");
const ITEM_STRIDE: usize = 36;
/// Compact uptake header: cell, tx, ty, demand, affinity slot. Expanded on device.
const HEADER_STRIDE: usize = 5;
/// Per-tick sense inputs: x, y, due, five intrinsics. Slot and scale stay resident.
/// Largest dynamic job row (x, y, due, 5 intrinsics, then 16 memory values for recurrent worlds).
/// Worlds without memory use the shorter `PLAIN_STRIDES` rows, as before.
const DYN_STRIDE: usize = 8 + MEMORY_WIDTH;
const STATIC_STRIDE: usize = 2;
const SOURCE_STRIDE: usize = 44;
/// Floats per slot in the resident traits buffer: 33 motion sensitivities + padding.
const TRAITS_STRIDE: usize = 36;
const JOB_STRIDE: usize = 12 + MEMORY_WIDTH;
const SENSE_OUT_STRIDE: usize = 40 + MEMORY_WIDTH;
const MAX_LEVELS: usize = 8;
const EVICT_AFTER_CALLS: u64 = 64;
const MAX_LAYERS: usize = 3;

/// Row sizes (floats) of the dynamic job words, the expanded sense job, and the sense output.
#[derive(Clone, Copy)]
struct Strides {
    dynamic: usize,
    job: usize,
    out: usize,
}
const PLAIN_STRIDES: Strides = Strides { dynamic: 8, job: 12, out: 40 };
const MEMORY_STRIDES: Strides = Strides { dynamic: 8 + MEMORY_WIDTH, job: 12 + MEMORY_WIDTH, out: 40 + MEMORY_WIDTH };

mod front;

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
    /// CPU time copying the mapped sense/MLP output after the wait has returned.
    /// Filled only while profiling. Already included in `gpu_roundtrip_ms`.
    pub readback_copy_ms: f64,
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
    pub resident_last: ResidentTiming,
}

struct Slab {
    dims: Vec<(usize, usize)>,
    /// Floats per slot for each layer: `in*out + out`. A `packed` slab (the recurrent
    /// network, too many layers for one binding each) has a single entry: all layers back to back.
    stride: Vec<usize>,
    packed: bool,
    /// The packed slab stores 16-bit weights, two per word (`stride[0]` counts words, not weights).
    half: bool,
    /// Largest binding a page may use, and a cap on slots per page (tests / tuning).
    max_bind: u64,
    page_cap: usize,
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
    /// Bumped when this slab's buffers are replaced, so cached bind groups die with them.
    epoch: u64,
}

/// Organisms per page: as many as fit one binding, `st_words` words (of 4 bytes) each.
fn slots_per_page(max_bind: u64, st_words: usize) -> usize {
    (max_bind / (st_words as u64 * 4)).clamp(1, 1 << 20) as usize
}

impl Slab {
    /// Switch a still-empty packed slab to 16-bit weights: half the words per organism, so a page holds twice as many.
    fn enable_half(&mut self) {
        debug_assert!(self.packed && self.pages.is_empty());
        self.half = true;
        self.stride[0] = self.stride[0].div_ceil(2);
        self.page_slots = slots_per_page(self.max_bind, self.stride[0]).min(self.page_cap);
    }

    fn cap(&self) -> usize {
        if self.pages.is_empty() { 0 } else { (self.pages.len() - 1) * self.page_slots + self.last_cap }
    }
}

/// All layers of one network back to back, each transposed to `[in][out]` with its biases after it.
fn transpose_packed<T: Copy>(dims: &[(usize, usize)], params: &[T], zero: T) -> Vec<T> {
    let total: usize = dims.iter().map(|&(i, o)| i * o + o).sum();
    let mut tw = Vec::with_capacity(total + 1);
    let mut off = 0;
    for &(i, o) in dims {
        let base = tw.len();
        tw.resize(base + i * o + o, zero);
        for oo in 0..o {
            for ii in 0..i {
                tw[base + ii * o + oo] = params[off + oo * i + ii];
            }
        }
        tw[base + i * o..base + i * o + o].copy_from_slice(&params[off + i * o..off + i * o + o]);
        off += i * o + o;
    }
    tw
}

struct SensePipelines {
    pipeline: wgpu::ComputePipeline,
    /// The 8-layer recurrent network (one packed weight binding, memory in and out).
    deep: wgpu::ComputePipeline,
    /// The same network reading 16-bit weights from the packed slab.
    deep_half: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    pyr_level0: wgpu::ComputePipeline,
    pyr_down: wgpu::ComputePipeline,
    pyr_layout: wgpu::BindGroupLayout,
}

struct ResidentPipelines {
    layout: wgpu::BindGroupLayout,
    bin_count: wgpu::ComputePipeline,
    bin_scan: wgpu::ComputePipeline,
    bin_scatter: wgpu::ComputePipeline,
    bin_sort: wgpu::ComputePipeline,
    uptake_demand: wgpu::ComputePipeline,
    uptake_org: wgpu::ComputePipeline,
    uptake_debit: wgpu::ComputePipeline,
    deposit_gather: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    expand: wgpu::ComputePipeline,
    expand_layout: wgpu::BindGroupLayout,
    expand_uniform: wgpu::Buffer,
}

/// Device-resident field state: the committed field plus exchange work buffers.
struct ResidentState {
    key: (usize, usize),
    cells: usize,
    ce: wgpu::Buffer,
    cm: wgpu::Buffer,
    committed_energy: f64,
    pending_energy: f64,
    // Work buffers (grown on demand).
    items_cap: usize,
    items: wgpu::Buffer,
    headers: wgpu::Buffer,
    expand_bind: Option<wgpu::BindGroup>,
    counts: wgpu::Buffer,
    starts: wgpu::Buffer,
    sorted: wgpu::Buffer,
    cell_aux: wgpu::Buffer,
    outv: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bind: Option<(u64, wgpu::BindGroup)>,
}

/// Where the time of the most recent resident exchange step went.
///
/// The `*_ms` fields are CPU wall time. `begin_ms`, `deposit_ms` and `commit_ms`
/// stop at `queue.submit` and do not wait for the GPU. `uptake_ms` and `finish_ms`
/// include the wait those calls already use to read results back; that wait is not
/// GPU kernel time. The timestamp spans are optional `TIMESTAMP_QUERY` deltas for
/// the passes inside a call that already waited, and only when profiling is on.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentTiming {
    pub begin_ms: f64,
    pub uptake_ms: f64,
    pub deposit_ms: f64,
    pub finish_ms: f64,
    pub commit_ms: f64,
    /// GPU timestamp-query span for the uptake submission. `None` when profiling
    /// is off, the adapter lacks `TIMESTAMP_QUERY`, or this call did not wait.
    pub uptake_timestamp_span_ms: Option<f64>,
    /// GPU timestamp-query span for the finish submission. Same caveat as uptake.
    pub finish_timestamp_span_ms: Option<f64>,
    /// CPU time building the uptake item buffer (cell, weights, demand, affinity).
    /// Filled only while profiling. Already included in `uptake_ms`.
    pub uptake_pack_ms: f64,
    /// CPU time for `write_buffer` and command recording before `queue.submit`.
    /// Filled only while profiling. Already included in `uptake_ms`.
    pub uptake_upload_encode_ms: f64,
    /// CPU time from `queue.submit` through the mapped uptake readback.
    /// Includes the GPU timestamp span. Not kernel time. Already included in `uptake_ms`.
    pub uptake_wait_readback_ms: f64,
    /// CPU time copying the mapped finish rows after the wait returned.
    /// Filled only while profiling. Already included in `finish_ms`.
    pub finish_readback_copy_ms: f64,
    /// GPU timestamp span of the deposit passes. Deposit itself does not wait;
    /// this span is read in the finish poll that already waits. `None` when
    /// profiling is off, timestamps are unsupported, or there was nothing to deposit.
    pub deposit_timestamp_span_ms: Option<f64>,
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

/// Per-pass GPU timer for the single-wait front half (diagnostic, on with `--profile`): every
/// pass of the sensing/uptake submission gets its own begin/end timestamp; averages go to stderr
/// every 128 ticks.
pub(crate) struct PassTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
    active: std::cell::Cell<bool>,
    names: std::cell::RefCell<Vec<&'static str>>,
    sums: std::cell::RefCell<Vec<(&'static str, f64)>>,
    ticks: std::cell::Cell<u32>,
}

impl PassTimer {
    const MAX_PASSES: usize = 32;

    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> PassTimer {
        let bytes = (Self::MAX_PASSES * 16) as u64;
        PassTimer {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("passes"), ty: wgpu::QueryType::Timestamp, count: (Self::MAX_PASSES * 2) as u32 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor { label: Some("passes-resolve"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            readback: device.create_buffer(&wgpu::BufferDescriptor { label: Some("passes-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            period_ns: queue.get_timestamp_period() as f64,
            active: std::cell::Cell::new(false),
            names: Default::default(),
            sums: Default::default(),
            ticks: std::cell::Cell::new(0),
        }
    }

    fn begin(&self, on: bool) {
        self.active.set(on);
        self.names.borrow_mut().clear();
    }

    /// Timestamp writes for the next pass, or `None` when not recording.
    fn tw(&self, name: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.active.get() {
            return None;
        }
        let mut names = self.names.borrow_mut();
        let k = names.len();
        if k >= Self::MAX_PASSES {
            return None;
        }
        names.push(name);
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: Some(2 * k as u32), end_of_pass_write_index: Some(2 * k as u32 + 1) })
    }

    fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        let k = self.names.borrow().len() as u32;
        if k > 0 {
            enc.resolve_query_set(&self.set, 0..2 * k, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, k as u64 * 16);
        }
        self.active.set(false);
    }

    fn add(&self, name: &'static str, ms: f64) {
        let mut sums = self.sums.borrow_mut();
        match sums.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += ms,
            None => sums.push((name, ms)),
        }
    }

    /// Read the timestamps (the caller has already waited for the submission) and report every 128 ticks.
    fn collect(&self, device: &wgpu::Device) {
        let k = self.names.borrow().len();
        if k == 0 {
            return;
        }
        let slice = self.readback.slice(0..k as u64 * 16);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // The submission has completed, so this poll returns at once.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return;
        }
        {
            let view = slice.get_mapped_range().expect("pass timestamps mapped");
            for (i, &name) in self.names.borrow().iter().enumerate() {
                let t0 = u64::from_le_bytes(view[i * 16..i * 16 + 8].try_into().unwrap());
                let t1 = u64::from_le_bytes(view[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                self.add(name, t1.saturating_sub(t0) as f64 * self.period_ns / 1e6);
            }
        }
        self.readback.unmap();
        let t = self.ticks.get() + 1;
        self.ticks.set(t);
        if t % 128 == 0 {
            let sums = self.sums.borrow();
            let total: f64 = sums.iter().map(|e| e.1).sum::<f64>() / 128.0;
            let line = sums.iter().map(|(n, ms)| format!("{n} {:.2}", ms / 128.0)).collect::<Vec<_>>().join(" | ");
            eprintln!("gpu pass ms/tick (front half, avg of 128): {line} | sum {total:.2}");
            drop(sums);
            self.sums.borrow_mut().clear();
        }
    }
}

struct Io {
    cap: usize,
    slots: wgpu::Buffer,
    inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    staging: wgpu::Buffer,
}

pub struct GpuBackend {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
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
    /// Timestamp queries for resident uptake/finish. Used only while profiling,
    /// and only on submissions that already wait for readback.
    resident_timestamps: Option<Timestamps>,
    pass_timer: Option<PassTimer>,
    /// Deposit passes do not wait. When profiling, their timestamp span is
    /// resolved here and read in the later finish poll. Separate query set so
    /// it does not collide with uptake/finish queries.
    deposit_timestamps: Option<Timestamps>,
    /// A deposit submission wrote timestamps that finish has not read yet.
    deposit_ts_armed: bool,
    /// Opt-in. Does not change dispatch results. Default false.
    profiling: bool,
    /// True while the device copy of the field equals the observation field just
    /// returned by `transport_and_sources` (lets fused sensing skip its upload).
    field_resident: bool,
    field_gpu: Option<FieldGpu>,
    /// Bumped whenever `field_gpu` is rebuilt (its buffers change identity).
    field_gen: u64,
    res_pipes: ResidentPipelines,
    resident: Option<ResidentState>,
    sense_io: Option<Io>,
    /// Field-step readback submitted but not polled yet, so the CPU can pack
    /// sense jobs while that GPU work runs. `None` unless `submit_transport` is ahead.
    pending_field: Option<PendingField>,
    affinity: Option<wgpu::Buffer>,
    affinity_cap: u32,
    affinity_len: u32,
    affinity_free: Vec<u32>,
    affinity_of: HashMap<u64, u32>,
    /// Stencils computed while a sense dispatch was in flight, in sense-job order.
    stencil_ids: Vec<u64>,
    stencil_xy: Vec<(u64, u64)>,
    stencil_val: Vec<(u32, f32, f32)>,
    job_dyn: Option<wgpu::Buffer>,
    job_static: Option<wgpu::Buffer>,
    job_expand_bind: Option<wgpu::BindGroup>,
    job_expand_pipe: Option<wgpu::ComputePipeline>,
    job_expand_layout: Option<wgpu::BindGroupLayout>,
    job_expand_uniform: Option<wgpu::Buffer>,
    job_cap: usize,
    sense_bind: Option<SenseBind>,
    sense_words: Option<[u32; 76]>,
    /// The current sense call carries recurrent memory (wider job and output rows).
    mem_mode: bool,
    sense_static_sig: Vec<(u64, u32, u32)>,
    /// Queue the whole front half of a resident tick and wait once (see `front.rs`).
    single_wait: bool,
    front: front::FrontState,
    /// Scratch copies of a CPU field, used only when a viewer samples a field that is not resident.
    /// Never aliases the simulation buffers.
    preview_scratch: Option<PreviewScratch>,
    pub info: GpuInfo,
    pub stats: GpuStats,
}

/// CPU-field upload target for the live viewer. Not part of a tick.
struct PreviewScratch {
    cells: usize,
    generation: u64,
    energy: wgpu::Buffer,
    moment: wgpu::Buffer,
}

/// Planes the viewer can color without changing simulation state.
///
/// When `on_device` is set, `energy` and `moment` are the working field buffers
/// (`fe` / `fm`). Between ticks those hold the committed field: the resident
/// commit copies them into the committed buffers at the start of the next tick,
/// and this borrow does not flush that copy or map anything.
#[derive(Clone)]
pub struct FieldPlanes {
    pub energy: wgpu::Buffer,
    pub moment: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    /// `hx * hy`, the cell area `fl-view` divides energy by.
    pub cell_area: f32,
    /// Changes when the underlying buffers are reallocated.
    pub generation: u64,
}

struct PendingField {
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ts_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    start: u64,
    end: u64,
    cells: usize,
    resident: bool,
    subcycles: u32,
    round_started: std::time::Instant,
}

struct SenseInflight {
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ts_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    bytes: u64,
}

struct SenseBind {
    key: (usize, usize, usize, u64, u64),
    group: wgpu::BindGroup,
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
        pollster::block_on(Self::open(None, 0))
    }

    /// A new backend with no organism or field state, on the same device, so a window that draws with this
    /// device keeps working (the viewer swaps worlds with it). The pipelines are built again. Field
    /// generations continue from this backend's, so a painter that caches by generation never mistakes the
    /// new world's field buffers for the old ones.
    pub fn fresh(&self) -> Result<GpuBackend, String> {
        pollster::block_on(Self::open(Some((self.instance.clone(), self.adapter.clone(), self.device.clone(), self.queue.clone())), self.field_gen))
    }

    async fn open(existing: Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)>, field_gen: u64) -> Result<GpuBackend, String> {
        let (instance, adapter, device, queue, limits, ts_ok) = match existing {
            Some((instance, adapter, device, queue)) => {
                let limits = device.limits();
                let ts_ok = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
                (instance, adapter, device, queue, limits, ts_ok)
            }
            None => {
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None, apply_limit_buckets: false })
                    .await
                    .map_err(|e| format!("no GPU adapter found: {e}"))?;
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
                (instance, adapter, device, queue, limits, ts_ok)
            }
        };
        let ai = adapter.get_info();
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
                let packed = is_recurrent(CATALOG[t]);
                let mut stride: Vec<usize> = dims.iter().map(|&(i, o)| i * o + o).collect();
                if packed {
                    stride = vec![stride.iter().sum()];
                }
                let max_bind = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
                let page_slots = slots_per_page(max_bind, *stride.iter().max().expect("layers"));
                Slab { dims, stride, packed, half: false, max_bind, page_cap: usize::MAX, pages: Vec::new(), traits: Vec::new(), page_slots, last_cap: 0, owner: Vec::new(), last_used: Vec::new(), free: Vec::new(), epoch: 0 }
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
        let deep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sense-deep"),
            layout: Some(&spl),
            module: &sense_module,
            entry_point: Some("main_deep"),
            compilation_options: Default::default(),
            cache: None,
        });
        let deep_half_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sense-deep-half"),
            layout: Some(&spl),
            module: &sense_module,
            entry_point: Some("main_deep_h"),
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
            deep: deep_pipeline,
            deep_half: deep_half_pipeline,
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
        let pyramid_timestamps = ts_ok.then(&mk_ts);
        let field_timestamps = ts_ok.then(&mk_ts);
        let resident_timestamps = ts_ok.then(&mk_ts);
        let deposit_timestamps = ts_ok.then(&mk_ts);
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
        let res_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("resident"), source: wgpu::ShaderSource::Wgsl(RESIDENT_SHADER.into()) });
        let res_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident"),
            entries: &[
                uentry(0),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("resident"), bind_group_layouts: &[Some(&res_layout)], immediate_size: 0 });
        let mkr = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&rpl),
                module: &res_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let expand_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("expand-items"), source: wgpu::ShaderSource::Wgsl(EXPAND_SHADER.into()) });
        let expand_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("expand-items"),
            entries: &[uentry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
        });
        let epl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("expand-items"), bind_group_layouts: &[Some(&expand_layout)], immediate_size: 0 });
        let expand_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("expand_items"),
            layout: Some(&epl),
            module: &expand_module,
            entry_point: Some("expand_items"),
            compilation_options: Default::default(),
            cache: None,
        });
        let expand_uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("expand-items-dims"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let job_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("expand-jobs"), source: wgpu::ShaderSource::Wgsl(JOB_EXPAND_SHADER.into()) });
        let job_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("expand-jobs"),
            entries: &[uentry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
        });
        let jpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("expand-jobs"), bind_group_layouts: &[Some(&job_layout)], immediate_size: 0 });
        let job_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("expand_jobs"),
            layout: Some(&jpl),
            module: &job_module,
            entry_point: Some("expand_jobs"),
            compilation_options: Default::default(),
            cache: None,
        });
        let job_uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("expand-jobs-dims"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let res_pipes = ResidentPipelines {
            bin_count: mkr("bin_count"),
            bin_scan: mkr("bin_scan"),
            bin_scatter: mkr("bin_scatter"),
            bin_sort: mkr("bin_sort"),
            uptake_demand: mkr("uptake_demand"),
            uptake_org: mkr("uptake_org"),
            uptake_debit: mkr("uptake_debit"),
            deposit_gather: mkr("deposit_gather"),
            finish: mkr("finish"),
            layout: res_layout,
            expand: expand_pipe,
            expand_layout,
            expand_uniform,
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
        Ok(GpuBackend {
            instance,
            adapter,
            device,
            queue,
            pipeline,
            layout,
            uniform,
            dummy,
            slabs,
            slot_of: HashMap::new(),
            io: None,
            call: 0,
            timestamps,
            pyramid_timestamps,
            pass_timer: None,
            input_scratch: Vec::new(),
            sense,
            field_pipes,
            field_timestamps,
            resident_timestamps,
            deposit_timestamps,
            deposit_ts_armed: false,
            profiling: false,
            field_resident: false,
            field_gpu: None,
            field_gen,
            res_pipes,
            resident: None,
            sense_io: None,
            pending_field: None,
            affinity: None,
            affinity_cap: 0,
            affinity_len: 0,
            affinity_free: Vec::new(),
            affinity_of: HashMap::new(),
            stencil_ids: Vec::new(),
            stencil_xy: Vec::new(),
            stencil_val: Vec::new(),
            job_dyn: None,
            job_static: None,
            job_expand_bind: None,
            job_expand_pipe: Some(job_pipe),
            job_expand_layout: Some(job_layout),
            job_expand_uniform: Some(job_uniform),
            job_cap: 0,
            sense_bind: None,
            sense_words: None,
            mem_mode: false,
            sense_static_sig: Vec::new(),
            single_wait: false,
            front: front::FrontState::default(),
            preview_scratch: None,
            info,
            stats: GpuStats::default(),
        })
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Buffers holding field planes for a display sample.
    ///
    /// `on_device`: return the resident working field. Does not submit, wait, or
    /// modify those buffers. `!on_device`: upload `field` into scratch buffers
    /// that the tick never reads.
    pub fn field_planes(&mut self, field: &Field, on_device: bool) -> Result<FieldPlanes, String> {
        let width = field.gx as u32;
        let height = field.gy as u32;
        let cell_area = (field.hx * field.hy) as f32;
        if on_device {
            let fg = self.field_gpu.as_ref().ok_or("the resident field is not on the device yet")?;
            return Ok(FieldPlanes { energy: fg.fe.clone(), moment: fg.fm.clone(), width, height, cell_area, generation: self.field_gen });
        }
        let cells = field.cells();
        let needs = self.preview_scratch.as_ref().is_none_or(|s| s.cells < cells);
        if needs {
            use wgpu::BufferUsages as U;
            let mk = |label, floats: usize| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (floats * 4) as u64,
                    usage: U::STORAGE | U::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            self.preview_scratch = Some(PreviewScratch { cells, generation: self.field_gen.wrapping_add(1).max(1), energy: mk("preview-e", cells), moment: mk("preview-m", cells * 32) });
        }
        let scratch = self.preview_scratch.as_ref().expect("scratch");
        self.queue.write_buffer(&scratch.energy, 0, f32_bytes(&field.e));
        self.queue.write_buffer(&scratch.moment, 0, f32_bytes(&field.m));
        Ok(FieldPlanes {
            energy: scratch.energy.clone(),
            moment: scratch.moment.clone(),
            width,
            height,
            cell_area,
            generation: scratch.generation,
        })
    }

    /// Record resident uptake/finish timestamp spans. Off by default, so the
    /// unprofiled command stream does not write those queries. Does not add a
    /// wait to begin, deposit, or commit, and does not request timestamp
    /// features beyond `TIMESTAMP_QUERY` (already optional at device open).
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    pub fn profiling(&self) -> bool {
        self.profiling
    }

    /// `TIMESTAMP_QUERY` was available on this device. False when spans cannot be read.
    pub fn timestamps_supported(&self) -> bool {
        self.timestamps.is_some()
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
            s.page_cap = slots.max(1);
            s.page_slots = s.page_slots.min(s.page_cap);
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
        slab.epoch = slab.epoch.wrapping_add(1);
        self.sense_bind = None;
        self.stats.slab_growths += 1;
        Ok(())
    }

    fn upload(&mut self, t: usize, slot: usize, params: &Weights, motion: &[f32]) {
        let slab = &self.slabs[t];
        {
            let (page, local) = (slot / slab.page_slots, slot % slab.page_slots);
            let mut tr = [0.0f32; TRAITS_STRIDE];
            tr[..motion.len()].copy_from_slice(motion);
            self.queue.write_buffer(&slab.traits[page], (local * TRAITS_STRIDE * 4) as u64, f32_bytes(&tr));
        }
        if slab.packed {
            // Every layer transposed to `[in][out]` (as the kernels read it), biases after it, layers back to back.
            let (page, local) = (slot / slab.page_slots, slot % slab.page_slots);
            let at = (local * slab.stride[0] * 4) as u64;
            if slab.half {
                // 16-bit patterns, little-endian: element 2k is the low half of word k (WGSL `unpack2x16float`).
                let mut tw = transpose_packed(&slab.dims, &params.to_f16_bits(), 0u16);
                if tw.len() % 2 == 1 {
                    tw.push(0);
                }
                self.queue.write_buffer(&slab.pages[page][0], at, bytemuck::cast_slice(&tw));
            } else {
                let tw = transpose_packed(&slab.dims, &params.to_f32(), 0.0f32);
                self.queue.write_buffer(&slab.pages[page][0], at, f32_bytes(&tw));
            }
            return;
        }
        let params = params.to_f32();
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
        let copy_started = self.profiling.then(std::time::Instant::now);
        let out: Vec<f32> = {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            match bytemuck::try_cast_slice::<u8, f32>(&view) {
                Ok(f) => f.to_vec(),
                Err(_) => view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            }
        };
        if let Some(started) = copy_started {
            self.stats.last.readback_copy_ms += started.elapsed().as_secs_f64() * 1e3;
        }
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
                    if matches!(g.params, Weights::F16(_)) && self.slabs[t].packed && self.slabs[t].pages.is_empty() {
                        self.slabs[t].enable_half();
                    }
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
                        if let Some(slot) = self.affinity_of.remove(&g) {
                            self.affinity_free.push(slot);
                        }
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
        self.field_gen += 1;
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

    fn strides(&self) -> Strides {
        if self.mem_mode { MEMORY_STRIDES } else { PLAIN_STRIDES }
    }

    fn ensure_sense_io(&mut self, n: usize) {
        if self.sense_io.as_ref().is_some_and(|io| io.cap >= n) && self.job_cap >= n {
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
        self.job_dyn = Some(mk("sense-dyn", cap * DYN_STRIDE * 4, U::STORAGE | U::COPY_DST));
        self.job_static = Some(mk("sense-static", cap * STATIC_STRIDE * 4, U::STORAGE | U::COPY_DST));
        self.job_cap = cap;
        self.job_expand_bind = None;
        self.sense_bind = None;
        self.sense_static_sig.clear();
        self.sense_words = None;
    }

    fn ensure_job_expand_bind(&mut self) {
        if self.job_expand_bind.is_some() {
            return;
        }
        let io = self.sense_io.as_ref().expect("sense io");
        let dynb = self.job_dyn.as_ref().expect("dynamic jobs");
        let stb = self.job_static.as_ref().expect("static jobs");
        let layout = self.job_expand_layout.as_ref().expect("job expand layout");
        let uniform = self.job_expand_uniform.as_ref().expect("job expand uniform");
        self.job_expand_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand-jobs"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dynb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: stb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: io.inputs.as_entire_binding() },
            ],
        }));
    }

    fn ensure_sense_bind(&mut self, t: usize, page: usize) {
        let cap = self.sense_io.as_ref().expect("sense io").cap;
        let epoch = self.slabs[t].epoch;
        let key = (t, page, cap, self.field_gen, epoch);
        if self.sense_bind.as_ref().is_some_and(|b| b.key == key) {
            return;
        }
        let io = self.sense_io.as_ref().expect("sense io");
        let fg = self.field_gpu.as_ref().expect("field resources");
        let slab = &self.slabs[t];
        let wb = |l: usize| slab.pages[page].get(l).unwrap_or(&self.dummy).as_entire_binding();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        self.sense_bind = Some(SenseBind { key, group });
    }

    /// Record one fused sense dispatch and map its output, without waiting.
    /// The job buffer the shader reads is filled by `expand_jobs` from the
    /// dynamic upload plus the resident slot/scale buffer, byte-for-byte the
    /// same 28-float records (12 sense words plus the 16 memory values).
    fn submit_sense_group(&mut self, t: usize, page: usize, n: usize, dynamic: &[f32], statics: &[f32], statics_changed: bool, sensors: &fl_core::config::SensorsCfg) -> Result<SenseInflight, String> {
        let prep = std::time::Instant::now();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sense") });
        let stamp = self.timestamps.is_some();
        self.encode_sense_group(&mut enc, t, page, n, dynamic, statics, statics_changed, sensors, stamp)?;
        let io = self.sense_io.as_ref().expect("sense io");
        let bytes = (n * self.strides().out * 4) as u64;
        enc.copy_buffer_to_buffer(&io.outputs, 0, &io.staging, 0, bytes);
        if let Some(ts) = &self.timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.queue.submit([enc.finish()]);
        self.stats.last.host_prep_ms += prep.elapsed().as_secs_f64() * 1e3;
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
        Ok(SenseInflight { rx, ts_rx, bytes })
    }

    /// Record the job-expand and fused sense passes of one (topology, page) group
    /// into `enc`. Uploads the group's dynamic (and, when changed, static) job words.
    /// `stamp` writes the sense timestamp pair; the caller resolves and reads it.
    #[allow(clippy::too_many_arguments)]
    fn encode_sense_group(&mut self, enc: &mut wgpu::CommandEncoder, t: usize, page: usize, n: usize, dynamic: &[f32], statics: &[f32], statics_changed: bool, sensors: &fl_core::config::SensorsCfg, stamp: bool) -> Result<(), String> {
        self.ensure_sense_io(n);
        self.ensure_job_expand_bind();
        self.ensure_sense_bind(t, page);
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
        // lu[1].w and lu[2].w: job and output row strides.
        let st = self.strides();
        w[19] = st.job as u32;
        w[23] = st.out as u32;
        if self.sense_words != Some(w) {
            self.queue.write_buffer(&self.sense.uniform, 0, bytemuck::cast_slice(&w));
            self.sense_words = Some(w);
        }
        let dynb = self.job_dyn.as_ref().expect("dynamic jobs");
        self.queue.write_buffer(dynb, 0, f32_bytes(dynamic));
        if statics_changed {
            let stb = self.job_static.as_ref().expect("static jobs");
            self.queue.write_buffer(stb, 0, f32_bytes(statics));
        }
        let expand_grid = grid_for((n as u32).div_ceil(64));
        let st = self.strides();
        let dims = [n as u32, expand_grid.0, st.dynamic as u32, st.job as u32];
        let uniform = self.job_expand_uniform.as_ref().expect("job expand uniform");
        self.queue.write_buffer(uniform, 0, bytemuck::cast_slice(&dims));
        {
            let tw = self.timestamps.as_ref().filter(|_| stamp).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: None,
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("expand-jobs"), timestamp_writes: tw });
            pass.set_pipeline(self.job_expand_pipe.as_ref().expect("job expand"));
            pass.set_bind_group(0, self.job_expand_bind.as_ref().expect("job expand bind"), &[]);
            pass.dispatch_workgroups(expand_grid.0, expand_grid.1, 1);
        }
        {
            let tw = self.timestamps.as_ref().filter(|_| stamp).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sense"), timestamp_writes: tw });
            pass.set_pipeline(if slab.half { &self.sense.deep_half } else if slab.packed { &self.sense.deep } else { &self.sense.pipeline });
            pass.set_bind_group(0, &self.sense_bind.as_ref().expect("sense bind").group, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        Ok(())
    }

    fn finish_sense_group(&mut self, inflight: SenseInflight, mine: Option<&[usize]>, out: &mut [SenseOutput]) -> Result<(), String> {
        let round_started = std::time::Instant::now();
        let SenseInflight { rx, ts_rx, bytes } = inflight;
        let io = self.sense_io.as_ref().expect("sense io");
        let slice = io.staging.slice(0..bytes);
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        if let (Some(ts), Some(trx)) = (&self.timestamps, ts_rx) {
            if let Some(k) = read_timestamps(&self.device, ts, trx) {
                *self.stats.last.kernel_ms.get_or_insert(0.0) += k;
            }
        }
        let copy_started = self.profiling.then(std::time::Instant::now);
        {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let rows: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("sense readback alignment: {e:?}"))?;
            let stride = self.strides().out;
            let n = rows.len() / stride;
            for r in 0..n {
                let k = mine.map(|m| m[r]).unwrap_or(r);
                let row = &rows[r * stride..(r + 1) * stride];
                let o = &mut out[k];
                o.logits.copy_from_slice(&row[..OUTPUT_WIDTH]);
                o.motion_drive = [row[36] as f64, row[37] as f64];
                o.density0 = row[38] as f64;
                if self.mem_mode {
                    o.memory.copy_from_slice(&row[40..40 + MEMORY_WIDTH]);
                }
            }
        }
        if let Some(started) = copy_started {
            self.stats.last.readback_copy_ms += started.elapsed().as_secs_f64() * 1e3;
        }
        io.staging.unmap();
        self.stats.last.gpu_roundtrip_ms += round_started.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Cell, tx, ty for every sense job, computed while the sense dispatch is
    /// already on the queue. Uptake reuses these bits instead of recomputing them.
    fn remember_stencils(&mut self, jobs: &[SenseJob<'_>]) {
        if self.resident.is_none() {
            return;
        }
        let Ok((gx, gy, hx, hy)) = self.res_geom() else { return };
        self.stencil_ids.clear();
        self.stencil_xy.clear();
        self.stencil_val.clear();
        self.stencil_ids.reserve(jobs.len());
        self.stencil_xy.reserve(jobs.len());
        self.stencil_val.reserve(jobs.len());
        for j in jobs {
            let (k, tx, ty) = stencil_base(gx, gy, hx, hy, j.x, j.y);
            self.stencil_ids.push(j.genome.id);
            self.stencil_xy.push((j.x.to_bits(), j.y.to_bits()));
            self.stencil_val.push((k, tx, ty));
        }
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
        self.mem_mode = jobs.first().is_some_and(|j| is_recurrent(j.genome.hidden()));
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
        // One fused dispatch per (topology, weight page). A single page is the
        // steady state of the 10k/20k targets: skip the filter and overlap the
        // stencil precompute with that one in-flight dispatch.
        let mut groups: Vec<(usize, usize)> = placed.iter().map(|&(t, slot)| (t, slot / self.slabs[t].page_slots)).collect();
        groups.sort_unstable();
        groups.dedup();
        let single = groups.len() == 1;
        for (gi, (t, page)) in groups.into_iter().enumerate() {
            let bk = std::time::Instant::now();
            // Grow the resident job buffers before comparing the static signature,
            // so a reallocation cannot skip the upload into a fresh buffer.
            self.ensure_sense_io(jobs.len());
            let ps = self.slabs[t].page_slots;
            let (mine, slots): (Option<Vec<usize>>, Vec<u32>) = if single {
                (None, placed.iter().map(|&(_, slot)| (slot % ps) as u32).collect())
            } else {
                let mine: Vec<usize> = (0..jobs.len()).filter(|&k| placed[k].0 == t && placed[k].1 / ps == page).collect();
                let slots = mine.iter().map(|&k| (placed[k].1 % ps) as u32).collect();
                (Some(mine), slots)
            };
            let n = slots.len();
            let mut dynamic = Vec::with_capacity(n * DYN_STRIDE);
            let mut statics = Vec::with_capacity(n * STATIC_STRIDE);
            let mut sig = Vec::with_capacity(n);
            for (r, &slot) in slots.iter().enumerate() {
                let k = mine.as_ref().map(|m| m[r]).unwrap_or(r);
                let j = &jobs[k];
                let scale = j.genome.scale() as f32;
                dynamic.extend_from_slice(&[j.x as f32, j.y as f32, if j.due { 1.0 } else { 0.0 }]);
                dynamic.extend_from_slice(&j.intrinsic);
                if self.mem_mode {
                    dynamic.extend_from_slice(j.memory);
                }
                statics.extend_from_slice(&[f32::from_bits(slot), scale]);
                sig.push((j.genome.id, slot, scale.to_bits()));
            }
            let statics_changed = sig != self.sense_static_sig;
            if statics_changed {
                self.sense_static_sig = sig;
            }
            self.stats.last.bookkeeping_ms += bk.elapsed().as_secs_f64() * 1e3;
            let inflight = self.submit_sense_group(t, page, n, &dynamic, &statics, statics_changed, sensors)?;
            if gi == 0 {
                // The dispatch is queued. Stencil bases depend only on positions,
                // so this CPU work overlaps the sense kernel instead of the uptake pack.
                let overlap = std::time::Instant::now();
                self.remember_stencils(jobs);
                self.stats.last.bookkeeping_ms += overlap.elapsed().as_secs_f64() * 1e3;
            }
            self.finish_sense_group(inflight, mine.as_deref(), out)?;
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
        self.run_field_step(field, cfg, sources, n, false, false)
    }
}

impl GpuBackend {
    /// The field step. `resident`: the working field is already on the device and stays
    /// there (only the accounting is read back); otherwise `field` is uploaded first and
    /// the observation field is downloaded into it.
    /// `defer` maps the accounting readback and returns without polling, so the
    /// caller can do CPU work before `finish_field_read`. Only used for the
    /// resident field, whose planes stay on the device.
    fn run_field_step(&mut self, field: &mut Field, cfg: &Config, sources: &[Source], n: u64, resident: bool, defer: bool) -> Result<FieldAccounting, String> {
        self.field_resident = false;
        self.stats.field_last = FieldTiming::default();
        let host = std::time::Instant::now();
        self.ensure_field(field, cfg.pyramid_levels(), sources.len())?;
        self.front.pyr_done = false;
        let with_pyramid = self.single_wait && resident && defer && !self.profiling;
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
        if !resident {
            self.queue.write_buffer(&fg.fe, 0, f32_bytes(&field.e));
            self.queue.write_buffer(&fg.fm, 0, f32_bytes(&field.m));
        }
        let fp = &self.field_pipes;
        let mut enc = match self.front.open_enc.take() {
            Some(enc) => enc,
            None => self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("field") }),
        };
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
        if !resident {
            enc.copy_buffer_to_buffer(&fg.fe, 0, &fg.fstaging, 0, cells * 4);
            enc.copy_buffer_to_buffer(&fg.fm, 0, &fg.fstaging, cells * 4, cells * 32 * 4);
        }
        if with_pyramid {
            // The sensor pyramid of the observation field, in the same submission.
            self.encode_pyramid(&mut enc, None);
            self.front.pyr_done = true;
        }
        enc.copy_buffer_to_buffer(&fg.acct, 0, &fg.fstaging, cells * 33 * 4, cells * 3 * 4);
        if let Some(ts) = &self.field_timestamps {
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        self.stats.field_last.host_ms = host.elapsed().as_secs_f64() * 1e3;
        let round = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        let start = if resident { cells * 33 * 4 } else { 0 };
        let end = cells * 36 * 4;
        let slice = fg.fstaging.slice(start..end);
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
        let subcycles = if diffusion > 0.0 { sub } else { 1 };
        let pending = PendingField { rx, ts_rx, start, end, cells: cells as usize, resident, subcycles, round_started: round };
        if defer {
            self.pending_field = Some(pending);
            self.field_resident = true;
            return Ok(FieldAccounting { subcycles, ..Default::default() });
        }
        self.finish_field_read(pending, field)
    }

    fn finish_field_read(&mut self, pending: PendingField, field: &mut Field) -> Result<FieldAccounting, String> {
        let PendingField { rx, ts_rx, start, end, cells, resident, subcycles, round_started } = pending;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let slice = fg.fstaging.slice(start..end);
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU field readback: {e}"))?;
        if let (Some(ts), Some(trx)) = (&self.field_timestamps, ts_rx) {
            self.stats.field_last.kernel_ms = read_timestamps(&self.device, ts, trx);
        }
        let mut acct = FieldAccounting { subcycles, ..Default::default() };
        {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let all: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("field readback alignment: {e:?}"))?;
            let c = cells;
            let a = if resident {
                &all[..c * 3]
            } else {
                field.e.copy_from_slice(&all[..c]);
                field.m.copy_from_slice(&all[c..c * 33]);
                &all[c * 33..c * 36]
            };
            for cell in 0..c {
                acct.decay_removed += a[cell * 3] as f64;
                acct.injected += a[cell * 3 + 1] as f64;
                acct.tiny_removed += a[cell * 3 + 2] as f64;
            }
        }
        fg.fstaging.unmap();
        self.stats.field_last.roundtrip_ms = round_started.elapsed().as_secs_f64() * 1e3;
        self.field_resident = true;
        Ok(acct)
    }
}

/// Bilinear stencil base cell and fractions (`grid-cic-v1`), as the CPU computes them.
fn stencil_base(field_gx: usize, field_gy: usize, hx: f64, hy: f64, x: f64, y: f64) -> (u32, f32, f32) {
    let fx = x / hx - 0.5;
    let fy = y / hy - 0.5;
    let ix = fx.floor();
    let iy = fy.floor();
    let x0 = (ix as i64).rem_euclid(field_gx as i64) as usize;
    let y0 = (iy as i64).rem_euclid(field_gy as i64) as usize;
    ((y0 * field_gx + x0) as u32, (fx - ix) as f32, (fy - iy) as f32)
}

impl GpuBackend {
    fn res_geom(&self) -> Result<(usize, usize, f64, f64), String> {
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let (gx, gy) = (fg.key.0, fg.key.1);
        let (hx, hy) = (fg.levels[0].3 as f64, fg.levels[0].4 as f64);
        Ok((gx, gy, hx, hy))
    }

    fn deposit_items(&self, deps: &[FieldDeposit], hx: f64, hy: f64) -> Vec<f32> {
        let (gx, gy, _, _) = self.res_geom().expect("geometry");
        let mut v = Vec::with_capacity(deps.len() * ITEM_STRIDE);
        for d in deps {
            let (key, tx, ty, q, dir) = match *d {
                FieldDeposit::Stencil { x, y, q, ref dir } => {
                    let (k, tx, ty) = stencil_base(gx, gy, hx, hy, x, y);
                    (k, tx, ty, q, dir)
                }
                FieldDeposit::Cell { cell, q, ref dir } => (cell as u32, 0.0, 0.0, q, dir),
            };
            v.extend_from_slice(&[f32::from_bits(key), tx, ty, q as f32]);
            v.extend_from_slice(dir);
        }
        v
    }

    /// Make sure work buffers hold `n` items and the bind group matches the current field buffers.
    fn ensure_res_work(&mut self, n: usize) -> Result<(), String> {
        let gen = self.field_gen;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let st = self.resident.as_mut().ok_or("resident state not initialized")?;
        if n > st.items_cap {
            let cap = n.next_power_of_two().max(1024);
            use wgpu::BufferUsages as U;
            let mk = |label, size: usize, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
            let out_floats = (cap * 4).max(st.cells * 4);
            st.items = mk("items", cap * ITEM_STRIDE * 4, U::STORAGE | U::COPY_DST);
            st.headers = mk("headers", cap * HEADER_STRIDE * 4, U::STORAGE | U::COPY_DST);
            st.expand_bind = None;
            st.sorted = mk("sorted", cap * 4, U::STORAGE);
            st.outv = mk("res-out", out_floats * 4, U::STORAGE | U::COPY_SRC);
            st.staging = mk("res-staging", out_floats * 4, U::MAP_READ | U::COPY_DST);
            st.items_cap = cap;
            st.bind = None;
        }
        if st.bind.as_ref().is_none_or(|(g, _)| *g != gen) {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident"),
                layout: &self.res_pipes.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: st.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fg.fe.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fg.fm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: st.items.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: st.counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: st.starts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: st.sorted.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: st.cell_aux.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: st.outv.as_entire_binding() },
                ],
            });
            st.bind = Some((gen, bg));
        }
        Ok(())
    }

    fn write_res_uniform(&self, n_items: usize, ch: Option<&fl_core::config::ChemistryCfg>, tiny: f64) {
        let st = self.resident.as_ref().expect("resident");
        let (gx, gy) = st.key;
        let items_grid = grid_for((n_items.max(1) as u32).div_ceil(64));
        let cells_grid = grid_for((st.cells as u32).div_ceil(64));
        let (emax, gain, thr, kappa) = ch.map_or((0.0, 0.0, 0.0, 0.0), |c| (c.maximum_efficiency, c.affinity_gain, c.discrimination_threshold, c.toxicity_sensitivity));
        let w: [u32; 12] = [
            gx as u32,
            gy as u32,
            n_items as u32,
            items_grid.0,
            (emax as f32).to_bits(),
            (gain as f32).to_bits(),
            (thr as f32).to_bits(),
            (kappa as f32).to_bits(),
            (tiny as f32).to_bits(),
            cells_grid.0,
            0,
            0,
        ];
        self.queue.write_buffer(&st.uniform, 0, bytemuck::cast_slice(&w));
    }

    /// `stamp` writes a TIMESTAMP_QUERY at the start of the first binning pass.
    /// The matching end stamp is written by the caller on its last pass.
    /// `BinStamp::Off` leaves the command stream unchanged. This does not wait.
    /// `begin` writes the opening timestamp on the first binning pass.
    /// Uptake turns that off because the item-expand pass already wrote it.
    fn encode_binning(&self, enc: &mut wgpu::CommandEncoder, n: usize, stamp: BinStamp, begin: bool) {
        let st = self.resident.as_ref().expect("resident");
        let p = &self.res_pipes;
        let items_grid = grid_for((n.max(1) as u32).div_ceil(64));
        let cells_grid = grid_for((st.cells as u32).div_ceil(64));
        enc.clear_buffer(&st.counts, 0, None);
        let ts = match stamp {
            BinStamp::Off => None,
            BinStamp::Resident => self.resident_timestamps.as_ref(),
            BinStamp::Deposit => self.deposit_timestamps.as_ref(),
        };
        let begin_tw = if begin {
            ts.map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: None,
            })
        } else {
            None
        };
        let pt = |name| self.pass_timer.as_ref().and_then(|t| t.tw(name));
        dispatch_resident(st, enc, &p.bin_count, items_grid, begin_tw.or_else(|| pt("bin_count")));
        dispatch_resident(st, enc, &p.bin_scan, (1, 1), pt("bin_scan"));
        dispatch_resident(st, enc, &p.bin_scatter, items_grid, pt("bin_scatter"));
        dispatch_resident(st, enc, &p.bin_sort, cells_grid, pt("bin_sort"));
    }

    /// Read the resident staging buffer. When `with_timestamp` is set, the
    /// timestamp readback is mapped before the same `poll` the data already
    /// waits on — no second synchronization.
    /// `with_timestamp` reads the uptake/finish query in this same poll.
    /// `with_deposit_timestamp` reads the deposit query in that same poll.
    /// The last value is the CPU time spent copying the mapped rows, and is 0
    /// when profiling is off.
    /// Map the resident staging buffer and run `body` on the floats in place.
    /// The copy timer covers `body` only, so callers that reduce in the mapping
    /// do not pay a second allocation. The last value is 0 when profiling is off.
    fn read_res_out<T>(&self, floats: usize, with_timestamp: bool, with_deposit_timestamp: bool, body: impl FnOnce(&[f32]) -> T) -> Result<(T, Option<f64>, Option<f64>, f64), String> {
        let st = self.resident.as_ref().expect("resident");
        let slice = st.staging.slice(0..(floats * 4) as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let ts_rx = if with_timestamp {
            self.resident_timestamps.as_ref().map(|ts| {
                let (ttx, trx) = mpsc::channel();
                ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = ttx.send(r);
                });
                trx
            })
        } else {
            None
        };
        let dep_rx = if with_deposit_timestamp {
            self.deposit_timestamps.as_ref().map(|ts| {
                let (ttx, trx) = mpsc::channel();
                ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = ttx.send(r);
                });
                trx
            })
        } else {
            None
        };
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        let span = match (self.resident_timestamps.as_ref(), ts_rx) {
            (Some(ts), Some(trx)) => read_timestamps(&self.device, ts, trx),
            _ => None,
        };
        let deposit_span = match (self.deposit_timestamps.as_ref(), dep_rx) {
            (Some(ts), Some(trx)) => read_timestamps(&self.device, ts, trx),
            _ => None,
        };
        let copy_started = self.profiling.then(std::time::Instant::now);
        let out = {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let rows = bytemuck::try_cast_slice::<u8, f32>(&view).map_err(|e| format!("readback alignment: {e:?}"))?;
            body(rows)
        };
        let copy_ms = copy_started.map(|t| t.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        st.staging.unmap();
        Ok((out, span, deposit_span, copy_ms))
    }

    /// Apply deposit items to the working field (no readback).
    fn encode_deposits(&mut self, enc: &mut wgpu::CommandEncoder, deps: &[FieldDeposit], stamp: bool) -> Result<(), String> {
        if deps.is_empty() {
            return Ok(());
        }
        let (_, _, hx, hy) = self.res_geom()?;
        let items = self.deposit_items(deps, hx, hy);
        self.ensure_res_work(deps.len())?;
        let st = self.resident.as_ref().expect("resident");
        self.queue.write_buffer(&st.items, 0, f32_bytes(&items));
        self.write_res_uniform(deps.len(), None, 0.0);
        self.encode_binning(enc, deps.len(), if stamp { BinStamp::Deposit } else { BinStamp::Off }, stamp);
        let cells_grid = grid_for((st.cells as u32).div_ceil(64));
        let pipe = &self.res_pipes.deposit_gather;
        let end = if stamp {
            self.deposit_timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            })
        } else {
            None
        };
        dispatch_resident(st, enc, pipe, cells_grid, end);
        if stamp {
            if let Some(ts) = &self.deposit_timestamps {
                enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
                self.deposit_ts_armed = true;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum BinStamp {
    Off,
    Resident,
    Deposit,
}

fn dispatch_resident(st: &ResidentState, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, grid: (u32, u32), tw: Option<wgpu::ComputePassTimestampWrites<'_>>) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("resident"), timestamp_writes: tw });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &st.bind.as_ref().expect("bind").1, &[]);
    pass.dispatch_workgroups(grid.0, grid.1, 1);
}

impl GpuBackend {
    fn take_stencil(&self, j: &UptakeJob<'_>, gx: usize, gy: usize, hx: f64, hy: f64, cursor: &mut usize) -> (u32, f32, f32) {
        let xb = j.x.to_bits();
        let yb = j.y.to_bits();
        while *cursor < self.stencil_ids.len() {
            if self.stencil_ids[*cursor] == j.genome.id && self.stencil_xy[*cursor] == (xb, yb) {
                let s = self.stencil_val[*cursor];
                *cursor += 1;
                return s;
            }
            *cursor += 1;
        }
        stencil_base(gx, gy, hx, hy, j.x, j.y)
    }

    fn affinity_slot(&mut self, id: u64, affinity: &[f32]) -> Result<u32, String> {
        if let Some(&slot) = self.affinity_of.get(&id) {
            return Ok(slot);
        }
        let slot = if let Some(slot) = self.affinity_free.pop() {
            slot
        } else {
            if self.affinity_len == self.affinity_cap {
                self.grow_affinity()?;
            }
            let slot = self.affinity_len;
            self.affinity_len += 1;
            slot
        };
        let buf = self.affinity.as_ref().ok_or("affinity buffer")?;
        self.queue.write_buffer(buf, slot as u64 * 32 * 4, f32_bytes(affinity));
        self.affinity_of.insert(id, slot);
        Ok(slot)
    }

    fn grow_affinity(&mut self) -> Result<(), String> {
        let new_cap = (self.affinity_cap.max(1) * 2).max(1024);
        use wgpu::BufferUsages as U;
        let fresh = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("affinity"),
            size: new_cap as u64 * 32 * 4,
            usage: U::STORAGE | U::COPY_DST | U::COPY_SRC,
            mapped_at_creation: false,
        });
        if let Some(old) = self.affinity.take() {
            if self.affinity_len > 0 {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("affinity-grow") });
                enc.copy_buffer_to_buffer(&old, 0, &fresh, 0, self.affinity_len as u64 * 32 * 4);
                self.queue.submit([enc.finish()]);
            }
        }
        self.affinity = Some(fresh);
        self.affinity_cap = new_cap;
        if let Some(st) = self.resident.as_mut() {
            st.expand_bind = None;
        }
        Ok(())
    }

    fn ensure_expand_bind(&mut self) -> Result<(), String> {
        if self.resident.as_ref().is_some_and(|st| st.expand_bind.is_some()) {
            return Ok(());
        }
        let affinity = self.affinity.as_ref().ok_or("affinity buffer")?.clone();
        let st = self.resident.as_ref().ok_or("resident")?;
        let headers = st.headers.clone();
        let items = st.items.clone();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand-items"),
            layout: &self.res_pipes.expand_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.res_pipes.expand_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: headers.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: affinity.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: items.as_entire_binding() },
            ],
        });
        self.resident.as_mut().expect("resident").expand_bind = Some(bg);
        Ok(())
    }
}

impl ResidentBackend for GpuBackend {
    fn upload_committed(&mut self, field: &Field, cfg: &Config) -> Result<(), String> {
        self.ensure_field(field, cfg.pyramid_levels(), cfg.sources.count as usize)?;
        let cells = field.cells();
        use wgpu::BufferUsages as U;
        let mk = |label, size: usize, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        let st = ResidentState {
            key: (field.gx, field.gy),
            cells,
            ce: mk("committed-e", cells * 4, U::STORAGE | U::COPY_SRC | U::COPY_DST),
            cm: mk("committed-m", cells * 32 * 4, U::STORAGE | U::COPY_SRC | U::COPY_DST),
            committed_energy: field.total_energy(),
            pending_energy: 0.0,
            items_cap: 0,
            items: mk("items", 16, U::STORAGE | U::COPY_DST),
            headers: mk("headers", 16, U::STORAGE | U::COPY_DST),
            expand_bind: None,
            counts: mk("counts", cells * 4, U::STORAGE | U::COPY_DST),
            starts: mk("starts", (cells + 1) * 4, U::STORAGE),
            sorted: mk("sorted", 16, U::STORAGE),
            cell_aux: mk("cell-aux", cells * 4, U::STORAGE),
            outv: mk("res-out", 16, U::STORAGE | U::COPY_SRC),
            staging: mk("res-staging", 16, U::MAP_READ | U::COPY_DST),
            uniform: mk("res-uniform", 48, U::UNIFORM | U::COPY_DST),
            bind: None,
        };
        self.queue.write_buffer(&st.ce, 0, f32_bytes(&field.e));
        self.queue.write_buffer(&st.cm, 0, f32_bytes(&field.m));
        self.front.commit_pending = false;
        self.front.open_enc = None;
        self.front.deps_n = None;
        self.resident = Some(st);
        self.ensure_res_work(1)?;
        Ok(())
    }

    fn download_committed(&mut self, field: &mut Field) -> Result<(), String> {
        self.flush_commit();
        let st = self.resident.as_ref().ok_or("nothing uploaded")?;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let cells = st.cells as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("download") });
        enc.copy_buffer_to_buffer(&st.ce, 0, &fg.fstaging, 0, cells * 4);
        enc.copy_buffer_to_buffer(&st.cm, 0, &fg.fstaging, cells * 4, cells * 32 * 4);
        self.queue.submit([enc.finish()]);
        let slice = fg.fstaging.slice(0..cells * 33 * 4);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let all: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("readback alignment: {e:?}"))?;
            let c = cells as usize;
            field.e.resize(c, 0.0);
            field.m.resize(c * 32, 0.0);
            field.e.copy_from_slice(&all[..c]);
            field.m.copy_from_slice(&all[c..c * 33]);
        }
        fg.fstaging.unmap();
        Ok(())
    }

    fn committed_energy(&self) -> f64 {
        self.resident.as_ref().map_or(0.0, |s| s.committed_energy)
    }

    fn begin_tick(&mut self, ops: &[FieldDeposit]) -> Result<(), String> {
        let t = std::time::Instant::now();
        self.stats.resident_last = ResidentTiming::default();
        self.field_resident = false;
        let st = self.resident.as_ref().ok_or("nothing uploaded")?;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let cells = st.cells as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("begin") });
        if self.front.commit_pending {
            // The previous tick's commit, folded in here: the working field already
            // is the committed one, so it is copied out instead of copied in.
            enc.copy_buffer_to_buffer(&fg.fe, 0, &st.ce, 0, cells * 4);
            enc.copy_buffer_to_buffer(&fg.fm, 0, &st.cm, 0, cells * 32 * 4);
            self.front.commit_pending = false;
        } else {
            enc.copy_buffer_to_buffer(&st.ce, 0, &fg.fe, 0, cells * 4);
            enc.copy_buffer_to_buffer(&st.cm, 0, &fg.fm, 0, cells * 32 * 4);
        }
        self.deposit_ts_armed = false;
        self.front.deps_n = None;
        self.encode_deposits(&mut enc, ops, false)?;
        if self.single_wait {
            // Rides with the first field-step submission.
            self.front.open_enc = Some(enc);
        } else {
            self.queue.submit([enc.finish()]);
        }
        self.stats.resident_last.begin_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    fn submit_transport(&mut self, cfg: &Config, sources: &[Source], n: u64) -> Result<(), String> {
        if self.pending_field.is_some() {
            let mut discard = Field { gx: 1, gy: 1, hx: 1.0, hy: 1.0, e: Vec::new(), m: Vec::new() };
            let pending = self.pending_field.take().expect("pending");
            self.finish_field_read(pending, &mut discard)?;
        }
        let st = self.resident.as_ref().ok_or("nothing uploaded")?;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let mut geom = Field { gx: st.key.0, gy: st.key.1, hx: fg.levels[0].3 as f64, hy: fg.levels[0].4 as f64, e: Vec::new(), m: Vec::new() };
        let (hx, hy) = cfg.cell_size();
        geom.hx = hx;
        geom.hy = hy;
        self.run_field_step(&mut geom, cfg, sources, n, true, true)?;
        Ok(())
    }

    fn finish_transport(&mut self) -> Result<FieldAccounting, String> {
        let pending = self.pending_field.take().ok_or("transport was not submitted")?;
        let mut discard = Field { gx: 1, gy: 1, hx: 1.0, hy: 1.0, e: Vec::new(), m: Vec::new() };
        self.finish_field_read(pending, &mut discard)
    }

    fn uptake(&mut self, ch: &fl_core::config::ChemistryCfg, jobs: &[UptakeJob<'_>], out: &mut [UptakeOut]) -> Result<f64, String> {
        let t = std::time::Instant::now();
        if jobs.is_empty() {
            return Ok(0.0);
        }
        let pack_t = self.profiling.then(std::time::Instant::now);
        let (gx, gy, hx, hy) = self.res_geom()?;
        let mut headers = Vec::with_capacity(jobs.len() * HEADER_STRIDE);
        let mut cursor = 0usize;
        for j in jobs {
            let (k, tx, ty) = self.take_stencil(j, gx, gy, hx, hy, &mut cursor);
            let slot = self.affinity_slot(j.genome.id, &j.genome.affinity)?;
            headers.extend_from_slice(&[f32::from_bits(k), tx, ty, j.demand as f32, f32::from_bits(slot)]);
        }
        if self.profiling {
            self.stats.resident_last.uptake_pack_ms = pack_t.map(|s| s.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        }
        let upload_t = self.profiling.then(std::time::Instant::now);
        self.ensure_res_work(jobs.len())?;
        self.ensure_expand_bind()?;
        let st = self.resident.as_ref().expect("resident");
        self.queue.write_buffer(&st.headers, 0, f32_bytes(&headers));
        self.write_res_uniform(jobs.len(), Some(ch), 0.0);
        let items_grid = grid_for((jobs.len() as u32).div_ceil(64));
        let dims = [jobs.len() as u32, items_grid.0, 0, 0];
        self.queue.write_buffer(&self.res_pipes.expand_uniform, 0, bytemuck::cast_slice(&dims));
        let stamp = self.profiling && self.resident_timestamps.is_some();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("uptake") });
        {
            let tw = if stamp {
                self.resident_timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites {
                    query_set: &ts.set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                })
            } else {
                None
            };
            let st = self.resident.as_ref().expect("resident");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("expand-items"), timestamp_writes: tw });
            pass.set_pipeline(&self.res_pipes.expand);
            pass.set_bind_group(0, st.expand_bind.as_ref().expect("expand bind"), &[]);
            pass.dispatch_workgroups(items_grid.0, items_grid.1, 1);
        }
        self.encode_binning(&mut enc, jobs.len(), if stamp { BinStamp::Resident } else { BinStamp::Off }, false);
        let st = self.resident.as_ref().expect("resident");
        let p = &self.res_pipes;
        let cells_grid = grid_for((st.cells as u32).div_ceil(64));
        dispatch_resident(st, &mut enc, &p.uptake_demand, cells_grid, None);
        dispatch_resident(st, &mut enc, &p.uptake_org, items_grid, None);
        let end = if stamp {
            self.resident_timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            })
        } else {
            None
        };
        dispatch_resident(st, &mut enc, &p.uptake_debit, cells_grid, end);
        enc.copy_buffer_to_buffer(&st.outv, 0, &st.staging, 0, (jobs.len() * 16) as u64);
        if stamp {
            let ts = self.resident_timestamps.as_ref().expect("timestamp query set");
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        if self.profiling {
            self.stats.resident_last.uptake_upload_encode_ms = upload_t.map(|s| s.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        }
        let wait_t = self.profiling.then(std::time::Instant::now);
        self.queue.submit([enc.finish()]);
        let (granted, span, _, _copy_ms) = self.read_res_out(jobs.len() * 4, stamp, false, |res| {
            let mut granted = 0.0;
            for (k, o) in out.iter_mut().enumerate() {
                *o = UptakeOut { absorbed: res[k * 4] as f64, useful_gain: res[k * 4 + 1] as f64, toxicity_requested: res[k * 4 + 2] as f64 };
                granted += o.absorbed;
            }
            granted
        })?;
        if self.profiling {
            self.stats.resident_last.uptake_wait_readback_ms = wait_t.map(|s| s.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        }
        if stamp {
            self.stats.resident_last.uptake_timestamp_span_ms = span;
        }
        self.stats.resident_last.uptake_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(granted)
    }

    fn sense_uptake(&mut self, cfg: &Config, extra_levels: usize, jobs: &[SenseJob<'_>], sense_out: &mut [SenseOutput], uptake_out: &mut [UptakeOut]) -> Option<Result<(FieldAccounting, f64), String>> {
        if !self.single_wait {
            return None;
        }
        Some(self.sense_uptake_impl(cfg, extra_levels, jobs, sense_out, uptake_out))
    }

    fn deposit(&mut self, deposits: &[FieldDeposit]) -> Result<(), String> {
        let t = std::time::Instant::now();
        if self.single_wait && !self.profiling {
            // Packed now, queued with the finish pass: one submission for the back half.
            let (_, _, hx, hy) = self.res_geom()?;
            self.front.deps_items = self.deposit_items(deposits, hx, hy);
            self.front.deps_n = Some(deposits.len());
            self.stats.resident_last.deposit_ms = t.elapsed().as_secs_f64() * 1e3;
            return Ok(());
        }
        let stamp = self.profiling && self.deposit_timestamps.is_some() && !deposits.is_empty();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("deposit") });
        self.encode_deposits(&mut enc, deposits, stamp)?;
        self.queue.submit([enc.finish()]);
        self.stats.resident_last.deposit_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    fn finish(&mut self, tiny: f64) -> Result<FinishReport, String> {
        let t = std::time::Instant::now();
        self.ensure_res_work(1)?;
        self.write_res_uniform(0, None, tiny);
        let st = self.resident.as_ref().expect("resident");
        let cells = st.cells;
        let stamp = self.profiling && self.resident_timestamps.is_some();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("finish") });
        if let Some(n_dep) = self.front.deps_n.take() {
            if n_dep > 0 {
                self.ensure_res_work(n_dep)?;
                let st = self.resident.as_ref().expect("resident");
                self.queue.write_buffer(&st.items, 0, f32_bytes(&self.front.deps_items));
                // One uniform serves both: the deposit kernels read the item count, finish reads `tiny`.
                self.write_res_uniform(n_dep, None, tiny);
                self.encode_binning(&mut enc, n_dep, BinStamp::Off, false);
                dispatch_resident(st, &mut enc, &self.res_pipes.deposit_gather, grid_for((st.cells as u32).div_ceil(64)), None);
            }
        }
        let st = self.resident.as_ref().expect("resident");
        let pipe = &self.res_pipes.finish;
        let tw = if stamp {
            self.resident_timestamps.as_ref().map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            })
        } else {
            None
        };
        dispatch_resident(st, &mut enc, pipe, grid_for((cells as u32).div_ceil(64)), tw);
        enc.copy_buffer_to_buffer(&st.outv, 0, &st.staging, 0, (cells * 16) as u64);
        if stamp {
            let ts = self.resident_timestamps.as_ref().expect("timestamp query set");
            enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
            enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
        }
        let read_deposit_ts = self.deposit_ts_armed;
        self.queue.submit([enc.finish()]);
        let (rep, span, deposit_span, copy_ms) = self.read_res_out(cells * 4, stamp, read_deposit_ts, |rows| {
            let mut rep = FinishReport { worst_cone: f64::NEG_INFINITY, min_energy: f64::INFINITY, ..Default::default() };
            for c in 0..cells {
                let e = rows[c * 4] as f64;
                rep.field_energy += e;
                rep.tiny_removed += rows[c * 4 + 1] as f64;
                rep.worst_cone = rep.worst_cone.max(rows[c * 4 + 2] as f64);
                rep.min_energy = rep.min_energy.min(e);
                rep.cone_projections += (rows[c * 4 + 3] > 0.5) as u32;
            }
            if rep.worst_cone < 0.0 {
                rep.worst_cone = -1.0;
            }
            rep
        })?;
        self.deposit_ts_armed = false;
        if stamp {
            self.stats.resident_last.finish_timestamp_span_ms = span;
        }
        if read_deposit_ts {
            self.stats.resident_last.deposit_timestamp_span_ms = deposit_span;
        }
        if self.profiling {
            self.stats.resident_last.finish_readback_copy_ms = copy_ms;
        }
        self.resident.as_mut().expect("resident").pending_energy = rep.field_energy;
        self.stats.resident_last.finish_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(rep)
    }

    fn commit(&mut self) -> Result<(), String> {
        let t = std::time::Instant::now();
        let st = self.resident.as_mut().ok_or("nothing uploaded")?;
        let fg = self.field_gpu.as_ref().ok_or("resident field not initialized")?;
        let cells = st.cells as u64;
        if self.single_wait {
            // The copy rides with the next tick's first submission (or is flushed
            // by a download or an abort, whichever comes first).
            self.front.commit_pending = true;
        } else {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("commit") });
            enc.copy_buffer_to_buffer(&fg.fe, 0, &st.ce, 0, cells * 4);
            enc.copy_buffer_to_buffer(&fg.fm, 0, &st.cm, 0, cells * 32 * 4);
            self.queue.submit([enc.finish()]);
        }
        st.committed_energy = st.pending_energy;
        self.field_resident = false;
        self.stats.resident_last.commit_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    fn abort(&mut self) {
        // The working field is simply discarded; the next begin_tick recopies the committed field.
        // An encoder left open by begin_tick may carry the previous tick's commit: submit it.
        if let Some(enc) = self.front.open_enc.take() {
            self.queue.submit([enc.finish()]);
        }
        self.flush_commit();
        self.front.deps_n = None;
        self.front.pyr_done = false;
        self.field_resident = false;
    }
}

impl ControllerBackend for GpuBackend {
    fn name(&self) -> &str {
        "wgpu-fused-mlp-f32"
    }

    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        if jobs.iter().any(|j| is_recurrent(j.genome.hidden())) {
            return Err("the controller-only GPU backend has no memory; run recurrent networks with --backend gpu-fused or later".into());
        }
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

    const ZERO_MEMORY: [f32; MEMORY_WIDTH] = [0.0; MEMORY_WIDTH];

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
            .map(|i| SenseJob { organism: o.id[i], genome: &genomes[i], x: o.x[i], y: o.y[i], intrinsic: [0.3, 0.1, -0.2, 0.05, 0.0], due: true, memory: &ZERO_MEMORY })
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

    fn recurrent_small() -> Config {
        let mut c = vivid_small();
        c.controller.hidden_widths = vec![32; 8];
        c.architecture_mutation.maximum_parameters = 40_000;
        c.population.initial_count = 600;
        c
    }

    /// The 8-layer recurrent kernel gives the CPU reference's logits and next memory.
    #[test]
    fn recurrent_fused_sensing_matches_cpu_reference() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(recurrent_small()).unwrap();
        for _ in 0..30 {
            w.step().unwrap();
        }
        let cfg = w.cfg.clone();
        let o = &w.orgs;
        assert!(o.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)), "the world built up memory to read back");
        let genomes: Vec<fl_core::genome::Genome> = (0..o.len())
            .map(|i| {
                let mut g = (*o.genome[i]).clone();
                g.id = 2_000_000 + i as u64;
                g
            })
            .collect();
        let jobs: Vec<SenseJob<'_>> = (0..o.len())
            .map(|i| SenseJob { organism: o.id[i], genome: &genomes[i], x: o.x[i], y: o.y[i], intrinsic: [0.3, 0.1, -0.2, 0.05, 0.0], due: true, memory: &o.memory[i] })
            .collect();
        let mut a = vec![SenseOutput::default(); jobs.len()];
        let mut b = vec![SenseOutput::default(); jobs.len()];
        let lv = cfg.pyramid_levels();
        gpu.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut a).unwrap();
        fl_core::ReferenceSenseInfer.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut b).unwrap();
        let (mut dl, mut dm, mut mag) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(&b) {
            for k in 0..OUTPUT_WIDTH {
                dl = dl.max((x.logits[k] - y.logits[k]).abs() as f64);
            }
            for k in 0..MEMORY_WIDTH {
                dm = dm.max((x.memory[k] - y.memory[k]).abs() as f64);
                mag = mag.max(y.memory[k].abs() as f64);
            }
        }
        eprintln!("recurrent fused vs reference: logits {dl:.2e}, memory {dm:.2e} (largest memory value {mag:.2})");
        assert!(mag > 0.0, "the reference produced a memory to compare");
        assert!(dl < 2e-3, "logits differ by {dl}");
        assert!(dm < 2e-3, "memory differs by {dm}");
    }

    /// A recurrent job that is not due skips the network on the GPU but still gets its motion drive and
    /// density; the due ones match the CPU reference as before.
    #[test]
    fn recurrent_kernel_skips_jobs_that_are_not_due() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(recurrent_small()).unwrap();
        for _ in 0..30 {
            w.step().unwrap();
        }
        let cfg = w.cfg.clone();
        let o = &w.orgs;
        let genomes: Vec<fl_core::genome::Genome> = (0..o.len())
            .map(|i| {
                let mut g = (*o.genome[i]).clone();
                g.id = 4_000_000 + i as u64;
                g
            })
            .collect();
        let jobs: Vec<SenseJob<'_>> = (0..o.len())
            .map(|i| SenseJob { organism: o.id[i], genome: &genomes[i], x: o.x[i], y: o.y[i], intrinsic: [0.3, 0.1, -0.2, 0.05, 0.0], due: i % 3 == 0, memory: &o.memory[i] })
            .collect();
        let mut a = vec![SenseOutput::default(); jobs.len()];
        let mut b = vec![SenseOutput::default(); jobs.len()];
        let lv = cfg.pyramid_levels();
        gpu.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut a).unwrap();
        fl_core::ReferenceSenseInfer.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut b).unwrap();
        let mut due = 0;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            for k in 0..2 {
                assert!((x.motion_drive[k] - y.motion_drive[k]).abs() <= 1e-3 * (1.0 + y.motion_drive[k].abs()), "job {i} drive {k}");
            }
            assert!((x.density0 - y.density0).abs() <= 1e-4 * (1.0 + y.density0.abs()), "job {i} density");
            if jobs[i].due {
                due += 1;
                for k in 0..OUTPUT_WIDTH {
                    assert!((x.logits[k] - y.logits[k]).abs() < 2e-3, "job {i} logit {k}");
                }
                for k in 0..MEMORY_WIDTH {
                    assert!((x.memory[k] - y.memory[k]).abs() < 2e-3, "job {i} memory {k}");
                }
            }
        }
        assert!(due > 10 && due < jobs.len(), "a mix of due and idle jobs: {due} of {}", jobs.len());
    }

    /// A fresh backend on the same device carries nothing over from the world that came before it: a new
    /// world (organism ids start over) steps exactly as it does on a brand new backend.
    #[test]
    fn fresh_backend_forgets_the_previous_world() {
        let Ok(mut old) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        old.set_single_wait(true);
        let mut first = fl_core::World::new(recurrent_small()).unwrap();
        for _ in 0..8 {
            first.step_inference(fl_core::Inference::Resident(&mut old)).unwrap();
        }
        let mut cfg = recurrent_small();
        cfg.population.initial_count = 450;
        cfg.population.maximum_count = 900;
        let old_gen = old.field_gen;
        let mut reused = old.fresh().unwrap();
        reused.set_single_wait(true);
        let mut a = fl_core::World::new(cfg.clone()).unwrap();
        let Ok(mut other) = GpuBackend::new() else { return };
        other.set_single_wait(true);
        let mut b = fl_core::World::new(cfg).unwrap();
        for _ in 0..8 {
            a.step_inference(fl_core::Inference::Resident(&mut reused)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut other)).unwrap();
        }
        assert!(a.orgs.len() > 400, "the new world has its own, larger population");
        assert!(old_gen >= 1 && reused.field_gen > old_gen, "field generations keep counting up across a fresh backend ({} after {old_gen})", reused.field_gen);
        assert_eq!(a.state_hash(), b.state_hash(), "a fresh backend steps like a new one");
    }

    fn recurrent_half_small() -> Config {
        let mut c = recurrent_small();
        c.controller.canonical_weight_dtype = "f16".into();
        c
    }

    /// The 16-bit kernel reads exactly the weights the 16-bit genome holds: same logits and memory as the CPU reference,
    /// and the slab really is half the size.
    #[test]
    fn half_precision_recurrent_kernel_matches_cpu_reference() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(recurrent_half_small()).unwrap();
        for _ in 0..30 {
            w.step().unwrap();
        }
        let cfg = w.cfg.clone();
        let o = &w.orgs;
        assert!(o.genome.iter().all(|g| matches!(g.params, Weights::F16(_))));
        assert!(o.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)));
        let genomes: Vec<fl_core::genome::Genome> = (0..o.len())
            .map(|i| {
                let mut g = (*o.genome[i]).clone();
                g.id = 3_000_000 + i as u64;
                g
            })
            .collect();
        let jobs: Vec<SenseJob<'_>> = (0..o.len())
            .map(|i| SenseJob { organism: o.id[i], genome: &genomes[i], x: o.x[i], y: o.y[i], intrinsic: [0.3, 0.1, -0.2, 0.05, 0.0], due: true, memory: &o.memory[i] })
            .collect();
        let mut a = vec![SenseOutput::default(); jobs.len()];
        let mut b = vec![SenseOutput::default(); jobs.len()];
        let lv = cfg.pyramid_levels();
        gpu.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut a).unwrap();
        fl_core::ReferenceSenseInfer.sense_and_infer(&w.field, &cfg.sensors, lv, &jobs, &mut b).unwrap();
        let (mut dl, mut dm) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(&b) {
            for k in 0..OUTPUT_WIDTH {
                dl = dl.max((x.logits[k] - y.logits[k]).abs() as f64);
            }
            for k in 0..MEMORY_WIDTH {
                dm = dm.max((x.memory[k] - y.memory[k]).abs() as f64);
            }
        }
        eprintln!("16-bit recurrent fused vs reference: logits {dl:.2e}, memory {dm:.2e}");
        assert!(dl < 2e-3, "logits differ by {dl}");
        assert!(dm < 2e-3, "memory differs by {dm}");
        let deep = CATALOG.iter().position(|h| is_recurrent(h)).unwrap();
        assert!(gpu.slabs[deep].half);
        assert_eq!(gpu.slabs[deep].stride[0], 12_980 / 2);
        // Half the bytes per organism of the 32-bit slab (which strides 12,980 words).
        assert_eq!(gpu.resident_bytes(), (gpu.slabs[deep].cap() * 6_490 * 4) as u64);
    }

    /// A 16-bit recurrent world runs on both resident paths, is bit-reproducible and closes its ledger.
    #[test]
    fn half_precision_world_runs_on_the_resident_paths() {
        let Ok(mut g1) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut g2 = GpuBackend::new().unwrap();
        g1.set_single_wait(true);
        g2.set_single_wait(true);
        let mut a = fl_core::World::new(recurrent_half_small()).unwrap();
        let mut b = fl_core::World::new(recurrent_half_small()).unwrap();
        for _ in 0..100 {
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
        }
        a.sync_field(&mut g1).unwrap();
        b.sync_field(&mut g2).unwrap();
        assert_eq!(a.state_hash(), b.state_hash());
        assert!(a.orgs.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)));
        assert!(a.orgs.genome.iter().all(|g| matches!(g.params, Weights::F16(_))));
        let l = &a.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        assert!((a.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5);
    }

    /// A recurrent world runs on the classic and single-wait resident paths, agrees with the
    /// CPU early on, is bit-reproducible, and closes its ledger.
    #[test]
    fn recurrent_world_runs_on_the_resident_paths() {
        let Ok(mut classic) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut g1 = GpuBackend::new().unwrap();
        let mut g2 = GpuBackend::new().unwrap();
        g1.set_single_wait(true);
        g2.set_single_wait(true);
        let mut cpu = fl_core::World::new(recurrent_small()).unwrap();
        let mut r = fl_core::World::new(recurrent_small()).unwrap();
        let mut a = fl_core::World::new(recurrent_small()).unwrap();
        let mut b = fl_core::World::new(recurrent_small()).unwrap();
        for _ in 0..10 {
            cpu.step().unwrap();
            r.step_inference(fl_core::Inference::Resident(&mut classic)).unwrap();
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
        }
        let dm = |x: &fl_core::world::Organisms, y: &fl_core::world::Organisms| {
            assert_eq!(x.len(), y.len());
            (0..x.len()).flat_map(|i| (0..MEMORY_WIDTH).map(move |k| (i, k))).map(|(i, k)| (x.memory[i][k] - y.memory[i][k]).abs()).fold(0.0f32, f32::max)
        };
        let (d_classic, d_fast) = (dm(&cpu.orgs, &r.orgs), dm(&cpu.orgs, &a.orgs));
        eprintln!("recurrent memory vs CPU after 10 ticks: classic {d_classic:.2e}, single-wait {d_fast:.2e}");
        assert!(d_classic < 5e-3 && d_fast < 5e-3);
        assert!(a.orgs.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)));
        assert_eq!(dm(&a.orgs, &b.orgs), 0.0);
        for _ in 10..120 {
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
        }
        a.sync_field(&mut g1).unwrap();
        b.sync_field(&mut g2).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "two recurrent single-wait runs are bit-identical");
        let l = &a.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        assert!((a.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5);
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

    #[test]
    fn resident_tick_matches_cpu_reference() {
        let Ok(mut gpu) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut w = fl_core::World::new(vivid_small()).unwrap();
        for _ in 0..120 {
            w.step().unwrap();
        }
        let cfg = w.cfg.clone();
        let mut cpu = fl_core::ReferenceResident::default();
        let mut srcs = w.sources.clone();
        for s in &mut srcs {
            s.advance_to(w.tick, &cfg, &w.rng);
        }
        let o = &w.orgs;
        let dirs: Vec<[f32; 32]> = (0..o.len()).map(|i| o.genome[i].affinity).collect();
        let ops = vec![
            FieldDeposit::Cell { cell: 77, q: 3.0, dir: [0.0; 32] },
            FieldDeposit::Stencil { x: 10.3, y: 99.7, q: 1.5, dir: dirs[0] },
        ];
        let jobs: Vec<UptakeJob<'_>> = (0..o.len()).map(|i| UptakeJob { x: o.x[i], y: o.y[i], demand: 0.003 * (1 + i % 5) as f64, genome: &o.genome[i] }).collect();
        let deps: Vec<FieldDeposit> = (0..o.len()).step_by(3).map(|i| FieldDeposit::Stencil { x: o.x[i], y: o.y[i], q: 0.01, dir: dirs[i] }).collect();
        let mut fa = w.field.clone();
        let mut fb = w.field.clone();
        let mut ua = vec![UptakeOut::default(); jobs.len()];
        let mut ub = vec![UptakeOut::default(); jobs.len()];
        let mut reports = Vec::new();
        for (be, uo) in [(&mut gpu as &mut dyn ResidentBackend, &mut ua), (&mut cpu as &mut dyn ResidentBackend, &mut ub)] {
            be.upload_committed(&w.field, &cfg).unwrap();
            be.begin_tick(&ops).unwrap();
            let acct = be.transport_resident(&cfg, &srcs, w.tick).unwrap();
            let granted = be.uptake(&cfg.chemistry, &jobs, uo).unwrap();
            be.deposit(&deps).unwrap();
            let fin = be.finish(cfg.world.tiny_energy_threshold).unwrap();
            be.commit().unwrap();
            reports.push((acct, granted, fin, be.committed_energy()));
        }
        gpu.download_committed(&mut fa).unwrap();
        cpu.download_committed(&mut fb).unwrap();
        let rel = |x: f64, y: f64| (x - y).abs() / y.abs().max(1e-9);
        let (ga, gb) = (&reports[0], &reports[1]);
        let mut du = 0.0f64;
        for (x, y) in ua.iter().zip(&ub) {
            du = du.max((x.absorbed - y.absorbed).abs() / (1e-6 + y.absorbed.abs()));
            du = du.max((x.useful_gain - y.useful_gain).abs() / (1e-6 + y.absorbed.abs()));
            du = du.max((x.toxicity_requested - y.toxicity_requested).abs() / (1e-6 + y.absorbed.abs()));
        }
        let emax = fb.e.iter().fold(0.0f32, |m, &v| m.max(v)) as f64;
        let de = fa.e.iter().zip(&fb.e).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / emax;
        let dm = fa.m.iter().zip(&fb.m).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / emax;
        eprintln!(
            "resident vs cpu: uptake {du:.2e}, granted {:.2e}, field e {de:.2e} m {dm:.2e}, energy {:.2e}, injected {:.2e}, cone {:.2e}/{:.2e}, projections {}/{}",
            rel(ga.1, gb.1), rel(ga.3, gb.3), rel(ga.0.injected, gb.0.injected), ga.2.worst_cone, gb.2.worst_cone, ga.2.cone_projections, gb.2.cone_projections
        );
        assert!(du < 1e-4, "uptake differs by {du}");
        assert!(rel(ga.1, gb.1) < 1e-5, "granted total");
        assert!(de < 1e-5 && dm < 1e-5, "committed field differs (e {de}, m {dm})");
        assert!(rel(ga.3, gb.3) < 1e-6, "committed energy");
        assert!((ga.2.tiny_removed - gb.2.tiny_removed).abs() < 1e-9);
    }

    #[test]
    fn resident_gpu_world_is_deterministic_and_closes_its_ledger() {
        let Ok(mut g1) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut g2 = GpuBackend::new().unwrap();
        let mut a = fl_core::World::new(vivid_small()).unwrap();
        let mut b = fl_core::World::new(vivid_small()).unwrap();
        a.schedule(50, fl_core::Command::EraseDisk { x: 60.0, y: 60.0, radius: 12.0, disposal: fl_core::Disposal::Recycle }).unwrap();
        b.schedule(50, fl_core::Command::EraseDisk { x: 60.0, y: 60.0, radius: 12.0, disposal: fl_core::Disposal::Recycle }).unwrap();
        for _ in 0..200 {
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
        }
        a.sync_field(&mut g1).unwrap();
        b.sync_field(&mut g2).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "two GPU-resident runs are bit-identical");
        for _ in 200..640 {
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
        }
        a.sync_field(&mut g1).unwrap();
        let l = &a.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        let rel = (a.total_energy() - expected).abs() / l.transferred.max(1.0);
        eprintln!("resident world: births {}, user deaths {}, population {}, residual {:.2e} ({rel:.2e} of transfers)", a.totals.births, a.totals.deaths_user, a.orgs.len(), l.residual);
        assert!(rel < 1e-5);
        assert!(a.totals.births > 0 && a.totals.deaths_user > 0);
        assert!(a.orgs.len() > 500);
    }

    /// Live tuning commands (source power and speed, diffusion, decay, chemistry drift)
    /// reach the device: the single-wait GPU world follows the CPU reference through the
    /// change, injects exactly what the new power says, and its ledger still closes.
    #[test]
    fn live_tuning_reaches_the_gpu_and_matches_the_cpu() {
        use fl_core::world::Param;
        let Ok(mut g) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        g.set_single_wait(true);
        let mut cpu = fl_core::World::new(vivid_small()).unwrap();
        let mut gpu = fl_core::World::new(vivid_small()).unwrap();
        let base = vivid_small();
        for w in [&mut cpu, &mut gpu] {
            w.schedule(10, fl_core::Command::SetParam { param: Param::SourcePower, value: base.sources.power_per_source * 2.0 }).unwrap();
            w.schedule(10, fl_core::Command::SetParam { param: Param::SourceSpeed, value: base.sources.path_speed * 3.0 + 1.0 }).unwrap();
            w.schedule(10, fl_core::Command::SetParam { param: Param::Diffusion, value: base.world.diffusion * 1.5 }).unwrap();
            w.schedule(10, fl_core::Command::SetParam { param: Param::Decay, value: base.world.decay_per_second * 2.0 }).unwrap();
            w.schedule(10, fl_core::Command::DriftChemistry { source: Some(0), degrees: 40.0 }).unwrap();
        }
        let (mut before, mut after) = (0.0, 0.0);
        for k in 0..20 {
            cpu.step().unwrap();
            let r = gpu.step_inference(fl_core::Inference::Resident(&mut g)).unwrap();
            if k == 9 {
                before = r.ledger.injected_sources;
            }
            if k == 10 {
                after = r.ledger.injected_sources;
            }
        }
        gpu.sync_field(&mut g).unwrap();
        assert!((after / before - 2.0).abs() < 1e-4, "device injection {before} -> {after}");
        assert_eq!(gpu.cfg.sources.power_per_source, base.sources.power_per_source * 2.0);
        assert_eq!(cpu.sources, gpu.sources, "sources (positions, chemistry) agree");
        let total: f64 = cpu.field.e.iter().map(|&v| v as f64).sum();
        let worst = cpu.field.e.iter().zip(&gpu.field.e).map(|(&a, &b)| (a as f64 - b as f64).abs()).fold(0.0, f64::max);
        eprintln!("live tuning: field energy {total:.3}, worst cell difference {worst:.2e}");
        assert!(worst / total.max(1e-9) < 1e-3, "field differs from the CPU reference after the change");
        let l = &gpu.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        assert!((gpu.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5, "ledger closes through the change");
    }

    /// The single-wait path (`--backend gpu-realtime`) against the classic resident path.
    /// Only the uptake request differs (f32 on the device instead of f64 on the host), so
    /// a short run stays close; two single-wait runs are bit-identical; the ledger closes
    /// through commands, births, deaths and several topology groups; and a field copied
    /// back mid-run (which flushes the folded commit) matches a run that never looked.
    #[test]
    fn single_wait_world_matches_resident_and_closes_its_ledger() {
        let Ok(mut classic) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut g1 = GpuBackend::new().unwrap();
        let mut g2 = GpuBackend::new().unwrap();
        g1.set_single_wait(true);
        g2.set_single_wait(true);
        let cfg = || {
            let mut c = vivid_small();
            // Topology changes at most births: several (topology, page) groups per tick.
            c.architecture_mutation.probability_per_birth = 0.5;
            c
        };
        let mut r = fl_core::World::new(cfg()).unwrap();
        let mut a = fl_core::World::new(cfg()).unwrap();
        let mut b = fl_core::World::new(cfg()).unwrap();
        for w in [&mut r, &mut a, &mut b] {
            w.schedule(50, fl_core::Command::EraseDisk { x: 60.0, y: 60.0, radius: 12.0, disposal: fl_core::Disposal::Recycle }).unwrap();
        }
        for _ in 0..24 {
            r.step_inference(fl_core::Inference::Resident(&mut classic)).unwrap();
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
        }
        // Early agreement with the classic path, before chaos amplifies rounding.
        let rel = |x: f64, y: f64| (x - y).abs() / x.abs().max(y.abs()).max(1e-30);
        assert_eq!(r.orgs.len(), a.orgs.len());
        let worst_e = (0..r.orgs.len()).map(|i| rel(r.orgs.energy[i], a.orgs.energy[i])).fold(0.0, f64::max);
        let worst_x = (0..r.orgs.len()).map(|i| (r.orgs.x[i] - a.orgs.x[i]).abs()).fold(0.0, f64::max);
        eprintln!("single-wait vs classic after 24 ticks: organism energy {worst_e:.2e} rel, position {worst_x:.2e}, transferred {:.2e} rel", rel(r.ledger.transferred, a.ledger.transferred));
        assert!(worst_e < 1e-4 && worst_x < 1e-4);
        assert!(rel(r.ledger.transferred, a.ledger.transferred) < 1e-5);
        for k in 24..640 {
            a.step_inference(fl_core::Inference::Resident(&mut g1)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut g2)).unwrap();
            if k % 97 == 0 {
                // Looking at the field must not change the run.
                a.sync_field(&mut g1).unwrap();
            }
        }
        a.sync_field(&mut g1).unwrap();
        b.sync_field(&mut g2).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "two single-wait runs are bit-identical");
        let l = &a.ledger;
        let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
        let relres = (a.total_energy() - expected).abs() / l.transferred.max(1.0);
        let topologies: std::collections::BTreeSet<u8> = a.orgs.genome.iter().map(|g| g.topology).collect();
        eprintln!("single-wait world: births {}, user deaths {}, population {}, topologies {}, residual {:.2e} ({relres:.2e} of transfers)", a.totals.births, a.totals.deaths_user, a.orgs.len(), topologies.len(), l.residual);
        assert!(relres < 1e-5);
        assert!(a.totals.births > 0 && a.totals.deaths_user > 0);
        assert!(topologies.len() > 1, "the run exercised several topology groups");
        assert!(a.orgs.len() > 500);
    }

    /// Sensing and inference are the same kernels on both paths: after one tick from
    /// the same world, every held output is equal bit for bit.
    #[test]
    fn single_wait_sensing_is_bit_identical_on_the_first_tick() {
        let Ok(mut classic) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut fast = GpuBackend::new().unwrap();
        fast.set_single_wait(true);
        let mut a = fl_core::World::new(vivid_small()).unwrap();
        let mut b = fl_core::World::new(vivid_small()).unwrap();
        a.step_inference(fl_core::Inference::Resident(&mut classic)).unwrap();
        b.step_inference(fl_core::Inference::Resident(&mut fast)).unwrap();
        assert_eq!(a.orgs.held, b.orgs.held, "identical sensing and inference on the first tick");
    }

    fn tiny_world() -> Config {
        let mut c = Config::reference();
        c.world.width = 16.0;
        c.world.height = 16.0;
        c.world.grid_width = 16;
        c.world.grid_height = 16;
        c.population.initial_count = 24;
        c.population.maximum_count = 64;
        c.population.initial_energy = 4.0;
        c.sources.count = 1;
        c.sources.power_per_source = 8.0;
        c.sources.sigma_world_units = 2.0;
        c.sensors.maximum_scale_world_units = 4.0;
        c.controller.hidden_widths = vec![8];
        c
    }

    #[test]
    fn profiling_does_not_change_resident_results() {
        let Ok(mut off) = GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let Ok(mut on) = GpuBackend::new() else {
            eprintln!("second GPU device unavailable; skipping");
            return;
        };
        assert!(!off.profiling(), "profiling is off unless requested");
        on.set_profiling(true);
        let cfg = tiny_world();
        let mut a = fl_core::World::new(cfg.clone()).unwrap();
        let mut b = fl_core::World::new(cfg).unwrap();
        b.set_profiling(true);
        for _ in 0..6 {
            a.step_inference(fl_core::Inference::Resident(&mut off)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut on)).unwrap();
        }
        a.sync_field(&mut off).unwrap();
        b.sync_field(&mut on).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "timestamp queries must not change field or organism bits");
        assert_eq!(a.ledger, b.ledger);
        assert_eq!(a.orgs, b.orgs);
        if on.timestamps_supported() {
            assert!(
                on.stats.resident_last.finish_timestamp_span_ms.is_some(),
                "finish already waits for readback; the timestamp span is read in that same wait"
            );
        } else {
            assert!(on.stats.resident_last.finish_timestamp_span_ms.is_none());
            assert!(on.stats.resident_last.uptake_timestamp_span_ms.is_none());
        }
    }
}
