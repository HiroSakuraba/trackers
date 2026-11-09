//! Opt-in execution profiler.
//!
//! Disabled by default. While disabled, [`StageProfiler::start`] does not read
//! the clock. The profiler is execution telemetry: it is not an input to the
//! tick, and [`crate::World::state_hash`] ignores it.
//!
//! CPU stages are `std::time::Instant` wall spans. GPU submit/wait spans are
//! CPU time blocked on a submission the step already waited for; they are not
//! GPU kernel time. Optional `TIMESTAMP_QUERY` spans are a separate clock and
//! are never added to the CPU wall totals.

use serde_json::{json, Value};

/// CPU-side stages. Values are wall time, except the thread-sum counters on
/// [`StageProfiler`], which are called out as non-additive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CpuStage {
    /// Sensor pyramid build and per-organism sampling on the CPU.
    Sensing = 0,
    /// CPU field transport, sources, decay, tiny clearing, deposits, and
    /// organism chemistry settlement (toxicity, maintenance).
    ChemistryField = 1,
    /// CPU scalar network `forward` only.
    Networks = 2,
    /// Host copies, command recording, `queue.submit`, and waits the step
    /// already performed. Not GPU kernel time.
    TransfersSync = 3,
    /// Candidate setup, motion/emission/reproduction bookkeeping, validation,
    /// publication.
    Other = 4,
    /// Wall time inside an external backend call (`evaluate` or fused
    /// `sense_and_infer`). For a GPU backend this is a device call (host work
    /// plus submit/wait). For a CPU reference backend it is CPU work that the
    /// world cannot see inside.
    ExternalEvaluate = 5,
    /// Headless frame rasterization. Outside the step.
    Render = 6,
    /// Checkpoint encoding. Outside the step.
    Checkpoint = 7,
    /// Copy of a resident field back to the host after the run, or for a
    /// frame/checkpoint. Outside the step wall. The wait is not kernel time.
    FieldDownload = 8,
}

pub const CPU_STAGE_COUNT: usize = 9;

impl CpuStage {
    pub const ALL: [CpuStage; CPU_STAGE_COUNT] = [
        CpuStage::Sensing,
        CpuStage::ChemistryField,
        CpuStage::Networks,
        CpuStage::TransfersSync,
        CpuStage::Other,
        CpuStage::ExternalEvaluate,
        CpuStage::Render,
        CpuStage::Checkpoint,
        CpuStage::FieldDownload,
    ];

    pub fn id(self) -> &'static str {
        match self {
            CpuStage::Sensing => "sensing",
            CpuStage::ChemistryField => "chemistry_field",
            CpuStage::Networks => "networks",
            CpuStage::TransfersSync => "transfers_sync",
            CpuStage::Other => "other",
            CpuStage::ExternalEvaluate => "external_evaluate",
            CpuStage::Render => "render",
            CpuStage::Checkpoint => "checkpoint",
            CpuStage::FieldDownload => "field_download",
        }
    }

    fn index(self) -> usize {
        self as u8 as usize
    }

    /// Stages that sit inside one committed step's wall time.
    pub fn in_step(self) -> bool {
        matches!(
            self,
            CpuStage::Sensing
                | CpuStage::ChemistryField
                | CpuStage::Networks
                | CpuStage::TransfersSync
                | CpuStage::Other
                | CpuStage::ExternalEvaluate
        )
    }
}

/// Clock source for a counter. These clocks are not interchangeable.
pub mod clock {
    /// `std::time::Instant` around CPU work.
    pub const CPU_WALL: &str = "cpu_wall";
    /// CPU wall time spent in `queue.submit` plus a wait the step already
    /// needed for readback. This includes GPU execution only as observed from
    /// the CPU. It is not GPU kernel time.
    pub const CPU_WALL_BLOCKED_ON_GPU: &str = "cpu_wall_blocked_on_gpu";
    /// CPU wall time through `queue.submit` only. The GPU is not waited on,
    /// so the GPU work may still be in flight. Not kernel time.
    pub const CPU_WALL_SUBMIT_NO_WAIT: &str = "cpu_wall_submit_no_wait";
    /// Delta of a `wgpu::Features::TIMESTAMP_QUERY` pair (first stamped pass
    /// to last stamped pass in that submission), converted with the device
    /// timestamp period. Not CPU time. Not an exclusive kernel trace.
    pub const GPU_TIMESTAMP_QUERY: &str = "gpu_timestamp_query";
    /// A count. `samples` is the total, not a duration.
    pub const COUNT: &str = "count";
}

/// A clock sample. When profiling is off this does not call `Instant::now`.
#[derive(Clone, Copy)]
pub struct Stamp {
    #[cfg(not(target_arch = "wasm32"))]
    t: Option<std::time::Instant>,
}

impl Stamp {
    #[inline(always)]
    pub fn start(enabled: bool) -> Self {
        Self::from_enabled(enabled)
    }

    /// `ON` is a const so the CPU reference sense loop can compile the off
    /// path with no per-organism clock read.
    #[inline(always)]
    pub fn capture<const ON: bool>() -> Self {
        if ON {
            Self::from_enabled(true)
        } else {
            Self::none()
        }
    }

    #[inline(always)]
    fn from_enabled(enabled: bool) -> Self {
        if !enabled {
            return Self::none();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                t: Some(std::time::Instant::now()),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::none()
        }
    }

    #[inline(always)]
    fn none() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            t: None,
        }
    }

    #[inline(always)]
    pub fn elapsed_ns(self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.t
                .map(|t| t.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
                .unwrap_or(0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }
}

/// Split one wall interval in proportion to two non-negative weights.
///
/// Used for the parallel CPU reference sense loop, where sensing and network
/// `forward` are interleaved per organism. The weights are summed per-thread
/// samples; the two returned parts add up to `wall_ns` and must not be added
/// again to the thread sums.
pub fn proportion(wall_ns: u64, weight_a: u64, weight_b: u64) -> (u64, u64) {
    let denom = weight_a.saturating_add(weight_b);
    if wall_ns == 0 || denom == 0 || weight_b == 0 {
        return (wall_ns, 0);
    }
    if weight_a == 0 {
        return (0, wall_ns);
    }
    let a = ((u128::from(wall_ns) * u128::from(weight_a)) / u128::from(denom)) as u64;
    (a, wall_ns - a)
}

/// Pieces of [`CpuStage::Other`]. They are already included in that stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OtherPart {
    /// Candidate clone, command application.
    Setup = 0,
    /// Copy fused sense results into the per-organism table.
    SenseScatter = 1,
    /// Store held controller outputs.
    HeldStore = 2,
    /// Copy resident uptake rows into settlements.
    UptakeScatter = 3,
    /// Motion, emission, and reproduction proposals.
    LifecycleMotion = 4,
    /// Births, deaths, retain, and child append.
    LifecycleCommit = 5,
    /// Invariant checks and publication of the candidate.
    Publish = 6,
}

pub const OTHER_PART_COUNT: usize = 7;

impl OtherPart {
    pub const ALL: [OtherPart; OTHER_PART_COUNT] = [
        OtherPart::Setup,
        OtherPart::SenseScatter,
        OtherPart::HeldStore,
        OtherPart::UptakeScatter,
        OtherPart::LifecycleMotion,
        OtherPart::LifecycleCommit,
        OtherPart::Publish,
    ];

    pub fn id(self) -> &'static str {
        match self {
            OtherPart::Setup => "setup",
            OtherPart::SenseScatter => "sense_scatter",
            OtherPart::HeldStore => "held_store",
            OtherPart::UptakeScatter => "uptake_scatter",
            OtherPart::LifecycleMotion => "lifecycle_motion",
            OtherPart::LifecycleCommit => "lifecycle_commit",
            OtherPart::Publish => "publish",
        }
    }

    fn index(self) -> usize {
        self as u8 as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfileTotals {
    pub ticks: u64,
    pub ns: [u64; CPU_STAGE_COUNT],
    pub entries: [u64; CPU_STAGE_COUNT],
    pub sensing_thread_ns: u64,
    pub networks_thread_ns: u64,
    pub other_part_ns: [u64; OTHER_PART_COUNT],
}

impl ProfileTotals {
    pub fn delta(self, earlier: Self) -> Self {
        let mut ns = [0u64; CPU_STAGE_COUNT];
        let mut entries = [0u64; CPU_STAGE_COUNT];
        for i in 0..CPU_STAGE_COUNT {
            ns[i] = self.ns[i].saturating_sub(earlier.ns[i]);
            entries[i] = self.entries[i].saturating_sub(earlier.entries[i]);
        }
        Self {
            ticks: self.ticks.saturating_sub(earlier.ticks),
            ns,
            entries,
            sensing_thread_ns: self
                .sensing_thread_ns
                .saturating_sub(earlier.sensing_thread_ns),
            networks_thread_ns: self
                .networks_thread_ns
                .saturating_sub(earlier.networks_thread_ns),
            other_part_ns: {
                let mut p = [0u64; OTHER_PART_COUNT];
                for i in 0..OTHER_PART_COUNT {
                    p[i] = self.other_part_ns[i].saturating_sub(earlier.other_part_ns[i]);
                }
                p
            },
        }
    }
}

/// Accumulated stage samples for one run (or one metrics interval).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageProfiler {
    enabled: bool,
    ticks: u64,
    total_ns: [u64; CPU_STAGE_COUNT],
    entries: [u64; CPU_STAGE_COUNT],
    /// Sum of per-thread sensing samples inside the CPU reference loop.
    /// Overlaps across workers. Not wall time. Not added to [`CpuStage::Sensing`].
    sensing_thread_ns: u64,
    /// Sum of per-thread `forward` samples. Same caveat as `sensing_thread_ns`.
    networks_thread_ns: u64,
    /// Breakdown of [`CpuStage::Other`]. Sum of these is that stage, not an extra cost.
    other_part_ns: [u64; OTHER_PART_COUNT],
}

impl Default for StageProfiler {
    fn default() -> Self {
        Self::disabled()
    }
}

impl StageProfiler {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ticks: 0,
            total_ns: [0; CPU_STAGE_COUNT],
            entries: [0; CPU_STAGE_COUNT],
            sensing_thread_ns: 0,
            networks_thread_ns: 0,
            other_part_ns: [0; OTHER_PART_COUNT],
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    pub fn stage_ns(&self, stage: CpuStage) -> u64 {
        self.total_ns[stage.index()]
    }

    pub fn entries(&self, stage: CpuStage) -> u64 {
        self.entries[stage.index()]
    }

    pub fn sensing_thread_ns(&self) -> u64 {
        self.sensing_thread_ns
    }

    pub fn networks_thread_ns(&self) -> u64 {
        self.networks_thread_ns
    }

    pub fn totals(&self) -> ProfileTotals {
        ProfileTotals {
            ticks: self.ticks,
            ns: self.total_ns,
            entries: self.entries,
            sensing_thread_ns: self.sensing_thread_ns,
            networks_thread_ns: self.networks_thread_ns,
            other_part_ns: self.other_part_ns,
        }
    }

    pub fn other_part_ns(&self, part: OtherPart) -> u64 {
        self.other_part_ns[part.index()]
    }

    /// Rebuild a profiler from a totals snapshot (interval deltas).
    pub fn from_totals(enabled: bool, totals: ProfileTotals) -> Self {
        Self {
            enabled,
            ticks: totals.ticks,
            total_ns: totals.ns,
            entries: totals.entries,
            sensing_thread_ns: totals.sensing_thread_ns,
            networks_thread_ns: totals.networks_thread_ns,
            other_part_ns: totals.other_part_ns,
        }
    }

    #[inline(always)]
    pub fn start(&self) -> Stamp {
        Stamp::start(self.enabled)
    }

    #[inline(always)]
    pub fn stop(&mut self, stage: CpuStage, stamp: Stamp) {
        if !self.enabled {
            return;
        }
        self.add_sample(stage, stamp.elapsed_ns());
    }

    /// Record one piece of [`CpuStage::Other`]. The time is included in `Other`
    /// and also in the named part. Disabled profilers do not read the stamp.
    pub fn stop_other(&mut self, part: OtherPart, stamp: Stamp) {
        if !self.enabled {
            return;
        }
        let ns = stamp.elapsed_ns();
        self.add_sample(CpuStage::Other, ns);
        let i = part.index();
        self.other_part_ns[i] = self.other_part_ns[i].saturating_add(ns);
    }

    pub fn add_sample(&mut self, stage: CpuStage, ns: u64) {
        if !self.enabled {
            return;
        }
        let i = stage.index();
        self.total_ns[i] = self.total_ns[i].saturating_add(ns);
        self.entries[i] = self.entries[i].saturating_add(1);
    }

    pub fn add_thread_ns(&mut self, sensing_ns: u64, networks_ns: u64) {
        if !self.enabled {
            return;
        }
        self.sensing_thread_ns = self.sensing_thread_ns.saturating_add(sensing_ns);
        self.networks_thread_ns = self.networks_thread_ns.saturating_add(networks_ns);
    }

    pub fn mark_tick(&mut self) {
        if self.enabled {
            self.ticks = self.ticks.saturating_add(1);
        }
    }

    pub fn absorb(&mut self, other: &StageProfiler) {
        if !self.enabled {
            return;
        }
        self.ticks = self.ticks.saturating_add(other.ticks);
        for i in 0..CPU_STAGE_COUNT {
            self.total_ns[i] = self.total_ns[i].saturating_add(other.total_ns[i]);
            self.entries[i] = self.entries[i].saturating_add(other.entries[i]);
        }
        self.sensing_thread_ns = self
            .sensing_thread_ns
            .saturating_add(other.sensing_thread_ns);
        self.networks_thread_ns = self
            .networks_thread_ns
            .saturating_add(other.networks_thread_ns);
        for i in 0..OTHER_PART_COUNT {
            self.other_part_ns[i] = self.other_part_ns[i].saturating_add(other.other_part_ns[i]);
        }
    }

    /// Rolled-up CPU wall totals in nanoseconds: sensing, chemistry/field,
    /// networks, transfers/sync, other, render, checkpoint, field download.
    ///
    /// When `external_evaluate_is_device_call` is set, the external-backend
    /// wall is included in transfers/sync. Otherwise it is returned separately
    /// as the last value and is not mixed into transfers.
    pub fn buckets_ns(&self, external_evaluate_is_device_call: bool) -> Buckets {
        let ext = self.stage_ns(CpuStage::ExternalEvaluate);
        let mut transfers = self.stage_ns(CpuStage::TransfersSync);
        let mut backend_cpu = 0u64;
        if external_evaluate_is_device_call {
            transfers = transfers.saturating_add(ext);
        } else {
            backend_cpu = ext;
        }
        Buckets {
            sensing: self.stage_ns(CpuStage::Sensing),
            chemistry_field: self.stage_ns(CpuStage::ChemistryField),
            networks: self.stage_ns(CpuStage::Networks),
            transfers_sync: transfers,
            other: self.stage_ns(CpuStage::Other),
            render: self.stage_ns(CpuStage::Render),
            checkpoint: self.stage_ns(CpuStage::Checkpoint),
            field_download: self.stage_ns(CpuStage::FieldDownload),
            backend_cpu,
        }
    }

    pub fn in_step_ns(&self) -> u64 {
        CpuStage::ALL
            .into_iter()
            .filter(|s| s.in_step())
            .map(|s| self.stage_ns(s))
            .fold(0u64, u64::saturating_add)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buckets {
    pub sensing: u64,
    pub chemistry_field: u64,
    pub networks: u64,
    pub transfers_sync: u64,
    pub other: u64,
    pub render: u64,
    pub checkpoint: u64,
    pub field_download: u64,
    /// External backend wall kept out of transfers when the backend is CPU.
    pub backend_cpu: u64,
}

/// One non-additive GPU or host-side annotation. Not part of the CPU wall sum.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuCounter {
    pub id: &'static str,
    pub clock: &'static str,
    pub total_ms: f64,
    pub samples: u64,
    pub note: &'static str,
}

/// Run metadata the profiler does not own.
#[derive(Clone, Debug)]
pub struct ProfileMeta {
    pub backend: String,
    pub workers: usize,
    pub simulation_seconds: f64,
    pub wall_seconds: f64,
    /// Sum of per-step `TickTiming::total_ms` over `ticks` (CPU wall of `step`).
    pub step_wall_ms: f64,
    pub population_final: usize,
    pub controller_evaluations: u64,
    /// `Some` when a GPU device was opened: whether `TIMESTAMP_QUERY` was available.
    pub timestamp_queries_supported: Option<bool>,
    /// GPU backends fold `external_evaluate` into transfers/sync because that
    /// wall is submit/wait plus host packing, not a CPU `forward`.
    pub external_evaluate_is_device_call: bool,
}

fn ms_from_ns(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn per_tick_ms(total_ms: f64, ticks: u64) -> f64 {
    if ticks == 0 {
        0.0
    } else {
        total_ms / ticks as f64
    }
}

fn stage_obj(total_ns: u64, entries: u64, ticks: u64) -> Value {
    let total_ms = ms_from_ns(total_ns);
    json!({
        "total_ns": total_ns,
        "total_ms": total_ms,
        "per_tick_ms": per_tick_ms(total_ms, ticks),
        "entries": entries,
    })
}

impl StageProfiler {
    /// Machine-readable profile. Times are milliseconds (`total_ms`) plus the
    /// integer nanosecond sums (`total_ns`) the milliseconds were derived from.
    pub fn document(&self, meta: &ProfileMeta, gpu: &[GpuCounter]) -> Value {
        let ticks = self.ticks;
        let b = self.buckets_ns(meta.external_evaluate_is_device_call);
        let accounted_ns = self.in_step_ns();
        let accounted_ms = ms_from_ns(accounted_ns);
        let unaccounted_ms = meta.step_wall_ms - accounted_ms;
        let mut other_parts = serde_json::Map::new();
        let mut other_part_sum = 0u64;
        for part in OtherPart::ALL {
            let ns = self.other_part_ns(part);
            other_part_sum = other_part_sum.saturating_add(ns);
            other_parts.insert(part.id().to_string(), stage_obj(ns, 0, ticks));
        }
        let mut stages = serde_json::Map::new();
        for stage in CpuStage::ALL {
            stages.insert(
                stage.id().to_string(),
                stage_obj(self.stage_ns(stage), self.entries(stage), ticks),
            );
        }
        json!({
            "schema": "frankenlife.stage-profile.v1",
            "profiling_enabled": self.enabled,
            "time_unit": "milliseconds",
            "ticks": ticks,
            "backend": meta.backend,
            "workers": meta.workers,
            "simulation_seconds": meta.simulation_seconds,
            "wall_seconds": meta.wall_seconds,
            "population_final": meta.population_final,
            "controller_evaluations": meta.controller_evaluations,
            "budget_ms_per_step_at_128hz": 1000.0 / 128.0,
            "step_wall_ms": {
                "total_ms": meta.step_wall_ms,
                "per_tick_ms": per_tick_ms(meta.step_wall_ms, ticks),
                "note": "Sum of the existing per-step wall clock (TickTiming.total_ms). This is the denominator for the CPU stage split.",
            },
            "cpu_wall_ms": {
                "sensing": stage_obj(b.sensing, self.entries(CpuStage::Sensing), ticks),
                "chemistry_field": stage_obj(b.chemistry_field, self.entries(CpuStage::ChemistryField), ticks),
                "networks": stage_obj(b.networks, self.entries(CpuStage::Networks), ticks),
                "transfers_sync": {
                    "total_ns": b.transfers_sync,
                    "total_ms": ms_from_ns(b.transfers_sync),
                    "per_tick_ms": per_tick_ms(ms_from_ns(b.transfers_sync), ticks),
                    "includes_external_device_call": meta.external_evaluate_is_device_call,
                    "external_evaluate_total_ns": self.stage_ns(CpuStage::ExternalEvaluate),
                    "external_evaluate_total_ms": ms_from_ns(self.stage_ns(CpuStage::ExternalEvaluate)),
                    "note": "Host copies, command recording, queue.submit, and waits the step already performed. When includes_external_device_call is true, the external backend wall is included. None of this is GPU kernel time.",
                },
                "other": stage_obj(b.other, self.entries(CpuStage::Other), ticks),
                "render": stage_obj(b.render, self.entries(CpuStage::Render), ticks),
                "checkpoint": stage_obj(b.checkpoint, self.entries(CpuStage::Checkpoint), ticks),
                "field_download": stage_obj(b.field_download, self.entries(CpuStage::FieldDownload), ticks),
                "backend_cpu": stage_obj(b.backend_cpu, self.entries(CpuStage::ExternalEvaluate), ticks),
            },
            "cpu_wall_stages_raw": stages,
            "cpu_wall_accounted_step_ms": {
                "total_ms": accounted_ms,
                "per_tick_ms": per_tick_ms(accounted_ms, ticks),
                "note": "Sum of in-step CPU wall stages (sensing, chemistry/field, networks, transfers/sync, other, external_evaluate). Render, checkpoint, and field download are outside the step.",
            },
            "unaccounted_step_ms": {
                "total_ms": unaccounted_ms,
                "per_tick_ms": per_tick_ms(unaccounted_ms, ticks),
                "note": "step_wall_ms minus in-step CPU stages. Gaps between timers, and the cost of the timer calls themselves. A small negative value is overlap or rounding, not a second clock.",
            },
            "other_breakdown": {
                "additive_with_cpu_wall": false,
                "included_in": "cpu_wall_ms.other",
                "sum_total_ns": other_part_sum,
                "note": "Parts of the other CPU wall. They sum to that stage; do not add them again. setup is the candidate clone and commands. lifecycle_motion is per-organism motion, emission, and reproduction. lifecycle_commit is births, deaths, and retain.",
                "parts": other_parts,
            },
            "cpu_thread_sum_ms": {
                "additive": false,
                "sensing_total_ns": self.sensing_thread_ns,
                "networks_total_ns": self.networks_thread_ns,
                "sensing_total_ms": ms_from_ns(self.sensing_thread_ns),
                "networks_total_ms": ms_from_ns(self.networks_thread_ns),
                "note": "Summed across worker threads inside the CPU reference sense/forward loop. Threads overlap, so this is not wall time and is not added to cpu_wall_ms. On that path the sensing vs networks wall split follows these proportions.",
            },
            "gpu": {
                "timestamp_queries_supported": meta.timestamp_queries_supported,
                "feature_used_when_supported": "wgpu::Features::TIMESTAMP_QUERY",
                "features_not_required": ["TIMESTAMP_QUERY_INSIDE_ENCODERS", "TIMESTAMP_QUERY_INSIDE_PASSES"],
                "do_not_add_to_cpu_wall": true,
                "counters": gpu.iter().map(|c| {
                    let per_tick = per_tick_ms(c.total_ms, ticks);
                    let per_sample = if c.samples == 0 { 0.0 } else { c.total_ms / c.samples as f64 };
                    json!({
                        "id": c.id,
                        "clock": c.clock,
                        "additive": false,
                        "total_ms": c.total_ms,
                        "samples": c.samples,
                        "per_tick_ms": per_tick,
                        "per_sample_ms": per_sample,
                        "note": c.note,
                    })
                }).collect::<Vec<_>>(),
            },
            "caveats": CAVEATS,
        })
    }

    /// Short console summary. GPU counters are listed only to label their clock.
    pub fn summary(&self, meta: &ProfileMeta, gpu: &[GpuCounter]) -> String {
        let b = self.buckets_ns(meta.external_evaluate_is_device_call);
        let accounted_ms = ms_from_ns(self.in_step_ns());
        let unaccounted = meta.step_wall_ms - accounted_ms;
        let step = per_tick_ms(meta.step_wall_ms, self.ticks);
        let line = |ns: u64| format!("{:.3}", per_tick_ms(ms_from_ns(ns), self.ticks));
        let mut gpu_lines: Vec<String> = Vec::new();
        for c in gpu {
            if c.samples == 0 && c.total_ms == 0.0 {
                continue;
            }
            let shown = if c.clock == clock::GPU_TIMESTAMP_QUERY {
                if c.samples == 0 {
                    continue;
                }
                format!(
                    "{:.3} ({}/{} samples)",
                    c.total_ms / c.samples as f64,
                    c.samples,
                    self.ticks
                )
            } else if c.clock == clock::COUNT {
                format!("{:.2}/tick ({} total)", c.samples as f64 / self.ticks.max(1) as f64, c.samples)
            } else {
                format!("{:.3}", per_tick_ms(c.total_ms, self.ticks))
            };
            gpu_lines.push(format!("{} {} [{}]", c.id, shown, c.clock));
        }
        let gpu_text = if gpu_lines.is_empty() {
            "none recorded".to_string()
        } else {
            gpu_lines.join(" | ")
        };
        let ts = match meta.timestamp_queries_supported {
            Some(true) => "TIMESTAMP_QUERY supported",
            Some(false) => "TIMESTAMP_QUERY not supported; timestamp spans omitted",
            None => "no GPU device",
        };
        format!(
            "stage profile: {ticks} ticks | step wall {step:.3} ms/tick | budget 7.8125 ms/tick at 128 Hz | {ts}\n\
             cpu wall ms/tick: sensing {s} | chemistry/field {c} | networks {n} | transfers/sync {t} | other {o} | render {r} | checkpoint {k} | field download {f}{backend}\n\
             unaccounted step ms/tick: {u:.3} (gaps between timers; Instant samples sit inside the open stage or this gap)\n\
             gpu (not added to cpu wall): {gpu_text}\n\
             transfers/sync and blocked-on-gpu clocks are submit and wait, not GPU kernel time. A gpu_timestamp_query span is the optional TIMESTAMP_QUERY delta for passes that already waited; it is not kernel-exclusive and is not a GTX 1080 measurement unless this process is that GPU.\n\
             thread-sum ms/tick (summed across workers, not wall, not added): sensing {tsum:.3} | networks {nsum:.3}\n\
             other parts ms/tick (already inside other): setup {os} | sense scatter {ss} | held {hs} | uptake scatter {us} | lifecycle motion {lm} | lifecycle commit {lc} | publish {pb}",
            os = line(self.other_part_ns(OtherPart::Setup)),
            ss = line(self.other_part_ns(OtherPart::SenseScatter)),
            hs = line(self.other_part_ns(OtherPart::HeldStore)),
            us = line(self.other_part_ns(OtherPart::UptakeScatter)),
            lm = line(self.other_part_ns(OtherPart::LifecycleMotion)),
            lc = line(self.other_part_ns(OtherPart::LifecycleCommit)),
            pb = line(self.other_part_ns(OtherPart::Publish)),
            ticks = self.ticks,
            s = line(b.sensing),
            c = line(b.chemistry_field),
            n = line(b.networks),
            t = line(b.transfers_sync),
            o = line(b.other),
            r = line(b.render),
            k = line(b.checkpoint),
            f = line(b.field_download),
            backend = if b.backend_cpu > 0 {
                format!(" | backend cpu {}", line(b.backend_cpu))
            } else {
                String::new()
            },
            u = per_tick_ms(unaccounted, self.ticks),
            tsum = per_tick_ms(ms_from_ns(self.sensing_thread_ns), self.ticks),
            nsum = per_tick_ms(ms_from_ns(self.networks_thread_ns), self.ticks),
        )
    }
}

const CAVEATS: [&str; 8] = [
    "time_unit is milliseconds. total_ns is the integer sum; total_ms = total_ns / 1e6. per_tick_ms divides by completed ticks.",
    "cpu_wall stages are std::time::Instant spans. They estimate the step wall. unaccounted_step_ms is the gap, including profiler overhead. Profiling adds those Instant calls only while enabled.",
    "On the parallel CPU reference path, sensing versus networks is one wall interval split in proportion to per-thread samples. cpu_thread_sum_ms overlaps across workers and is not added to cpu_wall.",
    "transfers_sync and cpu_wall_blocked_on_gpu include queue.submit and any wait the step already used for readback. That wait is not GPU kernel time. cpu_wall_submit_no_wait does not wait; the GPU work may still be in flight.",
    "gpu_timestamp_query is an optional wgpu TIMESTAMP_QUERY delta from the first stamped pass to the last in that submission, times the device timestamp period. It is not CPU time, not a vendor kernel profiler, and must not be added to cpu_wall. It is omitted when the adapter lacks TIMESTAMP_QUERY. TIMESTAMP_QUERY_INSIDE_ENCODERS and TIMESTAMP_QUERY_INSIDE_PASSES are not requested.",
    "Resident begin, deposit, and commit stay submit-without-wait while profiling. Uptake and finish already wait for readback; a timestamp span, when supported, is read in that same wait. No extra device poll is inserted.",
    "The unprofiled path does not record these stages and does not add resident timestamp writes. Enabling profiling does not change tick order, field values, networks, grid, or physics.",
    "A software Vulkan driver or a cloud GPU timing is not a measurement of a GeForce GTX 1080. Real-time at 128 Hz is 7.8125 ms per step.",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profiler_records_nothing_and_does_not_need_a_clock() {
        let mut p = StageProfiler::disabled();
        let s = p.start();
        p.stop(CpuStage::Sensing, s);
        p.add_sample(CpuStage::Networks, 5_000_000);
        p.add_thread_ns(9, 9);
        p.mark_tick();
        assert!(!p.is_enabled());
        assert_eq!(p.ticks(), 0);
        assert_eq!(p.stage_ns(CpuStage::Sensing), 0);
        assert_eq!(p.stage_ns(CpuStage::Networks), 0);
        assert_eq!(p.sensing_thread_ns(), 0);
    }

    #[test]
    fn proportion_partitions_one_wall_interval() {
        assert_eq!(proportion(100, 25, 75), (25, 75));
        assert_eq!(proportion(100, 1, 0), (100, 0));
        assert_eq!(proportion(100, 0, 5), (0, 100));
        assert_eq!(proportion(0, 5, 5), (0, 0));
        let (a, b) = proportion(10, 1, 2);
        assert_eq!(a + b, 10);
    }

    #[test]
    fn document_serializes_clocks_units_and_caveats() {
        let mut p = StageProfiler::disabled();
        p.set_enabled(true);
        p.add_sample(CpuStage::Sensing, 2_000_000);
        p.add_sample(CpuStage::ChemistryField, 4_000_000);
        p.add_sample(CpuStage::Networks, 0);
        p.add_sample(CpuStage::TransfersSync, 1_000_000);
        p.add_sample(CpuStage::ExternalEvaluate, 8_000_000);
        p.add_sample(CpuStage::Other, 500_000);
        p.add_thread_ns(3_000_000, 1_000_000);
        p.mark_tick();
        let meta = ProfileMeta {
            backend: "gpu-resident".into(),
            workers: 4,
            simulation_seconds: 1.0 / 128.0,
            wall_seconds: 0.02,
            step_wall_ms: 20.0,
            population_final: 32,
            controller_evaluations: 32,
            timestamp_queries_supported: Some(true),
            external_evaluate_is_device_call: true,
        };
        let gpu = [GpuCounter {
            id: "field_submit_and_wait",
            clock: clock::CPU_WALL_BLOCKED_ON_GPU,
            total_ms: 9.0,
            samples: 1,
            note: "CPU blocked on a field readback the step already waits for. Not GPU kernel time.",
        }, GpuCounter {
            id: "field_timestamp_span",
            clock: clock::GPU_TIMESTAMP_QUERY,
            total_ms: 7.5,
            samples: 1,
            note: "TIMESTAMP_QUERY span. Not CPU time. Do not add to cpu_wall.",
        }, GpuCounter {
            id: "resident_begin_submit",
            clock: clock::CPU_WALL_SUBMIT_NO_WAIT,
            total_ms: 0.25,
            samples: 1,
            note: "queue.submit only. Does not wait.",
        }];
        let doc = p.document(&meta, &gpu);
        let text = serde_json::to_string(&doc).unwrap();
        assert!(
            text.contains("\"schema\":\"frankenlife.stage-profile.v1\""),
            "{text}"
        );
        assert!(text.contains("\"time_unit\":\"milliseconds\""));
        assert!(text.contains("cpu_wall_blocked_on_gpu"));
        assert!(text.contains("cpu_wall_submit_no_wait"));
        assert!(text.contains("gpu_timestamp_query"));
        assert!(text.contains("not GPU kernel time") || text.contains("Not GPU kernel time"));
        assert!(text.contains("GTX 1080") || text.contains("GeForce GTX 1080"));
        assert!(text.contains("TIMESTAMP_QUERY_INSIDE_ENCODERS"));
        let transfers = &doc["cpu_wall_ms"]["transfers_sync"]["total_ns"];
        assert_eq!(
            transfers.as_u64(),
            Some(9_000_000),
            "device call folded into transfers"
        );
        assert!((doc["cpu_wall_ms"]["sensing"]["total_ms"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((doc["budget_ms_per_step_at_128hz"].as_f64().unwrap() - 7.8125).abs() < 1e-9);
        assert_eq!(
            doc["gpu"]["counters"][1]["clock"].as_str(),
            Some(clock::GPU_TIMESTAMP_QUERY)
        );
        assert_eq!(doc["gpu"]["do_not_add_to_cpu_wall"].as_bool(), Some(true));
        let summary = p.summary(&meta, &gpu);
        assert!(summary.contains("ms/tick"), "{summary}");
        assert!(summary.contains("7.8125"), "{summary}");
        assert!(summary.contains("chemistry/field"), "{summary}");
        assert!(summary.contains("transfers/sync"), "{summary}");
        assert!(summary.contains("not GPU kernel time"), "{summary}");
        assert!(summary.contains("GTX 1080"), "{summary}");
        assert!(summary.contains("field_submit_and_wait"), "{summary}");
        assert!(summary.contains("gpu_timestamp_query"), "{summary}");
        // CPU reference backend must not pretend the external call was a device wait.
        let mut meta_cpu = meta.clone();
        meta_cpu.external_evaluate_is_device_call = false;
        let doc_cpu = p.document(&meta_cpu, &[]);
        assert_eq!(
            doc_cpu["cpu_wall_ms"]["transfers_sync"]["total_ns"].as_u64(),
            Some(1_000_000)
        );
        assert_eq!(
            doc_cpu["cpu_wall_ms"]["backend_cpu"]["total_ns"].as_u64(),
            Some(8_000_000)
        );
    }

    #[test]
    fn delta_totals_subtract() {
        let mut p = StageProfiler::disabled();
        p.set_enabled(true);
        p.add_sample(CpuStage::Sensing, 10);
        p.mark_tick();
        let a = p.totals();
        p.add_sample(CpuStage::Sensing, 5);
        p.mark_tick();
        let d = p.totals().delta(a);
        assert_eq!(d.ticks, 1);
        assert_eq!(d.ns[CpuStage::Sensing.index()], 5);
        assert_eq!(d.entries[CpuStage::Sensing.index()], 1);
    }
}
