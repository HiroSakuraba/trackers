//! The world state and the ordered, transactional tick (plan §8).

use std::sync::Arc;

use serde::Serialize;

use crate::chemistry::{drag_force_coefficient, eta, grant_scale, rate_to_probability, settle, split_energy};
use crate::config::{CapacityPolicy, Config};
use crate::field::{min_image, wrap, Field, Stencil};
use crate::genome::{founder, mutate, Genome, MutationRecord, Weights};
use crate::nn::{forward, forward_state, is_recurrent, sigmoid, INPUT_WIDTH, MAX_HIDDEN, MEMORY_WIDTH, OUTPUT_WIDTH};
use crate::profile::{proportion, CpuStage, OtherPart, StageProfiler, Stamp};
use crate::rng::{Lane, SemanticRng};
use crate::sensors::{intrinsic, observe, Pyramid, PLANES};
use crate::sources::{init_sources, inject, isotropic_unit, Source};
use crate::CHANNELS;

/// Wall-clock stopwatch for execution telemetry. On `wasm32-unknown-unknown`,
/// where `std::time::Instant` is unavailable, it reports zero.
#[derive(Clone, Copy)]
struct Stopwatch(#[cfg(not(target_arch = "wasm32"))] std::time::Instant);

impl Stopwatch {
    fn start() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        return Stopwatch(std::time::Instant::now());
        #[cfg(target_arch = "wasm32")]
        return Stopwatch();
    }
    fn ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        return self.0.elapsed().as_secs_f64() * 1e3;
        #[cfg(target_arch = "wasm32")]
        return 0.0;
    }
}

/// Raw controller output layout (plan §6.3).
pub mod out {
    pub const SPLIT: usize = 0;
    pub const SHARE: usize = 1;
    pub const EMIT_GATE: usize = 2;
    pub const EMIT_DIR: usize = 3; // 32 entries
    pub const EMIT_RATE: usize = 35;
}

/// Memory of a newborn, and the stand-in input of a memoryless world.
pub const ZERO_MEMORY: [f32; MEMORY_WIDTH] = [0.0; MEMORY_WIDTH];

/// Hot organism state, structure-of-arrays, always ordered by stable ID.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Organisms {
    pub id: Vec<u64>,
    pub lineage: Vec<u64>,
    pub generation: Vec<u32>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub energy: Vec<f64>,
    pub birth_tick: Vec<u64>,
    pub cooldown_until: Vec<u64>,
    /// Previous committed `(E_next - E_prev) / dt`.
    pub energy_change_rate: Vec<f64>,
    pub phase: Vec<u32>,
    pub held: Vec<[f32; OUTPUT_WIDTH]>,
    pub held_tick: Vec<u64>,
    pub held_valid: Vec<bool>,
    /// Recurrent memory, updated at each evaluation. The column exists only in a world of recurrent
    /// genomes (one entry per organism); in any other world it stays empty and costs nothing.
    pub memory: Vec<[f32; MEMORY_WIDTH]>,
    pub genome: Vec<Arc<Genome>>,
}

impl Organisms {
    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.id.binary_search(&id).ok()
    }

    fn retain(&mut self, keep: &[bool]) {
        macro_rules! r {
            ($($f:ident),*) => {$(
                let mut i = 0;
                self.$f.retain(|_| { let k = keep[i]; i += 1; k });
            )*};
        }
        r!(id, lineage, generation, x, y, vx, vy, energy, birth_tick, cooldown_until, energy_change_rate, phase, held, held_tick, held_valid, memory, genome);
    }

    /// Copy every column of `src` except the genomes into `self`, keeping allocated
    /// capacity, so a tick does not allocate a fresh organism table. A tick's candidate
    /// reads genomes from the committed table (they are immutable and indexed alike
    /// until the commit), which saves touching every genome's reference count.
    fn copy_from_without_genomes(&mut self, src: &Organisms) {
        macro_rules! c {
            ($($f:ident),*) => {$(
                self.$f.clear();
                self.$f.extend_from_slice(&src.$f);
            )*};
        }
        c!(id, lineage, generation, x, y, vx, vy, energy, birth_tick, cooldown_until, energy_change_rate, phase, held, held_tick, held_valid, memory);
        self.genome.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, id: u64, lineage: u64, generation: u32, x: f64, y: f64, energy: f64, birth_tick: u64, cooldown_until: u64, phase: u32, genome: Arc<Genome>) {
        self.id.push(id);
        self.lineage.push(lineage);
        self.generation.push(generation);
        self.x.push(x);
        self.y.push(y);
        self.vx.push(0.0);
        self.vy.push(0.0);
        self.energy.push(energy);
        self.birth_tick.push(birth_tick);
        self.cooldown_until.push(cooldown_until);
        self.energy_change_rate.push(0.0);
        self.phase.push(phase);
        self.held.push([0.0; OUTPUT_WIDTH]);
        self.held_tick.push(0);
        self.held_valid.push(false);
        if is_recurrent(genome.hidden()) {
            self.memory.push(ZERO_MEMORY);
        }
        self.genome.push(genome);
    }
}

/// How an erase brush disposes of the removed organisms' energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Disposal {
    /// Removed from the modeled world (ledger output `O`).
    Remove,
    /// Dissipated (ledger `D`).
    Dissipate,
    /// Returned to the local field along the genetic affinity.
    Recycle,
}

/// Interventions, applied at the boundary of their recorded committed tick.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Command {
    EraseDisk { x: f64, y: f64, radius: f64, disposal: Disposal },
    /// Inject `amount` energy uniformly over the cells whose centers lie in the disk.
    EnergyBrush { x: f64, y: f64, radius: f64, amount: f64, direction: Option<[f32; CHANNELS]> },
    /// Set one live tuning parameter. It takes effect for the tick that starts at
    /// the command's boundary and stays in the configuration (and checkpoints).
    SetParam { param: Param, value: f64 },
    /// Rotate the chemical direction of one source (or every source) by `degrees`
    /// toward a deterministic random perpendicular direction.
    DriftChemistry { source: Option<u32>, degrees: f64 },
}

/// Configuration values that can be changed while a world runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Param {
    /// `sources.power_per_source` (energy per second per source).
    SourcePower,
    /// `sources.path_speed` (world units per second).
    SourceSpeed,
    /// `world.diffusion`.
    Diffusion,
    /// `world.decay_per_second`.
    Decay,
    /// `mutation.weight_probability`.
    MutationWeightProbability,
    /// `mutation.weight_standard_deviation`.
    MutationWeightSd,
    /// `mutation.replacement_probability`.
    MutationReplacementProbability,
    /// `mutation.chemistry_standard_deviation`.
    MutationChemistrySd,
    /// `mutation.sensing_log_scale_standard_deviation`.
    MutationSensingSd,
}

impl Param {
    pub const ALL: [Param; 9] = [
        Param::SourcePower,
        Param::SourceSpeed,
        Param::Diffusion,
        Param::Decay,
        Param::MutationWeightProbability,
        Param::MutationWeightSd,
        Param::MutationReplacementProbability,
        Param::MutationChemistrySd,
        Param::MutationSensingSd,
    ];

    pub fn get(self, cfg: &Config) -> f64 {
        match self {
            Param::SourcePower => cfg.sources.power_per_source,
            Param::SourceSpeed => cfg.sources.path_speed,
            Param::Diffusion => cfg.world.diffusion,
            Param::Decay => cfg.world.decay_per_second,
            Param::MutationWeightProbability => cfg.mutation.weight_probability,
            Param::MutationWeightSd => cfg.mutation.weight_standard_deviation,
            Param::MutationReplacementProbability => cfg.mutation.replacement_probability,
            Param::MutationChemistrySd => cfg.mutation.chemistry_standard_deviation,
            Param::MutationSensingSd => cfg.mutation.sensing_log_scale_standard_deviation,
        }
    }

    fn set(self, cfg: &mut Config, v: f64) {
        match self {
            Param::SourcePower => cfg.sources.power_per_source = v,
            Param::SourceSpeed => cfg.sources.path_speed = v,
            Param::Diffusion => cfg.world.diffusion = v,
            Param::Decay => cfg.world.decay_per_second = v,
            Param::MutationWeightProbability => cfg.mutation.weight_probability = v,
            Param::MutationWeightSd => cfg.mutation.weight_standard_deviation = v,
            Param::MutationReplacementProbability => cfg.mutation.replacement_probability = v,
            Param::MutationChemistrySd => cfg.mutation.chemistry_standard_deviation = v,
            Param::MutationSensingSd => cfg.mutation.sensing_log_scale_standard_deviation = v,
        }
    }

    /// Stable one-byte tag used by checkpoints.
    pub fn tag(self) -> u8 {
        Param::ALL.iter().position(|&p| p == self).expect("listed") as u8
    }

    pub fn from_tag(t: u8) -> Option<Param> {
        Param::ALL.get(t as usize).copied()
    }

    pub fn label(self) -> &'static str {
        match self {
            Param::SourcePower => "source power",
            Param::SourceSpeed => "source speed",
            Param::Diffusion => "diffusion",
            Param::Decay => "decay",
            Param::MutationWeightProbability => "mutation weight probability",
            Param::MutationWeightSd => "mutation weight sd",
            Param::MutationReplacementProbability => "mutation replacement probability",
            Param::MutationChemistrySd => "mutation chemistry sd",
            Param::MutationSensingSd => "mutation sensing sd",
        }
    }

    /// Reject values the configuration would reject, plus a cap on diffusion
    /// subcycles so one key press cannot stall the transport.
    pub fn check(self, cfg: &Config, v: f64) -> Result<(), String> {
        if !v.is_finite() || v < 0.0 {
            return Err(format!("{} must be finite and >= 0", self.label()));
        }
        match self {
            Param::MutationWeightProbability | Param::MutationReplacementProbability if v > 1.0 => Err(format!("{} must be in [0,1]", self.label())),
            Param::Diffusion => {
                let (hx, hy) = cfg.cell_size();
                let dt = cfg.dt();
                let sub = (v * dt / (hx * hx) + v * dt / (hy * hy)) / 0.5;
                if sub > 256.0 {
                    Err(format!("diffusion {v} needs more than 256 transport subcycles per tick"))
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScheduledCommand {
    pub tick: u64,
    pub seq: u64,
    pub command: Command,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Dissipation {
    pub decay: f64,
    pub uptake_inefficiency: f64,
    pub toxicity: f64,
    pub maintenance: f64,
    pub motion: f64,
    pub emission: f64,
    pub birth: f64,
    pub death: f64,
    pub tiny_reservoir: f64,
}

impl Dissipation {
    pub fn total(&self) -> f64 {
        self.decay + self.uptake_inefficiency + self.toxicity + self.maintenance + self.motion + self.emission + self.birth + self.death + self.tiny_reservoir
    }
    fn add(&mut self, o: &Dissipation) {
        self.decay += o.decay;
        self.uptake_inefficiency += o.uptake_inefficiency;
        self.toxicity += o.toxicity;
        self.maintenance += o.maintenance;
        self.motion += o.motion;
        self.emission += o.emission;
        self.birth += o.birth;
        self.death += o.death;
        self.tiny_reservoir += o.tiny_reservoir;
    }
}

/// `E_{n+1} - E_n = I - O - D + residual` for one committed tick.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TickLedger {
    pub energy_before: f64,
    pub energy_after: f64,
    pub field_energy_after: f64,
    pub organism_energy_after: f64,
    pub injected_sources: f64,
    pub injected_commands: f64,
    pub removed: f64,
    pub dissipated: Dissipation,
    /// Internal transfers (uptake grants + emission + child endowments), for relative residuals.
    pub transferred: f64,
    pub residual: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct CumulativeLedger {
    pub initial_endowment: f64,
    pub injected: f64,
    pub removed: f64,
    pub dissipated: Dissipation,
    pub transferred: f64,
    pub residual: f64,
    pub max_abs_tick_residual: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TickCounts {
    pub births: u32,
    pub deaths_exhaustion: u32,
    pub deaths_user: u32,
    pub reproduction_attempts: u32,
    pub rejected_births_viability: u32,
    pub rejected_births_mutation: u32,
    pub rejected_births_capacity: u32,
    pub topology_mutations: u32,
    pub controller_evaluations: u32,
    pub parameter_evaluations: u64,
    pub emitting: u32,
    pub cone_projections: u32,
    pub transport_subcycles: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TickTiming {
    pub total_ms: f64,
    pub transport_ms: f64,
    pub sense_and_infer_ms: f64,
    /// Part of `sense_and_infer_ms` spent inside an external backend's `evaluate`
    /// (0 for the built-in reference, where inference is fused with sensing).
    pub backend_inference_ms: f64,
    pub exchange_ms: f64,
    pub lifecycle_ms: f64,
    /// Candidate-state setup: cloning the committed world, scratch allocation, commands.
    pub setup_ms: f64,
    /// Phase 10: tiny clearing, invariant checks, ledger, publication.
    pub validate_publish_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TickReport {
    /// The newly committed tick.
    pub tick: u64,
    pub simulation_seconds: f64,
    pub population: usize,
    pub counts: TickCounts,
    pub ledger: TickLedger,
    /// Wall-clock timing: an execution measurement, never an ecological input.
    pub timing: TickTiming,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TickError {
    /// Research capacity policy: the candidate tick was discarded, nothing changed.
    CapacityPaused { tick: u64, live: usize, proposed_births: usize, maximum_count: usize, required_genome_bytes: u64, genome_quota_bytes: u64 },
    /// Numerical fault (`stop-and-preserve`): the committed world is untouched.
    NumericalFault { tick: u64, detail: String },
    Overflow { detail: String },
}

impl std::fmt::Display for TickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_else(|_| format!("{self:?}")))
    }
}

/// Per-organism sensing/controller result for one tick.
/// What later phases need from sensing: kept small (it exists for every organism every tick).
#[derive(Clone, Debug)]
struct Sense {
    density0: f64,
    out: [f32; OUTPUT_WIDTH],
    evaluated: bool,
    /// Next recurrent memory of a due evaluation. Boxed so a memoryless world pays eight
    /// bytes per organism, not sixty-four, in this table that every tick walks.
    mem: Option<Box<[f32; MEMORY_WIDTH]>>,
    /// `Σ_k motion_k · ∂_k` (x, y) over the sensed physical gradients of `[q, Z_0..Z_31]`.
    motion_drive: [f64; 2],
    /// Controller input of a due evaluation, kept only for an external controller backend.
    input: Option<Box<[f32; INPUT_WIDTH]>>,
}

impl Sense {
    fn blank() -> Sense {
        Sense { density0: 0.0, out: [0.0; OUTPUT_WIDTH], evaluated: false, mem: None, motion_drive: [0.0; 2], input: None }
    }
}

/// One organism's motion/emission/reproduction outcome, merged serially after the threaded pass.
#[derive(Clone, Copy, Default, Debug)]
struct MotionOut {
    ran: bool,
    dead: bool,
    has_tiny: bool,
    emitted: bool,
    has_deposit: bool,
    deaths: u8,
    attempts: u8,
    rejected_mutation: u8,
    rejected_viability: u8,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    energy: f64,
    d_motion: f64,
    tiny: f64,
    d_emission: f64,
    d_transferred: f64,
    deposit_q: f64,
}

struct BirthProposal {
    parent: usize,
    genome: Arc<Genome>,
    child_energy: f64,
    parent_energy: f64,
}


#[derive(Debug, Clone, PartialEq)]
pub struct World {
    pub cfg: Config,
    pub rng: SemanticRng,
    /// Last committed tick `n`.
    pub tick: u64,
    pub field: Field,
    pub orgs: Organisms,
    pub sources: Vec<Source>,
    pub next_org_id: u64,
    pub next_genome_id: u64,
    pub pending: Vec<ScheduledCommand>,
    pub(crate) next_command_seq: u64,
    pub ledger: CumulativeLedger,
    pub totals: TickCounts,
    /// Execution parallelism for sensing/inference; never changes scientific state.
    pub workers: usize,
    pub last_report: Option<TickReport>,
    pub(crate) pyramid_extra_levels: usize,
    /// The authoritative field lives on a resident device backend; `field` may be stale
    /// until `sync_field` is called.
    pub(crate) device_field: bool,
/// Execution telemetry. Off by default. Not part of the scientific state
/// (`state_hash` ignores it) and not written into checkpoints.
profile: StageProfiler,
/// Reused candidate organism storage. Not scientific state.
org_scratch: Reused<Organisms>,
    /// Reused `e_start` buffer for the per-tick energy delta.
    energy_scratch: Reused<Vec<f64>>,
    /// Reused per-organism sense table.
    sense_scratch: Reused<Vec<Sense>>,
    outs_scratch: Reused<Vec<SenseOutput>>,
    motion_scratch: Reused<Vec<MotionOut>>,
    dir_scratch: Reused<Vec<[f32; CHANNELS]>>,
    /// Genome-derived per-organism constants, aligned with `orgs` (rebuilt when it is not).
    genome_hot: Reused<Vec<GenomeHot>>,
    /// The cost parameters `genome_hot` was computed with.
    genome_hot_key: Reused<[u64; 4]>,
}

/// Per-organism values derived only from its (immutable) genome and the
/// configuration, kept next to the organism table so the tick does not chase
/// every genome pointer for them. Execution cache, not scientific state.
#[derive(Clone, Copy, Debug)]
struct GenomeHot {
    organism: u64,
    /// `basal + sensor * (scale / unit)^exponent`, energy per second.
    upkeep_rate: f64,
    params: u32,
}

impl GenomeHot {
    fn of(cfg: &Config, organism: u64, g: &Genome) -> GenomeHot {
        let co = &cfg.costs;
        GenomeHot {
            organism,
            upkeep_rate: co.basal_energy_per_second + co.sensor_energy_per_second_at_unit_scale * (g.scale() / co.sensor_unit_scale_world_units).powf(co.sensor_scale_cost_exponent),
            params: g.param_count() as u32,
        }
    }
}

/// Execution cache. `PartialEq` ignores it so a reused buffer cannot change world equality.
#[derive(Clone, Debug)]
struct Reused<T>(T);

impl<T> PartialEq for Reused<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T: Default> Default for Reused<T> {
    fn default() -> Self {
        Reused(T::default())
    }
}

impl World {
    pub fn new(cfg: Config) -> Result<World, Vec<String>> {
        cfg.validate()?;
        let rng = SemanticRng::new(cfg.seed);
        let mut field = Field::new(cfg.world.grid_width as usize, cfg.world.grid_height as usize, cfg.world.width, cfg.world.height);
        let e0 = cfg.field_initialization.energy_per_cell as f32;
        field.e.iter_mut().for_each(|v| *v = e0);
        let dt = cfg.dt();
        let k = cfg.controller.evaluation_interval_ticks;
        let cd = (cfg.reproduction.cooldown_seconds / dt).ceil() as u64;
        let mut orgs = Organisms::default();
        for id in 0..cfg.population.initial_count as u64 {
            let u = rng.uniform2(Lane::InitPosition, id, 0, 0, 0);
            let g = Arc::new(founder(&cfg, &rng, id, id));
            orgs.push(id, id, 0, u[0] * cfg.world.width, u[1] * cfg.world.height, cfg.population.initial_energy, 0, cd, (id % k as u64) as u32, g);
        }
        let sources = init_sources(&cfg, &rng);
        let n0 = cfg.population.initial_count as u64;
        let endowment = field.total_energy() + orgs.energy.iter().sum::<f64>();
        let workers = cfg.execution.cpu_workers as usize;
        let lv = cfg.pyramid_levels();
        Ok(World {
            cfg,
            rng,
            tick: 0,
            field,
            orgs,
            sources,
            next_org_id: n0,
            next_genome_id: n0,
            pending: Vec::new(),
            next_command_seq: 0,
            ledger: CumulativeLedger { initial_endowment: endowment, ..Default::default() },
            totals: TickCounts::default(),
            workers,
            last_report: None,
            pyramid_extra_levels: lv,
            device_field: false,
            profile: StageProfiler::disabled(),
            org_scratch: Reused::default(),
            energy_scratch: Reused::default(),
            sense_scratch: Reused::default(),
            outs_scratch: Reused::default(),
            motion_scratch: Reused::default(),
            dir_scratch: Reused::default(),
            genome_hot: Reused::default(),
            genome_hot_key: Reused::default(),
        })
    }

    /// Turn per-stage timing on or off. Off (the default) does not sample the clock
    /// for these stages. Either setting leaves the tick's scientific results unchanged.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profile.set_enabled(enabled);
    }

    pub fn stage_profile(&self) -> &StageProfiler {
        &self.profile
    }

    pub fn stage_profile_mut(&mut self) -> &mut StageProfiler {
        &mut self.profile
    }

    pub fn dt(&self) -> f64 {
        self.cfg.dt()
    }

    pub fn simulation_seconds(&self) -> f64 {
        self.tick as f64 * self.dt()
    }

    pub fn total_energy(&self) -> f64 {
        self.field.total_energy() + self.orgs.energy.iter().sum::<f64>()
    }

    /// Queue a command to apply at the boundary of committed tick `tick` (>= current tick).
    pub fn schedule(&mut self, tick: u64, command: Command) -> Result<u64, String> {
        if tick < self.tick {
            return Err(format!("command tick {tick} is before committed tick {}", self.tick));
        }
        if self.pending.len() >= self.cfg.execution.command_queue_capacity as usize {
            return Err("command queue is full".into());
        }
        match &command {
            Command::SetParam { param, value } => param.check(&self.cfg, *value)?,
            Command::DriftChemistry { source, degrees } => {
                if let Some(k) = source {
                    if *k >= self.cfg.sources.count {
                        return Err(format!("source {k} does not exist ({} sources)", self.cfg.sources.count));
                    }
                }
                if !degrees.is_finite() || degrees.abs() > 180.0 {
                    return Err("chemistry drift must be within 180 degrees".into());
                }
            }
            _ => {}
        }
        let seq = self.next_command_seq;
        self.next_command_seq += 1;
        self.pending.push(ScheduledCommand { tick, seq, command });
        Ok(seq)
    }

    /// Advance `n -> n+1`. On error the committed world is unchanged.
    pub fn step(&mut self) -> Result<TickReport, TickError> {
        self.step_with(None)
    }

    /// Advance `n -> n+1`, evaluating due controllers with `backend`
    /// (`None` = the built-in parallel scalar reference). On error the
    /// committed world is unchanged.
    pub fn step_with(&mut self, backend: Option<&mut dyn ControllerBackend>) -> Result<TickReport, TickError> {
        match backend {
            Some(b) => self.step_inference(Inference::Controller(b)),
            None => self.step_inference(Inference::Reference),
        }
    }

    /// Advance `n -> n+1` with an explicit inference route. On error the committed world is unchanged
    /// (a resident device backend discards its working field).
    pub fn step_inference(&mut self, inference: Inference<'_>) -> Result<TickReport, TickError> {
        match inference {
            Inference::Resident(r) => {
                let res = self.step_core(Inference::Resident(&mut *r));
                if res.is_err() {
                    r.abort();
                }
                res
            }
            other => {
                if self.device_field {
                    return Err(TickError::NumericalFault {
                        tick: self.tick + 1,
                        detail: "the authoritative field is on a resident device: call detach_device (or keep stepping it resident) first".into(),
                    });
                }
                self.step_core(other)
            }
        }
    }

    /// True when the authoritative field lives on a resident device backend.
    pub fn field_on_device(&self) -> bool {
        self.device_field
    }

    /// Copy the device's committed field into `self.field` (for rendering, checkpoints,
    /// inspection, hashing). The device stays authoritative.
    pub fn sync_field(&mut self, r: &mut dyn ResidentBackend) -> Result<(), String> {
        if self.device_field {
            r.download_committed(&mut self.field)?;
        }
        Ok(())
    }

    /// Bring the field back to the CPU and stop using the resident device.
    pub fn detach_device(&mut self, r: &mut dyn ResidentBackend) -> Result<(), String> {
        self.sync_field(r)?;
        self.device_field = false;
        Ok(())
    }

    fn step_core(&mut self, inference: Inference<'_>) -> Result<TickReport, TickError> {
        let t_start = Stopwatch::start();
        let n = self.tick;
        // Live tuning commands for this boundary change a candidate configuration,
        // which replaces the committed one only when the tick commits. A tick that
        // is rejected and retried therefore never applies them twice.
        let tuned: Option<Config> = {
            let mut params: Vec<(u64, Param, f64)> = self
                .pending
                .iter()
                .filter(|c| c.tick == n)
                .filter_map(|c| match &c.command {
                    Command::SetParam { param, value } => Some((c.seq, *param, *value)),
                    _ => None,
                })
                .collect();
            if params.is_empty() {
                None
            } else {
                params.sort_by_key(|p| p.0);
                let mut c = self.cfg.clone();
                for (_, p, v) in params {
                    p.set(&mut c, v);
                }
                Some(c)
            }
        };
        let cfg = tuned.as_ref().unwrap_or(&self.cfg);
        let dt = cfg.dt();
        let event = n.checked_add(1).ok_or(TickError::Overflow { detail: "tick counter".into() })?;
        let tiny = cfg.world.tiny_energy_threshold;
        let mut timing = TickTiming::default();
        let mut counts = TickCounts::default();
        let mut dis = Dissipation::default();
        // Local accumulator so a rejected candidate (capacity pause, numerical
        // fault) does not land in the run profile. Disabled: `start` does not
        // read the clock.
        let profiling = self.profile.is_enabled();
        let mut prof = StageProfiler::disabled();
        if profiling {
            prof.set_enabled(true);
        }

        let (backend, fused, mut device, mut resident) = match inference {
            Inference::Reference => (None, None, None, None),
            Inference::Controller(b) => (Some(b), None, None, None),
            Inference::Fused(f) => (None, Some(f), None, None),
            Inference::Device(d) => (None, None, Some(d), None),
            Inference::Resident(r) => (None, None, None, Some(r)),
        };
        if let Some(r) = resident.as_mut() {
            if !self.device_field {
                r.upload_committed(&self.field, cfg).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident upload: {detail}") })?;
                self.device_field = true;
            }
        }
        let is_resident = resident.is_some();
        let field_energy_before = match resident.as_ref() {
            Some(r) => r.committed_energy(),
            None => self.field.total_energy(),
        };
        let mut led = TickLedger { energy_before: field_energy_before + self.orgs.energy.iter().sum::<f64>(), ..Default::default() };
        let t_setup = Stopwatch::start();
        let ps = prof.start();
        // Resident ticks work on a geometry-only descriptor: the planes live on the device.
        let mut field = if is_resident {
            Field { gx: self.field.gx, gy: self.field.gy, hx: self.field.hx, hy: self.field.hy, e: Vec::new(), m: Vec::new() }
        } else {
            self.field.clone()
        };
        let mut orgs = std::mem::take(&mut self.org_scratch.0);
        orgs.copy_from_without_genomes(&self.orgs);
        // Genomes of the committed organisms, by pre-tick index. The candidate's own
        // genome column only collects this tick's children until the commit.
        let genomes: &[Arc<Genome>] = &self.orgs.genome;
        let mut hot = std::mem::take(&mut self.genome_hot.0);
        let hot_key = {
            let c = &cfg.costs;
            [c.basal_energy_per_second.to_bits(), c.sensor_energy_per_second_at_unit_scale.to_bits(), c.sensor_unit_scale_world_units.to_bits(), c.sensor_scale_cost_exponent.to_bits()]
        };
        if self.genome_hot_key.0 != hot_key || hot.len() != orgs.len() || hot.iter().zip(&orgs.id).any(|(h, id)| h.organism != *id) {
            self.genome_hot_key.0 = hot_key;
            hot.clear();
            hot.extend((0..orgs.len()).map(|i| GenomeHot::of(cfg, orgs.id[i], &genomes[i])));
        }
        let mut sources = self.sources.clone();
        self.energy_scratch.0.clear();
        self.energy_scratch.0.extend_from_slice(&orgs.energy);
        let e_start = std::mem::take(&mut self.energy_scratch.0);
        let mut dead = vec![false; orgs.len()];
        let mut deposits = if is_resident { Field { gx: field.gx, gy: field.gy, hx: field.hx, hy: field.hy, e: Vec::new(), m: Vec::new() } } else { Field::new(field.gx, field.gy, field.width(), field.height()) };
        let mut ops: Vec<FieldDeposit> = Vec::new();
        let mut dep_list: Vec<FieldDeposit> = Vec::new();

        // ── Phase 1: admit commands recorded for this boundary ────────────
        let mut due: Vec<&ScheduledCommand> = self.pending.iter().filter(|c| c.tick == n).collect();
        due.sort_by_key(|c| c.seq);
        let mut speed_now = self.cfg.sources.path_speed;
        for sc in &due {
            match &sc.command {
                // Already applied to the candidate configuration above.
                Command::SetParam { param: Param::SourceSpeed, value } => {
                    // Keep every source where it is: move the path anchor so the
                    // center is continuous across the change of speed.
                    for s in sources.iter_mut() {
                        s.advance_to(n, &self.cfg, &self.rng);
                        s.reanchor_for_speed(n, &self.cfg, speed_now, *value);
                    }
                    speed_now = *value;
                }
                Command::SetParam { .. } => {}
                Command::DriftChemistry { source, degrees } => {
                    for s in sources.iter_mut().filter(|s| source.is_none_or(|k| k == s.id)) {
                        s.drift_chemistry(&self.rng, n, sc.seq, *degrees);
                    }
                }
                Command::EraseDisk { x, y, radius, disposal } => {
                    for i in 0..orgs.len() {
                        if dead[i] {
                            continue;
                        }
                        let dx = min_image(orgs.x[i] - x, field.width());
                        let dy = min_image(orgs.y[i] - y, field.height());
                        if dx * dx + dy * dy <= radius * radius {
                            let e = orgs.energy[i];
                            match disposal {
                                Disposal::Remove => led.removed += e,
                                Disposal::Dissipate => dis.death += e,
                                Disposal::Recycle => {
                                    if is_resident {
                                        ops.push(FieldDeposit::Stencil { x: orgs.x[i], y: orgs.y[i], q: e, dir: genomes[i].affinity });
                                    } else {
                                        let s = field.stencil(orgs.x[i], orgs.y[i]);
                                        field.deposit(&s, e, &genomes[i].affinity);
                                    }
                                }
                            }
                            orgs.energy[i] = 0.0;
                            dead[i] = true;
                            counts.deaths_user += 1;
                        }
                    }
                }
                Command::EnergyBrush { x, y, radius, amount, direction } => {
                    let mut cells = Vec::new();
                    for cy in 0..field.gy {
                        for cx in 0..field.gx {
                            let dx = min_image((cx as f64 + 0.5) * field.hx - x, field.width());
                            let dy = min_image((cy as f64 + 0.5) * field.hy - y, field.height());
                            if dx * dx + dy * dy <= radius * radius {
                                cells.push(cy * field.gx + cx);
                            }
                        }
                    }
                    if cells.is_empty() {
                        cells.push(field.stencil(*x, *y).cells[0]);
                    }
                    let dir = direction.as_ref().and_then(|d| isotropic_unit(&d.map(|v| v as f64))).unwrap_or([0.0; CHANNELS]);
                    let q = amount.max(0.0) / cells.len() as f64;
                    for c in cells {
                        if is_resident {
                            ops.push(FieldDeposit::Cell { cell: c, q, dir });
                        } else {
                            let s = Stencil { n: 1, cells: [c, 0, 0, 0], w: [1.0, 0.0, 0.0, 0.0] };
                            field.deposit(&s, q, &dir);
                        }
                    }
                    led.injected_commands += amount.max(0.0);
                }
            }
        }

        prof.stop_other(OtherPart::Setup, ps);
        if let Some(r) = resident.as_mut() {
            let ps = prof.start();
            r.begin_tick(&ops).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident begin: {detail}") })?;
            // Submit only: begin_tick does not wait. Not GPU kernel time.
            prof.stop(CpuStage::TransfersSync, ps);
        }
        timing.setup_ms = t_setup.ms();

        // ── Phase 2: transport, then scheduled sources → observation field ─
        let t = Stopwatch::start();
        let ps = prof.start();
        for s in &mut sources {
            s.advance_to(n, cfg, &self.rng);
        }
        prof.stop(CpuStage::ChemistryField, ps);
        // Resident transport is submitted and not waited on yet. Sense-job
        // packing below overlaps that GPU work; the accounting wait follows it.
        let mut transport_open = false;
        if let Some(r) = resident.as_mut() {
            let ps = prof.start();
            r.submit_transport(cfg, &sources, n).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident transport: {detail}") })?;
            prof.stop(CpuStage::TransfersSync, ps);
            transport_open = true;
        } else if let Some(d) = device.as_mut() {
            let ps = prof.start();
            let acct = d
                .transport_and_sources(&mut field, cfg, &sources, n)
                .map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("field backend {}: {detail}", FieldBackend::name(&**d)) })?;
            prof.stop(CpuStage::TransfersSync, ps);
            dis.decay += acct.decay_removed;
            counts.transport_subcycles = acct.subcycles;
            led.injected_sources = acct.injected;
            dis.tiny_reservoir += acct.tiny_removed;
        } else {
            let ps = prof.start();
            let acct = cpu_transport_and_sources(&mut field, cfg, &sources, n).map_err(|detail| TickError::NumericalFault { tick: event, detail })?;
            prof.stop(CpuStage::ChemistryField, ps);
            dis.decay += acct.decay_removed;
            counts.transport_subcycles = acct.subcycles;
            led.injected_sources = acct.injected;
            dis.tiny_reservoir += acct.tiny_removed;
        }
        if !transport_open {
            timing.transport_ms = t.ms();
        }

        // ── Phase 3–4: sensing pyramid, sample organisms, evaluate due controllers ──
        let t_sense = Stopwatch::start();
        let workers = self.workers;
        // Set by a device's single-wait front half: (organism index per job, uptake results, granted energy).
        let mut fused_uptake: Option<(Vec<usize>, Vec<UptakeOut>, f64)> = None;
        let sense_now = device.is_some() || resident.is_some() || fused.is_some();
        let mut senses = if sense_now {
            let ps = prof.start();
            let k_eval = cfg.controller.evaluation_interval_ticks as u64;
            let idx: Vec<usize> = (0..orgs.len()).filter(|&i| !dead[i]).collect();
            let jobs: Vec<SenseJob<'_>> = crate::par::map_collect(workers, idx.len(), |k| {
                let i = idx[k];
                let age = (n - orgs.birth_tick[i]) as f64 * dt;
                SenseJob {
                    organism: orgs.id[i],
                    genome: &genomes[i],
                    x: orgs.x[i],
                    y: orgs.y[i],
                    intrinsic: intrinsic(&cfg.sensors, orgs.energy[i], age, orgs.vx[i], orgs.vy[i], orgs.energy_change_rate[i]),
                    due: !orgs.held_valid[i] || (n + 1) % k_eval == orgs.phase[i] as u64,
                    memory: orgs.memory.get(i).unwrap_or(&ZERO_MEMORY),
                }
            });
            prof.stop(CpuStage::Sensing, ps);
            // Reused across ticks: a fresh megabyte every tick costs page faults.
            let mut outs = std::mem::take(&mut self.outs_scratch.0);
            outs.clear();
            outs.resize(jobs.len(), SenseOutput::default());
            let mut front_done = false;
            if transport_open {
                // Single-wait front half when the device offers it: transport accounting,
                // sensing, inference and the uptake exchange come back together.
                let mut up = vec![UptakeOut::default(); jobs.len()];
                let tb = Stopwatch::start();
                let ps = prof.start();
                if let Some(res) = resident.as_mut().expect("resident transport").sense_uptake(cfg, self.pyramid_extra_levels, &jobs, &mut outs, &mut up) {
                    prof.stop(CpuStage::ExternalEvaluate, ps);
                    timing.backend_inference_ms = tb.ms();
                    let (acct, granted) = res.map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident front: {detail}") })?;
                    dis.decay += acct.decay_removed;
                    counts.transport_subcycles = acct.subcycles;
                    led.injected_sources = acct.injected;
                    dis.tiny_reservoir += acct.tiny_removed;
                    timing.transport_ms = t.ms();
                    fused_uptake = Some((idx.clone(), up, granted));
                    transport_open = false;
                    front_done = true;
                }
            }
            if transport_open {
                let ps = prof.start();
                let acct = resident.as_mut().expect("resident transport").finish_transport().map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident transport: {detail}") })?;
                // The wait that remains after sense-job packing. Not GPU kernel time.
                prof.stop(CpuStage::TransfersSync, ps);
                dis.decay += acct.decay_removed;
                counts.transport_subcycles = acct.subcycles;
                led.injected_sources = acct.injected;
                dis.tiny_reservoir += acct.tiny_removed;
                timing.transport_ms = t.ms();
            }
            let fused: Option<&mut dyn SenseInferBackend> = match (device, resident.as_mut()) {
                (Some(d), _) => Some(d),
                (None, Some(r)) => Some(&mut **r),
                (None, None) => fused,
            };
            let fb = fused.expect("fused backend");
            let has_memory = !orgs.memory.is_empty();
            let mut senses_buf = std::mem::take(&mut self.sense_scratch.0);
            if senses_buf.len() != orgs.len() {
                senses_buf.resize(orgs.len(), Sense::blank());
            }
            if !front_done {
                let tb = Stopwatch::start();
                let ps = prof.start();
                let res = fb.sense_and_infer(&field, &cfg.sensors, self.pyramid_extra_levels, &jobs, &mut outs);
                // Whole backend call. A GPU implementation's submit/wait is not kernel time;
                // that split is reported by the backend, on its own clocks.
                prof.stop(CpuStage::ExternalEvaluate, ps);
                timing.backend_inference_ms = tb.ms();
                res.map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("backend {}: {detail}", fb.name()) })?;
            }
            let ps = prof.start();
            let scatter_one = |k: usize, i: usize, s: &mut Sense| -> bool {
                let o = &outs[k];
                let finite = o.motion_drive.iter().all(|v| v.is_finite()) && o.density0.is_finite() && (!jobs[k].due || o.logits.iter().all(|v| v.is_finite()));
                if !finite {
                    return false;
                }
                s.density0 = o.density0.max(0.0);
                s.motion_drive = o.motion_drive;
                s.evaluated = jobs[k].due;
                // The next memory goes straight into the organism column below, not into a boxed copy per organism.
                s.mem = None;
                s.input = None;
                s.out = if jobs[k].due { o.logits } else { orgs.held[i] };
                true
            };
            // Every organism is a job (nobody was removed by a command this tick): the rows line up one to one, so the loop can be split across threads.
            let mut all_finite = true;
            if idx.len() == orgs.len() && crate::par::threaded(workers, idx.len()) {
                let ok = std::sync::atomic::AtomicBool::new(true);
                crate::par::for_each_mut(workers, &mut senses_buf[..], |i, s| {
                    if !scatter_one(i, i, s) {
                        ok.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                all_finite = ok.load(std::sync::atomic::Ordering::Relaxed);
            } else {
                for (k, &i) in idx.iter().enumerate() {
                    if !scatter_one(k, i, &mut senses_buf[i]) {
                        all_finite = false;
                        break;
                    }
                }
            }
            if !all_finite {
                // Name the first offender, as the serial loop would.
                let k = (0..idx.len())
                    .find(|&k| {
                        let o = &outs[k];
                        !(o.motion_drive.iter().all(|v| v.is_finite()) && o.density0.is_finite() && (!jobs[k].due || o.logits.iter().all(|v| v.is_finite())))
                    })
                    .unwrap_or(0);
                return Err(TickError::NumericalFault { tick: event, detail: format!("backend {} produced non-finite results for organism {}", fb.name(), orgs.id[idx[k]]) });
            }
            if has_memory {
                // The next memory of every organism evaluated this tick goes straight into its column (the held-output store
                // below does not touch it in this path). The jobs hold the old memory by reference, so release them first.
                let dues: Vec<bool> = jobs.iter().map(|j| j.due).collect();
                drop(jobs);
                if idx.len() == orgs.len() && crate::par::threaded(workers, idx.len()) {
                    crate::par::for_each_mut(workers, &mut orgs.memory[..], |i, m| {
                        if dues[i] {
                            *m = outs[i].memory;
                        }
                    });
                } else {
                    for (k, &i) in idx.iter().enumerate() {
                        if dues[k] {
                            orgs.memory[i] = outs[k].memory;
                        }
                    }
                }
            }
            prof.stop_other(OtherPart::SenseScatter, ps);
            self.outs_scratch.0 = outs;
            senses_buf
        } else {
            let ps = prof.start();
            let pyr = Pyramid::build(&field, self.pyramid_extra_levels);
            prof.stop(CpuStage::Sensing, ps);
            let wall = prof.start();
            let infer = backend.is_none();
            let (senses, sense_thread, net_thread) = if profiling {
                sense_all::<true>(cfg, &pyr, &orgs, genomes, &dead, n, self.workers, infer)
            } else {
                sense_all::<false>(cfg, &pyr, &orgs, genomes, &dead, n, self.workers, infer)
            };
            if profiling {
                let (sense_part, net_part) = proportion(wall.elapsed_ns(), sense_thread, net_thread);
                prof.add_sample(CpuStage::Sensing, sense_part);
                if net_part > 0 || net_thread > 0 {
                    prof.add_sample(CpuStage::Networks, net_part);
                }
                prof.add_thread_ns(sense_thread, net_thread);
            }
            senses
        };
        if let Some(be) = backend {
            let ps = prof.start();
            let due: Vec<usize> = (0..senses.len()).filter(|&i| senses[i].evaluated).collect();
            let jobs: Vec<InferenceJob<'_>> =
                due.iter().map(|&i| InferenceJob { organism: orgs.id[i], genome: &genomes[i], input: senses[i].input.as_deref().expect("controller path keeps due inputs") }).collect();
            let mut outs = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
            prof.stop_other(OtherPart::SenseScatter, ps);
            let tb = Stopwatch::start();
            let ps = prof.start();
            let res = be.evaluate(&jobs, &mut outs);
            prof.stop(CpuStage::ExternalEvaluate, ps);
            timing.backend_inference_ms = tb.ms();
            res.map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("backend {}: {detail}", be.name()) })?;
            drop(jobs);
            let ps = prof.start();
            for (k, &i) in due.iter().enumerate() {
                if outs[k].iter().any(|v| !v.is_finite()) {
                    return Err(TickError::NumericalFault { tick: event, detail: format!("backend {} produced non-finite logits for organism {}", be.name(), orgs.id[i]) });
                }
                senses[i].out = outs[k];
            }
            prof.stop_other(OtherPart::SenseScatter, ps);
        }
        let ps = prof.start();
        {
            let senses = &senses;
            crate::par::for_each_mut3(workers, &mut orgs.held[..], &mut orgs.held_tick[..], &mut orgs.held_valid[..], |i, held, tick, valid| {
                let s = &senses[i];
                if s.evaluated {
                    *held = s.out;
                    *tick = event;
                    *valid = true;
                }
            });
            // Only the CPU-sensing path hands memory over here; the fused/device paths wrote it when they scattered.
            if !sense_now && !orgs.memory.is_empty() {
                crate::par::for_each_mut(workers, &mut orgs.memory[..], |i, m| {
                    if let (true, Some(next)) = (senses[i].evaluated, &senses[i].mem) {
                        *m = **next;
                    }
                });
            }
            let (evals, params) = crate::par::sum2(workers, senses.len(), |i| if senses[i].evaluated { (1, hot[i].params as u64) } else { (0, 0) });
            counts.controller_evaluations += evals as u32;
            counts.parameter_evaluations += params;
        }
        prof.stop_other(OtherPart::HeldStore, ps);
        timing.sense_and_infer_ms = t_sense.ms();

        // ── Phase 5: uptake requests and cell grants ──────────────────────
        let t = Stopwatch::start();
        let ch = &cfg.chemistry;
        let umax = ch.uptake_capacity_per_second;
        let ps = prof.start();
        let demand: Vec<f64> = crate::par::map_collect(workers, orgs.len(), |i| if dead[i] { 0.0 } else { let r = senses[i].density0; dt * umax * r / (ch.uptake_half_saturation_density + r) });
        prof.stop(CpuStage::ChemistryField, ps);
        let mut uptake_total = vec![0.0f64; orgs.len()];
        let mut settlements: Vec<crate::chemistry::Settlement> = Vec::with_capacity(orgs.len());
        if let Some(r) = resident.as_mut() {
            let (idx, outs, granted) = if let Some(front) = fused_uptake.take() {
                front
            } else {
                let ps = prof.start();
                let idx: Vec<usize> = (0..orgs.len()).filter(|&i| demand[i] > 0.0).collect();
                let jobs: Vec<UptakeJob<'_>> = idx.iter().map(|&i| UptakeJob { x: orgs.x[i], y: orgs.y[i], demand: demand[i], genome: &genomes[i] }).collect();
                prof.stop(CpuStage::ChemistryField, ps);
                let mut outs = vec![UptakeOut::default(); jobs.len()];
                let ps = prof.start();
                let granted = r.uptake(ch, &jobs, &mut outs).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident uptake: {detail}") })?;
                // Includes the readback wait uptake already performs. Not kernel time.
                prof.stop(CpuStage::TransfersSync, ps);
                (idx, outs, granted)
            };
            led.transferred += granted;
            let ps = prof.start();
            settlements = vec![Default::default(); orgs.len()];
            let settle_one = |k: usize, i: usize, st: &mut crate::chemistry::Settlement, total: &mut f64| -> bool {
                let o = outs[k];
                if !(o.absorbed.is_finite() && o.useful_gain.is_finite() && o.toxicity_requested.is_finite()) {
                    return false;
                }
                *st = crate::chemistry::Settlement {
                    absorbed: o.absorbed,
                    useful_gain: o.useful_gain,
                    toxicity_requested: o.toxicity_requested,
                    toxicity_paid: o.toxicity_requested.min(orgs.energy[i] + o.useful_gain),
                };
                *total = o.absorbed;
                true
            };
            let mut all_finite = true;
            if idx.len() == orgs.len() && crate::par::threaded(workers, idx.len()) {
                let ok = std::sync::atomic::AtomicBool::new(true);
                crate::par::for_each_mut2(workers, &mut settlements[..], &mut uptake_total[..], |i, st, total| {
                    if !settle_one(i, i, st, total) {
                        ok.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                all_finite = ok.load(std::sync::atomic::Ordering::Relaxed);
            } else {
                for (k, &i) in idx.iter().enumerate() {
                    let (st, total) = (&mut settlements[i], &mut uptake_total[i]);
                    if !settle_one(k, i, st, total) {
                        all_finite = false;
                        break;
                    }
                }
            }
            if !all_finite {
                let k = (0..idx.len()).find(|&k| !(outs[k].absorbed.is_finite() && outs[k].useful_gain.is_finite() && outs[k].toxicity_requested.is_finite())).unwrap_or(0);
                return Err(TickError::NumericalFault { tick: event, detail: format!("resident uptake produced non-finite results for organism {}", orgs.id[idx[k]]) });
            }
            prof.stop_other(OtherPart::UptakeScatter, ps);
        } else {
        let ps = prof.start();
        let stencils: Vec<Stencil> = (0..orgs.len()).map(|i| field.stencil(orgs.x[i], orgs.y[i])).collect();
        let mut cell_demand = vec![0.0f64; field.cells()];
        for i in 0..orgs.len() {
            if demand[i] > 0.0 {
                let s = &stencils[i];
                for k in 0..s.n {
                    cell_demand[s.cells[k]] += demand[i] * s.w[k];
                }
            }
        }
        let mut granted_cell = vec![0.0f64; field.cells()];
        for i in 0..orgs.len() {
            if demand[i] <= 0.0 {
                settlements.push(Default::default());
                continue;
            }
            let s = &stencils[i];
            let a = &genomes[i].affinity;
            let mut packets = [(0.0f64, 0.0f64); 4];
            for k in 0..s.n {
                let c = s.cells[k];
                let ec = field.e[c] as f64;
                let u = grant_scale(ec, cell_demand[c]) * demand[i] * s.w[k];
                granted_cell[c] += u;
                let az = if ec > 0.0 {
                    let m = field.moment(c);
                    (0..CHANNELS).map(|j| a[j] as f64 * m[j] as f64).sum::<f64>() / ec
                } else {
                    0.0
                };
                packets[k] = (u, eta(ch.maximum_efficiency, ch.affinity_gain, ch.discrimination_threshold, az));
            }
            let st = settle(orgs.energy[i], ch.toxicity_sensitivity, packets[..s.n].iter().copied());
            uptake_total[i] = st.absorbed;
            settlements.push(st);
        }
        for c in 0..field.cells() {
            let g = granted_cell[c];
            if g > 0.0 {
                let e = field.e[c] as f64;
                let e_new = (e - g).max(0.0);
                let e_new32 = e_new as f32;
                let f = if e > 0.0 { e_new32 as f64 / e } else { 0.0 };
                field.e[c] = e_new32;
                for v in &mut field.m[c * CHANNELS..(c + 1) * CHANNELS] {
                    *v = (*v as f64 * f) as f32;
                }
                led.transferred += g;
            }
        }
        prof.stop(CpuStage::ChemistryField, ps);
        }

        // ── Phase 6: settle gain, toxicity, maintenance ───────────────────
        // Per-organism work runs on threads; every shared total is then added up in organism
        // order, exactly as the serial loop did, so the sums are bit-identical.
        let co = &cfg.costs;
        let ps = prof.start();
        #[derive(Clone, Copy, Default)]
        struct Upkeep {
            ran: bool,
            ineff: f64,
            tox: f64,
            maint: f64,
            tiny: f64,
            has_tiny: bool,
            died: bool,
        }
        let mut upkeep = vec![Upkeep::default(); orgs.len()];
        crate::par::for_each_mut3(workers, &mut orgs.energy[..], &mut dead[..], &mut upkeep[..], |i, energy, dead_i, u| {
            if *dead_i {
                return;
            }
            u.ran = true;
            let st = settlements[i];
            *energy += st.net();
            u.ineff = st.absorbed - st.useful_gain;
            u.tox = st.toxicity_paid;
            let cost = dt * hot[i].upkeep_rate + if senses[i].evaluated { co.neural_energy_per_parameter_evaluation * hot[i].params as f64 } else { 0.0 };
            let paid = cost.min(energy.max(0.0));
            *energy -= paid;
            u.maint = paid;
            if paid < cost {
                *energy = energy.max(0.0);
                u.tiny = *energy;
                u.has_tiny = true;
                *energy = 0.0;
                *dead_i = true;
                u.died = true;
                return;
            }
            if *energy <= tiny {
                u.tiny = energy.max(0.0);
                u.has_tiny = true;
                *energy = 0.0;
                *dead_i = true;
                u.died = true;
            }
        });
        for u in upkeep.iter().filter(|u| u.ran) {
            dis.uptake_inefficiency += u.ineff;
            dis.toxicity += u.tox;
            dis.maintenance += u.maint;
            if u.has_tiny {
                dis.tiny_reservoir += u.tiny;
            }
            counts.deaths_exhaustion += u.died as u32;
        }
        prof.stop(CpuStage::ChemistryField, ps);
        timing.exchange_ms = t.ms();

        // ── Phase 7: motion → emission → reproduction proposals ──────────
        let t = Stopwatch::start();
        let ps = prof.start();
        let drag_keep = (-ch.drag_per_second * dt).exp();
        let drag_coef = drag_force_coefficient(ch.drag_per_second, dt);
        let em = &cfg.emission;
        let rp = &cfg.reproduction;
        let (w, h) = (field.width(), field.height());
        let mut proposals: Vec<BirthProposal>;
        // One organism's motion, emission and reproduction attempt run on threads; its new state and its
        // contributions to the shared totals are recorded, then merged below in organism order.
        let tr0 = std::time::Instant::now();
        let mut mo = std::mem::take(&mut self.motion_scratch.0);
        mo.clear();
        mo.resize(orgs.len(), MotionOut::default());
        let mut dirs = std::mem::take(&mut self.dir_scratch.0);
        if dirs.len() < orgs.len() {
            dirs.resize(orgs.len(), [0.0; CHANNELS]);
        }
        let new_births = std::sync::Mutex::new(Vec::<BirthProposal>::new());
        let rng = &self.rng;
        crate::par::for_each_mut2(workers, &mut mo[..], &mut dirs[..orgs.len()], |i, r, dir_slot| {
            if dead[i] {
                return;
            }
            r.ran = true;
            let (mut x, mut y, mut energy) = (orgs.x[i], orgs.y[i], orgs.energy[i]);
            r.dead = false;
            'organism: {
                // Motion (field-mediated chemotaxis; no pairwise forces).
                // Genome bytes are immutable here; a shared reference avoids an Arc clone per organism.
                let g = &*genomes[i];
                let s = &senses[i];
                let gu = if umax > 0.0 { (uptake_total[i] / (dt * umax)).clamp(0.0, 1.0) } else { 0.0 };
                let (mut fx, mut fy) = (s.motion_drive[0], s.motion_drive[1]);
                fx *= gu;
                fy *= gu;
                let fn_ = (fx * fx + fy * fy).sqrt();
                if fn_ > ch.maximum_acceleration {
                    let f = ch.maximum_acceleration / fn_;
                    fx *= f;
                    fy *= f;
                }
                let mut vx = drag_keep * orgs.vx[i] + drag_coef * fx;
                let mut vy = drag_keep * orgs.vy[i] + drag_coef * fy;
                let sp = (vx * vx + vy * vy).sqrt();
                if sp > ch.maximum_speed {
                    let f = ch.maximum_speed / sp;
                    vx *= f;
                    vy *= f;
                }
                let mut cost = co.motion_energy_per_speed_squared_second * (vx * vx + vy * vy) * dt;
                if cost > energy {
                    let f = (energy / cost).sqrt();
                    vx *= f;
                    vy *= f;
                    cost = energy;
                }
                energy -= cost;
                r.d_motion = cost;
                r.vx = vx;
                r.vy = vy;
                x = wrap(x + dt * vx, w);
                y = wrap(y + dt * vy, h);
                if energy <= tiny {
                    r.tiny = energy.max(0.0);
                    r.has_tiny = true;
                    energy = 0.0;
                    r.dead = true;
                    r.deaths += 1;
                    break 'organism;
                }

                // Paid emission at the post-motion position, visible next tick.
                let o = &s.out;
                if sigmoid(o[out::EMIT_GATE] as f64) > em.gate_threshold {
                    let raw: [f64; CHANNELS] = std::array::from_fn(|k| o[out::EMIT_DIR + k] as f64);
                    let rn = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if rn >= em.direction_zero_threshold {
                        if let Some(dir) = isotropic_unit(&raw) {
                            let frac = rate_to_probability(em.maximum_fractional_rate_per_second * sigmoid(o[out::EMIT_RATE] as f64), dt);
                            let spend = energy * frac;
                            if spend > 0.0 {
                                energy -= spend;
                                *dir_slot = dir;
                                r.has_deposit = true;
                                r.deposit_q = em.transfer_efficiency * spend;
                                r.d_emission = (1.0 - em.transfer_efficiency) * spend;
                                r.d_transferred = spend;
                                r.emitted = true;
                                if energy <= tiny {
                                    r.tiny = energy.max(0.0);
                                    r.has_tiny = true;
                                    energy = 0.0;
                                    r.dead = true;
                                    r.deaths += 1;
                                    break 'organism;
                                }
                            }
                        }
                    }
                }

                // Reproduction proposal (one Bernoulli attempt per eligible tick).
                if n >= orgs.cooldown_until[i] && energy >= rp.minimum_parent_energy_before_attempt {
                    let lambda = rp.maximum_attempt_rate_per_second * sigmoid(o[out::SPLIT] as f64);
                    let p = rate_to_probability(lambda, dt);
                    let u = rng.uniform(Lane::ReproductionAttempt, orgs.id[i], event, 0, 0);
                    if u < p {
                        r.attempts += 1;
                        let beta = rp.minimum_child_fraction + (rp.maximum_child_fraction - rp.minimum_child_fraction) * sigmoid(o[out::SHARE] as f64);
                        let (ce, pe) = split_energy(energy, co.birth_energy, beta);
                        if ce >= rp.minimum_child_energy && pe >= rp.minimum_parent_energy_after_split {
                            match mutate(cfg, rng, g, orgs.id[i], event, 0) {
                                Ok(genome) => new_births.lock().expect("births lock").push(BirthProposal { parent: i, genome, child_energy: ce, parent_energy: pe }),
                                Err(_) => r.rejected_mutation += 1,
                            }
                        } else {
                            r.rejected_viability += 1;
                        }
                    }
                }
            }
            r.x = x;
            r.y = y;
            r.energy = energy;
        });
        let tr2 = std::time::Instant::now();
        // In organism order, as the serial loop did.
        for (i, r) in mo.iter().enumerate() {
            if !r.ran {
                continue;
            }
            dis.motion += r.d_motion;
            if r.has_tiny {
                dis.tiny_reservoir += r.tiny;
            }
            if r.has_deposit {
                if is_resident {
                    dep_list.push(FieldDeposit::Stencil { x: r.x, y: r.y, q: r.deposit_q, dir: dirs[i] });
                } else {
                    let st = field.stencil(r.x, r.y);
                    deposits.deposit(&st, r.deposit_q, &dirs[i]);
                }
            }
            if r.emitted {
                dis.emission += r.d_emission;
                led.transferred += r.d_transferred;
                counts.emitting += 1;
            }
            counts.deaths_exhaustion += r.deaths as u32;
            counts.reproduction_attempts += r.attempts as u32;
            counts.rejected_births_mutation += r.rejected_mutation as u32;
            counts.rejected_births_viability += r.rejected_viability as u32;
            orgs.x[i] = r.x;
            orgs.y[i] = r.y;
            orgs.vx[i] = r.vx;
            orgs.vy[i] = r.vy;
            orgs.energy[i] = r.energy;
            dead[i] = r.dead;
        }
        proposals = new_births.into_inner().expect("births lock");
        proposals.sort_by_key(|p| p.parent);
        if profiling && n % 128 == 0 {
            let tr3 = std::time::Instant::now();
            eprintln!("trace motion n={} setup {:.2} par {:.2} merge {:.2} ms (row {} B)", orgs.len(), 0.0, (tr2 - tr0).as_secs_f64() * 1e3, (tr3 - tr2).as_secs_f64() * 1e3, std::mem::size_of::<MotionOut>());
        }
        self.motion_scratch.0 = mo;
        self.dir_scratch.0 = dirs;
        prof.stop_other(OtherPart::LifecycleMotion, ps);
        let ps = prof.start();

        // ── Phase 8: capacities, births, deaths ───────────────────────────
        let survivors: Vec<usize> = (0..orgs.len()).filter(|&i| !dead[i]).collect();
        let wb = cfg.controller.weight_dtype().bytes();
        let live_bytes: u64 = survivors.iter().map(|&i| hot[i].params as u64 * wb).sum();
        let birth_bytes: u64 = proposals.iter().map(|p| p.genome.param_count() as u64 * wb).sum();
        let max_count = cfg.population.maximum_count as usize;
        let quota = cfg.execution.maximum_logical_genome_bytes;
        let mut accepted = proposals.len();
        if survivors.len() + proposals.len() > max_count || live_bytes + birth_bytes > quota {
            match cfg.capacity_policy() {
                CapacityPolicy::PauseBeforeCommit => {
                    return Err(TickError::CapacityPaused {
                        tick: event,
                        live: survivors.len(),
                        proposed_births: proposals.len(),
                        maximum_count: max_count,
                        required_genome_bytes: live_bytes + birth_bytes,
                        genome_quota_bytes: quota,
                    })
                }
                CapacityPolicy::RejectBirthsStableOrder => {
                    let (mut count, mut bytes) = (survivors.len(), live_bytes);
                    accepted = 0;
                    for p in &proposals {
                        let b = p.genome.param_count() as u64 * 4;
                        if count + 1 > max_count || bytes + b > quota {
                            break;
                        }
                        count += 1;
                        bytes += b;
                        accepted += 1;
                    }
                    counts.rejected_births_capacity = (proposals.len() - accepted) as u32;
                }
            }
        }
        let cd_ticks = (rp.cooldown_seconds / dt).ceil() as u64;
        let k_eval = cfg.controller.evaluation_interval_ticks as u64;
        let mut next_org = self.next_org_id;
        let mut next_genome = self.next_genome_id;
        let mut children: Vec<(u64, u64, u32, f64, f64, f64, Arc<Genome>)> = Vec::new();
        for p in proposals.into_iter().take(accepted) {
            let i = p.parent;
            orgs.energy[i] = p.parent_energy;
            dis.birth += co.birth_energy;
            led.transferred += p.child_energy;
            orgs.cooldown_until[i] = event.checked_add(cd_ticks).ok_or(TickError::Overflow { detail: "cooldown".into() })?;
            let mut genome = p.genome;
            Arc::get_mut(&mut genome).expect("fresh child genome is unique").id = next_genome;
            next_genome += 1;
            if genome.mutation.as_ref().is_some_and(|m: &MutationRecord| m.topology_changed) {
                counts.topology_mutations += 1;
            }
            let u = self.rng.uniform2(Lane::ChildOffset, orgs.id[i], event, 0, 0);
            let r = rp.child_position_offset_radius * u[0].sqrt();
            let th = std::f64::consts::TAU * u[1];
            let cx = wrap(orgs.x[i] + r * th.cos(), w);
            let cy = wrap(orgs.y[i] + r * th.sin(), h);
            children.push((next_org, orgs.lineage[i], orgs.generation[i] + 1, cx, cy, p.child_energy, genome));
            next_org = next_org.checked_add(1).ok_or(TickError::Overflow { detail: "organism id".into() })?;
            counts.births += 1;
        }
        // Natural deaths with residual energy recycle part of it along their affinity.
        let recycle = cfg.death.residual_energy_recycle_fraction;
        for i in 0..orgs.len() {
            if dead[i] && orgs.energy[i] > 0.0 {
                let e = orgs.energy[i];
                if is_resident {
                    dep_list.push(FieldDeposit::Stencil { x: orgs.x[i], y: orgs.y[i], q: recycle * e, dir: genomes[i].affinity });
                } else {
                    let st = field.stencil(orgs.x[i], orgs.y[i]);
                    deposits.deposit(&st, recycle * e, &genomes[i].affinity);
                }
                dis.death += (1.0 - recycle) * e;
                orgs.energy[i] = 0.0;
            }
        }
        for i in 0..orgs.len() {
            if !dead[i] {
                orgs.energy_change_rate[i] = (orgs.energy[i] - e_start[i]) / dt;
            }
        }
        self.energy_scratch.0 = e_start;
        let keep: Vec<bool> = dead.iter().map(|d| !d).collect();
        orgs.retain(&keep);
        {
            let mut i = 0;
            hot.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
        }
        for (id, lin, gen, x, y, e, g) in children {
            let deadline = event + cd_ticks;
            hot.push(GenomeHot::of(cfg, id, &g));
            orgs.push(id, lin, gen, x, y, e, event, deadline, (id % k_eval) as u32, g);
        }

        prof.stop_other(OtherPart::LifecycleCommit, ps);
        // ── Phase 9: paid deposits and recycling enter the next field ─────
        if let Some(r) = resident.as_mut() {
            let ps = prof.start();
            r.deposit(&dep_list).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident deposit: {detail}") })?;
            // Submit only: deposit does not wait. Not GPU kernel time.
            prof.stop(CpuStage::TransfersSync, ps);
        } else {
            let ps = prof.start();
            field.add_assign(&deposits);
            prof.stop(CpuStage::ChemistryField, ps);
        }
        timing.lifecycle_ms = t.ms();

        // ── Phase 10: tiny-reservoir clearing, validate invariants, publish ─
        let t_validate = Stopwatch::start();
        let field_energy_after;
        if let Some(r) = resident.as_mut() {
            let ps = prof.start();
            let fin = r.finish(tiny).map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident finish: {detail}") })?;
            // Includes the readback wait finish already performs. Not kernel time.
            prof.stop(CpuStage::TransfersSync, ps);
            dis.tiny_reservoir += fin.tiny_removed;
            if !(fin.min_energy >= 0.0) || !fin.min_energy.is_finite() {
                return Err(TickError::NumericalFault { tick: event, detail: format!("negative or non-finite field energy {}", fin.min_energy) });
            }
            if fin.worst_cone > 1e-3 {
                return Err(TickError::NumericalFault { tick: event, detail: format!("moment cone violated by {}", fin.worst_cone) });
            }
            counts.cone_projections = fin.cone_projections;
            field_energy_after = fin.field_energy;
        } else {
            let ps = prof.start();
            dis.tiny_reservoir += field.clear_tiny(tiny);
            let (worst, min_e) = field.cone_report();
            if !(min_e >= 0.0) || !min_e.is_finite() {
                return Err(TickError::NumericalFault { tick: event, detail: format!("negative or non-finite field energy {min_e}") });
            }
            if worst > 1e-3 {
                return Err(TickError::NumericalFault { tick: event, detail: format!("moment cone violated by {worst}") });
            }
            if worst > 0.0 {
                counts.cone_projections = field.project_cone() as u32;
            }
            field_energy_after = field.total_energy();
            prof.stop(CpuStage::ChemistryField, ps);
        }
        let ps = prof.start();
        if let Some(i) = (0..orgs.len()).find(|&i| !(orgs.energy[i] > 0.0) || !orgs.energy[i].is_finite() || !orgs.x[i].is_finite() || !orgs.y[i].is_finite()) {
            return Err(TickError::NumericalFault { tick: event, detail: format!("organism {} published with energy {}", orgs.id[i], orgs.energy[i]) });
        }
        led.dissipated = dis;
        led.field_energy_after = field_energy_after;
        led.organism_energy_after = orgs.energy.iter().sum();
        led.energy_after = led.field_energy_after + led.organism_energy_after;
        led.residual = led.energy_after - led.energy_before - (led.injected_sources + led.injected_commands - led.removed - dis.total());
        prof.stop_other(OtherPart::Publish, ps);

        if let Some(r) = resident.as_mut() {
            let ps = prof.start();
            r.commit().map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("resident commit: {detail}") })?;
            // Submit only: commit does not wait. Not GPU kernel time.
            prof.stop(CpuStage::TransfersSync, ps);
        } else {
            let ps = prof.start();
            self.field = field;
            prof.stop_other(OtherPart::Publish, ps);
        }
        let ps = prof.start();
        // Commit the genome column: survivors keep their committed genomes (moved,
        // not cloned), then this tick's children.
        {
            let mut column = std::mem::take(&mut self.orgs.genome);
            let mut i = 0;
            column.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
            column.append(&mut orgs.genome);
            orgs.genome = column;
        }
        self.genome_hot.0 = hot;
        self.org_scratch.0 = std::mem::replace(&mut self.orgs, orgs);
        self.sources = sources;
        if tuned.is_some() {
            self.cfg = cfg.clone();
        }
        self.tick = event;
        self.next_org_id = next_org;
        self.next_genome_id = next_genome;
        self.pending.retain(|c| c.tick != n);
        let cl = &mut self.ledger;
        cl.injected += led.injected_sources + led.injected_commands;
        cl.removed += led.removed;
        cl.dissipated.add(&dis);
        cl.transferred += led.transferred;
        cl.residual += led.residual;
        cl.max_abs_tick_residual = cl.max_abs_tick_residual.max(led.residual.abs());
        let tt = &mut self.totals;
        tt.births += counts.births;
        tt.deaths_exhaustion += counts.deaths_exhaustion;
        tt.deaths_user += counts.deaths_user;
        tt.reproduction_attempts += counts.reproduction_attempts;
        tt.rejected_births_viability += counts.rejected_births_viability;
        tt.rejected_births_mutation += counts.rejected_births_mutation;
        tt.rejected_births_capacity += counts.rejected_births_capacity;
        tt.topology_mutations += counts.topology_mutations;
        tt.controller_evaluations += counts.controller_evaluations;
        tt.parameter_evaluations += counts.parameter_evaluations;
        tt.emitting += counts.emitting;
        tt.cone_projections += counts.cone_projections;
        prof.stop_other(OtherPart::Publish, ps);
        prof.mark_tick();
        self.profile.absorb(&prof);
        self.sense_scratch.0 = senses;
        timing.validate_publish_ms = t_validate.ms();
        timing.total_ms = t_start.ms();
        let report = TickReport { tick: event, simulation_seconds: event as f64 * dt, population: self.orgs.len(), counts, ledger: led, timing };
        self.last_report = Some(report);
        Ok(report)
    }

    /// Deterministic digest of canonical scientific state (excludes execution settings).
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        eat(self.tick);
        eat(self.next_org_id);
        for v in &self.field.e {
            eat(v.to_bits() as u64);
        }
        for v in &self.field.m {
            eat(v.to_bits() as u64);
        }
        let o = &self.orgs;
        for i in 0..o.len() {
            for v in [o.id[i], o.lineage[i], o.birth_tick[i], o.cooldown_until[i], o.held_tick[i], o.genome[i].id] {
                eat(v);
            }
            for v in [o.x[i], o.y[i], o.vx[i], o.vy[i], o.energy[i], o.energy_change_rate[i]] {
                eat(v.to_bits());
            }
            for v in &o.held[i] {
                eat(v.to_bits() as u64);
            }
            match &o.genome[i].params {
                Weights::F32(w) => {
                    for v in w {
                        eat(v.to_bits() as u64);
                    }
                }
                Weights::F16(w) => {
                    for &v in w {
                        eat(v as u64);
                    }
                }
            }
            if let Some(m) = o.memory.get(i) {
                for v in m {
                    eat(v.to_bits() as u64);
                }
            }
        }
        h
    }

    /// Inspect one organism by stable identity.
    pub fn inspect(&self, id: u64) -> Option<Inspection> {
        let i = self.orgs.index_of(id)?;
        let o = &self.orgs;
        let g = &o.genome[i];
        let dt = self.dt();
        let pyr = Pyramid::build(&self.field, self.pyramid_extra_levels);
        let obs = observe(&pyr.sample(o.x[i], o.y[i], g.scale()), &self.cfg.sensors);
        let age = (self.tick - o.birth_tick[i]) as f64 * dt;
        let intr = intrinsic(&self.cfg.sensors, o.energy[i], age, o.vx[i], o.vy[i], o.energy_change_rate[i]);
        let mut observation: Vec<f32> = obs.field_inputs.to_vec();
        observation.extend_from_slice(&intr);
        let decoded = o.held_valid[i].then(|| {
            let h = &o.held[i];
            let em = &self.cfg.emission;
            let rp = &self.cfg.reproduction;
            let dn: f64 = h[out::EMIT_DIR..out::EMIT_DIR + CHANNELS].iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
            Decoded {
                reproduction_rate_per_second: rp.maximum_attempt_rate_per_second * sigmoid(h[out::SPLIT] as f64),
                child_share: rp.minimum_child_fraction + (rp.maximum_child_fraction - rp.minimum_child_fraction) * sigmoid(h[out::SHARE] as f64),
                emission_gate: sigmoid(h[out::EMIT_GATE] as f64),
                emission_gate_open: sigmoid(h[out::EMIT_GATE] as f64) > em.gate_threshold && dn >= em.direction_zero_threshold,
                emission_fraction_per_tick: rate_to_probability(em.maximum_fractional_rate_per_second * sigmoid(h[out::EMIT_RATE] as f64), dt),
            }
        });
        Some(Inspection {
            id,
            lineage: o.lineage[i],
            generation: o.generation[i],
            genome_id: g.id,
            parent_genome_id: g.parent,
            position: [o.x[i], o.y[i]],
            velocity: [o.vx[i], o.vy[i]],
            energy: o.energy[i],
            age_seconds: age,
            birth_tick: o.birth_tick[i],
            cooldown_until_tick: o.cooldown_until[i],
            energy_change_rate: o.energy_change_rate[i],
            topology: g.hidden().to_vec(),
            parameters: g.param_count(),
            sensing_scale: g.scale(),
            affinity: g.affinity.to_vec(),
            motion_response: g.motion.to_vec(),
            held_output_valid: o.held_valid[i],
            held_output_tick: o.held_tick[i],
            raw_outputs: o.held[i].to_vec(),
            decoded,
            observation_of_committed_field: observation,
            mutation: g.mutation.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Decoded {
    pub reproduction_rate_per_second: f64,
    pub child_share: f64,
    pub emission_gate: f64,
    pub emission_gate_open: bool,
    pub emission_fraction_per_tick: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Inspection {
    pub id: u64,
    pub lineage: u64,
    pub generation: u32,
    pub genome_id: u64,
    pub parent_genome_id: Option<u64>,
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    pub energy: f64,
    pub age_seconds: f64,
    pub birth_tick: u64,
    pub cooldown_until_tick: u64,
    pub energy_change_rate: f64,
    pub topology: Vec<usize>,
    pub parameters: usize,
    pub sensing_scale: f64,
    pub affinity: Vec<f32>,
    pub motion_response: Vec<f32>,
    pub held_output_valid: bool,
    pub held_output_tick: u64,
    pub raw_outputs: Vec<f32>,
    pub decoded: Option<Decoded>,
    /// Diagnostic re-observation of the *committed* field (not the input of the held decision).
    pub observation_of_committed_field: Vec<f32>,
    pub mutation: Option<MutationRecord>,
}

/// One independent controller evaluation: this organism's own genome and input.
pub struct InferenceJob<'a> {
    pub organism: u64,
    pub genome: &'a Genome,
    pub input: &'a [f32; INPUT_WIDTH],
}

/// Pluggable population-inference backend (plan §10, §18.3 "core backend port").
///
/// A backend evaluates many *independent* networks; it must never share
/// weights between organisms. Backends other than the built-in scalar
/// reference evaluator generally differ from it in floating-point
/// accumulation order, so a run that uses one has its own determinism class
/// and is compared to the reference by tolerance (plan §10.3, §10.7), not by
/// state hash.
pub trait ControllerBackend {
    fn name(&self) -> &str;
    /// Fill `out[j]` with the 36 raw logits of `jobs[j]`.
    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String>;
}

/// How a tick evaluates sensing and controllers.
pub enum Inference<'a> {
    /// Built-in CPU sensing and scalar reference inference (bitwise reference).
    Reference,
    /// CPU sensing; controllers evaluated by an external backend.
    Controller(&'a mut dyn ControllerBackend),
    /// Sensing pyramid, sampling and inference all done by one backend.
    Fused(&'a mut dyn SenseInferBackend),
    /// Field transport/sources plus fused sensing and inference on one device backend.
    Device(&'a mut dyn DeviceBackend),
    /// The field stays on the device across ticks; exchange and deposits run there too.
    Resident(&'a mut dyn ResidentBackend),
}

/// Energy accounting of one field transport + source step (all in cell-integrated energy).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FieldAccounting {
    /// Energy removed by exponential decay (dissipation).
    pub decay_removed: f64,
    /// Energy actually stored by source injection.
    pub injected: f64,
    /// Energy cleared from tiny reservoirs (dissipation).
    pub tiny_removed: f64,
    pub subcycles: u32,
}

/// Phase-2 backend: transport the field, inject scheduled sources, clear tiny
/// reservoirs, producing the observation field in place (plan §4.5, §11).
pub trait FieldBackend {
    fn name(&self) -> &str;
    /// `sources` are already advanced to tick `n`.
    fn transport_and_sources(&mut self, field: &mut Field, cfg: &Config, sources: &[Source], n: u64) -> Result<FieldAccounting, String>;
}

/// A backend that runs both the field step and fused sensing + inference, so the
/// observation field can stay on the device between them.
pub trait DeviceBackend: SenseInferBackend + FieldBackend {}
impl<T: SenseInferBackend + FieldBackend> DeviceBackend for T {}

/// The CPU field step (same arithmetic as the built-in path).
pub fn cpu_transport_and_sources(field: &mut Field, cfg: &Config, sources: &[Source], n: u64) -> Result<FieldAccounting, String> {
    let mut scratch = Field { gx: 0, gy: 0, hx: field.hx, hy: field.hy, e: Vec::new(), m: Vec::new() };
    let (decay_removed, subcycles) = field.transport(cfg.world.diffusion, cfg.world.decay_per_second, cfg.dt(), &mut scratch);
    let injected = inject(field, sources, n, cfg)?;
    let tiny_removed = field.clear_tiny(cfg.world.tiny_energy_threshold);
    Ok(FieldAccounting { decay_removed, injected, tiny_removed, subcycles })
}

/// CPU reference for the whole device port (reproduces the built-in state hash).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceDevice;

impl FieldBackend for ReferenceDevice {
    fn name(&self) -> &str {
        "cpu-reference-field"
    }
    fn transport_and_sources(&mut self, field: &mut Field, cfg: &Config, sources: &[Source], n: u64) -> Result<FieldAccounting, String> {
        cpu_transport_and_sources(field, cfg, sources, n)
    }
}

impl SenseInferBackend for ReferenceDevice {
    fn name(&self) -> &str {
        "cpu-reference-fused"
    }
    fn sense_and_infer(&mut self, field: &Field, sensors: &crate::config::SensorsCfg, extra_levels: usize, jobs: &[SenseJob<'_>], out: &mut [SenseOutput]) -> Result<(), String> {
        ReferenceSenseInfer.sense_and_infer(field, sensors, extra_levels, jobs, out)
    }
}

/// A field addition: a bilinear-stencil packet (emission, recycling) or an exact cell (energy brush).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldDeposit {
    Stencil { x: f64, y: f64, q: f64, dir: [f32; CHANNELS] },
    Cell { cell: usize, q: f64, dir: [f32; CHANNELS] },
}

/// One organism's uptake request (resident path).
pub struct UptakeJob<'a> {
    pub x: f64,
    pub y: f64,
    pub demand: f64,
    pub genome: &'a Genome,
}

/// Packet-chemistry result for one organism before the toxicity clip (which needs its energy).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UptakeOut {
    pub absorbed: f64,
    pub useful_gain: f64,
    pub toxicity_requested: f64,
}

/// Phase-10 summary of the working field.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FinishReport {
    pub tiny_removed: f64,
    /// Worst relative cone excess before projection (violations up to 1e-3 are projected).
    pub worst_cone: f64,
    pub min_energy: f64,
    pub cone_projections: u32,
    /// Total field energy after clearing and projection.
    pub field_energy: f64,
}

/// A backend that keeps the authoritative field on a device across ticks (plan §9.5, §10.5).
///
/// Each tick: `begin_tick` copies the committed field to a working field and applies
/// boundary commands; then `transport_resident`, `sense_and_infer`, `uptake` (which also
/// debits granted energy), `deposit` and `finish` act on the working field; `commit`
/// publishes it, `abort` discards it.
pub trait ResidentBackend: SenseInferBackend {
    fn upload_committed(&mut self, field: &Field, cfg: &Config) -> Result<(), String>;
    fn download_committed(&mut self, field: &mut Field) -> Result<(), String>;
    fn committed_energy(&self) -> f64;
    fn begin_tick(&mut self, ops: &[FieldDeposit]) -> Result<(), String>;
    /// Queue the field step without waiting. The GPU implementation returns
    /// after `queue.submit`, so the caller can pack sense jobs while it runs.
    fn submit_transport(&mut self, cfg: &Config, sources: &[Source], n: u64) -> Result<(), String>;
    /// Wait for the field step submitted by [`Self::submit_transport`] and return its accounting.
    fn finish_transport(&mut self) -> Result<FieldAccounting, String>;
    fn transport_resident(&mut self, cfg: &Config, sources: &[Source], n: u64) -> Result<FieldAccounting, String> {
        self.submit_transport(cfg, sources, n)?;
        self.finish_transport()
    }
    /// Fills `out` and debits granted energy from the working field; returns total granted energy.
    fn uptake(&mut self, chemistry: &crate::config::ChemistryCfg, jobs: &[UptakeJob<'_>], out: &mut [UptakeOut]) -> Result<f64, String>;
    /// Optional single-wait front half for a tick whose transport was already queued
    /// with [`Self::submit_transport`]: sensing, inference, the uptake request
    /// (`dt * umax * r / (K + r)` from the sensed level-0 density, evaluated on the
    /// device) and the uptake exchange, all read back in one wait.
    ///
    /// `uptake_out[k]` belongs to `jobs[k]`. Returns the transport accounting and the
    /// total granted energy. `None` (the default) means the backend does not offer it
    /// and the caller uses `finish_transport`, `sense_and_infer` and `uptake`.
    fn sense_uptake(&mut self, _cfg: &Config, _extra_levels: usize, _jobs: &[SenseJob<'_>], _sense_out: &mut [SenseOutput], _uptake_out: &mut [UptakeOut]) -> Option<Result<(FieldAccounting, f64), String>> {
        None
    }
    fn deposit(&mut self, deposits: &[FieldDeposit]) -> Result<(), String>;
    fn finish(&mut self, tiny: f64) -> Result<FinishReport, String>;
    fn commit(&mut self) -> Result<(), String>;
    fn abort(&mut self);
}

fn apply_deposit(field: &mut Field, d: &FieldDeposit) {
    match *d {
        FieldDeposit::Stencil { x, y, q, ref dir } => {
            let s = field.stencil(x, y);
            field.deposit(&s, q, dir);
        }
        FieldDeposit::Cell { cell, q, ref dir } => {
            let s = Stencil { n: 1, cells: [cell, 0, 0, 0], w: [1.0, 0.0, 0.0, 0.0] };
            field.deposit(&s, q, dir);
        }
    }
}

/// CPU reference of the resident port: the same arithmetic as the built-in path
/// (reproduces its state hash and ledger), with the field held "on the device".
#[derive(Debug, Default, Clone)]
pub struct ReferenceResident {
    committed: Option<Field>,
    working: Option<Field>,
    pending_energy: f64,
    committed_energy: f64,
    pending_transport: Option<FieldAccounting>,
}

impl SenseInferBackend for ReferenceResident {
    fn name(&self) -> &str {
        "cpu-reference-resident"
    }
    fn sense_and_infer(&mut self, _field: &Field, sensors: &crate::config::SensorsCfg, extra_levels: usize, jobs: &[SenseJob<'_>], out: &mut [SenseOutput]) -> Result<(), String> {
        let w = self.working.as_ref().ok_or("no working field")?;
        ReferenceSenseInfer.sense_and_infer(w, sensors, extra_levels, jobs, out)
    }
}

impl ResidentBackend for ReferenceResident {
    fn upload_committed(&mut self, field: &Field, _cfg: &Config) -> Result<(), String> {
        self.committed_energy = field.total_energy();
        self.committed = Some(field.clone());
        Ok(())
    }
    fn download_committed(&mut self, field: &mut Field) -> Result<(), String> {
        *field = self.committed.clone().ok_or("nothing uploaded")?;
        Ok(())
    }
    fn committed_energy(&self) -> f64 {
        self.committed_energy
    }
    fn begin_tick(&mut self, ops: &[FieldDeposit]) -> Result<(), String> {
        let mut w = self.committed.clone().ok_or("nothing uploaded")?;
        for d in ops {
            apply_deposit(&mut w, d);
        }
        self.working = Some(w);
        Ok(())
    }
    fn submit_transport(&mut self, cfg: &Config, sources: &[Source], n: u64) -> Result<(), String> {
        let acct = cpu_transport_and_sources(self.working.as_mut().ok_or("no working field")?, cfg, sources, n)?;
        self.pending_transport = Some(acct);
        Ok(())
    }
    fn finish_transport(&mut self) -> Result<FieldAccounting, String> {
        self.pending_transport.take().ok_or_else(|| "transport was not submitted".into())
    }
    fn uptake(&mut self, ch: &crate::config::ChemistryCfg, jobs: &[UptakeJob<'_>], out: &mut [UptakeOut]) -> Result<f64, String> {
        let field = self.working.as_mut().ok_or("no working field")?;
        let stencils: Vec<Stencil> = jobs.iter().map(|j| field.stencil(j.x, j.y)).collect();
        let mut cell_demand = vec![0.0f64; field.cells()];
        for (j, s) in jobs.iter().zip(&stencils) {
            for k in 0..s.n {
                cell_demand[s.cells[k]] += j.demand * s.w[k];
            }
        }
        let mut granted_cell = vec![0.0f64; field.cells()];
        for ((j, s), o) in jobs.iter().zip(&stencils).zip(out.iter_mut()) {
            let a = &j.genome.affinity;
            let mut packets = [(0.0f64, 0.0f64); 4];
            for k in 0..s.n {
                let c = s.cells[k];
                let ec = field.e[c] as f64;
                let u = grant_scale(ec, cell_demand[c]) * j.demand * s.w[k];
                granted_cell[c] += u;
                let az = if ec > 0.0 {
                    let m = field.moment(c);
                    (0..CHANNELS).map(|q| a[q] as f64 * m[q] as f64).sum::<f64>() / ec
                } else {
                    0.0
                };
                packets[k] = (u, eta(ch.maximum_efficiency, ch.affinity_gain, ch.discrimination_threshold, az));
            }
            // The toxicity clip needs the organism's energy; the caller applies it.
            let st = settle(f64::INFINITY, ch.toxicity_sensitivity, packets[..s.n].iter().copied());
            *o = UptakeOut { absorbed: st.absorbed, useful_gain: st.useful_gain, toxicity_requested: st.toxicity_requested };
        }
        let mut transferred = 0.0;
        for c in 0..field.cells() {
            let g = granted_cell[c];
            if g > 0.0 {
                let e = field.e[c] as f64;
                let e_new = (e - g).max(0.0);
                let e_new32 = e_new as f32;
                let f = if e > 0.0 { e_new32 as f64 / e } else { 0.0 };
                field.e[c] = e_new32;
                for v in &mut field.m[c * CHANNELS..(c + 1) * CHANNELS] {
                    *v = (*v as f64 * f) as f32;
                }
                transferred += g;
            }
        }
        Ok(transferred)
    }
    fn deposit(&mut self, deposits: &[FieldDeposit]) -> Result<(), String> {
        let field = self.working.as_mut().ok_or("no working field")?;
        let mut acc = Field::new(field.gx, field.gy, field.width(), field.height());
        for d in deposits {
            apply_deposit(&mut acc, d);
        }
        field.add_assign(&acc);
        Ok(())
    }
    fn finish(&mut self, tiny: f64) -> Result<FinishReport, String> {
        let field = self.working.as_mut().ok_or("no working field")?;
        let tiny_removed = field.clear_tiny(tiny);
        let (worst, min_e) = field.cone_report();
        let mut proj = 0;
        if worst > 0.0 && worst <= 1e-3 {
            proj = field.project_cone() as u32;
        }
        let total = field.total_energy();
        self.pending_energy = total;
        Ok(FinishReport { tiny_removed, worst_cone: worst, min_energy: min_e, cone_projections: proj, field_energy: total })
    }
    fn commit(&mut self) -> Result<(), String> {
        self.committed = Some(self.working.take().ok_or("no working field")?);
        self.committed_energy = self.pending_energy;
        Ok(())
    }
    fn abort(&mut self) {
        self.working = None;
    }
}

/// One organism's sensing (+ evaluation when `due`) request for a fused backend.
pub struct SenseJob<'a> {
    pub organism: u64,
    pub genome: &'a Genome,
    pub x: f64,
    pub y: f64,
    /// The five intrinsic inputs (energy, age, vx, vy, energy change), already normalized.
    pub intrinsic: [f32; 5],
    /// Whether the controller must be evaluated this tick.
    pub due: bool,
    /// The organism's recurrent memory going in (ignored by memoryless genomes).
    pub memory: &'a [f32; MEMORY_WIDTH],
}

/// What a fused backend returns per organism.
#[derive(Debug, Clone, Copy)]
pub struct SenseOutput {
    /// 36 raw logits (meaningful only when the job was `due`).
    pub logits: [f32; OUTPUT_WIDTH],
    /// `Σ_k motion_k · ∂q/∂x_k` style drive: the genome's 33 motion sensitivities
    /// dotted with the sensed physical gradients of `[q, Z_0..Z_31]` (x, y).
    pub motion_drive: [f64; 2],
    /// Local level-0 energy density (uptake request input).
    pub density0: f64,
    /// Next recurrent memory (meaningful only for a recurrent genome when the job was `due`).
    pub memory: [f32; MEMORY_WIDTH],
}

impl Default for SenseOutput {
    fn default() -> Self {
        SenseOutput { logits: [0.0; OUTPUT_WIDTH], motion_drive: [0.0; 2], density0: 0.0, memory: [0.0; MEMORY_WIDTH] }
    }
}

/// Fused sensing + inference backend (plan §10, §11: device-resident sensing).
/// Implementations other than [`ReferenceSenseInfer`] are tolerance-equivalent,
/// not bitwise, and form their own determinism class.
pub trait SenseInferBackend {
    fn name(&self) -> &str;
    fn sense_and_infer(&mut self, field: &Field, sensors: &crate::config::SensorsCfg, extra_levels: usize, jobs: &[SenseJob<'_>], out: &mut [SenseOutput]) -> Result<(), String>;
}

/// The motion drive as fl-core computes it (fixed summation order).
pub fn motion_drive(motion: &[f32; PLANES], gx: &[f64; PLANES], gy: &[f64; PLANES]) -> (f64, f64) {
    let (mut fx, mut fy) = (0.0, 0.0);
    for k in 0..PLANES {
        fx += motion[k] as f64 * gx[k];
        fy += motion[k] as f64 * gy[k];
    }
    (fx, fy)
}

/// CPU reference implementation of the fused port (same arithmetic as the built-in path).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceSenseInfer;

impl SenseInferBackend for ReferenceSenseInfer {
    fn name(&self) -> &str {
        "cpu-reference-fused"
    }
    fn sense_and_infer(&mut self, field: &Field, sensors: &crate::config::SensorsCfg, extra_levels: usize, jobs: &[SenseJob<'_>], out: &mut [SenseOutput]) -> Result<(), String> {
        let pyr = Pyramid::build(field, extra_levels);
        let mut scratch = vec![0.0f32; 2 * MAX_HIDDEN];
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            let obs = observe(&pyr.sample(j.x, j.y, j.genome.scale()), sensors);
            let (dx, dy) = motion_drive(&j.genome.motion, &obs.grad_x, &obs.grad_y);
            o.motion_drive = [dx, dy];
            o.density0 = pyr.density0(j.x, j.y);
            if j.due {
                let mut input = [0.0f32; INPUT_WIDTH];
                input[..99].copy_from_slice(&obs.field_inputs);
                input[99..].copy_from_slice(&j.intrinsic);
                forward_state(j.genome.hidden(), &j.genome.params.to_f32(), &input, j.memory, &mut scratch, &mut o.logits, &mut o.memory);
            }
        }
        Ok(())
    }
}

/// The scalar reference evaluator as a backend (same arithmetic as `World::step`).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceBackend;

impl ControllerBackend for ReferenceBackend {
    fn name(&self) -> &str {
        "cpu-reference-scalar"
    }
    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        if jobs.iter().any(|j| is_recurrent(j.genome.hidden())) {
            return Err("this controller-only backend has no memory; recurrent networks need the CPU path or a fused GPU backend".into());
        }
        let mut scratch = vec![0.0f32; 2 * MAX_HIDDEN];
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            forward(j.genome.hidden(), &j.genome.params.to_f32(), j.input, &mut scratch, o);
        }
        Ok(())
    }
}


#[allow(clippy::too_many_arguments)]
fn sense_one<const PROFILE: bool>(cfg: &Config, pyr: &Pyramid, orgs: &Organisms, genomes: &[Arc<Genome>], i: usize, n: u64, scratch: &mut [f32], infer: bool, sense_ns: &mut u64, net_ns: &mut u64) -> Sense {
    let t_sense = Stamp::capture::<PROFILE>();
    let g = &genomes[i];
    let raw = pyr.sample(orgs.x[i], orgs.y[i], g.scale());
    let obs = observe(&raw, &cfg.sensors);
    let density0 = pyr.density0(orgs.x[i], orgs.y[i]);
    let k = cfg.controller.evaluation_interval_ticks as u64;
    let due = !orgs.held_valid[i] || (n + 1) % k == orgs.phase[i] as u64;
    let mut out = orgs.held[i];
    let mem_in = orgs.memory.get(i).unwrap_or(&ZERO_MEMORY);
    let has_memory = !orgs.memory.is_empty();
    if due {
        let age = (n - orgs.birth_tick[i]) as f64 * cfg.dt();
        let intr = intrinsic(&cfg.sensors, orgs.energy[i], age, orgs.vx[i], orgs.vy[i], orgs.energy_change_rate[i]);
        let mut input = [0.0f32; INPUT_WIDTH];
        input[..99].copy_from_slice(&obs.field_inputs);
        input[99..].copy_from_slice(&intr);
        let (dx, dy) = motion_drive(&g.motion, &obs.grad_x, &obs.grad_y);
        charge::<PROFILE>(t_sense, sense_ns);
        if infer {
            let t_net = Stamp::capture::<PROFILE>();
            let mut mem = *mem_in;
            forward_state(g.hidden(), &g.params.to_f32(), &input, mem_in, scratch, &mut out, &mut mem);
            charge::<PROFILE>(t_net, net_ns);
            return Sense { density0, out, evaluated: true, mem: has_memory.then(|| Box::new(mem)), motion_drive: [dx, dy], input: None };
        }
        return Sense { density0, out, evaluated: true, mem: None, motion_drive: [dx, dy], input: Some(Box::new(input)) };
    }
    let (dx, dy) = motion_drive(&g.motion, &obs.grad_x, &obs.grad_y);
    charge::<PROFILE>(t_sense, sense_ns);
    Sense { density0, out, evaluated: false, mem: None, motion_drive: [dx, dy], input: None }
}

#[inline(always)]
fn charge<const PROFILE: bool>(stamp: Stamp, acc: &mut u64) {
    if PROFILE {
        *acc += stamp.elapsed_ns();
    }
}

#[allow(clippy::too_many_arguments)]
fn sense_all<const PROFILE: bool>(cfg: &Config, pyr: &Pyramid, orgs: &Organisms, genomes: &[Arc<Genome>], dead: &[bool], n: u64, workers: usize, infer: bool) -> (Vec<Sense>, u64, u64) {
    let len = orgs.len();
    let run = |range: std::ops::Range<usize>| -> (Vec<Sense>, u64, u64) {
        let mut scratch = vec![0.0f32; 2 * MAX_HIDDEN];
        let mut sense_ns = 0u64;
        let mut net_ns = 0u64;
        let v = range
            .map(|i| {
                if dead[i] {
                    Sense::blank()
                } else {
                    sense_one::<PROFILE>(cfg, pyr, orgs, genomes, i, n, &mut scratch, infer, &mut sense_ns, &mut net_ns)
                }
            })
            .collect();
        (v, sense_ns, net_ns)
    };
    let workers = workers.max(1).min(len.max(1));
    if workers == 1 || len < 256 {
        return run(0..len);
    }
    let chunk = len.div_ceil(workers);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|wi| {
                let r = (wi * chunk).min(len)..((wi + 1) * chunk).min(len);
                s.spawn(move || run(r))
            })
            .collect();
        let mut all = Vec::with_capacity(len);
        let mut sense_ns = 0u64;
        let mut net_ns = 0u64;
        for h in handles {
            let (part, s_ns, n_ns) = h.join().expect("sensing worker panicked");
            all.extend(part);
            sense_ns += s_ns;
            net_ns += n_ns;
        }
        (all, sense_ns, net_ns)
    })
}
