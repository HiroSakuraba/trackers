//! The world state and the ordered, transactional tick (plan §8).

use std::sync::Arc;

use serde::Serialize;

use crate::chemistry::{drag_force_coefficient, eta, grant_scale, rate_to_probability, settle, split_energy};
use crate::config::{CapacityPolicy, Config};
use crate::field::{min_image, wrap, Field, Stencil};
use crate::genome::{founder, mutate, Genome, MutationRecord};
use crate::nn::{forward, sigmoid, INPUT_WIDTH, MAX_HIDDEN, OUTPUT_WIDTH};
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
        r!(id, lineage, generation, x, y, vx, vy, energy, birth_tick, cooldown_until, energy_change_rate, phase, held, held_tick, held_valid, genome);
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
#[derive(Clone)]
struct Sense {
    density0: f64,
    out: [f32; OUTPUT_WIDTH],
    evaluated: bool,
    /// `Σ_k motion_k · ∂_k` (x, y) over the sensed physical gradients of `[q, Z_0..Z_31]`.
    motion_drive: [f64; 2],
    /// Controller input of a due evaluation, kept only for an external controller backend.
    input: Option<Box<[f32; INPUT_WIDTH]>>,
}

impl Sense {
    fn blank() -> Sense {
        Sense { density0: 0.0, out: [0.0; OUTPUT_WIDTH], evaluated: false, motion_drive: [0.0; 2], input: None }
    }
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
        })
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

    /// Advance `n -> n+1` with an explicit inference route. On error the committed world is unchanged.
    pub fn step_inference(&mut self, inference: Inference<'_>) -> Result<TickReport, TickError> {
        let t_start = Stopwatch::start();
        let cfg = &self.cfg;
        let dt = cfg.dt();
        let n = self.tick;
        let event = n.checked_add(1).ok_or(TickError::Overflow { detail: "tick counter".into() })?;
        let tiny = cfg.world.tiny_energy_threshold;
        let mut timing = TickTiming::default();
        let mut counts = TickCounts::default();
        let mut led = TickLedger { energy_before: self.total_energy(), ..Default::default() };
        let mut dis = Dissipation::default();

        let (backend, fused, mut device) = match inference {
            Inference::Reference => (None, None, None),
            Inference::Controller(b) => (Some(b), None, None),
            Inference::Fused(f) => (None, Some(f), None),
            Inference::Device(d) => (None, None, Some(d)),
        };
        let t_setup = Stopwatch::start();
        let mut field = self.field.clone();
        let mut orgs = self.orgs.clone();
        let mut sources = self.sources.clone();
        let e_start = orgs.energy.clone();
        let mut dead = vec![false; orgs.len()];
        let mut deposits = Field::new(field.gx, field.gy, field.width(), field.height());

        // ── Phase 1: admit commands recorded for this boundary ────────────
        let mut due: Vec<&ScheduledCommand> = self.pending.iter().filter(|c| c.tick == n).collect();
        due.sort_by_key(|c| c.seq);
        for sc in &due {
            match &sc.command {
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
                                    let s = field.stencil(orgs.x[i], orgs.y[i]);
                                    field.deposit(&s, e, &orgs.genome[i].affinity);
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
                        let s = Stencil { n: 1, cells: [c, 0, 0, 0], w: [1.0, 0.0, 0.0, 0.0] };
                        field.deposit(&s, q, &dir);
                    }
                    led.injected_commands += amount.max(0.0);
                }
            }
        }

        timing.setup_ms = t_setup.ms();

        // ── Phase 2: transport, then scheduled sources → observation field ─
        let t = Stopwatch::start();
        for s in &mut sources {
            s.advance_to(n, cfg, &self.rng);
        }
        if let Some(d) = device.as_mut() {
            let acct = d
                .transport_and_sources(&mut field, cfg, &sources, n)
                .map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("field backend {}: {detail}", FieldBackend::name(&**d)) })?;
            dis.decay += acct.decay_removed;
            counts.transport_subcycles = acct.subcycles;
            led.injected_sources = acct.injected;
            dis.tiny_reservoir += acct.tiny_removed;
        } else {
            let acct = cpu_transport_and_sources(&mut field, cfg, &sources, n).map_err(|detail| TickError::NumericalFault { tick: event, detail })?;
            dis.decay += acct.decay_removed;
            counts.transport_subcycles = acct.subcycles;
            led.injected_sources = acct.injected;
            dis.tiny_reservoir += acct.tiny_removed;
        }
        timing.transport_ms = t.ms();

        // ── Phase 3–4: sensing pyramid, sample organisms, evaluate due controllers ──
        let t = Stopwatch::start();
        let fused: Option<&mut dyn SenseInferBackend> = match device {
            Some(d) => Some(d),
            None => fused,
        };
        let mut senses = if let Some(fb) = fused {
            let k_eval = cfg.controller.evaluation_interval_ticks as u64;
            let idx: Vec<usize> = (0..orgs.len()).filter(|&i| !dead[i]).collect();
            let jobs: Vec<SenseJob<'_>> = idx
                .iter()
                .map(|&i| {
                    let age = (n - orgs.birth_tick[i]) as f64 * dt;
                    SenseJob {
                        organism: orgs.id[i],
                        genome: &orgs.genome[i],
                        x: orgs.x[i],
                        y: orgs.y[i],
                        intrinsic: intrinsic(&cfg.sensors, orgs.energy[i], age, orgs.vx[i], orgs.vy[i], orgs.energy_change_rate[i]),
                        due: !orgs.held_valid[i] || (n + 1) % k_eval == orgs.phase[i] as u64,
                    }
                })
                .collect();
            let mut outs = vec![SenseOutput::default(); jobs.len()];
            let tb = Stopwatch::start();
            let res = fb.sense_and_infer(&field, &cfg.sensors, self.pyramid_extra_levels, &jobs, &mut outs);
            timing.backend_inference_ms = tb.ms();
            res.map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("backend {}: {detail}", fb.name()) })?;
            let mut senses = vec![Sense::blank(); orgs.len()];
            for (k, &i) in idx.iter().enumerate() {
                let o = &outs[k];
                let finite = o.motion_drive.iter().all(|v| v.is_finite()) && o.density0.is_finite() && (!jobs[k].due || o.logits.iter().all(|v| v.is_finite()));
                if !finite {
                    return Err(TickError::NumericalFault { tick: event, detail: format!("backend {} produced non-finite results for organism {}", fb.name(), orgs.id[i]) });
                }
                senses[i].density0 = o.density0.max(0.0);
                senses[i].motion_drive = o.motion_drive;
                senses[i].evaluated = jobs[k].due;
                senses[i].out = if jobs[k].due { o.logits } else { orgs.held[i] };
            }
            senses
        } else {
            let pyr = Pyramid::build(&field, self.pyramid_extra_levels);
            sense_all(cfg, &pyr, &orgs, &dead, n, self.workers, backend.is_none())
        };
        if let Some(be) = backend {
            let due: Vec<usize> = (0..senses.len()).filter(|&i| senses[i].evaluated).collect();
            let jobs: Vec<InferenceJob<'_>> =
                due.iter().map(|&i| InferenceJob { organism: orgs.id[i], genome: &orgs.genome[i], input: senses[i].input.as_deref().expect("controller path keeps due inputs") }).collect();
            let mut outs = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
            let tb = Stopwatch::start();
            let res = be.evaluate(&jobs, &mut outs);
            timing.backend_inference_ms = tb.ms();
            res.map_err(|detail| TickError::NumericalFault { tick: event, detail: format!("backend {}: {detail}", be.name()) })?;
            drop(jobs);
            for (k, &i) in due.iter().enumerate() {
                if outs[k].iter().any(|v| !v.is_finite()) {
                    return Err(TickError::NumericalFault { tick: event, detail: format!("backend {} produced non-finite logits for organism {}", be.name(), orgs.id[i]) });
                }
                senses[i].out = outs[k];
            }
        }
        for (i, s) in senses.iter().enumerate() {
            if s.evaluated {
                orgs.held[i] = s.out;
                orgs.held_tick[i] = event;
                orgs.held_valid[i] = true;
                counts.controller_evaluations += 1;
                counts.parameter_evaluations += orgs.genome[i].param_count() as u64;
            }
        }
        timing.sense_and_infer_ms = t.ms();

        // ── Phase 5: uptake requests and cell grants ──────────────────────
        let t = Stopwatch::start();
        let ch = &cfg.chemistry;
        let umax = ch.uptake_capacity_per_second;
        let stencils: Vec<Stencil> = (0..orgs.len()).map(|i| field.stencil(orgs.x[i], orgs.y[i])).collect();
        let demand: Vec<f64> = (0..orgs.len())
            .map(|i| if dead[i] { 0.0 } else { let r = senses[i].density0; dt * umax * r / (ch.uptake_half_saturation_density + r) })
            .collect();
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
        let mut uptake_total = vec![0.0f64; orgs.len()];
        let mut settlements = Vec::with_capacity(orgs.len());
        for i in 0..orgs.len() {
            if demand[i] <= 0.0 {
                settlements.push(Default::default());
                continue;
            }
            let s = &stencils[i];
            let a = &orgs.genome[i].affinity;
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

        // ── Phase 6: settle gain, toxicity, maintenance ───────────────────
        let co = &cfg.costs;
        let exhaust = |e: &mut f64, dead: &mut bool, dis: &mut Dissipation, counts: &mut TickCounts| {
            if *e <= tiny {
                dis.tiny_reservoir += e.max(0.0);
                *e = 0.0;
                if !*dead {
                    *dead = true;
                    counts.deaths_exhaustion += 1;
                }
                true
            } else {
                false
            }
        };
        for i in 0..orgs.len() {
            if dead[i] {
                continue;
            }
            let st = settlements[i];
            orgs.energy[i] += st.net();
            dis.uptake_inefficiency += st.absorbed - st.useful_gain;
            dis.toxicity += st.toxicity_paid;
            let g = &orgs.genome[i];
            let cost = dt * (co.basal_energy_per_second + co.sensor_energy_per_second_at_unit_scale * (g.scale() / co.sensor_unit_scale_world_units).powf(co.sensor_scale_cost_exponent))
                + if senses[i].evaluated { co.neural_energy_per_parameter_evaluation * g.param_count() as f64 } else { 0.0 };
            let paid = cost.min(orgs.energy[i].max(0.0));
            orgs.energy[i] -= paid;
            dis.maintenance += paid;
            if paid < cost {
                orgs.energy[i] = orgs.energy[i].max(0.0);
                dis.tiny_reservoir += orgs.energy[i];
                orgs.energy[i] = 0.0;
                dead[i] = true;
                counts.deaths_exhaustion += 1;
                continue;
            }
            exhaust(&mut orgs.energy[i], &mut dead[i], &mut dis, &mut counts);
        }
        timing.exchange_ms = t.ms();

        // ── Phase 7: motion → emission → reproduction proposals ──────────
        let t = Stopwatch::start();
        let drag_keep = (-ch.drag_per_second * dt).exp();
        let drag_coef = drag_force_coefficient(ch.drag_per_second, dt);
        let em = &cfg.emission;
        let rp = &cfg.reproduction;
        let (w, h) = (field.width(), field.height());
        let mut proposals: Vec<BirthProposal> = Vec::new();
        for i in 0..orgs.len() {
            if dead[i] {
                continue;
            }
            // Motion (field-mediated chemotaxis; no pairwise forces).
            let g = orgs.genome[i].clone();
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
            if cost > orgs.energy[i] {
                let f = (orgs.energy[i] / cost).sqrt();
                vx *= f;
                vy *= f;
                cost = orgs.energy[i];
            }
            orgs.energy[i] -= cost;
            dis.motion += cost;
            orgs.vx[i] = vx;
            orgs.vy[i] = vy;
            orgs.x[i] = wrap(orgs.x[i] + dt * vx, w);
            orgs.y[i] = wrap(orgs.y[i] + dt * vy, h);
            if exhaust(&mut orgs.energy[i], &mut dead[i], &mut dis, &mut counts) {
                continue;
            }

            // Paid emission at the post-motion position, visible next tick.
            let o = &s.out;
            if sigmoid(o[out::EMIT_GATE] as f64) > em.gate_threshold {
                let raw: [f64; CHANNELS] = std::array::from_fn(|k| o[out::EMIT_DIR + k] as f64);
                let rn = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
                if rn >= em.direction_zero_threshold {
                    if let Some(dir) = isotropic_unit(&raw) {
                        let frac = rate_to_probability(em.maximum_fractional_rate_per_second * sigmoid(o[out::EMIT_RATE] as f64), dt);
                        let spend = orgs.energy[i] * frac;
                        if spend > 0.0 {
                            orgs.energy[i] -= spend;
                            let st = field.stencil(orgs.x[i], orgs.y[i]);
                            deposits.deposit(&st, em.transfer_efficiency * spend, &dir);
                            dis.emission += (1.0 - em.transfer_efficiency) * spend;
                            led.transferred += spend;
                            counts.emitting += 1;
                            if exhaust(&mut orgs.energy[i], &mut dead[i], &mut dis, &mut counts) {
                                continue;
                            }
                        }
                    }
                }
            }

            // Reproduction proposal (one Bernoulli attempt per eligible tick).
            if n >= orgs.cooldown_until[i] && orgs.energy[i] >= rp.minimum_parent_energy_before_attempt {
                let lambda = rp.maximum_attempt_rate_per_second * sigmoid(o[out::SPLIT] as f64);
                let p = rate_to_probability(lambda, dt);
                let u = self.rng.uniform(Lane::ReproductionAttempt, orgs.id[i], event, 0, 0);
                if u < p {
                    counts.reproduction_attempts += 1;
                    let beta = rp.minimum_child_fraction + (rp.maximum_child_fraction - rp.minimum_child_fraction) * sigmoid(o[out::SHARE] as f64);
                    let (ce, pe) = split_energy(orgs.energy[i], co.birth_energy, beta);
                    if ce >= rp.minimum_child_energy && pe >= rp.minimum_parent_energy_after_split {
                        match mutate(cfg, &self.rng, &g, orgs.id[i], event, 0) {
                            Ok(genome) => proposals.push(BirthProposal { parent: i, genome, child_energy: ce, parent_energy: pe }),
                            Err(_) => counts.rejected_births_mutation += 1,
                        }
                    } else {
                        counts.rejected_births_viability += 1;
                    }
                }
            }
        }

        // ── Phase 8: capacities, births, deaths ───────────────────────────
        let survivors: Vec<usize> = (0..orgs.len()).filter(|&i| !dead[i]).collect();
        let live_bytes: u64 = survivors.iter().map(|&i| orgs.genome[i].param_count() as u64 * 4).sum();
        let birth_bytes: u64 = proposals.iter().map(|p| p.genome.param_count() as u64 * 4).sum();
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
                let st = field.stencil(orgs.x[i], orgs.y[i]);
                deposits.deposit(&st, recycle * e, &orgs.genome[i].affinity);
                dis.death += (1.0 - recycle) * e;
                orgs.energy[i] = 0.0;
            }
        }
        for i in 0..orgs.len() {
            if !dead[i] {
                orgs.energy_change_rate[i] = (orgs.energy[i] - e_start[i]) / dt;
            }
        }
        let keep: Vec<bool> = dead.iter().map(|d| !d).collect();
        orgs.retain(&keep);
        for (id, lin, gen, x, y, e, g) in children {
            let deadline = event + cd_ticks;
            orgs.push(id, lin, gen, x, y, e, event, deadline, (id % k_eval) as u32, g);
        }

        // ── Phase 9: paid deposits and recycling enter the next field ─────
        field.add_assign(&deposits);
        timing.lifecycle_ms = t.ms();

        // ── Phase 10: tiny-reservoir clearing, validate invariants, publish ─
        let t_validate = Stopwatch::start();
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
        if let Some(i) = (0..orgs.len()).find(|&i| !(orgs.energy[i] > 0.0) || !orgs.energy[i].is_finite() || !orgs.x[i].is_finite() || !orgs.y[i].is_finite()) {
            return Err(TickError::NumericalFault { tick: event, detail: format!("organism {} published with energy {}", orgs.id[i], orgs.energy[i]) });
        }
        led.dissipated = dis;
        led.field_energy_after = field.total_energy();
        led.organism_energy_after = orgs.energy.iter().sum();
        led.energy_after = led.field_energy_after + led.organism_energy_after;
        led.residual = led.energy_after - led.energy_before - (led.injected_sources + led.injected_commands - led.removed - dis.total());

        self.field = field;
        self.orgs = orgs;
        self.sources = sources;
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
            for v in &o.genome[i].params {
                eat(v.to_bits() as u64);
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
}

impl Default for SenseOutput {
    fn default() -> Self {
        SenseOutput { logits: [0.0; OUTPUT_WIDTH], motion_drive: [0.0; 2], density0: 0.0 }
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
                forward(j.genome.hidden(), &j.genome.params, &input, &mut scratch, &mut o.logits);
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
        let mut scratch = vec![0.0f32; 2 * MAX_HIDDEN];
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            forward(j.genome.hidden(), &j.genome.params, j.input, &mut scratch, o);
        }
        Ok(())
    }
}

fn sense_one(cfg: &Config, pyr: &Pyramid, orgs: &Organisms, i: usize, n: u64, scratch: &mut [f32], infer: bool) -> Sense {
    let g = &orgs.genome[i];
    let raw = pyr.sample(orgs.x[i], orgs.y[i], g.scale());
    let obs = observe(&raw, &cfg.sensors);
    let density0 = pyr.density0(orgs.x[i], orgs.y[i]);
    let k = cfg.controller.evaluation_interval_ticks as u64;
    let due = !orgs.held_valid[i] || (n + 1) % k == orgs.phase[i] as u64;
    let mut out = orgs.held[i];
    if due {
        let age = (n - orgs.birth_tick[i]) as f64 * cfg.dt();
        let intr = intrinsic(&cfg.sensors, orgs.energy[i], age, orgs.vx[i], orgs.vy[i], orgs.energy_change_rate[i]);
        let mut input = [0.0f32; INPUT_WIDTH];
        input[..99].copy_from_slice(&obs.field_inputs);
        input[99..].copy_from_slice(&intr);
        let (dx, dy) = motion_drive(&g.motion, &obs.grad_x, &obs.grad_y);
        if infer {
            forward(g.hidden(), &g.params, &input, scratch, &mut out);
            return Sense { density0, out, evaluated: true, motion_drive: [dx, dy], input: None };
        }
        return Sense { density0, out, evaluated: true, motion_drive: [dx, dy], input: Some(Box::new(input)) };
    }
    let (dx, dy) = motion_drive(&g.motion, &obs.grad_x, &obs.grad_y);
    Sense { density0, out, evaluated: false, motion_drive: [dx, dy], input: None }
}

fn sense_all(cfg: &Config, pyr: &Pyramid, orgs: &Organisms, dead: &[bool], n: u64, workers: usize, infer: bool) -> Vec<Sense> {
    let len = orgs.len();
    let run = |range: std::ops::Range<usize>| -> Vec<Sense> {
        let mut scratch = vec![0.0f32; 2 * MAX_HIDDEN];
        range.map(|i| if dead[i] { Sense::blank() } else { sense_one(cfg, pyr, orgs, i, n, &mut scratch, infer) }).collect()
    };
    let workers = workers.max(1).min(len.max(1));
    if workers == 1 || len < 256 {
        return run(0..len);
    }
    let chunk = len.div_ceil(workers);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers).map(|wi| {
            let r = (wi * chunk).min(len)..((wi + 1) * chunk).min(len);
            s.spawn(move || run(r))
        }).collect();
        let mut all = Vec::with_capacity(len);
        for h in handles {
            all.extend(h.join().expect("sensing worker panicked"));
        }
        all
    })
}
