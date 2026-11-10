//! `frankenlife.config.v1` schema (design plan Appendix A) and validation (A.2).
//!
//! Unknown keys are refused. Model-affecting algorithm identifiers must name a
//! family this crate actually implements; an unrecognized identifier is an
//! error, never permission to substitute another algorithm. Infrastructure
//! sections that this slice does not implement yet are parsed, checked for
//! basic sanity and reported as "declared, not implemented".

use serde::{Deserialize, Serialize};

use crate::nn::{catalog_index, param_count, CATALOG, INPUT_WIDTH, OUTPUT_WIDTH};
use crate::CHANNELS;

/// How controller weights are stored (`controller.canonical_weight_dtype`). Arithmetic is always f32;
/// with `F16` every weight is rounded to the nearest 16-bit float whenever a genome is created or
/// mutated, so the stored genome, the checkpoint and the GPU copy all hold the same values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    F32,
    F16,
}

impl WeightDtype {
    pub fn bytes(self) -> u64 {
        match self {
            WeightDtype::F32 => 4,
            WeightDtype::F16 => 2,
        }
    }
}

pub const REFERENCE_PRESET: &str = include_str!("../../../presets/reference-field-ecology.toml");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema: String,
    pub name: String,
    pub model: String,
    pub seed: u64,
    pub world: WorldCfg,
    pub population: PopulationCfg,
    pub field_initialization: FieldInitCfg,
    pub sources: SourcesCfg,
    pub chemistry: ChemistryCfg,
    pub sensors: SensorsCfg,
    pub controller: ControllerCfg,
    pub architecture_mutation: ArchMutationCfg,
    pub mutation: MutationCfg,
    pub costs: CostsCfg,
    pub emission: EmissionCfg,
    pub reproduction: ReproductionCfg,
    pub death: DeathCfg,
    pub execution: ExecutionCfg,
    pub presentation: PresentationCfg,
    pub recording: RecordingCfg,
    pub persistence: PersistenceCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorldCfg {
    pub width: f64,
    pub height: f64,
    pub grid_width: u32,
    pub grid_height: u32,
    pub boundary: String,
    pub moment_channels: u32,
    pub field_dtype: String,
    pub tick_numerator: u32,
    pub tick_denominator: u32,
    pub transport_scheme: String,
    pub diffusion: f64,
    pub decay_per_second: f64,
    pub stability_policy: String,
    pub exchange_kernel: String,
    pub tiny_energy_policy: String,
    pub tiny_energy_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PopulationCfg {
    pub initial_count: u32,
    pub maximum_count: u32,
    pub initial_energy: f64,
    pub initial_velocity: String,
    pub initial_positions: String,
    pub capacity_policy: String,
    pub automatic_reseed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FieldInitCfg {
    pub energy_per_cell: f64,
    pub composition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SourcesCfg {
    pub count: u32,
    pub power_per_source: f64,
    pub sigma_world_units: f64,
    pub spatial_normalization: String,
    pub kernel: String,
    pub kernel_cutoff_sigma: f64,
    pub chemical_directions: String,
    pub chemical_direction_policy: String,
    pub path: String,
    pub path_speed: f64,
    pub path_direction_interval_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChemistryCfg {
    pub family: String,
    pub heritable_traits: Vec<String>,
    pub maximum_efficiency: f64,
    pub affinity_norm_limit: f64,
    pub affinity_gain: f64,
    pub discrimination_threshold: f64,
    pub uptake_capacity_per_second: f64,
    pub uptake_half_saturation_density: f64,
    pub toxicity_sensitivity: f64,
    pub chemistry_evaluation: String,
    pub motion_response_norm_limit: f64,
    pub maximum_acceleration: f64,
    pub drag_per_second: f64,
    pub maximum_speed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SensorsCfg {
    pub schema: String,
    pub normalization: String,
    pub energy_density_scale: f64,
    pub gradient_length_scale_world_units: f64,
    pub composition_energy_floor_density: f64,
    pub internal_energy_scale: f64,
    pub age_scale_seconds: f64,
    pub velocity_scale: f64,
    pub energy_change_scale_per_second: f64,
    pub initial_scale_world_units: f64,
    pub minimum_scale_world_units: f64,
    pub maximum_scale_world_units: f64,
    pub pyramid_filter: String,
    pub pyramid_scale_blend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControllerCfg {
    pub family: String,
    pub input_width: u32,
    pub hidden_widths: Vec<u32>,
    pub output_width: u32,
    pub hidden_activation: String,
    pub output_activation: String,
    pub canonical_weight_dtype: String,
    pub accumulation_dtype: String,
    pub evaluation_interval_ticks: u32,
    pub phase_policy: String,
    pub autograd: bool,
    pub initializer: String,
    pub maximum_absolute_parameter: f64,
}

impl ControllerCfg {
    /// Storage type of the weights (validated to be `f32` or `f16`).
    pub fn weight_dtype(&self) -> WeightDtype {
        if self.canonical_weight_dtype == "f16" {
            WeightDtype::F16
        } else {
            WeightDtype::F32
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArchMutationCfg {
    pub enabled: bool,
    pub catalog: String,
    pub maximum_parameters: u64,
    pub probability_per_birth: f64,
    pub inheritance_map: String,
    pub depth_change_map: String,
    pub target_selection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MutationCfg {
    pub rng: String,
    pub weight_probability: f64,
    pub weight_standard_deviation: f64,
    pub replacement_probability: f64,
    pub chemistry_standard_deviation: f64,
    pub sensing_log_scale_standard_deviation: f64,
    pub reject_effective_noop: bool,
    pub maximum_candidate_proposals_per_birth: u32,
    pub exhausted_proposals_policy: String,
    pub meta_evolution: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CostsCfg {
    pub basal_energy_per_second: f64,
    pub sensor_energy_per_second_at_unit_scale: f64,
    pub sensor_unit_scale_world_units: f64,
    pub sensor_scale_cost_exponent: f64,
    pub neural_energy_per_parameter_evaluation: f64,
    pub motion_energy_per_speed_squared_second: f64,
    pub birth_energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmissionCfg {
    pub gate_threshold: f64,
    pub maximum_fractional_rate_per_second: f64,
    pub transfer_efficiency: f64,
    pub direction_zero_threshold: f64,
    pub zero_direction_policy: String,
    pub position: String,
    pub visibility: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReproductionCfg {
    pub maximum_attempt_rate_per_second: f64,
    pub minimum_parent_energy_before_attempt: f64,
    pub minimum_child_energy: f64,
    pub minimum_parent_energy_after_split: f64,
    pub minimum_child_fraction: f64,
    pub maximum_child_fraction: f64,
    pub cooldown_seconds: f64,
    pub child_position_offset_radius: f64,
    pub newborn_activation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeathCfg {
    pub residual_energy_recycle_fraction: f64,
    pub corpse_direction: String,
    pub numerical_fault_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCfg {
    pub backend: String,
    pub determinism: String,
    pub math_mode: String,
    pub cpu_workers: u32,
    pub command_queue_capacity: u32,
    pub control_queue_capacity: u32,
    pub command_durability: String,
    pub render_queue_capacity: u32,
    pub maximum_logical_genome_bytes: u64,
    pub maximum_allocated_genome_bytes: u64,
    pub maximum_total_bytes: u64,
    pub checkpoint_retained_bytes: u64,
    pub observer_pressure_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PresentationCfg {
    pub engine: String,
    pub width: u32,
    pub height: u32,
    pub frames_per_second: u32,
    pub color_mapping: String,
    pub background: String,
    pub point_radius_pixels: f64,
    pub bloom_enabled: bool,
    pub bloom_strength: f64,
    pub trails_enabled: bool,
    pub interpolation: String,
    pub show_clock_telemetry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecordingCfg {
    pub enabled: bool,
    pub clock: String,
    pub include_provenance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PersistenceCfg {
    pub catalog: String,
    pub checkpoint_mode: String,
    pub checkpoint_interval_simulation_seconds: f64,
    pub retain_recent_checkpoints: u32,
    pub preserve_pinned_discoveries: bool,
    pub metrics_interval_ticks: u32,
}

/// Capacity policy at a tick whose births would exceed a fixed logical quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CapacityPolicy {
    /// Research mode: discard the candidate tick and report the shortfall.
    PauseBeforeCommit,
    /// Exhibition mode: accept births in stable parent-ID order until full.
    RejectBirthsStableOrder,
}

/// Result of successful validation.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub dt_seconds: f64,
    pub default_parameter_count: usize,
    pub largest_catalog_parameter_count: usize,
    pub worst_case_live_genome_bytes: u64,
    pub logical_genome_quota_bytes: u64,
    pub transport_subcycles: u32,
    pub sensor_pyramid_levels: usize,
    pub source_power_total_per_second: f64,
    pub mandatory_cost_per_default_organism_per_second: f64,
    pub notes: Vec<String>,
}

impl Config {
    pub fn reference() -> Config {
        Config::from_toml(REFERENCE_PRESET).expect("embedded reference preset parses")
    }

    pub fn from_toml(text: &str) -> Result<Config, String> {
        toml::from_str(text).map_err(|e| format!("config parse error: {e}"))
    }

    pub fn dt(&self) -> f64 {
        self.world.tick_numerator as f64 / self.world.tick_denominator as f64
    }

    pub fn capacity_policy(&self) -> CapacityPolicy {
        match self.population.capacity_policy.as_str() {
            "reject-births-stable-order" => CapacityPolicy::RejectBirthsStableOrder,
            _ => CapacityPolicy::PauseBeforeCommit,
        }
    }

    pub fn cell_size(&self) -> (f64, f64) {
        (
            self.world.width / self.world.grid_width as f64,
            self.world.height / self.world.grid_height as f64,
        )
    }

    /// Number of extra pyramid levels above level 0 required by the maximum sensing scale.
    pub fn pyramid_levels(&self) -> usize {
        let (hx, hy) = self.cell_size();
        let s0 = hx.max(hy);
        let ratio = self.sensors.maximum_scale_world_units / s0;
        let mut l = 0usize;
        while (1u64 << l) as f64 + 1e-12 < ratio {
            l += 1;
        }
        l.max(1)
    }

    /// Validate ranges, identifiers and simultaneous constraints (design plan A.2).
    pub fn validate(&self) -> Result<ValidationReport, Vec<String>> {
        let mut err: Vec<String> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut need = |ok: bool, msg: String| {
            if !ok {
                err.push(msg);
            }
        };
        let id = |field: &str, got: &str, allowed: &[&str]| -> Option<String> {
            if allowed.contains(&got) {
                None
            } else {
                Some(format!("{field} = {got:?} is not an implemented family (allowed: {allowed:?})"))
            }
        };
        let mut ids: Vec<Option<String>> = vec![
            id("schema", &self.schema, &["frankenlife.config.v1"]),
            id("model", &self.model, &["field-ecology-v1"]),
            id("world.boundary", &self.world.boundary, &["periodic"]),
            id("world.field_dtype", &self.world.field_dtype, &["f32"]),
            id("world.transport_scheme", &self.world.transport_scheme, &["positive-five-point-v1"]),
            id("world.stability_policy", &self.world.stability_policy, &["subcycle"]),
            id("world.exchange_kernel", &self.world.exchange_kernel, &["grid-cic-v1"]),
            id("world.tiny_energy_policy", &self.world.tiny_energy_policy, &["dissipate-and-record"]),
            id("population.initial_velocity", &self.population.initial_velocity, &["zero"]),
            id("population.initial_positions", &self.population.initial_positions, &["uniform"]),
            id(
                "population.capacity_policy",
                &self.population.capacity_policy,
                &["pause-before-commit", "reject-births-stable-order"],
            ),
            id("field_initialization.composition", &self.field_initialization.composition, &["zero"]),
            id("sources.spatial_normalization", &self.sources.spatial_normalization, &["fixed-total-power"]),
            id("sources.kernel", &self.sources.kernel, &["normalized-truncated-minimum-image-gaussian-v1"]),
            id("sources.chemical_directions", &self.sources.chemical_directions, &["seeded-unit-vectors"]),
            id("sources.chemical_direction_policy", &self.sources.chemical_direction_policy, &["fixed-per-source"]),
            id("sources.path", &self.sources.path, &["seeded-smooth-torus-v1"]),
            id("chemistry.family", &self.chemistry.family, &["bounded-affinity-v1"]),
            id("chemistry.chemistry_evaluation", &self.chemistry.chemistry_evaluation, &["per-granted-cell-packet"]),
            id("sensors.schema", &self.sensors.schema, &["energy-moment-gradient-104-v1"]),
            id("sensors.normalization", &self.sensors.normalization, &["fixed-local-scales-v1"]),
            id("sensors.pyramid_filter", &self.sensors.pyramid_filter, &["positive-box-v1"]),
            id("sensors.pyramid_scale_blend", &self.sensors.pyramid_scale_blend, &["log-linear-v1"]),
            id("controller.family", &self.controller.family, &["independent-dense-mlp-v1"]),
            id("controller.hidden_activation", &self.controller.hidden_activation, &["tanh"]),
            id("controller.output_activation", &self.controller.output_activation, &["linear"]),
            id("controller.canonical_weight_dtype", &self.controller.canonical_weight_dtype, &["f32", "f16"]),
            id("controller.accumulation_dtype", &self.controller.accumulation_dtype, &["f32"]),
            id("controller.phase_policy", &self.controller.phase_policy, &["stable-identity"]),
            id("controller.initializer", &self.controller.initializer, &["fan-in-scaled-symmetric-v1"]),
            id("architecture_mutation.catalog", &self.architecture_mutation.catalog, &["dense-104-36-catalog-v1"]),
            id(
                "architecture_mutation.inheritance_map",
                &self.architecture_mutation.inheritance_map,
                &["overlap-plus-zero-output-v1"],
            ),
            id(
                "architecture_mutation.depth_change_map",
                &self.architecture_mutation.depth_change_map,
                &["prefix-and-identity-bridge-v1"],
            ),
            id(
                "architecture_mutation.target_selection",
                &self.architecture_mutation.target_selection,
                &["uniform-other-catalog-member"],
            ),
            id("mutation.rng", &self.mutation.rng, &["fl-philox-semantic-v1"]),
            id(
                "mutation.exhausted_proposals_policy",
                &self.mutation.exhausted_proposals_policy,
                &["reject-birth-no-charge"],
            ),
            id("emission.zero_direction_policy", &self.emission.zero_direction_policy, &["no-emission-no-charge"]),
            id("emission.position", &self.emission.position, &["post-motion"]),
            id("emission.visibility", &self.emission.visibility, &["next-tick"]),
            id("reproduction.newborn_activation", &self.reproduction.newborn_activation, &["next-tick"]),
            id("death.corpse_direction", &self.death.corpse_direction, &["genetic-affinity"]),
            id("death.numerical_fault_policy", &self.death.numerical_fault_policy, &["stop-and-preserve"]),
            id("execution.backend", &self.execution.backend, &["cpu-reference"]),
            id("execution.math_mode", &self.execution.math_mode, &["specified"]),
        ];
        for h in &self.chemistry.heritable_traits {
            ids.push(id(
                "chemistry.heritable_traits[]",
                h,
                &["affinity_direction", "motion_response", "sensing_scale"],
            ));
        }
        for e in ids.into_iter().flatten() {
            need(false, e);
        }

        // Every float must be finite.
        let floats = self.all_floats();
        for (name, v) in &floats {
            need(v.is_finite(), format!("{name} must be finite (got {v})"));
        }

        let w = &self.world;
        need(w.width > 0.0 && w.height > 0.0, "world size must be positive".into());
        need(w.grid_width >= 2 && w.grid_height >= 2, "grid must be at least 2x2".into());
        need(w.moment_channels as usize == CHANNELS, format!("world.moment_channels must be {CHANNELS} for field-ecology-v1"));
        need(w.tick_numerator > 0 && w.tick_denominator > 0, "tick rational must be positive".into());
        need(w.diffusion >= 0.0, "world.diffusion must be >= 0".into());
        need(w.decay_per_second >= 0.0, "world.decay_per_second must be >= 0".into());
        need(w.tiny_energy_threshold >= 0.0, "world.tiny_energy_threshold must be >= 0".into());

        let dt = self.dt();
        let (hx, hy) = self.cell_size();
        let rsum = w.diffusion * dt / (hx * hx) + w.diffusion * dt / (hy * hy);
        let subcycles = ((rsum / 0.5).ceil() as u32).max(1);

        let p = &self.population;
        need(p.initial_count >= 1, "population.initial_count must be >= 1".into());
        need(p.maximum_count >= p.initial_count, "population.maximum_count must be >= initial_count".into());
        need(p.initial_energy > 0.0, "population.initial_energy must be > 0".into());
        need(!p.automatic_reseed, "automatic reseeding is not part of the baseline".into());
        need(self.field_initialization.energy_per_cell >= 0.0, "field energy_per_cell must be >= 0".into());

        let s = &self.sources;
        need(s.power_per_source >= 0.0, "sources.power_per_source must be >= 0".into());
        need(s.sigma_world_units > 0.0, "sources.sigma_world_units must be > 0".into());
        need(s.kernel_cutoff_sigma > 0.0, "sources.kernel_cutoff_sigma must be > 0".into());
        need(s.path_speed >= 0.0, "sources.path_speed must be >= 0".into());
        let seg_ticks = s.path_direction_interval_seconds / dt;
        need(
            s.path_direction_interval_seconds > 0.0 && (seg_ticks - seg_ticks.round()).abs() < 1e-9,
            "sources.path_direction_interval_seconds / dt must be a positive integer".into(),
        );

        let c = &self.chemistry;
        need((0.0..=1.0).contains(&c.maximum_efficiency), "chemistry.maximum_efficiency must be in [0,1]".into());
        need(c.affinity_norm_limit > 0.0 && c.affinity_norm_limit <= 1.0, "chemistry.affinity_norm_limit must be in (0,1]".into());
        need(c.affinity_gain > 0.0, "chemistry.affinity_gain must be > 0".into());
        need(c.discrimination_threshold > 0.0, "baseline chemistry.discrimination_threshold must be > 0".into());
        need(c.uptake_capacity_per_second >= 0.0, "chemistry.uptake_capacity_per_second must be >= 0".into());
        need(c.uptake_half_saturation_density > 0.0, "chemistry.uptake_half_saturation_density must be > 0".into());
        need(c.toxicity_sensitivity > 0.0, "baseline chemistry.toxicity_sensitivity must be > 0".into());
        need(c.motion_response_norm_limit >= 0.0, "chemistry.motion_response_norm_limit must be >= 0".into());
        need(c.maximum_acceleration >= 0.0, "chemistry.maximum_acceleration must be >= 0".into());
        need(c.drag_per_second >= 0.0, "chemistry.drag_per_second must be >= 0".into());
        need(c.maximum_speed >= 0.0, "chemistry.maximum_speed must be >= 0".into());
        need(
            c.maximum_speed * dt <= 0.5 * hx.min(hy) + 1e-12,
            format!("maximum_speed * dt = {} exceeds 0.5 * min cell width = {}", c.maximum_speed * dt, 0.5 * hx.min(hy)),
        );

        let se = &self.sensors;
        for (n, v) in [
            ("energy_density_scale", se.energy_density_scale),
            ("gradient_length_scale_world_units", se.gradient_length_scale_world_units),
            ("composition_energy_floor_density", se.composition_energy_floor_density),
            ("internal_energy_scale", se.internal_energy_scale),
            ("age_scale_seconds", se.age_scale_seconds),
            ("velocity_scale", se.velocity_scale),
            ("energy_change_scale_per_second", se.energy_change_scale_per_second),
        ] {
            need(v > 0.0, format!("sensors.{n} must be > 0"));
        }
        let s0 = hx.max(hy);
        need(
            se.minimum_scale_world_units >= s0 - 1e-12,
            format!("sensors.minimum_scale_world_units must be >= base level spacing {s0}"),
        );
        need(
            se.minimum_scale_world_units <= se.initial_scale_world_units
                && se.initial_scale_world_units <= se.maximum_scale_world_units,
            "sensors scales must satisfy minimum <= initial <= maximum".into(),
        );
        let levels = self.pyramid_levels();
        let div = 1u32 << levels;
        need(
            w.grid_width % div == 0 && w.grid_height % div == 0,
            format!("grid must be divisible by 2^{levels} = {div} for the admitted sensing scale"),
        );

        let ct = &self.controller;
        need(ct.input_width as usize == INPUT_WIDTH, format!("controller.input_width must be {INPUT_WIDTH}"));
        need(ct.output_width as usize == OUTPUT_WIDTH, format!("controller.output_width must be {OUTPUT_WIDTH}"));
        let hidden: Vec<usize> = ct.hidden_widths.iter().map(|&h| h as usize).collect();
        need(catalog_index(&hidden).is_some(), format!("controller.hidden_widths {hidden:?} not in catalog {CATALOG:?}"));
        need(ct.evaluation_interval_ticks >= 1, "controller.evaluation_interval_ticks must be >= 1".into());
        need(!ct.autograd, "autograd is not part of the baseline".into());
        need(ct.maximum_absolute_parameter > 0.0, "controller.maximum_absolute_parameter must be > 0".into());

        let am = &self.architecture_mutation;
        need((0.0..=1.0).contains(&am.probability_per_birth), "architecture probability must be in [0,1]".into());
        if catalog_index(&hidden).is_some() {
            need(
                param_count(&hidden) as u64 <= am.maximum_parameters,
                "default topology exceeds architecture_mutation.maximum_parameters".into(),
            );
        }

        let m = &self.mutation;
        for (n, v) in [
            ("weight_probability", m.weight_probability),
            ("replacement_probability", m.replacement_probability),
        ] {
            need((0.0..=1.0).contains(&v), format!("mutation.{n} must be in [0,1]"));
        }
        for (n, v) in [
            ("weight_standard_deviation", m.weight_standard_deviation),
            ("chemistry_standard_deviation", m.chemistry_standard_deviation),
            ("sensing_log_scale_standard_deviation", m.sensing_log_scale_standard_deviation),
        ] {
            need(v >= 0.0, format!("mutation.{n} must be >= 0"));
        }
        need(m.maximum_candidate_proposals_per_birth >= 1, "at least one mutation proposal per birth".into());
        need(!m.meta_evolution, "meta_evolution is not implemented in field-ecology-v1".into());
        if m.reject_effective_noop {
            let can_change = (m.weight_probability > 0.0 && m.weight_standard_deviation > 0.0)
                || m.replacement_probability > 0.0
                || m.chemistry_standard_deviation > 0.0
                || m.sensing_log_scale_standard_deviation > 0.0
                || (am.enabled && am.probability_per_birth > 0.0);
            need(can_change, "strict mutation requires at least one operator able to change a genome".into());
        }

        let co = &self.costs;
        for (n, v) in [
            ("basal_energy_per_second", co.basal_energy_per_second),
            ("sensor_energy_per_second_at_unit_scale", co.sensor_energy_per_second_at_unit_scale),
            ("neural_energy_per_parameter_evaluation", co.neural_energy_per_parameter_evaluation),
            ("motion_energy_per_speed_squared_second", co.motion_energy_per_speed_squared_second),
            ("birth_energy", co.birth_energy),
        ] {
            need(v >= 0.0, format!("costs.{n} must be >= 0"));
        }
        need(co.sensor_unit_scale_world_units > 0.0, "costs.sensor_unit_scale_world_units must be > 0".into());

        let e = &self.emission;
        need((0.0..=1.0).contains(&e.gate_threshold), "emission.gate_threshold must be in [0,1]".into());
        need((0.0..=1.0).contains(&e.transfer_efficiency), "emission.transfer_efficiency must be in [0,1]".into());
        need(e.maximum_fractional_rate_per_second >= 0.0, "emission rate must be >= 0".into());
        need(e.direction_zero_threshold > 0.0, "emission.direction_zero_threshold must be > 0".into());

        let r = &self.reproduction;
        need(r.maximum_attempt_rate_per_second >= 0.0, "reproduction attempt rate must be >= 0".into());
        need(
            0.0 < r.minimum_child_fraction && r.minimum_child_fraction <= r.maximum_child_fraction && r.maximum_child_fraction < 1.0,
            "reproduction child fractions must satisfy 0 < min <= max < 1".into(),
        );
        need(r.minimum_child_energy > 0.0 && r.minimum_parent_energy_after_split > 0.0, "viability minima must be > 0".into());
        // Some admitted split must be viable at the attempt threshold.
        let avail = r.minimum_parent_energy_before_attempt - co.birth_energy;
        need(
            avail * r.maximum_child_fraction >= r.minimum_child_energy
                || avail * (1.0 - r.minimum_child_fraction) >= r.minimum_parent_energy_after_split,
            "inconsistent birth viability: no split can ever succeed".into(),
        );
        need(r.cooldown_seconds >= 0.0 && r.child_position_offset_radius >= 0.0, "cooldown/offset must be >= 0".into());
        need(
            (0.0..=1.0).contains(&self.death.residual_energy_recycle_fraction),
            "death.residual_energy_recycle_fraction must be in [0,1]".into(),
        );
        need(self.execution.cpu_workers >= 1, "execution.cpu_workers must be >= 1".into());
        need(
            self.execution.maximum_logical_genome_bytes <= self.execution.maximum_allocated_genome_bytes,
            "logical genome quota exceeds allocated genome budget".into(),
        );
        need(self.persistence.metrics_interval_ticks >= 1, "persistence.metrics_interval_ticks must be >= 1".into());
        need(
            self.initial_genome_bytes() <= self.execution.maximum_logical_genome_bytes,
            "initial population genomes exceed the logical genome quota".into(),
        );

        if self.presentation.engine == "franken-manim" {
            notes.push("presentation.engine = franken-manim: declared, not linked; the headless CPU rasterizer in the `frankenlife` CLI renders committed epochs instead".into());
        }
        notes.push(format!("persistence.catalog = {}: declared, not implemented in this slice (metrics go to JSONL)", self.persistence.catalog));
        notes.push(format!("execution.command_durability = {}: declared, not implemented (in-memory command queue)", self.execution.command_durability));
        notes.push(format!("execution.determinism = {}: CPU reference is bitwise deterministic for a fixed build and worker count independent", self.execution.determinism));
        if !err.is_empty() {
            return Err(err);
        }

        let default_p = param_count(&hidden);
        let largest = CATALOG.iter().map(|h| param_count(h)).filter(|&n| n as u64 <= am.maximum_parameters).max().unwrap_or(default_p);
        let pc = self.mandatory_cost_per_second(default_p, se.initial_scale_world_units);
        Ok(ValidationReport {
            dt_seconds: dt,
            default_parameter_count: default_p,
            largest_catalog_parameter_count: largest,
            worst_case_live_genome_bytes: p.maximum_count as u64 * largest as u64 * self.controller.weight_dtype().bytes(),
            logical_genome_quota_bytes: self.execution.maximum_logical_genome_bytes,
            transport_subcycles: subcycles,
            sensor_pyramid_levels: levels + 1,
            source_power_total_per_second: s.count as f64 * s.power_per_source,
            mandatory_cost_per_default_organism_per_second: pc,
            notes,
        })
    }

    /// Mandatory per-second cost of one organism (A.1), assuming evaluation at the configured cadence.
    pub fn mandatory_cost_per_second(&self, params: usize, scale: f64) -> f64 {
        let co = &self.costs;
        co.basal_energy_per_second
            + co.sensor_energy_per_second_at_unit_scale * (scale / co.sensor_unit_scale_world_units).powf(co.sensor_scale_cost_exponent)
            + co.neural_energy_per_parameter_evaluation * params as f64 / (self.dt() * self.controller.evaluation_interval_ticks as f64)
    }

    fn initial_genome_bytes(&self) -> u64 {
        let hidden: Vec<usize> = self.controller.hidden_widths.iter().map(|&h| h as usize).collect();
        if catalog_index(&hidden).is_none() {
            return 0;
        }
        self.population.initial_count as u64 * param_count(&hidden) as u64 * self.controller.weight_dtype().bytes()
    }

    fn all_floats(&self) -> Vec<(String, f64)> {
        // Round-trip through TOML values to enumerate every float field generically.
        let v = toml::Value::try_from(self).expect("config serializes");
        let mut out = Vec::new();
        fn walk(prefix: &str, v: &toml::Value, out: &mut Vec<(String, f64)>) {
            match v {
                toml::Value::Float(f) => out.push((prefix.to_string(), *f)),
                toml::Value::Table(t) => {
                    for (k, v) in t {
                        let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                        walk(&p, v, out);
                    }
                }
                _ => {}
            }
        }
        walk("", &v, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_preset_validates() {
        let c = Config::reference();
        let r = c.validate().expect("reference validates");
        assert_eq!(r.default_parameter_count, 9_988);
        assert_eq!(r.largest_catalog_parameter_count, 29_316);
        assert_eq!(r.sensor_pyramid_levels, 5);
        // A.2: 16,384 organisms at 29,316 F32 parameters = 1,832.25 MiB > 1 GiB quota.
        assert_eq!(r.worst_case_live_genome_bytes as f64 / (1024.0 * 1024.0), 1832.25);
    }

    #[test]
    fn unknown_keys_and_families_are_refused() {
        let bad = REFERENCE_PRESET.replace("seed = 42", "seed = 42\nsurprise = 1");
        assert!(Config::from_toml(&bad).is_err());
        let mut c = Config::reference();
        c.world.transport_scheme = "magic".into();
        assert!(c.validate().is_err());
        let mut c = Config::reference();
        c.controller.hidden_widths = vec![50];
        assert!(c.validate().is_err());
        let mut c = Config::reference();
        c.chemistry.maximum_speed = 100.0;
        assert!(c.validate().is_err());
    }
}
