//! Heritable genomes, random initialization and bounded strict mutation (plan §7.1–7.3, A.3).

use std::borrow::Cow;
use std::sync::Arc;

use serde::Serialize;

use crate::config::{Config, WeightDtype};
use crate::half::{f16_to_f32, f32_to_f16, round_f16};
use crate::nn::{catalog_index, fan_in_table, is_recurrent, map_topology, param_count, CATALOG};
use crate::rng::{Lane, SemanticRng};
use crate::sources::isotropic_unit;
use crate::CHANNELS;

pub const MOTION_WIDTH: usize = CHANNELS + 1;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MutationRecord {
    pub candidate_index: u32,
    pub topology_changed: bool,
    pub parameters_perturbed: u32,
    pub parameters_replaced: u32,
    pub rejected_candidates: u32,
}

/// Controller weights as stored in a genome: 32-bit floats, or 16-bit floats (half the memory,
/// on the CPU, in checkpoints and in GPU slabs). Both hold plain parameter values in the
/// `nn` layout; the network arithmetic is f32 either way.
#[derive(Debug, Clone, PartialEq)]
pub enum Weights {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

impl Weights {
    /// Store `v` in the requested type; `F16` rounds every value to the nearest 16-bit float.
    pub fn from_f32(dtype: WeightDtype, v: Vec<f32>) -> Weights {
        match dtype {
            WeightDtype::F32 => Weights::F32(v),
            WeightDtype::F16 => Weights::F16(v.iter().map(|&x| f32_to_f16(x)).collect()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Weights::F32(v) => v.len(),
            Weights::F16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> WeightDtype {
        match self {
            Weights::F32(_) => WeightDtype::F32,
            Weights::F16(_) => WeightDtype::F16,
        }
    }

    /// The values as f32 (borrowed for `F32`, widened exactly for `F16`).
    pub fn to_f32(&self) -> Cow<'_, [f32]> {
        match self {
            Weights::F32(v) => Cow::Borrowed(v),
            Weights::F16(v) => Cow::Owned(v.iter().map(|&h| f16_to_f32(h)).collect()),
        }
    }

    /// The raw 16-bit patterns, rounding to f16 first when the weights are stored as f32.
    pub fn to_f16_bits(&self) -> Cow<'_, [u16]> {
        match self {
            Weights::F16(v) => Cow::Borrowed(v),
            Weights::F32(v) => Cow::Owned(v.iter().map(|&x| f32_to_f16(x)).collect()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Genome {
    pub id: u64,
    pub parent: Option<u64>,
    /// Index into `nn::CATALOG`.
    pub topology: u8,
    pub params: Weights,
    /// Unit chemical affinity direction `a_i`.
    pub affinity: [f32; CHANNELS],
    /// Energy/composition gradient sensitivities, Euclidean norm within the fixed budget.
    pub motion: [f32; MOTION_WIDTH],
    /// Natural log of the sensing scale in world units.
    pub log_scale: f32,
    pub mutation: Option<MutationRecord>,
}

impl Genome {
    pub fn hidden(&self) -> &'static [usize] {
        CATALOG[self.topology as usize]
    }

    pub fn scale(&self) -> f64 {
        (self.log_scale as f64).exp()
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

fn uniform_symmetric(u: f64, fan_in: usize) -> f32 {
    let a = (3.0 / fan_in as f64).sqrt();
    (-a + 2.0 * a * u) as f32
}

/// Founder genome for stable organism `org_id` (`fan-in-scaled-symmetric-v1`, isotropic traits).
pub fn founder(cfg: &Config, rng: &SemanticRng, org_id: u64, genome_id: u64) -> Genome {
    let hidden: Vec<usize> = cfg.controller.hidden_widths.iter().map(|&h| h as usize).collect();
    let topology = catalog_index(&hidden).expect("validated topology");
    let n = param_count(&hidden);
    let fans = fan_in_table(&hidden);
    let mut params = vec![0.0f32; n];
    let dtype = cfg.controller.weight_dtype();
    let mut layer = 0;
    let offsets = crate::nn::layer_offsets(&hidden);
    let dims = crate::nn::layer_dims(&hidden);
    for p in 0..n {
        while p >= fans[layer].0 {
            layer += 1;
        }
        let is_bias = p >= offsets[layer] + dims[layer].0 * dims[layer].1;
        if !is_bias {
            let u = crate::rng::unit_half_open(rng.words(Lane::InitWeights, org_id, p as u64, 0, 0)[0]);
            params[p] = uniform_symmetric(u, fans[layer].1);
        }
    }
    let params = Weights::from_f32(dtype, params);
    let mut z = [0.0f64; CHANNELS];
    let mut attempt = 0;
    let affinity = loop {
        rng.normals(Lane::InitAffinity, org_id, attempt, 0, &mut z);
        if let Some(a) = isotropic_unit(&z) {
            break a;
        }
        attempt += 1;
    };
    let mut z = [0.0f64; MOTION_WIDTH];
    let mut attempt = 0;
    let motion = loop {
        rng.normals(Lane::InitMotion, org_id, attempt, 0, &mut z);
        if let Some(a) = isotropic_unit(&z) {
            let lim = cfg.chemistry.motion_response_norm_limit;
            let mut m = [0.0f32; MOTION_WIDTH];
            for k in 0..MOTION_WIDTH {
                m[k] = (a[k] as f64 * lim) as f32;
            }
            break project_ball(m, lim);
        }
        attempt += 1;
    };
    Genome {
        id: genome_id,
        parent: None,
        topology,
        params,
        affinity,
        motion,
        log_scale: cfg.sensors.initial_scale_world_units.ln() as f32,
        mutation: None,
    }
}

fn project_ball<const N: usize>(mut v: [f32; N], lim: f64) -> [f32; N] {
    let n = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if n > lim {
        let f = lim / n * (1.0 - 1e-7);
        for x in v.iter_mut() {
            *x = (*x as f64 * f) as f32;
        }
    }
    v
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MutationReject {
    /// Every candidate was invalid or an effective no-op after canonical rounding.
    ProposalsExhausted,
}

/// Strict bounded mutation. Draws are addressed by
/// `(parent_org_id, event_tick, candidate_index, lane, draw_index)`.
pub fn mutate(cfg: &Config, rng: &SemanticRng, parent: &Genome, parent_org: u64, event_tick: u64, child_genome_id: u64) -> Result<Arc<Genome>, MutationReject> {
    let mc = &cfg.mutation;
    let am = &cfg.architecture_mutation;
    let max_abs = cfg.controller.maximum_absolute_parameter;
    let dtype = cfg.controller.weight_dtype();
    let parent_params = parent.params.to_f32();
    let mut rejected = 0u32;
    for cand in 0..mc.maximum_candidate_proposals_per_birth as u64 {
        let key = |lane: Lane, draw: u64| rng.words(lane, parent_org, event_tick, cand, draw);
        // 1. Topology.
        let mut topology = parent.topology;
        let mut params: Vec<f32>;
        let mut topology_changed = false;
        let u_topo = crate::rng::unit_open(key(Lane::MutationTopology, 0)[0]);
        if am.enabled && u_topo < am.probability_per_birth {
            let others: Vec<u8> = (0..CATALOG.len() as u8)
                .filter(|&t| t != parent.topology && is_recurrent(CATALOG[t as usize]) == is_recurrent(parent.hidden()) && param_count(CATALOG[t as usize]) as u64 <= am.maximum_parameters)
                .collect();
            if !others.is_empty() {
                let pick = crate::rng::unit_half_open(key(Lane::MutationTopology, 1)[0]);
                topology = others[((pick * others.len() as f64) as usize).min(others.len() - 1)];
                topology_changed = true;
            }
        }
        if topology_changed {
            let new_h = CATALOG[topology as usize];
            params = map_topology(parent.hidden(), &parent_params, new_h, |_, fan_in, k| {
                let u = crate::rng::unit_half_open(key(Lane::MutationNewNeuron, k as u64)[0]);
                uniform_symmetric(u, fan_in)
            });
        } else {
            params = parent_params.to_vec();
        }
        // 2. Controller parameters (replacement takes priority over perturbation).
        let fans = fan_in_table(CATALOG[topology as usize]);
        let mut layer = 0;
        let (mut n_pert, mut n_rep) = (0u32, 0u32);
        let mut valid = true;
        let mut changed = topology_changed;
        for p in 0..params.len() {
            while p >= fans[layer].0 {
                layer += 1;
            }
            let w = key(Lane::MutationParamSelect, p as u64);
            let u_rep = crate::rng::unit_open(w[0]);
            let u_pert = crate::rng::unit_open(w[1]);
            let old = params[p];
            let new = if u_rep < mc.replacement_probability {
                n_rep += 1;
                let u = crate::rng::unit_half_open(key(Lane::MutationParamValue, p as u64)[0]);
                uniform_symmetric(u, fans[layer].1)
            } else if u_pert < mc.weight_probability {
                n_pert += 1;
                let z = rng.normal2(Lane::MutationParamValue, parent_org, event_tick, cand, p as u64)[0];
                (old as f64 + z * mc.weight_standard_deviation) as f32
            } else {
                continue;
            };
            // Stored weights are f16: judge validity and change on the value that will be stored.
            let new = if dtype == WeightDtype::F16 { round_f16(new) } else { new };
            if !new.is_finite() || (new as f64).abs() > max_abs {
                valid = false;
                break;
            }
            if new.to_bits() != old.to_bits() {
                changed = true;
            }
            params[p] = new;
        }
        if !valid {
            rejected += 1;
            continue;
        }
        // 3. Traits: affinity (renormalize), motion (radial projection), sensing scale (reflect).
        let mut z = [0.0f64; CHANNELS];
        rng.normals(Lane::MutationAffinity, parent_org, event_tick, cand, &mut z);
        let mut a = [0.0f64; CHANNELS];
        for k in 0..CHANNELS {
            a[k] = parent.affinity[k] as f64 + mc.chemistry_standard_deviation * z[k];
        }
        let Some(affinity) = isotropic_unit(&a) else {
            rejected += 1;
            continue;
        };
        let mut zm = [0.0f64; MOTION_WIDTH];
        rng.normals(Lane::MutationMotion, parent_org, event_tick, cand, &mut zm);
        let mut motion = [0.0f32; MOTION_WIDTH];
        for k in 0..MOTION_WIDTH {
            motion[k] = (parent.motion[k] as f64 + mc.chemistry_standard_deviation * zm[k]) as f32;
        }
        let motion = project_ball(motion, cfg.chemistry.motion_response_norm_limit);
        let (lo, hi) = (cfg.sensors.minimum_scale_world_units.ln(), cfg.sensors.maximum_scale_world_units.ln());
        let zs = rng.normal2(Lane::MutationSensing, parent_org, event_tick, cand, 0)[0];
        let mut ls = parent.log_scale as f64 + mc.sensing_log_scale_standard_deviation * zs;
        if hi > lo {
            let span = hi - lo;
            let mut t = (ls - lo).rem_euclid(2.0 * span);
            if t > span {
                t = 2.0 * span - t;
            }
            ls = lo + t;
        } else {
            ls = lo;
        }
        let log_scale = (ls as f32).clamp(lo as f32, hi as f32);
        if !motion.iter().all(|v| v.is_finite()) || !log_scale.is_finite() {
            rejected += 1;
            continue;
        }
        changed |= affinity != parent.affinity || motion != parent.motion || log_scale.to_bits() != parent.log_scale.to_bits();
        if mc.reject_effective_noop && !changed {
            rejected += 1;
            continue;
        }
        return Ok(Arc::new(Genome {
            id: child_genome_id,
            parent: Some(parent.id),
            topology,
            params: Weights::from_f32(dtype, params),
            affinity,
            motion,
            log_scale,
            mutation: Some(MutationRecord {
                candidate_index: cand as u32,
                topology_changed,
                parameters_perturbed: n_pert,
                parameters_replaced: n_rep,
                rejected_candidates: rejected,
            }),
        }));
    }
    Err(MutationReject::ProposalsExhausted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn founders_respect_bounds() {
        let cfg = Config::reference();
        let rng = SemanticRng::new(cfg.seed);
        let g = founder(&cfg, &rng, 3, 3);
        assert_eq!(g.params.len(), 9_988);
        let a = (3.0f64 / 104.0).sqrt() as f32;
        let p = g.params.to_f32();
        assert!(p[..104 * 64].iter().all(|v| v.abs() <= a));
        assert!(p[104 * 64..104 * 64 + 64].iter().all(|&v| v == 0.0));
        let n: f64 = g.affinity.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
        let nm: f64 = g.motion.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
        assert!(nm <= cfg.chemistry.motion_response_norm_limit + 1e-9);
        assert_eq!(g, founder(&cfg, &rng, 3, 3));
        assert_ne!(g.params, founder(&cfg, &rng, 4, 4).params);
    }

    #[test]
    fn mutation_is_addressed_and_bounded() {
        let cfg = Config::reference();
        let rng = SemanticRng::new(cfg.seed);
        let g = founder(&cfg, &rng, 1, 1);
        let c1 = mutate(&cfg, &rng, &g, 1, 500, 99).unwrap();
        let c2 = mutate(&cfg, &rng, &g, 1, 500, 99).unwrap();
        assert_eq!(c1, c2, "operational retry repeats proposals");
        let c3 = mutate(&cfg, &rng, &g, 1, 501, 99).unwrap();
        assert_ne!(c1.params, c3.params, "a later tick gets fresh proposals");
        assert_ne!(c1.affinity, g.affinity);
        assert!(c1.scale() >= cfg.sensors.minimum_scale_world_units - 1e-6);
        assert!(c1.scale() <= cfg.sensors.maximum_scale_world_units + 1e-6);
    }

    #[test]
    fn f16_genomes_hold_rounded_weights_and_inherit_exactly() {
        let mut cfg = Config::reference();
        cfg.controller.canonical_weight_dtype = "f16".into();
        let rng = SemanticRng::new(cfg.seed);
        let wide = founder(&Config::reference(), &rng, 3, 3);
        let g = founder(&cfg, &rng, 3, 3);
        let Weights::F16(bits) = &g.params else { panic!("f16 genome expected") };
        assert_eq!(bits.len(), 9_988);
        // The 16-bit founder is the 32-bit founder rounded, weight by weight.
        let w = wide.params.to_f32();
        for (h, x) in bits.iter().zip(w.iter()) {
            assert_eq!(*h, f32_to_f16(*x));
        }
        // Children are f16 again, deterministic, and differ from the parent only in stored values.
        let c1 = mutate(&cfg, &rng, &g, 3, 700, 9).unwrap();
        let c2 = mutate(&cfg, &rng, &g, 3, 700, 9).unwrap();
        assert_eq!(c1, c2);
        assert!(matches!(c1.params, Weights::F16(_)));
        assert_ne!(c1.params, g.params);
        // Widening and re-narrowing changes nothing: everything stored is already representable.
        let wide_again = c1.params.to_f32().into_owned();
        assert_eq!(Weights::from_f32(WeightDtype::F16, wide_again), c1.params);
    }

    #[test]
    fn c16_all_noop_candidates_reject_birth() {
        let mut cfg = Config::reference();
        cfg.mutation.weight_probability = 0.0;
        cfg.mutation.replacement_probability = 0.0;
        cfg.mutation.chemistry_standard_deviation = 0.0;
        cfg.mutation.sensing_log_scale_standard_deviation = 0.0;
        cfg.architecture_mutation.enabled = false;
        cfg.sensors.initial_scale_world_units = 4.0; // interior of the reflect interval
        let rng = SemanticRng::new(cfg.seed);
        let g = founder(&cfg, &rng, 1, 1);
        assert_eq!(mutate(&cfg, &rng, &g, 1, 10, 2), Err(MutationReject::ProposalsExhausted));
        cfg.mutation.reject_effective_noop = false;
        let clone = mutate(&cfg, &rng, &g, 1, 10, 2).unwrap();
        assert_eq!(clone.params, g.params);
    }

    #[test]
    fn topology_mutation_moves_within_catalog() {
        let mut cfg = Config::reference();
        cfg.architecture_mutation.probability_per_birth = 1.0;
        let rng = SemanticRng::new(cfg.seed);
        let g = founder(&cfg, &rng, 1, 1);
        let c = mutate(&cfg, &rng, &g, 1, 10, 2).unwrap();
        assert_ne!(c.topology, g.topology);
        assert_eq!(c.params.len(), param_count(CATALOG[c.topology as usize]));
        assert!(c.mutation.as_ref().unwrap().topology_changed);
    }
}
