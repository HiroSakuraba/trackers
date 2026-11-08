//! FrankenTorch population-inference backend (plan §10.2, §10.5, §9.2).
//!
//! Each organism keeps its *own* weights. Organisms that share a catalog
//! topology are packed into persistent per-topology weight slabs, one slab per
//! affine layer laid out `[slot, out, in]` — exactly fl-core's per-genome
//! row-major `W[out][in]` layout, so uploading a genome is a straight copy.
//! A layer for the whole population is then one FrankenTorch batched matmul
//!
//! ```text
//!   Y[b] = W[b] · x[b]      lhs [B, out, in] × rhs [B, in, 1] → [B, out, 1]
//! ```
//!
//! via `bmm_tensor_contiguous_f32_into` (caller-owned output), followed by
//! bias and `tanh` on hidden layers. Slots persist across ticks: a genome is
//! uploaded once at first use, and released after it stops being evaluated.
//!
//! Executor boundary (plan §3.4): FrankenTorch's CPU batched matmul runs on its
//! own internal Rayon pool. That work is *not* Asupersync-scheduled data
//! parallelism, and `capability()` says so.
//!
//! Numerics (plan §10.3): accumulation order differs from fl-core's scalar
//! reference, so logits agree to a tolerance, not bitwise; a run using this
//! backend is its own determinism class.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use fl_core::nn::{layer_dims, CATALOG, OUTPUT_WIDTH};
use fl_core::{ControllerBackend, InferenceJob};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::bmm_tensor_contiguous_f32_into;

/// Evict a slot after this many evaluations of other genomes without it.
const EVICT_AFTER_CALLS: u64 = 64;

struct Slab {
    dims: Vec<(usize, usize)>,
    /// Per layer: `[cap, out, in]` weights and `[cap, out]` biases.
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    owner: Vec<Option<u64>>,
    last_used: Vec<u64>,
    free: Vec<usize>,
}

impl Slab {
    fn new(topology: usize) -> Slab {
        let dims = layer_dims(CATALOG[topology]);
        Slab { w: vec![Vec::new(); dims.len()], b: vec![Vec::new(); dims.len()], dims, owner: Vec::new(), last_used: Vec::new(), free: Vec::new() }
    }

    fn cap(&self) -> usize {
        self.owner.len()
    }

    fn upload(&mut self, slot: usize, params: &[f32]) {
        let mut off = 0;
        for (l, &(i, o)) in self.dims.iter().enumerate() {
            self.w[l][slot * o * i..(slot + 1) * o * i].copy_from_slice(&params[off..off + o * i]);
            off += o * i;
            self.b[l][slot * o..(slot + 1) * o].copy_from_slice(&params[off..off + o]);
            off += o;
        }
        debug_assert_eq!(off, params.len());
    }

    fn alloc(&mut self, genome: u64, params: &[f32], call: u64) -> usize {
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                for (l, &(i, o)) in self.dims.iter().enumerate() {
                    let (nw, nb) = (self.w[l].len() + o * i, self.b[l].len() + o);
                    self.w[l].resize(nw, 0.0);
                    self.b[l].resize(nb, 0.0);
                }
                self.owner.push(None);
                self.last_used.push(0);
                self.cap() - 1
            }
        };
        self.upload(slot, params);
        self.owner[slot] = Some(genome);
        self.last_used[slot] = call;
        slot
    }
}

/// Counters for one backend instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct TorchStats {
    pub calls: u64,
    pub evaluations: u64,
    pub genome_uploads: u64,
    pub evictions: u64,
    pub batched_matmuls: u64,
    pub padded_slot_evaluations: u64,
}

pub struct TorchBackend {
    slabs: Vec<Slab>,
    slot_of: HashMap<u64, (usize, usize)>,
    call: u64,
    pub stats: TorchStats,
    act: Vec<f32>,
    next: Vec<f32>,
}

impl Default for TorchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TorchBackend {
    pub fn new() -> TorchBackend {
        TorchBackend { slabs: (0..CATALOG.len()).map(Slab::new).collect(), slot_of: HashMap::new(), call: 0, stats: TorchStats::default(), act: Vec::new(), next: Vec::new() }
    }

    /// Human-readable statement of what this backend is and is not.
    pub fn capability() -> &'static str {
        "frankentorch-kernel-cpu bmm_tensor_contiguous_f32_into over persistent per-topology slabs; \
         independent weights per organism; executor = FrankenTorch's internal Rayon pool (not Asupersync-scheduled); \
         tolerance-equivalent to the fl-core scalar reference, not bitwise"
    }

    /// Bytes of resident weight slabs (allocated capacity, including free slots).
    pub fn resident_bytes(&self) -> u64 {
        self.slabs.iter().map(|s| (s.w.iter().map(Vec::len).sum::<usize>() + s.b.iter().map(Vec::len).sum::<usize>()) as u64 * 4).sum()
    }

    fn evict_stale(&mut self) {
        for (t, slab) in self.slabs.iter_mut().enumerate() {
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
            // Compact a mostly-empty slab by dropping it; live genomes re-upload on next use.
            if slab.cap() >= 64 && slab.free.len() * 2 > slab.cap() {
                for g in slab.owner.iter().flatten() {
                    self.slot_of.remove(g);
                }
                *slab = Slab::new(t);
            }
        }
    }
}

impl ControllerBackend for TorchBackend {
    fn name(&self) -> &str {
        "frankentorch-bmm-f32"
    }

    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        self.call += 1;
        self.stats.calls += 1;
        self.stats.evaluations += jobs.len() as u64;
        // 1. Resolve (or upload) each job's slot.
        let mut placed: Vec<(usize, usize)> = Vec::with_capacity(jobs.len());
        for j in jobs {
            let g = j.genome;
            let t = g.topology as usize;
            let slot = match self.slot_of.get(&g.id) {
                Some(&(tt, s)) if tt == t => s,
                _ => {
                    let s = self.slabs[t].alloc(g.id, &g.params, self.call);
                    self.slot_of.insert(g.id, (t, s));
                    self.stats.genome_uploads += 1;
                    s
                }
            };
            self.slabs[t].last_used[slot] = self.call;
            placed.push((t, slot));
        }
        // 2. One batched pass per topology that has work.
        for t in 0..self.slabs.len() {
            let mine: Vec<usize> = (0..jobs.len()).filter(|&k| placed[k].0 == t).collect();
            if mine.is_empty() {
                continue;
            }
            let slab = &self.slabs[t];
            let cap = slab.cap();
            self.stats.padded_slot_evaluations += (cap - mine.len()) as u64;
            let (in0, _) = slab.dims[0];
            self.act.clear();
            self.act.resize(cap * in0, 0.0);
            for &k in &mine {
                let s = placed[k].1;
                self.act[s * in0..(s + 1) * in0].copy_from_slice(jobs[k].input);
            }
            let last = slab.dims.len() - 1;
            for (l, &(i, o)) in slab.dims.iter().enumerate() {
                let lhs = TensorMeta::from_shape(vec![cap, o, i], DType::F32, Device::Cpu);
                let rhs = TensorMeta::from_shape(vec![cap, i, 1], DType::F32, Device::Cpu);
                self.next.clear();
                self.next.resize(cap * o, 0.0);
                bmm_tensor_contiguous_f32_into(&slab.w[l], &self.act, &lhs, &rhs, &mut self.next).map_err(|e| format!("bmm layer {l}: {e:?}"))?;
                self.stats.batched_matmuls += 1;
                let bias = &slab.b[l];
                for (y, bv) in self.next.iter_mut().zip(bias.iter()) {
                    *y += *bv;
                    if l < last {
                        *y = y.tanh();
                    }
                }
                std::mem::swap(&mut self.act, &mut self.next);
            }
            for &k in &mine {
                let s = placed[k].1;
                out[k].copy_from_slice(&self.act[s * OUTPUT_WIDTH..(s + 1) * OUTPUT_WIDTH]);
            }
        }
        self.evict_stale();
        Ok(())
    }
}

/// Largest absolute logit difference between `backend` and the scalar reference on `jobs`.
pub fn max_abs_diff_vs_reference(backend: &mut dyn ControllerBackend, jobs: &[InferenceJob<'_>]) -> Result<f32, String> {
    let mut a = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
    let mut b = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
    backend.evaluate(jobs, &mut a)?;
    fl_core::ReferenceBackend.evaluate(jobs, &mut b)?;
    Ok(a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0, f32::max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::genome::founder;
    use fl_core::nn::INPUT_WIDTH;
    use fl_core::rng::SemanticRng;
    use fl_core::Config;

    #[test]
    fn independent_weights_match_reference_within_tolerance() {
        let mut cfg = Config::reference();
        let rng = SemanticRng::new(9);
        let mut genomes = Vec::new();
        for (k, h) in [vec![8u32], vec![64, 32], vec![32], vec![64, 32], vec![128, 96]].into_iter().enumerate() {
            cfg.controller.hidden_widths = h;
            for r in 0..7u64 {
                genomes.push(founder(&cfg, &rng, k as u64 * 100 + r, k as u64 * 100 + r));
            }
        }
        let inputs: Vec<[f32; INPUT_WIDTH]> = (0..genomes.len()).map(|g| std::array::from_fn(|i| (((g * 131 + i * 17) % 200) as f32 / 100.0 - 1.0))).collect();
        let jobs: Vec<InferenceJob<'_>> = genomes.iter().zip(&inputs).map(|(g, x)| InferenceJob { organism: g.id, genome: g, input: x }).collect();
        let mut be = TorchBackend::new();
        let d = max_abs_diff_vs_reference(&mut be, &jobs).unwrap();
        assert!(d < 1e-4, "max |Δlogit| = {d}");
        // Distinct genomes give distinct outputs: no accidental weight sharing.
        let mut out = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
        be.evaluate(&jobs, &mut out).unwrap();
        assert_ne!(out[7], out[8]);
        assert_eq!(be.stats.genome_uploads, jobs.len() as u64, "second call reuses resident slots");
    }
}
