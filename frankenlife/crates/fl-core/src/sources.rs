//! Seeded, moving Gaussian energy sources (plan §4.5, A.3 "Source paths/deposits").

use crate::config::Config;
use crate::field::{min_image, wrap, Field};
use crate::rng::{Lane, SemanticRng};
use crate::CHANNELS;

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub id: u32,
    /// Position at the start of the current path segment.
    pub anchor: [f64; 2],
    /// Current segment ordinal; within-segment time derives from integer ticks.
    pub ordinal: u64,
    pub start_dir: [f64; 2],
    pub end_dir: [f64; 2],
    /// Fixed unit chemical direction.
    pub chem: [f32; CHANNELS],
}

fn travel_dir(rng: &SemanticRng, id: u32, ordinal: u64) -> [f64; 2] {
    let u = rng.uniform(Lane::SourcePathDirection, id as u64, ordinal, 0, 0);
    let t = std::f64::consts::TAU * u;
    [t.cos(), t.sin()]
}

/// Isotropic unit vector from independent normals (sequential reduction order).
pub fn isotropic_unit<const N: usize>(z: &[f64; N]) -> Option<[f32; N]> {
    let n = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(n > 1e-12) {
        return None;
    }
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = (z[i] / n) as f32;
    }
    // Rounding to f32 can leave the norm a hair above one; scale down if so.
    let n32 = out.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    if n32 > 1.0 {
        for v in out.iter_mut() {
            *v = (*v as f64 / n32 * (1.0 - 1e-7)) as f32;
        }
    }
    Some(out)
}

pub fn init_sources(cfg: &Config, rng: &SemanticRng) -> Vec<Source> {
    (0..cfg.sources.count)
        .map(|id| {
            let u = rng.uniform2(Lane::SourceCenter, id as u64, 0, 0, 0);
            let mut z = [0.0f64; CHANNELS];
            let mut attempt = 0u64;
            let chem = loop {
                rng.normals(Lane::SourceChemistry, id as u64, attempt, 0, &mut z);
                if let Some(a) = isotropic_unit(&z) {
                    break a;
                }
                attempt += 1;
            };
            Source {
                id,
                anchor: [u[0] * cfg.world.width, u[1] * cfg.world.height],
                ordinal: 0,
                start_dir: travel_dir(rng, id, 0),
                end_dir: travel_dir(rng, id, 1),
                chem,
            }
        })
        .collect()
}

pub fn segment_ticks(cfg: &Config) -> u64 {
    (cfg.sources.path_direction_interval_seconds / cfg.dt()).round() as u64
}

fn displacement(speed: f64, t_seg: f64, a: [f64; 2], b: [f64; 2], u: f64) -> [f64; 2] {
    let s = u * u * u - u * u * u * u / 2.0;
    [speed * t_seg * (a[0] * u + (b[0] - a[0]) * s), speed * t_seg * (a[1] * u + (b[1] - a[1]) * s)]
}

impl Source {
    /// Roll the path forward so that `ordinal == segment containing tick n`.
    pub fn advance_to(&mut self, n: u64, cfg: &Config, rng: &SemanticRng) {
        let seg = segment_ticks(cfg);
        let target = n / seg;
        let t_seg = cfg.sources.path_direction_interval_seconds;
        while self.ordinal < target {
            let d = displacement(cfg.sources.path_speed, t_seg, self.start_dir, self.end_dir, 1.0);
            self.anchor = [wrap(self.anchor[0] + d[0], cfg.world.width), wrap(self.anchor[1] + d[1], cfg.world.height)];
            self.ordinal += 1;
            self.start_dir = self.end_dir;
            self.end_dir = travel_dir(rng, self.id, self.ordinal + 1);
        }
    }

    /// Change the path speed at tick `n` without moving the source: shifts the
    /// segment anchor so the center stays continuous across the change.
    /// Requires `advance_to(n)` (with the old speed) first.
    pub fn reanchor_for_speed(&mut self, n: u64, cfg: &Config, old_speed: f64, new_speed: f64) {
        let seg = segment_ticks(cfg);
        debug_assert_eq!(self.ordinal, n / seg);
        let u = (n % seg) as f64 / seg as f64;
        let t_seg = cfg.sources.path_direction_interval_seconds;
        let before = displacement(old_speed, t_seg, self.start_dir, self.end_dir, u);
        let after = displacement(new_speed, t_seg, self.start_dir, self.end_dir, u);
        self.anchor = [wrap(self.anchor[0] + before[0] - after[0], cfg.world.width), wrap(self.anchor[1] + before[1] - after[1], cfg.world.height)];
    }

    /// Rotate the chemical direction by `degrees` toward a deterministic random
    /// direction perpendicular to it, addressed by `(source, tick, command seq)`.
    pub fn drift_chemistry(&mut self, rng: &SemanticRng, tick: u64, seq: u64, degrees: f64) {
        let mut z = [0.0f64; CHANNELS];
        rng.normals(Lane::SourceChemistryDrift, self.id as u64, tick, seq, &mut z);
        let c: [f64; CHANNELS] = std::array::from_fn(|k| self.chem[k] as f64);
        let dot: f64 = (0..CHANNELS).map(|k| z[k] * c[k]).sum();
        let mut u = [0.0f64; CHANNELS];
        for k in 0..CHANNELS {
            u[k] = z[k] - dot * c[k];
        }
        let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm > 1e-12) {
            return;
        }
        let (s, co) = degrees.to_radians().sin_cos();
        let mut next = [0.0f64; CHANNELS];
        for k in 0..CHANNELS {
            next[k] = c[k] * co + u[k] / norm * s;
        }
        if let Some(unit) = isotropic_unit(&next) {
            self.chem = unit;
        }
    }

    /// Center at the midpoint time `(n + 1/2) dt` of the advance `n -> n+1`.
    /// Requires `advance_to(n)` first.
    pub fn center_mid(&self, n: u64, cfg: &Config) -> [f64; 2] {
        let seg = segment_ticks(cfg);
        debug_assert_eq!(self.ordinal, n / seg);
        let u = ((n % seg) as f64 + 0.5) / seg as f64;
        let d = displacement(cfg.sources.path_speed, cfg.sources.path_direction_interval_seconds, self.start_dir, self.end_dir, u);
        [wrap(self.anchor[0] + d[0], cfg.world.width), wrap(self.anchor[1] + d[1], cfg.world.height)]
    }
}

/// Normalized truncated minimum-image Gaussian weights over field cells.
/// Returns `(cell, weight)` with weights summing to one, or `None` for empty support.
pub fn kernel_weights(field: &Field, center: [f64; 2], sigma: f64, cutoff_sigma: f64) -> Option<Vec<(usize, f64)>> {
    let (w, h) = (field.width(), field.height());
    let cut = cutoff_sigma * sigma;
    let rx = (cut / field.hx).ceil() as i64 + 1;
    let ry = (cut / field.hy).ceil() as i64 + 1;
    let (gx, gy) = (field.gx as i64, field.gy as i64);
    let cx = (center[0] / field.hx - 0.5).floor() as i64;
    let cy = (center[1] / field.hy - 0.5).floor() as i64;
    let xs: Vec<i64> = if 2 * rx + 1 >= gx { (0..gx).collect() } else { (cx - rx..=cx + rx).map(|v| v.rem_euclid(gx)).collect() };
    let ys: Vec<i64> = if 2 * ry + 1 >= gy { (0..gy).collect() } else { (cy - ry..=cy + ry).map(|v| v.rem_euclid(gy)).collect() };
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    let mut total = 0.0;
    for &iy in &ys {
        let py = (iy as f64 + 0.5) * field.hy;
        let dy = min_image(py - center[1], h);
        for &ix in &xs {
            let px = (ix as f64 + 0.5) * field.hx;
            let dx = min_image(px - center[0], w);
            let d2 = dx * dx + dy * dy;
            if d2 <= cut * cut {
                let k = (-d2 / (2.0 * sigma * sigma)).exp();
                if k > 0.0 {
                    out.push(((iy * gx + ix) as usize, k));
                    total += k;
                }
            }
        }
    }
    if !(total > 0.0) || !total.is_finite() {
        return None;
    }
    for p in &mut out {
        p.1 /= total;
    }
    Some(out)
}

/// Inject every source's `P * dt` into `field`. Returns total injected energy.
pub fn inject(field: &mut Field, sources: &[Source], n: u64, cfg: &Config) -> Result<f64, String> {
    let q_total = cfg.sources.power_per_source * cfg.dt();
    let mut injected = 0.0;
    for s in sources {
        let c = s.center_mid(n, cfg);
        let ws = kernel_weights(field, c, cfg.sources.sigma_world_units, cfg.sources.kernel_cutoff_sigma)
            .ok_or_else(|| format!("source {} has empty kernel support", s.id))?;
        for (cell, w) in ws {
            let q = q_total * w;
            let old = field.e[cell] as f64;
            let new = (old + q) as f32;
            injected += new as f64 - old; // account what was actually stored
            field.e[cell] = new;
            let m = &mut field.m[cell * CHANNELS..(cell + 1) * CHANNELS];
            for k in 0..CHANNELS {
                m[k] = (m[k] as f64 + q * s.chem[k] as f64) as f32;
            }
        }
    }
    Ok(injected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_speed_mid_segment_does_not_move_the_source() {
        let mut cfg = Config::reference();
        cfg.sources.path_speed = 2.0;
        let rng = SemanticRng::new(cfg.seed);
        let seg = segment_ticks(&cfg);
        for n in [3, seg / 2 + 7, seg - 1, seg + seg / 3] {
            let mut s = init_sources(&cfg, &rng).remove(0);
            s.advance_to(n, &cfg, &rng);
            let before = s.center_mid(n, &cfg);
            let mut fast = cfg.clone();
            fast.sources.path_speed = 9.0;
            let mut kept = s.clone();
            kept.reanchor_for_speed(n, &cfg, 2.0, 9.0);
            let after = kept.center_mid(n, &fast);
            let jump = |a: [f64; 2], b: [f64; 2]| (min_image(a[0] - b[0], cfg.world.width).powi(2) + min_image(a[1] - b[1], cfg.world.height).powi(2)).sqrt();
            // Only the half tick of extra travel at the new speed remains.
            assert!(jump(before, after) < 9.0 * cfg.dt(), "tick {n}: moved {}", jump(before, after));
            // Without the re-anchor the same change teleports the source.
            let teleport = s.center_mid(n, &fast);
            assert!(jump(before, teleport) > 0.5 || n < 100, "tick {n}: teleport only {}", jump(before, teleport));
        }
    }

    #[test]
    fn chemistry_drift_rotates_by_the_requested_angle_deterministically() {
        let cfg = Config::reference();
        let rng = SemanticRng::new(cfg.seed);
        let s0 = init_sources(&cfg, &rng).remove(0);
        for deg in [5.0f64, 25.0, 90.0] {
            let mut a = s0.clone();
            a.drift_chemistry(&rng, 40, 7, deg);
            let mut b = s0.clone();
            b.drift_chemistry(&rng, 40, 7, deg);
            assert_eq!(a.chem, b.chem, "same address, same drift");
            let norm: f64 = a.chem.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
            let dot: f64 = (0..CHANNELS).map(|k| a.chem[k] as f64 * s0.chem[k] as f64).sum();
            assert!((dot - deg.to_radians().cos()).abs() < 1e-5, "angle {deg}: dot {dot}");
        }
        let mut c = s0.clone();
        c.drift_chemistry(&rng, 41, 7, 25.0);
        let mut d = s0.clone();
        d.drift_chemistry(&rng, 40, 7, 25.0);
        assert_ne!(c.chem, d.chem, "a later press drifts somewhere else");
    }

    #[test]
    fn c17_kernel_normalizes_at_any_resolution() {
        for g in [16usize, 32, 128] {
            let f = Field::new(g, g, 128.0, 128.0);
            for c in [[0.0, 0.0], [64.3, 1.2], [127.9, 127.9]] {
                let w = kernel_weights(&f, c, 10.0, 4.0).unwrap();
                let s: f64 = w.iter().map(|p| p.1).sum();
                assert!((s - 1.0).abs() < 1e-12, "{g} {s}");
                let mut cells: Vec<usize> = w.iter().map(|p| p.0).collect();
                cells.sort();
                cells.dedup();
                assert_eq!(cells.len(), w.len(), "no duplicate support cells");
            }
        }
    }

    #[test]
    fn c17_paths_are_continuous_and_speed_bounded() {
        let cfg = Config::reference();
        let rng = SemanticRng::new(cfg.seed);
        let mut s = init_sources(&cfg, &rng);
        let seg = segment_ticks(&cfg);
        let chem0 = s[0].chem;
        let dt = cfg.dt();
        let mut prev: Option<[f64; 2]> = None;
        for n in (seg - 50)..(seg + 50) {
            s[0].advance_to(n, &cfg, &rng);
            let c = s[0].center_mid(n, &cfg);
            if let Some(p) = prev {
                let dx = min_image(c[0] - p[0], cfg.world.width);
                let dy = min_image(c[1] - p[1], cfg.world.height);
                let v = (dx * dx + dy * dy).sqrt() / dt;
                assert!(v <= cfg.sources.path_speed + 1e-9, "speed {v}");
            }
            prev = Some(c);
        }
        assert_eq!(s[0].chem, chem0);
    }

    #[test]
    fn injection_matches_power() {
        let cfg = Config::reference();
        let rng = SemanticRng::new(cfg.seed);
        let mut srcs = init_sources(&cfg, &rng);
        let mut f = Field::new(128, 128, 128.0, 128.0);
        for s in &mut srcs {
            s.advance_to(0, &cfg, &rng);
        }
        let q = inject(&mut f, &srcs, 0, &cfg).unwrap();
        let expect = cfg.sources.count as f64 * cfg.sources.power_per_source * cfg.dt();
        assert!((q - expect).abs() < 1e-4, "{q} vs {expect}");
        assert!((f.total_energy() - q).abs() < 1e-6);
        assert!(f.cone_report().0 < 1e-5);
    }
}
