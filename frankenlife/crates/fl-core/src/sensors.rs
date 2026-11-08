//! Density sensor pyramid and the 104-wide observation vector (plan §6.1).
//!
//! Level 0 holds densities `R = e/(hx hy)`, `M = m/(hx hy)`. Each coarser level
//! is the arithmetic 2x2 average of the finer densities (never of integrated
//! energy). At a level, raw `R, M` and their analytic derivatives come from the
//! same cell-centered bilinear interpolant; levels are blended log-linearly in
//! scale *before* any normalization.

use crate::config::SensorsCfg;
use crate::field::Field;
use crate::CHANNELS;

/// Number of field planes: energy density plus 32 moment densities.
pub const PLANES: usize = CHANNELS + 1;
/// Observation vector width.
pub const SENSOR_WIDTH: usize = 104;

#[derive(Debug, Clone)]
pub struct Level {
    pub gx: usize,
    pub gy: usize,
    pub hx: f64,
    pub hy: f64,
    /// Cell-major planes: `d[c * 33 + 0] = R`, `d[c * 33 + 1 + k] = M_k`.
    pub d: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Pyramid {
    pub levels: Vec<Level>,
    /// Spacing of level 0 used for scale mapping: `max(hx, hy)`.
    pub s0: f64,
}

/// Raw blended sample: values and physical x/y derivatives for `[R, M_0..M_31]`.
#[derive(Debug, Clone, Copy)]
pub struct RawSample {
    pub v: [f64; PLANES],
    pub dx: [f64; PLANES],
    pub dy: [f64; PLANES],
}

/// Normalized observation plus the unsaturated physical derivatives used by motion.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// The first 99 sensor inputs (values, x derivatives, y derivatives).
    pub field_inputs: [f32; 99],
    /// Physical gradient of `q` (index 0) and of `Z_k` (index 1 + k), x and y.
    pub grad_x: [f64; PLANES],
    pub grad_y: [f64; PLANES],
}

impl Pyramid {
    pub fn build(field: &Field, extra_levels: usize) -> Pyramid {
        let area = field.hx * field.hy;
        let mut d0 = vec![0.0f32; field.cells() * PLANES];
        for c in 0..field.cells() {
            d0[c * PLANES] = (field.e[c] as f64 / area) as f32;
            for k in 0..CHANNELS {
                d0[c * PLANES + 1 + k] = (field.m[c * CHANNELS + k] as f64 / area) as f32;
            }
        }
        let mut levels = vec![Level { gx: field.gx, gy: field.gy, hx: field.hx, hy: field.hy, d: d0 }];
        for _ in 0..extra_levels {
            let p = levels.last().unwrap();
            let (gx, gy) = (p.gx / 2, p.gy / 2);
            let mut d = vec![0.0f32; gx * gy * PLANES];
            for y in 0..gy {
                for x in 0..gx {
                    let c = y * gx + x;
                    let f = |xx: usize, yy: usize| (yy * p.gx + xx) * PLANES;
                    let (a, b, cc, dd) = (f(2 * x, 2 * y), f(2 * x + 1, 2 * y), f(2 * x, 2 * y + 1), f(2 * x + 1, 2 * y + 1));
                    for k in 0..PLANES {
                        d[c * PLANES + k] =
                            ((p.d[a + k] as f64 + p.d[b + k] as f64 + p.d[cc + k] as f64 + p.d[dd + k] as f64) * 0.25) as f32;
                    }
                }
            }
            levels.push(Level { gx, gy, hx: p.hx * 2.0, hy: p.hy * 2.0, d });
        }
        Pyramid { levels, s0: field.hx.max(field.hy) }
    }

    /// Bilinear value and analytic derivatives of all planes at one level.
    pub fn sample_level(&self, l: usize, x: f64, y: f64) -> RawSample {
        let lv = &self.levels[l];
        let fx = x / lv.hx - 0.5;
        let fy = y / lv.hy - 0.5;
        let ix = fx.floor();
        let iy = fy.floor();
        let tx = fx - ix;
        let ty = fy - iy;
        let x0 = (ix as i64).rem_euclid(lv.gx as i64) as usize;
        let y0 = (iy as i64).rem_euclid(lv.gy as i64) as usize;
        let x1 = (x0 + 1) % lv.gx;
        let y1 = (y0 + 1) % lv.gy;
        let o00 = (y0 * lv.gx + x0) * PLANES;
        let o10 = (y0 * lv.gx + x1) * PLANES;
        let o01 = (y1 * lv.gx + x0) * PLANES;
        let o11 = (y1 * lv.gx + x1) * PLANES;
        let mut s = RawSample { v: [0.0; PLANES], dx: [0.0; PLANES], dy: [0.0; PLANES] };
        for k in 0..PLANES {
            let (v00, v10, v01, v11) = (lv.d[o00 + k] as f64, lv.d[o10 + k] as f64, lv.d[o01 + k] as f64, lv.d[o11 + k] as f64);
            s.v[k] = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11;
            s.dx[k] = ((1.0 - ty) * (v10 - v00) + ty * (v11 - v01)) / lv.hx;
            s.dy[k] = ((1.0 - tx) * (v01 - v00) + tx * (v11 - v01)) / lv.hy;
        }
        s
    }

    /// Log-linear blend between the two levels bracketing `scale`.
    pub fn sample(&self, x: f64, y: f64, scale: f64) -> RawSample {
        let top = self.levels.len() - 1;
        let lg = (scale / self.s0).log2().clamp(0.0, top as f64);
        let mut l = lg.floor() as usize;
        if l >= top {
            l = top - 1;
        }
        let t = (lg - l as f64).clamp(0.0, 1.0);
        let a = self.sample_level(l, x, y);
        if t == 0.0 {
            return a;
        }
        let b = self.sample_level(l + 1, x, y);
        let mut s = a;
        for k in 0..PLANES {
            s.v[k] = (1.0 - t) * a.v[k] + t * b.v[k];
            s.dx[k] = (1.0 - t) * a.dx[k] + t * b.dx[k];
            s.dy[k] = (1.0 - t) * a.dy[k] + t * b.dy[k];
        }
        s
    }

    /// Local energy density at level 0 (uptake request input).
    pub fn density0(&self, x: f64, y: f64) -> f64 {
        let lv = &self.levels[0];
        let fx = x / lv.hx - 0.5;
        let fy = y / lv.hy - 0.5;
        let ix = fx.floor();
        let iy = fy.floor();
        let tx = fx - ix;
        let ty = fy - iy;
        let x0 = (ix as i64).rem_euclid(lv.gx as i64) as usize;
        let y0 = (iy as i64).rem_euclid(lv.gy as i64) as usize;
        let x1 = (x0 + 1) % lv.gx;
        let y1 = (y0 + 1) % lv.gy;
        let g = |xx: usize, yy: usize| lv.d[(yy * lv.gx + xx) * PLANES] as f64;
        ((1.0 - tx) * (1.0 - ty) * g(x0, y0) + tx * (1.0 - ty) * g(x1, y0) + (1.0 - tx) * ty * g(x0, y1) + tx * ty * g(x1, y1)).max(0.0)
    }
}

/// Normalize a blended raw sample into controller inputs and motion gradients.
pub fn observe(raw: &RawSample, cfg: &SensorsCfg) -> Observation {
    let rf = cfg.composition_energy_floor_density;
    let rs = cfg.energy_density_scale;
    let lg = cfg.gradient_length_scale_world_units;
    let r = raw.v[0].max(0.0);
    let den = r + rf;
    let q = r / (r + rs);
    let dq = rs / ((r + rs) * (r + rs));
    let mut o = Observation { field_inputs: [0.0; 99], grad_x: [0.0; PLANES], grad_y: [0.0; PLANES] };
    o.grad_x[0] = dq * raw.dx[0];
    o.grad_y[0] = dq * raw.dy[0];
    o.field_inputs[0] = q as f32;
    for k in 0..CHANNELS {
        let z = raw.v[1 + k] / den;
        o.field_inputs[1 + k] = z as f32;
        o.grad_x[1 + k] = (raw.dx[1 + k] - z * raw.dx[0]) / den;
        o.grad_y[1 + k] = (raw.dy[1 + k] - z * raw.dy[0]) / den;
    }
    for k in 0..PLANES {
        o.field_inputs[PLANES + k] = (lg * o.grad_x[k]).tanh() as f32;
        o.field_inputs[2 * PLANES + k] = (lg * o.grad_y[k]).tanh() as f32;
    }
    o
}

/// The five intrinsic inputs, in table order.
pub fn intrinsic(cfg: &SensorsCfg, energy: f64, age_s: f64, vx: f64, vy: f64, de_dt: f64) -> [f32; 5] {
    [
        (energy / (energy + cfg.internal_energy_scale)) as f32,
        (age_s / (age_s + cfg.age_scale_seconds)) as f32,
        (vx / cfg.velocity_scale).tanh() as f32,
        (vy / cfg.velocity_scale).tanh() as f32,
        (de_dt / cfg.energy_change_scale_per_second).tanh() as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn c10_constant_density_identical_across_scales() {
        // h = 2, e = 8, m = 4a  =>  R = 2, M = a at every level.
        let mut f = Field::new(32, 32, 64.0, 64.0);
        for c in 0..f.cells() {
            f.e[c] = 8.0;
            f.m[c * CHANNELS + 5] = 4.0;
        }
        let p = Pyramid::build(&f, 3);
        let cfg = Config::reference().sensors;
        let mut first: Option<[f32; 99]> = None;
        for l in 0..4 {
            let s = p.sample_level(l, 13.7, 50.2);
            assert!((s.v[0] - 2.0).abs() < 1e-6);
            assert!((s.v[6] - 1.0).abs() < 1e-6);
            assert!(s.dx.iter().chain(s.dy.iter()).all(|d| d.abs() < 1e-9));
        }
        for scale in [2.0, 3.0, 5.5, 8.0, 16.0] {
            let o = observe(&p.sample(13.7, 50.2, scale), &cfg);
            let z = o.field_inputs[6] as f64;
            assert!((z - 1.0 / (2.0 + cfg.composition_energy_floor_density)).abs() < 1e-6);
            if let Some(f0) = first {
                assert_eq!(f0, o.field_inputs, "scale {scale}");
            } else {
                first = Some(o.field_inputs);
            }
        }
    }

    #[test]
    fn c10_quotient_derivative_matches_finite_difference() {
        let mut f = Field::new(16, 16, 16.0, 16.0);
        for y in 0..16 {
            for x in 0..16 {
                let c = y * 16 + x;
                f.e[c] = 1.0 + 0.3 * x as f32 + 0.1 * y as f32;
                f.m[c * CHANNELS + 2] = 0.2 * x as f32 - 0.05 * y as f32;
            }
        }
        let p = Pyramid::build(&f, 1);
        let cfg = Config::reference().sensors;
        let (x, y, h) = (6.3, 7.4, 1e-4);
        let z = |x: f64, y: f64| {
            let s = p.sample_level(0, x, y);
            s.v[3] / (s.v[0] + cfg.composition_energy_floor_density)
        };
        let o = observe(&p.sample_level(0, x, y), &cfg);
        let fd = (z(x + h, y) - z(x - h, y)) / (2.0 * h);
        assert!((o.grad_x[3] - fd).abs() < 1e-6, "{} vs {}", o.grad_x[3], fd);
    }
}
