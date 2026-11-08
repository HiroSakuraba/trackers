//! The energy/moment field: `e_c >= 0`, `m_c in R^32`, `||m_c|| <= e_c` (plan §4.2).
//!
//! Storage is genuinely typed F32, cell-major for moments (`m[c * 32 + k]`),
//! in cell-integrated units. Arithmetic that moves energy is carried out in
//! F64 and rounded once on store; the energy ledger reports the residual.

use crate::CHANNELS;

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub gx: usize,
    pub gy: usize,
    pub hx: f64,
    pub hy: f64,
    pub e: Vec<f32>,
    pub m: Vec<f32>,
}

/// A merged compact interpolation stencil (`grid-cic-v1`): up to four distinct cells,
/// nonnegative weights summing to one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stencil {
    pub n: usize,
    pub cells: [usize; 4],
    pub w: [f64; 4],
}

impl Field {
    pub fn new(gx: usize, gy: usize, width: f64, height: f64) -> Field {
        Field { gx, gy, hx: width / gx as f64, hy: height / gy as f64, e: vec![0.0; gx * gy], m: vec![0.0; gx * gy * CHANNELS] }
    }

    #[inline]
    pub fn cells(&self) -> usize {
        self.gx * self.gy
    }

    #[inline]
    pub fn width(&self) -> f64 {
        self.hx * self.gx as f64
    }

    #[inline]
    pub fn height(&self) -> f64 {
        self.hy * self.gy as f64
    }

    pub fn total_energy(&self) -> f64 {
        self.e.iter().map(|&v| v as f64).sum()
    }

    #[inline]
    pub fn moment(&self, c: usize) -> &[f32] {
        &self.m[c * CHANNELS..(c + 1) * CHANNELS]
    }

    /// Cell-centered bilinear stencil with periodic wrap; repeated wrapped cells are merged.
    pub fn stencil(&self, x: f64, y: f64) -> Stencil {
        let fx = x / self.hx - 0.5;
        let fy = y / self.hy - 0.5;
        let ix = fx.floor();
        let iy = fy.floor();
        let tx = fx - ix;
        let ty = fy - iy;
        let gx = self.gx as i64;
        let gy = self.gy as i64;
        let x0 = (ix as i64).rem_euclid(gx) as usize;
        let y0 = (iy as i64).rem_euclid(gy) as usize;
        let x1 = (x0 + 1) % self.gx;
        let y1 = (y0 + 1) % self.gy;
        let raw = [
            (y0 * self.gx + x0, (1.0 - tx) * (1.0 - ty)),
            (y0 * self.gx + x1, tx * (1.0 - ty)),
            (y1 * self.gx + x0, (1.0 - tx) * ty),
            (y1 * self.gx + x1, tx * ty),
        ];
        let mut s = Stencil { n: 0, cells: [0; 4], w: [0.0; 4] };
        for (c, w) in raw {
            if let Some(k) = s.cells[..s.n].iter().position(|&q| q == c) {
                s.w[k] += w;
            } else {
                s.cells[s.n] = c;
                s.w[s.n] = w;
                s.n += 1;
            }
        }
        s
    }

    /// Add a packet `(q, q * dir)` spread over `stencil` (energy and moment share the weights).
    /// `dir` must satisfy `||dir|| <= 1`.
    pub fn deposit(&mut self, s: &Stencil, q: f64, dir: &[f32; CHANNELS]) {
        for k in 0..s.n {
            let c = s.cells[k];
            let qc = q * s.w[k];
            self.e[c] = (self.e[c] as f64 + qc) as f32;
            let m = &mut self.m[c * CHANNELS..(c + 1) * CHANNELS];
            for j in 0..CHANNELS {
                m[j] = (m[j] as f64 + qc * dir[j] as f64) as f32;
            }
        }
    }

    /// Add another field elementwise (used for next-tick deposit accumulators).
    pub fn add_assign(&mut self, other: &Field) {
        for (a, b) in self.e.iter_mut().zip(&other.e) {
            *a += *b;
        }
        for (a, b) in self.m.iter_mut().zip(&other.m) {
            *a += *b;
        }
    }

    pub fn clear(&mut self) {
        self.e.iter_mut().for_each(|v| *v = 0.0);
        self.m.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Positive five-point diffusion (subcycled) followed by exponential decay.
    /// Returns energy removed by decay (charged to dissipation by the caller).
    pub fn transport(&mut self, diffusion: f64, decay_per_second: f64, dt: f64, scratch: &mut Field) -> (f64, u32) {
        let rx = diffusion * dt / (self.hx * self.hx);
        let ry = diffusion * dt / (self.hy * self.hy);
        let sub = (((rx + ry) / 0.5).ceil() as u32).max(1);
        if diffusion > 0.0 {
            let (rx, ry) = (rx / sub as f64, ry / sub as f64);
            for _ in 0..sub {
                self.diffuse_once(rx, ry, scratch);
            }
        }
        let mut removed = 0.0;
        if decay_per_second > 0.0 {
            let f = (-decay_per_second * dt).exp();
            for c in 0..self.cells() {
                let old = self.e[c] as f64;
                let new = (old * f) as f32;
                removed += old - new as f64;
                self.e[c] = new;
                for v in &mut self.m[c * CHANNELS..(c + 1) * CHANNELS] {
                    *v = (*v as f64 * f) as f32;
                }
            }
        }
        (removed, sub)
    }

    fn diffuse_once(&mut self, rx: f64, ry: f64, out: &mut Field) {
        debug_assert!(rx + ry <= 0.5 + 1e-12);
        let (gx, gy) = (self.gx, self.gy);
        let center = 1.0 - 2.0 * rx - 2.0 * ry;
        if out.e.len() != self.e.len() {
            *out = Field::new(gx, gy, self.width(), self.height());
        }
        for y in 0..gy {
            let yu = (y + gy - 1) % gy;
            let yd = (y + 1) % gy;
            for x in 0..gx {
                let xl = (x + gx - 1) % gx;
                let xr = (x + 1) % gx;
                let c = y * gx + x;
                let (l, r, u, d) = (y * gx + xl, y * gx + xr, yu * gx + x, yd * gx + x);
                out.e[c] = (center * self.e[c] as f64 + rx * (self.e[l] as f64 + self.e[r] as f64) + ry * (self.e[u] as f64 + self.e[d] as f64)) as f32;
                for k in 0..CHANNELS {
                    let g = |i: usize| self.m[i * CHANNELS + k] as f64;
                    out.m[c * CHANNELS + k] = (center * g(c) + rx * (g(l) + g(r)) + ry * (g(u) + g(d))) as f32;
                }
            }
        }
        std::mem::swap(&mut self.e, &mut out.e);
        std::mem::swap(&mut self.m, &mut out.m);
    }

    /// Tiny-reservoir clearing: cells holding at most `threshold` energy (or any
    /// f32-subnormal energy, where relative precision is lost) are emptied, energy
    /// and moment together. Returns the energy removed (charged to dissipation).
    pub fn clear_tiny(&mut self, threshold: f64) -> f64 {
        let mut removed = 0.0;
        for c in 0..self.cells() {
            let e = self.e[c];
            if e != 0.0 && ((e as f64) <= threshold || !e.is_normal()) || (e == 0.0 && self.moment(c).iter().any(|&v| v != 0.0)) {
                removed += e as f64;
                self.e[c] = 0.0;
                self.m[c * CHANNELS..(c + 1) * CHANNELS].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        removed
    }

    /// Largest relative cone violation `||m|| / e - 1` (positive means violated) and min energy.
    pub fn cone_report(&self) -> (f64, f64) {
        let mut worst = f64::NEG_INFINITY;
        let mut min_e = f64::INFINITY;
        for c in 0..self.cells() {
            let e = self.e[c] as f64;
            min_e = min_e.min(e);
            let n = norm(self.moment(c));
            if e > 0.0 {
                worst = worst.max(n / e - 1.0);
            } else if n > 0.0 {
                worst = f64::INFINITY;
            }
        }
        (worst, min_e)
    }

    /// Project rounding-level cone excess back onto `||m|| <= e`. Energy is untouched.
    /// Returns the number of cells adjusted.
    pub fn project_cone(&mut self) -> usize {
        let mut n_adj = 0;
        for c in 0..self.cells() {
            let e = self.e[c] as f64;
            let m = &mut self.m[c * CHANNELS..(c + 1) * CHANNELS];
            let n = norm(m);
            if n > e {
                let f = if n > 0.0 { e / n } else { 0.0 };
                for v in m.iter_mut() {
                    *v = (*v as f64 * f * (1.0 - 1e-7)) as f32;
                }
                n_adj += 1;
            }
        }
        n_adj
    }
}

#[inline]
pub fn norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

/// Minimum-image displacement on a periodic axis of length `l`.
#[inline]
pub fn min_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

#[inline]
pub fn wrap(x: f64, l: f64) -> f64 {
    let r = x.rem_euclid(l);
    if r >= l {
        0.0
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(k: usize) -> [f32; CHANNELS] {
        let mut a = [0.0; CHANNELS];
        a[k] = 1.0;
        a
    }

    #[test]
    fn c3_opposing_signatures_keep_energy() {
        let mut f = Field::new(4, 4, 4.0, 4.0);
        let s = Stencil { n: 1, cells: [5, 0, 0, 0], w: [1.0, 0.0, 0.0, 0.0] };
        let a = unit(3);
        let mut neg = a;
        neg[3] = -1.0;
        f.deposit(&s, 3.0, &a);
        f.deposit(&s, 5.0, &neg);
        assert_eq!(f.e[5], 8.0);
        assert_eq!(f.moment(5)[3], -2.0);
        assert!(norm(f.moment(5)) <= f.e[5] as f64);
    }

    #[test]
    fn c6_diffusion_known_answer_and_cone() {
        let mut f = Field::new(8, 8, 8.0, 8.0);
        let c = 3 * 8 + 3;
        f.e[c] = 1.0;
        f.m[c * CHANNELS] = 0.6;
        f.m[c * CHANNELS + 1] = -0.8;
        let mut scratch = f.clone();
        // r = D dt / h^2 = 0.2 per axis would be r_x + r_y = 0.4 <= 0.5: single step.
        f.diffuse_once(0.2, 0.2, &mut scratch);
        assert!((f.e[c] - 0.2).abs() < 1e-7);
        for n in [c - 1, c + 1, c - 8, c + 8] {
            assert!((f.e[n] - 0.2).abs() < 1e-7);
        }
        assert!((f.total_energy() - 1.0).abs() < 1e-6);
        assert!(f.cone_report().0 <= 1e-6);
    }

    #[test]
    fn diffusion_subcycles_instead_of_negative_center() {
        let mut f = Field::new(8, 8, 8.0, 8.0);
        f.e[0] = 1.0;
        let mut s = f.clone();
        let (_, sub) = f.transport(1.0, 0.0, 1.0, &mut s); // r_x + r_y = 2.0
        assert_eq!(sub, 4);
        assert!(f.e.iter().all(|&v| v >= 0.0));
        assert!((f.total_energy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stencil_weights_sum_to_one_and_merge_on_tiny_grids() {
        let f = Field::new(1, 1, 2.0, 2.0);
        let s = f.stencil(0.3, 1.7);
        assert_eq!(s.n, 1);
        assert!((s.w[0] - 1.0).abs() < 1e-12);
        let f = Field::new(16, 16, 16.0, 16.0);
        for &(x, y) in &[(0.0, 0.0), (15.99, 0.01), (7.25, 3.5), (0.5, 0.5)] {
            let s = f.stencil(x, y);
            assert!((s.w[..s.n].iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(s.w[..s.n].iter().all(|&w| w >= 0.0));
        }
    }

    #[test]
    fn tiny_reservoirs_clear_energy_and_moment_together() {
        let mut f = Field::new(4, 4, 4.0, 4.0);
        f.e[0] = 1e-40; // f32 subnormal: relative precision is gone
        f.m[0] = 9e-41;
        f.e[1] = 1e-13;
        f.m[CHANNELS] = -1e-13;
        f.e[2] = 0.5;
        let removed = f.clear_tiny(1e-12);
        assert!((removed - (1e-40f32 as f64 + 1e-13f32 as f64)).abs() < 1e-20);
        assert_eq!((f.e[0], f.m[0], f.e[1], f.m[CHANNELS]), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(f.e[2], 0.5);
    }

    #[test]
    fn c4_emission_stencil_shares_weights() {
        let mut f = Field::new(8, 8, 8.0, 8.0);
        let s = f.stencil(3.3, 4.8);
        assert_eq!(s.n, 4);
        let a = unit(0);
        f.deposit(&s, 3.0, &a); // 4 units spent at efficiency 0.75
        assert!((f.total_energy() - 3.0).abs() < 1e-6);
        for k in 0..s.n {
            let c = s.cells[k];
            assert!((f.moment(c)[0] - f.e[c]).abs() < 1e-7);
        }
    }
}
