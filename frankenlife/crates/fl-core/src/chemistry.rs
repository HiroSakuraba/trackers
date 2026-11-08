//! Pure exchange arithmetic shared by the tick and the known-answer fixtures
//! (plan §5.2–5.5, §7.4, A.1).

/// Two-pass demand/grant: `alpha_c = min(1, e_c / A_c)`.
#[inline]
pub fn grant_scale(available: f64, demand: f64) -> f64 {
    if demand > 0.0 {
        (available / demand).min(1.0)
    } else {
        0.0
    }
}

/// Bounded affinity chemistry `eta(z) = eta_max * tanh(k a.z - b)`.
#[inline]
pub fn eta(eta_max: f64, gain: f64, threshold: f64, a_dot_z: f64) -> f64 {
    eta_max * (gain * a_dot_z - threshold).tanh()
}

/// Settlement of an organism's granted packets.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Settlement {
    pub absorbed: f64,
    pub useful_gain: f64,
    pub toxicity_requested: f64,
    pub toxicity_paid: f64,
}

impl Settlement {
    /// Net internal energy change.
    pub fn net(&self) -> f64 {
        self.useful_gain - self.toxicity_paid
    }
    /// Energy dissipated by uptake inefficiency and toxicity.
    pub fn dissipated(&self) -> f64 {
        self.absorbed - self.useful_gain + self.toxicity_paid
    }
}

/// Apply §5.3 per packet: `packets` yields `(granted u, eta)`.
pub fn settle(energy: f64, kappa: f64, packets: impl IntoIterator<Item = (f64, f64)>) -> Settlement {
    let mut s = Settlement::default();
    for (u, e) in packets {
        s.absorbed += u;
        s.useful_gain += e.max(0.0) * u;
        s.toxicity_requested += kappa * (-e).max(0.0) * u;
    }
    s.toxicity_paid = s.toxicity_requested.min(energy + s.useful_gain);
    s
}

/// Drag velocity coefficient `-expm1(-gamma dt) / gamma` (`dt` at `gamma = 0`).
#[inline]
pub fn drag_force_coefficient(gamma: f64, dt: f64) -> f64 {
    if gamma == 0.0 {
        dt
    } else {
        -(-gamma * dt).exp_m1() / gamma
    }
}

/// Probability of at least one event in a constant-rate interval.
#[inline]
pub fn rate_to_probability(rate: f64, dt: f64) -> f64 {
    -(-rate * dt).exp_m1()
}

/// `(child, parent_after)` energies for available `e`, birth cost `cb`, share `beta`.
#[inline]
pub fn split_energy(e: f64, cb: f64, beta: f64) -> (f64, f64) {
    let rest = e - cb;
    let child = beta * rest;
    (child, rest - child)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c1_contended_uptake() {
        let e = 10.0;
        let (r1, r2) = (8.0, 12.0);
        let alpha = grant_scale(e, r1 + r2);
        let (u1, u2) = (alpha * r1, alpha * r2);
        assert_eq!((u1, u2), (4.0, 6.0));
        let s1 = settle(1.0, 0.2, [(u1, 0.5)]);
        let s2 = settle(1.0, 0.2, [(u2, 0.25)]);
        assert_eq!((s1.net(), s2.net()), (2.0, 1.5));
        assert_eq!(s1.dissipated() + s2.dissipated(), 6.5);
        // Field loses 10; particles gain 3.5; global change = -6.5 = -dissipation.
        assert_eq!(-(u1 + u2) + s1.net() + s2.net(), -6.5);
    }

    #[test]
    fn c2_toxicity_clipping() {
        // Energy 2 absorbs 3 units with zero useful gain and requested toxicity 5.
        let kappa = 5.0 / 3.0;
        let s = settle(2.0, kappa, [(3.0, -1.0)]);
        assert!((s.toxicity_requested - 5.0).abs() < 1e-12);
        assert_eq!(s.toxicity_paid, 2.0);
        assert_eq!(s.net(), -2.0);
        assert_eq!(s.dissipated(), 5.0);
    }

    #[test]
    fn c9_packet_chemistry_is_not_mixture_chemistry() {
        let f = |az: f64| eta(0.8, 4.0, 0.15, az);
        let s = settle(10.0, 0.2, [(1.0, f(1.0)), (1.0, f(-1.0))]);
        assert!((s.net() - 0.639355310069).abs() < 1e-11, "{}", s.net());
        let mix = settle(10.0, 0.2, [(2.0, f(0.0))]);
        assert!((mix.net() - -0.047643210759).abs() < 1e-11, "{}", mix.net());
    }

    #[test]
    fn c5_birth_split() {
        let (c, p) = split_energy(10.0, 1.0, 1.0 / 3.0);
        assert!((c - 3.0).abs() < 1e-12 && (p - 6.0).abs() < 1e-12);
        assert!((10.0 - c - p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn c11_small_rate_arithmetic() {
        let k = drag_force_coefficient(1e-6, 1.0 / 128.0);
        assert!((k - 0.00781249996948).abs() < 1e-14, "{k}");
        let naive = (1.0f32 - (-1e-6f32 / 128.0).exp()) / 1e-6;
        assert!((naive as f64 - 0.0078125).abs() > 1e-9 || naive == 0.0, "naive f32 loses precision");
        assert_eq!(rate_to_probability(0.0, 1.0 / 128.0), 0.0);
        assert!(rate_to_probability(1e-12, 1.0 / 128.0) > 0.0);
        assert_eq!(drag_force_coefficient(0.0, 0.25), 0.25);
    }
}
