//! Declared display mappings shared by every FrankenLife view.
//!
//! These functions only *read* committed state. Colors are pure functions of
//! stable identities or genetic/field values — never per-frame randomness —
//! so the same organism has the same color in every frame, run and viewer.

#![forbid(unsafe_code)]

use fl_core::field::Field;

/// Teal, violet, magenta, pale cyan, amber, lime (plan §2.3).
pub const PALETTE: [[f32; 3]; 6] = [
    [0.18, 0.83, 0.75],
    [0.55, 0.36, 0.96],
    [0.91, 0.47, 0.98],
    [0.65, 0.95, 0.99],
    [0.98, 0.75, 0.14],
    [0.64, 0.90, 0.21],
];

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `founder-lineage-v1`: palette entry by lineage hash with a small stable tint.
pub fn lineage_color(lineage: u64) -> [f32; 3] {
    let h = mix64(lineage.wrapping_add(0x9E37_79B9_7F4A_7C15));
    let a = PALETTE[(h % 6) as usize];
    let b = PALETTE[((h >> 8) % 6) as usize];
    let t = ((h >> 16) & 0xff) as f32 / 255.0 * 0.35;
    [a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t]
}

/// Fixed projection basis of the 32-D chemical space onto RGB.
pub fn chem_basis() -> [[f32; 32]; 3] {
    let mut b = [[0.0f32; 32]; 3];
    for (r, row) in b.iter_mut().enumerate() {
        let mut n = 0.0f32;
        for (k, v) in row.iter_mut().enumerate() {
            let h = mix64((r as u64) << 32 | k as u64);
            *v = (h >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
            n += *v * *v;
        }
        let n = n.sqrt();
        row.iter_mut().for_each(|v| *v /= n);
    }
    b
}

/// `chemistry-projection-v1`: a 32-D chemical vector mapped to a luminous RGB.
pub fn chem_color(v: &[f32], basis: &[[f32; 32]; 3]) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for r in 0..3 {
        c[r] = basis[r].iter().zip(v).map(|(a, b)| a * b).sum();
    }
    let m = c.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-6);
    let base = [0.25, 0.35, 0.55];
    [
        (base[0] + 0.75 * (c[0] / m).max(-0.3)).clamp(0.0, 1.0),
        (base[1] + 0.65 * (c[1] / m).max(-0.3)).clamp(0.0, 1.0),
        (base[2] + 0.60 * (c[2] / m).max(-0.3)).clamp(0.0, 1.0),
    ]
}

/// Linear RGB per field cell. `raw` = grey energy density only (diagnostic).
pub fn field_cell_rgb(field: &Field, basis: &[[f32; 32]; 3], raw: bool) -> Vec<f32> {
    let area = (field.hx * field.hy) as f32;
    let cells = field.cells();
    let mut out = vec![0.0f32; cells * 3];
    for c in 0..cells {
        let e = field.e[c];
        let r = e / area;
        if raw {
            let v = 1.0 - (-r * 0.5).exp();
            out[c * 3..c * 3 + 3].copy_from_slice(&[v, v, v]);
            continue;
        }
        if e <= 0.0 {
            continue;
        }
        let glow = (1.0 - (-r * 0.35).exp()) * 0.22;
        let z: Vec<f32> = field.moment(c).iter().map(|m| m / e).collect();
        let col = chem_color(&z, basis);
        for k in 0..3 {
            out[c * 3 + k] = col[k] * glow;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_are_stable_and_in_range() {
        assert_eq!(lineage_color(17), lineage_color(17));
        assert_ne!(lineage_color(17), lineage_color(18));
        let b = chem_basis();
        let c = chem_color(&[0.3; 32], &b);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
