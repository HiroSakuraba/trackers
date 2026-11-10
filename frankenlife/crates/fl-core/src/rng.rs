//! `fl-philox-semantic-v1`: counter-based randomness addressed by meaning.
//!
//! Every draw is a pure function of `(seed, lane, a, b, c, d)`. Lanes name the
//! purpose (reproduction attempt, mutation parameter selection, ...) and the
//! address fields carry stable identities and ticks, never storage slots or
//! scheduling order. The 5-field address is folded into a 128-bit Philox
//! counter with SplitMix64 finalizers; the key comes from the run seed.
//! Retrying a tick repeats its draws; a later tick receives fresh ones.

/// Purpose lanes. Values are frozen: changing one changes model identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Lane {
    InitPosition = 1,
    InitWeights = 2,
    InitAffinity = 3,
    InitMotion = 4,
    SourceCenter = 10,
    SourceChemistry = 11,
    SourcePathDirection = 12,
    /// Random perpendicular direction of a live chemistry drift command.
    SourceChemistryDrift = 13,
    ReproductionAttempt = 20,
    ChildOffset = 21,
    MutationTopology = 30,
    MutationParamSelect = 31,
    MutationParamValue = 32,
    MutationAffinity = 33,
    MutationMotion = 34,
    MutationSensing = 35,
    MutationNewNeuron = 36,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticRng {
    key: [u32; 2],
}

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = a as u64 * b as u64;
    ((p >> 32) as u32, p as u32)
}

/// Philox4x32 with 10 rounds (Salmon et al., SC'11).
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo(M0, ctr[0]);
        let (hi1, lo1) = mulhilo(M1, ctr[2]);
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    ctr
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl SemanticRng {
    pub fn new(seed: u64) -> Self {
        let k = splitmix(seed ^ 0x4652_414E_4B45_4E4C); // "FRANKENL"
        SemanticRng { key: [k as u32, (k >> 32) as u32] }
    }

    /// Two raw 64-bit words for one semantic address.
    #[inline]
    pub fn words(&self, lane: Lane, a: u64, b: u64, c: u64, d: u64) -> [u64; 2] {
        let h0 = splitmix(splitmix((lane as u64) << 32 ^ a) ^ b);
        let h1 = splitmix(splitmix(c ^ 0xA5A5_5A5A_0F0F_F0F0) ^ d.rotate_left(17) ^ (lane as u64));
        let out = philox4x32_10([h0 as u32, (h0 >> 32) as u32, h1 as u32, (h1 >> 32) as u32], self.key);
        [
            out[0] as u64 | (out[1] as u64) << 32,
            out[2] as u64 | (out[3] as u64) << 32,
        ]
    }

    /// Two uniforms in the open interval (0, 1).
    #[inline]
    pub fn uniform2(&self, lane: Lane, a: u64, b: u64, c: u64, d: u64) -> [f64; 2] {
        let w = self.words(lane, a, b, c, d);
        [unit_open(w[0]), unit_open(w[1])]
    }

    #[inline]
    pub fn uniform(&self, lane: Lane, a: u64, b: u64, c: u64, d: u64) -> f64 {
        unit_open(self.words(lane, a, b, c, d)[0])
    }

    /// Two independent standard normals (Box–Muller, open first uniform, no cached spare).
    #[inline]
    pub fn normal2(&self, lane: Lane, a: u64, b: u64, c: u64, d: u64) -> [f64; 2] {
        let w = self.words(lane, a, b, c, d);
        let u1 = unit_open(w[0]);
        let u2 = unit_half_open(w[1]);
        let r = (-2.0 * u1.ln()).sqrt();
        let t = std::f64::consts::TAU * u2;
        [r * t.cos(), r * t.sin()]
    }

    /// Fill `out` with standard normals drawn at consecutive `d` addresses.
    pub fn normals(&self, lane: Lane, a: u64, b: u64, c: u64, out: &mut [f64]) {
        for (pair, chunk) in out.chunks_mut(2).enumerate() {
            let z = self.normal2(lane, a, b, c, pair as u64);
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = z[k];
            }
        }
    }
}

/// Leading 52 bits `k` mapped to `(k + 0.5) 2^-52`.
#[inline]
pub fn unit_open(w: u64) -> f64 {
    ((w >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64)
}

/// Leading 52 bits `k` mapped to `k 2^-52`.
#[inline]
pub fn unit_half_open(w: u64) -> f64 {
    (w >> 12) as f64 * (1.0 / (1u64 << 52) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_known_answer() {
        // Random123 known-answer vectors for philox4x32-10.
        assert_eq!(philox4x32_10([0, 0, 0, 0], [0, 0]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(
            philox4x32_10([0xffffffff; 4], [0xffffffff; 2]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
        );
        assert_eq!(
            philox4x32_10([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0]),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
        );
    }

    #[test]
    fn unit_mappings_hit_bounds() {
        assert!(unit_open(0) > 0.0);
        assert!(unit_open(u64::MAX) < 1.0);
        assert_eq!(unit_half_open(0), 0.0);
        assert!(unit_half_open(u64::MAX) < 1.0);
    }

    #[test]
    fn addresses_are_pure_and_distinct() {
        let r = SemanticRng::new(42);
        let a = r.words(Lane::ReproductionAttempt, 7, 100, 0, 0);
        assert_eq!(a, r.words(Lane::ReproductionAttempt, 7, 100, 0, 0));
        assert_ne!(a, r.words(Lane::ReproductionAttempt, 7, 101, 0, 0));
        assert_ne!(a, r.words(Lane::ChildOffset, 7, 100, 0, 0));
        assert_ne!(a, SemanticRng::new(43).words(Lane::ReproductionAttempt, 7, 100, 0, 0));
    }

    #[test]
    fn normals_have_unit_moments() {
        let r = SemanticRng::new(1);
        let mut v = vec![0.0; 200_000];
        r.normals(Lane::MutationMotion, 1, 2, 3, &mut v);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
    }
}
