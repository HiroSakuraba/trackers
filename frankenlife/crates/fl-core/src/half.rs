//! IEEE 754 binary16 ("f16") conversion, written out so the core needs no extra crate and the
//! rounding (round to nearest, ties to even) is the same everywhere: CPU reference, checkpoints
//! and the GPU kernels, which unpack with WGSL's `unpack2x16float`.

/// Round an `f32` to the nearest `f16` (ties to even). Out-of-range values become infinity,
/// NaN stays NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return if man == 0 { sign | 0x7c00 } else { sign | 0x7e00 | (man >> 13) as u16 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or underflow to zero): shift the full significand into the 10 fraction bits.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut r = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // a carry into the exponent is correct, including up to infinity
    }
    sign | r as u16
}

/// Exact widening of an `f16` to `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// `x` rounded to the nearest value an `f16` can hold, as an `f32`.
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "rounds up to infinity");
        assert_eq!(f32_to_f16(1.0e9), 0x7c00);
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001, "smallest subnormal");
        assert_eq!(f32_to_f16(1.0e-9), 0x0000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        // Halfway between 1.0 and the next f16 (1 + 2^-10) rounds to even, i.e. down; just above rounds up.
        assert_eq!(f32_to_f16(1.0 + 0.5 * 2f32.powi(-10)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 0.5 * 2f32.powi(-10) + 1e-7), 0x3c01);
        assert_eq!(f32_to_f16(1.0 + 1.5 * 2f32.powi(-10)), 0x3c02, "tie rounds to the even neighbour");
    }

    #[test]
    fn every_f16_survives_a_round_trip() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
            }
        }
    }

    #[test]
    fn rounding_is_to_the_nearest_neighbour() {
        let mut s = 0x1234_5678u32;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            // Spread over many binades: sign/exponent/mantissa straight from the random bits.
            let x = f32::from_bits(s);
            if !x.is_finite() || x.abs() > 65000.0 {
                continue;
            }
            let h = f32_to_f16(x);
            let r = f16_to_f32(h) as f64;
            let err = (r - x as f64).abs();
            for nb in [h.wrapping_sub(1), h.wrapping_add(1)] {
                let n = f16_to_f32(nb);
                if n.is_finite() {
                    assert!(err <= (n as f64 - x as f64).abs() + 1e-30, "{x} -> {r}, neighbour {n} is closer");
                }
            }
        }
    }
}
