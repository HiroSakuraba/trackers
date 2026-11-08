//! Independent dense MLP controllers (plan §6.2–6.5, A.3).
//!
//! Every organism owns its own parameter vector. Organisms that share a
//! topology share only the kernel and the packed layout:
//! for each layer, row-major `W[out][in]` followed by `b[out]`.
//! Hidden layers use `tanh`; the final affine layer is linear and yields 36
//! raw logits that are decoded only by the output contract.

pub const INPUT_WIDTH: usize = 104;
pub const OUTPUT_WIDTH: usize = 36;
pub const MAX_HIDDEN: usize = 128;

/// `dense-104-36-catalog-v1`.
pub const CATALOG: [&[usize]; 5] = [&[8], &[32], &[64, 32], &[96, 64], &[128, 96]];

pub fn catalog_index(hidden: &[usize]) -> Option<u8> {
    CATALOG.iter().position(|h| *h == hidden).map(|i| i as u8)
}

/// `(fan_in, fan_out)` for each affine layer.
pub fn layer_dims(hidden: &[usize]) -> Vec<(usize, usize)> {
    let mut dims = Vec::with_capacity(hidden.len() + 1);
    let mut prev = INPUT_WIDTH;
    for &h in hidden {
        dims.push((prev, h));
        prev = h;
    }
    dims.push((prev, OUTPUT_WIDTH));
    dims
}

pub fn param_count(hidden: &[usize]) -> usize {
    layer_dims(hidden).iter().map(|&(i, o)| i * o + o).sum()
}

/// Offsets of each layer's weight block (bias block follows at `+ in*out`).
pub fn layer_offsets(hidden: &[usize]) -> Vec<usize> {
    let mut off = 0;
    layer_dims(hidden)
        .iter()
        .map(|&(i, o)| {
            let here = off;
            off += i * o + o;
            here
        })
        .collect()
}

/// Fan-in of the layer that owns parameter index `p` (weights and biases alike).
pub fn fan_in_table(hidden: &[usize]) -> Vec<(usize, usize)> {
    // (end_exclusive, fan_in) per layer.
    let mut end = 0;
    layer_dims(hidden)
        .iter()
        .map(|&(i, o)| {
            end += i * o + o;
            (end, i)
        })
        .collect()
}

#[inline]
fn dot(w: &[f32], x: &[f32]) -> f32 {
    // Eight independent accumulators in a fixed order: deterministic and vectorizable.
    let mut acc = [0.0f32; 8];
    let chunks = w.len() / 8;
    for c in 0..chunks {
        let wb = &w[c * 8..c * 8 + 8];
        let xb = &x[c * 8..c * 8 + 8];
        for k in 0..8 {
            acc[k] += wb[k] * xb[k];
        }
    }
    let mut tail = 0.0f32;
    for k in chunks * 8..w.len() {
        tail += w[k] * x[k];
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// Scalar reference forward pass. `scratch` must hold `2 * MAX_HIDDEN` floats.
pub fn forward(hidden: &[usize], params: &[f32], input: &[f32; INPUT_WIDTH], scratch: &mut [f32], out: &mut [f32; OUTPUT_WIDTH]) {
    debug_assert_eq!(params.len(), param_count(hidden));
    let (a, b) = scratch.split_at_mut(MAX_HIDDEN);
    let mut cur: &mut [f32] = a;
    let mut nxt: &mut [f32] = b;
    let mut off = 0;
    let mut in_w = INPUT_WIDTH;
    for (li, &h) in hidden.iter().enumerate() {
        let w = &params[off..off + in_w * h];
        let bias = &params[off + in_w * h..off + in_w * h + h];
        for o in 0..h {
            let x: &[f32] = if li == 0 { &input[..] } else { &cur[..in_w] };
            nxt[o] = (bias[o] + dot(&w[o * in_w..(o + 1) * in_w], x)).tanh();
        }
        off += in_w * h + h;
        in_w = h;
        std::mem::swap(&mut cur, &mut nxt);
    }
    let w = &params[off..off + in_w * OUTPUT_WIDTH];
    let bias = &params[off + in_w * OUTPUT_WIDTH..];
    for o in 0..OUTPUT_WIDTH {
        out[o] = bias[o] + dot(&w[o * in_w..(o + 1) * in_w], &cur[..in_w]);
    }
}

/// Sign-stable logistic sigmoid.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `overlap-plus-zero-output-v1` / `prefix-and-identity-bridge-v1` parameter map from
/// `old` to `new` topology. `init(layer, fan_in, flat_index)` supplies initializer values
/// for added incoming connections; outgoing influence of added neurons is zero and takes
/// precedence. Biases of added neurons are zero; overlapping biases are copied.
pub fn map_topology(old_h: &[usize], old: &[f32], new_h: &[usize], mut init: impl FnMut(usize, usize, usize) -> f32) -> Vec<f32> {
    let od = layer_dims(old_h);
    let nd = layer_dims(new_h);
    let oo = layer_offsets(old_h);
    let no = layer_offsets(new_h);
    let mut p = vec![0.0f32; param_count(new_h)];
    let ow = |l: usize, o: usize, i: usize| old[oo[l] + o * od[l].0 + i];
    let ob = |l: usize, o: usize| old[oo[l] + od[l].0 * od[l].1 + o];
    let n_layers = nd.len();
    let old_last = od.len() - 1;

    if old_h.len() == new_h.len() {
        // Equal depth: copy the leading overlapping subnetwork.
        for l in 0..n_layers {
            let (ni, nout) = nd[l];
            let (oi, oout) = od[l];
            for o in 0..nout {
                for i in 0..ni {
                    let added_input = l > 0 && i >= oi;
                    let v = if added_input {
                        0.0
                    } else if o >= oout {
                        init(l, ni, no[l] + o * ni + i)
                    } else {
                        ow(l, o, i)
                    };
                    p[no[l] + o * ni + i] = v;
                }
                p[no[l] + ni * nout + o] = if o < oout { ob(l, o) } else { 0.0 };
            }
        }
        return p;
    }

    // Depth change. First hidden affine map: copy overlap, initialize added rows.
    let (ni0, nout0) = nd[0];
    let oout0 = od[0].1;
    for o in 0..nout0 {
        for i in 0..ni0 {
            p[no[0] + o * ni0 + i] = if o < oout0 { ow(0, o, i) } else { init(0, ni0, no[0] + o * ni0 + i) };
        }
        p[no[0] + ni0 * nout0 + o] = if o < oout0 { ob(0, o) } else { 0.0 };
    }
    let old_h0 = old_h[0];
    if old_h.len() == 1 && new_h.len() == 2 {
        // Insert a rectangular identity bridge (zero bias) between hidden layers.
        let (a, b) = (new_h[0], new_h[1]);
        for j in 0..b {
            for i in 0..a {
                // Outgoing-zeroing: added first-layer neurons (i >= old_h0) feed nothing.
                p[no[1] + j * a + i] = if i == j && i < old_h0 { 1.0 } else { 0.0 };
            }
        }
        // Output: copy leading old output columns through the bridge, zero unmatched.
        for o in 0..OUTPUT_WIDTH {
            for j in 0..b {
                p[no[2] + o * b + j] = if j < old_h0 && j < a { ow(old_last, o, j) } else { 0.0 };
            }
            p[no[2] + b * OUTPUT_WIDTH + o] = ob(old_last, o);
        }
    } else if old_h.len() == 2 && new_h.len() == 1 {
        // Discard the middle layer; copy leading rows of the old output map.
        let h = new_h[0];
        let old_b = old_h[1];
        for o in 0..OUTPUT_WIDTH {
            for j in 0..h {
                p[no[1] + o * h + j] = if j < old_b && j < old_h0 { ow(old_last, o, j) } else { 0.0 };
            }
            p[no[1] + h * OUTPUT_WIDTH + o] = ob(old_last, o);
        }
    } else {
        unreachable!("catalog depths are 1 or 2");
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parameter_counts() {
        let counts: Vec<usize> = CATALOG.iter().map(|h| param_count(h)).collect();
        assert_eq!(counts, vec![1_164, 4_548, 9_988, 18_628, 29_316]);
    }

    fn rand_params(h: &[usize], seed: u32) -> Vec<f32> {
        let mut s = seed as u64 | 1;
        (0..param_count(h))
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
            })
            .collect()
    }

    fn input(seed: u32) -> [f32; INPUT_WIDTH] {
        let mut x = [0.0; INPUT_WIDTH];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (((i as u32).wrapping_mul(2654435761u32) ^ seed) % 1000) as f32 / 500.0 - 1.0;
        }
        x
    }

    #[test]
    fn c11_final_layer_is_linear_and_sigmoid_applies_once() {
        let h = [8usize];
        let mut p = vec![0.0f32; param_count(&h)];
        let n = p.len();
        p[n - OUTPUT_WIDTH] = 8.0; // output bias 0
        let mut out = [0.0; OUTPUT_WIDTH];
        let mut scratch = vec![0.0; 2 * MAX_HIDDEN];
        forward(&h, &p, &input(1), &mut scratch, &mut out);
        assert_eq!(out[0], 8.0);
        assert!((sigmoid(out[0] as f64) - 0.99966465).abs() < 1e-8);
        assert!((sigmoid((out[0] as f64).tanh()) - 0.73105853).abs() < 1e-6);
        assert!(sigmoid(-800.0) >= 0.0 && sigmoid(-800.0).is_finite());
    }

    #[test]
    fn c16_equal_depth_widening_preserves_function() {
        let mut scratch = vec![0.0; 2 * MAX_HIDDEN];
        for (a, b) in [(0usize, 1usize), (2, 3), (3, 4), (2, 4)] {
            let (ha, hb) = (CATALOG[a], CATALOG[b]);
            let p = rand_params(ha, 7);
            let q = map_topology(ha, &p, hb, |_, fi, k| ((k % 13) as f32 - 6.0) / (fi as f32));
            for s in 0..5 {
                let x = input(s);
                let (mut o1, mut o2) = ([0.0; OUTPUT_WIDTH], [0.0; OUTPUT_WIDTH]);
                forward(ha, &p, &x, &mut scratch, &mut o1);
                forward(hb, &q, &x, &mut scratch, &mut o2);
                for k in 0..OUTPUT_WIDTH {
                    assert!((o1[k] - o2[k]).abs() < 1e-5, "{a}->{b}: {} vs {}", o1[k], o2[k]);
                }
            }
        }
    }

    #[test]
    fn c16_every_directed_transition_maps_deterministically() {
        let mut scratch = vec![0.0; 2 * MAX_HIDDEN];
        for a in 0..CATALOG.len() {
            for b in 0..CATALOG.len() {
                if a == b {
                    continue;
                }
                let p = rand_params(CATALOG[a], 3);
                let q1 = map_topology(CATALOG[a], &p, CATALOG[b], |_, _, _| 0.01);
                let q2 = map_topology(CATALOG[a], &p, CATALOG[b], |_, _, _| 0.01);
                assert_eq!(q1, q2);
                assert_eq!(q1.len(), param_count(CATALOG[b]));
                let mut o = [0.0; OUTPUT_WIDTH];
                forward(CATALOG[b], &q1, &input(9), &mut scratch, &mut o);
                assert!(o.iter().all(|v| v.is_finite()));
                // Output biases are copied in every direction.
                let ob = &p[p.len() - OUTPUT_WIDTH..];
                let nb = &q1[q1.len() - OUTPUT_WIDTH..];
                assert_eq!(ob, nb);
            }
        }
        // An identity tanh bridge is not function preserving.
        assert!((1.0f64.tanh() - 0.76159416).abs() < 1e-8);
        assert!((1.0f64.tanh().tanh() - 0.64201499).abs() < 1e-8);
    }
}
