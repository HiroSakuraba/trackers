// Fused sensing + independent-network inference: one workgroup per organism.
//
// 1. Sample R and the 32 moment densities at the organism's sensing scale
//    (bilinear values and analytic derivatives at two pyramid levels, blended
//    log-linearly *before* normalization), one plane per thread.
// 2. Normalize into the 104-wide input vector in workgroup memory (plan §6.1
//    ordering) and compute the motion drive: the genome's 33 resident motion
//    sensitivities dotted with the physical gradients of [q, Z_0..Z_31].
// 3. Run the organism's own MLP from workgroup memory (as mlp.wgsl).
//
// Outputs per job (stride 56): 36 logits, motion drive x, y, level-0 density, pad, 16 memory.
// The MLP runs for every job; logits of jobs that were not due are ignored on
// the CPU (barriers cannot sit under data-dependent branches).

struct SenseDims {
    a: vec4<u32>,                // n_jobs, n_layers, d0 (104), d1
    b: vec4<u32>,                // d2, d3, grid_x, n_levels
    c: vec4<f32>,                // s0, composition floor density, energy density scale, gradient length
    lu: array<vec4<u32>, 8>,     // per level: gx, gy, offset (cells), 0 (level 0: first row of the group)
    lf: array<vec4<f32>, 8>,     // per level: hx, hy, 0, 0
}

@group(0) @binding(0) var<uniform> dims: SenseDims;
@group(0) @binding(1) var<storage, read> jobs: array<f32>;
@group(0) @binding(2) var<storage, read> pyr: array<f32>;
@group(0) @binding(3) var<storage, read> traits: array<f32>;
@group(0) @binding(4) var<storage, read> w0: array<f32>;
// The same binding read as packed 16-bit floats (two per word) by `main_deep_h`.
@group(0) @binding(4) var<storage, read> w0h: array<u32>;
@group(0) @binding(5) var<storage, read> w1: array<f32>;
@group(0) @binding(6) var<storage, read> w2: array<f32>;
@group(0) @binding(7) var<storage, read_write> outputs: array<f32>;

const WG: u32 = 64u;
const PL: u32 = 33u;
// Row strides come from the uniform (lu[1].w and lu[2].w): 12 and 40 without memory, 28 and 56 with it.
// Job row: slot bits, x, y, scale, due, 5 intrinsic, 2 pad, 16 memory.
// Output row: 36 logits, drive x/y, density, pad, 16 memory (+40).
const TS: u32 = 36u;   // traits stride: 33 motion sensitivities + pad
const OUT_W: u32 = 36u;

var<workgroup> a: array<f32, 128>;
var<workgroup> b: array<f32, 128>;
var<workgroup> rv: array<f32, 33>;
var<workgroup> rdx: array<f32, 33>;
var<workgroup> rdy: array<f32, 33>;
var<workgroup> gqx: array<f32, 33>;
var<workgroup> gqy: array<f32, 33>;
var<workgroup> due_wg: u32;

fn wrap_index(i: i32, n: u32) -> u32 {
    let m = i32(n);
    return u32(((i % m) + m) % m);
}

// Bilinear value and analytic x/y derivatives of plane k at pyramid level l.
fn sample(l: u32, x: f32, y: f32, k: u32) -> vec3<f32> {
    let gx = dims.lu[l].x;
    let gy = dims.lu[l].y;
    let off = dims.lu[l].z;
    let hx = dims.lf[l].x;
    let hy = dims.lf[l].y;
    let fx = x / hx - 0.5;
    let fy = y / hy - 0.5;
    let ixf = floor(fx);
    let iyf = floor(fy);
    let tx = fx - ixf;
    let ty = fy - iyf;
    let x0 = wrap_index(i32(ixf), gx);
    let y0 = wrap_index(i32(iyf), gy);
    let x1 = (x0 + 1u) % gx;
    let y1 = (y0 + 1u) % gy;
    let v00 = pyr[(off + y0 * gx + x0) * PL + k];
    let v10 = pyr[(off + y0 * gx + x1) * PL + k];
    let v01 = pyr[(off + y1 * gx + x0) * PL + k];
    let v11 = pyr[(off + y1 * gx + x1) * PL + k];
    let v = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11;
    let dx = ((1.0 - ty) * (v10 - v00) + ty * (v11 - v01)) / hx;
    let dy = ((1.0 - tx) * (v01 - v00) + tx * (v11 - v01)) / hy;
    return vec3<f32>(v, dx, dy);
}

// Steps 1 and 2, shared by the shallow and the recurrent kernel: sample the field, build
// the 104 normalized inputs in `a`, and write the motion drive and density of this job.
fn prepare(t: u32, job: u32, jb: u32) {
    let slot = bitcast<u32>(jobs[jb]);
    let x = jobs[jb + 1u];
    let y = jobs[jb + 2u];
    let scale = jobs[jb + 3u];
    let top = dims.b.w - 1u;
    let lgs = clamp(log2(scale / dims.c.x), 0.0, f32(top));
    var l = u32(floor(lgs));
    if (l >= top) {
        l = top - 1u;
    }
    let tb = clamp(lgs - f32(l), 0.0, 1.0);

    // 1. Raw blended samples, one plane per thread.
    if (t < PL) {
        var s = sample(l, x, y, t);
        if (tb > 0.0) {
            s = (1.0 - tb) * s + tb * sample(l + 1u, x, y, t);
        }
        rv[t] = s.x;
        rdx[t] = s.y;
        rdy[t] = s.z;
    }
    workgroupBarrier();

    // 2. Normalized inputs and physical gradients.
    let rf = dims.c.y;
    let rs = dims.c.z;
    let lgl = dims.c.w;
    let r = max(rv[0], 0.0);
    let den = r + rf;
    if (t < PL) {
        var val: f32;
        var gxv: f32;
        var gyv: f32;
        if (t == 0u) {
            let dq = rs / ((r + rs) * (r + rs));
            val = r / (r + rs);
            gxv = dq * rdx[0];
            gyv = dq * rdy[0];
        } else {
            let z = rv[t] / den;
            val = z;
            gxv = (rdx[t] - z * rdx[0]) / den;
            gyv = (rdy[t] - z * rdy[0]) / den;
        }
        a[t] = val;
        a[PL + t] = tanh(lgl * gxv);
        a[2u * PL + t] = tanh(lgl * gyv);
        gqx[t] = gxv;
        gqy[t] = gyv;
    } else if (t < PL + 5u) {
        a[99u + (t - PL)] = jobs[jb + 5u + (t - PL)];
    }
    workgroupBarrier();

    if (t == 0u) {
        var fx = 0.0;
        var fy = 0.0;
        for (var k = 0u; k < PL; k++) {
            let m = traits[slot * TS + k];
            fx += m * gqx[k];
            fy += m * gqy[k];
        }
        outputs[job * dims.lu[2].w + 36u] = fx;
        outputs[job * dims.lu[2].w + 37u] = fy;
        outputs[job * dims.lu[2].w + 38u] = max(sample(0u, x, y, 0u).x, 0.0);
    }

}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let local = wg.y * dims.b.z + wg.x;
    if (local >= dims.a.x) {
        return;
    }
    // Row of this group's first job in the job and output buffers (0 unless
    // several groups share one submission).
    let job = dims.lu[0].w + local;
    let jb = job * dims.lu[1].w;
    let slot = bitcast<u32>(jobs[jb]);
    prepare(t, job, jb);

    // 3. The organism's own MLP, from workgroup memory.
    {
        let n_in = dims.a.z;
        let n_out = dims.a.w;
        let base = slot * (n_in * n_out + n_out);
        for (var o = t; o < n_out; o += WG) {
            var acc = w0[base + n_in * n_out + o];
            for (var i = 0u; i < n_in; i++) {
                acc += w0[base + i * n_out + o] * a[i];
            }
            b[o] = tanh(acc);
        }
    }
    workgroupBarrier();

    let last1 = dims.a.y == 2u;
    {
        let n_in = dims.a.w;
        let n_out = dims.b.x;
        let base = slot * (n_in * n_out + n_out);
        for (var o = t; o < n_out; o += WG) {
            var acc = w1[base + n_in * n_out + o];
            for (var i = 0u; i < n_in; i++) {
                acc += w1[base + i * n_out + o] * b[i];
            }
            if (last1) {
                a[o] = acc;
            } else {
                a[o] = tanh(acc);
            }
        }
    }
    workgroupBarrier();

    if (last1) {
        for (var o = t; o < OUT_W; o += WG) {
            outputs[job * dims.lu[2].w + o] = a[o];
        }
        return;
    }

    {
        let n_in = dims.b.x;
        let n_out = dims.b.y;
        let base = slot * (n_in * n_out + n_out);
        for (var o = t; o < n_out; o += WG) {
            var acc = w2[base + n_in * n_out + o];
            for (var i = 0u; i < n_in; i++) {
                acc += w2[base + i * n_out + o] * a[i];
            }
            outputs[job * dims.lu[2].w + o] = acc;
        }
    }
}

// The 8-layer recurrent network. All nine affine layers of an organism sit back to back
// in one packed buffer (w0), each transposed to [in][out] with its biases after it.
// Inputs are the 104 sensor/intrinsic values plus 16 memory values; outputs are 36 logits
// plus 16 raw memory values, which leave as tanh (the next memory) at output offset 40.
const D_TOTAL: u32 = 12980u;   // floats per organism: 3872 + 7 * 1056 + 1716
const D_L1: u32 = 3872u;       // first 32 -> 32 layer
const D_HB: u32 = 1056u;       // one 32 -> 32 layer: 32*32 + 32
const D_LAST: u32 = 11264u;    // output layer, 32 -> 52
const D_H: u32 = 32u;
const D_OUTW: u32 = 52u;
const D_IN0: u32 = 120u;       // 104 + 16

@compute @workgroup_size(64)
fn main_deep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let local = wg.y * dims.b.z + wg.x;
    if (local >= dims.a.x) {
        return;
    }
    let job = dims.lu[0].w + local;
    let jb = job * dims.lu[1].w;
    let slot = bitcast<u32>(jobs[jb]);
    prepare(t, job, jb);
    if (t < 16u) {
        a[104u + t] = jobs[jb + 12u + t];
    }
    if (t == 0u) {
        due_wg = select(0u, 1u, jobs[jb + 4u] > 0.5);
    }
    workgroupBarrier();
    // A job that is not due keeps its sensing outputs (motion drive, density) but skips the network:
    // the whole group agrees, so the barriers below stay uniform.
    if (workgroupUniformLoad(&due_wg) == 0u) {
        return;
    }

    // Each thread owns one output column and walks the layer's inputs in order (bias first, then
    // input 0, 1, 2, ...). Eight weights are read at a time so the loads overlap instead of
    // waiting one after another; the additions keep their order, so the answers are unchanged.
    let base = slot * D_TOTAL;
    if (t < D_H) {
            var acc = w0[base + D_IN0 * D_H + t];
            for (var i = 0u; i < 120u; i += 8u) {
                let p0 = w0[base + (i + 0u) * D_H + t];
                let p1 = w0[base + (i + 1u) * D_H + t];
                let p2 = w0[base + (i + 2u) * D_H + t];
                let p3 = w0[base + (i + 3u) * D_H + t];
                let p4 = w0[base + (i + 4u) * D_H + t];
                let p5 = w0[base + (i + 5u) * D_H + t];
                let p6 = w0[base + (i + 6u) * D_H + t];
                let p7 = w0[base + (i + 7u) * D_H + t];
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
        b[t] = tanh(acc);
    }
    workgroupBarrier();
    for (var l = 1u; l < 8u; l++) {
        let bb = base + D_L1 + (l - 1u) * D_HB;
        if (t < D_H) {
            if ((l & 1u) == 1u) {
            var acc = w0[bb + D_H * D_H + t];
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = w0[bb + (i + 0u) * D_H + t];
                let p1 = w0[bb + (i + 1u) * D_H + t];
                let p2 = w0[bb + (i + 2u) * D_H + t];
                let p3 = w0[bb + (i + 3u) * D_H + t];
                let p4 = w0[bb + (i + 4u) * D_H + t];
                let p5 = w0[bb + (i + 5u) * D_H + t];
                let p6 = w0[bb + (i + 6u) * D_H + t];
                let p7 = w0[bb + (i + 7u) * D_H + t];
                acc += p0 * b[i + 0u];
                acc += p1 * b[i + 1u];
                acc += p2 * b[i + 2u];
                acc += p3 * b[i + 3u];
                acc += p4 * b[i + 4u];
                acc += p5 * b[i + 5u];
                acc += p6 * b[i + 6u];
                acc += p7 * b[i + 7u];
            }
                a[t] = tanh(acc);
            } else {
            var acc = w0[bb + D_H * D_H + t];
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = w0[bb + (i + 0u) * D_H + t];
                let p1 = w0[bb + (i + 1u) * D_H + t];
                let p2 = w0[bb + (i + 2u) * D_H + t];
                let p3 = w0[bb + (i + 3u) * D_H + t];
                let p4 = w0[bb + (i + 4u) * D_H + t];
                let p5 = w0[bb + (i + 5u) * D_H + t];
                let p6 = w0[bb + (i + 6u) * D_H + t];
                let p7 = w0[bb + (i + 7u) * D_H + t];
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
                b[t] = tanh(acc);
            }
        }
        workgroupBarrier();
    }
    if (t < D_OUTW) {
        let bb = base + D_LAST;
            var acc = w0[bb + D_H * D_OUTW + t];
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = w0[bb + (i + 0u) * D_OUTW + t];
                let p1 = w0[bb + (i + 1u) * D_OUTW + t];
                let p2 = w0[bb + (i + 2u) * D_OUTW + t];
                let p3 = w0[bb + (i + 3u) * D_OUTW + t];
                let p4 = w0[bb + (i + 4u) * D_OUTW + t];
                let p5 = w0[bb + (i + 5u) * D_OUTW + t];
                let p6 = w0[bb + (i + 6u) * D_OUTW + t];
                let p7 = w0[bb + (i + 7u) * D_OUTW + t];
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
        if (t < OUT_W) {
            outputs[job * dims.lu[2].w + t] = acc;
        } else {
            outputs[job * dims.lu[2].w + 40u + (t - OUT_W)] = tanh(acc);
        }
    }
}

// One 16-bit weight of the packed recurrent slab (element index `i`, low half first).
fn wh(i: u32) -> f32 {
    let p = unpack2x16float(w0h[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

// `main_deep` with the weights stored as 16-bit floats: half the memory traffic, the same f32 arithmetic.
@compute @workgroup_size(64)
fn main_deep_h(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let local = wg.y * dims.b.z + wg.x;
    if (local >= dims.a.x) {
        return;
    }
    let job = dims.lu[0].w + local;
    let jb = job * dims.lu[1].w;
    let slot = bitcast<u32>(jobs[jb]);
    prepare(t, job, jb);
    if (t < 16u) {
        a[104u + t] = jobs[jb + 12u + t];
    }
    if (t == 0u) {
        due_wg = select(0u, 1u, jobs[jb + 4u] > 0.5);
    }
    workgroupBarrier();
    // A job that is not due keeps its sensing outputs (motion drive, density) but skips the network:
    // the whole group agrees, so the barriers below stay uniform.
    if (workgroupUniformLoad(&due_wg) == 0u) {
        return;
    }

    // Each thread owns one output column and walks the layer's inputs in order (bias first, then
    // input 0, 1, 2, ...). Eight weights are read at a time so the loads overlap instead of
    // waiting one after another; the additions keep their order, so the answers are unchanged.
    let base = slot * D_TOTAL;
    if (t < D_H) {
            var acc = wh(base + D_IN0 * D_H + t);
            for (var i = 0u; i < 120u; i += 8u) {
                let p0 = wh(base + (i + 0u) * D_H + t);
                let p1 = wh(base + (i + 1u) * D_H + t);
                let p2 = wh(base + (i + 2u) * D_H + t);
                let p3 = wh(base + (i + 3u) * D_H + t);
                let p4 = wh(base + (i + 4u) * D_H + t);
                let p5 = wh(base + (i + 5u) * D_H + t);
                let p6 = wh(base + (i + 6u) * D_H + t);
                let p7 = wh(base + (i + 7u) * D_H + t);
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
        b[t] = tanh(acc);
    }
    workgroupBarrier();
    for (var l = 1u; l < 8u; l++) {
        let bb = base + D_L1 + (l - 1u) * D_HB;
        if (t < D_H) {
            if ((l & 1u) == 1u) {
            var acc = wh(bb + D_H * D_H + t);
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = wh(bb + (i + 0u) * D_H + t);
                let p1 = wh(bb + (i + 1u) * D_H + t);
                let p2 = wh(bb + (i + 2u) * D_H + t);
                let p3 = wh(bb + (i + 3u) * D_H + t);
                let p4 = wh(bb + (i + 4u) * D_H + t);
                let p5 = wh(bb + (i + 5u) * D_H + t);
                let p6 = wh(bb + (i + 6u) * D_H + t);
                let p7 = wh(bb + (i + 7u) * D_H + t);
                acc += p0 * b[i + 0u];
                acc += p1 * b[i + 1u];
                acc += p2 * b[i + 2u];
                acc += p3 * b[i + 3u];
                acc += p4 * b[i + 4u];
                acc += p5 * b[i + 5u];
                acc += p6 * b[i + 6u];
                acc += p7 * b[i + 7u];
            }
                a[t] = tanh(acc);
            } else {
            var acc = wh(bb + D_H * D_H + t);
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = wh(bb + (i + 0u) * D_H + t);
                let p1 = wh(bb + (i + 1u) * D_H + t);
                let p2 = wh(bb + (i + 2u) * D_H + t);
                let p3 = wh(bb + (i + 3u) * D_H + t);
                let p4 = wh(bb + (i + 4u) * D_H + t);
                let p5 = wh(bb + (i + 5u) * D_H + t);
                let p6 = wh(bb + (i + 6u) * D_H + t);
                let p7 = wh(bb + (i + 7u) * D_H + t);
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
                b[t] = tanh(acc);
            }
        }
        workgroupBarrier();
    }
    if (t < D_OUTW) {
        let bb = base + D_LAST;
            var acc = wh(bb + D_H * D_OUTW + t);
            for (var i = 0u; i < 32u; i += 8u) {
                let p0 = wh(bb + (i + 0u) * D_OUTW + t);
                let p1 = wh(bb + (i + 1u) * D_OUTW + t);
                let p2 = wh(bb + (i + 2u) * D_OUTW + t);
                let p3 = wh(bb + (i + 3u) * D_OUTW + t);
                let p4 = wh(bb + (i + 4u) * D_OUTW + t);
                let p5 = wh(bb + (i + 5u) * D_OUTW + t);
                let p6 = wh(bb + (i + 6u) * D_OUTW + t);
                let p7 = wh(bb + (i + 7u) * D_OUTW + t);
                acc += p0 * a[i + 0u];
                acc += p1 * a[i + 1u];
                acc += p2 * a[i + 2u];
                acc += p3 * a[i + 3u];
                acc += p4 * a[i + 4u];
                acc += p5 * a[i + 5u];
                acc += p6 * a[i + 6u];
                acc += p7 * a[i + 7u];
            }
        if (t < OUT_W) {
            outputs[job * dims.lu[2].w + t] = acc;
        } else {
            outputs[job * dims.lu[2].w + 40u + (t - OUT_W)] = tanh(acc);
        }
    }
}
