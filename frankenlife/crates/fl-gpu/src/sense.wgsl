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
// Outputs per job (stride 40): 36 logits, motion drive x, y, level-0 density.
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
@group(0) @binding(5) var<storage, read> w1: array<f32>;
@group(0) @binding(6) var<storage, read> w2: array<f32>;
@group(0) @binding(7) var<storage, read_write> outputs: array<f32>;

const WG: u32 = 64u;
const PL: u32 = 33u;
const JS: u32 = 12u;   // job stride: slot bits, x, y, scale, due, 5 intrinsic, 2 pad
const OS: u32 = 40u;   // output stride
const TS: u32 = 36u;   // traits stride: 33 motion sensitivities + pad
const OUT_W: u32 = 36u;

var<workgroup> a: array<f32, 128>;
var<workgroup> b: array<f32, 128>;
var<workgroup> rv: array<f32, 33>;
var<workgroup> rdx: array<f32, 33>;
var<workgroup> rdy: array<f32, 33>;
var<workgroup> gqx: array<f32, 33>;
var<workgroup> gqy: array<f32, 33>;

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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let local = wg.y * dims.b.z + wg.x;
    if (local >= dims.a.x) {
        return;
    }
    // Row of this group's first job in the job and output buffers (0 unless
    // several groups share one submission).
    let job = dims.lu[0].w + local;
    let jb = job * JS;
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
        outputs[job * OS + 36u] = fx;
        outputs[job * OS + 37u] = fy;
        outputs[job * OS + 38u] = max(sample(0u, x, y, 0u).x, 0.0);
    }

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
            outputs[job * OS + o] = a[o];
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
            outputs[job * OS + o] = acc;
        }
    }
}
