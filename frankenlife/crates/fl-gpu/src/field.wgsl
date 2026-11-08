// Field step on the GPU (plan §4.5): positive five-point diffusion (subcycled),
// exponential decay, normalized truncated minimum-image Gaussian sources, and
// tiny-reservoir clearing. Energy plane `e[c]`, moments `m[c*32 + k]`
// (cell-integrated, same layout as fl-core).
//
// Accounting per cell goes to `acct[c*3 + {0,1,2}]` = decay removed, energy
// injected, tiny energy cleared; the CPU sums these in f64 in cell order.
//
// Bind groups: AB (src A -> dst B), BA (src B -> dst A) for diffusion
// ping-pong, and INPLACE (dst A, src dummies) for the in-place kernels.

struct F {
    gx: u32,
    gy: u32,
    n_sources: u32,
    grid_x: u32,     // workgroups along x (64 threads each)
    rx: f32,         // per-subcycle D dt / hx^2
    ry: f32,
    decay: f32,      // removed fraction -expm1(-lambda dt) (avoids cancellation in 1 - exp)
    tiny: f32,       // tiny-energy threshold
    width: f32,
    height: f32,
    hx: f32,
    hy: f32,
}

@group(0) @binding(0) var<uniform> f: F;
@group(0) @binding(1) var<storage, read> se: array<f32>;
@group(0) @binding(2) var<storage, read> sm: array<f32>;
@group(0) @binding(3) var<storage, read_write> de: array<f32>;
@group(0) @binding(4) var<storage, read_write> dm: array<f32>;
@group(0) @binding(5) var<storage, read_write> acct: array<f32>;
// Per source (stride 44): cx, cy, sigma, cutoff, q_total, W (written by source_norm),
// x0, y0, bw, bh (u32 bits: support box start and size), chem[32], pad[2].
@group(0) @binding(6) var<storage, read_write> src: array<f32>;

const PL: u32 = 33u;
const SS: u32 = 44u;

fn tid(gid: vec3<u32>) -> u32 {
    return gid.y * f.grid_x * 64u + gid.x;
}

fn sget(c: u32, p: u32) -> f32 {
    if (p == 0u) {
        return se[c];
    }
    return sm[c * 32u + p - 1u];
}

fn min_image(d: f32, l: f32) -> f32 {
    return d - l * round(d / l);
}

// One diffusion subcycle, one thread per (cell, plane), plane fastest.
@compute @workgroup_size(64)
fn diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = tid(gid);
    let cells = f.gx * f.gy;
    if (g >= cells * PL) {
        return;
    }
    let c = g / PL;
    let p = g % PL;
    let x = c % f.gx;
    let y = c / f.gx;
    let xl = (x + f.gx - 1u) % f.gx;
    let xr = (x + 1u) % f.gx;
    let yu = (y + f.gy - 1u) % f.gy;
    let yd = (y + 1u) % f.gy;
    // Flux form: each neighbour exchange rx*(n - u) is exactly antisymmetric between
    // the two cells, so the total is conserved up to unbiased rounding. (The algebraically
    // equal center*u + rx*(...) form, with center = 1 - 2rx - 2ry rounded to f32, has
    // weights that do not sum to exactly one and drifts systematically.)
    let u = sget(c, p);
    let v = u
        + f.rx * ((sget(y * f.gx + xl, p) - u) + (sget(y * f.gx + xr, p) - u))
        + f.ry * ((sget(yu * f.gx + x, p) - u) + (sget(yd * f.gx + x, p) - u));
    if (p == 0u) {
        de[c] = v;
    } else {
        dm[c * 32u + p - 1u] = v;
    }
}

// Exponential decay in place; plane 0 records the removed energy.
@compute @workgroup_size(64)
fn decay(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = tid(gid);
    let cells = f.gx * f.gy;
    if (g >= cells * PL) {
        return;
    }
    let c = g / PL;
    let p = g % PL;
    if (p == 0u) {
        let old = de[c];
        let nw = old - old * f.decay;
        de[c] = nw;
        acct[c * 3u] = old - nw;
    } else {
        let i = c * 32u + p - 1u;
        dm[i] = dm[i] - dm[i] * f.decay;
    }
}

var<workgroup> partial: array<f32, 256>;

// Normalization of each source's truncated kernel over its discrete support:
// one workgroup per source sums exp(-d^2 / 2 sigma^2) over the support box.
@compute @workgroup_size(256)
fn source_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let s = wg.x;
    let b = s * SS;
    let cx = src[b];
    let cy = src[b + 1u];
    let sigma = src[b + 2u];
    let cut = src[b + 3u];
    let x0 = bitcast<u32>(src[b + 6u]);
    let y0 = bitcast<u32>(src[b + 7u]);
    let bw = bitcast<u32>(src[b + 8u]);
    let bh = bitcast<u32>(src[b + 9u]);
    var acc = 0.0;
    for (var i = t; i < bw * bh; i += 256u) {
        let ix = (x0 + i % bw) % f.gx;
        let iy = (y0 + i / bw) % f.gy;
        let dx = min_image((f32(ix) + 0.5) * f.hx - cx, f.width);
        let dy = min_image((f32(iy) + 0.5) * f.hy - cy, f.height);
        let d2 = dx * dx + dy * dy;
        if (d2 <= cut * cut) {
            acc += exp(-d2 / (2.0 * sigma * sigma));
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w = w >> 1u) {
        if (t < w) {
            partial[t] += partial[t + w];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        src[b + 5u] = partial[0];
    }
}

// Inject every source into each cell (gather: no write races), in place.
@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid(gid);
    if (c >= f.gx * f.gy) {
        return;
    }
    let px = (f32(c % f.gx) + 0.5) * f.hx;
    let py = (f32(c / f.gx) + 0.5) * f.hy;
    let e_old = de[c];
    var add_e = 0.0;
    for (var s = 0u; s < f.n_sources; s++) {
        let b = s * SS;
        let cut = src[b + 3u];
        let w_total = src[b + 5u];
        let dx = min_image(px - src[b], f.width);
        let dy = min_image(py - src[b + 1u], f.height);
        let d2 = dx * dx + dy * dy;
        if (d2 <= cut * cut && w_total > 0.0) {
            let sigma = src[b + 2u];
            let q = src[b + 4u] * exp(-d2 / (2.0 * sigma * sigma)) / w_total;
            add_e += q;
            for (var k = 0u; k < 32u; k++) {
                dm[c * 32u + k] += q * src[b + 10u + k];
            }
        }
    }
    let e_new = e_old + add_e;
    de[c] = e_new;
    acct[c * 3u + 1u] = e_new - e_old;
}

// Tiny-reservoir clearing (energy and moment together), in place.
@compute @workgroup_size(64)
fn tiny(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid(gid);
    if (c >= f.gx * f.gy) {
        return;
    }
    let e = de[c];
    var clear = false;
    if (e != 0.0) {
        clear = e <= f.tiny || abs(e) < 1.1754944e-38;
    } else {
        for (var k = 0u; k < 32u; k++) {
            if (dm[c * 32u + k] != 0.0) {
                clear = true;
            }
        }
    }
    if (clear) {
        acct[c * 3u + 2u] = e;
        de[c] = 0.0;
        for (var k = 0u; k < 32u; k++) {
            dm[c * 32u + k] = 0.0;
        }
    } else {
        acct[c * 3u + 2u] = 0.0;
    }
}
