// Build uptake headers on the device from the sense output that is already there.
//
// The uptake request of an organism is `dt * umax * r / (K + r)` with `r` the
// sensed level-0 energy density (clamped at zero). Computing it here lets the
// uptake exchange follow sensing in the same queue, with no readback between.
//
// hdr_in stride 4 (global item order): [base cell bits, tx, ty, affinity slot bits].
// sense_out stride 40, or 56 with memory (rows of the current sense group): density at +38.
// headers stride 5 (global item order): [base cell bits, tx, ty, demand, affinity slot bits].

struct D {
    n: u32,      // rows in this group
    grid_x: u32,
    base: u32,   // global item index of this group's first row
    out_stride: u32,   // sense output row: 40, or 56 with memory
    coef: f32,   // dt * umax
    half: f32,   // uptake half-saturation density K
    _a: f32,
    _b: f32,
}

@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> hdr_in: array<f32>;
@group(0) @binding(2) var<storage, read> sense_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> headers: array<f32>;

@compute @workgroup_size(64)
fn make_headers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * d.grid_x * 64u + gid.x;
    if (i >= d.n) {
        return;
    }
    let g = d.base + i;
    let r = max(sense_out[i * d.out_stride + 38u], 0.0);
    let h = g * 5u;
    let s = g * 4u;
    headers[h] = hdr_in[s];
    headers[h + 1u] = hdr_in[s + 1u];
    headers[h + 2u] = hdr_in[s + 2u];
    headers[h + 3u] = d.coef * r / (d.half + r);
    headers[h + 4u] = hdr_in[s + 3u];
}
