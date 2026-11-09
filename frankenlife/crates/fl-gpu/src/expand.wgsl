// Expand compact uptake headers into the 36-float item layout the exchange
// kernels already read. This is a copy: header floats and affinity floats are
// written unchanged, in the same order the CPU used to pack on the host.
//
// Header stride 5: [base cell as f32 bits, tx, ty, demand, affinity slot as f32 bits].
// Affinity stride 32. Item stride 36: [cell, tx, ty, demand, affinity[32]].

struct Expand {
    n: u32,
    grid_x: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> ex: Expand;
@group(0) @binding(1) var<storage, read> headers: array<f32>;
@group(0) @binding(2) var<storage, read> affinity: array<f32>;
@group(0) @binding(3) var<storage, read_write> item_out: array<f32>;

@compute @workgroup_size(64)
fn expand_items(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * ex.grid_x * 64u + gid.x;
    if (i >= ex.n) {
        return;
    }
    let h = i * 5u;
    let b = i * 36u;
    item_out[b] = headers[h];
    item_out[b + 1u] = headers[h + 1u];
    item_out[b + 2u] = headers[h + 2u];
    item_out[b + 3u] = headers[h + 3u];
    let slot = bitcast<u32>(headers[h + 4u]);
    let a = slot * 32u;
    for (var q = 0u; q < 32u; q++) {
        item_out[b + 4u + q] = affinity[a + q];
    }
}
