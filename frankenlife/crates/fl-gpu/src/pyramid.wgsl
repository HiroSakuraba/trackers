// Density sensor pyramid (plan §6.1), cell-major with 33 planes per cell:
// pyr[(level_offset + cell) * 33 + 0] = R, [+1+k] = M_k.
// `level0` converts the cell-integrated field to densities; `down` averages
// 2x2 blocks of the finer level's densities (never integrated energy).

struct P {
    gx: u32,        // destination level width (cells)
    gy: u32,        // destination level height
    src_off: u32,   // source level offset (cells)
    dst_off: u32,   // destination level offset (cells)
    src_gx: u32,    // source level width
    inv_area: f32,  // 1 / (hx * hy) for level 0
    grid_x: u32,    // workgroups along x in this dispatch
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fe: array<f32>;
@group(0) @binding(2) var<storage, read> fm: array<f32>;
@group(0) @binding(3) var<storage, read_write> pyr: array<f32>;

const PL: u32 = 33u;

@compute @workgroup_size(64)
fn level0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.y * p.grid_x * 64u + gid.x;
    if (c >= p.gx * p.gy) {
        return;
    }
    pyr[c * PL] = fe[c] * p.inv_area;
    for (var k = 0u; k < 32u; k++) {
        pyr[c * PL + 1u + k] = fm[c * 32u + k] * p.inv_area;
    }
}

@compute @workgroup_size(64)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.y * p.grid_x * 64u + gid.x;
    if (c >= p.gx * p.gy) {
        return;
    }
    let x = c % p.gx;
    let y = c / p.gx;
    let a = (p.src_off + (2u * y) * p.src_gx + 2u * x) * PL;
    let b = (p.src_off + (2u * y) * p.src_gx + 2u * x + 1u) * PL;
    let cc = (p.src_off + (2u * y + 1u) * p.src_gx + 2u * x) * PL;
    let d = (p.src_off + (2u * y + 1u) * p.src_gx + 2u * x + 1u) * PL;
    let o = (p.dst_off + c) * PL;
    for (var k = 0u; k < PL; k++) {
        pyr[o + k] = (pyr[a + k] + pyr[b + k] + pyr[cc + k] + pyr[d + k]) * 0.25;
    }
}
