// Rebuild the 12-float (28 with memory) sense job the fused shader already reads.
// Dynamic stride 8 (24 with memory): [x, y, due, intrinsic0..4, memory0..15]. Those change every tick.
// Static stride 2: [slot as f32 bits, scale]. Those stay resident until the
// organism set or its slot changes. Padding floats are the same zeros the
// CPU pack used to write.

struct Expand {
    n: u32,
    grid_x: u32,
    dyn_stride: u32,   // 8, or 24 with memory
    job_stride: u32,   // 12, or 28 with memory
}

@group(0) @binding(0) var<uniform> ex: Expand;
@group(0) @binding(1) var<storage, read> dynamic: array<f32>;
@group(0) @binding(2) var<storage, read> statics: array<f32>;
@group(0) @binding(3) var<storage, read_write> jobs: array<f32>;

@compute @workgroup_size(64)
fn expand_jobs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * ex.grid_x * 64u + gid.x;
    if (i >= ex.n) {
        return;
    }
    let d = i * ex.dyn_stride;
    let s = i * 2u;
    let j = i * ex.job_stride;
    jobs[j] = statics[s];
    jobs[j + 1u] = dynamic[d];
    jobs[j + 2u] = dynamic[d + 1u];
    jobs[j + 3u] = statics[s + 1u];
    jobs[j + 4u] = dynamic[d + 2u];
    for (var k = 0u; k < 5u; k++) {
        jobs[j + 5u + k] = dynamic[d + 3u + k];
    }
    jobs[j + 10u] = 0.0;
    jobs[j + 11u] = 0.0;
    if (ex.job_stride > 12u) {
        for (var k = 0u; k < 16u; k++) {
            jobs[j + 12u + k] = dynamic[d + 8u + k];
        }
    }
}
