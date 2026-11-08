// Fused independent-network MLP: one workgroup per organism.
//
// Weights for a slot and layer are stored transposed, [in][out], then `out`
// biases, so thread `o` reads w[base + i*out + o]: consecutive threads read
// consecutive floats. Hidden layers apply tanh; the final layer is linear
// (sigmoid and other decoding happen on the CPU in fl-core).

struct Dims {
    n_jobs: u32,
    n_layers: u32, // 2 or 3 affine layers
    d0: u32,       // input width (104)
    d1: u32,       // output of layer 0
    d2: u32,       // output of layer 1
    d3: u32,       // output of layer 2 (when n_layers == 3)
    grid_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<storage, read> w0: array<f32>;
@group(0) @binding(5) var<storage, read> w1: array<f32>;
@group(0) @binding(6) var<storage, read> w2: array<f32>;

const WG: u32 = 64u;
const IN_W: u32 = 104u;
const OUT_W: u32 = 36u;

var<workgroup> a: array<f32, 128>;
var<workgroup> b: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let job = wg.y * dims.grid_x + wg.x;
    if (job >= dims.n_jobs) {
        return;
    }
    let slot = slots[job];

    for (var i = t; i < dims.d0; i += WG) {
        a[i] = inputs[job * IN_W + i];
    }
    workgroupBarrier();

    // Layer 0: a (d0) -> b (d1), always a hidden layer.
    {
        let n_in = dims.d0;
        let n_out = dims.d1;
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

    // Layer 1: b (d1) -> a (d2); hidden when there are three layers, else the output.
    let last1 = dims.n_layers == 2u;
    {
        let n_in = dims.d1;
        let n_out = dims.d2;
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
            outputs[job * OUT_W + o] = a[o];
        }
        return;
    }

    // Layer 2: a (d2) -> output (d3), linear.
    {
        let n_in = dims.d2;
        let n_out = dims.d3;
        let base = slot * (n_in * n_out + n_out);
        for (var o = t; o < n_out; o += WG) {
            var acc = w2[base + n_in * n_out + o];
            for (var i = 0u; i < n_in; i++) {
                acc += w2[base + i * n_out + o] * a[i];
            }
            outputs[job * OUT_W + o] = acc;
        }
    }
}
