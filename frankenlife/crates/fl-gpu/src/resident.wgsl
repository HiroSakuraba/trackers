// Resident-field exchange on the GPU: deterministic per-cell reductions for uptake
// demand, granted energy and deposits, plus the phase-10 summary.
//
// Items (stride 36 floats): [base cell (u32 bits), tx, ty, amount, vec32].
//   uptake items: amount = demand, vec32 = the organism's affinity direction
//   deposit items: amount = q, vec32 = the packet's unit direction
// An item with base cell (x0, y0) and fractions (tx, ty) covers the cells
// (x0+dx, y0+dy), dx, dy in {0, 1}, with weights wx(dx) * wy(dy) (grid-cic-v1).
//
// Determinism: items are binned by base cell with a counting sort (integer
// atomics only count), each bin is then sorted by item index, and every
// per-cell sum visits the four neighbouring bins in a fixed order. No float
// result depends on thread scheduling.

struct R {
    gx: u32,
    gy: u32,
    n_items: u32,
    grid_items: u32, // workgroups along x for item-indexed kernels
    emax: f32,      // maximum efficiency
    gain: f32,      // affinity gain k
    thr: f32,       // discrimination threshold b
    kappa: f32,     // toxicity sensitivity
    tiny: f32,      // tiny-energy threshold
    grid_cells: u32, // workgroups along x for cell-indexed kernels
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var<storage, read_write> fe: array<f32>;
@group(0) @binding(2) var<storage, read_write> fm: array<f32>;
@group(0) @binding(3) var<storage, read> items: array<f32>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> starts: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(7) var<storage, read_write> cell_aux: array<f32>;
@group(0) @binding(8) var<storage, read_write> outv: array<f32>;

const IS: u32 = 36u;

fn tid_items(gid: vec3<u32>) -> u32 {
    return gid.y * r.grid_items * 64u + gid.x;
}

fn tid_cells(gid: vec3<u32>) -> u32 {
    return gid.y * r.grid_cells * 64u + gid.x;
}

fn key_of(i: u32) -> u32 {
    return bitcast<u32>(items[i * IS]);
}

// ---- binning: count, scan, scatter, sort ----

@compute @workgroup_size(64)
fn bin_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = tid_items(gid);
    if (i >= r.n_items) {
        return;
    }
    atomicAdd(&counts[key_of(i)], 1u);
}

var<workgroup> chunk_sum: array<u32, 256>;

// Exclusive prefix sum of counts into starts[0..cells], starts[cells] = total.
// One workgroup; also zeroes counts so they can serve as scatter cursors.
@compute @workgroup_size(256)
fn bin_scan(@builtin(local_invocation_index) t: u32) {
    let cells = r.gx * r.gy;
    let per = (cells + 255u) / 256u;
    let a = min(t * per, cells);
    let b = min(a + per, cells);
    var s = 0u;
    for (var c = a; c < b; c++) {
        s += atomicLoad(&counts[c]);
    }
    chunk_sum[t] = s;
    workgroupBarrier();
    if (t == 0u) {
        var run = 0u;
        for (var k = 0u; k < 256u; k++) {
            let v = chunk_sum[k];
            chunk_sum[k] = run;
            run += v;
        }
        starts[cells] = run;
    }
    workgroupBarrier();
    var off = chunk_sum[t];
    for (var c = a; c < b; c++) {
        let v = atomicLoad(&counts[c]);
        starts[c] = off;
        off += v;
        atomicStore(&counts[c], 0u);
    }
}

@compute @workgroup_size(64)
fn bin_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = tid_items(gid);
    if (i >= r.n_items) {
        return;
    }
    let k = key_of(i);
    let pos = starts[k] + atomicAdd(&counts[k], 1u);
    sorted[pos] = i;
}

// Sort each bin by item index (bins are small): makes bin order deterministic.
@compute @workgroup_size(64)
fn bin_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid_cells(gid);
    if (c >= r.gx * r.gy) {
        return;
    }
    let a = starts[c];
    let b = starts[c + 1u];
    for (var i = a + 1u; i < b; i++) {
        let v = sorted[i];
        var j = i;
        while (j > a && sorted[j - 1u] > v) {
            sorted[j] = sorted[j - 1u];
            j -= 1u;
        }
        sorted[j] = v;
    }
}

// ---- per-cell gathers over the four neighbouring bins ----

fn bin_of(cx: u32, cy: u32, dx: u32, dy: u32) -> u32 {
    let bx = (cx + r.gx - dx) % r.gx;
    let by = (cy + r.gy - dy) % r.gy;
    return by * r.gx + bx;
}

fn corner_weight(i: u32, dx: u32, dy: u32) -> f32 {
    let tx = items[i * IS + 1u];
    let ty = items[i * IS + 2u];
    var wx = 1.0 - tx;
    if (dx == 1u) {
        wx = tx;
    }
    var wy = 1.0 - ty;
    if (dy == 1u) {
        wy = ty;
    }
    return wx * wy;
}

// A_c: total uptake demand on each cell.
@compute @workgroup_size(64)
fn uptake_demand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid_cells(gid);
    if (c >= r.gx * r.gy) {
        return;
    }
    let cx = c % r.gx;
    let cy = c / r.gx;
    var a = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let dx = k & 1u;
        let dy = k >> 1u;
        let bn = bin_of(cx, cy, dx, dy);
        for (var p = starts[bn]; p < starts[bn + 1u]; p++) {
            let i = sorted[p];
            a += items[i * IS + 3u] * corner_weight(i, dx, dy);
        }
    }
    cell_aux[c] = a;
}

fn grant_scale(e: f32, a: f32) -> f32 {
    if (a > 0.0) {
        return min(e / a, 1.0);
    }
    return 0.0;
}

// Per organism: granted packets and packetwise chemistry on the observation field.
@compute @workgroup_size(64)
fn uptake_org(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = tid_items(gid);
    if (i >= r.n_items) {
        return;
    }
    let base = key_of(i);
    let x0 = base % r.gx;
    let y0 = base / r.gx;
    let d = items[i * IS + 3u];
    var absorbed = 0.0;
    var gain_sum = 0.0;
    var tox = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let dx = k & 1u;
        let dy = k >> 1u;
        let c = ((y0 + dy) % r.gy) * r.gx + (x0 + dx) % r.gx;
        let ec = fe[c];
        let u = grant_scale(ec, cell_aux[c]) * d * corner_weight(i, dx, dy);
        var az = 0.0;
        if (ec > 0.0) {
            for (var q = 0u; q < 32u; q++) {
                az += items[i * IS + 4u + q] * fm[c * 32u + q];
            }
            az = az / ec;
        }
        let eta = r.emax * tanh(r.gain * az - r.thr);
        absorbed += u;
        gain_sum += max(eta, 0.0) * u;
        tox += r.kappa * max(-eta, 0.0) * u;
    }
    outv[i * 4u] = absorbed;
    outv[i * 4u + 1u] = gain_sum;
    outv[i * 4u + 2u] = tox;
    outv[i * 4u + 3u] = 0.0;
}

// Debit each cell by the energy granted from it, and its moment proportionally.
@compute @workgroup_size(64)
fn uptake_debit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid_cells(gid);
    if (c >= r.gx * r.gy) {
        return;
    }
    let cx = c % r.gx;
    let cy = c / r.gx;
    let e = fe[c];
    let alpha = grant_scale(e, cell_aux[c]);
    var g = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let dx = k & 1u;
        let dy = k >> 1u;
        let bn = bin_of(cx, cy, dx, dy);
        for (var p = starts[bn]; p < starts[bn + 1u]; p++) {
            let i = sorted[p];
            g += alpha * items[i * IS + 3u] * corner_weight(i, dx, dy);
        }
    }
    if (g > 0.0) {
        let e_new = max(e - g, 0.0);
        var f = 0.0;
        if (e > 0.0) {
            f = e_new / e;
        }
        fe[c] = e_new;
        for (var q = 0u; q < 32u; q++) {
            fm[c * 32u + q] = fm[c * 32u + q] * f;
        }
    }
}

// Add all deposit packets touching each cell (energy and moment share the weights).
@compute @workgroup_size(64)
fn deposit_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid_cells(gid);
    if (c >= r.gx * r.gy) {
        return;
    }
    let cx = c % r.gx;
    let cy = c / r.gx;
    var add_e = 0.0;
    var add_m: array<f32, 32>;
    for (var q = 0u; q < 32u; q++) {
        add_m[q] = 0.0;
    }
    var touched = false;
    for (var k = 0u; k < 4u; k++) {
        let dx = k & 1u;
        let dy = k >> 1u;
        let bn = bin_of(cx, cy, dx, dy);
        for (var p = starts[bn]; p < starts[bn + 1u]; p++) {
            let i = sorted[p];
            let qc = items[i * IS + 3u] * corner_weight(i, dx, dy);
            add_e += qc;
            for (var q = 0u; q < 32u; q++) {
                add_m[q] += qc * items[i * IS + 4u + q];
            }
            touched = true;
        }
    }
    if (touched) {
        fe[c] = fe[c] + add_e;
        for (var q = 0u; q < 32u; q++) {
            fm[c * 32u + q] = fm[c * 32u + q] + add_m[q];
        }
    }
}

// Phase 10 per cell: tiny clearing, cone check (and projection of rounding-level
// excess up to 1e-3), summary row [energy after, tiny removed, cone excess, projected].
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = tid_cells(gid);
    if (c >= r.gx * r.gy) {
        return;
    }
    var e = fe[c];
    var removed = 0.0;
    var clear = false;
    if (e != 0.0) {
        clear = e <= r.tiny || abs(e) < 1.1754944e-38;
    } else {
        for (var q = 0u; q < 32u; q++) {
            if (fm[c * 32u + q] != 0.0) {
                clear = true;
            }
        }
    }
    if (clear) {
        removed = e;
        e = 0.0;
        fe[c] = 0.0;
        for (var q = 0u; q < 32u; q++) {
            fm[c * 32u + q] = 0.0;
        }
    }
    var n2 = 0.0;
    for (var q = 0u; q < 32u; q++) {
        let v = fm[c * 32u + q];
        n2 += v * v;
    }
    let n = sqrt(n2);
    var excess = -1.0;
    if (e > 0.0) {
        excess = n / e - 1.0;
    } else if (n > 0.0) {
        excess = 3.0e38;
    }
    var projected = 0.0;
    if (excess > 0.0 && excess <= 1e-3) {
        let f = e / n * (1.0 - 1e-7);
        for (var q = 0u; q < 32u; q++) {
            fm[c * 32u + q] = fm[c * 32u + q] * f;
        }
        projected = 1.0;
    }
    outv[c * 4u] = e;
    outv[c * 4u + 1u] = removed;
    outv[c * 4u + 2u] = excess;
    outv[c * 4u + 3u] = projected;
}
