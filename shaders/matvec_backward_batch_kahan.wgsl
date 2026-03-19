// Batched backward: d_x[pos] = W^T @ d_y[pos] — Kahan compensated summation.
// One thread per (pos, j). Sequential accumulation matches forward Kahan precision.

struct Params {
    out_dim: u32,
    in_dim: u32,
    n_pos: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> d_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec_backward_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;
    let n_pos = params.n_pos;

    let pos = flat_id / in_dim;
    let j = flat_id % in_dim;

    if (pos >= n_pos) { return; }

    let dy_base = pos * out_dim;

    // Kahan compensated summation: d_x[j] = sum_i W[i][j] * d_y[i]
    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    for (var i: u32 = 0u; i < out_dim; i++) {
        let product = w[i * in_dim + j] * d_y[dy_base + i];
        let y_val = product - comp;
        let t = sum + y_val;
        comp = (t - sum) - y_val;
        sum = t;
    }

    d_x[pos * in_dim + j] = sum;
}
