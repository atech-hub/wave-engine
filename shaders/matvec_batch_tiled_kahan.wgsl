// Batched matrix-vector: tiled workgroup reduction with Kahan compensation.
// MUST match matvec_backward_batch_tiled_kahan.wgsl accumulation pattern exactly.
// One workgroup per (row, pos). 256 threads per workgroup.
// Dispatch: (out_dim, n_pos, 1) workgroups.
//
// 256 threads for 768-dim: each thread accumulates only 3 elements (nearly exact).
// Tree reduction: 8 levels (log2(256)) with compensated addition.
// Carries Kahan compensation through tree to eliminate precision gap.

struct Params {
    out_dim: u32,
    in_dim: u32,
    n_pos: u32,
    use_bias: u32,
}

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial_acc: array<f32, 256>;
var<workgroup> partial_comp: array<f32, 256>;

@compute @workgroup_size(256)
fn matvec_batch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;
    let n_pos = params.n_pos;

    let row = wid.x;  // one workgroup per output row
    let pos = wid.y;  // one workgroup per position

    if (pos >= n_pos) { return; }

    let x_base = pos * in_dim;
    let w_base = row * in_dim;

    // Each thread accumulates its strided chunk with Kahan compensation
    // At 768-dim / 256 threads = 3 elements per thread — nearly exact
    var acc: f32 = 0.0;
    var comp: f32 = 0.0;
    var j: u32 = tid;
    loop {
        if (j >= in_dim) { break; }
        let product = w[w_base + j] * x[x_base + j];
        let y_val = product - comp;
        let t = acc + y_val;
        comp = (t - acc) - y_val;
        acc = t;
        j += 256u;
    }
    partial_acc[tid] = acc;
    partial_comp[tid] = comp;
    workgroupBarrier();

    // Compensated tree reduction — carries error through all 8 levels
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            let a = partial_acc[tid];
            let b_val = partial_acc[tid + stride];
            let sum = a + b_val;
            // Capture low-order bits lost in the addition
            let err = (a - sum) + b_val;
            partial_comp[tid] += partial_comp[tid + stride] + err;
            partial_acc[tid] = sum;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var result = partial_acc[0] + partial_comp[0];
        if (params.use_bias == 1u) {
            result += b[row];
        }
        y[pos * out_dim + row] = result;
    }
}
