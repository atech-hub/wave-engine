// Batched backward: d_x[pos] = W^T @ d_y[pos] — tiled with Kahan compensation.
// MUST match matvec_batch_tiled_kahan.wgsl accumulation pattern exactly.
// One workgroup per (j, pos). 256 threads per workgroup.
// Dispatch: (in_dim, n_pos, 1) workgroups.
//
// 256 threads for 768-dim: each thread accumulates only 3 elements (nearly exact).
// Tree reduction: 8 levels (log2(256)) with compensated addition.
// Carries Kahan compensation through tree to eliminate precision gap.

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

var<workgroup> partial_acc: array<f32, 256>;
var<workgroup> partial_comp: array<f32, 256>;

@compute @workgroup_size(256)
fn matvec_backward_batch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let j = wid.x;      // input element
    let pos = wid.y;    // position
    let out_dim = params.out_dim;
    let in_dim = params.in_dim;

    if (pos >= params.n_pos || j >= in_dim) { return; }

    let dy_base = pos * out_dim;

    // Each thread accumulates its strided chunk with Kahan compensation
    // At 768-dim / 256 threads = 3 elements per thread — nearly exact
    var acc: f32 = 0.0;
    var comp: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= out_dim) { break; }
        let product = w[i * in_dim + j] * d_y[dy_base + i];
        let y_val = product - comp;
        let t = acc + y_val;
        comp = (t - acc) - y_val;
        acc = t;
        i += 256u;
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
            let err = (a - sum) + b_val;
            partial_comp[tid] += partial_comp[tid + stride] + err;
            partial_acc[tid] = sum;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        d_x[pos * in_dim + j] = partial_acc[0] + partial_comp[0];
    }
}
