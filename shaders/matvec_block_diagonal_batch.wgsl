// Block-diagonal batched matvec: y[pos] = BlockDiag(W) @ x[pos] + b
//
// N groups of group_size dims each. Each thread computes one output element.
// Thread determines which group from its output index.
// Only reads weights from its group's block.
//
// 6 groups of 128×128 replaces dense 768×768:
//   Dense:  768 × 768 = 589,824 params
//   Block:  6 × 128 × 128 = 98,304 params (6x reduction)
//
// One dispatch replaces dense matvec_batch.wgsl.

struct Params {
    group_size: u32,   // 128 (= n_embd / n_groups)
    n_groups: u32,     // 6
    n_pos: u32,
    n_embd: u32,       // 768 (= group_size * n_groups)
}

// Weight layout: groups concatenated.
// w[g] starts at offset g * group_size * group_size
@group(0) @binding(0) var<storage, read> w: array<f32>;       // [n_groups * gs * gs]
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [n_pos * n_embd]
@group(0) @binding(2) var<storage, read> b: array<f32>;       // [n_embd]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [n_pos * n_embd]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn matvec_block_diagonal_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let gs = params.group_size;
    let n_embd = params.n_embd;
    let n_pos = params.n_pos;

    let pos = flat_id / n_embd;
    let out_i = flat_id % n_embd;

    if (pos >= n_pos) { return; }

    let group = out_i / gs;
    let local_i = out_i % gs;
    let w_base = group * gs * gs + local_i * gs;
    let x_base = pos * n_embd + group * gs;

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < gs; j++) {
        sum += w[w_base + j] * x[x_base + j];
    }
    sum += b[out_i];

    y[pos * n_embd + out_i] = sum;
}
