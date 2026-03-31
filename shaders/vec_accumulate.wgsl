// In-place vector accumulation: a[i] += b[i]
// Used by fused ODE backward to accumulate parameter gradients.

struct Params { len: u32, _p1: u32, _p2: u32, _p3: u32, }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn vec_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.len) { return; }
    a[id.x] = a[id.x] + b[id.x];
}
