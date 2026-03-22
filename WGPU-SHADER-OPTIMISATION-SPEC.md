# WGPU SHADER OPTIMISATIONS — Perturbative ODE + Block-Diagonal out_proj
# Date: 2026-03-22 (updated post-refactor)
# For: Code (Claude Code)
# Priority: After 100-iter validation, before next major training run
# Goal: Bring perturbative ODE + block-diagonal to the wgpu tier for AMD/cross-platform speed
#
# NOTE: Updated for post-refactor directory structure (common/, wgpu_tier/, candle_tier/)
# and OutProjWeights enum. All file paths and Rust-side patterns updated.

---

## Overview

Two new WGSL shaders replace the current fused RK4 ODE chain and dense matvec_batch
out_proj. Together they should cut wgpu FFN time dramatically:

| Component   | Current                          | New                               |
|-------------|----------------------------------|-----------------------------------|
| ODE forward | 16-step RK4 (12 dispatches/step) | Perturbative (1 dispatch total)   |
| ODE buffers | 8 scratch buffers (r,s,k1-k4,mid,new)| 2 temp buffers (r_lin, s_lin) |
| out_proj    | Dense matvec_batch (768×768)     | Block-diagonal (6×128×128 fused)  |
| ODE backward| kerr_backward_batch × 16 steps   | Single perturbative backward      |

## File Map (post-refactor)

| Component | File |
|-----------|------|
| GpuBackend struct + pipelines | `src/wgpu_tier/pipelines.rs` |
| Forward dispatch ops | `src/wgpu_tier/ops_forward.rs` |
| Backward dispatch ops | `src/wgpu_tier/ops_backward.rs` |
| Resident weight buffers | `src/wgpu_tier/resident.rs` |
| FFN full GPU pipeline | `src/wgpu_tier/ffn_full_gpu.rs` |
| FFN ping-pong buffers | `src/wgpu_tier/ffn_gpu.rs` |
| GPU backend trait impl | `src/wgpu_tier/gpu_backend.rs` |
| Backend dispatch | `src/wgpu_tier/dispatch.rs` |
| Buffer pool | `src/wgpu_tier/buffers.rs` |
| Weight structs (KerrWeights etc.) | `src/common/model.rs` |
| OutProjWeights enum | `src/common/model.rs` |
| FFN routing (CPU/GPU) | `src/common/ffn.rs` |
| FFT ODE (CPU path) | `src/common/fft_ode.rs` |
| Candle perturbative ODE | `src/candle_tier/ode.rs` |
| WGSL shaders | `shaders/*.wgsl` |

---

## Shader 1: kerr_perturbative_batch.wgsl

### Math (matches candle_tier/ode.rs exactly)

Given input r[k], s[k] per band, and parameters gamma[], omega[], alpha, beta:

```
Step 1 — Linear solution (damping + rotation):
  decay[k] = exp(-softplus(gamma[k]))
  r_lin[k] = decay[k] * (r[k]*cos(omega[k]) - s[k]*sin(omega[k]))
  s_lin[k] = decay[k] * (r[k]*sin(omega[k]) + s[k]*cos(omega[k]))

Step 2 — Self-phase modulation:
  mag_sq[k] = r_lin[k]² + s_lin[k]²

Step 3 — Cross-phase modulation (stencil [1,1,0,1,1]):
  ns[k] = mag_sq[k-2] + mag_sq[k-1] + mag_sq[k+1] + mag_sq[k+2]
  (with bounds checks, 0 for out-of-range)

Step 4 — Phase perturbation:
  delta_phi[k] = alpha * mag_sq[k] + beta * ns[k]

Step 5 — Apply correction:
  r_out[k] = r_lin[k] - delta_phi[k] * s_lin[k]
  s_out[k] = s_lin[k] + delta_phi[k] * r_lin[k]
```

### WGSL Implementation

```wgsl
// Perturbative Kerr-ODE: single-pass analytical approximation.
// Replaces 16-step RK4 (192 dispatches) with ONE dispatch.
//
// One thread per (pos, band). All bands computed independently.
// No scratch buffers needed — everything computed in-register.
//
// Lab-validated: MSE 0.000005 vs RK4-16 baseline.
// Trains BETTER than RK4-16 (loss 2.97 vs 3.07 at 100 iters).

struct Params {
    n_bands: u32,
    n_pos: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read_write> r_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(3) var<storage, read_write> s_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(4) var<storage, read> gamma: array<f32>;      // [n_bands] (pre-softplus)
@group(0) @binding(5) var<storage, read> omega: array<f32>;      // [n_bands]
@group(0) @binding(6) var<storage, read> decay: array<f32>;      // [n_bands] = exp(-softplus(gamma))
@group(0) @binding(7) var<storage, read> cos_w: array<f32>;      // [n_bands] = cos(omega)
@group(0) @binding(8) var<storage, read> sin_w: array<f32>;      // [n_bands] = sin(omega)
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(64)
fn kerr_perturbative_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let n = params.n_bands;
    let n_pos = params.n_pos;

    let pos = flat_id / n;
    let band = flat_id % n;

    if (pos >= n_pos) {
        return;
    }

    let base = pos * n;
    let idx = base + band;

    let r = r_in[idx];
    let s = s_in[idx];

    // Step 1: Linear solution — damping + base rotation
    let d = decay[band];
    let cw = cos_w[band];
    let sw = sin_w[band];
    let r_lin = d * (r * cw - s * sw);
    let s_lin = d * (r * sw + s * cw);

    // Step 2: Self-phase modulation
    let mag_sq = r_lin * r_lin + s_lin * s_lin;

    // Step 3: Cross-phase modulation — recompute neighbours in-thread
    // (option b: avoids second pass, 4 extra trig ops per thread)
    var ns: f32 = 0.0;
    if (band >= 2u) {
        let ni = base + band - 2u;
        let d2 = decay[band - 2u];
        let c2 = cos_w[band - 2u];
        let s2 = sin_w[band - 2u];
        let rl = d2 * (r_in[ni] * c2 - s_in[ni] * s2);
        let sl = d2 * (r_in[ni] * s2 + s_in[ni] * c2);
        ns += rl * rl + sl * sl;
    }
    if (band >= 1u) {
        let ni = base + band - 1u;
        let d1 = decay[band - 1u];
        let c1 = cos_w[band - 1u];
        let s1 = sin_w[band - 1u];
        let rl = d1 * (r_in[ni] * c1 - s_in[ni] * s1);
        let sl = d1 * (r_in[ni] * s1 + s_in[ni] * c1);
        ns += rl * rl + sl * sl;
    }
    if (band + 1u < n) {
        let ni = base + band + 1u;
        let dp = decay[band + 1u];
        let cp = cos_w[band + 1u];
        let sp = sin_w[band + 1u];
        let rl = dp * (r_in[ni] * cp - s_in[ni] * sp);
        let sl = dp * (r_in[ni] * sp + s_in[ni] * cp);
        ns += rl * rl + sl * sl;
    }
    if (band + 2u < n) {
        let ni = base + band + 2u;
        let dp = decay[band + 2u];
        let cp = cos_w[band + 2u];
        let sp = sin_w[band + 2u];
        let rl = dp * (r_in[ni] * cp - s_in[ni] * sp);
        let sl = dp * (r_in[ni] * sp + s_in[ni] * cp);
        ns += rl * rl + sl * sl;
    }

    // Step 4: Phase perturbation
    let delta_phi = params.alpha * mag_sq + params.beta * ns;

    // Step 5: Apply correction
    r_out[idx] = r_lin - delta_phi * s_lin;
    s_out[idx] = s_lin + delta_phi * r_lin;
}
```

### Alternative: Two-Pass Version (exact Candle match)

If numerical exactness vs Candle is required (for checkpoint compatibility):

**Pass 1: `kerr_perturbative_linear.wgsl`**
- Compute r_lin, s_lin, mag_sq for all (pos, band)
- Write to intermediate buffers

**Pass 2: `kerr_perturbative_nonlinear.wgsl`**
- Read neighbours' mag_sq from intermediate buffer
- Compute delta_phi, apply correction
- Write r_out, s_out

Two dispatches instead of one, but still 2 vs 192 for RK4-16.
Code should decide based on whether single-dispatch or exact-match matters more.

### Rust-side changes

**File: `src/wgpu_tier/ops_forward.rs`**

Replace `gpu_kerr_ode_batch_fused` with:

```rust
pub(crate) fn gpu_kerr_ode_perturbative_batch(
    &self, weights: &KerrWeights, xs: &[Vec<f32>]
) -> Vec<Vec<f32>> {
    let n_pos = xs.len();
    let n_bands = weights.gamma_raw.len();

    // Deinterleave: [r0,s0,r1,s1,...] → separate r[] and s[]
    let (r_flat, s_flat) = deinterleave(xs, n_bands);

    // Precompute decay, cos_w, sin_w (could cache in resident buffers)
    let decay: Vec<f32> = weights.gamma_raw.iter()
        .map(|&g| (-softplus(g)).exp()).collect();
    let cos_w: Vec<f32> = weights.omega.iter().map(|&w| w.cos()).collect();
    let sin_w: Vec<f32> = weights.omega.iter().map(|&w| w.sin()).collect();

    // Upload to GPU
    let r_buf = self.storage_buf("r", &r_flat);
    let s_buf = self.storage_buf("s", &s_flat);
    let r_out = self.output_buf("r_out", n_pos * n_bands);
    let s_out = self.output_buf("s_out", n_pos * n_bands);
    let decay_buf = self.storage_buf("decay", &decay);
    let cos_buf = self.storage_buf("cos_w", &cos_w);
    let sin_buf = self.storage_buf("sin_w", &sin_w);
    let gamma_buf = self.storage_buf("gamma", &weights.gamma_raw);
    let omega_buf = self.storage_buf("omega", &weights.omega);
    let params_buf = self.uniform_buf("params", &PerturbativeParams {
        n_bands: n_bands as u32,
        n_pos: n_pos as u32,
        alpha: weights.alpha,
        beta: weights.beta,
    });

    // Bind group, ONE dispatch — all positions, all bands
    let wg = ((n_pos * n_bands + 63) / 64) as u32;
    let mut encoder = self.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.kerr_perturbative_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }
    self.queue.submit(Some(encoder.finish()));

    // ONE readback, reinterleave
    reinterleave(&self.readback(&r_out, n_pos*n_bands),
                 &self.readback(&s_out, n_pos*n_bands),
                 n_bands, n_pos)
}
```

### Backward: kerr_perturbative_backward_batch.wgsl

**Recommendation:** Keep identity backward for now (matches current wgpu training).
The Candle tier gets true gradients via autograd. The wgpu tier uses identity backward
plus Adam — this already works (wgpu tier trains, loss descends). True perturbative
backward can be added later as a refinement.

### Buffers eliminated

Current RK4 fused requires per-step:
- r_buf, s_buf (state)
- k1r, k1s, k2r, k2s, k3r, k3s, k4r, k4s (derivatives)
- r_mid, s_mid (midpoints)
- r_new, s_new (output)
= 14 buffers

Perturbative requires:
- r_in, s_in (input)
- r_out, s_out (output)
= 4 buffers (2 reusable)

### Performance estimate

Current: 16 steps × (1 deriv dispatch + 4 vec_scale_add + 2 rk4_combine + 2 copy)
       = 16 × ~9 dispatches = 144 dispatches + 32 buffer copies

Perturbative: 1 dispatch (or 2 for two-pass variant)

At 768-dim, batch=8: 384 bands × 8 positions = 3072 threads.
One workgroup of 64 = 48 workgroups. Trivial for any GPU.

---

## Shader 2: matvec_block_diagonal_batch.wgsl

### What it replaces

Current: `matvec_batch.wgsl` with dense weight matrix
New: 6 groups of 128×128, fused into one dispatch

### WGSL Implementation

```wgsl
// Block-diagonal batched matvec: y[pos] = BlockDiag(W) @ x[pos] + b
//
// N groups of group_size dims each. Each thread computes one output element.
// Thread determines which group from its output index.
// Only reads weights from its group's block.

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
```

### Backward shaders

**d_x shader:** Same structure but transposed weight access.
**d_W shader:** Outer product per group — reuse outer_product.wgsl pattern.
(See original spec above for full WGSL — unchanged by refactor.)

### Rust-side: resident buffer integration (UPDATED for OutProjWeights enum)

**File: `src/wgpu_tier/resident.rs`**

The ResidentWeightBuffers now upload through the OutProjWeights enum methods:

```rust
// In the FfnWeights::KerrDualMaestro(w) arm of from_model():

// OLD (pre-enum — direct access):
// out_proj_w: create_buf(device, queue, &flatten_weights(&w.out_proj.w)),
// out_proj_b: create_buf(device, queue, &w.out_proj.b),

// NEW (through OutProjWeights enum):
out_proj_w: create_buf(device, queue, &w.out_proj.weights_flat()),
out_proj_b: create_buf(device, queue, &w.out_proj.bias_flat()),

// Same for update_from_model():
// OLD:
// queue.write_buffer(out_proj_w, 0, bytemuck::cast_slice(&flatten_weights(&w.out_proj.w)));
// queue.write_buffer(out_proj_b, 0, bytemuck::cast_slice(&w.out_proj.b));

// NEW:
queue.write_buffer(out_proj_w, 0, bytemuck::cast_slice(&w.out_proj.weights_flat()));
queue.write_buffer(out_proj_b, 0, bytemuck::cast_slice(&w.out_proj.bias_flat()));
```

**Buffer size:** The resident buffer size changes based on the enum variant:

```rust
// Dense: n_embd * n_embd = 590,592 floats
// BlockDiagonal(6 groups): 6 * 128 * 128 = 98,304 floats
// Use: w.out_proj.param_count() - w.out_proj.dim()  (total minus bias)
// Or:  w.out_proj.weights_flat().len()
```

**Shader selection:** The dispatch code in `src/wgpu_tier/dispatch.rs` (or wherever
`kerr_dual_maestro_add` lives) needs to check which pipeline to use:

```rust
// Select out_proj shader based on OutProjWeights variant
if w.out_proj.is_block_diagonal() {
    // Use matvec_block_diagonal_batch pipeline
    // Params: group_size, n_groups, n_pos, n_embd
} else {
    // Use standard matvec_batch pipeline
    // Params: out_dim, in_dim, n_pos, use_bias
}
```

---

## Integration plan (UPDATED for post-refactor layout)

### Phase 1: Perturbative ODE shader (biggest win, simplest change)
1. Create `shaders/kerr_perturbative_batch.wgsl` (from spec above)
2. Add pipeline + layout to GpuBackend struct in `src/wgpu_tier/pipelines.rs`
3. Add `PerturbativeParams` uniform struct in `src/wgpu_tier/pipelines.rs`
4. Upload precomputed decay/cos_w/sin_w as resident buffers in `src/wgpu_tier/resident.rs`
5. Add `gpu_kerr_ode_perturbative_batch` in `src/wgpu_tier/ops_forward.rs`
6. Replace `gpu_kerr_ode_batch_fused` call in `src/wgpu_tier/dispatch.rs` or `gpu_backend.rs`
7. Remove deinterleave/reinterleave dispatches (handle in Rust or in shader)
8. Test: `--gpu --layers 4 --iters 100` — loss must descend, no NaN, match CPU within 0.3

### Phase 2: Block-diagonal out_proj shader
1. Create `shaders/matvec_block_diagonal_batch.wgsl`
2. Create `shaders/matvec_block_diagonal_backward_batch.wgsl`
3. Add pipelines to GpuBackend in `src/wgpu_tier/pipelines.rs`
4. Update resident weight upload in `src/wgpu_tier/resident.rs` to use `weights_flat()`
5. Add shader selection in dispatch based on `w.out_proj.is_block_diagonal()`
6. Test: `--gpu --out-proj-groups 6 --layers 4 --iters 100`

### Phase 3: Cleanup
1. Remove old RK4 scratch buffer allocations from `src/wgpu_tier/ops_forward.rs`
2. Remove vec_scale_add and rk4_combine pipelines from `src/wgpu_tier/pipelines.rs`
3. Simplify FfnFullBuffers in `src/wgpu_tier/ffn_full_gpu.rs`
4. Keep RK4 behind a `--rk4-ode` flag as fallback

### Testing

```bash
# Quick sanity: must not NaN, loss must descend
cargo run --release -- data/input.txt --gpu --layers 4 --iters 100 --seq 64 --no-curriculum

# With block-diagonal
cargo run --release -- data/input.txt --gpu --layers 4 --iters 100 --seq 64 --no-curriculum --out-proj-groups 6

# Cross-tier comparison (all should reach similar loss at 100 iters)
cargo run --release -- data/input.txt --layers 4 --iters 100 --seq 64 --no-curriculum           # CPU
cargo run --release -- data/input.txt --gpu --layers 4 --iters 100 --seq 64 --no-curriculum     # wgpu
cargo run --release --features candle-backend -- data/input.txt --candle --layers 4 --iters 100  # Candle
```

---

## Decisions for Code

1. **Single-pass perturbative** (recompute neighbours in-thread). Switch to two-pass
   only if numerical results diverge too much from Candle tier.

2. **Keep identity backward for now.** wgpu tier trains with identity backward + Adam.

3. **Resident buffer caching:** decay, cos_w, sin_w are constant for a given model.
   Upload once to ResidentWeightBuffers, reuse every block. Don't re-upload per dispatch.

4. **Don't remove RK4 code yet.** Keep behind `--rk4-ode` flag as proven fallback.

5. **Use OutProjWeights enum methods** for all weight access in resident buffers:
   `weights_flat()`, `bias_flat()`, `is_block_diagonal()`, `param_count()`.
   Never access `.w` or `.b` directly on KerrDualMaestroWeights.out_proj.

---

## What NOT to change

- Candle tier (`src/candle_tier/`): already uses perturbative ODE, don't touch
- CPU tier (`src/cpu/`): uses RK4-16 via `common/fft_ode.rs`, keep as-is
- Attention shaders: no changes
- Maestro shaders: no changes (use matvec_batch which stays)
- Training loop, optimizer, LR schedule: no changes
- OutProjWeights enum in `common/model.rs`: no changes (already correct)
