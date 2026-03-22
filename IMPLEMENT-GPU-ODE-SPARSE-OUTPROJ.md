# IMPLEMENT NOW — GPU-Native ODE + Sparse out_proj
# Date: 2026-03-22
# For: Code (Claude Code)
# From: Desktop + Marco
# Priority: CRITICAL — implement for the next run after grammar finishes
#
# These should have been in the first IMPLEMENT-NOW spec. They are the
# two biggest remaining performance bottlenecks.

---

## Optimisation A: GPU-Native Perturbative ODE (ELIMINATE CPU TRANSFERS)

### The Problem

The current ODE CustomOp does this 48 times per iteration:

```rust
fn kerr_ode_batch(x: &Tensor, params: &OdeParams) -> Result<Tensor> {
    let x_cpu = x.to_device(&Device::Cpu)?;   // ← GPU→CPU TRANSFER
    let op = KerrOdeCustomOp { ... };
    let result = x_cpu.apply_op1(op)?;         // ← CPU computation
    result.to_device(x.device())               // ← CPU→GPU TRANSFER
}
```

That's 48 GPU↔CPU round trips per forward pass (24 blocks × 2 transfers each).
Plus another 48 in the backward pass. Total: ~96 data transfers per iteration.
Each transfer involves copying a [batch×seq_len, 768] tensor.

### The Solution

The perturbative ODE is just tensor math. Every operation can be expressed
as Candle tensor operations that run on GPU natively. No CustomOp needed.
No CPU transfer needed. The data stays on GPU the entire time.

### The Math (what the perturbative ODE does)

For input x of shape [n_pos, n_embd]:
1. Split into r and s components: r = x[..., 0::2], s = x[..., 1::2]  (384 bands each)
2. Linear solution: r_lin = exp(-γ) * (r*cos(ω) - s*sin(ω))
                    s_lin = exp(-γ) * (r*sin(ω) + s*cos(ω))
3. Self-phase: mag_sq = r_lin² + s_lin²
4. Cross-phase: neighbour_sum = shifted sums of mag_sq (±1, ±2 positions)
5. Correction: delta_phi = α * mag_sq + β * neighbour_sum
6. Output: r_out = r_lin - delta_phi * s_lin
           s_out = s_lin + delta_phi * r_lin
7. Interleave r_out and s_out back to [n_pos, n_embd]

### Implementation in Candle Tensor Ops

Replace the entire `kerr_ode_batch` function with this:

```rust
/// GPU-native perturbative ODE — zero CPU transfers.
/// All operations are Candle tensor ops that run on GPU via cuBLAS/CUDA.
fn kerr_ode_gpu(x: &Tensor, params: &GpuOdeParams) -> Result<Tensor> {
    let (n_pos, n_embd) = x.dims2()?;
    let n_bands = n_embd / 2;

    // Split into r (even indices) and s (odd indices)
    // x shape: [n_pos, n_embd] → r shape: [n_pos, n_bands], s shape: [n_pos, n_bands]
    let x_reshaped = x.reshape((n_pos, n_bands, 2))?;
    let r = x_reshaped.narrow(2, 0, 1)?.squeeze(2)?;  // [n_pos, n_bands]
    let s = x_reshaped.narrow(2, 1, 1)?.squeeze(2)?;  // [n_pos, n_bands]

    // Step 1: Linear solution (damping + rotation)
    // params.decay, params.cos_w, params.sin_w are precomputed tensors [1, n_bands]
    let r_lin = ((&r * &params.cos_w)? - (&s * &params.sin_w)?)?.broadcast_mul(&params.decay)?;
    let s_lin = ((&r * &params.sin_w)? + (&s * &params.cos_w)?)?.broadcast_mul(&params.decay)?;

    // Step 2: Self-phase modulation
    let mag_sq = (&r_lin * &r_lin + &s_lin * &s_lin)?;  // [n_pos, n_bands]

    // Step 3: Cross-phase modulation (neighbour sum)
    // Pad mag_sq with zeros on both sides, then sum shifted versions
    let zeros = Tensor::zeros((n_pos, 2), DType::F32, x.device())?;
    let padded = Tensor::cat(&[&zeros, &mag_sq, &zeros], 1)?;  // [n_pos, n_bands+4]

    // Neighbours: positions k-2, k-1, k+1, k+2 relative to k
    // With padding of 2, original band k is at position k+2 in padded
    // So k-2 = padded[k], k-1 = padded[k+1], k+1 = padded[k+3], k+2 = padded[k+4]
    let ns_m2 = padded.narrow(1, 0, n_bands)?;      // k-2
    let ns_m1 = padded.narrow(1, 1, n_bands)?;      // k-1
    let ns_p1 = padded.narrow(1, 3, n_bands)?;      // k+1
    let ns_p2 = padded.narrow(1, 4, n_bands)?;      // k+2
    let neighbour_sum = (ns_m2 + ns_m1 + ns_p1 + ns_p2)?;

    // Step 4: Phase correction
    let delta_phi = (mag_sq * params.alpha as f64 + neighbour_sum * params.beta as f64)?;

    // Step 5: Apply correction
    let r_out = (&r_lin - &delta_phi * &s_lin)?;
    let s_out = (&s_lin + &delta_phi * &r_lin)?;

    // Step 6: Interleave back to [n_pos, n_embd]
    let r_expanded = r_out.unsqueeze(2)?;  // [n_pos, n_bands, 1]
    let s_expanded = s_out.unsqueeze(2)?;  // [n_pos, n_bands, 1]
    let interleaved = Tensor::cat(&[&r_expanded, &s_expanded], 2)?;  // [n_pos, n_bands, 2]
    let output = interleaved.reshape((n_pos, n_embd))?;

    Ok(output)
}

/// Precomputed ODE parameters as GPU tensors (computed once at model init).
struct GpuOdeParams {
    decay: Tensor,    // [1, n_bands] — exp(-softplus(gamma_raw))
    cos_w: Tensor,    // [1, n_bands] — cos(omega)
    sin_w: Tensor,    // [1, n_bands] — sin(omega)
    alpha: f32,
    beta: f32,
}

impl GpuOdeParams {
    fn from_ode_params(params: &OdeParams, device: &Device) -> Result<Self> {
        let n_bands = params.gamma_raw.len();
        fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }

        let decay_vals: Vec<f32> = params.gamma_raw.iter()
            .map(|&g| (-softplus(g)).exp()).collect();
        let cos_vals: Vec<f32> = params.omega.iter().map(|&w| w.cos()).collect();
        let sin_vals: Vec<f32> = params.omega.iter().map(|&w| w.sin()).collect();

        Ok(Self {
            decay: Tensor::from_vec(decay_vals, (1, n_bands), device)?,
            cos_w: Tensor::from_vec(cos_vals, (1, n_bands), device)?,
            sin_w: Tensor::from_vec(sin_vals, (1, n_bands), device)?,
            alpha: params.alpha,
            beta: params.beta,
        })
    }
}
```

### What changes in the model

1. **At model init:** Add `gpu_ode_params: GpuOdeParams` to each `CandleBlock`.
   Precompute once from the existing `OdeParams`.

2. **In forward pass:** Replace:
   ```rust
   let effective_ode_out = kerr_ode_batch(&precond, &block.ode_params)?;
   ```
   with:
   ```rust
   let effective_ode_out = kerr_ode_gpu(&precond, &block.gpu_ode_params)?;
   ```

3. **Remove:** The entire `KerrOdeCustomOp` struct and `kerr_ode_batch` function.
   They're no longer needed.

4. **Backward:** Candle's autograd handles this automatically. All the operations
   (multiply, add, narrow, cat, reshape) have built-in backward implementations.
   The ODE params (decay, cos_w, sin_w) are constants (not in VarMap), so no
   gradients flow to them. The gradient flows through r_out and s_out back to
   the precond input — same identity-backward behaviour as before, but now
   Candle computes it on GPU instead of us manually specifying it.

### Expected Impact

| Metric | Current (CPU CustomOp) | GPU-native |
|--------|----------------------|------------|
| GPU↔CPU transfers | 96 per iteration | **0** |
| ODE compute location | CPU (sequential per sample) | **GPU (parallel all samples)** |
| Synchronisation points | 48 per forward pass | **0** |
| ODE time estimate | ~0.5s/iter | **~0.01s/iter** |

The GPU-native ODE eliminates every CPU↔GPU transfer in the forward pass.
The attention still runs on CPU, but that's separate.

### Risks and Mitigations

**Risk 1:** Candle's narrow/cat/reshape may create intermediate tensors.
**Mitigation:** With device.synchronize() after optimizer step, intermediates
are freed. The VRAM overhead is small (a few [n_pos, n_bands] tensors).

**Risk 2:** The padded neighbour sum may have edge effects.
**Mitigation:** The zero-padding handles boundaries correctly — band 0 has
no k-2 or k-1 neighbours (they're zero), same as the CPU version's
`if k >= 2` checks.

**Risk 3:** Numerical precision may differ slightly from CPU.
**Mitigation:** The perturbative ODE has MSE 0.000005 vs RK4-16.
GPU f32 vs CPU f32 differences are 1e-7 level. Not a concern.

---

## Optimisation B: Sparse out_proj (BANDED MATRIX)

### The Problem

Each block has a dense out_proj: Linear(768, 768) = 589,824 parameters.
24 blocks × 589,824 = 14.2M parameters in out_proj alone.
This is 39% of iter time — the single biggest compute bottleneck.

### The Solution

Replace dense out_proj with a banded matrix. Instead of every output dim
connecting to every input dim (768×768), each output dim connects only to
nearby input dims (within a bandwidth of W).

### Why This Should Work

The ODE creates structured output — nearby bands are correlated through
the ±2 stencil coupling. Band 50's output is physically related to bands
48-52. A banded out_proj respects this structure by only mixing nearby bands.

### Implementation

Candle doesn't have a native banded matmul, but we can implement it
efficiently using grouped convolutions or manual masking.

**Approach A: Mask the weight matrix (simplest, for testing)**

```rust
/// Banded linear layer — masks dense weight to zero outside bandwidth.
/// During training, gradients only flow to non-zero weights.
struct BandedLinear {
    inner: Linear,       // Standard Candle Linear (in VarMap, gets gradients)
    mask: Tensor,        // Binary mask [out_dim, in_dim] — 1.0 within band, 0.0 outside
}

impl BandedLinear {
    fn new(in_dim: usize, out_dim: usize, bandwidth: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let inner = linear_uniform(in_dim, out_dim, vb)?;

        // Create banded mask: output[i] connects to input[i-bw..i+bw]
        let mut mask_data = vec![0.0f32; out_dim * in_dim];
        let half_bw = bandwidth / 2;
        for i in 0..out_dim {
            let start = if i >= half_bw { i - half_bw } else { 0 };
            let end = (i + half_bw + 1).min(in_dim);
            for j in start..end {
                mask_data[i * in_dim + j] = 1.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, (out_dim, in_dim), device)?;

        Ok(Self { inner, mask })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply mask to weight before matmul
        // masked_weight = weight * mask (zero out connections outside bandwidth)
        let w = self.inner.weight();
        let masked_w = (w * &self.mask)?;
        let b = self.inner.bias().unwrap();

        // Manual matmul with masked weight: x @ masked_w.T + b
        x.matmul(&masked_w.t()?)?.broadcast_add(b)
    }
}
```

**The mask approach has a subtlety:** The full 768×768 weight matrix still
exists in memory and gets gradients computed for all 589K entries. The mask
zeros them out in forward, so the model learns as if banded, but the memory
and compute savings only come from the sparsity pattern, not from a smaller
matrix.

**Approach B: Block-diagonal (more efficient)**

Reshape the 768-dim vector into groups, apply small dense matmuls per group:

```rust
/// Block-diagonal out_proj — groups of bands processed independently.
/// 768 dims / 12 groups = 64 dims per group.
/// Each group: Linear(64, 64) = 4,096 params.
/// Total: 12 × 4,096 = 49,152 params (vs 589,824 dense = 12x reduction).
struct BlockDiagonalLinear {
    groups: Vec<Linear>,
    n_groups: usize,
    group_size: usize,
}

impl BlockDiagonalLinear {
    fn new(n_embd: usize, n_groups: usize, vb: VarBuilder) -> Result<Self> {
        let group_size = n_embd / n_groups;
        let mut groups = Vec::new();
        for g in 0..n_groups {
            let group_vb = vb.pp(format!("group_{g}"));
            groups.push(linear_uniform(group_size, group_size, group_vb)?);
        }
        Ok(Self { groups, n_groups, group_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n_pos, n_embd) = x.dims2()?;
        let mut outputs = Vec::new();

        for g in 0..self.n_groups {
            let start = g * self.group_size;
            let group_input = x.narrow(1, start, self.group_size)?;
            let group_output = self.groups[g].forward(&group_input)?;
            outputs.push(group_output);
        }

        Tensor::cat(&outputs, 1)  // Concatenate group outputs along dim 1
    }
}
```

**Approach B is recommended because:**
- True parameter reduction (49K vs 589K per block)
- True compute reduction (12 small matmuls vs 1 large matmul)
- cuBLAS handles batched small matmuls efficiently
- Clean Candle implementation with proper VarMap integration
- Each group learns to mix nearby bands — matches the ODE stencil structure

### Configuration

The number of groups controls the tradeoff:

| Groups | Group size | Params/block | vs Dense | Bandwidth equiv |
|--------|-----------|-------------|----------|-----------------|
| 6 | 128 | 98K | 6x fewer | Each band mixes with ±64 neighbours |
| 12 | 64 | 49K | 12x fewer | Each band mixes with ±32 neighbours |
| 24 | 32 | 25K | 24x fewer | Each band mixes with ±16 neighbours |
| 48 | 16 | 12K | 48x fewer | Each band mixes with ±8 neighbours |

Start with 12 groups (64 dims each) — matches the 12 attention heads.

### What Changes in the Model

In `CandleBlock`, replace:
```rust
out_proj: Linear,  // 768×768 dense
```
with:
```rust
out_proj: BlockDiagonalLinear,  // 12 groups × 64×64
```

The VarMap names change from `block.N.out_proj.weight/bias` to
`block.N.out_proj.group_M.weight/bias`. This means checkpoints from
the old format won't load — fresh init required. That's fine for the
next run.

### Expected Impact

| Metric | Current (dense) | Block-diagonal (12 groups) |
|--------|----------------|---------------------------|
| out_proj params/block | 589,824 | **49,152** (12x fewer) |
| out_proj total (24 blocks) | 14.2M | **1.2M** |
| Total model params (char) | ~15.5M | **~2.5M** |
| out_proj compute/block | 768×768 matmul | 12 × 64×64 matmuls |
| Share of iter time | 39% | **~5%** |

### Risk

**The big risk:** If the model needs long-range band mixing (band 50 talks to
band 300), block-diagonal kills that. The loss will degrade.

**The test:** Run 200 iters with block-diagonal. Compare loss at iter 200
against the dense baseline. If loss is within 0.2: block-diagonal works.
If loss is > 0.5 worse: increase group size or revert to dense.

**The lab bench already gave us a clue:** The perturbative ODE (which respects
local structure) trains BETTER than RK4-16. This suggests the architecture
benefits from locality. Block-diagonal out_proj is the same principle
applied to the translator layer.

---

## Implementation Order for Code

### Step 1: GPU-native ODE (highest priority)
1. Add `GpuOdeParams` struct with precomputed tensors
2. Add `kerr_ode_gpu` function using Candle tensor ops
3. Precompute `GpuOdeParams` in model init for each block
4. Replace `kerr_ode_batch` call with `kerr_ode_gpu` in forward
5. Remove the `KerrOdeCustomOp` struct entirely
6. Test: 10 iters, compare loss and VRAM against current

### Step 2: Block-diagonal out_proj (after GPU ODE verified)
1. Add `BlockDiagonalLinear` struct
2. Replace `out_proj: Linear` with `BlockDiagonalLinear` in CandleBlock
3. Update model init to create grouped linears
4. Update `extract_wchk_params` for new weight layout
5. Test: 200 iters on grammar corpus, compare loss against dense

### Step 3: Verify both together
1. Run 500 iters with both optimisations on grammar corpus
2. Compare: loss trajectory, VRAM usage, iter time
3. Expected: 3-4x faster iter time, 80% fewer params, similar loss

---

## What NOT to Change

- Don't touch the attention path (frozen, CPU — separate optimisation)
- Don't change the maestro dimensions (validated at 16)
- Don't modify the training loop, optimizer, LR schedule, or checkpointing
- Don't change the tokenizer or data loading

Only the ODE forward and out_proj are modified. Everything else stays.

---

## Expected Combined Impact

| | Current | After both optimisations |
|---|---|---|
| GPU↔CPU transfers | 96/iter | **~0** (only attention remains on CPU) |
| ODE compute | CPU, 0.5s | **GPU, ~0.01s** |
| out_proj params | 14.2M | **1.2M** |
| out_proj compute | 39% of iter | **~5% of iter** |
| Total model params (char) | 15.5M | **~2.5M** |
| VRAM estimate | 1873MB | **~1400MB** |
| Iter time estimate | 7.7s | **3-4s** |
| 3000 iters | 6.4 hours | **~3 hours** |

A 2.5M parameter model that trains in 3 hours on a consumer GPU.
That's the wave-engine's promise delivered.
