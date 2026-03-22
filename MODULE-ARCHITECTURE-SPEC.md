# WAVE-ENGINE MODULE ARCHITECTURE — Restructure Spec
# Date: 2026-03-22
# For: Code (Claude Code)
# Priority: Do this BEFORE block-diagonal consistency. This ENABLES that change.
# Goal: Clean module boundaries so tier-specific changes don't ripple everywhere.

---

## The Problem

Changing out_proj from dense to block-diagonal touches 13 files / 207 references
because every module directly accesses `out_proj.w` and `out_proj.b`. There's no
abstraction layer. When the internal structure changes, everything breaks.

## Current Module Map (36 files, flat)

```
src/
├── main.rs                 Entry point, CLI, orchestration
├── model.rs                Weight structs (shared types) ← THE HUB
│
├── SHARED (all tiers)
│   ├── backend.rs          ComputeBackend trait
│   ├── data.rs             Dataset loading
│   ├── bpe.rs              BPE tokenizer
│   ├── wave_embed.rs       Harmonic embeddings (frozen)
│   ├── wave_attn.rs        Harmonic attention (frozen)
│   ├── rng.rs              RNG
│   └── token_cache.rs      Token caching
│
├── CPU TIER
│   ├── wave_block.rs       CPU forward pass (FFN)
│   ├── fft_ode.rs          FFT-based ODE (CPU RK4)
│   └── backward.rs         Hand-derived gradients
│
├── WGPU TIER (11 files!)
│   ├── gpu.rs              WGPU device setup
│   ├── gpu_backend.rs      ComputeBackend impl
│   ├── gpu_buffers.rs      Buffer pool
│   ├── gpu_dispatch.rs     Dispatch methods (ComputeBackend)
│   ├── gpu_ops_forward.rs  Forward GPU ops
│   ├── gpu_ops_backward.rs Backward GPU ops
│   ├── gpu_pipelines.rs    Shader compilation + struct
│   ├── gpu_resident.rs     Resident weight buffers (37 out_proj refs!)
│   ├── gpu_persistent.rs   Persistent pipeline (19 out_proj refs!)
│   ├── gpu_validate.rs     GPU validation
│   ├── ffn_gpu.rs          FFN out_proj on GPU
│   ├── ffn_full_gpu.rs     Full FFN pipeline on GPU
│   └── ffn_backend.rs      FFN routing (CPU or GPU)
│
├── CANDLE TIER (self-contained — this is the model)
│   ├── candle_engine.rs    Candle training loop
│   ├── block_diagonal.rs   BlockDiagonalLinear for Candle
│   └── gpu_ode.rs          GPU-native perturbative ODE
│
├── TRAINING
│   ├── train.rs            Training loop
│   ├── init.rs             Weight initialization
│   ├── optim.rs            Adam optimizer (30 out_proj refs!)
│   ├── pipeline.rs         Forward/backward pipeline (46 out_proj refs!)
│   └── monitor.rs          Pipeline monitor
│
├── CHECKPOINT
│   ├── wave_checkpoint.rs  WCHK save/load
│   └── checkpoint.rs       Legacy checkpoint
│
├── OTHER
│   ├── weights.rs          Binary weight loader (legacy)
│   └── grad_test.rs        Gradient validation
│
└── shaders/                WGSL compute shaders (17 files)
```

## The Core Problem: out_proj is Naked

Every module directly accesses:
```rust
weights.out_proj.w    // Vec<Vec<f32>> — assumes dense
weights.out_proj.b    // Vec<f32>
```

When out_proj changes from `LinearWeights` to `BlockDiagonalWeights`, ALL of
these break. The fix: out_proj should expose OPERATIONS, not data.

---

## The Fix: OutProj Trait

### Step 1: Define the abstraction

In model.rs, add a trait (or enum) that hides the internal structure:

```rust
/// Abstract out_proj — can be dense or block-diagonal.
/// All consumers use this interface. Nobody accesses .w or .b directly.
#[derive(Clone)]
pub enum OutProjWeights {
    Dense(LinearWeights),
    BlockDiagonal(BlockDiagonalWeights),
}

impl OutProjWeights {
    /// Forward pass: y = out_proj(x)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        match self {
            OutProjWeights::Dense(lw) => linear(&lw.w, &lw.b, x),
            OutProjWeights::BlockDiagonal(bd) => block_diagonal_forward(bd, x),
        }
    }

    /// Number of trainable parameters
    pub fn param_count(&self) -> usize {
        match self {
            OutProjWeights::Dense(lw) => lw.w.len() * lw.w[0].len() + lw.b.len(),
            OutProjWeights::BlockDiagonal(bd) => {
                bd.n_groups * (bd.group_size * bd.group_size + bd.group_size)
            }
        }
    }

    /// Flatten parameters into a Vec<f32> (for checkpoint save)
    pub fn flatten(&self) -> Vec<f32> {
        match self {
            OutProjWeights::Dense(lw) => {
                let mut v = Vec::new();
                for row in &lw.w { v.extend(row); }
                v.extend(&lw.b);
                v
            }
            OutProjWeights::BlockDiagonal(bd) => {
                let mut v = Vec::new();
                for g in &bd.groups {
                    for row in &g.w { v.extend(row); }
                    v.extend(&g.b);
                }
                v
            }
        }
    }

    /// Unflatten parameters from a slice (for checkpoint load)
    pub fn unflatten(&mut self, params: &[f32], offset: &mut usize) {
        match self {
            OutProjWeights::Dense(lw) => {
                let n = lw.w[0].len();
                for row in &mut lw.w {
                    row.copy_from_slice(&params[*offset..*offset+n]);
                    *offset += n;
                }
                lw.b.copy_from_slice(&params[*offset..*offset+lw.b.len()]);
                *offset += lw.b.len();
            }
            OutProjWeights::BlockDiagonal(bd) => {
                let gs = bd.group_size;
                for g in &mut bd.groups {
                    for row in &mut g.w {
                        row.copy_from_slice(&params[*offset..*offset+gs]);
                        *offset += gs;
                    }
                    g.b.copy_from_slice(&params[*offset..*offset+gs]);
                    *offset += gs;
                }
            }
        }
    }

    /// Get flat weight references for optimizer (mutable)
    /// Returns Vec of (weight_slice, bias_slice) pairs
    pub fn param_slices_mut(&mut self) -> Vec<(&mut Vec<Vec<f32>>, &mut Vec<f32>)> {
        match self {
            OutProjWeights::Dense(lw) => vec![(&mut lw.w, &mut lw.b)],
            OutProjWeights::BlockDiagonal(bd) => {
                bd.groups.iter_mut().map(|g| (&mut g.w, &mut g.b)).collect()
            }
        }
    }

    /// For GPU: flatten weights into a single buffer for upload
    pub fn flat_weights_for_gpu(&self) -> (Vec<f32>, Vec<f32>) {
        match self {
            OutProjWeights::Dense(lw) => {
                let w_flat: Vec<f32> = lw.w.iter().flat_map(|r| r.iter().copied()).collect();
                (w_flat, lw.b.clone())
            }
            OutProjWeights::BlockDiagonal(bd) => {
                let mut w_flat = Vec::new();
                let mut b_flat = Vec::new();
                for g in &bd.groups {
                    for row in &g.w { w_flat.extend(row); }
                    b_flat.extend(&g.b);
                }
                (w_flat, b_flat)
            }
        }
    }

    /// Dimensions
    pub fn in_dim(&self) -> usize {
        match self {
            OutProjWeights::Dense(lw) => lw.w[0].len(),
            OutProjWeights::BlockDiagonal(bd) => bd.n_groups * bd.group_size,
        }
    }

    pub fn out_dim(&self) -> usize {
        match self {
            OutProjWeights::Dense(lw) => lw.w.len(),
            OutProjWeights::BlockDiagonal(bd) => bd.n_groups * bd.group_size,
        }
    }

    /// Is this block-diagonal?
    pub fn is_block_diagonal(&self) -> bool {
        matches!(self, OutProjWeights::BlockDiagonal(_))
    }

    /// Number of groups (1 for dense)
    pub fn n_groups(&self) -> usize {
        match self {
            OutProjWeights::Dense(_) => 1,
            OutProjWeights::BlockDiagonal(bd) => bd.n_groups,
        }
    }

    /// Group size (= n_embd for dense)
    pub fn group_size(&self) -> usize {
        match self {
            OutProjWeights::Dense(lw) => lw.w[0].len(),
            OutProjWeights::BlockDiagonal(bd) => bd.group_size,
        }
    }
}
```

### Step 2: Update KerrDualMaestroWeights

```rust
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: OutProjWeights,  // WAS: LinearWeights
}
```

Same for KerrMaestroAddWeights and PerBandLinearWeights.

### Step 3: Replace direct access patterns

Every file that does this:
```rust
// OLD — direct access, assumes dense:
let out = linear(&weights.out_proj.w, &weights.out_proj.b, &x);
```

Becomes:
```rust
// NEW — goes through abstraction:
let out = weights.out_proj.forward(&x);
```

Every optimizer step that does:
```rust
// OLD — iterates rows of out_proj.w:
for row in &mut block.ffn.out_proj.w { ... }
```

Becomes:
```rust
// NEW — uses param_slices_mut:
for (w, b) in block.ffn.out_proj.param_slices_mut() {
    for row in w { adam_step(row, ...); }
    adam_step(b, ...);
}
```

### Step 4: GPU weight upload

gpu_resident.rs currently uploads:
```rust
// OLD:
let w_flat = flatten_dense(&weights.out_proj.w);
upload(w_flat, n_embd * n_embd);
```

Becomes:
```rust
// NEW:
let (w_flat, b_flat) = weights.out_proj.flat_weights_for_gpu();
upload(w_flat, weights.out_proj.param_count() - b_flat.len());
```

The GPU shader (matvec_batch or matvec_block_diagonal_batch) is selected based
on `weights.out_proj.is_block_diagonal()`.

---

## Module Restructure (Optional — Nice-to-Have)

The flat file layout works but is confusing. A cleaner structure:

```
src/
├── main.rs
├── lib.rs                  (re-exports)
│
├── common/
│   ├── mod.rs
│   ├── model.rs            Weight structs + OutProjWeights trait
│   ├── config.rs           ModelConfig (with out_proj_groups)
│   ├── data.rs             Dataset
│   ├── bpe.rs              Tokenizer
│   ├── embed.rs            Harmonic embeddings
│   ├── attn.rs             Harmonic attention
│   ├── rng.rs              RNG
│   └── checkpoint.rs       WCHK format (version-aware)
│
├── cpu/
│   ├── mod.rs
│   ├── forward.rs          CPU forward (was wave_block.rs)
│   ├── backward.rs         CPU backward (was backward.rs)
│   ├── ode.rs              FFT-based RK4 ODE (was fft_ode.rs)
│   └── backend.rs          CpuBackend impl
│
├── wgpu/
│   ├── mod.rs
│   ├── device.rs           WGPU setup (was gpu.rs)
│   ├── backend.rs          GpuBackend impl (was gpu_backend.rs + gpu_dispatch.rs)
│   ├── buffers.rs          Buffer pool (was gpu_buffers.rs)
│   ├── pipelines.rs        Shader compilation (was gpu_pipelines.rs)
│   ├── resident.rs         Resident weights (was gpu_resident.rs)
│   ├── forward.rs          Forward ops (was gpu_ops_forward.rs)
│   ├── backward.rs         Backward ops (was gpu_ops_backward.rs)
│   ├── ffn.rs              FFN pipeline (was ffn_gpu.rs + ffn_full_gpu.rs + ffn_backend.rs)
│   ├── persistent.rs       Persistent pipeline
│   └── validate.rs         Validation
│
├── candle/
│   ├── mod.rs
│   ├── engine.rs           Candle training (was candle_engine.rs)
│   ├── ode.rs              GPU-native perturbative ODE (was gpu_ode.rs)
│   └── block_diag.rs       BlockDiagonalLinear (was block_diagonal.rs)
│
├── training/
│   ├── mod.rs
│   ├── loop.rs             Training loop (was train.rs)
│   ├── init.rs             Weight init
│   ├── optim.rs            Adam optimizer
│   └── pipeline.rs         Forward/backward pipeline
│
└── shaders/                (unchanged)
```

This restructure is a BIG change and should NOT be done at the same time as
the block-diagonal work. It's a separate PR. The OutProjWeights enum is the
critical fix — it can be done in the flat structure first.

---

## Implementation Order

### Phase 1: OutProjWeights enum (enables block-diagonal everywhere)
1. Add `BlockDiagonalWeights` struct to model.rs
2. Add `OutProjWeights` enum with Dense/BlockDiagonal variants
3. Add all methods (forward, flatten, unflatten, param_slices_mut, flat_weights_for_gpu)
4. Change `KerrDualMaestroWeights.out_proj` type to `OutProjWeights`
5. Change `PerBandLinearWeights.out_proj` type to `OutProjWeights`

### Phase 2: Update consumers (file by file)
6. wave_block.rs — use .forward() (6 refs)
7. init.rs — create OutProjWeights::Dense or ::BlockDiagonal based on config (3 refs)
8. ffn_backend.rs — use .forward() (16 refs)
9. backward.rs — gradient through OutProjWeights
10. optim.rs — use param_slices_mut() (30 refs)
11. pipeline.rs — use .forward(), .param_count() (46 refs)
12. wave_checkpoint.rs — use .flatten()/.unflatten()
13. gpu_resident.rs — use .flat_weights_for_gpu() (37 refs)
14. gpu_persistent.rs — same pattern (19 refs)
15. gpu_dispatch.rs — use .forward() for CPU fallback (11 refs)
16. ffn_gpu.rs — update buffer sizes (4 refs)
17. ffn_full_gpu.rs — update pipeline (27 refs)
18. gpu_validate.rs — update validation (5 refs)
19. weights.rs — update loader (4 refs)

### Phase 3: Test all tiers
20. CPU: --layers 4 --iters 10 → loss descends, no NaN
21. wgpu: --gpu --layers 4 --iters 10 → loss descends, no NaN
22. Candle: --candle --layers 4 --iters 10 → loss descends, no NaN
23. Cross-tier: train on Candle, serve in wave-server

### Phase 4 (later): Module restructure into directories
- cpu/, wgpu/, candle/, common/, training/
- Separate PR, no functional changes

---

## Key Principle

**Nobody touches out_proj.w or out_proj.b directly.**
**Everybody goes through OutProjWeights methods.**

When we add a new out_proj variant in the future (sparse, low-rank, KAN),
we add a variant to the enum and implement the methods. Zero changes to
any consumer module.

---

## What This Enables

Once OutProjWeights is in place:
- Block-diagonal works on ALL tiers automatically
- --out-proj-groups flag controls which variant is created in init.rs
- WCHK v2 stores the variant type, checkpoint load creates the right enum
- GPU upload works for both dense and block-diagonal
- Optimizer works for both
- Future: sparse out_proj, low-rank out_proj, KAN out_proj — just add enum variant

This is the same principle as ComputeBackend (trait over CPU/GPU) but for
weight storage. ComputeBackend abstracts WHERE compute happens. OutProjWeights
abstracts HOW the out_proj is structured.
