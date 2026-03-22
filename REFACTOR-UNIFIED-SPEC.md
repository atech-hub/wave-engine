# WAVE-ENGINE REFACTOR — Final Unified Spec
# Date: 2026-03-22
# Authors: Desktop (Opus) + Code — aligned through Marco
# Status: APPROVED — ready for execution
# Rule: NO CODE until this spec is reviewed. Execute phases in order.

---

## Executive Summary

The wave-engine has 36 source files in a flat directory. Changing out_proj from
dense to block-diagonal touched 13 files / 207 references and corrupted main.rs.
This spec fixes the root cause (no abstraction) and cleans up the codebase.

**Two principles:**
1. **Decouple first, reorganise second** — the OutProjWeights enum breaks the
   coupling. Directory restructure happens after, as a separate step.
2. **550-line file limit** — enforced during restructure. One or two exceptions
   tolerated, but every file should target ≤550 lines.

---

## Phase 1: OutProjWeights Enum (model.rs)
**Time estimate: 30 minutes**
**Risk: Low — additive, nothing breaks**

### 1a. Add BlockDiagonalWeights struct

```rust
/// Block-diagonal linear — groups of bands processed independently.
#[derive(Clone)]
pub struct BlockDiagonalWeights {
    pub groups: Vec<LinearWeights>,  // n_groups × (group_size → group_size)
    pub n_groups: usize,
    pub group_size: usize,
}
```

### 1b. Add OutProjWeights enum

```rust
/// Abstract out_proj — dense or block-diagonal.
/// ALL consumers use this interface. Nobody accesses .w or .b directly.
#[derive(Clone)]
pub enum OutProjWeights {
    Dense(LinearWeights),
    BlockDiagonal(BlockDiagonalWeights),
}
```

### 1c. Implement methods on OutProjWeights

Required methods (all must be implemented):

```rust
impl OutProjWeights {
    /// CPU forward: y = out_proj(x)
    pub fn forward(&self, x: &[f32]) -> Vec<f32>;

    /// Batched CPU forward: y[pos] = out_proj(x[pos]) for all positions
    pub fn forward_batch(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Total trainable parameter count
    pub fn param_count(&self) -> usize;

    /// Flatten all parameters into a contiguous Vec<f32> (checkpoint save)
    pub fn flatten_into(&self, out: &mut Vec<f32>);

    /// Unflatten parameters from a slice (checkpoint load)
    /// Advances the offset past consumed parameters.
    pub fn unflatten_from(&mut self, params: &[f32], offset: &mut usize);

    /// Mutable access to weight/bias pairs for optimizer
    /// Dense: returns 1 pair. BlockDiag: returns n_groups pairs.
    pub fn param_groups_mut(&mut self) -> Vec<(&mut Vec<Vec<f32>>, &mut Vec<f32>)>;

    /// Flat weight buffer for GPU upload (concatenated group weights)
    pub fn weights_flat(&self) -> Vec<f32>;

    /// Flat bias buffer for GPU upload (concatenated group biases = full bias)
    pub fn bias_flat(&self) -> Vec<f32>;

    /// Input/output dimension (same for square projection)
    pub fn dim(&self) -> usize;

    /// Number of groups (1 for dense)
    pub fn n_groups(&self) -> usize;

    /// Size of each group (= dim for dense)
    pub fn group_size(&self) -> usize;

    /// Is this block-diagonal?
    pub fn is_block_diagonal(&self) -> bool;
}
```

### 1d. Update weight structs to use the enum

```rust
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: OutProjWeights,       // WAS: LinearWeights
}

pub struct KerrMaestroAddWeights {
    pub kerr: KerrWeights,
    pub maestro: MaestroWeights,
    pub out_proj: OutProjWeights,       // WAS: LinearWeights
}

pub struct PerBandLinearWeights {
    pub band_w: Vec<[[f32; 2]; 2]>,
    pub band_b: Vec<[f32; 2]>,
    pub out_proj: OutProjWeights,       // WAS: LinearWeights
}
```

### 1e. Add out_proj_groups to ModelConfig

```rust
pub struct ModelConfig {
    pub n_bands: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_n_steps: usize,
    pub out_proj_groups: usize,  // NEW — 1=dense, 6=block-diagonal (768-dim)
}
```

Default: `out_proj_groups: 1` for `default_128()`, `out_proj_groups: 6` for 768-dim.

### 1f. Test: `cargo build --release` compiles with zero warnings

At this point nothing is USING the enum yet — the old code still compiles
because the struct field types changed but the variants wrap the same data.
Wait — actually changing the type of `out_proj` from `LinearWeights` to
`OutProjWeights` WILL break all consumers. So Phase 1 should:

**OPTION A (safe):** Add the enum but DON'T change the struct fields yet.
Add a `from_linear()` constructor and `to_linear()` accessor so Phase 2
can migrate file by file.

**OPTION B (faster):** Change the struct fields AND update all consumers
in one pass. Riskier but fewer intermediate states.

**Decision: OPTION A.** Add the enum and all methods. Add a temporary
`KerrDualMaestroWeights.out_proj_new: Option<OutProjWeights>` field.
Migrate consumers one at a time. When all consumers use `out_proj_new`,
rename it to `out_proj` and delete the old field. This way cargo build
passes after every file change.

Actually — simpler approach: **just change the field type and fix compiler
errors one file at a time.** The Rust compiler will tell you exactly which
lines break. Each fix is mechanical: `out_proj.w` → match on the enum or
call a method. The compiler is the migration tool.

**Final decision: Change the field type. Fix compiler errors file by file.
Commit after each file compiles.** This is the Rust way.

---

## Phase 2: Update Active Consumers
**Time estimate: 1-2 hours**
**Risk: Medium — compiler guides every change**

### Identify the 6 active files (ordered by reference count)

These are the files that are ACTUALLY USED in the current training paths.
Legacy/dead files are handled in Phase 4.

| File | out_proj refs | What to change |
|------|-------------|----------------|
| wave_block.rs | 6 | `linear(&op.w, &op.b, x)` → `op.forward(x)` |
| init.rs | 3 | Create `OutProjWeights::Dense(...)` or `::BlockDiagonal(...)` based on config |
| ffn_backend.rs | 16 | `linear(&op.w, &op.b, x)` → `op.forward(x)`, batch versions |
| train.rs / optim.rs | 30 | Adam step: iterate `op.param_groups_mut()` |
| wave_checkpoint.rs | ~10 | Save: `op.flatten_into()`. Load: `op.unflatten_from()` |
| backend.rs | ~5 | Trait methods that reference out_proj dimensions |

### For each file:

1. Change the code to use OutProjWeights methods
2. `cargo build --release` — must compile
3. Quick smoke test: `cargo run --release -- data/input.txt --layers 4 --iters 3 --seq 64 --no-curriculum`
4. Commit: `"refactor: migrate {filename} to OutProjWeights enum"`

### Patterns to search and replace:

```rust
// FORWARD (most common — ~40 of the 50 refs)
// OLD:
linear(&weights.out_proj.w, &weights.out_proj.b, &x)
// or:
let w = &block.ffn.out_proj.w;
let b = &block.ffn.out_proj.b;
let out = linear(w, b, &x);
// NEW:
weights.out_proj.forward(&x)

// PARAM COUNT
// OLD:
n_embd * n_embd + n_embd  // out_proj weight + bias
// NEW:
weights.out_proj.param_count()

// FLATTEN (checkpoint save)
// OLD:
for row in &block.ffn.out_proj.w { flat.extend(row); }
flat.extend(&block.ffn.out_proj.b);
// NEW:
block.ffn.out_proj.flatten_into(&mut flat);

// UNFLATTEN (checkpoint load)
// OLD:
for row in &mut block.ffn.out_proj.w { row.copy_from_slice(&params[idx..idx+n]); idx += n; }
block.ffn.out_proj.b.copy_from_slice(&params[idx..idx+n]); idx += n;
// NEW:
block.ffn.out_proj.unflatten_from(&params, &mut idx);

// OPTIMIZER (Adam step on weights)
// OLD:
for (i, row) in block.ffn.out_proj.w.iter_mut().enumerate() {
    for (j, val) in row.iter_mut().enumerate() { adam_step(val, ...); }
}
for val in block.ffn.out_proj.b.iter_mut() { adam_step(val, ...); }
// NEW:
for (w, b) in block.ffn.out_proj.param_groups_mut() {
    for row in w.iter_mut() {
        for val in row.iter_mut() { adam_step(val, ...); }
    }
    for val in b.iter_mut() { adam_step(val, ...); }
}

// GPU UPLOAD
// OLD:
let w_flat: Vec<f32> = weights.out_proj.w.iter().flat_map(|r| r.iter().copied()).collect();
gpu.upload(&w_flat);
// NEW:
let w_flat = weights.out_proj.weights_flat();
gpu.upload(&w_flat);
```

### Legacy files — DON'T update, just make them compile

For dead legacy files (pipeline.rs, gpu_persistent.rs, etc.), the quickest
fix is to add a temporary accessor:

```rust
impl OutProjWeights {
    /// TEMPORARY — legacy compatibility. Remove when legacy files are deleted.
    pub fn as_linear(&self) -> &LinearWeights {
        match self {
            OutProjWeights::Dense(lw) => lw,
            _ => panic!("Legacy code requires Dense out_proj"),
        }
    }
    pub fn as_linear_mut(&mut self) -> &mut LinearWeights {
        match self {
            OutProjWeights::Dense(lw) => lw,
            _ => panic!("Legacy code requires Dense out_proj"),
        }
    }
}
```

Then legacy code changes from `out_proj.w` to `out_proj.as_linear().w`.
This compiles, doesn't change behaviour, and the panic never fires because
legacy code is never called. These accessors get deleted in Phase 4.

---

## Phase 3: Test All Tiers
**Time estimate: 10 minutes**
**Risk: Low — validation only**

```bash
# CPU tier (RK4-16, uses wave_block.rs + ffn_backend.rs)
cargo run --release -- data/input.txt --layers 4 --iters 10 --seq 64 --no-curriculum
# Expected: loss ~4.4 → ~3.2, no NaN

# wgpu GPU tier (RK4-16 fused, uses gpu_dispatch.rs)
cargo run --release -- data/input.txt --layers 4 --iters 10 --seq 64 --no-curriculum --gpu
# Expected: loss ~4.4 → ~3.5, no NaN

# Candle CUDA tier (perturbative ODE, uses candle_engine.rs)
cargo run --release --features candle-backend -- data/input.txt --candle --layers 4 --iters 10 --seq 64 --no-curriculum
# Expected: loss ~4.8 → ~4.7 (warmup), no NaN

# Block-diagonal test (Candle, 6 groups)
cargo run --release --features candle-backend -- data/input.txt --candle --layers 4 --iters 10 --seq 64 --no-curriculum --out-proj-groups 6
# Expected: compiles and runs, loss descends

# Cross-tier: train on Candle with block-diag, load in wave-server
# (if WCHK v2 is implemented in this phase)
```

All three must pass before proceeding. If any tier fails, fix it before Phase 4.

---

## Phase 4: Delete Dead Legacy Modules
**Time estimate: 5 minutes**
**Risk: Low — dead code removal**

### Files to delete (7 files, ~4,895 lines):

| File | Lines | Reason |
|------|-------|--------|
| pipeline.rs | 1223 | Dead — kerr-engine forward/backward cache, replaced by train.rs |
| gpu_dispatch.rs | 1055 | Dead — old ComputeBackend dispatch, replaced by gpu_ops_forward.rs |
| gpu_persistent.rs | 736 | Dead — old persistent pipeline, never called |
| ffn_full_gpu.rs | 691 | Dead — disabled full FFN GPU pipeline |
| backward.rs | 554 | Dead — kerr-engine backward, replaced by train.rs backward |
| grad_test.rs | ~100 | Dead — gradient validation against PyTorch |
| weights.rs | ~100 | Dead — binary weight loader for legacy format |

### How to verify they're dead:

```bash
# Check if any active code imports from these modules
grep -rn "use crate::pipeline" src/main.rs src/train.rs src/candle_engine.rs
grep -rn "use crate::backward" src/main.rs src/train.rs
# etc. — if no active file imports them, they're dead
```

### After deletion:

1. Remove `mod` declarations from main.rs / lib.rs
2. `cargo build --release --features candle-backend` — must compile
3. Run Phase 3 tests again — all tiers must still pass
4. Remove `as_linear()` / `as_linear_mut()` temporary accessors from OutProjWeights
5. Commit: `"cleanup: delete 7 dead legacy modules (~4,900 lines)"`

---

## Phase 5: WCHK v2 Checkpoint Format
**Time estimate: 30 minutes**
**Risk: Medium — format change, backward compatibility**

### Changes to wave_checkpoint.rs:

1. Bump version: `WCHK` v1 → v2
2. Add `out_proj_groups` to header (after rk4_n_steps)
3. Param count formula uses `OutProjWeights::param_count()` instead of hardcoded `n_embd * n_embd`
4. Save: `block.ffn.out_proj.flatten_into(&mut flat)` (works for both variants)
5. Load: read `out_proj_groups` from header, create correct enum variant, `unflatten_from()`

### Backward compatibility:

- v1 checkpoints: assume `out_proj_groups = 1` (dense). Load into `OutProjWeights::Dense(...)`.
- v2 checkpoints: read groups from header. Create appropriate variant.
- Old checkpoints still load. New checkpoints include the group info.

### wave-server checkpoint.rs:

- Read `out_proj_groups` from WCHK v2 header (currently hardcodes 6)
- Create `BlockDiagonalWeights` or `LinearWeights` based on header value
- Both Candle and CPU/wgpu checkpoints load correctly

---

## Phase 6: Directory Restructure (SEPARATE PR)
**Time estimate: 2-3 hours**
**Risk: Medium — many file moves, import changes**

### Target structure:

```
src/
├── main.rs                 CLI dispatch only (≤300 lines)
│
├── core/                   SHARED — all tiers import from here
│   ├── mod.rs
│   ├── model.rs            Weight structs + OutProjWeights (≤550 lines)
│   ├── primitives.rs       linear(), layer_norm(), gelu(), softplus() (≤300 lines)
│   ├── config.rs           ModelConfig (≤100 lines)
│   ├── embed.rs            Harmonic embeddings (from wave_embed.rs)
│   ├── attn.rs             Harmonic attention (from wave_attn.rs)
│   ├── ffn.rs              Dual-maestro + ODE forward — shared CPU impl (from wave_block.rs)
│   ├── checkpoint.rs       WCHK v2 save/load (from wave_checkpoint.rs)
│   ├── data.rs             Dataset loading
│   ├── tokenizer.rs        BPE + char-level (from bpe.rs + token_cache.rs)
│   ├── rng.rs              Deterministic RNG
│   └── monitor.rs          Pipeline timing
│
├── cpu/                    CPU tier
│   ├── mod.rs
│   ├── train.rs            CPU training loop (from train.rs)
│   ├── backward.rs         CPU gradient computation (from current backward logic in train.rs)
│   └── adam.rs             CPU Adam optimizer (from optim.rs)
│
├── wgpu/                   wgpu GPU tier
│   ├── mod.rs
│   ├── device.rs           WGPU setup (from gpu.rs, ≤200 lines)
│   ├── backend.rs          GpuBackend struct + ComputeBackend impl (from gpu_backend.rs)
│   ├── pipelines.rs        Shader compilation (from gpu_pipelines.rs, ≤550 lines)
│   ├── forward.rs          Forward dispatch ops (from gpu_ops_forward.rs, ≤550 lines)
│   ├── ode.rs              Fused ODE dispatch — RK4 + perturbative (split from gpu_ops_forward.rs, ~200 lines)
│   ├── backward.rs         Backward dispatch ops (from gpu_ops_backward.rs)
│   ├── buffers.rs          Buffer pool (from gpu_buffers.rs)
│   ├── resident.rs         Resident weight buffers (from gpu_resident.rs)
│   ├── ffn.rs              FFN routing + ping-pong (from ffn_backend.rs + ffn_gpu.rs)
│   └── validate.rs         GPU validation (from gpu_validate.rs)
│
├── candle/                 Candle CUDA tier
│   ├── mod.rs
│   ├── model.rs            Candle model construction (split from candle_engine.rs, ≤400 lines)
│   ├── train.rs            Candle training loop (split from candle_engine.rs, ≤400 lines)
│   ├── ode.rs              GPU-native perturbative ODE (from gpu_ode.rs)
│   └── block_diag.rs       Candle BlockDiagonalLinear (from block_diagonal.rs)
│
└── shaders/                WGSL compute shaders (unchanged)
```

### File splits required (550-line limit):

| Original | Lines | Split into | Target lines |
|----------|-------|-----------|-------------|
| model.rs | 889 | core/model.rs + core/primitives.rs | ~550 + ~340 |
| candle_engine.rs | 794 | candle/model.rs + candle/train.rs | ~400 + ~400 |
| gpu_ops_forward.rs | 751 | wgpu/forward.rs + wgpu/ode.rs | ~550 + ~200 |
| gpu_pipelines.rs | 930 | wgpu/pipelines.rs (trim dead code) | ≤550 |
| main.rs | 1062 | main.rs (CLI only) + tier dispatch in modules | ≤300 |

### Execution:

1. Create directory structure: `mkdir -p src/{core,cpu,wgpu,candle}`
2. Move files one at a time, updating `mod.rs` and `use` paths
3. `cargo build` after each move — must compile
4. Split oversized files
5. Final test: all three tiers pass
6. Commit: `"refactor: restructure into core/cpu/wgpu/candle modules"`

---

## Rules for Code

1. **No sed on source files.** Use proper editor commands or write the full file.
2. **Commit after each file migration** in Phase 2. If something breaks, git revert one file.
3. **Compiler is the guide.** Change the type, fix what the compiler says, move on.
4. **Don't optimise during refactor.** No logic changes. Same behaviour, cleaner structure.
5. **Test after every phase.** All three tiers must pass before proceeding.
6. **550-line limit.** If a file exceeds this after migration, split it immediately.
7. **No dead code.** If it's not called, delete it. Don't migrate dead code into new directories.

---

## Success Criteria

After all phases:
- [ ] OutProjWeights enum with Dense + BlockDiagonal variants
- [ ] All 3 tiers compile and train (CPU, wgpu, Candle)
- [ ] --out-proj-groups flag works on all tiers
- [ ] WCHK v2 checkpoints save/load correctly across tiers
- [ ] wave-server loads any tier's checkpoint
- [ ] No file exceeds 550 lines (1-2 exceptions documented)
- [ ] Dead legacy code deleted (~4,900 lines removed)
- [ ] Source organised into core/, cpu/, wgpu/, candle/ directories
- [ ] Adding a new out_proj variant requires ZERO consumer changes

---

## What This Enables (future)

Once complete, the following become trivial:
- **Sparse out_proj:** Add `OutProjWeights::Sparse(...)` variant + methods
- **Low-rank out_proj:** Add `OutProjWeights::LowRank(...)` variant
- **KAN out_proj:** Add `OutProjWeights::KAN(...)` variant
- **Hybrid Qwen replacement:** Import core/model.rs, create BlockDiagonal with Qwen dims
- **New training tier:** Add a `tpu/` or `metal/` directory, import from core/
