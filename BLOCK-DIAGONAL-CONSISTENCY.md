# BLOCK-DIAGONAL CONSISTENCY — All Tiers Spec
# Date: 2026-03-22
# For: Code (Claude Code)
# Priority: Do this BEFORE the next training run
# Goal: All tiers produce identical architecture and checkpoint format

---

## Problem

Candle CUDA uses 6-group block-diagonal out_proj. CPU and wgpu still use
dense out_proj. Wave-server already expects block-diagonal WCHK. A model
trained on CPU/wgpu can't be served. This is broken.

## Target State

ALL tiers:
- out_proj is BlockDiagonal with configurable groups
- WCHK v2 format with out_proj_groups in header
- wave-server loads any tier's checkpoint
- `--out-proj-groups N` flag works on all tiers (default 6 for 768-dim)

---

## Changes Required

### 1. ModelConfig — Add out_proj_groups

File: `src/model.rs`

```rust
pub struct ModelConfig {
    pub n_bands: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_n_steps: usize,
    pub out_proj_groups: usize,  // NEW — default 6 for 768-dim, 1 = dense
}
```

Add to `default_128()`: `out_proj_groups: 1` (dense for small models)
Add to any `default_768()`: `out_proj_groups: 6`

### 2. model.rs — Add BlockDiagonalWeights, update KerrDualMaestroWeights

File: `src/model.rs`

Add struct (matching wave-server):
```rust
#[derive(Clone)]
pub struct BlockDiagonalWeights {
    pub groups: Vec<LinearWeights>,  // n_groups × (group_size, group_size)
    pub n_groups: usize,
    pub group_size: usize,
}
```

Change `KerrDualMaestroWeights`:
```rust
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: BlockDiagonalWeights,  // WAS: LinearWeights
}
```

Add block-diagonal forward helper:
```rust
pub fn block_diagonal_forward(weights: &BlockDiagonalWeights, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (g, group) in weights.groups.iter().enumerate() {
        let start = g * weights.group_size;
        let end = start + weights.group_size;
        let group_in = &x[start..end];
        let group_out = linear(&group.w, &group.b, group_in);
        out[start..end].copy_from_slice(&group_out);
    }
    out
}
```

### 3. init.rs — Initialize block-diagonal weights

File: `src/init.rs`

Where out_proj is currently initialized as a dense LinearWeights:
```rust
// OLD:
out_proj: LinearWeights { w: random_matrix(n_embd, n_embd), b: zeros(n_embd) }

// NEW:
out_proj: BlockDiagonalWeights {
    groups: (0..config.out_proj_groups).map(|_| {
        let gs = n_embd / config.out_proj_groups;
        LinearWeights { w: random_matrix(gs, gs), b: zeros(gs) }
    }).collect(),
    n_groups: config.out_proj_groups,
    group_size: n_embd / config.out_proj_groups,
}
```

### 4. wave_checkpoint.rs — WCHK v2 with out_proj_groups

File: `src/wave_checkpoint.rs`

**Save:**
- Change magic version to 2
- Add out_proj_groups to config header (after rk4_n_steps):
  `f.write_all(&(config.out_proj_groups as u32).to_le_bytes()).unwrap();`
- Param layout for out_proj changes from:
  `n_embd × n_embd weight + n_embd bias` (dense)
  to:
  `n_groups × (group_size × group_size weight + group_size bias)` (block-diagonal)

**Load:**
- Read version. If v1 AND out_proj is dense → convert to block-diagonal? 
  OR just fail with clear error message. (Prefer fail — old checkpoints 
  are incompatible anyway due to weight shapes)
- If v2: read out_proj_groups from header, compute param count correctly

**Param count formula (per block):**
```rust
let gs = n_embd / out_proj_groups;
let out_proj_params = out_proj_groups * (gs * gs + gs);
let per_block = n_embd*4  // two layer norms
    + md*n_embd + md       // maestro_in squeeze
    + n_embd*md + n_embd   // maestro_in process
    + md*n_embd + md       // maestro_out squeeze
    + n_embd*md + n_embd   // maestro_out process
    + out_proj_params;      // block-diagonal out_proj
```

### 5. Flatten/unflatten in main.rs or train.rs

File: wherever `flatten_params` and `unflatten_params` live

Update the out_proj section:
```rust
// OLD (dense):
for row in &block.ffn.out_proj.w { flat.extend(row); }
flat.extend(&block.ffn.out_proj.b);

// NEW (block-diagonal):
for group in &block.ffn.out_proj.groups {
    for row in &group.w { flat.extend(row); }
    flat.extend(&group.b);
}
```

Same pattern for unflatten (read group_size×group_size + group_size per group).

### 6. Forward pass — wave_block.rs and ffn_backend.rs

File: `src/wave_block.rs`

Where `dual_maestro_forward_cached` calls `linear(out_proj.w, out_proj.b, ...)`:
```rust
// OLD:
let projected = linear(&weights.out_proj.w, &weights.out_proj.b, &regulated);

// NEW:
let projected = block_diagonal_forward(&weights.out_proj, &regulated);
```

File: `src/ffn_backend.rs`

Same change — wherever the CPU path does the out_proj matmul.

### 7. wgpu out_proj shader

File: relevant shader in `src/gpu_*.rs`

The wgpu fused shader does a dense matmul for out_proj. This needs to
become block-diagonal. The shader should:
1. Compute which group each thread belongs to
2. Only read weights from that group's block
3. Output to that group's position

If this is complex, an alternative: do out_proj on CPU after the GPU
ODE (similar to how Candle handles it). The out_proj is small relative
to the ODE, so the CPU→GPU→CPU transfer cost is minimal.

### 8. main.rs — Wire the --out-proj-groups CLI flag

Already partially done. Make sure it propagates to:
- ModelConfig construction
- init_model
- wave_checkpoint save/load
- The WCHK header

### 9. wave-server — Update for WCHK v2

File: `wave-server/src/checkpoint.rs`

- Read WCHK v2: parse out_proj_groups from header
- Use it in count_trainable_params and unflatten_to_model
- Fall back: if v1, assume dense (out_proj_groups = 1) for backward compat
  Actually — wave-server already hardcodes 6 groups. Change it to read
  from the header instead.

Also add out_proj_groups to wave-server's ModelConfig.

---

## Testing

After implementing:

```bash
# CPU tier — must not NaN, loss must descend
cargo run --release -- data/input.txt --layers 4 --iters 10 --seq 64 --no-curriculum --out-proj-groups 6

# wgpu tier
cargo run --release -- data/input.txt --layers 4 --iters 10 --seq 64 --no-curriculum --gpu --out-proj-groups 6

# Candle tier (already works)
cargo run --release --features candle-backend -- data/input.txt --candle --layers 4 --iters 10 --seq 64 --no-curriculum --out-proj-groups 6

# Cross-tier checkpoint test:
# 1. Train 5 iters on Candle, save WCHK
# 2. Load in wave-server → must not crash
# 3. Train 5 iters on CPU, save WCHK
# 4. Load in wave-server → must not crash
```

## Param counts for reference (768-dim, 24 layers)

| Groups | group_size | out_proj/block | Total out_proj | Total model (char-135) |
|--------|-----------|---------------|---------------|----------------------|
| 1 (dense) | 768 | 590,592 | 14.2M | 15.5M |
| 6 | 128 | 99,456 | 2.4M | 3.7M |
| 12 | 64 | 49,920 | 1.2M | 2.5M |

## Order of implementation

1. ModelConfig + model.rs structs (foundations)
2. init.rs (create block-diagonal weights)
3. flatten/unflatten (param serialization)
4. wave_checkpoint.rs (WCHK v2)
5. wave_block.rs + ffn_backend.rs (CPU forward)
6. Test CPU tier
7. wgpu shader (GPU forward) — or CPU fallback for out_proj
8. Test wgpu tier
9. wave-server checkpoint.rs (read v2 header)
10. Cross-tier checkpoint test

## What NOT to change

- Candle tier: already works, don't touch
- Attention: no block-diagonal, stays dense
- Maestro: stays as-is
- ODE: no changes
- Training loop, optimizer, LR schedule: no changes
