# PROGRESSIVE-DIM-SCALING-SPEC.md — Scale trained models to larger dimensions

**Goal:** Take a trained 168-dim checkpoint and scale it to 256-dim (or any larger dimension), preserving learned weights where bands align and initialising new bands fresh.

**New file:** `src/common/scale.rs` — all scaling logic self-contained.

**CLI:** `wave-engine --scale <input.bin> --target-bands 128 --output <scaled.bin> [options]`

---

## Why the wave architecture makes this possible

In an MLP, dimension 84 is an arbitrary index — there's no way to map dim-84 of a 168-dim model to anything in a 256-dim model. Scaling requires retraining from scratch.

In wave-engine, band k is a specific harmonic frequency. Band 1 oscillates at ω=1/n_bands, band 2 at ω=2/n_bands, etc. Each band has a physical meaning: it carries a specific frequency of the wave basis. Band 1 at 168-dim and band 1 at 256-dim represent the same low-frequency structure — the model learned what to do with it.

Scaling from 84→128 bands means keeping bands 1–84 with their learned weights and adding bands 85–128 with fresh initialisation. The existing learned structure is preserved; new capacity is added on top.

---

## What transfers, what gets recomputed, what gets padded

### Recomputed (frozen, deterministic — no transplant):
- **wte** (harmonic embedding table): recomputed from vocab_size + new n_bands
- **wpe** (positional encoding table): recomputed for new dimension
- **Attention weights**: frozen harmonic heads, recomputed from new n_embd
- **ODE omega values**: ω_k = (k+1)/n_bands — changes because n_bands changed

### Transplanted with padding (learned weights preserved):
| Weight | Old shape | New shape | Padding strategy |
|--------|----------|-----------|-----------------|
| **LN weight** (ln, ln_ffn, ln_f) | [168] | [256] | Pad with 1.0 (identity) |
| **LN bias** (ln, ln_ffn, ln_f) | [168] | [256] | Pad with 0.0 (identity) |
| **ODE gamma_raw** | [84] | [128] | Pad with fresh init value |
| **Maestro squeeze.w** | [16 × 168] | [16 × 256] | Pad columns with 0.0 |
| **Maestro squeeze.b** | [16] | [16] | No change (maestro_dim fixed) |
| **Maestro process.w** | [168 × 16] | [256 × 16] | Pad rows with small random |
| **Maestro process.b** | [168] | [256] | Pad with 0.0 |
| **Out_proj (Dense)** | [168 × 168] | [256 × 256] | Top-left block preserved, rest small random |
| **lm_head** | [vocab × 168] | [vocab × 256] | Pad columns with small random |
| **ODE alpha, beta** | scalar | scalar | Keep same |

### Padding rationale:
- **LN: 1.0/0.0** — LayerNorm with weight=1, bias=0 is identity. New bands pass through unchanged until trained.
- **Maestro squeeze columns: 0.0** — New input dimensions contribute nothing to the bottleneck initially. The maestro learns to use new bands during fine-tuning.
- **Maestro process rows: small random** — New output dimensions get small random init so they break symmetry during training. 0.0 would mean new bands never get gradient.
- **Out_proj: top-left + random** — Existing band interactions preserved exactly. New-to-old and new-to-new interactions get small random init.
- **lm_head: small random** — Each vocab token gets small random values for new dimensions. The lm_head will learn what the new bands mean during fine-tuning.
- **Gamma_raw: fresh init** — Same initial damping as a new model. The ODE will learn appropriate damping for new bands.

---

## Implementation: `src/common/scale.rs`

### Public API

```rust
/// Configuration for dimension scaling
pub struct ScaleConfig {
    pub source_path: String,      // input checkpoint
    pub target_bands: usize,      // new n_bands (must be > source)
    pub target_head: usize,       // new n_head
    pub output_path: String,      // output checkpoint
    pub target_groups: usize,     // out_proj groups for target (1=dense)
    pub seed: u64,                // RNG seed for random padding
}

/// Scale a checkpoint from source dimensions to target dimensions.
/// Returns (scaled_params, new_dims, original_dims) for verification.
pub fn scale_checkpoint(config: &ScaleConfig) -> Result<(), String>
```

### Internal functions (all private to scale.rs)

```rust
/// Pad a 1D vector from old_len to new_len with a fill value
fn pad_1d(src: &[f32], new_len: usize, fill: f32) -> Vec<f32>

/// Pad a 2D weight matrix [out_dim × in_dim] → [new_out × new_in]
/// Top-left block preserved, rest filled with fill_fn
fn pad_2d(
    src: &[Vec<f32>],
    new_out: usize,
    new_in: usize,
    fill_fn: &mut dyn FnMut() -> f32,  // closure for random or constant fill
) -> Vec<Vec<f32>>

/// Transplant one block's weights from source to target dimensions
fn scale_block(
    src_params: &[f32],       // flat params for this block (source dims)
    src_dims: &BlockDims,     // n_embd, maestro_dim, n_bands for source
    tgt_dims: &BlockDims,     // same for target
    rng: &mut Rng,
) -> Vec<f32>                 // flat params for this block (target dims)

/// Print a diagnostic summary of what was transplanted vs padded
fn print_scale_report(
    src_dims: &BlockDims,
    tgt_dims: &BlockDims,
    n_layers: usize,
    vocab_size: usize,
)
```

### BlockDims helper (private)

```rust
struct BlockDims {
    n_bands: usize,
    n_embd: usize,        // n_bands * 2
    maestro_dim: usize,    // always 16
    out_proj_groups: usize,
}
```

---

## Param layout (must match flatten_params/unflatten_params exactly)

Per block, in order:
1. `ln.weight` [n_embd]
2. `ln.bias` [n_embd]
3. `ln_ffn.weight` [n_embd]
4. `ln_ffn.bias` [n_embd]
5. `maestro_in.squeeze.w` [maestro_dim × n_embd] (row-major)
6. `maestro_in.squeeze.b` [maestro_dim]
7. `maestro_in.process.w` [n_embd × maestro_dim] (row-major)
8. `maestro_in.process.b` [n_embd]
9. `maestro_out.squeeze.w` [maestro_dim × n_embd]
10. `maestro_out.squeeze.b` [maestro_dim]
11. `maestro_out.process.w` [n_embd × maestro_dim]
12. `maestro_out.process.b` [n_embd]
13. `out_proj` (Dense: [n_embd × n_embd] + [n_embd], BlockDiag: groups × ([gs × gs] + [gs]))

After all blocks:
14. `ln_f.weight` [n_embd]
15. `ln_f.bias` [n_embd]
16. `lm_head` [vocab × n_embd] (row-major)

**The scaling function reads source params in this order, pads each tensor, and writes target params in the same order.** No unflatten-to-struct needed — work directly on the flat arrays with offset tracking.

---

## Dense-to-Dense out_proj scaling

When scaling Dense out_proj (groups=1):

Source: [168 × 168] weight + [168] bias
Target: [256 × 256] weight + [256] bias

```
[  168×168 learned  |  168×88 random  ]     256 cols
[  88×168 random    |  88×88 random   ]
```

The top-left 168×168 block is the existing learned mixing. The new rows and columns handle new band interactions — small random init.

### Dense-to-BlockDiag scaling

If the target uses block-diagonal (for larger dims), the scaling is more complex. The source Dense matrix needs to be decomposed into groups. For 168→256 with 4 groups:

- Source: Dense [168 × 168]
- Target: 4 groups of [64 × 64]

Extract the diagonal blocks from the source that overlap with target groups. This is lossy — inter-group information is discarded.

**Recommendation:** Keep Dense at 256-dim (the operating regime says Dense or 4-group at 256-dim). If Dense→Dense, the scaling is clean and lossless for existing bands. Only consider BlockDiag at 384-dim+.

---

## Attention scaling

Attention is frozen — weights are recomputed from `init_model`, not loaded from checkpoint. The scaling function does NOT need to transplant attention weights. However, the attention architecture must be consistent with the new dimensions:

- `n_head` may change (4→8 at 256-dim) — this is a config parameter, not a transplant
- `head_dim = n_embd / n_head` — must divide evenly
- Phase projection weights [2 × n_embd] are reinitialised
- Value projection weights [head_dim × head_dim] are reinitialised

---

## ODE parameter scaling

```rust
// Source: 84 bands
gamma_raw = [g0, g1, ..., g83]   // learned damping
omega     = [ω0, ω1, ..., ω83]  // = [(k+1)/84 for k in 0..84]
alpha     = 0.1                   // scalar, keep
beta      = 0.1                   // scalar, keep

// Target: 128 bands
gamma_raw = [g0, g1, ..., g83, fresh, fresh, ..., fresh]  // pad with init value
omega     = [(k+1)/128 for k in 0..128]                    // RECOMPUTED (not padded)
alpha     = 0.1                                             // keep
beta      = 0.1                                             // keep
```

**Important:** omega is RECOMPUTED, not padded. At 84 bands, ω_84 = 84/84 = 1.0. At 128 bands, ω_84 = 84/128 = 0.656. The frequency assignments shift to accommodate the wider spectrum. This is correct — the ODE's frequency basis should span the full range at the new dimension.

**Note on gamma_raw:** ODE params (gamma, omega) are currently frozen during training (the training loop doesn't update them — gradients skip the ODE). So the learned gamma_raw values from the 168-dim model reflect the init value, not learned values. Padding with the same init value is consistent. If ODE param training is enabled in the future, the learned gammas for bands 1–84 would carry real information and should definitely be preserved.

---

## Optimizer state

The Adam optimizer has momentum (m) and variance (v) vectors of the same size as params. When scaling:

**Option A (recommended): Reset optimizer.** Start fresh Adam state at the new dimension. The scaled model is effectively a new model with a warm start — the optimizer should explore the new parameter space without momentum from the old dimensions. This is what `--resume` with a new learning rate does already.

**Option B: Pad optimizer state.** Pad m and v the same way as params (existing slots keep their values, new slots get 0.0). This preserves training momentum for existing bands but the new bands have no history. Could cause instability if old momentum is stale at new dimensions.

**Recommendation:** Option A. The checkpoint saves with `iter=0` and fresh Adam state. The user provides a learning rate for fine-tuning at the new dimension. This matches the cycling protocol — resume from best, fresh optimiser.

---

## CLI interface

```bash
# Scale 168-dim to 256-dim
wave-engine --scale model_a_gs_best.bin \
  --target-bands 128 --target-head 8 \
  --output model_256_from_168.bin

# Scale with specific out_proj groups
wave-engine --scale model_a_gs_best.bin \
  --target-bands 128 --target-head 8 \
  --out-proj-groups 4 \
  --output model_256_from_168.bin

# Then train at new dimension
wave-engine data/grammar_shakespeare.txt \
  --resume model_256_from_168.bin \
  --layers 4 --n-bands 128 --n-head 8 \
  --out-proj-groups 1 --alpha 0.1 --agc-ceiling 1.0 \
  --bpe --tokenizer data/tokenizer_512.json \
  --lr 1e-4 --iters 20000 \
  --checkpoint-name model_256_scaled.bin
```

### CLI plumbing in main.rs

Add to main.rs (before --candle and --analyze checks):

```rust
if std::env::args().any(|a| a == "--scale") {
    let source = std::env::args().skip_while(|a| a != "--scale").nth(1)
        .expect("--scale requires a checkpoint path");
    let target_bands: usize = parse_flag("--target-bands", 128);
    let target_head: usize = parse_flag("--target-head", 8);
    let output: String = parse_flag("--output", "scaled_checkpoint.bin".to_string());
    let groups: usize = parse_flag("--out-proj-groups", 1);

    common::scale::scale_checkpoint(&common::scale::ScaleConfig {
        source_path: source,
        target_bands,
        target_head,
        output_path: output,
        target_groups: groups,
        seed: 42,
    }).unwrap_or_else(|e| { eprintln!("Scale error: {e}"); std::process::exit(1); });
    return;
}
```

---

## Expected output

```
wave-engine v0.1.0  (8 threads, 16 available)

Scaling checkpoint: model_a_gs_best.bin
  Source: 84 bands (168-dim), 4 layers, 1 groups, 512 vocab
  Target: 128 bands (256-dim), 4 layers, 1 groups, 512 vocab

  Per-block weight scaling:
    LN weights:      168 → 256  (pad 1.0/0.0)
    Maestro squeeze: [16×168] → [16×256]  (pad cols 0.0)
    Maestro process: [168×16] → [256×16]  (pad rows random)
    ODE gamma_raw:   84 → 128  (pad fresh init)
    ODE omega:       84 → 128  (recomputed)
    Out_proj Dense:  [168×168] → [256×256]  (top-left preserved)

  Final layers:
    ln_f:    168 → 256  (pad 1.0/0.0)
    lm_head: [512×168] → [512×256]  (pad cols random)

  Source params:  171,234
  Target params:  394,752
  Transplanted:  171,234 (100% of source preserved)
  New (random):  223,518

  Saved: model_256_from_168.bin (WCHK v2, iter=0, fresh optimizer)
```

---

## Diagnostic measurements (run before and after scaling)

### Before scaling (168-dim trained model):
```bash
wave-engine --analyze --resume model_a_gs_best.bin \
  --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1 --alpha 0.1 \
  --bpe --tokenizer data/tokenizer_512.json
```

### After scaling, before fine-tuning (256-dim, transplanted weights):
```bash
wave-engine --analyze --resume model_256_from_168.bin \
  --layers 4 --n-bands 128 --n-head 8 --out-proj-groups 1 --alpha 0.1 \
  --bpe --tokenizer data/tokenizer_512.json
```

### After fine-tuning (256-dim, trained):
```bash
wave-engine --analyze --resume model_256_scaled_best.bin \
  --layers 4 --n-bands 128 --n-head 8 --out-proj-groups 1 --alpha 0.1 \
  --bpe --tokenizer data/tokenizer_512.json
```

### What to look for:
1. **Phase clustering:** Does the transplanted model retain its phase structure? Does fine-tuning improve it?
2. **Harmonic spectra:** Do the old bands (1–84) keep their harmonic assignments? Do new bands (85–128) develop their own? Does multi-harmonic structure persist at 256-dim where it collapsed at 168-dim?
3. **Semantic discrimination:** Does 256-dim cross the 1.5x threshold where 168-dim stayed at 1.0x?
4. **Band census:** Do the new bands specialise? Does the old 50/50 split evolve?
5. **Output quality:** Does the scaled model produce English immediately (transplanted weights work) or need full retraining?

---

## File structure

```
src/common/scale.rs     — all scaling logic (pad_1d, pad_2d, scale_block, scale_checkpoint)
src/common/mod.rs       — add: pub mod scale;
src/main.rs             — add: --scale CLI dispatch (~15 lines)
```

One new file. Two small edits. Everything else stays the same.

---

## Future extensions (NOT in this spec)

- **Scale down** (256→168): extract bands 1–84, discard 85–128. Useful for distillation.
- **Add layers:** Scale from 4L to 6L or 8L. New layers get fresh init, existing layers keep weights. Similar transplant pattern.
- **Asymmetric scaling:** Different bands for different layers (e.g., low layers keep 84 bands, deep layers get 128). Requires per-layer dimension tracking.
- **Merge two models:** Combine a 168-dim model's low bands with a different 168-dim model's structure. Requires alignment analysis.
- **Checkpoint v3:** Store source dimensions in the header so the scaling provenance is traceable.

---

## Test protocol

1. **Scale and verify:** Scale model_a_gs_best.bin (168-dim) to 256-dim. Verify output checkpoint loads and runs inference without crash.
2. **Weight preservation:** Load both source and target checkpoints, verify that bands 1–84 in the target match bands 1–84 in the source exactly (bit-identical for transplanted weights).
3. **Inference before training:** Serve the scaled checkpoint through wave-server. Does it produce output at all? Output quality will be degraded (new bands are random) but the old structure should give it a head start vs training from scratch.
4. **Fine-tune and compare:** Train 256-dim from the scaled checkpoint for 10K iters. Compare loss curve against 256-dim trained from scratch for 10K iters. The scaled model should start at a lower loss and converge faster.
5. **Harmonic diagnostics:** Run --analyze on all three checkpoints (168-dim trained, 256-dim scaled pre-training, 256-dim scaled post-training). Track the key questions from the investigation.

---

## DO NOT

- Do not modify checkpoint.rs — the existing save/load format handles any dimension. scale.rs produces a standard WCHK v2 checkpoint.
- Do not modify wave_model.rs — init_model already accepts arbitrary Dims. The scaled checkpoint is loaded through the normal unflatten path.
- Do not modify the training loop — the scaled checkpoint is loaded via --resume like any other checkpoint.
- Do not add ODE param training — that's a separate feature. Scale.rs just pads the frozen values.
- Do not add layer scaling in this spec — that's a future extension. This spec is dimension scaling only, same layer count.
