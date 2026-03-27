# TIED EMBEDDINGS — Implementation Guide
# Date: 2026-03-23
# From: Desktop
# For: Code
# Priority: DO NOW — training blocked without this

---

## What Changes

The lm_head weight matrix is REMOVED. Output logits are computed by
multiplying hidden states against the EXISTING embedding table (wte).
Since wte is frozen (harmonic embeddings, no gradient), the entire
lm_head becomes a zero-parameter, zero-gradient operation.

## Why This Fixes the NaN

| | Before (separate lm_head) | After (tied) |
|---|---|---|
| Model A trainable | 8,510,848 | 67,672 |
| lm_head % | 99.2% | 0% |
| ODE % | 0.01% | 1.0% |
| Gradient clipping | lm_head dominates | All gradients are model gradients |
| NaN risk | lm_head weights grow → logit overflow | No lm_head weights to grow |

## The 10 Changes (in order)

### 1. Model struct — remove lm_head

```rust
struct WavePacketModel {
    wte: Vec<Vec<f32>>,    // [vocab_size][n_embd] — frozen harmonic embeddings
    wpe: Vec<Vec<f32>>,    // [block_size][n_embd] — frozen positional embeddings
    blocks: Vec<WaveBlockWeights>,
    ln_f: LayerNormWeights,
    // lm_head: Vec<Vec<f32>>,  ← DELETE THIS LINE
    vocab_size: usize,
}
```

### 2. init_model — remove lm_head initialization

Delete these lines (~189-192):
```rust
// DELETE:
let lm_head: Vec<Vec<f32>> = (0..vocab_size)
    .map(|_| (0..N_EMBD).map(|_| rng.uniform(limit)).collect())
    .collect();
```

And update the struct construction:
```rust
WavePacketModel { wte, wpe, blocks, ln_f, vocab_size }
// was: WavePacketModel { wte, wpe, blocks, ln_f, lm_head, vocab_size }
```

### 3. Forward — use wte instead of lm_head

Line ~298-305, replace:
```rust
// OLD:
let logits: Vec<Vec<f32>> = post_ln_f.iter().map(|normed| {
    let mut l = vec![0.0f32; model.vocab_size];
    for v in 0..model.vocab_size {
        let mut sum = 0.0f32;
        for j in 0..N_EMBD { sum += model.lm_head[v][j] * normed[j]; }
        l[v] = sum;
    }
    l
}).collect();
```

With:
```rust
// NEW — tied embeddings: logits = hidden @ wte.T
let logits: Vec<Vec<f32>> = post_ln_f.iter().map(|normed| {
    let mut l = vec![0.0f32; model.vocab_size];
    for v in 0..model.vocab_size {
        let mut sum = 0.0f32;
        for j in 0..d.n_embd { sum += model.wte[v][j] * normed[j]; }
        l[v] = sum;
    }
    l
}).collect();
```

Same math, different weight source. wte is frozen harmonic embeddings.

### 4. Backward — remove lm_head gradient, use wte for d_hidden

Line ~432-445, replace:
```rust
// OLD:
let mut d_hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
for pos in 0..t {
    let mut d_h = vec![0.0f32; N_EMBD];
    for j in 0..N_EMBD {
        for v in 0..vocab_size {
            d_h[j] += model.lm_head[v][j] * d_logits[pos][v];
        }
    }
    d_hidden.push(d_h);
    // lm_head weight gradients
    for v in 0..vocab_size {
        for j in 0..N_EMBD {
            grads.lm_head[v][j] += d_logits[pos][v] * cache.post_ln_f[pos][j];
        }
    }
}
```

With:
```rust
// NEW — tied embeddings: d_hidden uses wte (frozen), no weight gradients
let mut d_hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
for pos in 0..t {
    let mut d_h = vec![0.0f32; d.n_embd];
    for j in 0..d.n_embd {
        for v in 0..vocab_size {
            d_h[j] += model.wte[v][j] * d_logits[pos][v];
        }
    }
    d_hidden.push(d_h);
    // No lm_head weight gradients — wte is frozen
}
```

This is the biggest win: the entire vocab_size × n_embd gradient
accumulation loop is DELETED. That's the 37GB allocation that crashed
RAM, the 99% gradient domination, and the logit overflow source — all
gone in one change.

### 5. Gradients struct — remove lm_head

```rust
struct Gradients {
    block_ln_w: Vec<Vec<f32>>,
    block_ln_b: Vec<Vec<f32>>,
    // ... all the block gradients stay ...
    ln_f_w: Vec<f32>,
    ln_f_b: Vec<f32>,
    // lm_head: Vec<Vec<f32>>,  ← DELETE THIS LINE
}
```

And remove its initialization in backward():
```rust
// DELETE:
lm_head: vec![vec![0.0; N_EMBD]; vocab_size],
```

### 6. count_trainable — remove lm_head count

```rust
fn count_trainable(model: &WavePacketModel) -> usize {
    let mut n = 0;
    // ... all the block counting stays ...
    n += d.n_embd * 2; // ln_f
    // DELETE: n += model.vocab_size * N_EMBD; // lm_head
    n
}
```

### 7. flatten_params — remove lm_head

Delete:
```rust
// DELETE:
for row in &model.lm_head { p.extend_from_slice(row); }
```

### 8. flatten_grads — remove lm_head

Delete:
```rust
// DELETE:
for row in &grads.lm_head { g.extend_from_slice(row); }
```

### 9. unflatten_params — remove lm_head

Delete:
```rust
// DELETE:
for row in &mut model.lm_head { row.copy_from_slice(&params[idx..idx+N_EMBD]); idx += N_EMBD; }
```

### 10. Candle tier — same change

In src/candle_tier/engine.rs, the lm_head is a separate trainable Linear layer.
Replace it with a matmul against the embedding table:

```rust
// OLD:
let logits = lm_head.forward(&post_ln_f)?;

// NEW:
let logits = post_ln_f.matmul(&wte.t()?)?;
```

And remove lm_head from VarMap/VarBuilder. The wte tensor is already
allocated — just use it transposed.

Also update extract_wchk_params() to NOT include lm_head params.

---

## Checkpoint Compatibility

Old checkpoints include lm_head weights. New checkpoints don't.
Two options:

**Option A (simple):** Bump WCHK version to v3. Old checkpoints can't resume.
This is fine — we're starting fresh training runs anyway.

**Option B (backward compat):** When loading, detect param count mismatch.
If old checkpoint has more params than model expects, skip the lm_head
portion at the end of the flat array. This lets you resume from old
checkpoints but ignores their lm_head weights (which is correct since
we're replacing them with wte).

Recommend Option A for simplicity. The old checkpoints trained on
corrupted weights anyway (the NaN issue).

---

## Logit Scale Consideration

The harmonic embedding table has specific magnitude properties
(cos/sin values, roughly in [-1, 1]). The logits from wte.T @ hidden
may have different scale than a randomly initialized lm_head.

If logits are too small (loss stays high, softmax too uniform):
add a learnable scalar temperature: `logits *= temperature`
That's 1 trainable parameter, not 8.4M.

If logits are fine: do nothing. Test first.

---

## Test After Implementation

```bash
# Model A — should train past iter 1000 without NaN at lr=3e-4:
wave-engine data/combined_10mb.txt --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 6 --iters 1000 --batch 4 --seq 128 --lr 3e-4

# Check:
# 1. No NaN
# 2. gnorm should NOT be 1.00 every iteration (gradient clipping not saturated)
# 3. Trainable params should be ~68K, not ~8.5M
# 4. Loss should descend from ~10.8 (log(50K))
```

Then test at 384-dim to confirm it scales:
```bash
wave-engine data/combined_10mb.txt --layers 8 --n-bands 192 --n-head 8 \
  --out-proj-groups 6 --iters 200 --batch 4 --seq 128 --lr 1e-4
```

---

## Summary

| Change | File | Lines affected |
|--------|------|---------------|
| Remove lm_head from struct | main.rs | ~3 lines |
| Remove lm_head init | main.rs | ~4 lines |
| Forward: wte instead of lm_head | main.rs | ~6 lines (replace) |
| Backward: remove lm_head grads | main.rs | ~10 lines (delete) |
| Gradients: remove lm_head | main.rs | ~3 lines |
| count_trainable: remove lm_head | main.rs | ~1 line |
| flatten_params: remove lm_head | main.rs | ~1 line |
| flatten_grads: remove lm_head | main.rs | ~1 line |
| unflatten_params: remove lm_head | main.rs | ~1 line |
| Candle tier: same changes | candle_tier/engine.rs | ~10 lines |

Total: ~40 lines changed across 2 files. Most are deletions.
The forward pass replacement is the only "new" code — and it's
the same math with a different weight source.
