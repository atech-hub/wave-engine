# Contributing to wave-engine

## About this project

This engine was built by an IT systems administrator (not a software engineer) collaborating with AI assistants. The architecture and direction come from Marco Da Cunha. Claude Desktop (Opus) handles theory, analysis, and documentation. Claude Code handles implementation and testing.

This is stated openly because it matters for how contributions work:

- **The maintainer understands the architecture deeply** but reviews code at a functional level, not a syntactic one.
- **PRs are evaluated on results** — does training still converge? Do the monitors show the expected improvements? Did anything break?
- **Clear descriptions matter more than clean code.** Explain what you changed, what it does, and how you tested it. Show numbers.

## How to contribute

1. **Fork the repo and create a branch** from `main`.
2. **Make your changes.** Write tests or show training results.
3. **Run the validation suite:**
   ```bash
   # CPU baseline (must match: loss ~2.81 at iter 99)
   cargo run --release -- data/input.txt --iters 100

   # GPU tier (must match: loss ~2.81 at iter 99 for --gpu safe, ~3.06 for fast)
   cargo run --release -- data/input.txt --iters 100 --gpu
   ```
4. **Submit a PR** with your results and a clear description.

## What needs building (priority order)

### Critical for real training

**Wikitext-103 data pipeline** — The engine currently trains on Shakespeare (65-character vocab, 1MB). Real training needs word-level or BPE tokenization of wikitext-103. The parquet files exist, BPE code exists in `bpe.rs`, but the pipeline isn't wired together. This is the #1 blocker for validating the architecture on real English.

**Checkpoint save/load with resume** — `checkpoint.rs` exists but may not match the current model format. Overnight training requires saving weights + optimizer state + iteration count and resuming cleanly.

**Text generation** — The engine trains but can't generate text. Needs an autoregressive sampling loop (temperature, top-k, top-p). The inference code exists in [kerr-server](https://github.com/atech-hub/kerr-server)'s `inference.rs` as a reference. Without this, you train blind and can't see what the model produces.

### Proven improvements to integrate

**Curriculum training** — Start at 192 bands, unlock to 384 over first 20% of iterations. Validated at +1.46 percentage points improvement on kerr-engine. The mechanism is known, needs wiring into the training loop.

**Stochastic resonance** — Add α=0.05 noise to ODE initial conditions. Validated at -8.8% perplexity improvement. A few lines of code.

### GPU acceleration

**Fused GPU ODE** — The ODE currently runs on CPU (5.5ms/block). The FFT shader (`fft_512.wgsl`) is compiled and validated. Making the entire RK4 loop GPU-resident (all buffers in VRAM, no CPU roundtrips) would eliminate per-step dispatch overhead. This is the path to moving the remaining 28% FFN bottleneck to GPU.

**GPU backward buffer reuse** — The backward `out_proj` creates fresh buffers each call (transpose upload, zero d_w/d_b). Caching transposed weights and reusing zeroed buffers would cut backward time in half.

## Architecture notes

The model uses **GPT-J parallel block formulation** — attention and FFN both read the same normalized input:
```
x = x + attn(LN(x)) + FFN(LN(x))    # parallel (GPT-J, this engine)
x = x + attn(LN(x)); x = x + FFN(LN(x))  # sequential (GPT-2, NOT this engine)
```

If you modify the forward pass, this distinction matters — the training data was generated with parallel blocks, and the backward pass assumes it.

**Frozen attention**: Attention weights do NOT update during training. Gradients flow through the attention output for the residual connection, but no attention weight gradients are computed. This is by design, not a bug.

**Maestro dim=16 is a universal constant.** Don't try scaling it with embedding dimension. Tested at 16/96/128/160 on 768-dim — all produce the same quality within 0.028 loss. The coordination task is fundamentally low-dimensional.

## Testing

Every change should be validated against the CPU baseline:

| Metric | Expected value |
|--------|---------------|
| Loss at iter 0 | ~4.4 (random init) |
| Loss at iter 99 | ~2.81 (CPU, 4 layers) |
| Loss descending | Monotonically with fluctuation |
| NaN | Never |
| Gradient norm | Clipped to 1.0, raw norm > 0 |

If you add a GPU feature, compare against CPU at the same iteration count. Document the quality gap and speed improvement.

## Code of conduct

Be honest about results. Document nulls alongside positives. If something doesn't work, say so — every null result in this project has led to a better understanding of the architecture.
