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

   # GPU tier (must converge, loss descending)
   cargo run --release -- data/input.txt --iters 100 --gpu
   ```
4. **Submit a PR** with your results and a clear description.

## What's already built

These features are complete and should not be broken by contributions:

- **Three training tiers** — CPU, wgpu (any GPU), Candle/CUDA (NVIDIA)
- **BPE tokenizer** — HuggingFace tokenizer.json format, tested with 50K vocab
- **Checkpoint save/load** — WCHK format with optimizer state and resume
- **Text generation** — Autoregressive sampling with temperature/top-k/top-p
- **Curriculum training** — Band unlocking, validated +1.46pp improvement
- **GPU fused ODE** — One submit, 192 dispatches, zero CPU readbacks
- **Pipeline monitor** — Per-section FFN timing (mae_in, ODE, mae_out, out_proj)
- **Ping-pong buffers** — Forward/backward GPU consistency for out_proj
- **FFT ODE** — OFDM-inspired stencil convolution, validated at 1.19e-7

## What needs building

### Real data validation (highest priority)

**Wikitext-103 training pipeline** — The engine currently trains on Shakespeare (65-character vocab, 1MB). Real training needs wikitext-103 with word-level or BPE tokenization. The BPE code works, the checkpoint saves, the pipeline is proven — but the architecture hasn't been validated on real English yet. This is the most important next step.

**Tied embeddings / vocab adapter** — At 50K BPE vocab, the lm_head is 38.6M parameters — bigger than the entire 24-layer model (15.5M). A learned adapter (768×768 = 590K params) mapping hidden states into the frozen harmonic embedding space would solve this.

### Proven improvements to integrate

**Stochastic resonance** — Add α=0.05 noise to ODE initial conditions. Validated at -8.8% perplexity improvement on kerr-engine. A few lines of code.

### Performance

**fp16 for linear ops** — Matvecs at fp16 would halve memory and PCIe transfer size. Keep ODE at fp32 (nonlinear dynamics are precision-sensitive). Test on out_proj first.

**Speculative decoding** — Generate multiple candidate tokens in parallel, verify against full model. Reduces per-token latency for serving.

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
