# wave-engine

Training engine for wave-coherent neural architectures. Replaces standard MLP layers with coupled harmonic oscillators (Kerr-ODE) governed by a differential equation. Three training tiers — CPU, cross-platform GPU (wgpu), and NVIDIA GPU (Candle/CUDA) — all producing the same model from one binary, one command.

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) research project.

## Quick Start

```bash
# Build
cargo build --release

# Train on CPU (best quality, any hardware)
./target/release/wave-engine data/input.txt --iters 200

# Train with GPU acceleration (any GPU — NVIDIA, AMD, Intel, Apple)
./target/release/wave-engine data/input.txt --iters 200 --gpu

# Train with NVIDIA CUDA (requires --features candle-backend)
cargo build --release --features candle-backend
./target/release/wave-engine data/input.txt --iters 200 --candle

# Train with BPE tokenizer
./target/release/wave-engine data/input.txt --iters 200 --bpe --tokenizer data/tokenizer.json

# See all options
./target/release/wave-engine --help
```

No Python. No pip. No CUDA toolkit. Build once, run anywhere.

## Model Configurations

Proven configurations for each dimension tier, with recommended settings, training results, and ideal use cases. Start with the tier that matches your hardware and goals.

| Tier | Dimension | Params | Speed | Best For | Details |
|------|-----------|--------|-------|----------|--------|
| **Research / Specialist** | 168-dim | 186K | 57-80ms | Small-vocab structured tasks — music, code, chemistry, DNA, arithmetic. Trains in minutes on any CPU. 0.988 phase clustering, bimodal band census confirmed. | [168-dim config](configs/168-dim/CONFIG.md) |
| **Mid-Range** | 256-dim | 258-579K | 140-210ms | Domain-specific text, structured English, code generation. Bridge between research and production. | [256-dim config](configs/256-dim/CONFIG.md) |
| **Power User** | 384-dim | ~500K | ~200ms | Coherent English with word-level BPE. Enough capacity for sentence-level generation. | [384-dim config](configs/384-dim/CONFIG.md) |
| **Production** | 768-dim | ~4M | 1.2s CPU / 4.3s GPU | Full English, 50K BPE vocabulary. 24-layer models trained on Candle/CUDA. | [768-dim config](configs/768-dim/CONFIG.md) |

Key scaling rules discovered through systematic testing:
- **Gradient balance:** Model params need ≥44% of total for effective learning. Below this, the lm_head starves the ODE/maestro.
- **Vocab matching:** Larger vocabularies need larger dimensions — 512 BPE for 168-dim, up to 50K BPE for 768-dim.
- **Multi-grid embeddings:** Required for BPE below 768-dim. Coprime dual-circle mapping gives 101x-11,800x token separation improvement.
- **Per-band ODE clamp:** Max magnitude 2.5 before ODE — prevents phase wrapping at all dimensions.

See [configs/README.md](configs/README.md) for the complete guide.

## Examples

### Your first training run

Download any plain text file (Shakespeare, Wikipedia, a novel — anything works) and save it as `data/input.txt`. Then:

```bash
./target/release/wave-engine data/input.txt --layers 4 --iters 200
```

This trains a 4-layer model on CPU for 200 iterations. You'll see loss descending from ~4.5 to ~2.5. The model saves to `checkpoint.bin` when done.

### Scale up with GPU

If you have any GPU (NVIDIA, AMD, Intel, Apple Silicon):

```bash
./target/release/wave-engine data/input.txt --layers 4 --iters 200 --gpu
```

Same model, same results, GPU-accelerated. The `--gpu` flag works on any GPU vendor — no CUDA required.

### NVIDIA CUDA (fastest)

If you have an NVIDIA GPU and want maximum speed:

```bash
cargo build --release --features candle-backend
./target/release/wave-engine data/input.txt --candle --layers 4 --iters 200
```

This uses the Candle/CUDA backend with perturbative ODE and block-diagonal output projection. 2.4x faster than CPU.

### Train with BPE tokenizer (recommended for real training)

Character-level tokenization is fine for experiments, but BPE produces better models:

```bash
# Download a GPT-2 tokenizer (or any HuggingFace tokenizer.json)
./target/release/wave-engine data/input.txt --iters 1000 --bpe --tokenizer data/tokenizer.json
```

### Production training (24 layers, diverse corpus)

For a real model that generates English:

```bash
# Combine your training data (grammar, literature, etc.)
cat grammar.txt literature.txt > data/training.txt

# Train 24 layers with BPE on CUDA (fastest)
./target/release/wave-engine data/training.txt --candle \
    --layers 24 --iters 5000 --seq 256 --batch 4 --lr 1e-4 \
    --bpe --tokenizer data/tokenizer.json
```

### Resume training from a checkpoint

```bash
./target/release/wave-engine data/input.txt --iters 5000 --resume checkpoint.bin
```

### Custom architecture dimensions

All tiers (CPU, wgpu, Candle) support runtime-configurable dimensions:

```bash
# 168-dim diagnostic model (fast, 57ms/iter CPU, use 512 BPE)
./target/release/wave-engine data/input.txt --n-bands 84 --n-head 4 --layers 4 \
    --bpe --tokenizer data/tokenizer_512.json

# 384-dim (coherent English, use 2K-4K BPE)
./target/release/wave-engine data/input.txt --n-bands 192 --n-head 8 --layers 8

# 768-dim default (production, 50K BPE)
./target/release/wave-engine data/input.txt --bpe --tokenizer data/tokenizer.json
```

Recommended BPE vocab per dimension (harmonic embedding minimum):
- 168-dim (84 bands): 512 vocab
- 384-dim (192 bands): 2K-4K vocab
- 768-dim (384 bands): 50K vocab

### Analyse your trained model

After training, inspect what the model learned using harmonic coherence diagnostics — the same `cos(n × Δθ)` math from the research framework:

```bash
# Wave structure report: semantic discrimination, depth curve, band census, phase clustering
./target/release/wave-engine --analyze --resume checkpoint.bin --layers 4

# With BPE tokenizer (for proper word-level semantic pairs)
./target/release/wave-engine --analyze --resume checkpoint.bin --layers 24 --out-proj-groups 6 \
    --bpe --tokenizer data/tokenizer.json
```

This runs a forward pass on curated test sentences, extracts phase angles at every layer, and sweeps harmonic coherence between known semantic pairs (cat/dog, noun/verb, etc.). Reports to console and `analysis/wave_report.json`.

### Serve your trained model

After training, serve it with [wave-server](https://github.com/atech-hub/wave-server):

```bash
# In the wave-server repo:
./target/release/wave-server ../wave-engine/checkpoint.bin --bpe ../wave-engine/data/tokenizer.json --port 8080

# Then connect any OpenAI-compatible chat UI to http://localhost:8080/v1
```

## Training Tiers

The engine provides three tiers to match your hardware. All tiers train the same model architecture and produce compatible checkpoints.

| Tier | Flag | Speed (4L) | Loss @ 200 | Params | Hardware |
|------|------|-----------|-----------|--------|----------|
| CPU | *(none)* | 520ms/iter | **2.52** | 2.63M | Any computer |
| wgpu | `--gpu` | 520ms/iter | **2.52** (identical to CPU) | 2.63M | Any GPU (Vulkan/Metal/DX12) |
| Candle CUDA | `--candle` | 213ms/iter | **2.81** (block-diagonal, 4x fewer FFN params) | 657K | NVIDIA only |

*Measured March 22 2026: 4 layers, seq=64, batch=4, 200 iters, Shakespeare, no curriculum, RTX 4070 Ti.*

**CPU** gives the best training quality on any hardware — a Raspberry Pi, a cloud VM, a 10-year-old laptop. **wgpu** runs on any GPU without CUDA, producing identical results to CPU (same loss at every iteration). **Candle CUDA** is 2.4x faster with block-diagonal output projection (6 groups of 128) and perturbative ODE — fewer parameters, faster convergence, slightly higher loss from cosine LR warmup.

**Realistic training times:** The engine runs on any hardware from a Raspberry Pi to an RTX 4090. A 4-layer experiment takes ~2 minutes on any machine. Training time scales with your hardware — the architecture is the same everywhere, only the speed differs.

## Usage

```
wave-engine <data> [options]

Arguments:
  <data>              Path to training data file (text)

Training:
  --iters N           Number of training iterations (default: 500)
  --batch N           Batch size (default: 4)
  --seq N             Sequence length (default: 256)
  --layers N          Number of parallel blocks (default: 24)
  --lr RATE           Learning rate (default: 1e-4 for 384+ bands, 3e-4 otherwise)

Tokenizer:
  --bpe               Use BPE tokenizer (default: character-level)
  --tokenizer FILE    Path to HuggingFace tokenizer.json (default: data/tokenizer.json)

Architecture (runtime configurable):
  --n-bands N         Harmonic frequency bands (default: 384, embedding dim = N×2)
  --n-head N          Attention heads (default: 12)
  --maestro-dim N     Maestro bottleneck width (default: 16)
  --rk4-steps N       ODE integration steps, 1=perturbative (default: 16)
  --out-proj-groups N Block-diagonal groups, 1=dense (default: 6)

GPU:
  --gpu               Enable wgpu GPU acceleration (any GPU)
  --candle            Use Candle/CUDA backend (NVIDIA only, requires candle-backend feature)

Monitoring:
  --monitor           Enable always-on pipeline timing (per-section FFN breakdown)

Resume:
  --resume FILE       Resume training from WCHK checkpoint
  --no-curriculum     Disable progressive band curriculum
```

### Always-On Monitors

These run automatically during training — no flags needed:

| Monitor | Output | Description |
|---------|--------|-------------|
| Loss + time | Console | Per-iteration loss and wall-clock time |
| Gradient norm | Console | `gnorm=` after each iteration |
| NaN recovery | Console | Detects NaN loss, skips step, logs count, continues |
| VRAM tracking | Console (Candle) | Real-time GPU memory via cudarc `mem_get_info` |
| JSONL telemetry | `training_log.jsonl` | Per-iteration: loss, lr, time_ms, vram_mb, nan_skips |
| Checkpoint guard | — | Refuses to save when loss is NaN/Inf/zero |
| Loss in filename | Checkpoint | `checkpoint_iter500_loss2.48.bin` for quick comparison |

The `--monitor` flag adds per-section FFN timing on top of these.

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Three training tiers | ✓ | CPU, wgpu (any GPU), Candle/CUDA (NVIDIA) |
| BPE tokenizer | ✓ | HuggingFace tokenizer.json format, tested with 50K vocab |
| Token cache | ✓ | BPE encoding cached to disk — 6 min encode → 13 sec reload |
| Checkpoint save/load | ✓ | WCHK format with optimizer state, iteration count, resume support |
| Text generation | ✓ | Temperature, top-k, top-p, repetition penalty sampling |
| Curriculum training | ✓ | Soft-mask band unlocking (+1.46pp validated), LN-safe at 24 layers |
| GPU fused ODE | ✓ | One submit, 192 dispatches, zero CPU readbacks between RK4 steps |
| Candle ODE | ✓ | CustomOp1 with identity backward, matching GELU and init |
| FFT ODE | ✓ | OFDM-inspired stencil convolution, validated at 1.19e-7 precision |
| GPU FFT shader | ✓ | 512-point radix-2 Cooley-Tukey in WGSL, compiled and validated |
| Ping-pong buffers | ✓ | Forward/backward read same GPU bits, eliminates precision mismatch |
| Pipeline monitor | ✓ | Per-section FFN timing (mae_in, ODE, mae_out, out_proj) |
| 256-thread Kahan shaders | ✓ | Compensated tree reduction, 0.19 gap (f32 non-associativity floor) |

## Architecture

Wave-engine implements a novel neural architecture where standard MLP feed-forward layers are replaced with coupled harmonic oscillators integrated via a fourth-order Runge-Kutta solver.

**Block structure (GPT-J parallel formulation):**
```
x = x + attention(LN(x)) + FFN(LN(x))
```

Attention and FFN receive the same normalized input and run in parallel. This is different from the sequential GPT-2 formulation and was chosen to enable within-block overlap between attention scoring and ODE integration.

**FFN: Dual-Maestro Kerr-ODE**
```
input → maestro_in (768→16→768) → ODE (RK4, 16 steps) → maestro_out (768→16→768) → out_proj (768→768)
```

The maestro layers are learned bottleneck coordinators (dim=16, a universal constant validated across 128-dim to 1536-dim). The ODE evolves 384 coupled oscillator bands through nonlinear Kerr dynamics — self-phase modulation, cross-phase modulation, and nearest-neighbour coupling.

**Attention: Frozen Harmonic Coherence**

Standard Q/K dot-product attention is replaced with phase-based scoring: `cos(n × Δφ)` where `n` is a learned harmonic number and `Δφ` is the phase difference between positions. Attention weights are frozen during training — only the FFN and layer norms learn. This reduces trainable parameters significantly and is validated to produce equivalent quality on the datasets tested.

**Default configuration:**
| Parameter | Value |
|-----------|-------|
| Layers | 24 (parallel blocks) |
| Embedding dim | 768 (384 bands × 2) |
| Attention heads | 12 |
| Maestro dim | 16 |
| RK4 steps | 16 |
| Block size | 256 |

**Trainable parameters at 24 layers:** ~15.5M (attention frozen)

**Parameter efficiency:** The Kerr-ODE FFN uses 640K parameters per block vs 4.72M for a standard 4x-expansion MLP — 7.4x fewer parameters. A 24-layer wave-engine at 768-dim has 15.5M trainable parameters doing the computation that a standard transformer needs ~115M for. The ODE itself is only 770 parameters per block (384 γ + 384 ω + α + β), replacing a dense 768×3072×768 matrix path with coupled oscillator dynamics that require ~100x fewer FLOPs.

## GPU Acceleration

### wgpu (cross-platform)

The `--gpu` flag enables GPU acceleration via wgpu, which works on any GPU supporting Vulkan, Metal, or DX12 — NVIDIA, AMD, Intel, Apple Silicon.

What runs on GPU:
- Frozen attention output projection (zero quality cost)
- Trained FFN output projection via ping-pong buffers (0.19 quality trade for 2.8x speedup)
- Fused ODE integration (one submit, 192 dispatches, all buffers VRAM-resident)

What stays on CPU:
- Maestro layers (dim=16, too small for GPU dispatch overhead)
- Attention scoring (phase-based, frozen)
- Layer normalization

The engine includes 32+ WGSL compute shaders covering matvec, outer product, attention backward, layer norm, FFT convolution, ODE integration, and RK4 stepping. An always-on pipeline monitor (`--monitor`) shows per-section timing so you can see exactly where every millisecond goes.

### Candle/CUDA (NVIDIA)

```bash
./target/release/wave-engine data/input.txt --iters 200 --candle
```

The Candle backend uses cuBLAS for all matrix operations with automatic forward/backward consistency via autograd. The Kerr-ODE runs as a `CustomOp1` with identity backward (ODE parameters frozen, gradient passthrough). Requires NVIDIA GPU with CUDA support.

### A note on GPU utilisation

If you open Task Manager or `nvidia-smi` during training, you'll see GPU utilisation oscillating between 20-63% instead of the 100% you might expect from PyTorch. **This is normal and by design — it means the architecture is working efficiently.**

A standard transformer keeps the GPU at 100% because every operation is a large dense matrix multiply that runs on GPU. The wave-engine replaces those dense MLP layers (4.72M parameters, 589K multiply-adds per block) with a Kerr-ODE (770 parameters, ~6K operations per block). The GPU simply has less work to do.

The oscillation pattern during Candle CUDA training:

| Phase | GPU% | What's happening |
|-------|------|-----------------|
| Forward/backward matmuls | 50-63% | cuBLAS computing projections across 24 layers |
| ODE integration | 20-30% | CPU running RK4 for 24 blocks (CustomOp1) |
| Optimizer step | ~20% | Parameter updates |

The GPU bursts during matrix operations, dips while the CPU runs the ODE, then bursts again. This is a 15.5M parameter model doing the work of a 115M parameter model — the GPU has less to compute because the architecture is more efficient, not because something is wrong.

**Healthy training indicators:** GPU 40-63% with regular oscillation, temperature under 60°C, VRAM stable (5-6GB at 24 layers), CPU under 15%. If you see GPU at 0% (crashed), 100% sustained (hung), or VRAM climbing continuously (memory leak), something is wrong.

## Pipeline Monitor

The `--monitor` flag enables per-section timing for every component:

```
[FFN fwd] mae_in: 0.5ms  ODE(FFT): 5.5ms  mae_out: 0.5ms  out_proj(GPU): 2.0ms  (64 elem)
[profile fwd] LN: 0.3ms  Attn: 17ms  FFN: 35ms  Total: 53ms
```

The monitor identified that `out_proj` (768×768 matmul) is 66% of FFN time — not the ODE (28%). This finding redirected GPU optimization from the ODE to `out_proj`, resulting in a 2.3x FFN speedup.

## Key Findings

These are validated through testing and documented honestly:

- **Maestro dim=16 is a universal constant.** Tested at dim 16/96/128/160 on 768-dim — all within 0.028 loss. The coordination task is fundamentally low-dimensional.
- **ODE is 28% of FFN time, not 80%.** The output projection (768×768 matmul) dominates at 66%. Per-section monitors revealed this.
- **GPU fast mode reaches CPU quality in the same wall-clock time.** CPU hits loss 2.50 at 36 seconds. GPU hits 2.54 at 36 seconds. Then GPU keeps going — more iterations in the same time.
- **The 0.19 GPU quality gap is f32 non-associativity.** Different addition order on GPU vs CPU gives a different but equally valid result. Proven by identical error at 64 and 256 GPU threads.
- **FFT-based ODE derivative matches sequential at 384 bands on CPU.** OFDM-inspired stencil convolution validated at 1.19e-7 precision. GPU FFT shader written and validated.
- **Frozen attention loses nothing on tested datasets.** Harmonic coherence scoring produces equivalent quality without training attention weights.
- **7.4x parameter efficiency holds across scale.** 640K FFN params per block vs 4.72M standard MLP. Ratio measured at both 128-dim and 768-dim.

## OFDM-Inspired ODE Acceleration

The Kerr-ODE's nearest-neighbour coupling is structurally identical to OFDM subcarrier interference in MIMO wireless systems. The stencil sum `ns[k] = mag²[k-2] + mag²[k-1] + mag²[k+1] + mag²[k+2]` is a convolution with kernel [1,1,0,1,1], which maps to FFT → multiply → IFFT in the frequency domain.

This connection is implemented in `fft_ode.rs` using rustfft, with a GPU FFT shader (`shaders/fft_512.wgsl`) compiled and validated at 1.67e-6 precision vs CPU. At 384 bands, CPU FFT matches sequential speed. The GPU fused ODE path chains all RK4 dispatches in a single command encoder submit with zero CPU readbacks between steps.

## Project Structure

```
src/
├── main.rs                  Training loop, CLI, model init, re-export shims
├── common/                  Shared modules (all tiers)
│   ├── model.rs             Weight structs, OutProjWeights enum, layer_norm, gelu, linear
│   ├── attn.rs              Harmonic coherence attention (frozen)
│   ├── block.rs             Parallel block (GPT-J formulation)
│   ├── ffn.rs               FFN routing via ComputeBackend
│   ├── embed.rs             Frozen harmonic + positional embeddings
│   ├── checkpoint.rs        WCHK v2 checkpoint save/load with resume
│   ├── fft_ode.rs           OFDM-inspired FFT ODE derivative
│   └── optim.rs             Adam optimizer
├── cpu/
│   └── train.rs             CPU/wgpu training loop
├── wgpu_tier/               Cross-platform GPU backend
│   ├── pipelines.rs         Pipeline + shader compilation (35 shaders)
│   ├── dispatch.rs          ComputeBackend trait implementation
│   ├── ops_forward.rs       Forward ops (fused RK4, perturbative, block-diagonal)
│   ├── ops_backward.rs      Backward ops (analytical gradients)
│   ├── resident.rs          Pre-uploaded weight buffers (zero per-iter allocation)
│   ├── buffers.rs           Buffer pool with cache-by-pointer
│   └── ffn_gpu.rs           Ping-pong buffer management
├── candle_tier/             NVIDIA CUDA backend
│   ├── engine.rs            Candle training loop with autograd
│   ├── ode.rs               GPU-native perturbative ODE
│   └── block_diag.rs        Block-diagonal linear via Candle
├── backend.rs               ComputeBackend trait (CPU/GPU abstraction)
├── bpe.rs                   BPE tokenizer (HuggingFace format)
├── monitor.rs               Always-on pipeline timing
└── rng.rs                   Deterministic PRNG

shaders/                     35 WGSL compute shaders
├── kerr_perturbative_batch.wgsl         Single-dispatch perturbative ODE (14x speedup)
├── matvec_block_diagonal_batch.wgsl     Block-diagonal batched matvec
├── matvec_batch_tiled_kahan.wgsl        Forward matmul (256-thread, Kahan compensated)
├── matvec_backward_batch_tiled_kahan.wgsl  Backward d_x
├── outer_product.wgsl                   Backward d_W (Kahan compensated)
├── fft_512.wgsl                         512-point radix-2 FFT
├── kerr_step_batch.wgsl                 Fused ODE forward (RK4)
├── kerr_backward_batch.wgsl             Analytical ODE backward
└── ...                                  Layer norm, GELU, attention, RK4, etc.
```

## Requirements

- **Rust** (edition 2024)
- **Any GPU** for `--gpu` mode (Vulkan, Metal, or DX12 support)
- **NVIDIA GPU + CUDA** for `--candle` mode (optional feature)

No Python, no pip, no CUDA toolkit for the default build. The Candle backend requires `CUDA_COMPUTE_CAP=89` (or your GPU's compute capability) set as an environment variable.

```bash
# Default build (CPU + wgpu)
cargo build --release

# With Candle/CUDA support
cargo build --release --features candle-backend
```

## Contributing

The maintainer ([Marco Da Cunha](https://github.com/atech-hub)) is an IT systems administrator, not a software engineer. This engine was built through collaboration with AI — Claude Desktop for architecture and analysis, Claude Code for implementation and testing. This is stated openly and honestly.

What this means for contributions:

- **Main branch is protected.** All changes go through pull requests.
- **Fork and branch.** Want to improve shaders, add new backends, optimize training? Fork the repo, create a branch, do your work, submit a PR.
- **The validation gate is test results.** Every PR must show that training still converges to baseline quality on at least one tier.
- **The maintainer merges based on testing and description, not code review.** Be clear about what you changed and why.

**Known targets for contributors:**

| Target | Impact | Difficulty |
|--------|--------|------------|
| Wikitext-103 training pipeline | Validate architecture on real English | Medium |
| Tied embeddings / vocab adapter | Solve 50K vocab lm_head explosion (38.6M → 590K params) | Medium |
| Stochastic resonance integration | Validated -8.8% perplexity improvement (α=0.05) | Small |
| Batched inference dispatch | Use engine's existing batch patterns in wave-server | Medium |
| fp16 for linear ops | Halve memory and PCIe transfer size | Medium |
| Speculative decoding | Lower per-token latency for serving | Large |

## Important: Model Compatibility

Models trained by wave-engine use a novel architecture (Kerr-ODE, harmonic coherence attention, block-diagonal projections) that is **not compatible** with standard inference tools. Trained checkpoints will **only** work with [wave-server](https://github.com/atech-hub/wave-server). They cannot be loaded by LM Studio, Ollama, llama.cpp, vLLM, or Hugging Face Transformers — these tools have no code path for ODE integration or harmonic attention.

Train with wave-engine → serve with wave-server → connect any OpenAI-compatible chat UI.

## Related

- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT, 1000+ cloners)
- [wave-server](https://github.com/atech-hub/wave-server) — OpenAI-compatible inference server with KV-cache and wave memory (public, Apache 2.0)
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Persistent wave memory state (public, Apache 2.0)
- [kerr-engine](https://github.com/atech-hub/kerr-engine) — First implementation, historical reference (public, Apache 2.0)

## References and Acknowledgments

The perturbative ODE and GPU architecture draw on published work from multiple fields:

- **Secondini et al. (2015)** — "[Fiber Nonlinearity Mitigation in WDM Systems: Enhanced Split-Step Fourier Method](https://arxiv.org/abs/1507.00921)." The ESSFM single-step approach inspired the perturbative Kerr-ODE: replace iterative numerical integration with a single analytical pass.
- **Lin et al. (2022)** — "[Perturbation-Aided Sample-Based Learned Digital Back-Propagation](https://arxiv.org/abs/2110.05563)." Informed the α/β phase correction structure — self-phase modulation + cross-phase modulation as learnable perturbation terms.
- **Pal et al. (2024)** — "[Coupled Lugiato-Lefèvre Equation for Nonlinear Frequency Comb Generation](https://arxiv.org/abs/2404.05646v2)." The physical basis for Kerr coupling terms between oscillator bands.
- **Ng (2026)** — "[RYS-XLarge: Repeated Blocks for Parameter Efficiency](https://huggingface.co/blog/rys-xlarge)." Inspiration for the repeated-blocks experiment (serve model through blocks twice for quality improvement).
- **Listopad (2025)** — "[ResonanceDB: Phase-Aware Vector Database](https://arxiv.org/abs/2509.09691)." Independent validation of the phase-aware approach to vector similarity, confirming that per-harmonic coherence captures structure invisible to cosine similarity.
- **Ping-pong buffer pattern** — Standard GPU compute technique for maintaining numerical consistency between forward and backward passes. Our implementation was informed by studying a wgpu Game of Life tutorial demonstrating double-buffered compute dispatch.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, analysis, documentation
- **Claude Code** — Implementation, testing, GPU infrastructure
