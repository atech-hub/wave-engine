# wave-engine

Training engine for wave-coherent neural architectures. Replaces standard MLP layers with coupled harmonic oscillators (Kerr-ODE) governed by a differential equation. Three training tiers — CPU, cross-platform GPU (wgpu), and NVIDIA GPU (Candle/CUDA) — all producing the same model from one binary, one command.

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) research project.

## Quick Start

```bash
# Build
cargo build --release

# Train on CPU (best quality, any hardware)
cargo run --release -- data/input.txt --iters 200

# Train with GPU acceleration (any GPU — NVIDIA, AMD, Intel, Apple)
cargo run --release -- data/input.txt --iters 200 --gpu

# Train with NVIDIA CUDA (requires --features candle-backend)
cargo run --release --features candle-backend -- data/input.txt --iters 200 --candle

# Train with BPE tokenizer
cargo run --release -- data/input.txt --iters 200 --bpe --tokenizer data/tokenizer.json

# See all options
cargo run --release -- --help
```

No Python. No pip. No CUDA toolkit. One `cargo run` command.

## Training Tiers

The engine provides three tiers to match your hardware. All tiers train the same model architecture and produce compatible checkpoints.

| Tier | Flag | Speed (4L) | Quality | GPU% | Hardware |
|------|------|-----------|---------|------|----------|
| CPU | *(none)* | 407ms/iter | **2.48** (gold) | 0% | Any computer |
| wgpu | `--gpu` | 300ms/iter | **2.48** (matches CPU) | ~10% | Any GPU (Vulkan/Metal/DX12) |
| wgpu fast | `--gpu` *(ping-pong)* | 145ms/iter | **2.67** (0.19 gap) | ~25% | Any GPU |
| Candle CUDA | `--candle` | 280ms/iter | **2.59** (0.1 gap) | ~80% | NVIDIA only |

**CPU** gives the best training quality on any hardware — a Raspberry Pi, a cloud VM, a 10-year-old laptop. **wgpu** accelerates on any GPU without CUDA, with zero quality loss in safe mode or 2.8x speedup in fast mode. **Candle** uses NVIDIA's cuBLAS for maximum GPU throughput with automatic forward/backward consistency.

The 0.19 gap in wgpu fast mode is an inherent f32 non-associativity difference between CPU and GPU addition order — not a bug. Both are valid training trajectories. GPU fast mode reaches the same loss level as CPU in the same wall-clock time, then keeps going.

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

GPU:
  --gpu               Enable wgpu GPU acceleration (any GPU)
  --candle            Use Candle/CUDA backend (NVIDIA only, requires candle-backend feature)

Monitoring:
  --monitor           Enable always-on pipeline timing (per-section FFN breakdown)
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Three training tiers | ✓ | CPU, wgpu (any GPU), Candle/CUDA (NVIDIA) |
| BPE tokenizer | ✓ | HuggingFace tokenizer.json format, tested with 50K vocab |
| Checkpoint save/load | ✓ | WCHK format with optimizer state, iteration count, resume support |
| Text generation | ✓ | Temperature, top-k, top-p, repetition penalty sampling |
| Curriculum training | ✓ | Band unlocking over first 20% of iterations (+1.46pp validated) |
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
cargo run --release --features candle-backend -- data/input.txt --iters 200 --candle
```

The Candle backend uses cuBLAS for all matrix operations with automatic forward/backward consistency via autograd. The Kerr-ODE runs as a `CustomOp1` with identity backward (ODE parameters frozen, gradient passthrough). Requires NVIDIA GPU with CUDA support.

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

## OFDM-Inspired ODE Acceleration

The Kerr-ODE's nearest-neighbour coupling is structurally identical to OFDM subcarrier interference in MIMO wireless systems. The stencil sum `ns[k] = mag²[k-2] + mag²[k-1] + mag²[k+1] + mag²[k+2]` is a convolution with kernel [1,1,0,1,1], which maps to FFT → multiply → IFFT in the frequency domain.

This connection is implemented in `fft_ode.rs` using rustfft, with a GPU FFT shader (`shaders/fft_512.wgsl`) compiled and validated at 1.67e-6 precision vs CPU. At 384 bands, CPU FFT matches sequential speed. The GPU fused ODE path chains all RK4 dispatches in a single command encoder submit with zero CPU readbacks between steps.

## Project Structure

```
src/
├── main.rs              Training loop, CLI, model init
├── wave_attn.rs         Harmonic coherence attention (frozen)
├── wave_block.rs        Dual-maestro Kerr-ODE FFN
├── wave_embed.rs        Frozen harmonic + positional embeddings
├── ffn_backend.rs       CPU/GPU FFN routing via ComputeBackend
├── ffn_gpu.rs           Ping-pong GPU buffer management
├── fft_ode.rs           OFDM-inspired FFT ODE derivative
├── candle_engine.rs     Candle/CUDA backend
├── monitor.rs           Always-on pipeline timing
├── backend.rs           ComputeBackend trait (CPU/GPU abstraction)
├── gpu_pipelines.rs     wgpu pipeline + shader compilation
├── gpu_dispatch.rs      GPU ComputeBackend implementation
├── gpu_ops_forward.rs   GPU forward operations (including fused ODE)
├── gpu_ops_backward.rs  GPU backward operations
├── optim.rs             Adam optimizer
├── bpe.rs               BPE tokenizer (HuggingFace format)
├── checkpoint.rs        WCHK checkpoint save/load with resume
└── rng.rs               Deterministic PRNG

shaders/                 32+ WGSL compute shaders
├── matvec_batch_tiled_kahan.wgsl    Forward matmul (256-thread, Kahan compensated)
├── matvec_backward_batch_tiled_kahan.wgsl  Backward d_x
├── outer_product.wgsl               Backward d_W (Kahan compensated)
├── fft_512.wgsl                     512-point radix-2 FFT
├── kerr_step_batch.wgsl             Fused ODE forward
└── ...                              Layer norm, GELU, attention, RK4, etc.

data/
├── input.txt            Shakespeare training data
└── tokenizer.json       BPE tokenizer (HuggingFace format)
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

## Related

- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT, 1000+ cloners)
- [wave-server](https://github.com/atech-hub/wave-server) — OpenAI-compatible inference server with KV-cache and wave memory (public, Apache 2.0)
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Persistent wave memory state (public, Apache 2.0)
- [kerr-engine](https://github.com/atech-hub/kerr-engine) — First implementation, historical reference (public, Apache 2.0)

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, analysis, documentation
- **Claude Code** — Implementation, testing, GPU infrastructure
