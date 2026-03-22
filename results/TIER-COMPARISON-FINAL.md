# Wave Engine — Tier Comparison

## Date: 2026-03-22 (updated with fresh measurements)

## Config: 4 layers, 768-dim, 200 iters, seq=64, batch=4, Shakespeare, no curriculum

| Tier | Flag | Speed | Loss @ 199 | Params | VRAM | Hardware |
|------|------|-------|-----------|--------|------|----------|
| CPU | *(none)* | 520ms/iter | **2.52** | 2.63M (dense) | — | Any computer |
| wgpu GPU | `--gpu` | 520ms/iter | **2.52** (identical) | 2.63M (dense) | — | Any GPU (Vulkan/Metal/DX12) |
| Candle CUDA | `--candle` | 213ms/iter | **2.81** | 657K (block-diag 6×128) | 1329MB stable | NVIDIA only |

## Analysis

### CPU and wgpu produce identical results
- Same loss at every single iteration (verified iter-by-iter)
- Same init seed, same math, same RK4-16 ODE
- wgpu dispatches to GPU shaders but the current config routes most work through CPU
- Both use dense out_proj (1 group)

### Candle CUDA is 2.4x faster with 4x fewer params
- 213ms/iter vs 520ms — cuBLAS + autograd
- Block-diagonal out_proj: 6 groups of 128×128 (657K total params vs 2.63M)
- Perturbative ODE (single-pass analytical, no RK4 steps)
- Loss gap (0.29) partly from cosine LR warmup consuming early iterations
- VRAM rock-solid at 1329MB — no leak (device.synchronize() after optimizer.step())

### Key differences between tiers

| Feature | CPU / wgpu | Candle CUDA |
|---------|-----------|-------------|
| ODE method | RK4-16 (sequential) | Perturbative (single-pass) |
| Out_proj | Dense 768×768 | Block-diagonal 6×128×128 |
| Params | 2.63M | 657K |
| LR schedule | Flat | Cosine with 100-iter warmup |
| Backward | Manual analytical | Autograd (true gradients) |

## Recommended Defaults

- `cargo run --release -- data/input.txt --iters 200` → CPU tier (best quality, any hardware)
- `cargo run --release -- data/input.txt --iters 200 --gpu` → wgpu (identical quality, any GPU)
- `cargo run --release --features candle-backend -- data/input.txt --iters 200 --candle` → Candle (fastest, NVIDIA)
