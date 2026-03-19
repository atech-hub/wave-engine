# Wave Packet Engine — Final Tier Comparison

## Date: 2026-03-19

## All tiers at 4 layers, 768-dim, 200 iters, Shakespeare

| Tier | Config | iter/s | Loss@200 | Gap from CPU | GPU% |
|------|--------|--------|----------|--------------|------|
| CPU (gold) | CpuBackend | 430ms | **2.52** | 0 | 0% |
| wgpu GPU (ping-pong) | CPU FFN + GPU out_proj ping-pong | 800ms (24L) | **2.74** | 0.22 | 22-27% |
| wgpu GPU (backend) | ComputeBackend routing | **125ms** | 3.27 | 0.75 | TBD |
| Candle CUDA (no ODE) | cuBLAS + autograd | 165ms | **2.79** | 0.27 | ~80% |

## Analysis

### CPU tier is the quality leader
- Best loss (2.52), slowest speed
- Every op at exact f32 precision
- No GPU needed — runs on anything

### Candle CUDA is the best GPU quality
- cuBLAS handles forward+backward matmul consistently
- Loss gap (0.27) is from missing ODE, not precision issues
- When CustomOp1 ODE is fixed, should match CPU quality
- 80% GPU utilisation — cuBLAS keeps the GPU busy

### wgpu ping-pong is the best cross-platform GPU
- Works on any GPU (NVIDIA, AMD, Intel, Apple)
- 0.22 loss gap is acceptable
- Ping-pong buffers ensure forward/backward consistency for out_proj
- 22-27% GPU — limited by CPU ODE bottleneck

### wgpu ComputeBackend is fastest but needs quality work
- 125ms/iter — fastest of all tiers
- 0.75 loss gap too large for production
- Gap comes from GPU out_proj × CPU regulated cross-precision product
- Fix: combine ComputeBackend routing with ping-pong for out_proj

## Recommended Defaults

- `cargo run --release -- data/input.txt 200` → CPU tier (best quality)
- `cargo run --release -- data/input.txt 200 --gpu` → ping-pong (best GPU quality)
- `cargo run --release --features candle-backend -- data/input.txt 200 --candle` → Candle (NVIDIA max speed)
