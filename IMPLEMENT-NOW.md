# IMPLEMENT NOW: Optimisations for Fresh Production Run
# Date: 2026-03-22
# For: Code (Claude Code)
# From: Desktop (Claude Desktop / Opus) + Marco
# Priority: DO THIS BEFORE STARTING THE NEXT RUN
#
# We've wasted two runs on NaN/checkpoint issues. The next run must be
# the definitive production run. Implement these validated optimisations
# BEFORE hitting start.

---

## Context

All checkpoints are corrupted (from no-clipping run). Fresh start required.
While we're starting fresh anyway, implement the lab-validated optimisations
so the next run is faster and produces more usable results.

---

## Optimisation 1: RK4-8 (VALIDATED — zero risk)

### What
Change RK4 steps from 16 to 8.

### Evidence
- Lab benchmark: MSE 0.000000 vs RK4-16 (identical)
- Lab training: loss 3.07 vs 3.07 (identical)
- ODE converges in 8 steps at current α=0.1, β=0.1

### Implementation
In `src/main.rs`, change:
```rust
const RK4_STEPS: usize = 16;
```
to:
```rust
const RK4_STEPS: usize = 8;
```

That's it. One line. The RK4_STEPS constant propagates to all ODE calls.

### Expected impact
~10% faster iter time (ODE is 28% of FFN, halving it saves ~14% of FFN = ~5-10% of iter).

---

## Optimisation 2: Perturbative ODE (VALIDATED — high confidence)

### What
Replace the iterative RK4 ODE solver with a single-pass analytical computation.

### Evidence
- Lab benchmark: MSE 0.000005 vs RK4-16 (excellent)
- Lab training: loss 2.97 vs 3.07 (BETTER than baseline)
- 14.1x faster in benchmark
- Zero sequential steps, fully parallelisable

### Implementation

#### Step A: Add perturbative function to wave_block.rs

Add this function alongside the existing `kerr_ode_forward_cpu`:

```rust
/// Perturbative ODE — single-pass analytical Kerr computation.
/// Based on first-order perturbation theory from telecom DSP.
/// Replaces 16 iterative RK4 steps with: damping + rotation + correction.
/// Lab-validated: MSE 0.000005 vs RK4-16, trains BETTER (2.97 vs 3.07).
fn kerr_ode_perturbative_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;

    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    // Step 1: Linear solution (damping + base rotation)
    let mut r_lin = vec![0.0f32; n_bands];
    let mut s_lin = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let r = x[k * 2];
        let s = x[k * 2 + 1];
        let decay = (-gamma[k]).exp();
        let cos_w = weights.omega[k].cos();
        let sin_w = weights.omega[k].sin();
        r_lin[k] = decay * (r * cos_w - s * sin_w);
        s_lin[k] = decay * (r * sin_w + s * cos_w);
    }

    // Step 2: First-order nonlinear correction (SPM + XPM)
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let mag_sq = r_lin[k] * r_lin[k] + s_lin[k] * s_lin[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r_lin[k-2]*r_lin[k-2] + s_lin[k-2]*s_lin[k-2]; }
        if k >= 1 { ns += r_lin[k-1]*r_lin[k-1] + s_lin[k-1]*s_lin[k-1]; }
        if k+1 < n_bands { ns += r_lin[k+1]*r_lin[k+1] + s_lin[k+1]*s_lin[k+1]; }
        if k+2 < n_bands { ns += r_lin[k+2]*r_lin[k+2] + s_lin[k+2]*s_lin[k+2]; }
        let delta_phi = weights.alpha * mag_sq + weights.beta * ns;
        out[k * 2]     = r_lin[k] - delta_phi * s_lin[k];
        out[k * 2 + 1] = s_lin[k] + delta_phi * r_lin[k];
    }
    out
}
```

#### Step B: Wire it in

In `wave_block.rs`, in the `dual_maestro_forward_cached` function, replace:
```rust
let kerr_out_all: Vec<Vec<f32>> = precond_all.iter()
    .map(|p| kerr_ode_forward_cpu(&weights.kerr, p)).collect();
```
with:
```rust
let kerr_out_all: Vec<Vec<f32>> = precond_all.iter()
    .map(|p| kerr_ode_perturbative_cpu(&weights.kerr, p)).collect();
```

#### Step C: Wire it in the Candle path too

In `candle_engine.rs`, the ODE runs as a Candle CustomOp1. The CPU forward
inside that custom op calls the same Kerr derivative logic. Find the ODE
forward in the Candle custom op and replace it with the perturbative version.

Search for `kerr_ode` or `rk4_step` in candle_engine.rs — that's where
the Candle ODE forward happens. Replace the RK4 loop with the perturbative
computation (same code as above).

If the Candle custom op is complex to modify, an alternative: just change
the RK4 step count to 8 in the Candle path (Optimisation 1) and use
perturbative only in the CPU path. We can swap the Candle path later.

### Expected impact
14x faster ODE computation. At 768-dim where ODE is 28% of FFN time:
~25% reduction in FFN time, ~10-15% reduction in total iter time.
Combined with RK4-8 as fallback: guaranteed improvement either way.

---

## Optimisation 3: Batch=8 (VALIDATED — confirmed VRAM holds)

### What
Double the batch size from 4 to 8.

### Evidence
- Tested: VRAM stays at 2641MB (Candle processes ODE sequentially)
- Iter time: 10.5s at batch=8 vs 5.7s at batch=4 (1.85x)
- Same tokens/sec but smoother gradients (8 samples averaged)
- At 5000 iters: 10.2M tokens seen (8.4% of dataset) vs 3M at batch=4

### Implementation
In `src/main.rs` or wherever BATCH_SIZE is defined, change from 4 to 8.
Or if it's a CLI arg, pass `--batch 8`.

### Expected impact
Loss curve will descend more consistently (less bounce). The model will
see 3x more data in the same number of iterations. This is the difference
between "word salad" and "broken English" at iter 5000.

---

## Optimisation 4: Checkpoint sanity check (PREVENT FUTURE DISASTERS)

### What
Before saving a checkpoint, check if loss is NaN or 0.0. If so, skip
the save and log a warning. Never overwrite a good checkpoint with garbage.

### Implementation
In the checkpoint save section of candle_engine.rs, add:

```rust
if (iter + 1) % 100 == 0 || iter == total_iters - 1 {
    // SANITY CHECK: don't save corrupted checkpoints
    if total_loss.is_nan() || total_loss == 0.0 {
        eprintln!("  WARNING: Loss is {total_loss} — skipping checkpoint save (corrupted)");
        continue; // or just skip the save block
    }
    // ... existing save code ...
}
```

Also add loss to the checkpoint filename for easy identification:
```rust
let st_path = format!("candle_checkpoint_iter{}_loss{:.2}.safetensors", iter + 1, total_loss);
```

### Expected impact
Prevents the exact disaster we just had. If the model goes NaN, old
good checkpoints are preserved. The filename includes loss so you can
immediately see which checkpoints are good without loading them.

---

## Optimisation 5: Warmup reduction (from earlier finding)

### What
Reduce cosine LR warmup from 200 to 100 iterations.

### Evidence
200-iter warmup keeps LR near zero for too long.
Loss with 200-iter warmup: 11.21→9.13 at iter 50.
Loss without warmup: 11.18→7.65 at iter 50.
100 iters is sufficient warmup for LR=1e-4.

### Implementation
Find the warmup parameter in the cosine LR schedule and change 200 → 100.

---

## Summary: The next run command

After all optimisations are implemented:

```bash
wave-engine data/wikitext.txt --candle --bpe --iters 5000 --seq 256 --no-curriculum --batch 8
```

Settings:
- Fresh init (no resume — all checkpoints corrupted)
- Batch=8 (smoother gradients, 10.2M tokens at 5000 iters)
- RK4-8 or perturbative ODE (lab-validated, faster)
- Grad clipping ON (already fixed)
- device.synchronize() ON (already fixed)
- Cosine LR warmup=100 (not 200)
- Checkpoints every 100 iters WITH loss sanity check
- JSONL telemetry every iter
- Loss in checkpoint filename

Expected: ~9-10s/iter, ~13 hours total, 10.2M tokens.
This is the run that tells us whether the architecture produces English.

---

## Implementation order for Code

1. Change RK4_STEPS = 8 (one line, zero risk)
2. Add perturbative function to wave_block.rs (copy from lab)
3. Wire perturbative into the CPU forward path
4. Wire perturbative into the Candle custom op (or keep RK4-8 there as fallback)
5. Change batch size to 8
6. Add checkpoint sanity check (NaN/0.0 guard)
7. Add loss to checkpoint filename
8. Reduce warmup to 100 iters
9. Test: run 10 iters, verify loss descends, VRAM stable, no NaN
10. Start the production run

Steps 1-8 are code changes. Step 9 is a 2-minute sanity test.
Step 10 is the 13-hour overnight run.
