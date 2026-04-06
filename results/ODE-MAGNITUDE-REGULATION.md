# ODE Magnitude Regulation — From Fixed Resistor to Adaptive Gain Control

## Date: 2026-03-25

## Origin

During 256-dim BPE training, the model showed a consistent V-shaped divergence pattern: loss descended for ~5K iterations, then rose despite cosine LR decay. Gradient monitors (implemented this session) revealed the cause: the per-band ODE magnitude clamp was fighting the maestro.

The maestro pre-conditioner learns to push band magnitudes higher as training progresses because higher magnitudes carry more information through the ODE. A fixed magnitude clamp at 2.5 (the original value) created a hard ceiling that the maestro hit at iter 4K, causing gradient distortion and progressive divergence.

## The Problem (New Ground)

This regulation problem doesn't exist in standard neural architectures:

- **Standard transformers:** MLP magnitudes are controlled by activation functions (ReLU, GELU) and layer normalisation. No oscillator, no phase wrapping.
- **Fiber optics (Kerr NLS equation):** Magnitude is controlled by launch power and distributed amplification. Parameters are fixed — the system doesn't learn to push harder.
- **Wave-engine:** The maestro is a LEARNED pre-conditioner that actively increases magnitudes as it discovers that higher magnitudes carry more signal through the ODE. No prior system combines a learning agent with a nonlinear oscillator that has physics-based stability limits.

The closest analogy is in electronics: a variable-gain amplifier feeding a nonlinear circuit with a maximum input threshold.

## Experimental Progression

Five tests, each teaching something specific. All at 256-dim, 12 layers, 512 BPE, 20K iterations, cosine LR schedule.

### Test A: Hard clamp at 2.5, lr=1e-4 (baseline)

| Metric | Value |
|--------|-------|
| Best loss | 4.16 |
| Rolling avg at 8-10K | 6.53 (rising) |
| V-shape | YES — severe |
| Clamp rate (late) | 5.9% of bands |
| Max maestro magnitude | 6.8 |

**Finding:** The maestro pushed to 6.8 — 2.7× above the 2.5 clamp. Clamp rate escalated from 1.3% to 5.9% over training. The maestro learned to push harder to compensate for the clamped signal, creating a feedback loop: push → clip → push harder → clip more → gradient distortion → V-shape divergence.

### Test B: Hard clamp at 2.5, lr=3e-5

| Metric | Value |
|--------|-------|
| Best loss | 4.65 |
| Rolling avg at 8-10K | 6.15 (flat) |
| V-shape | No |
| Clamp rate (late) | 2.3% |

**Finding:** Lower LR prevents the maestro from pushing hard enough to fight the clamp. Stable but slow — the model can't reach deep minima because the LR is too conservative.

### Test C: Hard clamp at 5.0, lr=1e-4

| Metric | Value |
|--------|-------|
| Best loss | 3.75 (best of all tests) |
| Rolling avg at 8-10K | 6.06 (flat) |
| Rolling avg at 14-16K | 6.28 (mild rise) |
| V-shape | Mild — delayed |
| Clamp rate progression | 0% → 13% → 30% → 50% → 92% |

**Finding:** Higher threshold gives the maestro room. Best loss of any test (3.75). But the maestro eventually outgrows 5.0 too — clamp rate reaches 92% by iter 14K. The V-shape is delayed but not eliminated. ANY fixed threshold will eventually be outgrown.

### Test D: Soft clamp (tanh compression), threshold 5.0, lr=1e-4

| Metric | Value |
|--------|-------|
| Best loss | 3.83 |
| Rolling avg at 14-16K | 6.02 (descending!) |
| V-shape | No — descending through iter 16K |
| Max magnitude (late) | 7.95 |
| Compression zone | 0.8% |

**Finding:** Smooth compression eliminates the V-shape. The maestro pushed to 7.95 but the tanh handled it gracefully — no hard wall, no cliff, no feedback loop. However, tanh compresses ALL magnitudes (even below threshold): at mag=4.0, threshold=5.0, tanh outputs 3.32 (17% reduction on normal signal). This over-compression slows learning compared to Test C.

### Test E: AGC (Automatic Gain Control) with knee compressor

First run: AGC adapted correctly (threshold 3.28 → 7.99 in 3600 iters) but a single NaN event at iter 3934 poisoned the EMA, collapsing the threshold to the 2.0 floor. Fix: NaN guard in observe().

Second run (with NaN guard): AGC adapted freely but the threshold climbed to 10.0+ because the 3-sigma formula didn't account for ODE physics limits. At threshold 10.0, magnitudes of 7+ entered the ODE, causing phase shift > 180° (chaotic regime). NaN rate reached 48%.

**Finding:** The AGC needs a physics-based ceiling. The ODE stability constraint at α=0.01 is:
- δφ = (α + 4β) × mag² 
- δφ < π/2 (90°) for stability
- mag < √(π/2 / 0.05) ≈ 5.6

The AGC should adapt freely within [2.0, 6.0] — the range bounded by ODE collapse (floor) and ODE chaos (ceiling).

## Key Finding: Physics-Bounded Adaptive Regulation

The regulation system must satisfy three constraints simultaneously:

1. **No under-regulation** (floor): threshold ≥ 2.0, or the model can't express enough magnitude for learning
2. **No over-regulation** (ceiling): threshold ≤ ~6.0, bounded by ODE phase stability (δφ < 90°)
3. **Adaptive within bounds**: the threshold tracks the maestro's natural operating range, not a hand-tuned constant

| Approach | Under-regulation | Over-regulation | Adaptive | Verdict |
|----------|-----------------|-----------------|----------|---------|
| Hard clamp 2.5 | Protected | YES — too tight | No | Fails at iter 4K |
| Hard clamp 5.0 | Protected | Delayed | No | Fails at iter 14K |
| Soft tanh 5.0 | Protected | Mild (over-compresses) | No | Works but slow |
| AGC (no ceiling) | Protected | NO — threshold → 10+ | Yes but unbounded | ODE blows up |
| **AGC + ceiling** | **Protected** | **Protected** | **Yes, bounded** | **Correct design** |

## The Electronics Analogy (Marco's insight)

The progression maps directly to electronics:

| Electronics | Wave-engine | Problem |
|-------------|------------|---------|
| Fixed resistor | Hard clamp at 2.5 | Clips signal, distorts waveform |
| Larger resistor | Hard clamp at 5.0 | Delays clipping, same eventual fate |
| Zener diode | Tanh soft clamp | Smooth compression but affects normal signal |
| AGC circuit | Adaptive threshold | Adapts to signal, only clips outliers |
| AGC with rail voltage | AGC + physics ceiling | Adapts within what the circuit can handle |

The "rail voltage" is the ODE stability limit — the maximum magnitude the nonlinear oscillator can process without phase wrapping. No amount of adaptation can exceed this physical constraint. The AGC must operate within [floor, ceiling] where both bounds are derived from physics, not tuning.

## ODE Stability Mathematics

The Kerr-ODE derivative at band k:
```
dφ/dt = ω_k + α × |Z_k|² + β × Σ|Z_neighbours|²
```

For stability, the phase change per RK4 step must be < π/2:
```
δφ = (α + 4β) × M² × dt < π/2
```

At α = β = 0.01, dt = 1.0/16 (RK4-16):
```
M < √(π/2 / ((0.01 + 0.04) × (1/16))) ≈ √(π/2 / 0.003125) ≈ 22.4
```

But with the perturbative ODE (single-step, dt=1.0):
```
M < √(π/2 / (0.01 + 0.04)) ≈ √(π/2 / 0.05) ≈ 5.6
```

The perturbative ODE is the binding constraint. The ceiling should be ~5.5 for the perturbative path and can be higher for RK4-16 (which subdivides the step).

## Status

- **Hard clamp at 2.5:** REPLACED
- **Hard clamp at 5.0:** TESTED — works as a stopgap
- **Soft tanh:** TESTED — eliminates V-shape, over-compresses normal signal
- **AGC without ceiling:** TESTED — adapts correctly but exceeds ODE stability
- **AGC with physics ceiling:** SPECIFIED — ready for implementation (max_threshold = 6.0)

## Cross-Reference

- Test data: Tests A-E, all at 256-dim 12L 512 BPE 20K iters
- ODE coupling scaling: investigations/wave-structure/INVESTIGATION.md
- Multi-grid embeddings: Pattern 53 in ENGINE-PATTERNS.md
- Maestro ceiling: results/MAESTRO-CEILING-FINDING.md
- Per-band clamp origin: "capacitors regulate current" analogy (Marco, 2026-03-23)

## For ENGINE-PATTERNS.md

Pattern candidate: "Physics-Bounded Adaptive Regulation"
- Any learned system feeding a nonlinear processor needs regulation
- The regulation must be adaptive (the learned system changes its operating range)
- The regulation must be bounded by the processor's physics (not just by convention)
- Floor = minimum useful signal. Ceiling = processor stability limit. AGC adapts between them.
- Derived from the observation: "we should let the model find its sweet spot, not us" (Marco, 2026-03-25)
