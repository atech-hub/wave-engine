# Model Configurations

Proven configurations for the wave-engine at each dimension tier. Each configuration includes recommended settings, test results, known limitations, and findings discovered at that scale.

Start with the tier that matches your hardware and goals:

| Tier | Dimension | Vocab | Params | Speed | Use Case |
|------|-----------|-------|--------|-------|----------|
| [**Research**](168-dim/CONFIG.md) | 168 | 512 BPE / char | 186K | 57-80ms/iter | Fast iteration, diagnostics, architecture experiments |
| **Power User** | 384 | 8K BPE | ~500K | ~200ms/iter | Coherent English, scaled diagnostics *(coming soon)* |
| **Production** | 768 | 50K BPE | ~4M | ~1.2s/iter (CPU) / 4.3s (GPU 24L) | Full English, reasoning *(coming soon)* |

## How to Use

1. Pick the tier that fits your hardware
2. Read the CONFIG.md for recommended settings
3. Run the quick test command (500 iters, ~1 min)
4. If pre-flight passes and loss descends, start the full training run

## Key Scaling Rules (apply to all tiers)

- **Gradient balance:** Model params need ≥44% of total trainable params for effective learning
- **Vocab matching:** Max vocab ≈ dimension × layers / 4 (rough guide, see each tier for exact values)
- **ODE coupling:** α=β=0.01 at ≤128 bands, 0.1 at 384+ bands
- **Embeddings:** Multi-grid coprime required for BPE at any dimension below 768
- **Per-band clamp:** 2.5 max magnitude before ODE — always on

## Contributing a Configuration

Trained a model at a new dimension? Create a folder and CONFIG.md following the [168-dim template](168-dim/CONFIG.md). Include:

1. Recommended command with all flags
2. Architecture table (params, layers, heads, groups)
3. Tokenizer comparison (if tested multiple)
4. Training results (best loss, rolling average, speed, NaN rate)
5. Wave diagnostics output (if available)
6. Known limitations
7. Pre-flight expected output
