# CONFIGURABLE ARCHITECTURE DIMENSIONS — CLI Flags for All Tiers
# Date: 2026-03-22
# For: Code (Claude Code)
# Priority: Next session — required before any more scaling tests
# Goal: All architecture dimensions configurable via CLI flags. No more editing source.

---

## The Problem

Architecture dimensions are compile-time constants in main.rs:
```rust
const N_BANDS: usize = 384;    // Can't change without recompile
const N_HEAD: usize = 12;      // Same
const MAESTRO_DIM: usize = 16; // Same
```

The Candle tier imports these via `use crate::{N_BANDS, N_EMBD, N_HEAD, ...}`.
To test 16,384-dim, Code had to edit source and rebuild. That's not acceptable
for an engine that targets multiple scales.

The CPU/wgpu path already has `TrainConfig` with some CLI flags (--layers, --batch,
--seq, --lr) but NOT architecture dimensions.

## The Fix

### Step 1: Add architecture flags to CLI

```bash
# New flags (with sensible defaults matching current 768-dim):
--n-bands 384       # Number of harmonic bands (n_embd = n_bands * 2)
--n-head 12         # Number of attention heads
--maestro-dim 16    # Maestro bottleneck width
--rk4-steps 16      # ODE integration steps (CPU/wgpu only)

# These already exist:
--layers 24         # Number of blocks
--out-proj-groups 6 # Block-diagonal groups
--batch 8           # Batch size
--seq 256           # Sequence length
--lr 1e-4           # Learning rate
--iters 5000        # Training iterations
```

### Step 2: Parse in main.rs and pass to ALL tiers

```rust
fn main() {
    // Parse architecture flags (shared by all tiers)
    let n_bands: usize = parse_flag("--n-bands", 384);
    let n_head: usize = parse_flag("--n-head", 12);
    let maestro_dim: usize = parse_flag("--maestro-dim", 16);
    let rk4_steps: usize = parse_flag("--rk4-steps", 16);
    let n_layers: usize = parse_flag("--layers", 24);
    let out_proj_groups: usize = parse_flag("--out-proj-groups", 1);
    let n_embd = n_bands * 2;

    // Validate
    assert_eq!(n_embd % n_head, 0, "n_embd ({n_embd}) must be divisible by n_head ({n_head})");
    assert_eq!(n_embd % out_proj_groups, 0, "n_embd must be divisible by out_proj_groups");

    // Auto-scale rules (validated findings):
    let lr_default = if n_bands > 256 { 1e-4 } else { 3e-4 };
    let lr: f64 = parse_flag("--lr", lr_default);

    // Build ModelConfig
    let config = ModelConfig {
        n_bands,
        n_head,
        n_layers,
        maestro_dim,
        block_size: parse_flag("--seq", 256),
        rk4_n_steps: rk4_steps,
        out_proj_groups,
    };

    if is_candle {
        candle_engine::engine::train_candle(&data_path, n_iters, &config, ...);
    } else {
        train::run_training(TrainConfig { config, ... });
    }
}
```

### Step 3: Update Candle engine to take ModelConfig

**File: `src/candle_tier/engine.rs`**

Change:
```rust
// OLD:
use crate::{N_BANDS, N_EMBD, N_HEAD, N_LAYERS, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};

pub fn train_candle(data_path: &str, n_iters: usize) -> Result<()> {
    // Uses compile-time N_BANDS, N_EMBD, etc. throughout
}
```

To:
```rust
// NEW:
use crate::common::model::ModelConfig;

pub fn train_candle(data_path: &str, n_iters: usize, config: &ModelConfig) -> Result<()> {
    let n_bands = config.n_bands;
    let n_embd = config.n_embd();
    let n_head = config.n_head;
    let n_layers = config.n_layers;
    let maestro_dim = config.maestro_dim;
    // ... use these local variables throughout instead of constants
}
```

### Step 4: Update CPU/wgpu path

The `TrainConfig` struct should include `ModelConfig`:
```rust
pub struct TrainConfig {
    pub config: ModelConfig,  // Architecture dimensions
    pub data_path: String,
    pub n_iters: usize,
    pub batch_size: usize,
    pub lr: f64,
    // ... other training params
}
```

And `init_model()` in main.rs should take `&ModelConfig` instead of using constants:
```rust
fn init_model(vocab_size: usize, seed: u64, config: &ModelConfig) -> WavePacketModel {
    let n_bands = config.n_bands;
    let n_embd = config.n_embd();
    // ... use config throughout
}
```

### Step 5: Remove compile-time constants

Once all code uses `ModelConfig`, remove from main.rs:
```rust
// DELETE these:
const N_BANDS: usize = 384;
const N_EMBD: usize = N_BANDS * 2;
const N_HEAD: usize = 12;
const N_LAYERS: usize = 24;
const MAESTRO_DIM: usize = 16;
const BLOCK_SIZE: usize = 256;
const RK4_STEPS: usize = 16;
```

Also remove from `common/model.rs` if there are any legacy constants there.

### Step 6: Update help text

```
ARCHITECTURE (runtime, configurable):
    --n-bands 384       Harmonic frequency bands (embedding dim = n_bands × 2)
    --n-head 12         Attention heads (n_embd must be divisible by n_head)
    --maestro-dim 16    Maestro bottleneck width
    --rk4-steps 16      ODE integration steps (CPU/wgpu tiers only)
    --out-proj-groups 1 Block-diagonal groups (6 = sweet spot at 768-dim)
    --layers 24         Number of transformer blocks

PRESETS:
    768-dim (default):  --n-bands 384 --n-head 12 --out-proj-groups 6
    Qwen 0.5B scale:    --n-bands 448 --n-head 14 --out-proj-groups 7
    LLaMA 8B scale:     --n-bands 2048 --n-head 32 --out-proj-groups 32
    Stress test:        --n-bands 8192 --n-head 128 --out-proj-groups 128
```

---

## Testing

```bash
# Default (768-dim, should match current behaviour)
cargo run --release --features candle-backend -- data/input.txt --candle --iters 10

# Qwen 0.5B scale
cargo run --release --features candle-backend -- data/input.txt --candle --iters 10 \
    --n-bands 448 --n-head 14 --out-proj-groups 7

# Stress test (16,384-dim)
cargo run --release --features candle-backend -- data/input.txt --candle --iters 10 \
    --n-bands 8192 --n-head 128 --out-proj-groups 128 --layers 4 --seq 32

# CPU tier with custom dims
cargo run --release -- data/input.txt --iters 10 \
    --n-bands 448 --n-head 14 --layers 4

# Verify default matches: compare loss with and without explicit flags
# Both should produce identical loss:
cargo run --release -- data/input.txt --iters 10 --layers 4
cargo run --release -- data/input.txt --iters 10 --layers 4 --n-bands 384 --n-head 12
```

---

## Validation rules (enforce in code)

```rust
// In main.rs after parsing:
assert!(config.n_bands > 0, "n_bands must be > 0");
assert!(config.n_head > 0, "n_head must be > 0");
assert_eq!(config.n_embd() % config.n_head, 0,
    "n_embd ({}) must be divisible by n_head ({})", config.n_embd(), config.n_head);
assert_eq!(config.n_embd() % config.out_proj_groups, 0,
    "n_embd ({}) must be divisible by out_proj_groups ({})", config.n_embd(), config.out_proj_groups);
assert!(config.rk4_n_steps > 0, "rk4_steps must be > 0");

// Auto-warnings:
if config.n_bands > 256 && lr > 1e-4 {
    eprintln!("WARNING: lr={lr} may be too high for {}-dim. Recommend lr=1e-4", config.n_embd());
}
if config.maestro_dim == 16 && config.n_embd() > 4096 {
    eprintln!("WARNING: maestro_dim=16 at {}-dim is {:.0}:1 compression. Consider --maestro-dim 32",
        config.n_embd(), config.n_embd() as f64 / 16.0);
}
```

---

## What NOT to change

- OutProjWeights enum: already configurable via --out-proj-groups
- WCHK checkpoint format: already stores dimensions in header
- Attention (frozen): just reads n_head from config
- ODE parameters: gamma, omega, alpha, beta are per-layer init, not architecture constants
