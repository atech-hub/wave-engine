# BLOCK-DIAGONAL OUT_PROJ — Implementation Spec for Code
# Date: 2026-03-22
# For: Code (Claude Code)
# Priority: Build now while grammar trains. Test AFTER training finishes.
# DO NOT patch into the running training. Fresh init test only.

---

## What This Is

Replace the dense out_proj (768×768 = 589,824 params per block) with
12 independent group projections (64×64 = 4,096 params each = 49,152 total).
This is a 12x parameter reduction in the single biggest compute bottleneck
(35% of iter time).

## Why It Should Work

The ODE couples nearby bands through the ±2 stencil. Band 50's output is
correlated with bands 48-52. A dense out_proj that mixes band 50 with band
300 is likely doing unnecessary work. Block-diagonal respects the ODE's
local structure — each group of 64 dims (32 bands) mixes internally.

The perturbative ODE training BETTER than RK4-16 already proved that
the architecture benefits from locality. Block-diagonal applies the same
principle to the translator layer.

## Implementation

### New struct: BlockDiagonalLinear

```rust
/// Block-diagonal linear layer — groups of bands processed independently.
/// N_EMBD / n_groups dims per group. Each group is a small dense Linear.
/// Total params: n_groups × (group_size × group_size + group_size)
struct BlockDiagonalLinear {
    groups: Vec<Linear>,
    n_groups: usize,
    group_size: usize,
}

impl BlockDiagonalLinear {
    fn new(n_embd: usize, n_groups: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!(n_embd % n_groups, 0, "n_embd must be divisible by n_groups");
        let group_size = n_embd / n_groups;
        let mut groups = Vec::new();
        for g in 0..n_groups {
            let group_vb = vb.pp(format!("g{g}"));
            groups.push(linear_uniform(group_size, group_size, group_vb)?);
        }
        Ok(Self { groups, n_groups, group_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n_pos, n_embd) = x.dims2()?;
        let mut outputs = Vec::with_capacity(self.n_groups);
        for g in 0..self.n_groups {
            let start = g * self.group_size;
            let group_input = x.narrow(1, start, self.group_size)?;
            let group_output = self.groups[g].forward(&group_input)?;
            outputs.push(group_output);
        }
        Tensor::cat(&outputs, 1)
    }
}
```

### Where to put it

Add the struct to `src/candle_engine.rs` alongside the existing model code.
Or create a new `src/block_diagonal.rs` module — Code's choice.

### How to wire it

In `CandleBlock`, change:
```rust
// OLD:
out_proj: Linear,
```
to:
```rust
// NEW:
out_proj: BlockDiagonalLinear,
```

In model init (`CandleWaveModel::new`), change:
```rust
// OLD:
let out_proj = linear_uniform(N_EMBD, N_EMBD, vs_block.pp("out_proj"))?;
```
to:
```rust
// NEW:
const OUT_PROJ_GROUPS: usize = 12;
let out_proj = BlockDiagonalLinear::new(N_EMBD, OUT_PROJ_GROUPS, vs_block.pp("out_proj"))?;
```

In the forward pass, the call stays the same:
```rust
let ffn_out = block.out_proj.forward(&regulated)?;
```

### VarMap names change

Old: `block.0.out_proj.weight`, `block.0.out_proj.bias`
New: `block.0.out_proj.g0.weight`, `block.0.out_proj.g0.bias`, ..., `block.0.out_proj.g11.weight`, etc.

This means old checkpoints WON'T load. That's fine — we test from fresh init.

### Update extract_wchk_params

The WCHK checkpoint extraction needs updating for the new weight layout.
Each block's out_proj section changes from one large weight matrix to
12 small ones. Code should update the extraction to iterate over groups.

### Configuration: 12 groups

12 groups of 64 dims each. This matches:
- 12 attention heads (same grouping)
- 32 bands per group (ODE stencil couples ±2 = 5 bands, well within 32)
- 49,152 params vs 589,824 dense (12x reduction)

## How to Test

DO NOT patch into the running grammar training.

After grammar training finishes:

```bash
# Fresh init, block-diagonal out_proj, 200 iters on grammar corpus
wave-engine data/grammar/grammar_corpus.txt --candle --iters 200 --batch 8 --no-curriculum
```

Compare loss at iter 200 against the grammar baseline:
- Grammar baseline at iter 200: ~2.52
- If block-diagonal at iter 200 is < 2.7: SUCCESS — keep it
- If block-diagonal at iter 200 is 2.7-3.0: INVESTIGATE — might need larger groups
- If block-diagonal at iter 200 is > 3.0: REVERT — model needs long-range mixing

If 12 groups is too aggressive, try 6 groups (128 dims each, 6x reduction).
If 6 groups also fails, the dense out_proj is necessary and we leave it.

## Expected Impact

| Metric | Dense (current) | Block-diagonal (12 groups) |
|--------|----------------|---------------------------|
| out_proj params/block | 589,824 | 49,152 |
| out_proj total (24 blocks) | 14.2M | 1.2M |
| Total model params (char) | 15.5M | ~2.5M |
| out_proj share of iter | ~35% | ~5% |
| Estimated iter time | 7.5s | ~5s |
| 3000 iters | 6.2 hours | ~4.2 hours |

## What NOT to change

- Don't touch the GPU ODE (just implemented, working)
- Don't change maestro dimensions (validated at 16)
- Don't modify the training loop, optimizer, or LR schedule
- Don't change attention
- Build in a separate file or behind a flag — easy to revert

## Summary for Code

1. Build `BlockDiagonalLinear` struct (separate file, doesn't touch running code)
2. Wait for grammar training to finish
3. Wire it into candle_engine.rs (replace out_proj: Linear)
4. Fresh init, 200 iters on grammar corpus
5. Compare loss at iter 200 against baseline
6. If good → full 3000-iter run
7. If bad → revert, investigate
