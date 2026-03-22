# REFACTOR SPEC — Addendum: Code's Concern Addressed
# Date: 2026-03-22
# Applies to: REFACTOR-UNIFIED-SPEC.md, Section 1d and Phase 2

---

## Code's Concern (valid, Desktop agrees)

> "The spec mentions updating PerBandLinearWeights and KerrMaestroAddWeights too —
> those are kerr-engine structs that get deleted in Phase 4. We should skip migrating
> them in Phase 2 and just give them the as_linear() hack."

## Resolution

**Only `KerrDualMaestroWeights` gets the OutProjWeights enum.**

### Evidence (from grep across all source files):

```
KerrDualMaestroWeights  → used in ffn_backend.rs, wave_block.rs, init.rs (ALL ACTIVE)
KerrMaestroAddWeights   → only init.rs active (2 refs), rest DEAD (gpu_dispatch, gpu_persistent, weights)
PerBandLinearWeights    → only init.rs active (2 refs), rest DEAD (gpu_dispatch, weights)
```

### Why this is safe:

The `FfnWeights` enum handles each variant in SEPARATE match arms:
```rust
match &block.ffn {
    FfnWeights::PerBand(w) => {
        // w.out_proj is LinearWeights → .w and .b work as-is
    }
    FfnWeights::KerrMaestro(w) => {
        // w.out_proj is LinearWeights → .w and .b work as-is
    }
    FfnWeights::KerrDualMaestro(w) => {
        // w.out_proj is OutProjWeights → use .forward(), .flatten_into(), etc.
    }
}
```

Different out_proj types per variant is type-safe because the match arms
never share a generic accessor. Each arm knows its own type.

### Updated Section 1d (replaces original):

```rust
// CHANGE — this is the active FFN for all 24 blocks:
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: OutProjWeights,       // WAS: LinearWeights → NOW: enum
}

// NO CHANGE — legacy single-maestro, deleted in Phase 4:
pub struct KerrMaestroAddWeights {
    pub kerr: KerrWeights,
    pub maestro: MaestroWeights,
    pub out_proj: LinearWeights,        // STAYS LinearWeights
}

// NO CHANGE — block 0 FFN, always dense:
pub struct PerBandLinearWeights {
    pub band_w: Vec<[[f32; 2]; 2]>,
    pub band_b: Vec<[f32; 2]>,
    pub out_proj: LinearWeights,        // STAYS LinearWeights
}
```

### Updated Phase 2 scope:

In every active file, ONLY the `FfnWeights::KerrDualMaestro(w)` match arms
need updating. The `PerBand` and `KerrMaestro` arms stay untouched — their
`w.out_proj.w` and `w.out_proj.b` still compile because the type didn't change.

| File | Total out_proj refs | Refs that ACTUALLY change |
|------|--------------------|-----------------------|
| optim.rs | 30 | ~10 (only KerrDualMaestro arms) |
| gpu_resident.rs | 37 | ~12 (only KerrDualMaestro arms) |
| ffn_backend.rs | 16 | ~8 (only KerrDualMaestro path) |
| wave_block.rs | 6 | ~4 (only KerrDualMaestro path) |
| wave_checkpoint.rs | ~10 | ~5 (only KerrDualMaestro section) |
| init.rs | 3 | ~2 (create OutProjWeights::Dense or ::BlockDiagonal) |

**Total: ~41 references to change, not 207.** The rest compile as-is.

### Consequence:

Phase 2 is roughly HALF the originally estimated work. The compiler will flag
exactly which lines need changing — they're all in KerrDualMaestro match arms.
The PerBand and KerrMaestro arms produce zero compiler errors.

---

## Action for Code

When executing the spec:
1. Read REFACTOR-UNIFIED-SPEC.md for the full plan
2. Apply this addendum: only change KerrDualMaestroWeights in Phase 1d
3. In Phase 2, only update KerrDualMaestro match arms
4. PerBand and KerrMaestro arms: don't touch, they compile as-is
5. No as_linear() hack needed — the types simply don't change
