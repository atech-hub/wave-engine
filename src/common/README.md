# src/common/

Model definitions, I/O, orchestration, and analysis tools.

- **Pure math primitives** → `math/` (purity contract: deterministic, no state, no I/O)
- **Training/diagnostic monitors** → `src/monitors/` (observation, not computation)
- **Everything else** stays here: model structs, checkpoints, data loading, FFN orchestration, generation, galaxy scan, AGC, embeddings, attention forward, CLI help
