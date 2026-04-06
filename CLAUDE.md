# Wave Engine — Claude Code Instructions

**FIRST:** Read `C:\claude\CLAUDE-CODE-MEMORY.md` before doing anything. It has established parameters, flag names, and closed decisions that MUST NOT be changed without Marco's approval.

**SECOND:** Read `C:\claude\project-state.md` for current status and running tasks.

## Quick Reference

- Flag names: `--n-bands` (NOT --bands), `--n-head` (NOT --heads)
- Established: `--maestro-dim 16 --rk4-steps 16 --out-proj-groups 1 --phase-native`
- Always `--phase-native` for new training runs
- Dense out_proj (`--out-proj-groups 1`) required at ≤384-dim
- Data path is positional arg 1 (before --candle), not --data
- Test at 168-dim first, then scale up
- Verify ALL parameters before launching long runs

## Collaboration

- Marco: direction and decisions (not a programmer)
- Desktop (Opus): specs in `specs/` — NEVER edits source
- Code: implements, tests, commits. Catches Desktop overclaims.
