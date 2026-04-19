# Pure Math Primitives

Every file in this folder satisfies the purity contract:

1. **Deterministic.** Given the same inputs, always returns the same outputs.
2. **No shared mutable state.** No static mut, no Mutex, no OnceLock, no atomics-for-state.
3. **No I/O.** No eprintln! outside #[cfg(test)], no file access, no network.
4. **No RNG unless passed in.** Functions may accept &mut Rng but never create or access global RNG.
5. **Thread-safe by construction.** Immutable inputs, owned outputs.

## What belongs here

- Linear algebra backward primitives (backward.rs)
- ODE derivative and RK4 step functions (ode_deriv.rs)
- ODE backward pass with cached intermediates (ode_backward.rs)
- Attention backward primitives (attn_backward.rs)
- Small math utilities: softplus, cross_entropy (core.rs)
- Loss functions that are pure (phase_loss.rs)

## What does NOT belong here

- Anything that touches AGC (stateful) → stays in common/agc.rs
- Anything that orchestrates forward+backward (policy) → stays in common/ffn.rs
- Anything that does file I/O → stays in common/
- Anything with a ComputeBackend trait boundary → stays in common/compute.rs

## The test

If you're adding a new file and aren't sure whether it belongs here:
can you write a #[test] that calls the function with fixed inputs and asserts
fixed outputs, with no setup, no teardown, no state? If yes, it belongs here.
If you need to init_agc() or create a model first, it doesn't.
