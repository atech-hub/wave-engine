# CANDLE-CUDA-VMM INTEGRATION SPEC
# For: Code (Claude Code)
# From: Desktop (Claude Desktop / Opus)
# Date: 2026-03-22
# Status: INVESTIGATION + IMPLEMENTATION

---

## Context

The Candle CUDA training path has a memory management problem. CUDA's default
allocator caches freed blocks instead of returning them to the GPU. Over hundreds
of training iterations, this causes:
- Memory growth (11.5GB → OOM at iter 350)
- Iter time climbing (6.3s → 8.9s) from fragmentation
- Unpredictable crashes on long runs

Code's scoping fix (explicit drop, scalar grad norms) reduced usage to 7.2GB
and stabilised iter times. But the underlying CUDA allocator behaviour remains —
it could fragment again at longer runs or larger scales.

`candle-cuda-vmm` (https://docs.rs/candle-cuda-vmm/latest/candle_cuda_vmm/)
provides safe Rust bindings to CUDA's Virtual Memory Management API. This gives
EXPLICIT control over GPU memory allocation from Rust — allocate, deallocate,
track physical usage — instead of hoping CUDA's C++ allocator cooperates.

---

## Step 1: Add dependency (investigation gate)

In Cargo.toml, under [dependencies], add:

```toml
candle-cuda-vmm = { version = "0.1", optional = true }
```

Add to the candle-backend feature:

```toml
candle-backend = ["dep:candle-core", "dep:candle-nn", "dep:candle-cuda-vmm"]
```

**IMPORTANT:** candle-cuda-vmm depends on candle-core ^0.9.2-alpha.1. Our
Cargo.toml has candle-core 0.9. Check if these resolve together. If there's
a version conflict, try:
- `candle-core = { version = "0.9.2-alpha", ... }` 
- Or pin to git: `candle-core = { git = "https://github.com/huggingface/candle", ... }`

If versions don't resolve → STOP. Document the conflict. We may need to
vendor or fork candle-cuda-vmm to match our candle-core version.

**Gate: Does `cargo build --release --features candle-backend` succeed?**

---

## Step 2: Check VMM support at runtime

Before using any VMM features, check if the hardware supports it:

```rust
#[cfg(feature = "candle-backend")]
fn check_vmm_support() -> bool {
    match candle_cuda_vmm::is_vmm_supported() {
        Ok(supported) => {
            if supported {
                eprintln!("[VMM] CUDA Virtual Memory Management supported");
            } else {
                eprintln!("[VMM] CUDA VMM not supported on this hardware");
            }
            supported
        }
        Err(e) => {
            eprintln!("[VMM] Failed to check VMM support: {e}");
            false
        }
    }
}
```

Call this at startup in train_candle(). If VMM is not supported, fall back
to the current allocation pattern (which works, just less controlled).

RTX 4070 Ti is Ada Lovelace (Compute Capability 8.9) — well above the 6.0
minimum. VMM should be supported.

---

## Step 3: Create a memory pool for training

At training startup, after device detection:

```rust
use candle_cuda_vmm::VirtualMemoryPool;

// Create pool sized to the GPU (RTX 4070 Ti = 12GB)
// Virtual capacity can be larger than physical — VMM handles the mapping
let mut vmm_pool = VirtualMemoryPool::new(
    12 * 1024 * 1024 * 1024,  // 12GB virtual address space
    2 * 1024 * 1024,           // 2MB pages (VMM granularity)
    device.clone(),
)?;

eprintln!("[VMM] Pool created: 12GB virtual, 2MB pages");
```

---

## Step 4: Monitor memory between iterations

The primary value even WITHOUT full integration is memory monitoring.
After each training iteration:

```rust
// At end of each iteration, after all tensors should be dropped:
let phys_usage = vmm_pool.physical_memory_usage();
let phys_mb = phys_usage / (1024 * 1024);

// Log to JSONL telemetry
// Add "vram_mb" field to existing telemetry line
if iter % 10 == 0 {
    eprintln!("[VMM] Iter {iter}: physical GPU memory = {phys_mb}MB");
}
```

**THIS ALONE tells us if memory is leaking.** If phys_mb is flat across
iterations, the current scoping fix is sufficient. If it climbs, VMM
can help manage it.

NOTE: `physical_memory_usage()` may only track memory allocated through
the VMM pool, not Candle's internal allocations. Test this — if it reads 0
while Candle is clearly using VRAM, it's only tracking VMM-allocated memory.
In that case, we need cudarc's device memory query instead:

```rust
// Alternative: query total device memory directly via cudarc
// Check if candle_core::cuda_backend exposes memory info
// Or use cudarc directly (candle-cuda-vmm depends on cudarc ^0.18)
```

---

## Step 5: Investigate Candle allocator integration (the hard part)

The BIG question: can VMM replace Candle's internal CUDA allocator?

Candle creates tensors via `Tensor::from_vec(..., device)` and internal ops.
Each tensor allocation goes through Candle's CUDA backend which calls
cudaMalloc/cudaFree. VMM provides a DIFFERENT allocation path — 
cuMemCreate/cuMemMap/cuMemRelease.

**Three approaches, in order of difficulty:**

### Approach A: Memory tracking only (easiest, do first)
Don't replace Candle's allocator. Just use VMM to MONITOR memory:
- Track physical_memory_usage() per iteration
- Log to JSONL telemetry
- Alert if memory grows beyond threshold
- This is pure instrumentation — zero risk to training

### Approach B: Manual pool for large allocations (medium)
Use VMM to pre-allocate a pool for the largest tensors (lm_head backward,
out_proj activations). Then create Candle tensors that point to VMM-managed
memory using unsafe Tensor::from_raw_storage().

This is advanced — need to check if Candle supports creating tensors from
externally-managed CUDA pointers. If CudaStorage can be constructed from
a raw device pointer, this works. If not, this approach is blocked.

```rust
// Pseudocode — check if this API exists:
let raw_ptr = vmm_pool.allocate(offset, size)?;
let storage = CudaStorage::from_raw_ptr(raw_ptr, size, device)?;
let tensor = Tensor::from_storage(storage, shape)?;
```

### Approach C: Custom CUDA allocator (hardest, maximum control)
Replace Candle's entire CUDA allocation strategy with VMM. This likely
requires forking Candle or using cudarc's allocator hooks:

```rust
// cudarc may support custom allocators — check docs
// If so, register VMM as the allocator before creating any tensors
```

This gives full control but is a significant engineering effort.

**RECOMMENDATION: Start with Approach A. Get memory numbers. Then assess
whether B or C is needed based on what the monitoring shows.**

---

## Step 6: Per-iteration memory cleanup

If VMM monitoring shows memory growing, add explicit cleanup between iters:

```rust
// After the batch loop, before the next iteration:
// Force Candle to sync the CUDA stream (ensures all ops complete)
// Then check if any VMM pages can be released

// cudarc may expose cudaDeviceSynchronize or stream sync
// Candle's device.synchronize() if it exists

// Then log physical usage to confirm cleanup happened
```

---

## Step 7: Fallback for non-VMM hardware

The entire VMM integration must be optional. If VMM isn't supported
(older GPU, no CUDA 11.2), training works exactly as before:

```rust
let vmm_pool: Option<VirtualMemoryPool> = if check_vmm_support() {
    Some(VirtualMemoryPool::new(...)?)
} else {
    None
};

// In training loop:
if let Some(ref pool) = vmm_pool {
    let usage = pool.physical_memory_usage();
    // log it
}
```

---

## Summary: What to do

1. Add candle-cuda-vmm to Cargo.toml → check it builds
2. Add is_vmm_supported() check at startup
3. Create VirtualMemoryPool at training start
4. Log physical_memory_usage() every 10 iters to JSONL telemetry
5. Run a 100-iter test → compare VMM-reported memory vs Task Manager
6. If VMM tracking works → we have perfect memory monitoring
7. If VMM shows memory growing → investigate Approach B (manual pool)

**DO NOT attempt Approach B or C before Approach A is tested.**
**DO NOT break the existing training path — VMM is additive, not replacing.**

The goal for today: Steps 1-5. Get memory numbers. That data tells us
whether deeper integration is needed or if the current scoping fix is
sufficient for 3000-iter runs.

---

## Files to modify

- `Cargo.toml` — add dependency
- `src/candle_engine.rs` — add VMM init, monitoring, optional pool
- `training_log.jsonl` — add vram_mb field to telemetry

---

## Reference

- Crate docs: https://docs.rs/candle-cuda-vmm/latest/candle_cuda_vmm/
- Requires: CUDA 11.2+, Compute Capability 6.0+
- License: MIT OR Apache-2.0 (compatible with our Apache-2.0)
- Depends on: candle-core ^0.9.2-alpha.1, cudarc ^0.18
