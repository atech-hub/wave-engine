//! Compute backend — auto-selection + GPU implementation.
//!
//! ComputeBackend trait and CpuBackend are in common/compute.rs.
//! This file adds GpuBackend (wgpu-specific) and auto-selection logic.

pub use crate::common::compute::{ComputeBackend, CpuBackend};
use crate::model::*;

/// GPU dispatch overhead crossover point (measured on RTX 4070 Ti, 2026-03-12).
/// Below this: CPU wins. Above this: GPU persistent pipeline wins.
/// Derived from benchmark: CPU linear O(n^2) crosses ~120us GPU dispatch overhead.
const GPU_CROSSOVER_DIM: usize = 768;

/// Auto-select the best backend for the given embedding dimension.
/// Returns a boxed trait object so the caller doesn't know which backend it got.
///
/// Rules:
/// - N_EMBD < GPU_CROSSOVER_DIM → CpuBackend (dispatch overhead exceeds compute)
/// - N_EMBD >= GPU_CROSSOVER_DIM and GPU available → GpuBackend
/// - N_EMBD >= GPU_CROSSOVER_DIM but no GPU → CpuBackend (fallback)
/// - --force-cpu / --force-gpu override auto-selection
pub fn auto_select(n_embd: usize, force_cpu: bool, force_gpu: bool, gpu_device: Option<usize>) -> Box<dyn ComputeBackend + Send + Sync> {
    if force_cpu {
        println!("  Backend: CPU (forced)");
        return Box::new(CpuBackend);
    }

    if force_gpu || n_embd >= GPU_CROSSOVER_DIM {
        match try_gpu_backend(gpu_device) {
            Some(gpu) => {
                println!("  Backend: GPU (n_embd={n_embd} >= {GPU_CROSSOVER_DIM} crossover)");
                return Box::new(gpu);
            }
            None => {
                if force_gpu {
                    println!("  Backend: CPU (GPU requested but unavailable)");
                } else {
                    println!("  Backend: CPU (no GPU detected, fallback)");
                }
                return Box::new(CpuBackend);
            }
        }
    }

    println!("  Backend: CPU (n_embd={n_embd} < {GPU_CROSSOVER_DIM} crossover)");
    Box::new(CpuBackend)
}

/// Try to initialize GPU backend. Returns None if no GPU available.
/// If gpu_device is Some(idx), select that specific adapter by index.
fn try_gpu_backend(gpu_device: Option<usize>) -> Option<crate::gpu_backend::GpuBackend> {
    let instance = wgpu::Instance::default();

    let adapter = if let Some(idx) = gpu_device {
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if idx >= adapters.len() {
            println!("  GPU device {idx} not found ({} adapters available)", adapters.len());
            return None;
        }
        // enumerate_adapters returns owned adapters; index into the vec
        let adapters: Vec<_> = adapters.into_iter().collect();
        let info = adapters[idx].get_info();
        println!("  GPU: {} ({:?}) [device {}]", info.name, info.backend, idx);
        // We can't easily pass the adapter to GpuBackend::new(), so we use
        // GpuBackend::with_device_index for explicit selection
        return Some(crate::gpu_backend::GpuBackend::with_device_index(idx));
    } else {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?
    };

    let name = adapter.get_info().name;
    let backend = adapter.get_info().backend;
    println!("  GPU: {name} ({backend:?})");

    Some(crate::gpu_backend::GpuBackend::new())
}


// ComputeBackend trait + CpuBackend moved to common/compute.rs
// GpuBackend impl is in wgpu_tier/dispatch.rs
