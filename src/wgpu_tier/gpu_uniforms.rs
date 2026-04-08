//! Uniform parameter structs for GPU shader dispatch.
//!
//! Each struct maps to a uniform buffer binding in a WGSL compute shader.
//! All are `#[repr(C)]` + Pod + Zeroable for safe GPU upload via bytemuck.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub use_bias: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormParams {
    pub dim: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrDerivParams {
    pub n_bands: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBwdParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GeluBwdParams {
    pub len: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AttnBwdParams {
    pub seq_len: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub n_embd: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OuterProductParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub compute_bias: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBwdBatchParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatvecBatchParams {
    pub out_dim: u32,
    pub in_dim: u32,
    pub n_pos: u32,
    pub use_bias: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormBatchParams {
    pub dim: u32,
    pub n_pos: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrDerivBatchParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VecScaleAddParams {
    pub len: u32,
    pub scale: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VecAccumulateParams {
    pub len: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FftConvolveParams {
    pub n_bands: u32,
    pub n_positions: u32,
    pub mode: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Rk4CombineParams {
    pub len: u32,
    pub dt_over_6: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DeinterleaveParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GeluParams {
    pub len: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VecAddParams {
    pub len: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KerrBwdBatchParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub alpha: f32,
    pub beta: f32,
    pub chi: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PerturbativeParams {
    pub n_bands: u32,
    pub n_pos: u32,
    pub alpha: f32,
    pub beta: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockDiagParams {
    pub group_size: u32,
    pub n_groups: u32,
    pub n_pos: u32,
    pub n_embd: u32,
}
