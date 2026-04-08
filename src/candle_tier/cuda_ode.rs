//! CUDA-native Kerr-ODE — fused AGC + RK4 forward + backward GPU kernels.
//!
//! Two CUDA kernels:
//!   Forward:  fused AGC + RK4 integration, shared memory stencil coupling
//!   Backward: gather-pattern deriv_backward through RK4, recomputes k-values from cached states
//!
//! State cache stays on GPU between forward and backward — zero CPU re-run.
//! Only param grad reduction (tiny) happens on CPU after backward kernel.
//!
//! Forward:  GPU kernel (kerr_ode_fwd)
//! Backward: GPU kernel (kerr_ode_bwd) + CPU param grad reduction

#[cfg(feature = "candle-backend")]
pub mod cuda_ode {
    use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Result, Tensor, Error};
    use candle_core::cuda_backend::{CudaDType, cudarc};
    use cudarc::driver::PushKernelArg;
    use std::sync::{Arc, Mutex, OnceLock};

    /// FWM backward device helper — per-band gradient accumulation from FWM quartets.
    /// For each of the 8 quartet-role memberships this band participates in, accumulates
    /// contributions to this band's d_r, d_s, and d_chi. Translated mechanically from
    /// CPU fwm_quartet_backward in ode_backward.rs.
    /// Convention: only role a accumulates d_chi per quartet to avoid double-counting.
    const CUDA_FWM_BACKWARD_DEVICE_FN: &str = r#"
__device__ inline void compute_fwm_band_backward(
    int band, int n, float chi,
    const float* __restrict__ sr, const float* __restrict__ ss,
    const float* __restrict__ sddr, const float* __restrict__ sdds,
    float* fwm_dr_out, float* fwm_ds_out, float* fwm_dchi_out
) {
    float fdr = 0.0f, fds = 0.0f, fdchi = 0.0f;

    // ════════ Family A: quartet (k-2, k+1, k-1, k) for k in [2, n-1) ════════

    // Role a (band == k-2): k = band+2, valid when band+2 in [2, n-1)
    if (band + 2 >= 2 && band + 2 < n - 1) {
        int kv = band + 2;
        int a = band, b = kv+1, c = kv-1, d = kv;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c], ddrd=sddr[d],ddsd=sdds[d];
        float p_cd_re = rc*rd - sc*sd;
        float p_cd_im = rc*sd + sc*rd;
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[a]
        fdr += ddrb*chi*p_cd_im - ddsb*chi*p_cd_re;
        fdr += ddrc*chi*(sb*rd - rb*sd) - ddsc*chi*(rb*rd + sb*sd);
        fdr += ddrd*chi*(sb*rc - rb*sc) - ddsd*chi*(rb*rc + sb*sc);
        // d_s[a]
        fds -= ddrb*chi*p_cd_re + ddsb*chi*p_cd_im;
        fds += ddrc*chi*(rb*rd + sb*sd) + ddsc*chi*(sb*rd - rb*sd);
        fds += ddrd*chi*(rb*rc + sb*sc) + ddsd*chi*(sb*rc - rb*sc);
        // d_chi — role a accumulates full quartet contribution
        fdchi += ddra*(rb*p_cd_im - sb*p_cd_re) + ddsa*(-(rb*p_cd_re + sb*p_cd_im));
        fdchi += ddrb*(ra*p_cd_im - sa*p_cd_re) + ddsb*(-(ra*p_cd_re + sa*p_cd_im));
        fdchi += ddrc*(p_ab_im*rd - p_ab_re*sd) + ddsc*(-(p_ab_re*rd + p_ab_im*sd));
        fdchi += ddrd*(p_ab_im*rc - p_ab_re*sc) + ddsd*(-(p_ab_re*rc + p_ab_im*sc));
    }

    // Role b (band == k+1): k = band-1, valid when band-1 in [2, n-1)
    if (band - 1 >= 2 && band - 1 < n - 1) {
        int kv = band - 1;
        int a = kv-2, b = band, c = kv-1, d = kv;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c], ddrd=sddr[d],ddsd=sdds[d];
        float p_cd_re = rc*rd - sc*sd;
        float p_cd_im = rc*sd + sc*rd;
        // d_r[b]
        fdr += ddra*chi*p_cd_im - ddsa*chi*p_cd_re;
        fdr += ddrc*chi*(sa*rd - ra*sd) - ddsc*chi*(ra*rd + sa*sd);
        fdr += ddrd*chi*(sa*rc - ra*sc) - ddsd*chi*(ra*rc + sa*sc);
        // d_s[b]
        fds -= ddra*chi*p_cd_re + ddsa*chi*p_cd_im;
        fds += ddrc*chi*(ra*rd + sa*sd) + ddsc*chi*(sa*rd - ra*sd);
        fds += ddrd*chi*(ra*rc + sa*sc) + ddsd*chi*(sa*rc - ra*sc);
        // d_chi: skip — role a already counted this quartet
    }

    // Role c (band == k-1): k = band+1, valid when band+1 in [2, n-1)
    if (band + 1 >= 2 && band + 1 < n - 1) {
        int kv = band + 1;
        int a = kv-2, b = kv+1, c = band, d = kv;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrd=sddr[d],ddsd=sdds[d];
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[c]
        fdr += ddra*chi*(rb*sd - sb*rd) - ddsa*chi*(rb*rd + sb*sd);
        fdr += ddrb*chi*(ra*sd - sa*rd) - ddsb*chi*(ra*rd + sa*sd);
        fdr += ddrd*chi*p_ab_im - ddsd*chi*p_ab_re;
        // d_s[c]
        fds += ddra*chi*(rb*rd + sb*sd) + ddsa*chi*(rb*sd - sb*rd);
        fds += ddrb*chi*(ra*rd + sa*sd) + ddsb*chi*(ra*sd - sa*rd);
        fds -= ddrd*chi*p_ab_re + ddsd*chi*p_ab_im;
        // d_chi: skip
    }

    // Role d (band == k): k = band, valid when band in [2, n-1)
    if (band >= 2 && band < n - 1) {
        int kv = band;
        int a = kv-2, b = kv+1, c = kv-1, d = band;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c];
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[d]
        fdr += ddra*chi*(rb*sc - sb*rc) - ddsa*chi*(rb*rc + sb*sc);
        fdr += ddrb*chi*(ra*sc - sa*rc) - ddsb*chi*(ra*rc + sa*sc);
        fdr += ddrc*chi*p_ab_im - ddsc*chi*p_ab_re;
        // d_s[d]
        fds += ddra*chi*(rb*rc + sb*sc) + ddsa*chi*(rb*sc - sb*rc);
        fds += ddrb*chi*(ra*rc + sa*sc) + ddsb*chi*(ra*sc - sa*rc);
        fds -= ddrc*chi*p_ab_re + ddsc*chi*p_ab_im;
        // d_chi: skip
    }

    // ════════ Family B: quartet (k-1, k+2, k, k+1) for k in [1, n-2) ════════

    // Role a (band == k-1): k = band+1, valid when band+1 in [1, n-2)
    if (band + 1 >= 1 && band + 1 < n - 2) {
        int kv = band + 1;
        int a = band, b = kv+2, c = kv, d = kv+1;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c], ddrd=sddr[d],ddsd=sdds[d];
        float p_cd_re = rc*rd - sc*sd;
        float p_cd_im = rc*sd + sc*rd;
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[a]
        fdr += ddrb*chi*p_cd_im - ddsb*chi*p_cd_re;
        fdr += ddrc*chi*(sb*rd - rb*sd) - ddsc*chi*(rb*rd + sb*sd);
        fdr += ddrd*chi*(sb*rc - rb*sc) - ddsd*chi*(rb*rc + sb*sc);
        // d_s[a]
        fds -= ddrb*chi*p_cd_re + ddsb*chi*p_cd_im;
        fds += ddrc*chi*(rb*rd + sb*sd) + ddsc*chi*(sb*rd - rb*sd);
        fds += ddrd*chi*(rb*rc + sb*sc) + ddsd*chi*(sb*rc - rb*sc);
        // d_chi — role a accumulates full quartet contribution
        fdchi += ddra*(rb*p_cd_im - sb*p_cd_re) + ddsa*(-(rb*p_cd_re + sb*p_cd_im));
        fdchi += ddrb*(ra*p_cd_im - sa*p_cd_re) + ddsb*(-(ra*p_cd_re + sa*p_cd_im));
        fdchi += ddrc*(p_ab_im*rd - p_ab_re*sd) + ddsc*(-(p_ab_re*rd + p_ab_im*sd));
        fdchi += ddrd*(p_ab_im*rc - p_ab_re*sc) + ddsd*(-(p_ab_re*rc + p_ab_im*sc));
    }

    // Role b (band == k+2): k = band-2, valid when band-2 in [1, n-2)
    if (band - 2 >= 1 && band - 2 < n - 2) {
        int kv = band - 2;
        int a = kv-1, b = band, c = kv, d = kv+1;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c], ddrd=sddr[d],ddsd=sdds[d];
        float p_cd_re = rc*rd - sc*sd;
        float p_cd_im = rc*sd + sc*rd;
        // d_r[b]
        fdr += ddra*chi*p_cd_im - ddsa*chi*p_cd_re;
        fdr += ddrc*chi*(sa*rd - ra*sd) - ddsc*chi*(ra*rd + sa*sd);
        fdr += ddrd*chi*(sa*rc - ra*sc) - ddsd*chi*(ra*rc + sa*sc);
        // d_s[b]
        fds -= ddra*chi*p_cd_re + ddsa*chi*p_cd_im;
        fds += ddrc*chi*(ra*rd + sa*sd) + ddsc*chi*(sa*rd - ra*sd);
        fds += ddrd*chi*(ra*rc + sa*sc) + ddsd*chi*(sa*rc - ra*sc);
        // d_chi: skip
    }

    // Role c (band == k): k = band, valid when band in [1, n-2)
    if (band >= 1 && band < n - 2) {
        int kv = band;
        int a = kv-1, b = kv+2, c = band, d = kv+1;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrd=sddr[d],ddsd=sdds[d];
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[c]
        fdr += ddra*chi*(rb*sd - sb*rd) - ddsa*chi*(rb*rd + sb*sd);
        fdr += ddrb*chi*(ra*sd - sa*rd) - ddsb*chi*(ra*rd + sa*sd);
        fdr += ddrd*chi*p_ab_im - ddsd*chi*p_ab_re;
        // d_s[c]
        fds += ddra*chi*(rb*rd + sb*sd) + ddsa*chi*(rb*sd - sb*rd);
        fds += ddrb*chi*(ra*rd + sa*sd) + ddsb*chi*(ra*sd - sa*rd);
        fds -= ddrd*chi*p_ab_re + ddsd*chi*p_ab_im;
        // d_chi: skip
    }

    // Role d (band == k+1): k = band-1, valid when band-1 in [1, n-2)
    if (band - 1 >= 1 && band - 1 < n - 2) {
        int kv = band - 1;
        int a = kv-1, b = kv+2, c = kv, d = band;
        float ra=sr[a],sa=ss[a], rb=sr[b],sb=ss[b], rc=sr[c],sc=ss[c], rd=sr[d],sd=ss[d];
        float ddra=sddr[a],ddsa=sdds[a], ddrb=sddr[b],ddsb=sdds[b];
        float ddrc=sddr[c],ddsc=sdds[c];
        float p_ab_re = ra*rb - sa*sb;
        float p_ab_im = ra*sb + sa*rb;
        // d_r[d]
        fdr += ddra*chi*(rb*sc - sb*rc) - ddsa*chi*(rb*rc + sb*sc);
        fdr += ddrb*chi*(ra*sc - sa*rc) - ddsb*chi*(ra*rc + sa*sc);
        fdr += ddrc*chi*p_ab_im - ddsc*chi*p_ab_re;
        // d_s[d]
        fds += ddra*chi*(rb*rc + sb*sc) + ddsa*chi*(rb*sc - sb*rc);
        fds += ddrb*chi*(ra*rc + sa*sc) + ddsb*chi*(ra*sc - sa*rc);
        fds -= ddrc*chi*p_ab_re + ddsc*chi*p_ab_im;
        // d_chi: skip
    }

    *fwm_dr_out = fdr; *fwm_ds_out = fds; *fwm_dchi_out = fdchi;
}
"#;

    /// FWM device helper — shared between forward and backward kernel sources.
    /// Each band accumulates up to 8 quartet-role memberships (4 roles x 2 families).
    const CUDA_FWM_DEVICE_FN: &str = r#"
__device__ inline void compute_fwm_band(
    int band, int n, float chi,
    const float* __restrict__ sr, const float* __restrict__ ss,
    float* fwm_dr_out, float* fwm_ds_out
) {
    float fdr = 0.0f, fds = 0.0f;
    // Family A: quartet (k-2, k+1, k-1, k)
    // Role a (band == k-2): k = band+2
    if (band + 2 >= 2 && band + 2 < n - 1) {
        int kv = band + 2; int b = kv+1, c = kv-1, d = kv;
        float rb=sr[b],sb=ss[b],rc=sr[c],sc=ss[c],rd=sr[d],sd=ss[d];
        float pcr=rc*rd-sc*sd, pci=rc*sd+sc*rd;
        fdr += chi*(rb*pci - sb*pcr); fds -= chi*(rb*pcr + sb*pci);
    }
    // Role b (band == k+1): k = band-1
    if (band - 1 >= 2 && band - 1 < n - 1) {
        int kv = band - 1; int a = kv-2, c = kv-1, d = kv;
        float ra=sr[a],sa=ss[a],rc=sr[c],sc=ss[c],rd=sr[d],sd=ss[d];
        float pcr=rc*rd-sc*sd, pci=rc*sd+sc*rd;
        fdr += chi*(ra*pci - sa*pcr); fds -= chi*(ra*pcr + sa*pci);
    }
    // Role c (band == k-1): k = band+1
    if (band + 1 >= 2 && band + 1 < n - 1) {
        int kv = band + 1; int a = kv-2, b = kv+1, d = kv;
        float ra=sr[a],sa=ss[a],rb=sr[b],sb=ss[b],rd=sr[d],sd=ss[d];
        float par=ra*rb-sa*sb, pai=ra*sb+sa*rb;
        fdr += chi*(pai*rd - par*sd); fds -= chi*(par*rd + pai*sd);
    }
    // Role d (band == k): k = band
    if (band >= 2 && band < n - 1) {
        int kv = band; int a = kv-2, b = kv+1, c = kv-1;
        float ra=sr[a],sa=ss[a],rb=sr[b],sb=ss[b],rc=sr[c],sc=ss[c];
        float par=ra*rb-sa*sb, pai=ra*sb+sa*rb;
        fdr += chi*(pai*rc - par*sc); fds -= chi*(par*rc + pai*sc);
    }
    // Family B: quartet (k-1, k+2, k, k+1)
    // Role a (band == k-1): k = band+1
    if (band + 1 >= 1 && band + 1 < n - 2) {
        int kv = band + 1; int b = kv+2, c = kv, d = kv+1;
        float rb=sr[b],sb=ss[b],rc=sr[c],sc=ss[c],rd=sr[d],sd=ss[d];
        float pcr=rc*rd-sc*sd, pci=rc*sd+sc*rd;
        fdr += chi*(rb*pci - sb*pcr); fds -= chi*(rb*pcr + sb*pci);
    }
    // Role b (band == k+2): k = band-2
    if (band - 2 >= 1 && band - 2 < n - 2) {
        int kv = band - 2; int a = kv-1, c = kv, d = kv+1;
        float ra=sr[a],sa=ss[a],rc=sr[c],sc=ss[c],rd=sr[d],sd=ss[d];
        float pcr=rc*rd-sc*sd, pci=rc*sd+sc*rd;
        fdr += chi*(ra*pci - sa*pcr); fds -= chi*(ra*pcr + sa*pci);
    }
    // Role c (band == k): k = band
    if (band >= 1 && band < n - 2) {
        int kv = band; int a = kv-1, b = kv+2, d = kv+1;
        float ra=sr[a],sa=ss[a],rb=sr[b],sb=ss[b],rd=sr[d],sd=ss[d];
        float par=ra*rb-sa*sb, pai=ra*sb+sa*rb;
        fdr += chi*(pai*rd - par*sd); fds -= chi*(par*rd + pai*sd);
    }
    // Role d (band == k+1): k = band-1
    if (band - 1 >= 1 && band - 1 < n - 2) {
        int kv = band - 1; int a = kv-1, b = kv+2, c = kv;
        float ra=sr[a],sa=ss[a],rb=sr[b],sb=ss[b],rc=sr[c],sc=ss[c];
        float par=ra*rb-sa*sb, pai=ra*sb+sa*rb;
        fdr += chi*(pai*rc - par*sc); fds -= chi*(par*rc + pai*sc);
    }
    *fwm_dr_out = fdr; *fwm_ds_out = fds;
}
"#;

    /// Forward CUDA kernel — fused AGC + RK4 integration with FWM.
    const CUDA_ODE_FWD_KERNEL: &str = r#"
extern "C" __global__ void kerr_ode_fwd(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ state_cache,
    const float* __restrict__ gamma,
    const float* __restrict__ omega,
    const float* __restrict__ rk4_w,
    float agc_ceiling, float alpha, float beta, float chi,
    int n_bands, int n_steps
) {
    const int pos = blockIdx.x;
    const int k = threadIdx.x;
    if (k >= n_bands) return;
    const float dt = 1.0f / (float)n_steps;
    const int embd = n_bands * 2;

    float r = input[pos * embd + k * 2];
    float s = input[pos * embd + k * 2 + 1];

    // Fused AGC
    float mag = sqrtf(r * r + s * s + 1e-12f);
    float scale = fminf(1.0f, agc_ceiling / mag);
    r *= scale; s *= scale;

    const float g = gamma[k], w = omega[k];
    const float w0 = rk4_w[0], w1 = rk4_w[1], w2 = rk4_w[2], w3 = rk4_w[3];
    extern __shared__ float smem[];
    float* smem_mag = smem;
    float* smem_r = smem + n_bands;
    float* smem_s = smem + 2 * n_bands;

    for (int step = 0; step < n_steps; step++) {
        state_cache[(pos * n_steps + step) * embd + k * 2]     = r;
        state_cache[(pos * n_steps + step) * embd + k * 2 + 1] = s;

        // k1
        smem_mag[k] = r*r + s*s; smem_r[k] = r; smem_s[k] = s; __syncthreads();
        float ns = 0.0f;
        if (k>=2) ns += smem_mag[k-2]; if (k>=1) ns += smem_mag[k-1];
        if (k+1<n_bands) ns += smem_mag[k+1]; if (k+2<n_bands) ns += smem_mag[k+2];
        float phi = w + alpha*smem_mag[k] + beta*ns;
        float k1r = -g*r - phi*s, k1s = -g*s + phi*r;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, smem_r, smem_s, &fdr, &fds);
            k1r += fdr; k1s += fds;
        }
        __syncthreads();

        // k2
        float r2=r+0.5f*dt*k1r, s2=s+0.5f*dt*k1s;
        smem_mag[k] = r2*r2+s2*s2; smem_r[k] = r2; smem_s[k] = s2; __syncthreads();
        ns=0.0f;
        if (k>=2) ns+=smem_mag[k-2]; if (k>=1) ns+=smem_mag[k-1];
        if (k+1<n_bands) ns+=smem_mag[k+1]; if (k+2<n_bands) ns+=smem_mag[k+2];
        phi = w+alpha*smem_mag[k]+beta*ns;
        float k2r=-g*r2-phi*s2, k2s=-g*s2+phi*r2;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, smem_r, smem_s, &fdr, &fds);
            k2r += fdr; k2s += fds;
        }
        __syncthreads();

        // k3
        float r3=r+0.5f*dt*k2r, s3=s+0.5f*dt*k2s;
        smem_mag[k] = r3*r3+s3*s3; smem_r[k] = r3; smem_s[k] = s3; __syncthreads();
        ns=0.0f;
        if (k>=2) ns+=smem_mag[k-2]; if (k>=1) ns+=smem_mag[k-1];
        if (k+1<n_bands) ns+=smem_mag[k+1]; if (k+2<n_bands) ns+=smem_mag[k+2];
        phi = w+alpha*smem_mag[k]+beta*ns;
        float k3r=-g*r3-phi*s3, k3s=-g*s3+phi*r3;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, smem_r, smem_s, &fdr, &fds);
            k3r += fdr; k3s += fds;
        }
        __syncthreads();

        // k4
        float r4=r+dt*k3r, s4=s+dt*k3s;
        smem_mag[k] = r4*r4+s4*s4; smem_r[k] = r4; smem_s[k] = s4; __syncthreads();
        ns=0.0f;
        if (k>=2) ns+=smem_mag[k-2]; if (k>=1) ns+=smem_mag[k-1];
        if (k+1<n_bands) ns+=smem_mag[k+1]; if (k+2<n_bands) ns+=smem_mag[k+2];
        phi = w+alpha*smem_mag[k]+beta*ns;
        float k4r=-g*r4-phi*s4, k4s=-g*s4+phi*r4;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, smem_r, smem_s, &fdr, &fds);
            k4r += fdr; k4s += fds;
        }
        __syncthreads();

        r += dt*(w0*k1r + w1*k2r + w2*k3r + w3*k4r);
        s += dt*(w0*k1s + w1*k2s + w2*k3s + w3*k4s);
    }
    output[pos*embd + k*2] = r;
    output[pos*embd + k*2+1] = s;
}
"#;

    /// Backward CUDA kernel — gather-pattern deriv_backward through RK4.
    /// Recomputes k-values from cached states (no extra memory from forward).
    /// Shared memory: 5 × n_bands floats (s_mag, s_ddr, s_dds, s_r, s_s).
    const CUDA_ODE_BWD_KERNEL: &str = r#"
extern "C" __global__ void kerr_ode_bwd(
    const float* __restrict__ d_output,      // [n_pos, n_embd]
    const float* __restrict__ state_cache,   // [n_pos, n_steps, n_embd]
    float* __restrict__ d_input,             // [n_pos, n_embd]
    float* __restrict__ d_gamma_out,         // [n_pos, n_bands]
    float* __restrict__ d_alpha_out,         // [n_pos]
    float* __restrict__ d_beta_out,          // [n_pos]
    float* __restrict__ d_chi_out,           // [n_pos]
    float* __restrict__ d_rk4w_out,          // [n_pos, 4]
    const float* __restrict__ gamma,         // [n_bands]
    const float* __restrict__ omega,         // [n_bands]
    const float* __restrict__ gamma_raw,     // [n_bands] for softplus derivative
    const float* __restrict__ rk4_w,         // [4]
    float alpha, float beta, float chi,
    int n_bands, int n_steps
) {
    const int pos = blockIdx.x;
    const int k = threadIdx.x;
    if (k >= n_bands) return;

    const float dt = 1.0f / (float)n_steps;
    const int embd = n_bands * 2;
    const float g = gamma[k], w_k = omega[k];
    const float rw0 = rk4_w[0], rw1 = rk4_w[1], rw2 = rk4_w[2], rw3 = rk4_w[3];

    // Shared memory: 5 arrays of n_bands floats
    extern __shared__ float smem[];
    float* s_mag = smem;                   // mag_sq for forward recomputation
    float* s_ddr = smem + n_bands;         // d_dr for gather-pattern backward
    float* s_dds = smem + 2 * n_bands;     // d_ds for gather-pattern backward
    float* s_r   = smem + 3 * n_bands;     // r state at eval point
    float* s_s   = smem + 4 * n_bands;     // s state at eval point

    // Load output gradient
    float dr = d_output[pos * embd + k * 2];
    float ds = d_output[pos * embd + k * 2 + 1];

    // Per-band parameter gradient accumulators
    float d_gamma_k = 0.0f;
    float d_alpha_k = 0.0f;
    float d_beta_k  = 0.0f;
    float d_chi_k   = 0.0f;
    float d_rk4w_0 = 0, d_rk4w_1 = 0, d_rk4w_2 = 0, d_rk4w_3 = 0;

    // Walk backward through RK4 steps
    for (int step = n_steps - 1; step >= 0; step--) {
        // Load cached state at step start
        float r0 = state_cache[(pos*n_steps + step)*embd + k*2];
        float s0 = state_cache[(pos*n_steps + step)*embd + k*2 + 1];

        // ════════ Recompute k1, k2, k3, k4 from (r0, s0) ════════

        // k1 at (r0, s0)
        s_mag[k] = r0*r0 + s0*s0; s_r[k] = r0; s_s[k] = s0; __syncthreads();
        float ns1 = 0.0f;
        if (k>=2) ns1 += s_mag[k-2]; if (k>=1) ns1 += s_mag[k-1];
        if (k+1<n_bands) ns1 += s_mag[k+1]; if (k+2<n_bands) ns1 += s_mag[k+2];
        float phi1 = w_k + alpha*s_mag[k] + beta*ns1;
        float k1r = -g*r0 - phi1*s0, k1s = -g*s0 + phi1*r0;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, s_r, s_s, &fdr, &fds);
            k1r += fdr; k1s += fds;
        }
        __syncthreads();

        // k2 at (r0 + 0.5*dt*k1)
        float r2 = r0+0.5f*dt*k1r, s2 = s0+0.5f*dt*k1s;
        s_mag[k] = r2*r2+s2*s2; s_r[k] = r2; s_s[k] = s2; __syncthreads();
        float ns2 = 0.0f;
        if (k>=2) ns2 += s_mag[k-2]; if (k>=1) ns2 += s_mag[k-1];
        if (k+1<n_bands) ns2 += s_mag[k+1]; if (k+2<n_bands) ns2 += s_mag[k+2];
        float phi2 = w_k+alpha*s_mag[k]+beta*ns2;
        float k2r = -g*r2-phi2*s2, k2s = -g*s2+phi2*r2;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, s_r, s_s, &fdr, &fds);
            k2r += fdr; k2s += fds;
        }
        __syncthreads();

        // k3 at (r0 + 0.5*dt*k2)
        float r3 = r0+0.5f*dt*k2r, s3 = s0+0.5f*dt*k2s;
        s_mag[k] = r3*r3+s3*s3; s_r[k] = r3; s_s[k] = s3; __syncthreads();
        float ns3 = 0.0f;
        if (k>=2) ns3 += s_mag[k-2]; if (k>=1) ns3 += s_mag[k-1];
        if (k+1<n_bands) ns3 += s_mag[k+1]; if (k+2<n_bands) ns3 += s_mag[k+2];
        float phi3 = w_k+alpha*s_mag[k]+beta*ns3;
        float k3r = -g*r3-phi3*s3, k3s = -g*s3+phi3*r3;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, s_r, s_s, &fdr, &fds);
            k3r += fdr; k3s += fds;
        }
        __syncthreads();

        // k4 at (r0 + dt*k3)
        float r4 = r0+dt*k3r, s4 = s0+dt*k3s;
        s_mag[k] = r4*r4+s4*s4; s_r[k] = r4; s_s[k] = s4; __syncthreads();
        float ns4 = 0.0f;
        if (k>=2) ns4 += s_mag[k-2]; if (k>=1) ns4 += s_mag[k-1];
        if (k+1<n_bands) ns4 += s_mag[k+1]; if (k+2<n_bands) ns4 += s_mag[k+2];
        float phi4 = w_k+alpha*s_mag[k]+beta*ns4;
        float k4r = -g*r4-phi4*s4, k4s = -g*s4+phi4*r4;
        if (chi != 0.0f && n_bands > 4) {
            float fdr, fds; compute_fwm_band(k, n_bands, chi, s_r, s_s, &fdr, &fds);
            k4r += fdr; k4s += fds;
        }
        __syncthreads();

        // ════════ RK4 backward ════════
        // d_r/d_s hold gradients w.r.t. state after this step
        // Gradients w.r.t. k-values from RK4 combination
        float dk1r = dr*dt*rw0, dk1s = ds*dt*rw0;
        float dk2r = dr*dt*rw1, dk2s = ds*dt*rw1;
        float dk3r = dr*dt*rw2, dk3s = ds*dt*rw2;
        float dk4r = dr*dt*rw3, dk4s = ds*dt*rw3;

        // RK4 weight gradients
        d_rk4w_0 += dr*dt*k1r + ds*dt*k1s;
        d_rk4w_1 += dr*dt*k2r + ds*dt*k2s;
        d_rk4w_2 += dr*dt*k3r + ds*dt*k3s;
        d_rk4w_3 += dr*dt*k4r + ds*dt*k4s;

        // ── k4 backward at (r4, s4) ──
        // Load state and incoming grads into shared memory for gather
        s_r[k] = r4; s_s[k] = s4; s_ddr[k] = dk4r; s_dds[k] = dk4s;
        __syncthreads();
        // Self-band contributions
        float eval_dr4 = dk4r * (-g - 2.0f*alpha*r4*s4)
                        + dk4s * (phi4 + 2.0f*alpha*r4*r4);
        float eval_ds4 = dk4r * (-phi4 - 2.0f*alpha*s4*s4)
                        + dk4s * (-g + 2.0f*alpha*r4*s4);
        // Cross-band (gather from neighbours)
        if (k>=2) { eval_dr4 += s_ddr[k-2]*(-2.0f*beta*r4*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*r4*s_r[k-2]);
                     eval_ds4 += s_ddr[k-2]*(-2.0f*beta*s4*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*s4*s_r[k-2]); }
        if (k>=1) { eval_dr4 += s_ddr[k-1]*(-2.0f*beta*r4*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*r4*s_r[k-1]);
                     eval_ds4 += s_ddr[k-1]*(-2.0f*beta*s4*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*s4*s_r[k-1]); }
        if (k+1<n_bands) { eval_dr4 += s_ddr[k+1]*(-2.0f*beta*r4*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*r4*s_r[k+1]);
                            eval_ds4 += s_ddr[k+1]*(-2.0f*beta*s4*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*s4*s_r[k+1]); }
        if (k+2<n_bands) { eval_dr4 += s_ddr[k+2]*(-2.0f*beta*r4*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*r4*s_r[k+2]);
                            eval_ds4 += s_ddr[k+2]*(-2.0f*beta*s4*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*s4*s_r[k+2]); }
        // FWM backward for k4
        if (chi != 0.0f && n_bands > 4) {
            float fwm_dr4, fwm_ds4, fwm_dchi4;
            compute_fwm_band_backward(k, n_bands, chi, s_r, s_s, s_ddr, s_dds, &fwm_dr4, &fwm_ds4, &fwm_dchi4);
            eval_dr4 += fwm_dr4;
            eval_ds4 += fwm_ds4;
            d_chi_k += fwm_dchi4;
        }
        // Param grads from k4
        d_gamma_k += dk4r*(-r4) + dk4s*(-s4);
        d_alpha_k += dk4r*(-(r4*r4+s4*s4)*s4) + dk4s*((r4*r4+s4*s4)*r4);
        d_beta_k  += dk4r*(-ns4*s4) + dk4s*(ns4*r4);
        __syncthreads();
        // Route: eval grads → d_r0 + d_k3 (k4 eval point = r0 + dt*k3)
        dr += eval_dr4; ds += eval_ds4;       // to d_r0
        dk3r += eval_dr4 * dt;                // chain through k4 eval point
        dk3s += eval_ds4 * dt;

        // ── k3 backward at (r3, s3) ── (dk3 now includes k4 chain contribution)
        s_r[k] = r3; s_s[k] = s3; s_ddr[k] = dk3r; s_dds[k] = dk3s;
        __syncthreads();
        float eval_dr3 = dk3r * (-g - 2.0f*alpha*r3*s3)
                        + dk3s * (phi3 + 2.0f*alpha*r3*r3);
        float eval_ds3 = dk3r * (-phi3 - 2.0f*alpha*s3*s3)
                        + dk3s * (-g + 2.0f*alpha*r3*s3);
        if (k>=2) { eval_dr3 += s_ddr[k-2]*(-2.0f*beta*r3*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*r3*s_r[k-2]);
                     eval_ds3 += s_ddr[k-2]*(-2.0f*beta*s3*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*s3*s_r[k-2]); }
        if (k>=1) { eval_dr3 += s_ddr[k-1]*(-2.0f*beta*r3*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*r3*s_r[k-1]);
                     eval_ds3 += s_ddr[k-1]*(-2.0f*beta*s3*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*s3*s_r[k-1]); }
        if (k+1<n_bands) { eval_dr3 += s_ddr[k+1]*(-2.0f*beta*r3*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*r3*s_r[k+1]);
                            eval_ds3 += s_ddr[k+1]*(-2.0f*beta*s3*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*s3*s_r[k+1]); }
        if (k+2<n_bands) { eval_dr3 += s_ddr[k+2]*(-2.0f*beta*r3*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*r3*s_r[k+2]);
                            eval_ds3 += s_ddr[k+2]*(-2.0f*beta*s3*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*s3*s_r[k+2]); }
        // FWM backward for k3
        if (chi != 0.0f && n_bands > 4) {
            float fwm_dr3, fwm_ds3, fwm_dchi3;
            compute_fwm_band_backward(k, n_bands, chi, s_r, s_s, s_ddr, s_dds, &fwm_dr3, &fwm_ds3, &fwm_dchi3);
            eval_dr3 += fwm_dr3;
            eval_ds3 += fwm_ds3;
            d_chi_k += fwm_dchi3;
        }
        d_gamma_k += dk3r*(-r3) + dk3s*(-s3);
        d_alpha_k += dk3r*(-(r3*r3+s3*s3)*s3) + dk3s*((r3*r3+s3*s3)*r3);
        d_beta_k  += dk3r*(-ns3*s3) + dk3s*(ns3*r3);
        __syncthreads();
        dr += eval_dr3; ds += eval_ds3;
        dk2r += eval_dr3 * 0.5f * dt;         // k3 eval = r0 + 0.5*dt*k2
        dk2s += eval_ds3 * 0.5f * dt;

        // ── k2 backward at (r2, s2) ──
        s_r[k] = r2; s_s[k] = s2; s_ddr[k] = dk2r; s_dds[k] = dk2s;
        __syncthreads();
        float eval_dr2 = dk2r * (-g - 2.0f*alpha*r2*s2)
                        + dk2s * (phi2 + 2.0f*alpha*r2*r2);
        float eval_ds2 = dk2r * (-phi2 - 2.0f*alpha*s2*s2)
                        + dk2s * (-g + 2.0f*alpha*r2*s2);
        if (k>=2) { eval_dr2 += s_ddr[k-2]*(-2.0f*beta*r2*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*r2*s_r[k-2]);
                     eval_ds2 += s_ddr[k-2]*(-2.0f*beta*s2*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*s2*s_r[k-2]); }
        if (k>=1) { eval_dr2 += s_ddr[k-1]*(-2.0f*beta*r2*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*r2*s_r[k-1]);
                     eval_ds2 += s_ddr[k-1]*(-2.0f*beta*s2*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*s2*s_r[k-1]); }
        if (k+1<n_bands) { eval_dr2 += s_ddr[k+1]*(-2.0f*beta*r2*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*r2*s_r[k+1]);
                            eval_ds2 += s_ddr[k+1]*(-2.0f*beta*s2*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*s2*s_r[k+1]); }
        if (k+2<n_bands) { eval_dr2 += s_ddr[k+2]*(-2.0f*beta*r2*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*r2*s_r[k+2]);
                            eval_ds2 += s_ddr[k+2]*(-2.0f*beta*s2*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*s2*s_r[k+2]); }
        // FWM backward for k2
        if (chi != 0.0f && n_bands > 4) {
            float fwm_dr2, fwm_ds2, fwm_dchi2;
            compute_fwm_band_backward(k, n_bands, chi, s_r, s_s, s_ddr, s_dds, &fwm_dr2, &fwm_ds2, &fwm_dchi2);
            eval_dr2 += fwm_dr2;
            eval_ds2 += fwm_ds2;
            d_chi_k += fwm_dchi2;
        }
        d_gamma_k += dk2r*(-r2) + dk2s*(-s2);
        d_alpha_k += dk2r*(-(r2*r2+s2*s2)*s2) + dk2s*((r2*r2+s2*s2)*r2);
        d_beta_k  += dk2r*(-ns2*s2) + dk2s*(ns2*r2);
        __syncthreads();
        dr += eval_dr2; ds += eval_ds2;
        dk1r += eval_dr2 * 0.5f * dt;         // k2 eval = r0 + 0.5*dt*k1
        dk1s += eval_ds2 * 0.5f * dt;

        // ── k1 backward at (r0, s0) ──
        s_r[k] = r0; s_s[k] = s0; s_ddr[k] = dk1r; s_dds[k] = dk1s;
        __syncthreads();
        float eval_dr1 = dk1r * (-g - 2.0f*alpha*r0*s0)
                        + dk1s * (phi1 + 2.0f*alpha*r0*r0);
        float eval_ds1 = dk1r * (-phi1 - 2.0f*alpha*s0*s0)
                        + dk1s * (-g + 2.0f*alpha*r0*s0);
        if (k>=2) { eval_dr1 += s_ddr[k-2]*(-2.0f*beta*r0*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*r0*s_r[k-2]);
                     eval_ds1 += s_ddr[k-2]*(-2.0f*beta*s0*s_s[k-2]) + s_dds[k-2]*(2.0f*beta*s0*s_r[k-2]); }
        if (k>=1) { eval_dr1 += s_ddr[k-1]*(-2.0f*beta*r0*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*r0*s_r[k-1]);
                     eval_ds1 += s_ddr[k-1]*(-2.0f*beta*s0*s_s[k-1]) + s_dds[k-1]*(2.0f*beta*s0*s_r[k-1]); }
        if (k+1<n_bands) { eval_dr1 += s_ddr[k+1]*(-2.0f*beta*r0*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*r0*s_r[k+1]);
                            eval_ds1 += s_ddr[k+1]*(-2.0f*beta*s0*s_s[k+1]) + s_dds[k+1]*(2.0f*beta*s0*s_r[k+1]); }
        if (k+2<n_bands) { eval_dr1 += s_ddr[k+2]*(-2.0f*beta*r0*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*r0*s_r[k+2]);
                            eval_ds1 += s_ddr[k+2]*(-2.0f*beta*s0*s_s[k+2]) + s_dds[k+2]*(2.0f*beta*s0*s_r[k+2]); }
        // FWM backward for k1
        if (chi != 0.0f && n_bands > 4) {
            float fwm_dr1, fwm_ds1, fwm_dchi1;
            compute_fwm_band_backward(k, n_bands, chi, s_r, s_s, s_ddr, s_dds, &fwm_dr1, &fwm_ds1, &fwm_dchi1);
            eval_dr1 += fwm_dr1;
            eval_ds1 += fwm_ds1;
            d_chi_k += fwm_dchi1;
        }
        d_gamma_k += dk1r*(-r0) + dk1s*(-s0);
        d_alpha_k += dk1r*(-(r0*r0+s0*s0)*s0) + dk1s*((r0*r0+s0*s0)*r0);
        d_beta_k  += dk1r*(-ns1*s0) + dk1s*(ns1*r0);
        __syncthreads();
        dr += eval_dr1; ds += eval_ds1;
        // k1 eval point IS (r0, s0) — no upstream k-value to chain to
    }

    // Write d_input
    d_input[pos*embd + k*2] = dr;
    d_input[pos*embd + k*2+1] = ds;

    // Write per-band d_gamma (apply softplus derivative: sigmoid(gamma_raw))
    float sp_deriv = 1.0f / (1.0f + expf(-gamma_raw[k]));
    d_gamma_out[pos * n_bands + k] = d_gamma_k * sp_deriv;

    // ── Reduce d_alpha, d_beta, d_rk4w across bands using atomicAdd to shared memory ──
    // (n_bands ≤ 384, so contention is minimal)
    if (k == 0) { s_mag[0] = 0.0f; }
    __syncthreads();
    atomicAdd(&s_mag[0], d_alpha_k);
    __syncthreads();
    if (k == 0) d_alpha_out[pos] = s_mag[0];
    __syncthreads();

    if (k == 0) { s_mag[0] = 0.0f; }
    __syncthreads();
    atomicAdd(&s_mag[0], d_beta_k);
    __syncthreads();
    if (k == 0) d_beta_out[pos] = s_mag[0];
    __syncthreads();

    if (k == 0) { s_mag[0] = 0.0f; }
    __syncthreads();
    atomicAdd(&s_mag[0], d_chi_k);
    __syncthreads();
    if (k == 0) d_chi_out[pos] = s_mag[0];
    __syncthreads();

    // RK4 weights: 4 reductions
    float d_ws[4] = {d_rk4w_0, d_rk4w_1, d_rk4w_2, d_rk4w_3};
    for (int wi = 0; wi < 4; wi++) {
        if (k == 0) { s_mag[0] = 0.0f; }
        __syncthreads();
        atomicAdd(&s_mag[0], d_ws[wi]);
        __syncthreads();
        if (k == 0) d_rk4w_out[pos * 4 + wi] = s_mag[0];
        __syncthreads();
    }
}
"#;

    /// Compiled PTX — forward kernel (cached).
    static COMPILED_FWD_PTX: OnceLock<String> = OnceLock::new();
    /// Compiled PTX — backward kernel (cached, compiled on first backward call).
    static COMPILED_BWD_PTX: OnceLock<String> = OnceLock::new();

    fn get_fwd_ptx() -> &'static str {
        COMPILED_FWD_PTX.get_or_init(|| {
            let src = format!("{}{}", CUDA_FWM_DEVICE_FN, CUDA_ODE_FWD_KERNEL);
            let ptx = cudarc::nvrtc::compile_ptx(&src)
                .expect("CUDA ODE forward kernel compilation failed");
            ptx.to_src()
        })
    }

    fn get_bwd_ptx() -> &'static str {
        COMPILED_BWD_PTX.get_or_init(|| {
            let src = format!("{}{}{}", CUDA_FWM_BACKWARD_DEVICE_FN, CUDA_FWM_DEVICE_FN, CUDA_ODE_BWD_KERNEL);
            let ptx = cudarc::nvrtc::compile_ptx(&src)
                .expect("CUDA ODE backward kernel compilation failed");
            ptx.to_src()
        })
    }

    // Re-use types from custom_ode (same SharedParamGrads, same training loop)
    pub use crate::candle_tier::custom_ode::custom_ode::{
        OdeParamGradsAccum, SharedParamGrads,
    };

    /// GPU-resident forward cache — state cache stays on GPU for backward kernel.
    struct OdeCudaCacheGpu {
        /// State cache on GPU: [n_pos, n_steps, n_embd]
        state_cache: cudarc::driver::CudaSlice<f32>,
        weights: crate::model::KerrWeights,
        n_pos: usize,
        n_embd: usize,
    }

    /// Forward cache — CPU fallback path.
    struct OdeCudaCacheCpu {
        caches: Vec<crate::common::ode_backward::OdeForwardCache>,
        weights: crate::model::KerrWeights,
    }

    /// Cache enum — GPU path (Phase 2) or CPU fallback.
    enum OdeCudaCache {
        Gpu(OdeCudaCacheGpu),
        Cpu(OdeCudaCacheCpu),
    }

    /// CUDA-native CustomOp — fused AGC + RK4 in one kernel launch.
    pub struct KerrOdeCudaOp {
        gamma: Vec<f32>,         // [n_bands] pre-softplus'd
        gamma_raw: Vec<f32>,     // [n_bands] raw (for backward chain rule)
        omega: Vec<f32>,         // [n_bands]
        alpha: f32,
        beta: f32,
        chi: f32,                // four-wave mixing strength (0.0 = off)
        rk4_weights: [f32; 4],
        rk4_steps: usize,
        n_bands: usize,
        layer_idx: usize,
        agc_ceiling: f32,
        cache: Arc<Mutex<Option<OdeCudaCache>>>,
        param_grads: SharedParamGrads,
    }

    impl KerrOdeCudaOp {
        pub fn new(
            gamma_raw: Vec<f32>, omega: Vec<f32>,
            alpha: f32, beta: f32, chi: f32, rk4_weights: [f32; 4],
            rk4_steps: usize, n_bands: usize, layer_idx: usize,
            agc_ceiling: f32, param_grads: SharedParamGrads,
        ) -> Self {
            let gamma = gamma_raw.iter().map(|&g| crate::common::math::softplus(g)).collect();
            Self {
                gamma, gamma_raw, omega, alpha, beta, chi, rk4_weights,
                rk4_steps, n_bands, layer_idx, agc_ceiling,
                cache: Arc::new(Mutex::new(None)),
                param_grads,
            }
        }

        fn make_weights(&self) -> crate::model::KerrWeights {
            crate::model::KerrWeights {
                gamma_raw: self.gamma_raw.clone(),
                omega: self.omega.clone(),
                alpha: self.alpha,
                beta: self.beta,
                rk4_n_steps: self.rk4_steps,
                phase_correction: vec![0.0; self.n_bands],
                rk4_weights: self.rk4_weights,
                chi: self.chi,
            }
        }
    }

    impl CustomOp1 for KerrOdeCudaOp {
        fn name(&self) -> &'static str { "kerr_ode_cuda" }

        /// CPU fallback — same as custom_ode.rs
        fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let input = match storage {
                CpuStorage::F32(data) => data,
                _ => return Err(Error::Msg("KerrOdeCudaOp expects F32".to_string())),
            };
            let dims = layout.dims();
            let (n_pos, n_embd) = (dims[0], dims[1]);
            let weights = self.make_weights();

            let mut outputs = vec![0.0f32; n_pos * n_embd];
            let mut caches = Vec::with_capacity(n_pos);
            for pos in 0..n_pos {
                let start = layout.start_offset() + pos * n_embd;
                let x = &input[start..start + n_embd];
                let (out, cache) = crate::common::ode_backward::ode_forward_with_cache(x, &weights);
                outputs[pos * n_embd..(pos + 1) * n_embd].copy_from_slice(&out);
                caches.push(cache);
            }
            *self.cache.lock().unwrap() = Some(OdeCudaCache::Cpu(OdeCudaCacheCpu {
                caches, weights: self.make_weights(),
            }));
            Ok((CpuStorage::F32(outputs), Shape::from_dims(&[n_pos, n_embd])))
        }

        /// CUDA forward — single kernel launch, state cache stays on GPU for backward.
        #[cfg(feature = "candle-backend")]
        fn cuda_fwd(&self, storage: &candle_core::CudaStorage, layout: &Layout)
            -> Result<(candle_core::CudaStorage, Shape)>
        {
            let dev = &storage.device;
            let dims = layout.dims();
            let (n_pos, n_embd) = (dims[0], dims[1]);

            // Compile forward PTX on first call
            let ptx_str = get_fwd_ptx();
            let func = dev.get_or_load_custom_func("kerr_ode_fwd", "kerr_ode_fwd_mod", ptx_str)?;

            // Upload constants to GPU
            let d_gamma = dev.clone_htod(&self.gamma)?;
            let d_omega = dev.clone_htod(&self.omega)?;
            let d_rk4_w = dev.clone_htod(&self.rk4_weights)?;

            // Allocate output + state cache on GPU
            let d_output = dev.alloc_zeros::<f32>(n_pos * n_embd)?;
            let cache_elems = n_pos * self.rk4_steps * n_embd;
            let d_cache = dev.alloc_zeros::<f32>(cache_elems)?;

            let input_slice = f32::as_cuda_slice(storage)?;

            // Launch forward kernel
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_pos as u32, 1, 1),
                block_dim: (self.n_bands as u32, 1, 1),
                shared_mem_bytes: (3 * self.n_bands * std::mem::size_of::<f32>()) as u32,
            };
            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(&d_output);
            builder.arg(&d_cache);
            builder.arg(&d_gamma);
            builder.arg(&d_omega);
            builder.arg(&d_rk4_w);
            builder.arg(&self.agc_ceiling);
            builder.arg(&self.alpha);
            builder.arg(&self.beta);
            builder.arg(&self.chi);
            let n_bands_i32 = self.n_bands as i32;
            let n_steps_i32 = self.rk4_steps as i32;
            builder.arg(&n_bands_i32);
            builder.arg(&n_steps_i32);
            unsafe {
                builder.launch(cfg)
                    .map_err(|e| Error::Msg(format!("CUDA fwd kernel launch: {e}")))?;
            }

            // State cache stays on GPU — no CPU forward re-run needed
            *self.cache.lock().unwrap() = Some(OdeCudaCache::Gpu(OdeCudaCacheGpu {
                state_cache: d_cache,
                weights: self.make_weights(),
                n_pos,
                n_embd,
            }));

            let out_storage = <f32 as CudaDType>::wrap_cuda_slice(d_output, dev.clone());
            Ok((out_storage, Shape::from_dims(&[n_pos, n_embd])))
        }

        fn bwd(&self, _arg: &Tensor, _node: &Tensor, output_grad: &Tensor) -> Result<Option<Tensor>> {
            let dims = output_grad.dims();
            let (n_pos, n_embd) = (dims[0], dims[1]);

            let cache_lock = self.cache.lock().unwrap();
            let ode_cache = cache_lock.as_ref()
                .ok_or_else(|| Error::Msg("ODE backward called without forward cache".to_string()))?;

            match ode_cache {
                OdeCudaCache::Gpu(gpu_cache) => {
                    self.bwd_gpu(output_grad, gpu_cache, n_pos, n_embd)
                }
                OdeCudaCache::Cpu(cpu_cache) => {
                    self.bwd_cpu(output_grad, cpu_cache, n_pos, n_embd)
                }
            }
        }
    }

    impl KerrOdeCudaOp {
        /// GPU backward — launches backward kernel, reduces param grads on CPU.
        fn bwd_gpu(&self, output_grad: &Tensor, cache: &OdeCudaCacheGpu,
                    n_pos: usize, n_embd: usize) -> Result<Option<Tensor>>
        {
            // Get the CUDA device from the output_grad tensor
            let dev = match output_grad.device() {
                candle_core::Device::Cuda(dev) => dev.clone(),
                _ => return Err(Error::Msg("GPU backward requires CUDA device".to_string())),
            };

            // Compile backward PTX
            let bwd_ptx = get_bwd_ptx();
            let func = dev.get_or_load_custom_func("kerr_ode_bwd", "kerr_ode_bwd_mod", bwd_ptx)?;

            // Get d_output on GPU
            let (d_output_storage, _d_output_layout) = output_grad.storage_and_layout();
            let d_output_cuda = match &*d_output_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => return Err(Error::Msg("GPU backward: output_grad not on CUDA".to_string())),
            };
            let d_output_slice = f32::as_cuda_slice(d_output_cuda)?;

            // Upload constants
            let d_gamma = dev.clone_htod(&self.gamma)?;
            let d_omega = dev.clone_htod(&self.omega)?;
            let d_gamma_raw = dev.clone_htod(&self.gamma_raw)?;
            let d_rk4_w = dev.clone_htod(&self.rk4_weights)?;

            // Allocate output buffers
            let d_input = dev.alloc_zeros::<f32>(n_pos * n_embd)?;
            let d_gamma_out = dev.alloc_zeros::<f32>(n_pos * self.n_bands)?;
            let d_alpha_out = dev.alloc_zeros::<f32>(n_pos)?;
            let d_beta_out = dev.alloc_zeros::<f32>(n_pos)?;
            let d_chi_out = dev.alloc_zeros::<f32>(n_pos)?;
            let d_rk4w_out = dev.alloc_zeros::<f32>(n_pos * 4)?;

            // Launch backward kernel
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_pos as u32, 1, 1),
                block_dim: (self.n_bands as u32, 1, 1),
                shared_mem_bytes: (5 * self.n_bands * std::mem::size_of::<f32>()) as u32,
            };
            let mut builder = func.builder();
            builder.arg(d_output_slice);
            builder.arg(&cache.state_cache);
            builder.arg(&d_input);
            builder.arg(&d_gamma_out);
            builder.arg(&d_alpha_out);
            builder.arg(&d_beta_out);
            builder.arg(&d_chi_out);
            builder.arg(&d_rk4w_out);
            builder.arg(&d_gamma);
            builder.arg(&d_omega);
            builder.arg(&d_gamma_raw);
            builder.arg(&d_rk4_w);
            builder.arg(&self.alpha);
            builder.arg(&self.beta);
            builder.arg(&self.chi);
            let n_bands_i32 = self.n_bands as i32;
            let n_steps_i32 = self.rk4_steps as i32;
            builder.arg(&n_bands_i32);
            builder.arg(&n_steps_i32);
            unsafe {
                builder.launch(cfg)
                    .map_err(|e| Error::Msg(format!("CUDA bwd kernel launch: {e}")))?;
            }

            // Copy param grads to CPU and reduce across positions
            let gamma_grads: Vec<f32> = dev.clone_dtoh(&d_gamma_out)?;
            let alpha_grads: Vec<f32> = dev.clone_dtoh(&d_alpha_out)?;
            let beta_grads: Vec<f32> = dev.clone_dtoh(&d_beta_out)?;
            let chi_grads: Vec<f32> = dev.clone_dtoh(&d_chi_out)?;
            let rk4w_grads: Vec<f32> = dev.clone_dtoh(&d_rk4w_out)?;

            // Reduce: sum across positions
            let mut total_d_gamma_raw = vec![0.0f32; self.n_bands];
            for pos in 0..n_pos {
                for k in 0..self.n_bands {
                    total_d_gamma_raw[k] += gamma_grads[pos * self.n_bands + k];
                }
            }
            let total_d_alpha: f32 = alpha_grads.iter().sum();
            let total_d_beta: f32 = beta_grads.iter().sum();
            let total_d_chi: f32 = chi_grads.iter().sum();
            let mut total_d_rk4_weights = [0.0f32; 4];
            for pos in 0..n_pos {
                for w in 0..4 {
                    total_d_rk4_weights[w] += rk4w_grads[pos * 4 + w];
                }
            }

            // Store param grads
            {
                let mut v = self.param_grads.lock().unwrap();
                if self.layer_idx < v.len() {
                    v[self.layer_idx] = Some(OdeParamGradsAccum {
                        d_gamma_raw: total_d_gamma_raw,
                        d_alpha: total_d_alpha,
                        d_beta: total_d_beta,
                        d_chi: total_d_chi,
                        d_rk4_weights: total_d_rk4_weights,
                    });
                }
            }

            // Copy d_input to CPU, create Tensor, move to device
            // (small cost: n_pos × n_embd floats — dwarfed by kernel savings)
            let d_input_cpu: Vec<f32> = dev.clone_dtoh(&d_input)?;
            let d_input_tensor = Tensor::from_vec(
                d_input_cpu, output_grad.shape(), output_grad.device(),
            )?;
            Ok(Some(d_input_tensor))
        }

        /// CPU backward fallback — same as before.
        fn bwd_cpu(&self, output_grad: &Tensor, cache: &OdeCudaCacheCpu,
                    n_pos: usize, n_embd: usize) -> Result<Option<Tensor>>
        {
            let d_output_flat = output_grad.flatten_all()?.to_vec1::<f32>()?;

            let mut d_inputs = vec![0.0f32; n_pos * n_embd];
            let mut total_d_gamma_raw = vec![0.0f32; self.n_bands];
            let mut total_d_alpha = 0.0f32;
            let mut total_d_beta = 0.0f32;
            let mut total_d_chi = 0.0f32;
            let mut total_d_rk4_weights = [0.0f32; 4];

            for pos in 0..n_pos {
                let d_out = &d_output_flat[pos * n_embd..(pos + 1) * n_embd];
                let (d_input, pg) = crate::common::ode_backward::ode_backward(
                    d_out, &cache.caches[pos], &cache.weights,
                );
                d_inputs[pos * n_embd..(pos + 1) * n_embd].copy_from_slice(&d_input);
                for k in 0..self.n_bands { total_d_gamma_raw[k] += pg.d_gamma_raw[k]; }
                total_d_alpha += pg.d_alpha;
                total_d_beta += pg.d_beta;
                total_d_chi += pg.d_chi;
                for w in 0..4 { total_d_rk4_weights[w] += pg.d_rk4_weights[w]; }
            }

            {
                let mut v = self.param_grads.lock().unwrap();
                if self.layer_idx < v.len() {
                    v[self.layer_idx] = Some(OdeParamGradsAccum {
                        d_gamma_raw: total_d_gamma_raw,
                        d_alpha: total_d_alpha,
                        d_beta: total_d_beta,
                        d_chi: total_d_chi,
                        d_rk4_weights: total_d_rk4_weights,
                    });
                }
            }

            let d_input_tensor = Tensor::from_vec(d_inputs, output_grad.shape(), output_grad.device())?;
            Ok(Some(d_input_tensor))
        }
    }
}
