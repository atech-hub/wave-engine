// Batched Kerr-ODE derivative backward: compute input + parameter gradients
// for ALL positions in one dispatch.
//
// One thread per (pos, band). Outputs per (pos, band):
//   d_r, d_s           — input gradients
//   d_gamma, d_omega   — per-band parameter gradients
//   d_alpha_partial     — d_phi[k] * mag_sq[k] (reduce on CPU for d_alpha)
//   d_beta_partial      — d_phi[k] * ns[k]     (reduce on CPU for d_beta)
//
// The transpose convolution (d_ns → d_mag_sq) is computed per-thread using
// the same neighbor pattern as the forward [1,1,0,1,1] kernel.

struct Params {
    n_bands: u32,
    n_pos: u32,
    alpha: f32,
    beta: f32,
    chi: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Forward cache inputs (saved during forward derivative eval)
@group(0) @binding(0) var<storage, read> r: array<f32>;         // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s: array<f32>;         // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;     // [n_bands]
@group(0) @binding(3) var<storage, read> omega: array<f32>;     // [n_bands]

// Upstream gradients
@group(0) @binding(4) var<storage, read> d_dr: array<f32>;      // [n_pos * n_bands]
@group(0) @binding(5) var<storage, read> d_ds: array<f32>;      // [n_pos * n_bands]

// Outputs
@group(0) @binding(6) var<storage, read_write> d_r_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(7) var<storage, read_write> d_s_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(8) var<storage, read_write> d_gamma_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(9) var<storage, read_write> d_omega_out: array<f32>;  // [n_pos * n_bands]
@group(0) @binding(10) var<storage, read_write> d_alpha_partial: array<f32>; // [n_pos * n_bands]
@group(0) @binding(11) var<storage, read_write> d_beta_partial: array<f32>;  // [n_pos * n_bands]

@group(0) @binding(12) var<uniform> params: Params;
@group(0) @binding(13) var<storage, read_write> d_chi_partial: array<f32>;  // [n_pos * n_bands]

@compute @workgroup_size(64)
fn kerr_backward_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let n = params.n_bands;
    let n_pos = params.n_pos;

    let pos = flat_id / n;
    let band = flat_id % n;

    if (pos >= n_pos) {
        return;
    }

    let base = pos * n;
    let idx = base + band;

    // Read cached forward state
    let r_k = r[idx];
    let s_k = s[idx];
    let g_k = gamma[band];

    // Recompute forward intermediates for this band
    let mag_sq_k = r_k * r_k + s_k * s_k;

    // Neighbour sum (conv1d [1,1,0,1,1])
    var ns_k: f32 = 0.0;
    if (band >= 2u) {
        let i = base + band - 2u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band >= 1u) {
        let i = base + band - 1u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band + 1u < n) {
        let i = base + band + 1u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }
    if (band + 2u < n) {
        let i = base + band + 2u;
        ns_k += r[i] * r[i] + s[i] * s[i];
    }

    let phi_k = omega[band] + params.alpha * mag_sq_k + params.beta * ns_k;

    // Read upstream gradients
    let ddr_k = d_dr[idx];
    let dds_k = d_ds[idx];

    // dr[k] = -gamma[k]*r[k] - phi[k]*s[k]
    // ds[k] = -gamma[k]*s[k] + phi[k]*r[k]

    // d_gamma[k] = d_dr[k] * (-r[k]) + d_ds[k] * (-s[k])
    let dg_k = ddr_k * (-r_k) + dds_k * (-s_k);

    // d_phi[k] = d_dr[k] * (-s[k]) + d_ds[k] * r[k]
    let dphi_k = ddr_k * (-s_k) + dds_k * r_k;

    // d_omega[k] = d_phi[k]
    let dom_k = dphi_k;

    // d_alpha partial = d_phi[k] * mag_sq[k]
    let da_k = dphi_k * mag_sq_k;

    // d_beta partial = d_phi[k] * ns[k]
    let db_k = dphi_k * ns_k;

    // d_mag_sq[k] from phi: d_phi[k] * alpha
    var d_mag_sq_k: f32 = dphi_k * params.alpha;

    // d_mag_sq from ns: transpose convolution
    // ns[j] uses mag_sq[k] when k == j-2, j-1, j+1, or j+2
    // So d_mag_sq[k] += d_ns[k-2] + d_ns[k-1] + d_ns[k+1] + d_ns[k+2]
    // where d_ns[j] = d_phi[j] * beta
    // We need d_phi for neighbors — recompute from d_dr/d_ds
    if (band >= 2u) {
        let j = band - 2u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band >= 1u) {
        let j = band - 1u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band + 1u < n) {
        let j = band + 1u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }
    if (band + 2u < n) {
        let j = band + 2u;
        let ji = base + j;
        let dphi_j = d_dr[ji] * (-s[ji]) + d_ds[ji] * r[ji];
        d_mag_sq_k += dphi_j * params.beta;
    }

    // d_r[k] = d_dr[k] * (-gamma[k]) + d_ds[k] * phi[k] + d_mag_sq[k] * 2*r[k]
    var dr_k = ddr_k * (-g_k) + dds_k * phi_k + d_mag_sq_k * 2.0 * r_k;

    // d_s[k] = d_dr[k] * (-phi[k]) + d_ds[k] * (-gamma[k]) + d_mag_sq[k] * 2*s[k]
    var ds_k = ddr_k * (-phi_k) + dds_k * (-g_k) + d_mag_sq_k * 2.0 * s_k;

    // ── FWM backward ──────────────────────────────────────────────────────
    // Per-thread quartet enumeration: each band gathers gradient contributions
    // from all quartets it participates in. Matches the forward FWM pattern
    // in kerr_step_batch.wgsl. CPU reference: fwm_quartet_backward in ode_backward.rs.
    var fwm_dchi: f32 = 0.0;

    if (params.chi != 0.0 && n > 4u) {
        let chi = params.chi;

        // ── Family A: quartet (k-2, k+1, k-1, k) for k in [2, n-1) ──

        // Role a (band == k-2, k = band+2): valid when band+2 in [2, n-1)
        if (band + 2u >= 2u && band + 2u < n - 1u) {
            let bb = base + band + 3u;
            let cc = base + band + 1u;
            let dd = base + band + 2u;
            let rb_v = r[bb]; let sb_v = s[bb];
            let rc_v = r[cc]; let sc_v = s[cc];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_cd_re = rc_v*rd_v - sc_v*sd_v;
            let p_cd_im = rc_v*sd_v + sc_v*rd_v;

            dr_k += ddrb_v * chi * p_cd_im;
            dr_k -= ddsb_v * chi * p_cd_re;
            dr_k += ddrc_v * chi * (sb_v*rd_v - rb_v*sd_v);
            dr_k -= ddsc_v * chi * (rb_v*rd_v + sb_v*sd_v);
            dr_k += ddrd_v * chi * (sb_v*rc_v - rb_v*sc_v);
            dr_k -= ddsd_v * chi * (rb_v*rc_v + sb_v*sc_v);

            ds_k -= ddrb_v * chi * p_cd_re;
            ds_k -= ddsb_v * chi * p_cd_im;
            ds_k += ddrc_v * chi * (rb_v*rd_v + sb_v*sd_v);
            ds_k += ddsc_v * chi * (sb_v*rd_v - rb_v*sd_v);
            ds_k += ddrd_v * chi * (rb_v*rc_v + sb_v*sc_v);
            ds_k += ddsd_v * chi * (sb_v*rc_v - rb_v*sc_v);

            // d_chi: role a owns the full quartet contribution (roles b/c/d skip)
            let ra_v = r_k; let sa_v = s_k;
            let ddra_v = ddr_k; let ddsa_v = dds_k;
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;
            fwm_dchi += ddra_v * (rb_v*p_cd_im - sb_v*p_cd_re)
                      - ddsa_v * (rb_v*p_cd_re + sb_v*p_cd_im)
                      + ddrb_v * (ra_v*p_cd_im - sa_v*p_cd_re)
                      - ddsb_v * (ra_v*p_cd_re + sa_v*p_cd_im)
                      + ddrc_v * (p_ab_im*rd_v - p_ab_re*sd_v)
                      - ddsc_v * (p_ab_re*rd_v + p_ab_im*sd_v)
                      + ddrd_v * (p_ab_im*rc_v - p_ab_re*sc_v)
                      - ddsd_v * (p_ab_re*rc_v + p_ab_im*sc_v);
        }

        // Role b (band == k+1, k = band-1): valid when band-1 in [2, n-1)
        if (band >= 3u && band - 1u < n - 1u) {
            let aa = base + band - 3u;
            let cc = base + band - 2u;
            let dd = base + band - 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rc_v = r[cc]; let sc_v = s[cc];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_cd_re = rc_v*rd_v - sc_v*sd_v;
            let p_cd_im = rc_v*sd_v + sc_v*rd_v;

            dr_k += ddra_v * chi * p_cd_im;
            dr_k -= ddsa_v * chi * p_cd_re;
            dr_k += ddrc_v * chi * (sa_v*rd_v - ra_v*sd_v);
            dr_k -= ddsc_v * chi * (ra_v*rd_v + sa_v*sd_v);
            dr_k += ddrd_v * chi * (sa_v*rc_v - ra_v*sc_v);
            dr_k -= ddsd_v * chi * (ra_v*rc_v + sa_v*sc_v);

            ds_k -= ddra_v * chi * p_cd_re;
            ds_k -= ddsa_v * chi * p_cd_im;
            ds_k += ddrc_v * chi * (ra_v*rd_v + sa_v*sd_v);
            ds_k += ddsc_v * chi * (sa_v*rd_v - ra_v*sd_v);
            ds_k += ddrd_v * chi * (ra_v*rc_v + sa_v*sc_v);
            ds_k += ddsd_v * chi * (sa_v*rc_v - ra_v*sc_v);
        }

        // Role c (band == k-1, k = band+1): valid when band+1 in [2, n-1)
        if (band + 1u >= 2u && band + 1u < n - 1u) {
            let aa = base + band - 1u;
            let bb = base + band + 2u;
            let dd = base + band + 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rb_v = r[bb]; let sb_v = s[bb];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;

            dr_k += ddra_v * chi * (rb_v*sd_v - sb_v*rd_v);
            dr_k -= ddsa_v * chi * (rb_v*rd_v + sb_v*sd_v);
            dr_k += ddrb_v * chi * (ra_v*sd_v - sa_v*rd_v);
            dr_k -= ddsb_v * chi * (ra_v*rd_v + sa_v*sd_v);
            dr_k += ddrd_v * chi * p_ab_im;
            dr_k -= ddsd_v * chi * p_ab_re;

            ds_k += ddra_v * chi * (rb_v*rd_v + sb_v*sd_v);
            ds_k += ddsa_v * chi * (rb_v*sd_v - sb_v*rd_v);
            ds_k += ddrb_v * chi * (ra_v*rd_v + sa_v*sd_v);
            ds_k += ddsb_v * chi * (ra_v*sd_v - sa_v*rd_v);
            ds_k -= ddrd_v * chi * p_ab_re;
            ds_k -= ddsd_v * chi * p_ab_im;
        }

        // Role d (band == k, k = band): valid when band in [2, n-1)
        if (band >= 2u && band < n - 1u) {
            let aa = base + band - 2u;
            let bb = base + band + 1u;
            let cc = base + band - 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rb_v = r[bb]; let sb_v = s[bb];
            let rc_v = r[cc]; let sc_v = s[cc];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;

            dr_k += ddra_v * chi * (rb_v*sc_v - sb_v*rc_v);
            dr_k -= ddsa_v * chi * (rb_v*rc_v + sb_v*sc_v);
            dr_k += ddrb_v * chi * (ra_v*sc_v - sa_v*rc_v);
            dr_k -= ddsb_v * chi * (ra_v*rc_v + sa_v*sc_v);
            dr_k += ddrc_v * chi * p_ab_im;
            dr_k -= ddsc_v * chi * p_ab_re;

            ds_k += ddra_v * chi * (rb_v*rc_v + sb_v*sc_v);
            ds_k += ddsa_v * chi * (rb_v*sc_v - sb_v*rc_v);
            ds_k += ddrb_v * chi * (ra_v*rc_v + sa_v*sc_v);
            ds_k += ddsb_v * chi * (ra_v*sc_v - sa_v*rc_v);
            ds_k -= ddrc_v * chi * p_ab_re;
            ds_k -= ddsc_v * chi * p_ab_im;
        }

        // ── Family B: quartet (k-1, k+2, k, k+1) for k in [1, n-2) ──

        // Role a (band == k-1, k = band+1): valid when band+1 in [1, n-2)
        if (band + 1u >= 1u && band + 1u < n - 2u) {
            let bb = base + band + 3u;
            let cc = base + band + 1u;
            let dd = base + band + 2u;
            let rb_v = r[bb]; let sb_v = s[bb];
            let rc_v = r[cc]; let sc_v = s[cc];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_cd_re = rc_v*rd_v - sc_v*sd_v;
            let p_cd_im = rc_v*sd_v + sc_v*rd_v;

            dr_k += ddrb_v * chi * p_cd_im;
            dr_k -= ddsb_v * chi * p_cd_re;
            dr_k += ddrc_v * chi * (sb_v*rd_v - rb_v*sd_v);
            dr_k -= ddsc_v * chi * (rb_v*rd_v + sb_v*sd_v);
            dr_k += ddrd_v * chi * (sb_v*rc_v - rb_v*sc_v);
            dr_k -= ddsd_v * chi * (rb_v*rc_v + sb_v*sc_v);

            ds_k -= ddrb_v * chi * p_cd_re;
            ds_k -= ddsb_v * chi * p_cd_im;
            ds_k += ddrc_v * chi * (rb_v*rd_v + sb_v*sd_v);
            ds_k += ddsc_v * chi * (sb_v*rd_v - rb_v*sd_v);
            ds_k += ddrd_v * chi * (rb_v*rc_v + sb_v*sc_v);
            ds_k += ddsd_v * chi * (sb_v*rc_v - rb_v*sc_v);

            // d_chi: role a owns the full quartet contribution
            let ra_v = r_k; let sa_v = s_k;
            let ddra_v = ddr_k; let ddsa_v = dds_k;
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;
            fwm_dchi += ddra_v * (rb_v*p_cd_im - sb_v*p_cd_re)
                      - ddsa_v * (rb_v*p_cd_re + sb_v*p_cd_im)
                      + ddrb_v * (ra_v*p_cd_im - sa_v*p_cd_re)
                      - ddsb_v * (ra_v*p_cd_re + sa_v*p_cd_im)
                      + ddrc_v * (p_ab_im*rd_v - p_ab_re*sd_v)
                      - ddsc_v * (p_ab_re*rd_v + p_ab_im*sd_v)
                      + ddrd_v * (p_ab_im*rc_v - p_ab_re*sc_v)
                      - ddsd_v * (p_ab_re*rc_v + p_ab_im*sc_v);
        }

        // Role b (band == k+2, k = band-2): valid when band-2 in [1, n-2)
        if (band >= 3u && band - 2u < n - 2u) {
            let aa = base + band - 3u;
            let cc = base + band - 2u;
            let dd = base + band - 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rc_v = r[cc]; let sc_v = s[cc];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_cd_re = rc_v*rd_v - sc_v*sd_v;
            let p_cd_im = rc_v*sd_v + sc_v*rd_v;

            dr_k += ddra_v * chi * p_cd_im;
            dr_k -= ddsa_v * chi * p_cd_re;
            dr_k += ddrc_v * chi * (sa_v*rd_v - ra_v*sd_v);
            dr_k -= ddsc_v * chi * (ra_v*rd_v + sa_v*sd_v);
            dr_k += ddrd_v * chi * (sa_v*rc_v - ra_v*sc_v);
            dr_k -= ddsd_v * chi * (ra_v*rc_v + sa_v*sc_v);

            ds_k -= ddra_v * chi * p_cd_re;
            ds_k -= ddsa_v * chi * p_cd_im;
            ds_k += ddrc_v * chi * (ra_v*rd_v + sa_v*sd_v);
            ds_k += ddsc_v * chi * (sa_v*rd_v - ra_v*sd_v);
            ds_k += ddrd_v * chi * (ra_v*rc_v + sa_v*sc_v);
            ds_k += ddsd_v * chi * (sa_v*rc_v - ra_v*sc_v);
        }

        // Role c (band == k, k = band): valid when band in [1, n-2)
        if (band >= 1u && band < n - 2u) {
            let aa = base + band - 1u;
            let bb = base + band + 2u;
            let dd = base + band + 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rb_v = r[bb]; let sb_v = s[bb];
            let rd_v = r[dd]; let sd_v = s[dd];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrd_v = d_dr[dd]; let ddsd_v = d_ds[dd];
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;

            dr_k += ddra_v * chi * (rb_v*sd_v - sb_v*rd_v);
            dr_k -= ddsa_v * chi * (rb_v*rd_v + sb_v*sd_v);
            dr_k += ddrb_v * chi * (ra_v*sd_v - sa_v*rd_v);
            dr_k -= ddsb_v * chi * (ra_v*rd_v + sa_v*sd_v);
            dr_k += ddrd_v * chi * p_ab_im;
            dr_k -= ddsd_v * chi * p_ab_re;

            ds_k += ddra_v * chi * (rb_v*rd_v + sb_v*sd_v);
            ds_k += ddsa_v * chi * (rb_v*sd_v - sb_v*rd_v);
            ds_k += ddrb_v * chi * (ra_v*rd_v + sa_v*sd_v);
            ds_k += ddsb_v * chi * (ra_v*sd_v - sa_v*rd_v);
            ds_k -= ddrd_v * chi * p_ab_re;
            ds_k -= ddsd_v * chi * p_ab_im;
        }

        // Role d (band == k+1, k = band-1): valid when band-1 in [1, n-2)
        if (band >= 2u && band - 1u < n - 2u) {
            let aa = base + band - 2u;
            let bb = base + band + 1u;
            let cc = base + band - 1u;
            let ra_v = r[aa]; let sa_v = s[aa];
            let rb_v = r[bb]; let sb_v = s[bb];
            let rc_v = r[cc]; let sc_v = s[cc];
            let ddra_v = d_dr[aa]; let ddsa_v = d_ds[aa];
            let ddrb_v = d_dr[bb]; let ddsb_v = d_ds[bb];
            let ddrc_v = d_dr[cc]; let ddsc_v = d_ds[cc];
            let p_ab_re = ra_v*rb_v - sa_v*sb_v;
            let p_ab_im = ra_v*sb_v + sa_v*rb_v;

            dr_k += ddra_v * chi * (rb_v*sc_v - sb_v*rc_v);
            dr_k -= ddsa_v * chi * (rb_v*rc_v + sb_v*sc_v);
            dr_k += ddrb_v * chi * (ra_v*sc_v - sa_v*rc_v);
            dr_k -= ddsb_v * chi * (ra_v*rc_v + sa_v*sc_v);
            dr_k += ddrc_v * chi * p_ab_im;
            dr_k -= ddsc_v * chi * p_ab_re;

            ds_k += ddra_v * chi * (rb_v*rc_v + sb_v*sc_v);
            ds_k += ddsa_v * chi * (rb_v*sc_v - sb_v*rc_v);
            ds_k += ddrb_v * chi * (ra_v*rc_v + sa_v*sc_v);
            ds_k += ddsb_v * chi * (ra_v*sc_v - sa_v*rc_v);
            ds_k -= ddrc_v * chi * p_ab_re;
            ds_k -= ddsc_v * chi * p_ab_im;
        }
    }

    // Write outputs
    d_r_out[idx] = dr_k;
    d_s_out[idx] = ds_k;
    d_gamma_out[idx] = dg_k;
    d_omega_out[idx] = dom_k;
    d_alpha_partial[idx] = da_k;
    d_beta_partial[idx] = db_k;
    d_chi_partial[idx] = fwm_dchi;
}
