// Batched Kerr-ODE derivative: compute dr/ds for ALL positions in one dispatch.
//
// One thread per (pos, band). Shared parameters (gamma, omega, alpha, beta)
// are read once by all positions. Replaces N_POS separate kerr_step dispatches.
//
// Data layout: r_in/s_in/dr_out/ds_out are [n_pos * n_bands], contiguous per position.

struct Params {
    n_bands: u32,
    n_pos: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read_write> dr_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(3) var<storage, read_write> ds_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(4) var<storage, read> gamma: array<f32>;      // [n_bands]
@group(0) @binding(5) var<storage, read> omega: array<f32>;      // [n_bands]
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> alpha_beta_chi: array<f32>; // [3]: alpha, beta, chi

@compute @workgroup_size(64)
fn kerr_derivative_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat_id = id.x;
    let n = params.n_bands;
    let n_pos = params.n_pos;

    let pos = flat_id / n;
    let band = flat_id % n;

    if (pos >= n_pos) {
        return;
    }

    let base = pos * n;
    let r = r_in[base + band];
    let s = s_in[base + band];
    let mag_sq = r * r + s * s;

    // Conv1d with kernel [1, 1, 0, 1, 1] and padding=2
    var ns: f32 = 0.0;

    if (band >= 2u) {
        let idx = base + band - 2u;
        let r2 = r_in[idx]; let s2 = s_in[idx];
        ns += r2 * r2 + s2 * s2;
    }
    if (band >= 1u) {
        let idx = base + band - 1u;
        let r1 = r_in[idx]; let s1 = s_in[idx];
        ns += r1 * r1 + s1 * s1;
    }
    if (band + 1u < n) {
        let idx = base + band + 1u;
        let rp1 = r_in[idx]; let sp1 = s_in[idx];
        ns += rp1 * rp1 + sp1 * sp1;
    }
    if (band + 2u < n) {
        let idx = base + band + 2u;
        let rp2 = r_in[idx]; let sp2 = s_in[idx];
        ns += rp2 * rp2 + sp2 * sp2;
    }

    let alpha = alpha_beta_chi[0];
    let beta = alpha_beta_chi[1];
    let chi = alpha_beta_chi[2];

    // Clamp magnitude terms to prevent phi overflow at 768-dim+.
    // GPU FP differences can cause |Z| to drift, making mag_sq/ns explode.
    // Clamp at 2500 (50² — matches RK4 magnitude bound of 50.0).
    let mag_sq_c = min(mag_sq, 2500.0);
    let ns_c = min(ns, 10000.0);

    let phi = omega[band] + alpha * mag_sq_c + beta * ns_c;
    let g = gamma[band];

    var dr_val: f32 = -g * r - phi * s;
    var ds_val: f32 = -g * s + phi * r;

    // Four-wave mixing: Hamiltonian energy-conserving cubic coupling.
    // Each band participates in up to 8 quartets (4 roles × 2 families).
    // We enumerate only the quartets that include this band.
    if (chi != 0.0 && n > 4u) {
        // Helper: compute one quartet's contribution to this band's derivative.
        // Given quartet (a,b,c,d) and this band's role, accumulate into dr_val/ds_val.

        // Family A: quartet = (k-2, k+1, k-1, k) for k in [2, n-1)
        // Role: band == k-2 → k = band+2 (valid if band+2 in [2,n-1) → band in [0,n-3))
        if (band + 2u >= 2u && band + 2u < n - 1u) {
            let k = band + 2u;
            let a = k - 2u; let b = k + 1u; let c = k - 1u; let d = k;
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;
            dr_val += chi * (rb*pcd_im - sb*pcd_re);
            ds_val -= chi * (rb*pcd_re + sb*pcd_im);
        }
        // Role: band == k+1 → k = band-1 (valid if band-1 in [2,n-1) → band in [3,n))
        if (band >= 3u && band < n) {
            let k = band - 1u;
            let a = k - 2u; let b = k + 1u; let c = k - 1u; let d = k;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;
            dr_val += chi * (ra*pcd_im - sa*pcd_re);
            ds_val -= chi * (ra*pcd_re + sa*pcd_im);
        }
        // Role: band == k-1 → k = band+1 (valid if band+1 in [2,n-1) → band in [1,n-2))
        if (band >= 1u && band + 1u < n - 1u) {
            let k = band + 1u;
            let a = k - 2u; let b = k + 1u; let c = k - 1u; let d = k;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
            dr_val += chi * (pab_im*rd - pab_re*sd);
            ds_val -= chi * (pab_re*rd + pab_im*sd);
        }
        // Role: band == k → k = band (valid if band in [2,n-1))
        if (band >= 2u && band < n - 1u) {
            let k = band;
            let a = k - 2u; let b = k + 1u; let c = k - 1u; let d = k;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
            dr_val += chi * (pab_im*rc - pab_re*sc);
            ds_val -= chi * (pab_re*rc + pab_im*sc);
        }

        // Family B: quartet = (k-1, k+2, k, k+1) for k in [1, n-2)
        // Role: band == k-1 → k = band+1 (valid if band+1 in [1,n-2) → band in [0,n-3))
        if (band + 1u >= 1u && band + 1u < n - 2u) {
            let k = band + 1u;
            let a = k - 1u; let b = k + 2u; let c = k; let d = k + 1u;
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;
            dr_val += chi * (rb*pcd_im - sb*pcd_re);
            ds_val -= chi * (rb*pcd_re + sb*pcd_im);
        }
        // Role: band == k+2 → k = band-2 (valid if band-2 in [1,n-2) → band in [3,n))
        if (band >= 3u && band < n) {
            let k = band - 2u;
            let a = k - 1u; let b = k + 2u; let c = k; let d = k + 1u;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;
            dr_val += chi * (ra*pcd_im - sa*pcd_re);
            ds_val -= chi * (ra*pcd_re + sa*pcd_im);
        }
        // Role: band == k → k = band (valid if band in [1,n-2))
        if (band >= 1u && band < n - 2u) {
            let k = band;
            let a = k - 1u; let b = k + 2u; let c = k; let d = k + 1u;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rd = r_in[base + d]; let sd = s_in[base + d];
            let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
            dr_val += chi * (pab_im*rd - pab_re*sd);
            ds_val -= chi * (pab_re*rd + pab_im*sd);
        }
        // Role: band == k+1 → k = band-1 (valid if band-1 in [1,n-2) → band in [2,n-1))
        if (band >= 2u && band < n - 1u) {
            let k = band - 1u;
            let a = k - 1u; let b = k + 2u; let c = k; let d = k + 1u;
            let ra = r_in[base + a]; let sa = s_in[base + a];
            let rb = r_in[base + b]; let sb = s_in[base + b];
            let rc = r_in[base + c]; let sc = s_in[base + c];
            let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
            dr_val += chi * (pab_im*rc - pab_re*sc);
            ds_val -= chi * (pab_re*rc + pab_im*sc);
        }
    }

    dr_out[base + band] = dr_val;
    ds_out[base + band] = ds_val;
}
