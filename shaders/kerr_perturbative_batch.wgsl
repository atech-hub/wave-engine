// Perturbative Kerr-ODE: single-pass analytical approximation.
// Replaces 16-step RK4 (192 dispatches) with ONE dispatch.
//
// One thread per (pos, band). All bands computed independently.
// No scratch buffers needed — everything computed in-register.
//
// Lab-validated: MSE 0.000005 vs RK4-16 baseline.
// Trains BETTER than RK4-16 (loss 2.97 vs 3.07 at 100 iters).

struct Params {
    n_bands: u32,
    n_pos: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> r_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(1) var<storage, read> s_in: array<f32>;       // [n_pos * n_bands]
@group(0) @binding(2) var<storage, read_write> r_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(3) var<storage, read_write> s_out: array<f32>; // [n_pos * n_bands]
@group(0) @binding(4) var<storage, read> decay: array<f32>;      // [n_bands] = exp(-softplus(gamma))
@group(0) @binding(5) var<storage, read> cos_w: array<f32>;      // [n_bands] = cos(omega)
@group(0) @binding(6) var<storage, read> sin_w: array<f32>;      // [n_bands] = sin(omega)
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn kerr_perturbative_batch(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let r = r_in[idx];
    let s = s_in[idx];

    // Step 1: Linear solution — damping + base rotation
    let d = decay[band];
    let cw = cos_w[band];
    let sw = sin_w[band];
    let r_lin = d * (r * cw - s * sw);
    let s_lin = d * (r * sw + s * cw);

    // Step 2: Self-phase modulation
    let mag_sq = r_lin * r_lin + s_lin * s_lin;

    // Step 3: Cross-phase modulation — recompute neighbours in-thread
    // Each thread independently computes its neighbour's linear solution
    // (4 extra trig ops per thread, avoids second dispatch)
    var ns: f32 = 0.0;

    if (band >= 2u) {
        let ni = base + band - 2u;
        let d2 = decay[band - 2u];
        let c2 = cos_w[band - 2u];
        let s2 = sin_w[band - 2u];
        let rl = d2 * (r_in[ni] * c2 - s_in[ni] * s2);
        let sl = d2 * (r_in[ni] * s2 + s_in[ni] * c2);
        ns += rl * rl + sl * sl;
    }
    if (band >= 1u) {
        let ni = base + band - 1u;
        let d1 = decay[band - 1u];
        let c1 = cos_w[band - 1u];
        let s1 = sin_w[band - 1u];
        let rl = d1 * (r_in[ni] * c1 - s_in[ni] * s1);
        let sl = d1 * (r_in[ni] * s1 + s_in[ni] * c1);
        ns += rl * rl + sl * sl;
    }
    if (band + 1u < n) {
        let ni = base + band + 1u;
        let dp = decay[band + 1u];
        let cp = cos_w[band + 1u];
        let sp = sin_w[band + 1u];
        let rl = dp * (r_in[ni] * cp - s_in[ni] * sp);
        let sl = dp * (r_in[ni] * sp + s_in[ni] * cp);
        ns += rl * rl + sl * sl;
    }
    if (band + 2u < n) {
        let ni = base + band + 2u;
        let dp = decay[band + 2u];
        let cp = cos_w[band + 2u];
        let sp = sin_w[band + 2u];
        let rl = dp * (r_in[ni] * cp - s_in[ni] * sp);
        let sl = dp * (r_in[ni] * sp + s_in[ni] * cp);
        ns += rl * rl + sl * sl;
    }

    // Step 4: Phase perturbation
    let delta_phi = params.alpha * mag_sq + params.beta * ns;

    // Step 5: Apply correction
    r_out[idx] = r_lin - delta_phi * s_lin;
    s_out[idx] = s_lin + delta_phi * r_lin;
}
