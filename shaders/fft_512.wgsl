// 512-point radix-2 FFT in WGSL — Cooley-Tukey decimation-in-time.
//
// OFDM-inspired: computes FFT convolution for Kerr-ODE stencil coupling.
// Replaces CPU rustfft for the neighbour-sum computation:
//   ns = IFFT(FFT(mag_sq) * kernel_fft)
//
// One workgroup = one FFT. 256 threads per workgroup (512/2 butterflies per stage).
// 9 stages (log2(512) = 9). All data in workgroup shared memory.
//
// Usage:
//   Pass 1 (mode=0): FFT forward on mag_sq, multiply by kernel_fft, IFFT
//   The kernel_fft is precomputed once and uploaded as a storage buffer.
//
// Dispatch: (n_positions, 1, 1) workgroups — one FFT per sequence position.

const N: u32 = 512u;
const HALF_N: u32 = 256u;
const LOG_N: u32 = 9u;
const PI: f32 = 3.14159265358979323846;

struct Params {
    n_bands: u32,       // actual data length (384), rest is zero-padded
    n_positions: u32,   // number of positions to process
    mode: u32,          // 0 = full pipeline (FFT + multiply + IFFT + extract)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;          // [n_positions * n_bands] mag_sq values
@group(0) @binding(1) var<storage, read> kernel_re: array<f32>;      // [512] precomputed kernel FFT real parts
@group(0) @binding(2) var<storage, read> kernel_im: array<f32>;      // [512] precomputed kernel FFT imag parts
@group(0) @binding(3) var<storage, read_write> output: array<f32>;   // [n_positions * n_bands] neighbour sums
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory: real and imaginary parts of the 512-point complex array
var<workgroup> re: array<f32, 512>;
var<workgroup> im: array<f32, 512>;

// Bit-reverse a 9-bit index
fn bit_reverse_9(x: u32) -> u32 {
    var v = x;
    v = ((v & 0x1F0u) >> 5u) | ((v & 0x00Fu) << 5u);  // swap groups
    // Full 9-bit reversal
    v = ((v >> 1u) & 0x055u) | ((v & 0x055u) << 1u);   // swap adjacent bits
    v = ((v >> 2u) & 0x033u) | ((v & 0x033u) << 2u);   // swap pairs
    v = ((v >> 4u) & 0x00Fu) | ((v & 0x00Fu) << 4u);   // swap nibbles
    // Now we have 8-bit reversal in bits 0-7, plus bit 8
    // For 9 bits: reverse all 9 then keep lower 9
    var r: u32 = 0u;
    var tmp = x;
    for (var i: u32 = 0u; i < LOG_N; i++) {
        r = (r << 1u) | (tmp & 1u);
        tmp >>= 1u;
    }
    return r;
}

@compute @workgroup_size(256)
fn fft_convolve(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let pos = wg.x;
    if (pos >= params.n_positions) { return; }

    let n_bands = params.n_bands;
    let in_base = pos * n_bands;

    // ─── Step 1: Load mag_sq into shared memory with bit-reversal permutation ───
    // Each thread handles 2 elements (512 elements / 256 threads)
    let idx0 = tid;
    let idx1 = tid + HALF_N;

    let br0 = bit_reverse_9(idx0);
    let br1 = bit_reverse_9(idx1);

    // Load with zero-padding: only first n_bands elements have data
    if (br0 < n_bands) {
        re[idx0] = input[in_base + br0];
    } else {
        re[idx0] = 0.0;
    }
    im[idx0] = 0.0;

    if (br1 < n_bands) {
        re[idx1] = input[in_base + br1];
    } else {
        re[idx1] = 0.0;
    }
    im[idx1] = 0.0;

    workgroupBarrier();

    // ─── Step 2: FFT butterfly stages (9 stages for 512 points) ───
    for (var stage: u32 = 0u; stage < LOG_N; stage++) {
        let half_size = 1u << stage;           // half the butterfly group size
        let group_size = half_size << 1u;      // full butterfly group size
        let twiddle_stride = N >> (stage + 1u); // twiddle factor spacing

        // Each thread does one butterfly per pair of elements it handles
        // Thread tid handles butterfly within its group
        let group0 = idx0 / group_size;
        let pos_in_group0 = idx0 % group_size;

        if (pos_in_group0 < half_size) {
            let k = pos_in_group0;
            let top = group0 * group_size + k;
            let bot = top + half_size;

            // Twiddle factor: W_N^(k * twiddle_stride) = exp(-2πi * k * twiddle_stride / N)
            let angle = -2.0 * PI * f32(k * twiddle_stride) / f32(N);
            let tw_re = cos(angle);
            let tw_im = sin(angle);

            // Complex multiply: twiddle * bottom
            let b_re = re[bot] * tw_re - im[bot] * tw_im;
            let b_im = re[bot] * tw_im + im[bot] * tw_re;

            // Butterfly
            let t_re = re[top];
            let t_im = im[top];
            re[top] = t_re + b_re;
            im[top] = t_im + b_im;
            re[bot] = t_re - b_re;
            im[bot] = t_im - b_im;
        }

        let group1 = idx1 / group_size;
        let pos_in_group1 = idx1 % group_size;

        if (pos_in_group1 < half_size) {
            let k = pos_in_group1;
            let top = group1 * group_size + k;
            let bot = top + half_size;

            let angle = -2.0 * PI * f32(k * twiddle_stride) / f32(N);
            let tw_re = cos(angle);
            let tw_im = sin(angle);

            let b_re = re[bot] * tw_re - im[bot] * tw_im;
            let b_im = re[bot] * tw_im + im[bot] * tw_re;

            let t_re = re[top];
            let t_im = im[top];
            re[top] = t_re + b_re;
            im[top] = t_im + b_im;
            re[bot] = t_re - b_re;
            im[bot] = t_im - b_im;
        }

        workgroupBarrier();
    }

    // ─── Step 3: Pointwise multiply with precomputed kernel FFT ───
    // Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    {
        let a_re = re[idx0];
        let a_im = im[idx0];
        let k_re = kernel_re[idx0];
        let k_im = kernel_im[idx0];
        re[idx0] = a_re * k_re - a_im * k_im;
        im[idx0] = a_re * k_im + a_im * k_re;
    }
    {
        let a_re = re[idx1];
        let a_im = im[idx1];
        let k_re = kernel_re[idx1];
        let k_im = kernel_im[idx1];
        re[idx1] = a_re * k_re - a_im * k_im;
        im[idx1] = a_re * k_im + a_im * k_re;
    }

    workgroupBarrier();

    // ─── Step 4: IFFT = conjugate → FFT → conjugate → scale by 1/N ───
    // Conjugate
    im[idx0] = -im[idx0];
    im[idx1] = -im[idx1];
    workgroupBarrier();

    // Bit-reversal permutation for IFFT
    // We need to re-permute the data. Use a temp approach: copy to local, barrier, write back.
    let br0_val_re = re[idx0];
    let br0_val_im = im[idx0];
    let br1_val_re = re[idx1];
    let br1_val_im = im[idx1];
    workgroupBarrier();

    let dest0 = bit_reverse_9(idx0);
    let dest1 = bit_reverse_9(idx1);
    re[dest0] = br0_val_re;
    im[dest0] = br0_val_im;
    re[dest1] = br1_val_re;
    im[dest1] = br1_val_im;
    workgroupBarrier();

    // FFT butterfly stages again (same code, reused for IFFT)
    for (var stage: u32 = 0u; stage < LOG_N; stage++) {
        let half_size = 1u << stage;
        let group_size = half_size << 1u;
        let twiddle_stride = N >> (stage + 1u);

        let group0_i = idx0 / group_size;
        let pos_in_group0_i = idx0 % group_size;

        if (pos_in_group0_i < half_size) {
            let k = pos_in_group0_i;
            let top = group0_i * group_size + k;
            let bot = top + half_size;

            let angle = -2.0 * PI * f32(k * twiddle_stride) / f32(N);
            let tw_re = cos(angle);
            let tw_im = sin(angle);

            let b_re = re[bot] * tw_re - im[bot] * tw_im;
            let b_im = re[bot] * tw_im + im[bot] * tw_re;

            let t_re = re[top];
            let t_im = im[top];
            re[top] = t_re + b_re;
            im[top] = t_im + b_im;
            re[bot] = t_re - b_re;
            im[bot] = t_im - b_im;
        }

        let group1_i = idx1 / group_size;
        let pos_in_group1_i = idx1 % group_size;

        if (pos_in_group1_i < half_size) {
            let k = pos_in_group1_i;
            let top = group1_i * group_size + k;
            let bot = top + half_size;

            let angle = -2.0 * PI * f32(k * twiddle_stride) / f32(N);
            let tw_re = cos(angle);
            let tw_im = sin(angle);

            let b_re = re[bot] * tw_re - im[bot] * tw_im;
            let b_im = re[bot] * tw_im + im[bot] * tw_re;

            let t_re = re[top];
            let t_im = im[top];
            re[top] = t_re + b_re;
            im[top] = t_im + b_im;
            re[bot] = t_re - b_re;
            im[bot] = t_im - b_im;
        }

        workgroupBarrier();
    }

    // Conjugate and scale by 1/N
    let scale = 1.0 / f32(N);
    re[idx0] = re[idx0] * scale;
    im[idx0] = -im[idx0] * scale;
    re[idx1] = re[idx1] * scale;
    im[idx1] = -im[idx1] * scale;

    workgroupBarrier();

    // ─── Step 5: Extract real parts for first n_bands elements ───
    if (idx0 < n_bands) {
        output[pos * n_bands + idx0] = re[idx0];
    }
    if (idx1 < n_bands) {
        output[pos * n_bands + idx1] = re[idx1];
    }
}
