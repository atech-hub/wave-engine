"""Merge two WCHK checkpoints by averaging weights. Standalone script."""
import struct
import sys

def read_wchk(path):
    with open(path, 'rb') as f:
        data = f.read()
    magic = data[:4]
    assert magic == b'WCHK', f"Not WCHK: {magic}"
    header = data[:72]
    remaining = data[72:]
    n_params = len(remaining) // (4 * 3)  # m, v, params
    offset = n_params * 4 * 2  # skip adam m and v
    weights = list(struct.unpack(f'<{n_params}f', remaining[offset:offset + n_params * 4]))
    return header, weights, n_params

def save_merged(header, weights, n_params, path):
    with open(path, 'wb') as f:
        # Write original header (reset iter to 0)
        f.write(header[:44])
        f.write(struct.pack('<Q', 0))  # iteration = 0
        f.write(header[52:64])  # lr + rng
        f.write(struct.pack('<Q', 0))  # adam_t = 0
        # Zero optimizer state
        for _ in range(n_params): f.write(struct.pack('<f', 0.0))  # m
        for _ in range(n_params): f.write(struct.pack('<f', 0.0))  # v
        # Merged weights
        for w in weights: f.write(struct.pack('<f', w))

if __name__ == '__main__':
    a_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_cpu_a.bin'
    b_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoint_cpu_b.bin'
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'merged_checkpoint.bin'
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    h_a, w_a, n_a = read_wchk(a_path)
    h_b, w_b, n_b = read_wchk(b_path)
    print(f"A: {n_a:,} params, B: {n_b:,} params")
    assert n_a == n_b, f"Param count mismatch: {n_a} vs {n_b}"

    merged = [alpha * a + (1.0 - alpha) * b for a, b in zip(w_a, w_b)]
    save_merged(h_a, merged, n_a, out_path)
    print(f"Merged ({alpha:.0%} A + {1-alpha:.0%} B) -> {out_path}")
