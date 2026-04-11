#!/usr/bin/env python3
"""Grid-Level Encoding Probe — separate embedding structure from learned structure.

The multi-grid embedding uses coprime moduli (m1, m2). This probe generates
phase patterns at specific grid positions and feeds them to the model to test:

1. On-grid vs off-grid: does the model process grid-native positions differently
   from positions that fall between grid points?
2. Grid-1 vs grid-2: does the n=9 cluster at grid-2 bands activate when we
   encode directly into grid-2?
3. Interpolated positions: what happens at fractional grid positions that have
   no training signal?

Usage:
    python scripts/grid_encoding_probe.py --resume <checkpoint> [options]
"""

import subprocess
import sys
import math
import json
import os

PI = math.pi


def encode_grid_position(n_bands, m1, m2, position, grid="both"):
    """Generate per-band phases for a grid position.

    Args:
        n_bands: total bands (half per grid)
        m1, m2: coprime moduli
        position: integer or float grid position
        grid: "both", "grid1", "grid2"

    Returns: list of (band, phase_radians) pairs for --encode-phases
    """
    half = n_bands // 2
    pairs = []

    if grid in ("both", "grid1"):
        theta1 = (position % m1) * 2.0 * PI / m1
        for n in range(half):
            phase = (n + 1) * theta1
            pairs.append((n, phase))

    if grid in ("both", "grid2"):
        theta2 = (position % m2) * 2.0 * PI / m2
        for n in range(half):
            band = half + n
            phase = (n + 1) * theta2
            pairs.append((band, phase))

    return pairs


def format_encode_phases(pairs):
    """Format as --encode-phases string."""
    return ",".join(f"{band}:{phase:.6f}" for band, phase in pairs)


def run_encode_phases(engine, checkpoint, data, phases_str, n_bands=84, n_layers=4, alpha=0.1, beta=0.2):
    """Run --encode-phases and parse output."""
    cmd = [
        engine, "--encode-phases", phases_str,
        "--resume", checkpoint,
        "--layers", str(n_layers),
        "--n-bands", str(n_bands),
        "--n-head", "4",
        "--out-proj-groups", "1",
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--data", data,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    cosines = {}
    for line in result.stdout.split("\n"):
        if "cos(input, output) per layer:" in line:
            parts = line.split("L")
            for p in parts[1:]:
                if ":" in p:
                    try:
                        layer_str, val_str = p.split(":")
                        cosines[int(layer_str.strip())] = float(val_str.strip())
                    except ValueError:
                        pass

    # Parse decoder readout
    lm_top = []
    phase_top = []
    for line in result.stdout.split("\n"):
        if "lm_head top" in line:
            parts = line.split(")")
            for p in parts[:-1]:
                if "(" in p:
                    try:
                        score = float(p.split("(")[-1].strip())
                        tok = p.split("(")[0].strip().split()[-1]
                        lm_top.append((tok, score))
                    except ValueError:
                        pass
        if "phase-native top" in line:
            parts = line.split(")")
            for p in parts[:-1]:
                if "(" in p:
                    try:
                        score = float(p.split("(")[-1].strip())
                        tok = p.split("(")[0].strip().split()[-1]
                        phase_top.append((tok, score))
                    except ValueError:
                        pass

    return {"cosines": cosines, "lm_top": lm_top[:3], "phase_top": phase_top[:3]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grid-Level Encoding Probe")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--data", default="data/grammar_lesson_1.txt")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--n-bands", type=int, default=84)
    parser.add_argument("--m1", type=int, default=9)
    parser.add_argument("--m2", type=int, default=11)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    engine = args.engine
    if engine is None:
        for c in ["target/release/wave-engine.exe", "target/debug/wave-engine.exe"]:
            if os.path.exists(c):
                engine = os.path.abspath(c)
                break
    if engine is None:
        print("ERROR: wave-engine not found")
        sys.exit(1)

    n_bands = args.n_bands
    m1, m2 = args.m1, args.m2
    half = n_bands // 2

    print(f"Grid encoding probe: m1={m1}, m2={m2}, {n_bands} bands")
    print(f"  Grid-1: {m1} positions, {half} bands (0-{half-1})")
    print(f"  Grid-2: {m2} positions, {half} bands ({half}-{n_bands-1})")
    print()

    results = {"m1": m1, "m2": m2, "n_bands": n_bands, "tests": []}

    # ─── Test 1: On-grid positions (all m1 × m2 = vocab-like positions) ───
    print("=== Test 1: On-grid positions ===")
    print(f"  Encoding positions 0..{m1*m2-1} (on-grid, both grids)")
    on_grid_cosines = []
    for pos in range(min(m1 * m2, 30)):  # cap at 30 to keep runtime sane
        pairs = encode_grid_position(n_bands, m1, m2, pos)
        phases_str = format_encode_phases(pairs)
        r = run_encode_phases(engine, args.resume, args.data, phases_str, n_bands)
        l3 = r["cosines"].get(3, 0)
        on_grid_cosines.append(l3)
        if pos < 10 or pos % 10 == 0:
            cos_str = "  ".join(f"L{k}:{v:.2f}" for k, v in sorted(r["cosines"].items()))
            print(f"  pos={pos:3d}  {cos_str}")

    avg_on = sum(on_grid_cosines) / len(on_grid_cosines) if on_grid_cosines else 0
    print(f"  Average L3 cos (on-grid): {avg_on:.3f}")

    # ─── Test 2: Off-grid (fractional positions) ───
    print(f"\n=== Test 2: Off-grid positions (fractional) ===")
    off_grid_cosines = []
    for frac in [0.5, 1.5, 2.5, 3.5, 4.5, 0.25, 0.75, 1.33, 2.67, 3.14]:
        pairs = encode_grid_position(n_bands, m1, m2, frac)
        phases_str = format_encode_phases(pairs)
        r = run_encode_phases(engine, args.resume, args.data, phases_str, n_bands)
        l3 = r["cosines"].get(3, 0)
        off_grid_cosines.append(l3)
        cos_str = "  ".join(f"L{k}:{v:.2f}" for k, v in sorted(r["cosines"].items()))
        print(f"  pos={frac:5.2f}  {cos_str}")

    avg_off = sum(off_grid_cosines) / len(off_grid_cosines) if off_grid_cosines else 0
    print(f"  Average L3 cos (off-grid): {avg_off:.3f}")

    # ─── Test 3: Grid-1 only vs Grid-2 only ───
    print(f"\n=== Test 3: Grid-1 only vs Grid-2 only ===")
    g1_cosines = []
    g2_cosines = []
    for pos in range(max(m1, m2)):
        # Grid-1 only
        pairs1 = encode_grid_position(n_bands, m1, m2, pos, grid="grid1")
        # Fill grid-2 bands with zero phase (magnitude 1)
        for b in range(half, n_bands):
            pairs1.append((b, 0.0))
        r1 = run_encode_phases(engine, args.resume, args.data, format_encode_phases(pairs1), n_bands)
        g1_cosines.append(r1["cosines"].get(3, 0))

        # Grid-2 only
        pairs2 = encode_grid_position(n_bands, m1, m2, pos, grid="grid2")
        for b in range(half):
            pairs2.append((b, 0.0))
        r2 = run_encode_phases(engine, args.resume, args.data, format_encode_phases(pairs2), n_bands)
        g2_cosines.append(r2["cosines"].get(3, 0))

        print(f"  pos={pos:2d}  grid1 L3={r1['cosines'].get(3,0):.3f}  grid2 L3={r2['cosines'].get(3,0):.3f}")

    avg_g1 = sum(g1_cosines) / len(g1_cosines)
    avg_g2 = sum(g2_cosines) / len(g2_cosines)
    print(f"  Average: grid1={avg_g1:.3f}  grid2={avg_g2:.3f}")

    # ─── Summary ───
    print(f"\n=== SUMMARY ===")
    print(f"  On-grid avg L3 cos:  {avg_on:.3f}")
    print(f"  Off-grid avg L3 cos: {avg_off:.3f}")
    print(f"  Grid-1 only avg:     {avg_g1:.3f}")
    print(f"  Grid-2 only avg:     {avg_g2:.3f}")
    print()

    if abs(avg_on - avg_off) > 0.05:
        print(f"  On-grid vs off-grid: {abs(avg_on - avg_off):.3f} difference — model distinguishes grid-native from foreign")
    else:
        print(f"  On-grid vs off-grid: {abs(avg_on - avg_off):.3f} difference — model treats both similarly")

    if abs(avg_g1 - avg_g2) > 0.05:
        print(f"  Grid-1 vs Grid-2: {abs(avg_g1 - avg_g2):.3f} difference — grids processed differently")
    else:
        print(f"  Grid-1 vs Grid-2: {abs(avg_g1 - avg_g2):.3f} difference — grids processed similarly")

    results["summary"] = {
        "on_grid_avg_l3": round(avg_on, 4),
        "off_grid_avg_l3": round(avg_off, 4),
        "grid1_avg_l3": round(avg_g1, 4),
        "grid2_avg_l3": round(avg_g2, 4),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON: {args.output}")


if __name__ == "__main__":
    main()
