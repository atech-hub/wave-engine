#!/usr/bin/env python3
"""Multi-Resolution Harmonic Probe — test whether structure exists at harmonics
beyond the standard n=1..12 sweep.

The Vedic Varga system encodes relationships at specific harmonic resolutions:
  n=9 (Navamsa, 108 buckets), n=27 (Nakshatra), n=60 (Shashtiamsa)

If the model builds structure at these higher harmonics that doesn't show at
n=1..12, we're missing relationships. If the distribution is uniform across
all harmonics, the higher ones are noise.

Usage:
    python scripts/multi_resolution_probe.py <scan_dir> [--output FILE] [--baseline-runs 5]

Reads phases.bin from an existing galaxy scan. No inference needed.
"""

import struct
import json
import sys
import math
import os
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy")
    sys.exit(1)


# Extended harmonic sweep — standard + Vedic Varga resolutions
HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 27, 36, 60]

# Standard (current engine sweep)
STANDARD_HARMONICS = {1, 2, 3, 4, 5, 6, 8, 12}

# Catalog for matching
CATALOG = [
    ("conjunction", 0.0, 8.0), ("opposition", 180.0, 8.0),
    ("trine", 120.0, 8.0), ("square", 90.0, 7.0),
    ("quintile", 72.0, 2.0), ("sextile", 60.0, 6.0),
    ("semi-square", 45.0, 2.0), ("semi-sextile", 30.0, 2.0),
    ("quincunx", 150.0, 2.0), ("sesquiquadrate", 135.0, 2.0),
    ("bi-quintile", 144.0, 2.0),
]


def load_phases(path):
    """Load phases.bin, return (phases[layer][pos][band], dims)."""
    with open(path, "rb") as f:
        magic, ver, n_layers, n_bands, n_pos, pad = struct.unpack("<6I", f.read(24))
        assert magic == 0x50484153, f"Bad magic: {magic:#x}"
        data = np.frombuffer(f.read(), dtype=np.float32)
        phases = data.reshape(n_layers, n_pos, n_bands)
    return phases, {"n_layers": n_layers, "n_bands": n_bands, "n_positions": n_pos}


def compute_mrl(diffs):
    """Mean resultant length — coherence at optimal offset."""
    s = np.mean(np.sin(diffs))
    c = np.mean(np.cos(diffs))
    return math.sqrt(s * s + c * c)


def match_catalog(angle_deg):
    """Match angle to catalog entry."""
    for name, angle, orb in CATALOG:
        d = abs(angle_deg - angle)
        d = min(d, 360.0 - d)
        if d <= orb:
            return name
    return None


def analyze_layer(phases_layer, n_bands, n_pos):
    """Analyze one layer: for each pair, find best harmonic by MRL."""
    results = []
    best_at_harmonic = defaultdict(int)  # how many pairs have their best MRL at each n
    extended_only_pairs = []  # pairs where best harmonic is NOT in standard set

    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            # Phase differences across positions
            diffs = phases_layer[:, j] - phases_layer[:, i]  # [n_pos]

            best_mrl = 0.0
            best_n = 1
            best_mean_cos = 0.0
            per_harmonic = {}

            for n in HARMONICS:
                scaled = n * diffs
                mrl = compute_mrl(scaled)
                mean_cos = float(np.mean(np.cos(scaled)))

                per_harmonic[n] = {"mrl": round(mrl, 4), "mean_cos": round(mean_cos, 4)}

                if mrl > best_mrl:
                    best_mrl = mrl
                    best_n = n
                    best_mean_cos = mean_cos

            best_at_harmonic[best_n] += 1

            # Is this pair only visible at extended harmonics?
            if best_n not in STANDARD_HARMONICS and best_mrl > 0.3:
                # Also check if standard harmonics miss it
                std_best_mrl = max(per_harmonic[n]["mrl"] for n in STANDARD_HARMONICS if n in per_harmonic)
                if best_mrl > std_best_mrl * 1.2:  # 20% better at extended
                    extended_only_pairs.append({
                        "pair": (i, j),
                        "best_n": best_n,
                        "best_mrl": round(best_mrl, 4),
                        "std_best_mrl": round(std_best_mrl, 4),
                        "improvement": round(best_mrl / max(std_best_mrl, 1e-6), 2),
                    })

    return best_at_harmonic, extended_only_pairs


def random_baseline(n_bands, n_pos, n_runs=5):
    """Random phase baseline — where do best harmonics land by chance?"""
    rng = np.random.default_rng(42)
    combined = defaultdict(list)

    for run in range(n_runs):
        random_phases = rng.uniform(-math.pi, math.pi, (n_pos, n_bands)).astype(np.float32)
        hist, _ = analyze_layer(random_phases, n_bands, n_pos)
        for n, count in hist.items():
            combined[n].append(count)

    # Return max across runs (worst-case random)
    return {n: max(counts) for n, counts in combined.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-resolution harmonic probe")
    parser.add_argument("scan_dir", help="Galaxy scan directory (contains phases.bin)")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--baseline-runs", type=int, default=5, help="Random baseline iterations")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    phases_path = os.path.join(args.scan_dir, "phases.bin")
    if not os.path.exists(phases_path):
        print(f"ERROR: {phases_path} not found")
        sys.exit(1)

    print(f"Loading phases from {phases_path}...")
    phases, dims = load_phases(phases_path)
    n_layers = dims["n_layers"]
    n_bands = dims["n_bands"]
    n_pos = dims["n_positions"]
    total_pairs = n_bands * (n_bands - 1) // 2
    print(f"  {n_layers} layers, {n_bands} bands, {n_pos} positions, {total_pairs} pairs per layer")
    print(f"  Harmonics: {HARMONICS}")
    print(f"  Standard: {sorted(STANDARD_HARMONICS)}")

    # Random baseline
    print(f"\nComputing random baseline ({args.baseline_runs} runs)...")
    baseline = random_baseline(n_bands, n_pos, args.baseline_runs)

    results = {"dims": dims, "harmonics": HARMONICS, "layers": []}

    for layer in range(n_layers):
        print(f"\n=== Layer {layer} ===")
        hist, extended = analyze_layer(phases[layer], n_bands, n_pos)

        print(f"  Best-MRL harmonic distribution ({total_pairs} pairs):")
        print(f"  {'n':>4s}  {'count':>6s}  {'%':>6s}  {'baseline':>8s}  {'signal':>8s}")
        print(f"  {'-'*38}")
        for n in HARMONICS:
            count = hist.get(n, 0)
            pct = 100.0 * count / total_pairs
            base = baseline.get(n, 0)
            signal = "YES" if count > base * 1.5 else ""
            marker = " *" if n not in STANDARD_HARMONICS and signal else ""
            print(f"  {n:>4d}  {count:>6d}  {pct:>5.1f}%  {base:>8d}  {signal:>8s}{marker}")

        if extended:
            print(f"\n  Extended-only pairs (best at n>{max(STANDARD_HARMONICS)}, >20% improvement):")
            for p in sorted(extended, key=lambda x: -x["best_mrl"])[:15]:
                print(f"    bands ({p['pair'][0]},{p['pair'][1]}): n={p['best_n']} MRL={p['best_mrl']:.3f} "
                      f"(std best={p['std_best_mrl']:.3f}, {p['improvement']:.1f}x)")
        else:
            print(f"\n  No extended-only pairs found (standard harmonics sufficient)")

        # Summary stats
        std_count = sum(hist.get(n, 0) for n in STANDARD_HARMONICS)
        ext_count = sum(hist.get(n, 0) for n in HARMONICS if n not in STANDARD_HARMONICS)
        print(f"\n  Standard harmonics: {std_count}/{total_pairs} ({100*std_count/total_pairs:.1f}%)")
        print(f"  Extended harmonics: {ext_count}/{total_pairs} ({100*ext_count/total_pairs:.1f}%)")

        results["layers"].append({
            "layer": layer,
            "histogram": {str(n): hist.get(n, 0) for n in HARMONICS},
            "baseline": {str(n): baseline.get(n, 0) for n in HARMONICS},
            "extended_only_pairs": len(extended),
            "extended_pairs_detail": extended[:20],
            "standard_fraction": round(std_count / total_pairs, 4),
        })

    # Overall verdict
    print("\n=== VERDICT ===")
    all_ext = sum(r["extended_only_pairs"] for r in results["layers"])
    if all_ext > 0:
        print(f"  {all_ext} pairs across {n_layers} layers show better coherence at extended harmonics")
        print(f"  The model IS encoding structure at resolutions beyond n=12")
        print(f"  Worth baking into the engine: add these harmonics to the galaxy scan")
    else:
        print(f"  No pairs show significantly better coherence at extended harmonics")
        print(f"  Standard n={{1..12}} sweep is sufficient for this model")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON written to: {args.output}")


if __name__ == "__main__":
    main()
