#!/usr/bin/env python3
"""Directional Energy Flow Probe — does asymmetry cluster at catalog angles?

Wu Xing says generative (+72°) and destructive (+144°) cycles at the same
angles have opposite energy flow directions. Test: encode all token pairs
in both orders ("ab" vs "ba"), measure asymmetry, group by angular distance.

If asymmetry clusters at 72° and 144° more than at other angles, the
directed-cycle concept is physically real in the model.

Usage:
    python scripts/directional_probe.py --resume <checkpoint> [options]
"""

import subprocess
import sys
import math
import json
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)


def run_encode(engine, checkpoint, data, text, n_bands=84, n_layers=4, alpha=0.1, beta=0.2):
    """Run --encode and extract L3 cosine."""
    cmd = [
        engine, "--encode", text,
        "--resume", checkpoint,
        "--layers", str(n_layers),
        "--n-bands", str(n_bands),
        "--n-head", "4",
        "--out-proj-groups", "1",
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--data", data,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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
    return cosines


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Directional Energy Flow Probe")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--data", default="data/grammar_lesson_1.txt")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--n-bands", type=int, default=84)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-pairs", type=int, default=200, help="Max pairs to test (random sample if vocab too large)")
    args = parser.parse_args()

    engine = args.engine
    if engine is None:
        for c in [os.path.abspath("target/debug/wave-engine.exe"),
                   os.path.abspath("target/release/wave-engine.exe")]:
            if os.path.exists(c):
                engine = c
                break
    if engine is None:
        print("ERROR: wave-engine not found")
        sys.exit(1)

    # Build char vocab from data file
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    # Filter to printable chars that the model knows
    vocab = [c for c in chars if c.isprintable() and c != '"']
    print(f"Vocab: {len(vocab)} chars")

    # Select pairs to test
    all_pairs = []
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            if i < j:
                all_pairs.append((a, b))

    if len(all_pairs) > args.max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_pairs), args.max_pairs, replace=False)
        pairs = [all_pairs[i] for i in sorted(indices)]
        print(f"  Sampled {len(pairs)} of {len(all_pairs)} pairs")
    else:
        pairs = all_pairs
        print(f"  Testing all {len(pairs)} pairs")

    # Also load relate-vocab data for angular distances
    relate_file = None
    for candidate in ["grammar_vocab_relations.json", "grammar_energy_test.json"]:
        if os.path.exists(candidate):
            relate_file = candidate
            break

    angle_lookup = {}
    if relate_file:
        with open(relate_file) as f:
            rdata = json.load(f)
        for p in rdata.get("pairs", []):
            key = (p["a"], p["b"])
            angle_lookup[key] = p["angle"]
            angle_lookup[(p["b"], p["a"])] = p["angle"]
        print(f"  Angle data from {relate_file}: {len(angle_lookup)} pairs")

    # Run all pairs in both orders
    print(f"\n  Running {len(pairs)} pairs in both orders...")
    results = []
    for idx, (a, b) in enumerate(pairs):
        ab = a + b
        ba = b + a
        cos_ab = run_encode(engine, args.resume, args.data, ab, args.n_bands)
        cos_ba = run_encode(engine, args.resume, args.data, ba, args.n_bands)

        l3_ab = cos_ab.get(3, 0)
        l3_ba = cos_ba.get(3, 0)
        asym = l3_ab - l3_ba

        angle = angle_lookup.get((a, b), None)

        results.append({
            "a": a, "b": b,
            "l3_ab": round(l3_ab, 4), "l3_ba": round(l3_ba, 4),
            "asymmetry": round(asym, 4),
            "abs_asymmetry": round(abs(asym), 4),
            "angle": round(angle, 1) if angle is not None else None,
        })

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(pairs)} done...")

    # Group by angular distance bins
    print(f"\n=== Asymmetry by angular distance ===")
    bins = [(0, 15, "conjunction ~0"), (25, 35, "semi-sextile ~30"),
            (38, 52, "semi-square ~45"), (54, 66, "sextile ~60"),
            (66, 78, "quintile ~72"), (82, 98, "square ~90"),
            (112, 128, "trine ~120"), (132, 150, "sesquiq/biquint ~140"),
            (145, 155, "quincunx ~150"), (172, 188, "opposition ~180")]

    print(f"  {'Angle range':>25s}  {'n':>4s}  {'mean |asym|':>12s}  {'max |asym|':>10s}")
    print(f"  {'-'*55}")

    bin_results = []
    for lo, hi, label in bins:
        in_bin = [r for r in results if r["angle"] is not None and lo <= r["angle"] <= hi]
        if not in_bin:
            # Try wrapping (for angles near 360)
            in_bin = [r for r in results if r["angle"] is not None and (lo <= r["angle"] <= hi or lo <= (360-r["angle"]) <= hi)]
        if in_bin:
            abs_asym = [r["abs_asymmetry"] for r in in_bin]
            mean_a = sum(abs_asym) / len(abs_asym)
            max_a = max(abs_asym)
            print(f"  {label:>25s}  {len(in_bin):>4d}  {mean_a:>12.4f}  {max_a:>10.4f}")
            bin_results.append({"label": label, "lo": lo, "hi": hi, "n": len(in_bin),
                               "mean_abs_asym": round(mean_a, 4), "max_abs_asym": round(max_a, 4)})

    # Also compute global stats
    all_abs = [r["abs_asymmetry"] for r in results]
    global_mean = sum(all_abs) / len(all_abs)
    print(f"\n  Global mean |asymmetry|: {global_mean:.4f}")
    print(f"  Global max |asymmetry|: {max(all_abs):.4f}")

    # Top asymmetric pairs
    results.sort(key=lambda r: -r["abs_asymmetry"])
    print(f"\n=== Top 15 most asymmetric pairs ===")
    for r in results[:15]:
        angle_str = f"{r['angle']:.0f}" if r["angle"] is not None else "?"
        print(f"  '{r['a']}'{r['b']}' vs '{r['b']}'{r['a']}'  asym={r['asymmetry']:+.3f}  angle={angle_str}")

    # Wu Xing check: is asymmetry higher at 72° and 144° than at other angles?
    wu_xing_bins = [r for r in results if r["angle"] is not None and
                    ((66 <= r["angle"] <= 78) or (138 <= r["angle"] <= 150))]
    other_bins = [r for r in results if r["angle"] is not None and
                  not ((66 <= r["angle"] <= 78) or (138 <= r["angle"] <= 150))]

    if wu_xing_bins and other_bins:
        wu_mean = sum(r["abs_asymmetry"] for r in wu_xing_bins) / len(wu_xing_bins)
        oth_mean = sum(r["abs_asymmetry"] for r in other_bins) / len(other_bins)
        print(f"\n=== Wu Xing test ===")
        print(f"  Pairs at ~72/144: n={len(wu_xing_bins)}, mean |asym|={wu_mean:.4f}")
        print(f"  Other pairs:      n={len(other_bins)}, mean |asym|={oth_mean:.4f}")
        if wu_mean > oth_mean * 1.2:
            print(f"  Wu Xing angles show {wu_mean/oth_mean:.1f}x more asymmetry — DIRECTED CYCLES REAL")
        else:
            print(f"  Wu Xing angles NOT significantly more asymmetric — direction is general, not angle-specific")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"pairs": results, "bins": bin_results, "global_mean_abs_asym": round(global_mean, 4)}, f, indent=2)
        print(f"\n  JSON: {args.output}")


if __name__ == "__main__":
    main()
