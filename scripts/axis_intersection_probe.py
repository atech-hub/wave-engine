#!/usr/bin/env python3
"""Axis Intersection Probe — are the four catalog-analog axes independent?

Tests whether phase distinctiveness, dignity, directional asymmetry, and
targeted destruction measure the same underlying "structural importance"
or four genuinely different properties.

Framing discipline: we've caught three framing drifts in this investigation.
Don't claim correlations without checking scatter. Don't round up or down.
Don't dismiss outliers. If the data doesn't fit the buckets cleanly, report
"inconclusive" rather than forcing a verdict.

Usage:
    python scripts/axis_intersection_probe.py --resume <checkpoint> [options]
"""

import subprocess
import sys
import json
import math
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)


def get_l3_cos(engine, checkpoint, data, text, mode="encode"):
    """Run --encode or --encode-phases and extract L3 cosine."""
    flag = "--encode-phases" if mode == "phases" else "--encode"
    cmd = [engine, flag, text,
           "--resume", checkpoint, "--layers", "4", "--n-bands", "84",
           "--n-head", "4", "--out-proj-groups", "1",
           "--alpha", "0.1", "--beta", "0.2", "--data", data]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    for line in r.stdout.split("\n"):
        if "cos(input, output) per layer:" in line:
            parts = line.split("L3:")
            if len(parts) > 1:
                try:
                    return float(parts[1].strip())
                except ValueError:
                    pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Axis Intersection Probe")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--data", default="data/grammar_lesson_1.txt")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--relate-json", default="grammar_energy_test.json")
    parser.add_argument("--output", default=None)
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

    # ─── Axis 1: Phase distinctiveness ───
    print("=== Axis 1: Phase distinctiveness (from relate-vocab JSON) ===")
    with open(args.relate_json) as f:
        rdata = json.load(f)

    non_conj = defaultdict(int)
    total_pairs = defaultdict(int)
    for p in rdata["pairs"]:
        for tok in [p["a"], p["b"]]:
            total_pairs[tok] += 1
            if p.get("catalog") and p["catalog"] != "conjunction":
                non_conj[tok] += 1

    # phase_score = fraction of non-conjunction pairs (higher = more distinctive)
    phase_scores = {tok: non_conj[tok] / max(total_pairs[tok], 1) for tok in total_pairs}
    vocab = sorted(tok for tok in phase_scores if len(tok) == 1)
    print(f"  {len(vocab)} single-char tokens")

    # ─── Axis 2: Dignity (max context shift) ───
    print("\n=== Axis 2: Dignity (computing per-token max context shift) ===")
    context_chars = ["e", "t", ".", " ", "a"]
    dignity_scores = {}
    for i, tok in enumerate(vocab):
        solo = get_l3_cos(engine, args.resume, args.data, tok)
        if solo is None:
            continue
        shifts = []
        for ctx in context_chars:
            if ctx == tok:
                continue
            ab = get_l3_cos(engine, args.resume, args.data, tok + ctx)
            ba = get_l3_cos(engine, args.resume, args.data, ctx + tok)
            if ab is not None:
                shifts.append(abs(ab - solo))
            if ba is not None:
                shifts.append(abs(ba - solo))
        if shifts:
            dignity_scores[tok] = max(shifts)
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(vocab)} done...")
    print(f"  Computed for {len(dignity_scores)} tokens")

    # ─── Axis 3: Directional asymmetry (per-token mean |asym|) ───
    print("\n=== Axis 3: Directional asymmetry (computing per-token) ===")
    # Sample pairs for each token
    direction_scores = {}
    for i, tok in enumerate(vocab):
        asym_vals = []
        # Test against 6 partner tokens
        partners = [v for v in vocab if v != tok][:8]
        for partner in partners:
            ab = get_l3_cos(engine, args.resume, args.data, tok + partner)
            ba = get_l3_cos(engine, args.resume, args.data, partner + tok)
            if ab is not None and ba is not None:
                asym_vals.append(abs(ab - ba))
        if asym_vals:
            direction_scores[tok] = sum(asym_vals) / len(asym_vals)
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(vocab)} done...")
    print(f"  Computed for {len(direction_scores)} tokens")

    # ─── Axis 4: Destruction (solo encoding L3 cos — lower = more destroyed) ───
    print("\n=== Axis 4: Destruction (solo L3 cos per token) ===")
    destruction_scores = {}
    for i, tok in enumerate(vocab):
        cos = get_l3_cos(engine, args.resume, args.data, tok)
        if cos is not None:
            destruction_scores[tok] = 1.0 - cos  # invert: high = more destroyed
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(vocab)} done...")
    print(f"  Computed for {len(destruction_scores)} tokens")

    # ─── Intersect: tokens present in all four axes ───
    common = [t for t in vocab
              if t in phase_scores and t in dignity_scores
              and t in direction_scores and t in destruction_scores]
    print(f"\n  Common tokens across all 4 axes: {len(common)}")

    if len(common) < 10:
        print("  Too few tokens for meaningful analysis")
        return

    # Build arrays
    # Note: dignity is INVERTED (high dignity = context-dependent = low structural importance)
    # So dignity_inv = 1 - dignity_score (high = context-independent = structurally important)
    phase = np.array([phase_scores[t] for t in common])
    dignity_inv = np.array([1.0 - dignity_scores[t] for t in common])
    direction = np.array([direction_scores[t] for t in common])
    destruction = np.array([destruction_scores[t] for t in common])

    axes = {
        "phase": phase,
        "dignity_inv": dignity_inv,
        "direction": direction,
        "destruction": destruction,
    }
    axis_names = list(axes.keys())

    # ─── Pairwise correlations ───
    print("\n=== Pairwise correlations (Pearson r) ===")
    correlations = {}
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            a, b = axis_names[i], axis_names[j]
            r = np.corrcoef(axes[a], axes[b])[0, 1]
            correlations[(a, b)] = r
            print(f"  {a:15s} <-> {b:15s}:  r = {r:+.4f}")

    all_r = list(correlations.values())
    mean_r = np.mean(np.abs(all_r))
    print(f"\n  Mean |r|: {mean_r:.4f}")

    # ─── Top-10 intersection ───
    print("\n=== Top-10 intersection ===")
    top10s = {}
    for name, vals in axes.items():
        ranked = sorted(range(len(common)), key=lambda i: -vals[i])
        top10s[name] = set(common[i] for i in ranked[:10])
        top_list = [common[i] for i in ranked[:10]]
        print(f"  {name:15s} top 10: {top_list}")

    # Count appearances
    appearance_count = defaultdict(int)
    for tok in common:
        for name in axis_names:
            if tok in top10s[name]:
                appearance_count[tok] += 1

    in_4 = [t for t in common if appearance_count[t] == 4]
    in_3 = [t for t in common if appearance_count[t] == 3]
    in_2 = [t for t in common if appearance_count[t] == 2]
    in_1 = [t for t in common if appearance_count[t] == 1]
    print(f"\n  Tokens in all 4 top-10s: {in_4}")
    print(f"  Tokens in exactly 3:     {in_3}")
    print(f"  Tokens in exactly 2:     {in_2}")
    print(f"  Tokens in exactly 1:     {in_1}")

    # ─── Composite score ───
    print("\n=== Composite structural importance (top 15) ===")
    # Normalise each axis to [0,1]
    def normalise(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    composite = (normalise(phase) + normalise(dignity_inv) +
                 normalise(direction) + normalise(destruction)) / 4.0
    ranked = sorted(range(len(common)), key=lambda i: -composite[i])

    print(f"  {'rank':>4s}  {'token':>5s}  {'phase':>7s}  {'dig_inv':>7s}  {'direct':>7s}  {'destruct':>8s}  {'composite':>9s}")
    print(f"  {'-'*55}")
    for rank, idx in enumerate(ranked[:15]):
        t = common[idx]
        print(f"  {rank+1:4d}  '{t:>3s}'  {phase[idx]:7.3f}  {dignity_inv[idx]:7.3f}  {direction[idx]:7.3f}  {destruction[idx]:8.3f}  {composite[idx]:9.3f}")

    # ─── Verdict ───
    print("\n=== VERDICT ===")
    high_corr = sum(1 for r in all_r if abs(r) > 0.6)
    low_corr = sum(1 for r in all_r if abs(r) < 0.3)
    n_in_all4 = len(in_4)

    if high_corr >= 4 and n_in_all4 >= 7:
        verdict = "ONE_PROPERTY"
        print("  All axes measure ONE property. Engine should expose a unified composite score.")
    elif low_corr >= 4 and n_in_all4 <= 2:
        verdict = "INDEPENDENT"
        print("  Axes are INDEPENDENT. Engine should expose four separate, uncorrelated metrics.")
    elif high_corr >= 2 or n_in_all4 >= 3:
        verdict = "PARTIALLY_INDEPENDENT"
        print("  Axes are PARTIALLY INDEPENDENT. Engine should expose them separately")
        print("  but document the correlations.")
    else:
        verdict = "INCONCLUSIVE"
        print("  Inconclusive — data doesn't fit the interpretation buckets cleanly.")
        print("  Need more data or BPE-level testing.")

    print(f"\n  high_corr (|r|>0.6): {high_corr}/6")
    print(f"  low_corr  (|r|<0.3): {low_corr}/6")
    print(f"  tokens in all 4 top-10s: {n_in_all4}")

    if args.output:
        out = {
            "n_tokens": len(common),
            "correlations": {f"{a}_{b}": round(r, 4) for (a, b), r in correlations.items()},
            "mean_abs_r": round(mean_r, 4),
            "top10_intersection": {"in_4": in_4, "in_3": in_3, "in_2": in_2},
            "verdict": verdict,
            "composite_top15": [
                {"token": common[i], "composite": round(float(composite[i]), 4)}
                for i in ranked[:15]
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  JSON: {args.output}")


if __name__ == "__main__":
    main()
