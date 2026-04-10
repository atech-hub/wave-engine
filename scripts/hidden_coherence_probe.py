#!/usr/bin/env python3
"""Hidden Coherence Probe — detect phase coherence the galaxy scan might miss.

Three analyses on existing phases.bin data:
1. Quartet phase-sum trajectories (random / locked / rotating / oscillating)
2. Shifted-coherence per pair (optimal phase offset search)
3. Per-band spatial wavelength (phase drift across positions)

Plus a random-phase baseline to distinguish signal from noise.

Usage:
    python scripts/hidden_coherence_probe.py <scan_dir> [--baseline-runs 5] [--verbose]
"""

import struct
import json
import sys
import math
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy")
    sys.exit(1)


def load_phases(path):
    """Load phases.bin, return (phases[layer][pos][band], dims)."""
    with open(path, "rb") as f:
        magic, ver, n_layers, n_bands, n_pos, pad = struct.unpack("<6I", f.read(24))
        assert magic == 0x50484153, f"Bad magic: {magic:#x}"
        data = np.frombuffer(f.read(), dtype=np.float32)
        phases = data.reshape(n_layers, n_pos, n_bands)
    return phases, {"n_layers": n_layers, "n_bands": n_bands, "n_positions": n_pos}


def enumerate_fwm_quartets(n_bands):
    """Generate FWM quartets using the engine's ±2 stencil."""
    quartets = []
    for k in range(2, n_bands - 1):
        quartets.append((k - 2, k + 1, k - 1, k))
    for k in range(1, n_bands - 2):
        quartets.append((k - 1, k + 2, k, k + 1))
    return quartets


def wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def circular_variance(angles):
    """Circular variance of a set of angles. 0=locked, 1=uniform."""
    s = np.sin(angles)
    c = np.cos(angles)
    R = np.sqrt(np.mean(s) ** 2 + np.mean(c) ** 2)
    return 1.0 - R


def mean_resultant_length(angles):
    """Mean resultant length. 1=locked, 0=uniform."""
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return np.sqrt(s ** 2 + c ** 2)


# ─── Analysis 1: Quartet phase-sum trajectories ───

def analyze_quartets(phases_layer, quartets, verbose=False):
    """Classify quartet phase-sum trajectories."""
    n_pos = phases_layer.shape[0]
    results = {"total": len(quartets), "random": 0, "locked": 0,
               "rotating": 0, "oscillating": 0, "rotating_details": []}

    for a, b, c, d in quartets:
        # Phase-sum trajectory
        phi = phases_layer[:, a] + phases_layer[:, b] - phases_layer[:, c] - phases_layer[:, d]

        # Circular stats
        cv = circular_variance(phi)
        mrl = mean_resultant_length(phi)

        # Phase differences between consecutive positions
        dphi = np.array([wrap(phi[i + 1] - phi[i]) for i in range(n_pos - 1)])
        dphi_mean = np.mean(dphi)
        dphi_std = np.std(dphi)
        sign_changes = np.sum(np.diff(np.sign(dphi)) != 0) / max(len(dphi) - 1, 1)

        # Classify
        if cv < 0.2 and mrl > 0.8:
            results["locked"] += 1
        elif dphi_std < 0.5 and abs(dphi_mean) > 0.1:
            results["rotating"] += 1
            results["rotating_details"].append({
                "bands": [int(a), int(b), int(c), int(d)],
                "rate": round(float(dphi_mean), 4),
                "rate_std": round(float(dphi_std), 4),
            })
        elif sign_changes > 0.4 and dphi_std < 1.0:
            results["oscillating"] += 1
        else:
            results["random"] += 1

    # Sort rotating by abs rate (strongest first)
    results["rotating_details"].sort(key=lambda x: abs(x["rate"]), reverse=True)
    return results


# ─── Analysis 2: Shifted-coherence per pair ───

def analyze_shifted_pairs(phases_layer, harmonics=(1, 2, 3, 4, 6), verbose=False):
    """Find pairs with hidden coherence at a shifted phase offset."""
    n_pos, n_bands = phases_layer.shape
    results = {"total_tested": 0, "significant": 0, "top_pairs": []}
    all_gaps = []

    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            diff = phases_layer[:, i] - phases_layer[:, j]
            for n in harmonics:
                nd = n * diff
                # Unshifted coherence
                unshifted = abs(np.mean(np.cos(nd)))
                # Max shifted = mean resultant length (analytical optimum)
                S = np.mean(np.sin(nd))
                C = np.mean(np.cos(nd))
                max_shifted = np.sqrt(S ** 2 + C ** 2)
                optimal_phi = np.arctan2(S, C) / n

                results["total_tested"] += 1
                gap = max_shifted - unshifted

                if max_shifted > 0.5 and max_shifted / max(unshifted, 0.01) > 3.0:
                    results["significant"] += 1
                    all_gaps.append({
                        "bands": [int(i), int(j)],
                        "harmonic": int(n),
                        "unshifted": round(float(unshifted), 4),
                        "max_shifted": round(float(max_shifted), 4),
                        "optimal_phi": round(float(optimal_phi), 4),
                        "gap": round(float(gap), 4),
                    })

    # Top 20 by gap
    all_gaps.sort(key=lambda x: x["gap"], reverse=True)
    results["top_pairs"] = all_gaps[:20]
    return results


# ─── Analysis 3: Per-band spatial wavelength ───

def analyze_spatial_wavelength(phases_layer, verbose=False):
    """Check if bands have consistent phase drift across positions."""
    n_pos, n_bands = phases_layer.shape
    results = {"stationary": 0, "drifting": 0, "clustered": 0, "drifting_details": []}

    for k in range(n_bands):
        # Phase change between consecutive positions
        dtheta = np.array([wrap(phases_layer[i + 1, k] - phases_layer[i, k])
                           for i in range(n_pos - 1)])
        dt_mean = np.mean(dtheta)
        dt_std = np.std(dtheta)

        if dt_std < 0.5 and abs(dt_mean) > 0.05:
            results["drifting"] += 1
            wavelength = 2 * math.pi / abs(dt_mean) if abs(dt_mean) > 0.001 else float("inf")
            results["drifting_details"].append({
                "band": int(k),
                "mean_rate": round(float(dt_mean), 4),
                "wavelength": round(float(wavelength), 1),
                "rate_std": round(float(dt_std), 4),
            })
        elif dt_std > 1.0:
            results["stationary"] += 1
        else:
            # Check clustering by position mod
            best_reduction = 0
            best_mod = 0
            for mod in [2, 3, 4, 6]:
                groups = [[] for _ in range(mod)]
                for i, v in enumerate(dtheta):
                    groups[i % mod].append(v)
                within_var = np.mean([np.var(g) for g in groups if len(g) > 1])
                total_var = np.var(dtheta)
                if total_var > 0:
                    reduction = 1 - within_var / total_var
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_mod = mod
            if best_reduction > 0.3:
                results["clustered"] += 1
            else:
                results["stationary"] += 1

    results["drifting_details"].sort(key=lambda x: abs(x["mean_rate"]), reverse=True)
    return results


# ─── Random baseline ───

def run_baseline(phases_layer, quartets, harmonics, n_runs=5):
    """Shuffle positions per band, rerun analyses, return worst-case counts."""
    max_rotating = 0
    max_shifted = 0
    max_drifting = 0

    for _ in range(n_runs):
        shuffled = phases_layer.copy()
        for band in range(shuffled.shape[1]):
            np.random.shuffle(shuffled[:, band])

        q = analyze_quartets(shuffled, quartets)
        s = analyze_shifted_pairs(shuffled, harmonics)
        w = analyze_spatial_wavelength(shuffled)

        max_rotating = max(max_rotating, q["rotating"])
        max_shifted = max(max_shifted, s["significant"])
        max_drifting = max(max_drifting, w["drifting"])

    return {
        "quartets_rotating_max": max_rotating,
        "shifted_significant_max": max_shifted,
        "drifting_max": max_drifting,
    }


# ─── Output ───

def to_markdown(results, scan_dir):
    lines = []
    lines.append(f"# Hidden Coherence Probe — {scan_dir}")
    lines.append("")
    lines.append(f"**Verdict:** {results['verdict']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Layer | Rotating Q | Baseline | Shifted pairs | Baseline | Drifting bands | Baseline |")
    lines.append("|-------|-----------|----------|--------------|----------|---------------|----------|")

    for layer in results["layers"]:
        li = layer["layer"]
        q = layer["quartets"]
        s = layer["shifted_pairs"]
        w = layer["spatial_wavelength"]
        b = layer["baseline"]
        lines.append(f"| L{li} | {q['rotating']} | {b['quartets_rotating_max']} | "
                     f"{s['significant']} | {b['shifted_significant_max']} | "
                     f"{w['drifting']} | {b['drifting_max']} |")

    for layer in results["layers"]:
        li = layer["layer"]
        lines.append("")
        lines.append(f"## Layer {li}")

        q = layer["quartets"]
        lines.append(f"- Quartets: {q['total']} total — {q['random']} random, "
                     f"{q['locked']} locked, {q['rotating']} rotating, {q['oscillating']} oscillating")
        if q["rotating_details"]:
            lines.append("- Rotating quartets:")
            for rd in q["rotating_details"][:10]:
                lines.append(f"  - Bands {rd['bands']}: rate={rd['rate']}, std={rd['rate_std']}")

        s = layer["shifted_pairs"]
        lines.append(f"- Shifted pairs: {s['significant']}/{s['total_tested']} significant")
        if s["top_pairs"]:
            lines.append("- Top shifted pairs:")
            for sp in s["top_pairs"][:10]:
                lines.append(f"  - Bands {sp['bands']} n={sp['harmonic']}: "
                             f"unshifted={sp['unshifted']}, shifted={sp['max_shifted']}, "
                             f"phi={sp['optimal_phi']}")

        w = layer["spatial_wavelength"]
        lines.append(f"- Spatial: {w['stationary']} stationary, {w['drifting']} drifting, "
                     f"{w['clustered']} clustered")
        if w["drifting_details"]:
            lines.append("- Drifting bands:")
            for dd in w["drifting_details"][:10]:
                lines.append(f"  - Band {dd['band']}: rate={dd['mean_rate']}, "
                             f"wavelength={dd['wavelength']} positions, std={dd['rate_std']}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hidden coherence probe")
    parser.add_argument("scan_dir", help="Galaxy scan directory")
    parser.add_argument("--baseline-runs", type=int, default=5, help="Baseline shuffle runs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    phases_path = scan_dir / "phases.bin"
    if not phases_path.exists():
        print(f"ERROR: {phases_path} not found")
        sys.exit(1)

    print(f"Loading phases from {phases_path}...")
    phases, dims = load_phases(phases_path)
    print(f"  {dims['n_layers']} layers, {dims['n_bands']} bands, {dims['n_positions']} positions")

    quartets = enumerate_fwm_quartets(dims["n_bands"])
    harmonics = (1, 2, 3, 4, 6)
    print(f"  {len(quartets)} FWM quartets, {len(harmonics)} harmonics")

    results = {
        "scan_dir": str(scan_dir),
        "dims": dims,
        "layers": [],
        "verdict": "",
    }

    any_hidden = False

    for li in range(dims["n_layers"]):
        print(f"\nLayer {li}...")
        layer_phases = phases[li]

        print("  Analysis 1: quartet trajectories...")
        q = analyze_quartets(layer_phases, quartets, args.verbose)
        print(f"    random={q['random']} locked={q['locked']} rotating={q['rotating']} oscillating={q['oscillating']}")

        print("  Analysis 2: shifted-coherence pairs...")
        s = analyze_shifted_pairs(layer_phases, harmonics, args.verbose)
        print(f"    significant={s['significant']}/{s['total_tested']}")

        print("  Analysis 3: spatial wavelength...")
        w = analyze_spatial_wavelength(layer_phases, args.verbose)
        print(f"    stationary={w['stationary']} drifting={w['drifting']} clustered={w['clustered']}")

        print(f"  Baseline ({args.baseline_runs} runs)...")
        b = run_baseline(layer_phases, quartets, harmonics, args.baseline_runs)
        print(f"    rotating_max={b['quartets_rotating_max']} shifted_max={b['shifted_significant_max']} drifting_max={b['drifting_max']}")

        # Check if real exceeds baseline
        if q["rotating"] > b["quartets_rotating_max"]:
            any_hidden = True
        if s["significant"] > b["shifted_significant_max"] * 2:
            any_hidden = True
        if w["drifting"] > b["drifting_max"] * 2:
            any_hidden = True

        results["layers"].append({
            "layer": li,
            "quartets": q,
            "shifted_pairs": s,
            "spatial_wavelength": w,
            "baseline": b,
        })

    if any_hidden:
        results["verdict"] = "HIDDEN COHERENCE FOUND — investigate further"
    else:
        results["verdict"] = "NO HIDDEN COHERENCE DETECTED — April 9 quartet collapse is genuine"

    # Write output
    out_json = scan_dir / "hidden_coherence.json"
    out_md = scan_dir / "hidden_coherence.md"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_md, "w") as f:
        f.write(to_markdown(results, str(scan_dir)))

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {results['verdict']}")
    print(f"{'=' * 60}")
    print(f"Output: {out_json}")
    print(f"        {out_md}")


if __name__ == "__main__":
    main()
