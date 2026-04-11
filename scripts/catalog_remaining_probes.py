#!/usr/bin/env python3
"""Remaining catalog concept probes — Wu He, Liu Hai, Xiang Xing,
San Hui Fang, Sect, Reception.

Runs on existing grammar_vocab_relations.json + phases.bin data.
No new inference needed for most tests.

Usage:
    python scripts/catalog_remaining_probes.py <scan_dir> [--relate-json FILE]
"""

import struct
import json
import sys
import math
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)


def load_phases(path):
    with open(path, "rb") as f:
        magic, ver, n_layers, n_bands, n_pos, pad = struct.unpack("<6I", f.read(24))
        assert magic == 0x50484153, f"Bad magic: {magic:#x}"
        data = np.frombuffer(f.read(), dtype=np.float32)
        phases = data.reshape(n_layers, n_pos, n_bands)
    return phases, {"n_layers": n_layers, "n_bands": n_bands, "n_positions": n_pos}


def compute_mrl(diffs):
    s = np.mean(np.sin(diffs))
    c = np.mean(np.cos(diffs))
    return math.sqrt(s * s + c * c)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", help="Galaxy scan directory")
    parser.add_argument("--relate-json", default="grammar_energy_test.json")
    parser.add_argument("--m1", type=int, default=9)
    parser.add_argument("--m2", type=int, default=11)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    phases_path = os.path.join(args.scan_dir, "phases.bin")
    phases, dims = load_phases(phases_path)
    n_layers = dims["n_layers"]
    n_bands = dims["n_bands"]
    n_pos = dims["n_positions"]
    half = n_bands // 2
    m1, m2 = args.m1, args.m2

    with open(args.relate_json) as f:
        rdata = json.load(f)
    pairs = rdata["pairs"]

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # TEST A: Wu He — 180° on grid-1 vs 180° on grid-2
    # Same angle, different cycle. Do they behave differently?
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST A: Wu He (Heavenly Stems) — 180 on grid-1 vs grid-2")
    print("=" * 60)

    # Find pairs near 180° and classify by grid
    opp_g1 = []  # pairs where the 180° comes from grid-1 band differences
    opp_g2 = []  # pairs where the 180° comes from grid-2 band differences
    opp_cross = []  # cross-grid opposition pairs

    for p in pairs:
        angle = p["angle"]
        if abs(angle - 180) > 8 and abs(angle - (360 - 180)) > 8:
            continue
        a_tok, b_tok = p["a"], p["b"]
        mrl = p["mrl"]
        dsim = p.get("deform_sim", 0)

        # We can't directly map token labels to grid positions from the JSON
        # But we can check: do pairs at 180° have different energy signatures?
        opp_cross.append({"a": a_tok, "b": b_tok, "mrl": mrl, "deform_sim": dsim, "angle": angle})

    # Alternative: use phases.bin directly for grid-level 180° analysis
    # Grid-1 bands: 0..half-1, Grid-2 bands: half..n_bands-1
    g1_opp_mrl = []  # MRL of 180° band pairs within grid-1
    g2_opp_mrl = []  # MRL of 180° band pairs within grid-2

    for layer in [0, n_layers - 1]:  # check L0 and last layer
        # Within grid-1: pairs of bands where mean phase diff ~ 180°
        for i in range(half):
            for j in range(i + 1, half):
                diffs = phases[layer, :, j] - phases[layer, :, i]
                mean_diff = np.degrees(np.arctan2(np.mean(np.sin(diffs)), np.mean(np.cos(diffs)))) % 360
                if abs(mean_diff - 180) < 15:
                    mrl = compute_mrl(diffs)
                    g1_opp_mrl.append({"layer": layer, "i": i, "j": j, "mrl": mrl, "angle": mean_diff})

        # Within grid-2
        for i in range(half, n_bands):
            for j in range(i + 1, n_bands):
                diffs = phases[layer, :, j] - phases[layer, :, i]
                mean_diff = np.degrees(np.arctan2(np.mean(np.sin(diffs)), np.mean(np.cos(diffs)))) % 360
                if abs(mean_diff - 180) < 15:
                    mrl = compute_mrl(diffs)
                    g2_opp_mrl.append({"layer": layer, "i": i, "j": j, "mrl": mrl, "angle": mean_diff})

    g1_avg = np.mean([x["mrl"] for x in g1_opp_mrl]) if g1_opp_mrl else 0
    g2_avg = np.mean([x["mrl"] for x in g2_opp_mrl]) if g2_opp_mrl else 0

    print(f"  Grid-1 opposition pairs (bands 0-{half-1}): {len(g1_opp_mrl)}, mean MRL={g1_avg:.4f}")
    print(f"  Grid-2 opposition pairs (bands {half}-{n_bands-1}): {len(g2_opp_mrl)}, mean MRL={g2_avg:.4f}")
    if g1_avg > 0 and g2_avg > 0:
        ratio = g1_avg / g2_avg
        print(f"  Grid-1/Grid-2 ratio: {ratio:.3f}")
        if abs(ratio - 1.0) > 0.15:
            print(f"  DIFFERENT: 180 on grid-1 vs grid-2 produces different coherence")
        else:
            print(f"  SIMILAR: 180 behaves the same on both grids")

    # Token-level opposition with energy signature
    print(f"\n  Token-level oppositions (from relate-vocab): {len(opp_cross)}")
    if opp_cross:
        avg_dsim = np.mean([p["deform_sim"] for p in opp_cross])
        print(f"  Mean deform_sim of opposition pairs: {avg_dsim:.4f}")
        for p in sorted(opp_cross, key=lambda x: -x["mrl"])[:5]:
            print(f"    '{p['a']}' <-> '{p['b']}' angle={p['angle']:.0f} MRL={p['mrl']:.3f} deform_sim={p['deform_sim']:.3f}")

    results["wu_he"] = {
        "g1_opp_count": len(g1_opp_mrl), "g1_avg_mrl": round(g1_avg, 4),
        "g2_opp_count": len(g2_opp_mrl), "g2_avg_mrl": round(g2_avg, 4),
        "token_opp_count": len(opp_cross),
    }

    # ═══════════════════════════════════════════════════════════════
    # TEST B: San Hui Fang — adjacent-position clustering
    # Do tokens at adjacent grid positions cluster more than random?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST B: San Hui Fang (Seasonal Groupings) — adjacent clustering")
    print("=" * 60)

    # For each set of 3 adjacent positions on each grid, measure mean pairwise MRL
    # Build token-to-grid mapping
    all_tokens = sorted(set(p["a"] for p in pairs) | set(p["b"] for p in pairs))
    tok_idx = {t: i for i, t in enumerate(all_tokens)}

    # Pair MRL lookup
    pair_mrl = {}
    pair_dsim = {}
    for p in pairs:
        pair_mrl[(p["a"], p["b"])] = p["mrl"]
        pair_mrl[(p["b"], p["a"])] = p["mrl"]
        pair_dsim[(p["a"], p["b"])] = p.get("deform_sim", 0)
        pair_dsim[(p["b"], p["a"])] = p.get("deform_sim", 0)

    # Group tokens by grid-1 position
    g1_groups = defaultdict(list)
    for t in all_tokens:
        if t in tok_idx:
            g1_groups[tok_idx[t] % m1].append(t)

    # For 3 adjacent grid-1 positions, compute mean pairwise MRL
    adj_mrls = []
    non_adj_mrls = []
    for pos in range(m1):
        adj = set()
        for dp in [-1, 0, 1]:
            p = (pos + dp) % m1
            adj.update(g1_groups[p])
        adj = list(adj)
        # Pairwise MRL within adjacent group
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                key = (adj[i], adj[j])
                if key in pair_mrl:
                    adj_mrls.append(pair_mrl[key])

    # Random non-adjacent pairs for comparison
    import random
    random.seed(42)
    for _ in range(len(adj_mrls)):
        a, b = random.sample(all_tokens, 2)
        if (a, b) in pair_mrl:
            non_adj_mrls.append(pair_mrl[(a, b)])

    avg_adj = np.mean(adj_mrls) if adj_mrls else 0
    avg_non = np.mean(non_adj_mrls) if non_adj_mrls else 0
    print(f"  Adjacent-3 group pairs: {len(adj_mrls)}, mean MRL={avg_adj:.4f}")
    print(f"  Random pairs:           {len(non_adj_mrls)}, mean MRL={avg_non:.4f}")
    if avg_non > 0:
        print(f"  Ratio: {avg_adj/avg_non:.3f}x")
        if avg_adj / avg_non > 1.1:
            print(f"  CONFIRMED: adjacent positions cluster more than random")
        else:
            print(f"  NULL: no significant adjacency clustering")

    results["san_hui_fang"] = {
        "adj_pairs": len(adj_mrls), "adj_avg_mrl": round(avg_adj, 4),
        "random_pairs": len(non_adj_mrls), "random_avg_mrl": round(avg_non, 4),
    }

    # ═══════════════════════════════════════════════════════════════
    # TEST C: Xiang Xing — self-punishment
    # Does a token encoding through the model show self-conflict?
    # Measure: tokens where cos(in,out) is NEGATIVE (output opposes input)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST C: Xiang Xing (Self-Punishment) — tokens that oppose themselves")
    print("=" * 60)

    # Use per-token destruction from relate-vocab energy profiles
    profiles = rdata.get("energy_profiles", [])
    self_conflict = []
    for p in profiles:
        # Total energy ratio < 0.8 = heavy self-damping
        if p["total_energy_ratio"] < 0.80:
            self_conflict.append(p)

    self_conflict.sort(key=lambda x: x["total_energy_ratio"])
    print(f"  Tokens with heavy self-damping (energy < 0.80x):")
    for p in self_conflict[:10]:
        print(f"    '{p['token']}' energy={p['total_energy_ratio']:.3f}x  peak=b{p['peak_band']}({p['peak_ratio']:.1f}x)  damp=b{p['damp_band']}({p['damp_ratio']:.2f}x)")

    print(f"  Total self-damped tokens: {len(self_conflict)} / {len(profiles)}")

    # Check: are self-damped tokens the same ones that are phase-distinctive?
    phase_dist = {}
    total_p = defaultdict(int)
    non_conj = defaultdict(int)
    for p in pairs:
        for t in [p["a"], p["b"]]:
            total_p[t] += 1
            if p.get("catalog") and p["catalog"] != "conjunction":
                non_conj[t] += 1
    for t in total_p:
        phase_dist[t] = non_conj[t] / max(total_p[t], 1)

    self_damped_tokens = set(p["token"] for p in self_conflict)
    phase_top10 = set(sorted(phase_dist.keys(), key=lambda t: -phase_dist[t])[:10])
    overlap = self_damped_tokens & phase_top10
    print(f"\n  Self-damped AND phase-distinctive top-10: {overlap if overlap else 'none'}")

    results["xiang_xing"] = {
        "self_damped_count": len(self_conflict),
        "self_damped_tokens": [p["token"] for p in self_conflict[:10]],
    }

    # ═══════════════════════════════════════════════════════════════
    # TEST D: Liu Hai (Six Harms) — friction at mixed angles
    # The catalog describes specific friction pairs. Do pairs at
    # "uncomfortable" angles (not clean catalog matches) show
    # distinctive energy signatures?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST D: Liu Hai (Six Harms) — friction at non-catalog angles")
    print("=" * 60)

    # "Harm" angles: angles that DON'T match any catalog entry
    catalog_pairs = [p for p in pairs if p.get("catalog")]
    no_match_pairs = [p for p in pairs if not p.get("catalog")]

    cat_dsim = np.mean([p.get("deform_sim", 0) for p in catalog_pairs]) if catalog_pairs else 0
    no_dsim = np.mean([p.get("deform_sim", 0) for p in no_match_pairs]) if no_match_pairs else 0
    cat_mrl = np.mean([p["mrl"] for p in catalog_pairs]) if catalog_pairs else 0
    no_mrl = np.mean([p["mrl"] for p in no_match_pairs]) if no_match_pairs else 0

    print(f"  Catalog-matched pairs:   n={len(catalog_pairs):5d}  mean MRL={cat_mrl:.4f}  mean deform_sim={cat_dsim:.4f}")
    print(f"  No-catalog-match pairs:  n={len(no_match_pairs):5d}  mean MRL={no_mrl:.4f}  mean deform_sim={no_dsim:.4f}")
    print(f"  MRL ratio (catalog/none): {cat_mrl/max(no_mrl, 0.001):.3f}")
    print(f"  Energy sim ratio:         {cat_dsim/max(no_dsim, 0.001):.3f}")

    if cat_dsim > no_dsim * 1.1:
        print(f"  Catalog pairs MORE energy-similar than non-catalog — recognized structure")
    elif no_dsim > cat_dsim * 1.1:
        print(f"  Non-catalog pairs MORE energy-similar — 'harm' angles processed more uniformly")
    else:
        print(f"  No significant difference in energy processing")

    results["liu_hai"] = {
        "catalog_n": len(catalog_pairs), "catalog_mrl": round(cat_mrl, 4), "catalog_dsim": round(cat_dsim, 4),
        "no_match_n": len(no_match_pairs), "no_match_mrl": round(no_mrl, 4), "no_match_dsim": round(no_dsim, 4),
    }

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Wu He (180 grid-1 vs grid-2):  g1={g1_avg:.4f} g2={g2_avg:.4f} — {'DIFFERENT' if abs(g1_avg/max(g2_avg,0.001) - 1) > 0.15 else 'SIMILAR'}")
    print(f"  San Hui Fang (adjacency):       {avg_adj/max(avg_non,0.001):.3f}x — {'CONFIRMED' if avg_adj/max(avg_non,0.001) > 1.1 else 'NULL'}")
    print(f"  Xiang Xing (self-punishment):   {len(self_conflict)} self-damped tokens")
    print(f"  Liu Hai (harm angles):          dsim ratio {cat_dsim/max(no_dsim,0.001):.3f} — {'DIFFERENT' if abs(cat_dsim/max(no_dsim,0.001) - 1) > 0.1 else 'SIMILAR'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON: {args.output}")


if __name__ == "__main__":
    main()
