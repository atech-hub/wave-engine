#!/usr/bin/env python3
"""Galaxy scan summary — compact readable output from galaxy_map.json.

Usage:
    python scripts/summarize_galaxy.py <galaxy_dir>
    python scripts/summarize_galaxy.py --compare <dir_a> <dir_b>

Produces galaxy_summary.json + galaxy_summary.md (single scan)
or galaxy_diff.json + galaxy_diff.md (compare mode).

Standard library only — no numpy, no pandas.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def load_galaxy(path):
    p = Path(path)
    if p.is_dir():
        p = p / "galaxy_map.json"
    with open(p) as f:
        return json.load(f), str(p.parent)

def summarize_layer(layer_data, top_k=20, fwm_threshold=0.5):
    bands = layer_data.get("bands", [])
    top_pairs = layer_data.get("top_pairs", [])
    triads = layer_data.get("triads", [])
    quartets = layer_data.get("fwm_quartets", [])
    summary = layer_data.get("summary", {})

    # Band statistics
    n = len(bands)
    near_boundary = sum(1 for b in bands if b.get("cv", 1) < 0.2)
    near_center = sum(1 for b in bands if b.get("mag", 0) < 0.3)
    cvs = [b.get("cv", 0.5) for b in bands]
    mean_cv = sum(cvs) / max(len(cvs), 1)

    # CV histogram (10 bins from 0 to 1)
    cv_hist = [0] * 10
    for cv in cvs:
        idx = min(int(cv * 10), 9)
        cv_hist[idx] += 1

    # Relationship counts by catalog type — from top_pairs (subset of full matrix)
    # The full matrix counts are in summary.significant_by_type but that field
    # uses a different format. Count from what we have in the JSON.
    cat_counts = {}
    no_match = 0
    for p in top_pairs:
        cat = p.get("cat", "none")
        if cat == "none":
            no_match += 1
        else:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    # Note: these counts are from top-100 pairs only, not the full 3486

    # Grid distribution from summary
    grid = summary.get("grid", {})

    # Interesting FWM quartets
    high_quartets = [q for q in quartets if q.get("coh", 0) >= fwm_threshold]
    high_quartets.sort(key=lambda q: q.get("coh", 0), reverse=True)

    # Top pairs (truncate to top_k)
    top_k_pairs = top_pairs[:top_k]

    return {
        "layer": layer_data.get("layer", 0),
        "band_stats": {
            "count": n,
            "near_boundary": near_boundary,
            "near_center": near_center,
            "mean_cv": round(mean_cv, 3),
            "cv_histogram": cv_hist,
        },
        "relationships": cat_counts,
        "relationships_no_match": no_match,
        "grid": {
            "g1": round(grid.get("g1", 0), 3),
            "g2": round(grid.get("g2", 0), 3),
            "comp": round(grid.get("comp", 0), 3),
            "approx": round(grid.get("approx", 0), 3),
        },
        "top_pairs": [{
            "bands": [p.get("a", 0), p.get("b", 0)],
            "dist_deg": round(p.get("dist_deg", 0), 1),
            "peak_n": p.get("peak_n", 0),
            "peak_str": round(p.get("peak_str", 0), 3),
            "cat": p.get("cat", "none"),
            "grid": p.get("grid", "?"),
        } for p in top_k_pairs],
        "triads": triads,
        "fwm_quartets": {
            "total": len(quartets),
            "above_threshold": len(high_quartets),
            "threshold": fwm_threshold,
            "top_50": high_quartets[:50],
        },
        "summary": {
            "pairs": summary.get("pairs", 0),
            "triads": summary.get("triads", 0),
            "quartets": summary.get("quartets", 0),
            "fill": round(summary.get("fill", 0), 3),
            "center": round(summary.get("center", 0), 3),
        },
    }

def summarize_scan(data, top_k=20, fwm_threshold=0.5):
    layers = data.get("layers", [])
    result = {
        "schema_version": "1.0",
        "summary_generated_at": datetime.now().isoformat(),
        "metadata": {
            "n_layers": data.get("n_layers", len(layers)),
            "n_bands": data.get("n_bands", 0),
            "n_positions": data.get("n_positions", 0),
            "agc_ceiling": data.get("agc_ceiling", 0),
            "grids": data.get("grids", {}),
        },
        "layers": [summarize_layer(l, top_k, fwm_threshold) for l in layers],
    }

    # Global summary
    total_triads = sum(l["summary"]["triads"] for l in result["layers"])
    total_quartets_ht = sum(l["fwm_quartets"]["above_threshold"] for l in result["layers"])
    result["global"] = {
        "total_triads": total_triads,
        "total_fwm_above_threshold": total_quartets_ht,
    }
    return result

def to_markdown(summary, source_path=""):
    lines = []
    lines.append(f"# Galaxy Scan Summary")
    lines.append(f"**Source:** {source_path}")
    lines.append(f"**Generated:** {summary.get('summary_generated_at', '?')}")
    meta = summary.get("metadata", {})
    lines.append(f"**Config:** {meta.get('n_bands', '?')} bands, {meta.get('n_layers', '?')} layers, {meta.get('n_positions', '?')} positions")
    grids = meta.get("grids", {})
    lines.append(f"**Grids:** m1={grids.get('m1', '?')}, m2={grids.get('m2', '?')}")
    lines.append(f"**AGC ceiling:** {meta.get('agc_ceiling', '?')}")
    lines.append("")

    g = summary.get("global", {})
    lines.append(f"## Global: {g.get('total_triads', 0)} triads, {g.get('total_fwm_above_threshold', 0)} FWM quartets above threshold")
    lines.append("")

    for layer in summary.get("layers", []):
        li = layer["layer"]
        lines.append(f"## Layer {li}")
        bs = layer["band_stats"]
        lines.append(f"- Bands: {bs['count']}, near-boundary: {bs['near_boundary']}, near-center: {bs['near_center']}, mean CV: {bs['mean_cv']}")
        s = layer["summary"]
        lines.append(f"- Pairs: {s['pairs']}, triads: {s['triads']}, FWM quartets: {s['quartets']}")
        lines.append(f"- Sphere: fill={s['fill']}, center={s['center']}")

        # Relationships
        rels = layer.get("relationships", {})
        if rels:
            rel_strs = [f"{k}={v}" for k, v in sorted(rels.items(), key=lambda x: -x[1])]
            lines.append(f"- Catalog matches: {', '.join(rel_strs)}")

        # Grid
        g = layer["grid"]
        lines.append(f"- Grid: g1={g['g1']:.1%} g2={g['g2']:.1%} comp={g['comp']:.1%} approx={g['approx']:.1%}")

        # Top pairs
        lines.append(f"### Top {len(layer['top_pairs'])} pairs")
        lines.append("| Bands | Dist | Peak n | Strength | Catalog | Grid |")
        lines.append("|-------|------|--------|----------|---------|------|")
        for p in layer["top_pairs"]:
            lines.append(f"| {p['bands'][0]},{p['bands'][1]} | {p['dist_deg']}d | n={p['peak_n']} | {p['peak_str']} | {p['cat']} | {p['grid']} |")

        # Triads
        if layer["triads"]:
            lines.append(f"### Triads ({len(layer['triads'])})")
            for t in layer["triads"]:
                lines.append(f"- Bands {t.get('bands', '?')}, coherence={t.get('coh', '?')}")

        # FWM quartets
        fq = layer["fwm_quartets"]
        lines.append(f"### FWM quartets: {fq['above_threshold']}/{fq['total']} above {fq['threshold']} threshold")
        for q in fq["top_50"][:10]:
            lines.append(f"- Bands {q.get('bands', '?')}, sum={q.get('sum', '?')}, coh={q.get('coh', '?')}")

        lines.append("")

    return "\n".join(lines)

def compare_scans(sum_a, sum_b, source_a, source_b):
    """Compare two scan summaries and produce a diff."""
    warnings = []

    # Check confounds
    meta_a = sum_a.get("metadata", {})
    meta_b = sum_b.get("metadata", {})
    if meta_a.get("n_bands") != meta_b.get("n_bands"):
        warnings.append(f"ARCHITECTURE MISMATCH: n_bands {meta_a.get('n_bands')} vs {meta_b.get('n_bands')}")
    if meta_a.get("n_layers") != meta_b.get("n_layers"):
        warnings.append(f"ARCHITECTURE MISMATCH: n_layers {meta_a.get('n_layers')} vs {meta_b.get('n_layers')}")

    diff = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "source_a": source_a,
        "source_b": source_b,
        "warnings": warnings,
        "per_layer": [],
    }

    layers_a = sum_a.get("layers", [])
    layers_b = sum_b.get("layers", [])
    n_layers = min(len(layers_a), len(layers_b))

    for li in range(n_layers):
        la = layers_a[li]
        lb = layers_b[li]
        sa = la["summary"]
        sb = lb["summary"]
        layer_diff = {
            "layer": li,
            "triads": f"{sa['triads']} -> {sb['triads']} ({sb['triads'] - sa['triads']:+d})",
            "quartets": f"{sa['quartets']} -> {sb['quartets']} ({sb['quartets'] - sa['quartets']:+d})",
            "fill": f"{sa['fill']} -> {sb['fill']} ({sb['fill'] - sa['fill']:+.3f})",
            "grid_a": la["grid"],
            "grid_b": lb["grid"],
            "fwm_above_threshold_a": la["fwm_quartets"]["above_threshold"],
            "fwm_above_threshold_b": lb["fwm_quartets"]["above_threshold"],
        }

        # Relationship type diffs
        all_types = set(la.get("relationships", {}).keys()) | set(lb.get("relationships", {}).keys())
        rel_diffs = {}
        for t in sorted(all_types):
            va = la.get("relationships", {}).get(t, 0)
            vb = lb.get("relationships", {}).get(t, 0)
            if va != vb:
                rel_diffs[t] = f"{va} -> {vb} ({vb - va:+d})"
        layer_diff["relationship_changes"] = rel_diffs
        diff["per_layer"].append(layer_diff)

    return diff

def diff_to_markdown(diff):
    lines = []
    lines.append("# Galaxy Scan Comparison")
    lines.append(f"**A:** {diff['source_a']}")
    lines.append(f"**B:** {diff['source_b']}")
    lines.append(f"**Generated:** {diff['generated_at']}")
    lines.append("")

    if diff["warnings"]:
        lines.append("## WARNINGS")
        for w in diff["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    for ld in diff.get("per_layer", []):
        lines.append(f"## Layer {ld['layer']}")
        lines.append(f"- Triads: {ld['triads']}")
        lines.append(f"- FWM quartets: {ld['quartets']}")
        lines.append(f"- Sphere fill: {ld['fill']}")
        lines.append(f"- FWM above threshold: {ld['fwm_above_threshold_a']} vs {ld['fwm_above_threshold_b']}")
        if ld.get("relationship_changes"):
            lines.append("- Relationship changes:")
            for t, v in ld["relationship_changes"].items():
                lines.append(f"  - {t}: {v}")
        lines.append("")

    return "\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize galaxy scan output")
    parser.add_argument("path", nargs="?", help="Galaxy directory or galaxy_map.json")
    parser.add_argument("path_b", nargs="?", help="Second galaxy directory (compare mode)")
    parser.add_argument("--compare", action="store_true", help="Compare two scans")
    parser.add_argument("--top-k", type=int, default=20, help="Top pairs per layer")
    parser.add_argument("--fwm-threshold", type=float, default=0.5, help="FWM coherence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.compare:
        if not args.path or not args.path_b:
            print("Compare mode needs two paths: --compare <dir_a> <dir_b>")
            sys.exit(1)
        data_a, src_a = load_galaxy(args.path)
        data_b, src_b = load_galaxy(args.path_b)
        sum_a = summarize_scan(data_a, args.top_k, args.fwm_threshold)
        sum_b = summarize_scan(data_b, args.top_k, args.fwm_threshold)
        diff = compare_scans(sum_a, sum_b, src_a, src_b)
        out_dir = Path(args.output) if args.output else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "galaxy_diff.json", "w") as f:
            json.dump(diff, f, indent=2)
        with open(out_dir / "galaxy_diff.md", "w") as f:
            f.write(diff_to_markdown(diff))
        print(f"Diff written to {out_dir}/galaxy_diff.json + galaxy_diff.md")
    else:
        if not args.path:
            print("Usage: summarize_galaxy.py <galaxy_dir> [--top-k N] [--fwm-threshold F]")
            sys.exit(1)
        data, src = load_galaxy(args.path)
        summary = summarize_scan(data, args.top_k, args.fwm_threshold)
        out_dir = Path(args.output) if args.output else Path(src)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "galaxy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        md = to_markdown(summary, src)
        with open(out_dir / "galaxy_summary.md", "w") as f:
            f.write(md)
        print(f"Summary written to {out_dir}/galaxy_summary.json + galaxy_summary.md")
        # Print quick stats
        g = summary.get("global", {})
        print(f"  Triads: {g.get('total_triads', 0)}, FWM quartets above threshold: {g.get('total_fwm_above_threshold', 0)}")

if __name__ == "__main__":
    main()
