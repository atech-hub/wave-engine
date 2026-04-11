#!/usr/bin/env python3
"""Context/Dignity Probe — does a token's energy signature change with context?

The catalog's "dignity" concept (Part 5.1): same entity, different strength
per domain. Maps to: does the same token get processed differently by the ODE
depending on which other tokens are in the context window?

Test: encode a focus token in various context strings, compare energy
deformation vectors. High similarity = context-independent (no dignity).
Low similarity = context-dependent (dignity in action).

Uses the wave-engine's --encode mode to get per-layer cosine and decoder
readout for each context.

Usage:
    python scripts/context_dignity_probe.py --resume <checkpoint> [options]

Requires wave-engine binary in target/release/ or target/debug/.
"""

import subprocess
import sys
import json
import math
import os
from collections import defaultdict


def run_encode(engine_path, checkpoint, data_path, text, n_bands=84, n_layers=4, alpha=0.1, beta=0.2):
    """Run wave-engine --encode and parse output."""
    cmd = [
        engine_path, "--encode", text,
        "--resume", checkpoint,
        "--layers", str(n_layers),
        "--n-bands", str(n_bands),
        "--n-head", "4",
        "--out-proj-groups", "1",
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--data", data_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Parse cosine per layer
    cosines = {}
    for line in result.stdout.split("\n"):
        if "cos(input, output) per layer:" in line:
            parts = line.split("L")
            for p in parts[1:]:
                if ":" in p:
                    layer_str, val_str = p.split(":")
                    try:
                        cosines[int(layer_str.strip())] = float(val_str.strip())
                    except ValueError:
                        pass

    # Parse decoder readout
    lm_head_top = []
    phase_top = []
    for line in result.stdout.split("\n"):
        if "lm_head top" in line:
            # Extract token-score pairs
            parts = line.split(")")
            for p in parts[:-1]:
                if "(" in p:
                    score_str = p.split("(")[-1].strip()
                    tok = p.split("(")[0].strip().split()[-1]
                    try:
                        lm_head_top.append((tok, float(score_str)))
                    except ValueError:
                        pass
        if "phase-native top" in line:
            parts = line.split(")")
            for p in parts[:-1]:
                if "(" in p:
                    score_str = p.split("(")[-1].strip()
                    tok = p.split("(")[0].strip().split()[-1]
                    try:
                        phase_top.append((tok, float(score_str)))
                    except ValueError:
                        pass

    return {
        "text": text,
        "cosines": cosines,
        "lm_head_top3": lm_head_top[:3],
        "phase_top3": phase_top[:3],
        "raw_stdout": result.stdout,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Context/Dignity Probe")
    parser.add_argument("--resume", required=True, help="Checkpoint path")
    parser.add_argument("--data", default="data/grammar_lesson_1.txt", help="Data file for char vocab")
    parser.add_argument("--n-bands", type=int, default=84)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--engine", default=None, help="Path to wave-engine binary")
    parser.add_argument("--output", default=None, help="Output JSON")
    args = parser.parse_args()

    # Find engine binary
    engine = args.engine
    if engine is None:
        for candidate in ["target/release/wave-engine", "target/release/wave-engine.exe",
                         "target/debug/wave-engine", "target/debug/wave-engine.exe"]:
            if os.path.exists(candidate):
                engine = candidate
                break
    if engine is None:
        print("ERROR: wave-engine binary not found. Build with cargo build or specify --engine")
        sys.exit(1)

    print(f"Engine: {engine}")
    print(f"Checkpoint: {args.resume}")
    print()

    # Define test cases: focus token in various contexts
    # Each group tests ONE token in different contexts
    test_groups = {
        "s": {
            "description": "'s' — plural/verb marker, most phase-distinctive token",
            "contexts": [
                "s",           # alone
                "is",          # verb ending
                "as",          # preposition ending
                "cats",        # plural
                "his",         # possessive context
                "s.",          # sentence-end
                "st",          # consonant cluster
            ],
        },
        "e": {
            "description": "'e' — common letter, generic in both phase and energy",
            "contexts": [
                "e",           # alone
                "the",         # function word
                "are",         # verb ending
                "sentence",    # mid-word
                "e.",          # sentence-end
                "he",          # pronoun
                "en",          # common bigram
            ],
        },
        "a": {
            "description": "'a' — article/common vowel",
            "contexts": [
                "a",           # alone (also article)
                "an",          # article
                "at",          # preposition
                "was",         # past tense context
                "a.",          # sentence-end
                "na",          # reversed common pair
                "ba",          # rare bigram
            ],
        },
        ".": {
            "description": "'.' — punctuation, energy-distinctive",
            "contexts": [
                ".",           # alone
                ".A",          # sentence start (capital follows)
                "t.",          # sentence end
                "..",          # double period
                "n.",          # after consonant
                "e.",          # after vowel
            ],
        },
    }

    all_results = {}

    for focus_tok, group in test_groups.items():
        print(f"=== {group['description']} ===")
        group_results = []

        for ctx in group["contexts"]:
            result = run_encode(engine, args.resume, args.data, ctx,
                              args.n_bands, args.n_layers, args.alpha, args.beta)
            group_results.append(result)
            cos_str = "  ".join(f"L{k}:{v:.2f}" for k, v in sorted(result["cosines"].items()))
            print(f"  \"{ctx:>10s}\"  {cos_str}")

        # Compare cosines: how much does the per-layer processing change with context?
        if len(group_results) >= 2:
            # Compare each context against the solo (first) result
            solo = group_results[0]
            print(f"\n  Dignity (deviation from solo '{focus_tok}'):")
            for r in group_results[1:]:
                diffs = []
                for layer in sorted(solo["cosines"].keys()):
                    if layer in r["cosines"]:
                        d = abs(r["cosines"][layer] - solo["cosines"][layer])
                        diffs.append(d)
                avg_diff = sum(diffs) / len(diffs) if diffs else 0
                max_diff = max(diffs) if diffs else 0
                max_layer = diffs.index(max_diff) if diffs else 0
                print(f"    \"{r['text']:>10s}\"  avg_shift={avg_diff:.3f}  max_shift={max_diff:.3f} (L{max_layer})")

        print()
        all_results[focus_tok] = {
            "description": group["description"],
            "contexts": [{
                "text": r["text"],
                "cosines": r["cosines"],
                "lm_head_top3": r["lm_head_top3"],
                "phase_top3": r["phase_top3"],
            } for r in group_results],
        }

    # Summary
    print("=== DIGNITY SUMMARY ===")
    print("Token  Solo_cos(L3)  Max_context_shift  Most_shifted_context")
    print("-" * 65)
    for focus_tok, data in all_results.items():
        contexts = data["contexts"]
        if len(contexts) < 2:
            continue
        solo_l3 = contexts[0]["cosines"].get(3, 0)
        max_shift = 0
        max_ctx = ""
        for c in contexts[1:]:
            shift = abs(c["cosines"].get(3, 0) - solo_l3)
            if shift > max_shift:
                max_shift = shift
                max_ctx = c["text"]
        print(f"  '{focus_tok}'      {solo_l3:.3f}          {max_shift:.3f}              \"{max_ctx}\"")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON written to: {args.output}")


if __name__ == "__main__":
    main()
