#!/usr/bin/env python3
"""Analyse wave-engine training logs. Reads training_log.jsonl or console output."""

import json
import sys
import math
import re
import argparse


def load_jsonl(path):
    """Load training_log.jsonl entries."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_console_log(path):
    """Parse console output (grep-friendly format) as fallback."""
    entries = []
    with open(path) as f:
        for line in f:
            # Match: "    50    10.4696       4.5s  lr=0.000051  vram=2737MB"
            m = re.match(r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)s\s+lr=([\d.]+)', line)
            if m:
                entries.append({
                    'iter': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'time_s': float(m.group(3)),
                    'lr': float(m.group(4)),
                })
                continue
            # Match CPU format: "    50     2.8576    512.4ms  gnorm=1.00"
            m = re.match(r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)ms\s+gnorm', line)
            if m:
                entries.append({
                    'iter': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'time_s': float(m.group(3)) / 1000,
                })
    return entries


def analyze(entries, corpus_tokens=None, batch_size=4, seq_len=256):
    if not entries:
        print("No data found.")
        return

    n = len(entries)
    losses = [e['loss'] for e in entries]
    iters = [e.get('iter', i) for i, e in enumerate(entries)]
    times = [e.get('time_s', e.get('time_ms', 0) / 1000 if 'time_ms' in e else 0) for e in entries]

    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    best_iter = iters[best_idx]
    avg_time = sum(times) / len(times) if times else 0
    total_time_s = sum(times)

    print(f"\n{'='*55}")
    print(f"  TRAINING ANALYSIS")
    print(f"{'='*55}")
    print(f"  Iterations: {n} (iter {iters[0]} to {iters[-1]})")
    hours = total_time_s / 3600
    print(f"  Total time: {hours:.1f}h ({avg_time:.1f}s/iter avg)")
    print(f"  Start loss: {losses[0]:.4f}")
    print(f"  Best loss:  {best_loss:.4f} (iter {best_iter})")
    print(f"  Final loss: {losses[-1]:.4f}")

    # Rolling average in windows
    window = min(200, n // 5) if n > 10 else n
    if window > 0:
        print(f"\n--- Rolling Average (window={window}) ---")
        windows = []
        for start in range(0, n, window):
            end = min(start + window, n)
            chunk = losses[start:end]
            avg = sum(chunk) / len(chunk)
            best_w = min(chunk)
            iter_start = iters[start]
            iter_end = iters[end - 1]
            windows.append((iter_start, iter_end, avg, best_w))
            print(f"  iter {iter_start:5d}-{iter_end:5d}:  avg={avg:.4f}  best={best_w:.4f}")

    # Descent rate per 500 iters
    block = 500
    if n > block:
        print(f"\n--- Descent Rate (per {block} iters) ---")
        prev_avg = None
        for start in range(0, n, block):
            end = min(start + block, n)
            chunk = losses[start:end]
            avg = sum(chunk) / len(chunk)
            if prev_avg is not None:
                delta = avg - prev_avg
                direction = "descending" if delta < -0.05 else "rising" if delta > 0.05 else "flat"
                print(f"  iter {iters[start]:5d}-{iters[end-1]:5d}:  "
                      f"avg={avg:.2f}  delta={delta:+.3f}  ({direction})")
            else:
                print(f"  iter {iters[start]:5d}-{iters[end-1]:5d}:  avg={avg:.2f}  (baseline)")
            prev_avg = avg

    # Corpus passes
    tokens_per_iter = batch_size * seq_len
    total_tokens = n * tokens_per_iter
    if corpus_tokens:
        passes = total_tokens / corpus_tokens
        print(f"\n--- Corpus Coverage ---")
        print(f"  Tokens per iter: {tokens_per_iter:,}")
        print(f"  Total tokens seen: {total_tokens:,}")
        print(f"  Corpus size: {corpus_tokens:,}")
        print(f"  Passes: {passes:.2f}")

    # Stability (last 20% of training)
    tail_start = int(n * 0.8)
    tail = losses[tail_start:]
    if len(tail) > 2:
        tail_avg = sum(tail) / len(tail)
        tail_std = (sum((x - tail_avg) ** 2 for x in tail) / len(tail)) ** 0.5
        tail_min = min(tail)
        tail_max = max(tail)
        print(f"\n--- Stability (last 20%) ---")
        print(f"  Mean: {tail_avg:.4f}")
        print(f"  Std:  {tail_std:.4f}")
        print(f"  Range: {tail_min:.4f} - {tail_max:.4f}")

    # Diagnosis
    print(f"\n--- Diagnosis ---")
    # Check if last 30% is flat
    third = n // 3
    last_third = losses[-third:] if third > 0 else losses
    first_third = losses[:third] if third > 0 else losses
    last_avg = sum(last_third) / len(last_third)
    first_avg = sum(first_third) / len(first_third)

    if last_avg < first_avg * 0.7:
        print(f"  Status: DESCENDING (still learning)")
        print(f"  Recommendation: Continue training")
    elif last_avg < first_avg * 0.95:
        print(f"  Status: SLOWING (diminishing returns)")
        if corpus_tokens and passes < 3:
            print(f"  Recommendation: Continue — only {passes:.1f} passes, model needs more exposure")
        else:
            print(f"  Recommendation: Consider adding more data or increasing capacity")
    else:
        print(f"  Status: PLATEAU")
        lrs = [e.get('lr', None) for e in entries[-10:]]
        lrs = [l for l in lrs if l is not None]
        if lrs and max(lrs) < 2e-5:
            print(f"  Likely cause: LR decayed too fast (final lr={lrs[-1]:.6f})")
            print(f"  Fix: Resume with higher min_lr_ratio (0.3 instead of 0.1)")
        else:
            print(f"  Recommendation: Model may need more capacity or different data")

    print(f"\n  Best checkpoint: iter {best_iter} (loss {best_loss:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyse wave-engine training logs")
    parser.add_argument("log_file", help="training_log.jsonl or console log")
    parser.add_argument("--corpus-tokens", type=int, default=None,
                        help="Total tokens in corpus (for pass counting)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    args = parser.parse_args()

    # Try JSONL first, fall back to console log parsing
    entries = load_jsonl(args.log_file)
    if not entries:
        entries = load_console_log(args.log_file)

    analyze(entries, args.corpus_tokens, args.batch, args.seq)


if __name__ == "__main__":
    main()
