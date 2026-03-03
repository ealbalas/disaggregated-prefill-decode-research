#!/usr/bin/env python3
# utils/plot_results.py
#
# Reads all benchmark JSON files from $OUTDIR, parses the tag metadata,
# and generates plots grouped by sweep dimension.
#
# Usage:
#   python utils/plot_results.py
#   python utils/plot_results.py --results-dir /path/to/results --plots-dir /path/to/plots

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
parser.add_argument("--results-dir", default=f"{scratch}/disaggregated-prefill-decode-research/results")
parser.add_argument("--plots-dir",   default=f"{scratch}/disaggregated-prefill-decode-research/results/plots")
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR   = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "mean_ttft_ms":  "Mean TTFT (ms)",
    "mean_tpot_ms":  "Mean TPOT (ms)",
    "mean_itl_ms":   "Mean ITL (ms)",
    "mean_e2el_ms":  "Mean E2E Latency (ms)",
    "request_throughput": "Throughput (req/s)",
}

RATIOS = ["1p1d", "1p2d", "1p3d", "2p1d", "3p1d"]

# ── Load results ──────────────────────────────────────────────────────────────
def load_results():
    records = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue

        # Parse tag: {ratio}_{input}x{output}_rate{rate}_conc{concurrency}
        tag = data.get("metadata", {}).get("tag", f.stem)
        m = re.match(r"(\d+p\d+d)_(\d+)x(\d+)_rate([\d.]+)_conc(\d+)", tag)
        if not m:
            print(f"Skipping unrecognised tag: {tag}")
            continue

        records.append({
            "tag":         tag,
            "ratio":       m.group(1),
            "input_len":   int(m.group(2)),
            "output_len":  int(m.group(3)),
            "rate":        float(m.group(4)),
            "concurrency": int(m.group(5)),
            **{k: data.get(k) for k in METRICS},
        })

    print(f"Loaded {len(records)} result files.")
    return records

# ── Generic line plot ─────────────────────────────────────────────────────────
def line_plot(records, x_key, x_label, metric_key, metric_label,
              group_key, group_label, filter_fn, filename):
    grouped = defaultdict(list)
    for r in records:
        if filter_fn(r) and r.get(metric_key) is not None:
            grouped[r[group_key]].append((r[x_key], r[metric_key]))

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.tab10(np.linspace(0, 1, max(len(grouped), 1)))

    for (group_val, color) in zip(sorted(grouped), colors):
        pts = sorted(grouped[group_val])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"{group_label}={group_val}", color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {x_label}\ngrouped by {group_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")

# ── Plot families ─────────────────────────────────────────────────────────────
def plots_vs_request_rate(records):
    """Metric vs request rate, one line per ratio. Fixed input/output to median."""
    input_lens  = sorted({r["input_len"]  for r in records})
    output_lens = sorted({r["output_len"] for r in records})
    mid_in  = input_lens[len(input_lens) // 2]
    mid_out = output_lens[len(output_lens) // 2]

    for metric_key, metric_label in METRICS.items():
        line_plot(
            records, "rate", "Request Rate (req/s)", metric_key, metric_label,
            "ratio", "Ratio",
            lambda r: r["input_len"] == mid_in and r["output_len"] == mid_out,
            f"rate_vs_{metric_key}_by_ratio_{mid_in}x{mid_out}.png",
        )

def plots_vs_input_len(records):
    """Metric vs input length, one line per ratio. Fixed rate/concurrency to median."""
    rates  = sorted({r["rate"]        for r in records})
    concs  = sorted({r["concurrency"] for r in records})
    mid_r  = rates[len(rates) // 2]
    mid_c  = concs[len(concs) // 2]
    mid_out = sorted({r["output_len"] for r in records})[0]

    for metric_key, metric_label in METRICS.items():
        line_plot(
            records, "input_len", "Input Length (tokens)", metric_key, metric_label,
            "ratio", "Ratio",
            lambda r: r["rate"] == mid_r and r["concurrency"] == mid_c and r["output_len"] == mid_out,
            f"input_len_vs_{metric_key}_by_ratio.png",
        )

def plots_vs_output_len(records):
    """Metric vs output length, one line per ratio."""
    rates   = sorted({r["rate"]        for r in records})
    concs   = sorted({r["concurrency"] for r in records})
    mid_r   = rates[len(rates) // 2]
    mid_c   = concs[len(concs) // 2]
    mid_in  = sorted({r["input_len"]   for r in records})[0]

    for metric_key, metric_label in METRICS.items():
        line_plot(
            records, "output_len", "Output Length (tokens)", metric_key, metric_label,
            "ratio", "Ratio",
            lambda r: r["rate"] == mid_r and r["concurrency"] == mid_c and r["input_len"] == mid_in,
            f"output_len_vs_{metric_key}_by_ratio.png",
        )

def plots_vs_concurrency(records):
    """Metric vs concurrency, one line per ratio."""
    rates   = sorted({r["rate"]      for r in records})
    mid_r   = rates[len(rates) // 2]
    mid_in  = sorted({r["input_len"]  for r in records})[len(sorted({r["input_len"]  for r in records})) // 2]
    mid_out = sorted({r["output_len"] for r in records})[len(sorted({r["output_len"] for r in records})) // 2]

    for metric_key, metric_label in METRICS.items():
        line_plot(
            records, "concurrency", "Max Concurrency", metric_key, metric_label,
            "ratio", "Ratio",
            lambda r: r["rate"] == mid_r and r["input_len"] == mid_in and r["output_len"] == mid_out,
            f"concurrency_vs_{metric_key}_by_ratio.png",
        )

def ratio_comparison_bar(records):
    """Bar chart: mean of each metric per ratio, at median settings."""
    rates   = sorted({r["rate"]        for r in records})
    concs   = sorted({r["concurrency"] for r in records})
    ins     = sorted({r["input_len"]   for r in records})
    outs    = sorted({r["output_len"]  for r in records})
    mid_r, mid_c, mid_in, mid_out = (
        rates[len(rates)//2], concs[len(concs)//2],
        ins[len(ins)//2],     outs[len(outs)//2],
    )

    subset = [r for r in records if
              r["rate"] == mid_r and r["concurrency"] == mid_c
              and r["input_len"] == mid_in and r["output_len"] == mid_out]

    if not subset:
        return

    for metric_key, metric_label in METRICS.items():
        ratio_vals = {r["ratio"]: r[metric_key] for r in subset if r.get(metric_key) is not None}
        if not ratio_vals:
            continue

        ratios = sorted(ratio_vals, key=lambda x: RATIOS.index(x) if x in RATIOS else 99)
        vals   = [ratio_vals[r] for r in ratios]
        colors = cm.tab10(np.linspace(0, 1, len(ratios)))

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(ratios, vals, color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_xlabel("P:D Ratio")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Ratio\n(input={mid_in}, output={mid_out}, rate={mid_r}, conc={mid_c})")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out = PLOTS_DIR / f"ratio_bar_{metric_key}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out.name}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib...")
        os.system("pip install matplotlib --break-system-packages -q")
        import matplotlib.pyplot as plt

    records = load_results()
    if not records:
        print("No results found. Run the sweep first.")
        raise SystemExit(1)

    print(f"\nGenerating plots in {PLOTS_DIR} ...")
    plots_vs_request_rate(records)
    plots_vs_input_len(records)
    plots_vs_output_len(records)
    plots_vs_concurrency(records)
    ratio_comparison_bar(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
