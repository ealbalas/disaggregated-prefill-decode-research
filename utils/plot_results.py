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

RATIOS = ["colocated", "1p1d", "1p2d", "1p3d", "2p1d", "3p1d"]
COMPARE_RATIOS = ["colocated", "1p1d", "2p1d", "3p1d"]
COMPARE_COLORS = {r: cm.tab10(i / (len(COMPARE_RATIOS) - 1)) for i, r in enumerate(COMPARE_RATIOS)}

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
        # ratio is either NpMd (e.g. 1p1d) or "colocated"
        tag = data.get("metadata", {}).get("tag", f.stem)
        m = re.match(r"(colocated|\d+p\d+d)_(\d+)x(\d+)_rate([\d.]+)_conc(\d+)", tag)
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
            "goodput":     data.get("goodput"),
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

# ── Colocated vs Disaggregated comparison plots ───────────────────────────────

def _compare_subplots(records, x_key, x_label, metric_key, metric_label, filename):
    """One subplot per request rate, one line per COMPARE_RATIOS ratio."""
    rates = sorted({r["rate"] for r in records if r["ratio"] in COMPARE_RATIOS})
    if not rates:
        return

    fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 5), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    for ax, rate in zip(axes, rates):
        for ratio in COMPARE_RATIOS:
            pts = sorted(
                (r[x_key], r[metric_key])
                for r in records
                if r["ratio"] == ratio and r["rate"] == rate
                and r.get(metric_key) is not None
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=ratio, color=COMPARE_COLORS[ratio])

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f"rate={rate} req/s")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{metric_label} vs {x_label}\nColocated vs Disaggregated", y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def compare_latency_vs_input_len(records):
    """TTFT, E2EL, and TPOT vs input length, one subplot per rate."""
    _compare_subplots(records, "input_len", "Input Length (tokens)",
                      "mean_ttft_ms", "Mean TTFT (ms)",
                      "compare_ttft_vs_input_len.png")
    _compare_subplots(records, "input_len", "Input Length (tokens)",
                      "mean_e2el_ms", "Mean E2E Latency (ms)",
                      "compare_e2el_vs_input_len.png")
    _compare_subplots(records, "input_len", "Input Length (tokens)",
                      "mean_tpot_ms", "Mean TPOT (ms)",
                      "compare_tpot_vs_input_len.png")


def compare_throughput_vs_input_len(records):
    """Throughput vs input length, one subplot per rate."""
    _compare_subplots(records, "input_len", "Input Length (tokens)",
                      "request_throughput", "Throughput (req/s)",
                      "compare_throughput_vs_input_len.png")


def compare_ttft_speedup_vs_input_len(records):
    """Normalized TTFT speedup (colocated / disagg) vs input length.
    Values > 1 mean disaggregation is faster on TTFT."""
    rates = sorted({r["rate"] for r in records if r["ratio"] in COMPARE_RATIOS})
    if not rates:
        return

    # Build colocated lookup: (input_len, rate) -> mean_ttft_ms
    coloc = {
        (r["input_len"], r["rate"]): r["mean_ttft_ms"]
        for r in records
        if r["ratio"] == "colocated" and r.get("mean_ttft_ms") is not None
    }
    if not coloc:
        return

    disagg_ratios = [r for r in COMPARE_RATIOS if r != "colocated"]
    fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 5), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    for ax, rate in zip(axes, rates):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="break-even")
        for ratio in disagg_ratios:
            pts = sorted(
                (r["input_len"], coloc[(r["input_len"], rate)] / r["mean_ttft_ms"])
                for r in records
                if r["ratio"] == ratio and r["rate"] == rate
                and r.get("mean_ttft_ms") is not None
                and (r["input_len"], rate) in coloc
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=ratio, color=COMPARE_COLORS[ratio])

        ax.set_xlabel("Input Length (tokens)")
        ax.set_ylabel("TTFT Speedup (colocated / disagg)")
        ax.set_title(f"rate={rate} req/s")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("TTFT Speedup of Disaggregated vs Colocated\n(>1 = disaggregation wins)", y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "compare_ttft_speedup_vs_input_len.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def compare_tpot_overhead_by_ratio(records):
    """Bar chart of TPOT per ratio at median input length, per rate.
    Reveals whether KV transfer adds decode overhead."""
    ins = sorted({r["input_len"] for r in records if r["ratio"] in COMPARE_RATIOS})
    if not ins:
        return
    mid_in = ins[len(ins) // 2]

    rates = sorted({r["rate"] for r in records if r["ratio"] in COMPARE_RATIOS})
    fig, axes = plt.subplots(1, len(rates), figsize=(5 * len(rates), 5), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    for ax, rate in zip(axes, rates):
        ratio_vals = {
            r["ratio"]: r["mean_tpot_ms"]
            for r in records
            if r["ratio"] in COMPARE_RATIOS and r["rate"] == rate
            and r["input_len"] == mid_in and r.get("mean_tpot_ms") is not None
        }
        if not ratio_vals:
            continue
        ratios = [r for r in COMPARE_RATIOS if r in ratio_vals]
        vals = [ratio_vals[r] for r in ratios]
        colors = [COMPARE_COLORS[r] for r in ratios]
        bars = ax.bar(ratios, vals, color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_xlabel("P:D Ratio")
        ax.set_ylabel("Mean TPOT (ms)")
        ax.set_title(f"rate={rate} req/s")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"TPOT by Ratio (input={mid_in} tokens)\nDecode overhead from KV transfer", y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "compare_tpot_overhead_by_ratio.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def compare_throughput_vs_rate(records):
    """Throughput vs request rate, one line per ratio, fixed to median input length."""
    ins = sorted({r["input_len"] for r in records if r["ratio"] in COMPARE_RATIOS})
    if not ins:
        return
    mid_in = ins[len(ins) // 2]

    fig, ax = plt.subplots(figsize=(7, 5))
    for ratio in COMPARE_RATIOS:
        pts = sorted(
            (r["rate"], r["request_throughput"])
            for r in records
            if r["ratio"] == ratio and r["input_len"] == mid_in
            and r.get("request_throughput") is not None
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=ratio, color=COMPARE_COLORS[ratio])

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title(f"Throughput vs Request Rate (input={mid_in} tokens)\nColocated vs Disaggregated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "compare_throughput_vs_rate.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


def compare_goodput_vs_input_len(records):
    """Goodput vs input length, one subplot per rate.
    Goodput = requests/s that meet all SLOs (TTFT, TPOT, E2EL)."""
    has_goodput = [r for r in records if r["ratio"] in COMPARE_RATIOS and r.get("goodput") is not None]
    if not has_goodput:
        print("  Skipping goodput plot: no goodput data found in results.")
        return

    _compare_subplots(has_goodput, "input_len", "Input Length (tokens)",
                      "goodput", "Goodput (req/s meeting SLOs)",
                      "compare_goodput_vs_input_len.png")


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

    print("\nGenerating colocated vs disaggregated comparison plots...")
    compare_latency_vs_input_len(records)
    compare_throughput_vs_input_len(records)
    compare_ttft_speedup_vs_input_len(records)
    compare_tpot_overhead_by_ratio(records)
    compare_throughput_vs_rate(records)
    compare_goodput_vs_input_len(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
