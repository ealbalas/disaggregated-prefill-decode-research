#!/usr/bin/env python3
# utils/plot_principled_ratio_comparison.py
#
# Plots results from the principled_ratio_comparison experiment.
# Compares all P:D ratios plus colocated at operating points derived from
# the saturation sweep, with per-workload SLOs derived from measured baselines.
#
# Generates three figures:
#   1. Latency (TTFT and E2EL) — 3×2 grid (workloads × rates), grouped bars per ratio
#   2. Goodput (req/s meeting calibrated SLOs) — grouped bars per ratio
#   3. Throughput (req/s) — grouped bars per ratio
#
# Usage:
#   python utils/plot_principled_ratio_comparison.py
#   python utils/plot_principled_ratio_comparison.py --results-dir /path --plots-dir /path

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
_repo_root = Path(__file__).resolve().parent.parent
parser.add_argument(
    "--results-dir",
    default=str(_repo_root / "results" / "principled_ratio_comparison"),
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "principled_ratio_comparison" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Workload archetypes (input_len, output_len) → display label
WORKLOADS = {
    (1024, 64): "prefill-heavy\n(1024×64)",
    (256, 256): "balanced\n(256×256)",
    (64, 256): "decode-leaning\n(64×256)",
}

RATIOS = ["colocated", "1p1d", "1p2d", "1p3d", "2p1d", "3p1d"]
COLORS = {r: cm.tab10(i / max(len(RATIOS) - 1, 1)) for i, r in enumerate(RATIOS)}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    records = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue

        tag = data.get("tag") or data.get("metadata", {}).get("tag", f.stem)
        m = re.match(r"(colocated|\d+p\d+d)_(\d+)x(\d+)_rate([\d.]+)_conc(\d+)", tag)
        if not m:
            continue

        ratio = m.group(1)
        if ratio not in RATIOS:
            continue

        input_len = int(m.group(2))
        output_len = int(m.group(3))
        if (input_len, output_len) not in WORKLOADS:
            continue

        records.append(
            {
                "tag": tag,
                "ratio": ratio,
                "input_len": input_len,
                "output_len": output_len,
                "workload": WORKLOADS[(input_len, output_len)],
                "rate": float(m.group(4)),
                "concurrency": int(m.group(5)),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "mean_e2el_ms": data.get("mean_e2el_ms"),
                "request_throughput": data.get("request_throughput"),
                "output_throughput": data.get("output_throughput"),
                "goodput": data.get("request_goodput"),
            }
        )

    print(f"Loaded {len(records)} result files.")
    return records


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _active_ratios(records):
    seen = {r["ratio"] for r in records}
    return [r for r in RATIOS if r in seen]


def _grouped_bar_grid(records, metric_key, ylabel, title, filename):
    """3×2 grid: rows=workloads, cols=rates. Grouped bars per ratio."""
    rates = sorted({r["rate"] for r in records})
    workload_keys = list(WORKLOADS.keys())
    workload_labels = list(WORKLOADS.values())

    if not rates or not workload_keys:
        return

    active = _active_ratios(records)
    x = np.arange(len(active))
    width = 0.8 / max(len(active), 1)

    fig, axes = plt.subplots(
        len(workload_keys),
        len(rates),
        figsize=(6 * len(rates), 4 * len(workload_keys)),
        squeeze=False,
    )

    for row, (wl_key, wl_label) in enumerate(WORKLOADS.items()):
        for col, rate in enumerate(rates):
            ax = axes[row][col]
            bar_values = []
            bar_colors = []
            for ratio in active:
                pts = [
                    r[metric_key]
                    for r in records
                    if r["ratio"] == ratio
                    and (r["input_len"], r["output_len"]) == wl_key
                    and r["rate"] == rate
                    and r.get(metric_key) is not None
                ]
                bar_values.append(pts[0] if pts else 0)
                bar_colors.append(COLORS[ratio])

            ax.bar(x, bar_values, width, color=bar_colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(active, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{wl_label.replace(chr(10), ' ')} — rate={rate} req/s", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_latency(records):
    """TTFT and E2EL in a 3×2 grid per workload × rate."""
    for metric_key, ylabel, short in [
        ("mean_ttft_ms", "Mean TTFT (ms)", "TTFT"),
        ("mean_e2el_ms", "Mean E2E Latency (ms)", "E2EL"),
    ]:
        _grouped_bar_grid(
            records,
            metric_key,
            ylabel,
            f"{short} by Workload and Ratio (calibrated SLOs)",
            f"principled_ratio_comparison_{short.lower()}.png",
        )


def plot_goodput(records):
    has_goodput = [r for r in records if r.get("goodput") is not None]
    if not has_goodput:
        print("  Skipping goodput plot: no goodput data found.")
        return
    _grouped_bar_grid(
        has_goodput,
        "goodput",
        "Goodput (req/s meeting SLOs)",
        "Goodput by Workload and Ratio (calibrated SLOs)",
        "principled_ratio_comparison_goodput.png",
    )


def plot_throughput(records):
    _grouped_bar_grid(
        records,
        "request_throughput",
        "Throughput (req/s)",
        "Request Throughput by Workload and Ratio",
        "principled_ratio_comparison_throughput.png",
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib...")
        os.system("pip install matplotlib --break-system-packages -q")
        import matplotlib.pyplot as plt

    records = load_data()
    if not records:
        print("No results found. Run the experiment first.")
        raise SystemExit(1)

    print(f"\nGenerating plots in {PLOTS_DIR} ...")
    plot_latency(records)
    plot_goodput(records)
    plot_throughput(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
