#!/usr/bin/env python3
# utils/plot_throughput_comparison.py
#
# Plots results from the throughput_comparison experiment.
# Shows output_throughput and request_throughput across the full
# (input_len × output_len) workload grid for each config.
#
# Generates one figure per metric, with one subplot per request rate.
# Each subplot is a grouped bar chart: x=workload, bars=configs.
#
# Usage:
#   python utils/plot_throughput_comparison.py
#   python utils/plot_throughput_comparison.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
    "--results-dir", default=str(_repo_root / "results" / "throughput_comparison")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "throughput_comparison" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Config display order (colocated first, then disaggregated ratios)
CONFIG_ORDER = ["colocated", "1p1d", "1p2d", "1p3d", "2p1d", "3p1d"]


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

        records.append(
            {
                "tag": tag,
                "config": m.group(1),
                "input_len": int(m.group(2)),
                "output_len": int(m.group(3)),
                "rate": float(m.group(4)),
                "concurrency": int(m.group(5)),
                "request_throughput": data.get("request_throughput"),
                "output_throughput": data.get("output_throughput"),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "mean_e2el_ms": data.get("mean_e2el_ms"),
                "goodput": data.get("request_goodput"),
            }
        )

    print(f"Loaded {len(records)} result files.")
    return records


# ── Plot helpers ───────────────────────────────────────────────────────────────
def _active_configs(records):
    seen = {r["config"] for r in records}
    return [c for c in CONFIG_ORDER if c in seen]


def _workload_labels(records):
    """Sorted unique (input_len, output_len) pairs as display strings."""
    pairs = sorted({(r["input_len"], r["output_len"]) for r in records})
    return pairs, [f"{i}×{o}" for i, o in pairs]


def _make_colors(configs):
    return {c: cm.tab10(i / max(len(configs) - 1, 1)) for i, c in enumerate(configs)}


def _grouped_bar(records, metric_key, ylabel, title, filename):
    """One subplot per request rate. x=workload, grouped bars=configs."""
    rates = sorted({r["rate"] for r in records})
    if not rates:
        return

    active = _active_configs(records)
    colors = _make_colors(active)
    pairs, wl_labels = _workload_labels(records)

    fig, axes = plt.subplots(1, len(rates), figsize=(max(8, 3 * len(pairs)) * len(rates), 5), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    x = np.arange(len(pairs))
    width = 0.8 / max(len(active), 1)

    for ax, rate in zip(axes, rates):
        for idx, config in enumerate(active):
            values = []
            for pair in pairs:
                pts = [
                    r[metric_key]
                    for r in records
                    if r["config"] == config
                    and (r["input_len"], r["output_len"]) == pair
                    and r["rate"] == rate
                    and r.get(metric_key) is not None
                ]
                values.append(pts[0] if pts else 0)

            offset = (idx - len(active) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=config, color=colors[config], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(wl_labels, fontsize=9)
        ax.set_xlabel("Workload (input×output tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"rate = {rate} req/s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_throughput(records):
    _grouped_bar(
        records,
        "output_throughput",
        "Output Throughput (tok/s)",
        "Output Throughput by Workload and Config",
        "throughput_comparison_output_throughput.png",
    )
    _grouped_bar(
        records,
        "request_throughput",
        "Request Throughput (req/s)",
        "Request Throughput by Workload and Config",
        "throughput_comparison_request_throughput.png",
    )


def plot_latency(records):
    for metric_key, ylabel, name in [
        ("mean_ttft_ms", "Mean TTFT (ms)", "TTFT"),
        ("mean_tpot_ms", "Mean TPOT (ms)", "TPOT"),
        ("mean_e2el_ms", "Mean E2E Latency (ms)", "E2E Latency"),
    ]:
        _grouped_bar(
            records,
            metric_key,
            ylabel,
            f"{name} by Workload and Config",
            f"throughput_comparison_{metric_key}.png",
        )


def plot_goodput(records):
    has_goodput = [r for r in records if r.get("goodput") is not None]
    if not has_goodput:
        print("  Skipping goodput plot: no goodput data found.")
        return
    _grouped_bar(
        has_goodput,
        "goodput",
        "Goodput (req/s meeting SLOs)",
        "Goodput by Workload and Config",
        "throughput_comparison_goodput.png",
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
    plot_throughput(records)
    plot_latency(records)
    plot_goodput(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
