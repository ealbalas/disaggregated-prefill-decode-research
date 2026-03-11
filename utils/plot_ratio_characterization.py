#!/usr/bin/env python3
# utils/plot_ratio_characterization.py
#
# Plots results from the ratio_characterization experiment.
# Compares all P:D ratios across three workload archetypes:
#   - prefill-heavy (1024x64)
#   - balanced      (256x256)
#   - decode-heavy  (64x1024)
#
# Generates four figures: latency, throughput, output_throughput, and goodput.
# Each figure has one subplot per request rate, with grouped bars per workload
# and one line/bar per ratio.
#
# Usage:
#   python utils/plot_ratio_characterization.py
#   python utils/plot_ratio_characterization.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
    "--results-dir", default=str(_repo_root / "results" / "ratio_characterization")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "ratio_characterization" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Workload archetypes (input_len, output_len) → display label
WORKLOADS = {
    (1024, 64): "prefill-heavy\n(1024×64)",
    (256, 256): "balanced\n(256×256)",
    (64, 1024): "decode-heavy\n(64×1024)",
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


def _grouped_bar_per_rate(records, metric_key, ylabel, title, filename):
    """One subplot per request rate. Grouped bars: x=workload, groups=ratios."""
    rates = sorted({r["rate"] for r in records})
    if not rates:
        return

    active = _active_ratios(records)
    workload_labels = list(WORKLOADS.values())

    fig, axes = plt.subplots(1, len(rates), figsize=(7 * len(rates), 5), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    x = np.arange(len(workload_labels))
    width = 0.8 / max(len(active), 1)

    for ax, rate in zip(axes, rates):
        for idx, ratio in enumerate(active):
            values = []
            for wl_label in workload_labels:
                pts = [
                    r[metric_key]
                    for r in records
                    if r["ratio"] == ratio
                    and r["workload"] == wl_label
                    and r["rate"] == rate
                    and r.get(metric_key) is not None
                ]
                values.append(pts[0] if pts else 0)

            offset = (idx - len(active) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=ratio,
                color=COLORS[ratio],
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(workload_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"rate={rate} req/s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_latency(records):
    latency_metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)", "Time-to-First-Token"),
        ("mean_tpot_ms", "Mean TPOT (ms)", "Time-per-Output-Token"),
        ("mean_e2el_ms", "Mean E2E Latency (ms)", "End-to-End Latency"),
    ]

    rates = sorted({r["rate"] for r in records})
    if not rates:
        return

    active = _active_ratios(records)
    workload_labels = list(WORKLOADS.values())
    x = np.arange(len(workload_labels))
    width = 0.8 / max(len(active), 1)

    fig, axes = plt.subplots(
        len(latency_metrics),
        len(rates),
        figsize=(7 * len(rates), 4 * len(latency_metrics)),
        squeeze=False,
    )

    for row, (metric_key, ylabel, metric_name) in enumerate(latency_metrics):
        for col, rate in enumerate(rates):
            ax = axes[row][col]
            for idx, ratio in enumerate(active):
                values = []
                for wl_label in workload_labels:
                    pts = [
                        r[metric_key]
                        for r in records
                        if r["ratio"] == ratio
                        and r["workload"] == wl_label
                        and r["rate"] == rate
                        and r.get(metric_key) is not None
                    ]
                    values.append(pts[0] if pts else 0)

                offset = (idx - len(active) / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    values,
                    width,
                    label=ratio,
                    color=COLORS[ratio],
                    alpha=0.85,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(workload_labels, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric_name} — rate={rate} req/s")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Latency by Workload Type and Ratio", y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "ratio_characterization_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_throughput(records):
    _grouped_bar_per_rate(
        records,
        "request_throughput",
        "Throughput (req/s)",
        "Request Throughput by Workload Type and Ratio",
        "ratio_characterization_throughput.png",
    )
    _grouped_bar_per_rate(
        records,
        "output_throughput",
        "Output Throughput (tok/s)",
        "Output Throughput by Workload Type and Ratio",
        "ratio_characterization_output_throughput.png",
    )


def plot_goodput(records):
    has_goodput = [r for r in records if r.get("goodput") is not None]
    if not has_goodput:
        print("  Skipping goodput plot: no goodput data found.")
        return
    _grouped_bar_per_rate(
        has_goodput,
        "goodput",
        "Goodput (req/s meeting SLOs)",
        "Goodput by Workload Type and Ratio",
        "ratio_characterization_goodput.png",
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
    plot_throughput(records)
    plot_goodput(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
