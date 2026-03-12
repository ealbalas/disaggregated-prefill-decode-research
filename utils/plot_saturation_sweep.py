#!/usr/bin/env python3
# utils/plot_saturation_sweep.py
#
# Plots results from the saturation_sweep experiment.
# Shows throughput (req/s) vs request rate for colocated and 1p1d across three
# workload shapes. Marks the plateau region where throughput gain < 5% per step.
#
# Usage:
#   python utils/plot_saturation_sweep.py
#   python utils/plot_saturation_sweep.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
    "--results-dir", default=str(_repo_root / "results" / "saturation_sweep")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "saturation_sweep" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Workload archetypes (input_len, output_len) → display label
WORKLOADS = {
    (1024, 64): "prefill-heavy\n(1024×64)",
    (256, 256): "balanced\n(256×256)",
    (256, 128): "mid-balanced\n(256×128)",
}

CONFIGS = ["colocated", "1p1d"]
COLORS = {c: cm.tab10(i / max(len(CONFIGS) - 1, 1)) for i, c in enumerate(CONFIGS)}

# Plateau threshold: throughput gain below this fraction per rate step is plateau
PLATEAU_THRESHOLD = 0.05


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
        if ratio not in CONFIGS:
            continue

        input_len = int(m.group(2))
        output_len = int(m.group(3))
        if (input_len, output_len) not in WORKLOADS:
            continue

        records.append(
            {
                "tag": tag,
                "config": ratio,
                "input_len": input_len,
                "output_len": output_len,
                "workload": WORKLOADS[(input_len, output_len)],
                "rate": float(m.group(4)),
                "concurrency": int(m.group(5)),
                "request_throughput": data.get("request_throughput"),
            }
        )

    print(f"Loaded {len(records)} result files.")
    return records


# ── Find plateau onset ────────────────────────────────────────────────────────
def find_plateau_rate(rates, throughputs):
    """Return the first rate where gain from previous step < PLATEAU_THRESHOLD."""
    for i in range(1, len(rates)):
        if throughputs[i - 1] and throughputs[i - 1] > 0:
            gain = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1]
            if gain < PLATEAU_THRESHOLD:
                return rates[i - 1]
    return None


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_throughput_vs_rate(records):
    """One subplot per workload shape. Lines: colocated vs 1p1d."""
    workload_keys = list(WORKLOADS.keys())
    workload_labels = list(WORKLOADS.values())

    fig, axes = plt.subplots(
        1, len(workload_keys), figsize=(6 * len(workload_keys), 5), sharey=False
    )
    if len(workload_keys) == 1:
        axes = [axes]

    for ax, (wl_key, wl_label) in zip(axes, WORKLOADS.items()):
        for config in CONFIGS:
            pts = [
                r
                for r in records
                if r["config"] == config
                and (r["input_len"], r["output_len"]) == wl_key
                and r["request_throughput"] is not None
            ]
            if not pts:
                continue
            pts.sort(key=lambda r: r["rate"])
            rates = [p["rate"] for p in pts]
            throughputs = [p["request_throughput"] for p in pts]

            ax.plot(
                rates,
                throughputs,
                marker="o",
                label=config,
                color=COLORS[config],
                linewidth=2,
            )

            # Mark plateau onset
            plateau = find_plateau_rate(rates, throughputs)
            if plateau is not None:
                ax.axvline(
                    plateau,
                    color=COLORS[config],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title(wl_label.replace("\n", " "))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Throughput vs Request Rate — Saturation Sweep\n"
        "(dashed vertical = plateau onset, Δthroughput < 5% per step)",
        y=1.02,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "saturation_sweep_throughput.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_throughput_all_on_one(records):
    """All workload × config combinations on one axes for easy comparison."""
    fig, ax = plt.subplots(figsize=(9, 5))

    linestyles = {"colocated": "-", "1p1d": "--"}
    markers = {(1024, 64): "o", (256, 256): "s", (256, 128): "^"}
    color_by_workload = {
        (1024, 64): cm.tab10(0),
        (256, 256): cm.tab10(1),
        (256, 128): cm.tab10(2),
    }

    for wl_key, wl_label in WORKLOADS.items():
        for config in CONFIGS:
            pts = [
                r
                for r in records
                if r["config"] == config
                and (r["input_len"], r["output_len"]) == wl_key
                and r["request_throughput"] is not None
            ]
            if not pts:
                continue
            pts.sort(key=lambda r: r["rate"])
            rates = [p["rate"] for p in pts]
            throughputs = [p["request_throughput"] for p in pts]

            ax.plot(
                rates,
                throughputs,
                marker=markers[wl_key],
                linestyle=linestyles[config],
                color=color_by_workload[wl_key],
                label=f"{config} / {wl_label.split(chr(10))[0]}",
                linewidth=2,
            )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Request Rate — All Workloads\n(solid=colocated, dashed=1p1d)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "saturation_sweep_throughput_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
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

    records = load_data()
    if not records:
        print("No results found. Run the experiment first.")
        raise SystemExit(1)

    print(f"\nGenerating plots in {PLOTS_DIR} ...")
    plot_throughput_vs_rate(records)
    plot_throughput_all_on_one(records)

    print(f"\nAll plots saved to {PLOTS_DIR}")
