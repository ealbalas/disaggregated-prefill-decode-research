#!/usr/bin/env python3
# utils/plot_request_rate_test.py
#
# Plots results from the request_rate_test experiment.
# Fixed: input=1024, output=256, concurrency=32.
# Swept: request rate (2, 4, 8, 10 req/s).
# Generates throughput, latency, and goodput plots — x-axis = request rate,
# one line per ratio (colocated vs disaggregated).
#
# Usage:
#   python utils/plot_request_rate_test.py
#   python utils/plot_request_rate_test.py --results-dir /path/to/stats --plots-dir /path/to/plots

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
_repo_root = Path(__file__).resolve().parent.parent
parser.add_argument("--results-dir", default=str(_repo_root / "results" / "request_rate_test"))
parser.add_argument("--plots-dir",   default=str(_repo_root / "results" / "request_rate_test" / "plots"))
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR   = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Ratios included in this experiment (order determines legend order)
RATIOS = ["colocated", "1p1d", "1p2d", "2p1d"]
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

        records.append({
            "tag":                tag,
            "ratio":              ratio,
            "input_len":          int(m.group(2)),
            "output_len":         int(m.group(3)),
            "rate":               float(m.group(4)),
            "concurrency":        int(m.group(5)),
            "mean_ttft_ms":       data.get("mean_ttft_ms"),
            "mean_tpot_ms":       data.get("mean_tpot_ms"),
            "mean_e2el_ms":       data.get("mean_e2el_ms"),
            "request_throughput": data.get("request_throughput"),
            "output_throughput":  data.get("output_throughput"),
            "goodput":            data.get("request_goodput"),
        })

    print(f"Loaded {len(records)} result files.")
    return records


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _active_ratios(records):
    """Return only ratios that appear in the loaded data, preserving order."""
    seen = {r["ratio"] for r in records}
    return [r for r in RATIOS if r in seen]


def _single_plot(records, metric_key, ylabel, title, filename):
    """Single plot: x=request_rate, one line per ratio."""
    active = _active_ratios(records)
    fig, ax = plt.subplots(figsize=(8, 5))

    for ratio in active:
        pts = sorted(
            (r["rate"], r[metric_key])
            for r in records
            if r["ratio"] == ratio and r.get(metric_key) is not None
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=ratio, color=COLORS[ratio])

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_throughput(records):
    """Request throughput and output throughput vs request rate."""
    _single_plot(
        records, "request_throughput", "Throughput (req/s)",
        "Request Throughput vs Request Rate\n(input=1024, output=256, concurrency=32)",
        "request_rate_test_throughput.png",
    )
    _single_plot(
        records, "output_throughput", "Output Throughput (tok/s)",
        "Output Throughput vs Request Rate\n(input=1024, output=256, concurrency=32)",
        "request_rate_test_output_throughput.png",
    )


def plot_latency(records):
    """TTFT, TPOT, E2EL vs request rate — 3-row subplot."""
    active = _active_ratios(records)
    latency_metrics = [
        ("mean_ttft_ms",  "Mean TTFT (ms)",        "Time-to-First-Token"),
        ("mean_tpot_ms",  "Mean TPOT (ms)",        "Time-per-Output-Token"),
        ("mean_e2el_ms",  "Mean E2E Latency (ms)", "End-to-End Latency"),
    ]

    fig, axes = plt.subplots(len(latency_metrics), 1, figsize=(8, 4 * len(latency_metrics)))

    for ax, (metric_key, ylabel, metric_name) in zip(axes, latency_metrics):
        for ratio in active:
            pts = sorted(
                (r["rate"], r[metric_key])
                for r in records
                if r["ratio"] == ratio and r.get(metric_key) is not None
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=ratio, color=COLORS[ratio])

        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Latency vs Request Rate\n(input=1024, output=256, concurrency=32)",
        y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "request_rate_test_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_goodput(records):
    """Goodput (req/s meeting all SLOs) vs request rate."""
    has_goodput = [r for r in records if r.get("goodput") is not None]
    if not has_goodput:
        print("  Skipping goodput plot: no goodput data found.")
        return
    _single_plot(
        has_goodput, "goodput", "Goodput (req/s meeting SLOs)",
        "Goodput vs Request Rate\n(input=1024, output=256, concurrency=32)",
        "request_rate_test_goodput.png",
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
