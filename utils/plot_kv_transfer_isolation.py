#!/usr/bin/env python3
# utils/plot_kv_transfer_isolation.py
#
# Plots results from the kv_transfer_isolation experiment.
# Derives the Nixl KV transfer overhead as:
#   transfer_overhead_ms[input_len] = mean_ttft_1p1d - mean_ttft_colocated
# and visualizes:
#   1. transfer_overhead vs input_len (scatter + line with std error bars)
#   2. stacked bar of prefill_time (colocated TTFT) and transfer overhead per input_len
#
# Usage:
#   python utils/plot_kv_transfer_isolation.py
#   python utils/plot_kv_transfer_isolation.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
    "--results-dir", default=str(_repo_root / "results" / "kv_transfer_isolation")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "kv_transfer_isolation" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["colocated", "1p1d"]


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

        config = m.group(1)
        if config not in CONFIGS:
            continue

        input_len = int(m.group(2))
        output_len = int(m.group(3))

        records.append(
            {
                "tag": tag,
                "config": config,
                "input_len": input_len,
                "output_len": output_len,
                "rate": float(m.group(4)),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "std_ttft_ms": data.get("std_ttft_ms"),
                "p50_ttft_ms": data.get("p50_ttft_ms"),
                "p99_ttft_ms": data.get("p99_ttft_ms"),
            }
        )

    print(f"Loaded {len(records)} result files.")
    return records


# ── Derive transfer overhead ───────────────────────────────────────────────────
def derive_overhead(records):
    """
    Returns sorted list of dicts with keys:
      input_len, prefill_ms, prefill_std, transfer_ms, transfer_std, disagg_ms
    """
    by_input = {}
    for r in records:
        il = r["input_len"]
        if il not in by_input:
            by_input[il] = {}
        by_input[il][r["config"]] = r

    results = []
    for input_len in sorted(by_input.keys()):
        both = by_input[input_len]
        if "colocated" not in both or "1p1d" not in both:
            continue
        col = both["colocated"]
        dis = both["1p1d"]

        prefill_ms = col["mean_ttft_ms"] or 0
        prefill_std = col["std_ttft_ms"] or 0
        disagg_ms = dis["mean_ttft_ms"] or 0
        disagg_std = dis["std_ttft_ms"] or 0
        transfer_ms = disagg_ms - prefill_ms
        # Propagate std in quadrature for the difference
        transfer_std = (prefill_std**2 + disagg_std**2) ** 0.5

        results.append(
            {
                "input_len": input_len,
                "prefill_ms": prefill_ms,
                "prefill_std": prefill_std,
                "transfer_ms": transfer_ms,
                "transfer_std": transfer_std,
                "disagg_ms": disagg_ms,
            }
        )

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_transfer_overhead(overhead):
    """Scatter + line of transfer overhead vs input_len with error bars."""
    if not overhead:
        print("  Skipping transfer overhead plot: no paired data.")
        return

    input_lens = [o["input_len"] for o in overhead]
    transfer_ms = [o["transfer_ms"] for o in overhead]
    transfer_std = [o["transfer_std"] for o in overhead]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        input_lens,
        transfer_ms,
        yerr=transfer_std,
        marker="o",
        linewidth=2,
        capsize=4,
        color=cm.tab10(0),
        label="transfer overhead (1p1d TTFT − colocated TTFT)",
    )

    # Fit a linear trendline (transfer cost is approximately linear in KV size)
    if len(input_lens) >= 2:
        coeffs = np.polyfit(input_lens, transfer_ms, 1)
        x_fit = np.linspace(min(input_lens), max(input_lens), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color=cm.tab10(0),
            alpha=0.5,
            label=f"linear fit: {coeffs[0]:.3f}×input_len + {coeffs[1]:.1f}",
        )

    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("Transfer Overhead (ms)")
    ax.set_title("Nixl KV Transfer Overhead vs Input Length\n(output_len=1, rate=1 req/s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "kv_transfer_overhead.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_ttft_breakdown(overhead):
    """Stacked bar: prefill_time and transfer_overhead per input_len."""
    if not overhead:
        print("  Skipping TTFT breakdown plot: no paired data.")
        return

    input_lens = [o["input_len"] for o in overhead]
    prefill_ms = [o["prefill_ms"] for o in overhead]
    transfer_ms = [o["transfer_ms"] for o in overhead]
    prefill_std = [o["prefill_std"] for o in overhead]
    transfer_std = [o["transfer_std"] for o in overhead]

    x = np.arange(len(input_lens))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x, prefill_ms, width, label="prefill time (colocated TTFT)", color=cm.tab10(1), alpha=0.85)
    bars2 = ax.bar(
        x,
        transfer_ms,
        width,
        bottom=prefill_ms,
        label="KV transfer overhead",
        color=cm.tab10(0),
        alpha=0.85,
    )

    # Error bars on total (disagg TTFT)
    total_ms = [p + t for p, t in zip(prefill_ms, transfer_ms)]
    total_std = [(ps**2 + ts**2) ** 0.5 for ps, ts in zip(prefill_std, transfer_std)]
    ax.errorbar(x, total_ms, yerr=total_std, fmt="none", color="black", capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Breakdown: Prefill Time vs KV Transfer Overhead\n(output_len=1, rate=1 req/s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOTS_DIR / "kv_transfer_ttft_breakdown.png"
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

    overhead = derive_overhead(records)
    if not overhead:
        print("Could not derive transfer overhead: need both colocated and 1p1d results at the same input lengths.")
        raise SystemExit(1)

    print(f"\nKV transfer overhead by input length:")
    for o in overhead:
        print(
            f"  input={o['input_len']:4d}: prefill={o['prefill_ms']:.1f}ms, "
            f"transfer={o['transfer_ms']:.1f}ms ± {o['transfer_std']:.1f}ms"
        )

    print(f"\nGenerating plots in {PLOTS_DIR} ...")
    plot_transfer_overhead(overhead)
    plot_ttft_breakdown(overhead)

    print(f"\nAll plots saved to {PLOTS_DIR}")
