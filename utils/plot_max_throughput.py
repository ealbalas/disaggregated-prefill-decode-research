#!/usr/bin/env python3
# utils/plot_max_throughput.py
#
# Plots results from the max_throughput experiment.
#
# Prefill capacity: throughput (req/s) vs request rate, one curve per input_len
#   (all with output_len=1). The saturation point is where the throughput curve
#   diverges below the offered-load diagonal (throughput < rate).
#
# Decode capacity: same plot structure, one curve per output_len
#   (all with input_len=64).
#
# Also plots token throughput (tokens/s) vs rate to show capacity in absolute
# terms: input_token_throughput = request_throughput × input_len, and
# output_token_throughput = request_throughput × output_len.
#
# Usage:
#   python utils/plot_max_throughput.py
#   python utils/plot_max_throughput.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
parser.add_argument(
    "--results-dir", default=str(_repo_root / "results" / "max_throughput")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "max_throughput" / "plots"),
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["colocated", "1p1d"]
LINESTYLES = {"colocated": "-", "1p1d": "--"}

# Lengths used in each sweep
PREFILL_INPUT_LENS = [64, 128, 256, 512, 1024]
DECODE_OUTPUT_LENS = [64, 128, 256, 512, 1024]
DECODE_INPUT_LEN = 64


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
        rate = float(m.group(4))

        # Classify into prefill sweep (output_len==1) or decode sweep (input_len==DECODE_INPUT_LEN, output_len>1)
        if output_len == 1 and input_len in PREFILL_INPUT_LENS:
            phase = "prefill"
            sweep_len = input_len
        elif input_len == DECODE_INPUT_LEN and output_len in DECODE_OUTPUT_LENS:
            phase = "decode"
            sweep_len = output_len
        else:
            continue

        throughput = data.get("request_throughput")
        if throughput is None:
            continue

        records.append(
            {
                "tag": tag,
                "config": config,
                "phase": phase,
                "input_len": input_len,
                "output_len": output_len,
                "sweep_len": sweep_len,
                "rate": rate,
                "request_throughput": throughput,
                "token_throughput": throughput * sweep_len,
            }
        )

    print(f"Loaded {len(records)} result files.")
    return records


# ── Throughput vs rate (req/s) ─────────────────────────────────────────────────
def plot_req_throughput(records, phase, sweep_lens, x_label, title_suffix, filename):
    """One subplot per config. Lines: one per sweep_len."""
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(7 * len(CONFIGS), 5), sharey=False)
    if len(CONFIGS) == 1:
        axes = [axes]

    colors = {l: cm.tab10(i / max(len(sweep_lens) - 1, 1)) for i, l in enumerate(sweep_lens)}

    for ax, config in zip(axes, CONFIGS):
        all_rates = set()
        for sweep_len in sweep_lens:
            pts = [
                r
                for r in records
                if r["config"] == config
                and r["phase"] == phase
                and r["sweep_len"] == sweep_len
            ]
            if not pts:
                continue
            pts.sort(key=lambda r: r["rate"])
            rates = [p["rate"] for p in pts]
            throughputs = [p["request_throughput"] for p in pts]
            all_rates.update(rates)

            ax.plot(
                rates,
                throughputs,
                marker="o",
                label=f"{x_label}={sweep_len}",
                color=colors[sweep_len],
                linewidth=2,
            )

        # Offered-load diagonal (throughput = rate → not saturated)
        if all_rates:
            r_max = max(all_rates)
            ax.plot(
                [0, r_max],
                [0, r_max],
                color="gray",
                linestyle=":",
                linewidth=1.5,
                label="offered load (ideal)",
            )

        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title(f"{config} — {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Throughput vs Rate — {title_suffix}\n"
        "(curves diverge below diagonal = saturation reached)",
        y=1.02,
    )
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Token throughput vs rate ───────────────────────────────────────────────────
def plot_token_throughput(records, phase, sweep_lens, x_label, title_suffix, filename):
    """Token throughput (tokens/s) = request_throughput × sweep_len, vs rate."""
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(7 * len(CONFIGS), 5), sharey=False)
    if len(CONFIGS) == 1:
        axes = [axes]

    colors = {l: cm.tab10(i / max(len(sweep_lens) - 1, 1)) for i, l in enumerate(sweep_lens)}

    for ax, config in zip(axes, CONFIGS):
        for sweep_len in sweep_lens:
            pts = [
                r
                for r in records
                if r["config"] == config
                and r["phase"] == phase
                and r["sweep_len"] == sweep_len
            ]
            if not pts:
                continue
            pts.sort(key=lambda r: r["rate"])
            rates = [p["rate"] for p in pts]
            tok_throughputs = [p["token_throughput"] for p in pts]

            ax.plot(
                rates,
                tok_throughputs,
                marker="o",
                label=f"{x_label}={sweep_len}",
                color=colors[sweep_len],
                linewidth=2,
            )

        token_label = "Input tokens/s" if phase == "prefill" else "Output tokens/s"
        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel(token_label)
        ax.set_title(f"{config} — {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Token Throughput vs Rate — {title_suffix}\n"
        "(plateau = max token throughput capacity)",
        y=1.02,
    )
    plt.tight_layout()
    out = PLOTS_DIR / filename
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

    plot_req_throughput(
        records,
        phase="prefill",
        sweep_lens=PREFILL_INPUT_LENS,
        x_label="input_len",
        title_suffix="Prefill Capacity (output_len=1)",
        filename="prefill_req_throughput_vs_rate.png",
    )
    plot_token_throughput(
        records,
        phase="prefill",
        sweep_lens=PREFILL_INPUT_LENS,
        x_label="input_len",
        title_suffix="Prefill Capacity (output_len=1)",
        filename="prefill_token_throughput_vs_rate.png",
    )
    plot_req_throughput(
        records,
        phase="decode",
        sweep_lens=DECODE_OUTPUT_LENS,
        x_label="output_len",
        title_suffix="Decode Capacity (input_len=64)",
        filename="decode_req_throughput_vs_rate.png",
    )
    plot_token_throughput(
        records,
        phase="decode",
        sweep_lens=DECODE_OUTPUT_LENS,
        x_label="output_len",
        title_suffix="Decode Capacity (input_len=64)",
        filename="decode_token_throughput_vs_rate.png",
    )

    print(f"\nAll plots saved to {PLOTS_DIR}")
