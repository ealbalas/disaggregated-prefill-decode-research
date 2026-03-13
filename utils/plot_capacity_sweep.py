#!/usr/bin/env python3
# utils/plot_capacity_sweep.py
#
# Plots results from the capacity_sweep experiment.
# Shows output_throughput (tok/s) vs input_len (prefill sweep) and vs
# output_len (decode sweep) for each config, with a horizontal target line.
#
# Usage:
#   python utils/plot_capacity_sweep.py
#   python utils/plot_capacity_sweep.py --results-dir /path/to/stats --plots-dir /path/to/plots

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
    "--results-dir", default=str(_repo_root / "results" / "capacity_sweep")
)
parser.add_argument(
    "--plots-dir",
    default=str(_repo_root / "results" / "capacity_sweep" / "plots"),
)
parser.add_argument(
    "--target-tok-per-s",
    type=float,
    default=None,
    help="Target throughput threshold line (tok/s). Inferred from data if omitted.",
)
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
PLOTS_DIR = Path(args.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = ["colocated", "1p1d", "1p2d", "1p3d", "2p1d", "3p1d"]

# Tag pattern: {config}_{prefill|decode}_{input}x{output}_rate{rate}_conc{conc}
TAG_RE = re.compile(
    r"(colocated|\d+p\d+d)_(prefill|decode)_(\d+)x(\d+)_rate([\d.]+)_conc(\d+)"
)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    prefill_records, decode_records = [], []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue

        tag = data.get("tag") or data.get("metadata", {}).get("tag", f.stem)
        m = TAG_RE.match(tag)
        if not m:
            continue

        rec = {
            "config": m.group(1),
            "sweep": m.group(2),          # "prefill" or "decode"
            "input_len": int(m.group(3)),
            "output_len": int(m.group(4)),
            "rate": float(m.group(5)),
            "output_throughput": data.get("output_throughput"),
            "request_throughput": data.get("request_throughput"),
        }

        if rec["sweep"] == "prefill":
            prefill_records.append(rec)
        else:
            decode_records.append(rec)

    print(f"Loaded {len(prefill_records)} prefill records, {len(decode_records)} decode records.")
    return prefill_records, decode_records


# ── Plot helpers ───────────────────────────────────────────────────────────────
def _active_configs(records):
    seen = {r["config"] for r in records}
    return [c for c in CONFIG_ORDER if c in seen]


def _make_colors(configs):
    return {c: cm.tab10(i / max(len(configs) - 1, 1)) for i, c in enumerate(configs)}


def _capacity_curve(records, x_key, xlabel, title, filename, target):
    if not records:
        print(f"  Skipping {filename}: no data.")
        return

    active = _active_configs(records)
    colors = _make_colors(active)

    fig, ax = plt.subplots(figsize=(9, 5))

    for config in active:
        pts = sorted(
            [r for r in records if r["config"] == config and r.get("output_throughput") is not None],
            key=lambda r: r[x_key],
        )
        if not pts:
            continue
        xs = [p[x_key] for p in pts]
        ys = [p["output_throughput"] for p in pts]
        ax.plot(xs, ys, marker="o", label=config, color=colors[config])

    if target is not None:
        ax.axhline(target, color="red", linestyle="--", linewidth=1.2, label=f"target ({target} tok/s)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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

    prefill_records, decode_records = load_data()
    if not prefill_records and not decode_records:
        print("No results found. Run the experiment first.")
        raise SystemExit(1)

    # Infer target from the data if not provided — use the minimum passing value
    # across all records as a rough guide, or fall back to None.
    target = args.target_tok_per_s

    print(f"\nGenerating plots in {PLOTS_DIR} ...")

    _capacity_curve(
        prefill_records,
        x_key="input_len",
        xlabel="Input Length (tokens)",
        title="Prefill Capacity: Throughput vs Input Length (output_len=1)",
        filename="capacity_sweep_prefill.png",
        target=target,
    )

    _capacity_curve(
        decode_records,
        x_key="output_len",
        xlabel="Output Length (tokens)",
        title="Decode Capacity: Throughput vs Output Length",
        filename="capacity_sweep_decode.png",
        target=target,
    )

    print(f"\nAll plots saved to {PLOTS_DIR}")
    if target is None:
        print("Tip: pass --target-tok-per-s N to draw the threshold line on the plots.")
