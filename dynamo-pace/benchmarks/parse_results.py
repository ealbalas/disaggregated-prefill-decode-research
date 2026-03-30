#!/usr/bin/env python3
"""parse_results.py — Parse Dynamo benchmark JSON files into a summary table and plots.

Usage:
    python parse_results.py
    python parse_results.py --results-dir /path/to/stats --plots-dir /path/to/plots
    python parse_results.py --csv summary.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    scratch = "/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research"
    parser = argparse.ArgumentParser(description="Parse Dynamo benchmark results.")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(scratch, "results/dynamo"),
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join(scratch, "results/dynamo/plots"),
        help="Directory to write output plots.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write a summary CSV.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (print table only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
    "request_throughput",
    "output_throughput",
]


def load_result(path: Path) -> dict:
    """Load a single benchmark JSON file and return a flat record dict."""
    with open(path) as f:
        data = json.load(f)

    record = {"file": path.name}

    # Extract topology tag from filename (e.g. disagg_1p1d_job12345_in512_out128...)
    stem = path.stem
    parts = stem.split("_")
    record["tag"] = stem

    # Try to extract input/output lens and rate from filename
    for part in parts:
        if part.startswith("in") and part[2:].isdigit():
            record["input_len"] = int(part[2:])
        elif part.startswith("out") and part[3:].isdigit():
            record["output_len"] = int(part[3:])
        elif part.startswith("rate") and part[4:].replace(".", "").isdigit():
            record["rate"] = float(part[4:])

    for key in METRIC_KEYS:
        record[key] = data.get(key, float("nan"))

    return record


def load_all_results(results_dir: Path) -> list[dict]:
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"[parse] No JSON files found in {results_dir}", file=sys.stderr)
        return []

    records = []
    for path in json_files:
        try:
            records.append(load_result(path))
        except Exception as e:
            print(f"[parse] Warning: skipping {path.name}: {e}", file=sys.stderr)

    print(f"[parse] Loaded {len(records)} result files.")
    return records


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    ("tag",              "Config"),
    ("input_len",        "In"),
    ("output_len",       "Out"),
    ("rate",             "Rate"),
    ("mean_ttft_ms",     "TTFT mean"),
    ("p99_ttft_ms",      "TTFT p99"),
    ("mean_tpot_ms",     "TPOT mean"),
    ("p99_tpot_ms",      "TPOT p99"),
    ("mean_e2el_ms",     "E2EL mean"),
    ("p99_e2el_ms",      "E2EL p99"),
    ("request_throughput", "Req/s"),
]


def print_table(records: list[dict]) -> None:
    if not records:
        return

    headers = [col[1] for col in DISPLAY_COLS]
    rows = []
    for r in records:
        row = []
        for key, _ in DISPLAY_COLS:
            val = r.get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.1f}")
            else:
                row.append(str(val) if val != "" else "-")
        rows.append(row)

    # Compute column widths
    widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    sep = "  ".join("-" * w for w in widths)
    hdr = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))

    print()
    print(hdr)
    print(sep)
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], csv_path: Path) -> None:
    import csv

    if not records:
        return

    all_keys = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"[parse] CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(records: list[dict], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[parse] matplotlib not available — skipping plots.")
        return

    if not records:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group by tag (config name without timestamp/job suffix)
    def config_label(tag: str) -> str:
        # Strip trailing timestamp _YYYYMMDD_HHMMSS and job ID _jobNNN
        parts = tag.split("_")
        label_parts = []
        for p in parts:
            if p.startswith("job") and p[3:].isdigit():
                break
            if len(p) == 8 and p.isdigit():
                break
            label_parts.append(p)
        return "_".join(label_parts)

    for r in records:
        r["config"] = config_label(r["tag"])

    configs = sorted({r["config"] for r in records})

    # Plot 1: TTFT mean by config
    _bar_plot(
        records=records,
        configs=configs,
        metric="mean_ttft_ms",
        ylabel="Mean TTFT (ms)",
        title="Mean Time-to-First-Token by Config",
        outfile=plots_dir / "ttft_mean.png",
    )

    # Plot 2: TPOT mean by config
    _bar_plot(
        records=records,
        configs=configs,
        metric="mean_tpot_ms",
        ylabel="Mean TPOT (ms)",
        title="Mean Time-per-Output-Token by Config",
        outfile=plots_dir / "tpot_mean.png",
    )

    # Plot 3: E2E latency p99 by config
    _bar_plot(
        records=records,
        configs=configs,
        metric="p99_e2el_ms",
        ylabel="p99 E2E Latency (ms)",
        title="p99 End-to-End Latency by Config",
        outfile=plots_dir / "e2el_p99.png",
    )

    # Plot 4: Throughput by config
    _bar_plot(
        records=records,
        configs=configs,
        metric="request_throughput",
        ylabel="Request Throughput (req/s)",
        title="Request Throughput by Config",
        outfile=plots_dir / "throughput.png",
    )

    print(f"[parse] Plots saved to: {plots_dir}")


def _bar_plot(
    records: list[dict],
    configs: list[str],
    metric: str,
    ylabel: str,
    title: str,
    outfile: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Average metric across all runs per config
    values = []
    for cfg in configs:
        vals = [r[metric] for r in records if r.get("config") == cfg and r.get(metric) == r.get(metric)]
        values.append(float(np.mean(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    bars = ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)

    if not results_dir.exists():
        print(f"[parse] ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    records = load_all_results(results_dir)
    if not records:
        sys.exit(0)

    print_table(records)

    if args.csv:
        write_csv(records, Path(args.csv))

    if not args.no_plots:
        make_plots(records, plots_dir)


if __name__ == "__main__":
    main()
