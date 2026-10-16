#!/usr/bin/env python3
"""
Summarize results from selected experiments.
Output: summary.csv, summary_table.tex, and optional plots.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fl_backdoor.utils.paths import experiment_path, summary_path
from backend.results_scanner import extract_experiment_metrics as read_result_metrics
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"


# ---------- Shared tolerant CSV reader ----------
def extract_experiment_metrics(exp_dir: Path):
    """Compatibility wrapper using the same tolerant reader as the web history."""
    result = read_result_metrics(exp_dir)
    if result is None:
        return None
    return {key: value for key, value in result.items()
            if key not in {"curve_data", "warnings", "last_round", "dataset"}}


# ---------- Main ----------
def main():
    global RESULTS_DIR, SUMMARIES_DIR
    parser = argparse.ArgumentParser(description="Summarize selected experiments.")
    parser.add_argument("--experiments", nargs="+", help="List of experiment names to include.")
    parser.add_argument("--output-name", help="Name of the summary output directory (default: timestamp).")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating multi-experiment plots.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Experiment results root.")
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir.resolve()
    SUMMARIES_DIR = RESULTS_DIR / "summaries"

    # Determine which experiments to process
    if args.experiments:
        exp_dirs = [experiment_path(RESULTS_DIR, name) for name in args.experiments]
        if any(not directory.is_dir() for directory in exp_dirs):
            parser.error("One or more selected experiments do not exist.")
    else:
        # Default: all experiment subdirectories directly under results/
        exp_dirs = []
        for directory in RESULTS_DIR.iterdir() if RESULTS_DIR.is_dir() else []:
            try:
                path = experiment_path(RESULTS_DIR, directory.name)
                if path.is_dir():
                    exp_dirs.append(path)
            except ValueError:
                continue

    if not exp_dirs:
        print("No valid experiment directories found.")
        return

    rows = []
    for exp_dir in exp_dirs:
        metrics = read_result_metrics(exp_dir)
        if metrics is None or not any(metrics.get(key) is not None for key in ("accuracy", "asr", "loss")):
            print(f"Warning: {exp_dir.name} 没有可用指标，已跳过。")
            continue
        for warning in metrics.pop("warnings", []):
            print(f"Warning: {exp_dir.name}: {warning}")
        for key in ("curve_data", "dataset", "last_round"):
            metrics.pop(key, None)
        rows.append({"experiment": exp_dir.name, **metrics})
    if not rows:
        parser.error("没有可用于总结的实验指标。")

    # Reserve output only after finding usable data; never leave an empty report.
    if args.output_name:
        output_name = args.output_name
    else:
        import uuid
        output_name = datetime.now().strftime("summary_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = summary_path(RESULTS_DIR, output_name)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)  # Never overwrite a previous report.

    df_summary = pd.DataFrame(rows)

    # Define column order
    base_cols = ["experiment", "accuracy", "asr", "loss"]
    detect_cols = ["precision", "recall", "fpr", "auc", "tp", "fp", "fn", "tn"]
    agg_cols = [c for c in df_summary.columns if c.startswith("avg_")]
    final_cols = base_cols + [c for c in detect_cols if c in df_summary.columns] + agg_cols
    df_summary = df_summary[final_cols]

    csv_path = output_dir / "summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    latex_path = output_dir / "summary_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(df_summary.to_latex(index=False, float_format="%.4f"))
    print(f"LaTeX table saved to {latex_path}")

    # Optionally generate multi-experiment plots
    if not args.no_plots:
        plot_script = PROJECT_ROOT / "scripts" / "plot.py"
        import subprocess
        cmd = [sys.executable, str(plot_script), str(output_dir)]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        print(f"Plots generated in {output_dir}")

    print(f"\nSummary completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
