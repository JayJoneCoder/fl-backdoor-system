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
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fl_backdoor.utils.metrics import auc as _auc, roc_curve as _roc_curve
from fl_backdoor.utils.paths import experiment_path, summary_path
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"


# ---------- CSV readers ----------
def compute_auc_from_clients_csv(exp_dir: Path):
    clients_csv = None
    for f in exp_dir.glob("*_clients.csv"):
        clients_csv = f
        break
    if clients_csv is None:
        return None
    df = pd.read_csv(clients_csv)
    if "score" not in df.columns or "is_malicious" not in df.columns:
        return None
    df_valid = df[df["is_malicious"].isin([0, 1])].copy()
    df_valid["score"] = pd.to_numeric(df_valid["score"], errors="coerce")
    df_valid = df_valid[np.isfinite(df_valid["score"])]
    if df_valid.empty:
        return None
    y_true = df_valid["is_malicious"].astype(int).values
    y_score = df_valid["score"].values
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    value = _auc(fpr, tpr)
    return float(value) if np.isfinite(value) else None


def find_main_csv(exp_dir: Path):
    for f in exp_dir.glob("*.csv"):
        if "_metrics" not in f.name and "_clients" not in f.name:
            return f
    return None


def get_average_client_counts(exp_dir: Path) -> dict:
    clients_csv = None
    for f in exp_dir.glob("*_clients.csv"):
        clients_csv = f
        break
    if clients_csv is None:
        return {"avg_total_malicious": 0.0, "avg_total_benign": 0.0,
                "avg_kept_malicious": 0.0, "avg_kept_benign": 0.0}
    df = pd.read_csv(clients_csv)
    if "is_malicious" not in df.columns or "round" not in df.columns:
        return {"avg_total_malicious": 0.0, "avg_total_benign": 0.0,
                "avg_kept_malicious": 0.0, "avg_kept_benign": 0.0}
    df_valid = df[df["is_malicious"] >= 0].copy()
    if df_valid.empty:
        return {"avg_total_malicious": 0.0, "avg_total_benign": 0.0,
                "avg_kept_malicious": 0.0, "avg_kept_benign": 0.0}
    grouped = df_valid.groupby("round")
    avg_malicious = grouped["is_malicious"].sum().mean()
    avg_total = grouped.size().mean()
    avg_benign = avg_total - avg_malicious
    return {
        "avg_total_malicious": avg_malicious,
        "avg_total_benign": avg_benign,
        "avg_kept_malicious": avg_malicious,
        "avg_kept_benign": avg_benign,
    }


def extract_aggregation_averages(metrics_csv: Path) -> dict:
    key_mapping = {
        "defense-agg-malicious-removal-rate": "avg_malicious_removal_rate",
        "defense-agg-benign-removal-rate": "avg_benign_removal_rate",
        "defense-agg-kept-malicious": "avg_kept_malicious",
        "defense-agg-removed-malicious": "avg_removed_malicious",
        "defense-agg-removed-benign": "avg_removed_benign",
        "defense-agg-kept-benign": "avg_kept_benign",
        "defense-agg-total-malicious": "avg_total_malicious",
        "defense-agg-total-benign": "avg_total_benign",
        "defense-agg-selected-clients": "avg_selected_clients",
    }
    if not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    agg_rows = df[df["component"] == "aggregation"]
    if agg_rows.empty:
        return {}
    averages = {}
    for key, new_key in key_mapping.items():
        values = agg_rows[agg_rows["key"] == key]["value"]
        if not values.empty:
            numeric = pd.to_numeric(values, errors='coerce').dropna()
            averages[new_key] = numeric.mean() if not numeric.empty else 0.0
        else:
            averages[new_key] = 0.0
    return averages


def extract_experiment_metrics(exp_dir: Path):
    """Extract all relevant metrics from an experiment directory."""
    main_csv = find_main_csv(exp_dir)
    if main_csv is None:
        return None

    df_main = pd.read_csv(main_csv)
    if df_main.empty:
        return None
    last_row = df_main.iloc[-1]
    acc = last_row.get("accuracy", None)
    asr = last_row.get("asr", None)
    loss = last_row.get("loss", None)

    metrics_csv = exp_dir / f"{main_csv.stem}_metrics.csv"
    metrics = {}
    has_aggregation = False

    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        last_round = int(last_row.get("round", -1))

        # Detection metrics (last round)
        detection_rows = df_metrics[
            (df_metrics["round"] == last_round) & (df_metrics["component"] == "detection")
        ]
        for _, row in detection_rows.iterrows():
            key = row["key"]
            val = row["value"]
            if key in ["detect_precision", "detect_recall", "detect_fpr"]:
                metrics[key.replace("detect_", "")] = float(val) if pd.notna(val) else None
            elif key in ["detect_tp", "detect_fp", "detect_fn", "detect_tn"]:
                metrics[key.replace("detect_", "")] = float(val) if pd.notna(val) else None

        # Check if there are any aggregation records
        agg_rows = df_metrics[df_metrics["component"] == "aggregation"]
        if not agg_rows.empty:
            has_aggregation = True
            key_mapping = {
                "agg_malicious_removal_rate": "avg_malicious_removal_rate",
                "agg_benign_removal_rate": "avg_benign_removal_rate",
                "agg_kept_malicious": "avg_kept_malicious",
                "agg_removed_malicious": "avg_removed_malicious",
                "agg_removed_benign": "avg_removed_benign",
                "agg_kept_benign": "avg_kept_benign",
                "agg_total_malicious": "avg_total_malicious",
                "agg_total_benign": "avg_total_benign",
                "agg_selected_clients": "avg_selected_clients",
            }
            for key, new_key in key_mapping.items():
                values = agg_rows[agg_rows["key"] == key]["value"]
                if not values.empty:
                    numeric = pd.to_numeric(values, errors='coerce').dropna()
                    metrics[new_key] = numeric.mean() if not numeric.empty else 0.0
                else:
                    metrics[new_key] = 0.0

    # If no aggregation metrics found (baseline), compute from clients CSV
    if not has_aggregation:
        clients_csv = None
        for f in exp_dir.glob("*_clients.csv"):
            clients_csv = f
            break
        if clients_csv is not None:
            df_clients = pd.read_csv(clients_csv)
            if "is_malicious" in df_clients.columns and "round" in df_clients.columns:
                df_valid = df_clients[df_clients["is_malicious"] >= 0].copy()
                if not df_valid.empty:
                    avg_malicious = df_valid.groupby("round")["is_malicious"].sum().mean()
                    avg_total = df_valid.groupby("round").size().mean()
                    avg_benign = avg_total - avg_malicious
                    metrics["avg_total_malicious"] = avg_malicious
                    metrics["avg_total_benign"] = avg_benign
                    metrics["avg_kept_malicious"] = avg_malicious
                    metrics["avg_kept_benign"] = avg_benign
                    metrics["avg_malicious_removal_rate"] = 0.0
                    metrics["avg_benign_removal_rate"] = 0.0
                    metrics["avg_removed_malicious"] = 0.0
                    metrics["avg_removed_benign"] = 0.0
                    metrics["avg_selected_clients"] = 0.0

    # Compute AUC from clients CSV
    auc = compute_auc_from_clients_csv(exp_dir)
    if auc is not None:
        metrics["auc"] = auc

    result = {"accuracy": acc, "asr": asr, "loss": loss, **metrics}
    return result


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
        for directory in RESULTS_DIR.iterdir():
            try:
                path = experiment_path(RESULTS_DIR, directory.name)
                if path.is_dir():
                    exp_dirs.append(path)
            except ValueError:
                continue

    if not exp_dirs:
        print("No valid experiment directories found.")
        return

    # Create output directory
    if args.output_name:
        output_name = args.output_name
    else:
        import uuid
        output_name = datetime.now().strftime("summary_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = summary_path(RESULTS_DIR, output_name)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)  # Never overwrite a previous report.

    rows = []
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        metrics = extract_experiment_metrics(exp_dir)
        if metrics is None:
            print(f"Warning: Could not extract metrics from {exp_name}")
            continue
        row = {"experiment": exp_name, **metrics}
        rows.append(row)

    if not rows:
        print("No valid experiments.")
        return

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
    with open(latex_path, "w") as f:
        f.write(df_summary.to_latex(index=False, float_format="%.4f"))
    print(f"LaTeX table saved to {latex_path}")

    # Optionally generate multi-experiment plots
    if not args.no_plots:
        plot_script = PROJECT_ROOT / "scripts" / "plot.py"
        import subprocess
        cmd = [sys.executable, str(plot_script), str(output_dir)]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        print(f"Plots generated in {output_dir}")

    print(f"\nSummary completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
