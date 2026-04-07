#!/usr/bin/env python3
"""
Summarize results from all experiments in results/.
Output: summary.csv, summary_table.tex
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------- Helper functions ----------
def _roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    thresholds = np.unique(y_score)[::-1]
    tpr_list = []
    fpr_list = []
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    if total_pos == 0 or total_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([np.inf, -np.inf])
    tp = 0
    fp = 0
    prev_score = None
    for i, score in enumerate(y_score[desc_idx]):
        if prev_score is not None and score != prev_score:
            tpr_list.append(tp / total_pos)
            fpr_list.append(fp / total_neg)
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    tpr_list.append(tp / total_pos)
    fpr_list.append(fp / total_neg)
    fpr = np.array([0] + fpr_list)
    tpr = np.array([0] + tpr_list)
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    return fpr, tpr, thresholds


def _auc(fpr, tpr):
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    auc_value = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2.0
        auc_value += width * height
    return auc_value


def compute_auc_from_clients_csv(exp_dir: Path):
    """Compute AUC from clients CSV using score and is_malicious."""
    clients_csv = None
    for f in exp_dir.glob("*_clients.csv"):
        clients_csv = f
        break
    if clients_csv is None:
        return None
    df = pd.read_csv(clients_csv)
    if "score" not in df.columns or "is_malicious" not in df.columns:
        return None
    df_valid = df[df["is_malicious"] >= 0].copy()
    if df_valid.empty:
        return None
    y_true = df_valid["is_malicious"].astype(int).values
    y_score = df_valid["score"].values
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    return _auc(fpr, tpr)


def find_main_csv(exp_dir: Path):
    for f in exp_dir.glob("*.csv"):
        if "_metrics" not in f.name and "_clients" not in f.name:
            return f
    return None


def get_average_client_counts(exp_dir: Path) -> dict:
    """
    Compute average number of malicious and benign clients per round from clients CSV.
    Returns dict with keys: avg_total_malicious, avg_total_benign, avg_kept_malicious, avg_kept_benign
    (For no defense, kept = total).
    """
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
    # Filter valid labels (is_malicious >= 0)
    df_valid = df[df["is_malicious"] >= 0].copy()
    if df_valid.empty:
        return {"avg_total_malicious": 0.0, "avg_total_benign": 0.0,
                "avg_kept_malicious": 0.0, "avg_kept_benign": 0.0}
    # Group by round
    grouped = df_valid.groupby("round")
    # Average malicious per round
    avg_malicious = grouped["is_malicious"].sum().mean()
    # Average total clients per round (number of rows per round)
    avg_total = grouped.size().mean()
    avg_benign = avg_total - avg_malicious
    # For no defense, kept = total
    return {
        "avg_total_malicious": avg_malicious,
        "avg_total_benign": avg_benign,
        "avg_kept_malicious": avg_malicious,
        "avg_kept_benign": avg_benign,
    }


def extract_aggregation_averages(metrics_csv: Path) -> dict:
    """
    Compute average aggregation metrics across all rounds.
    Returns dict with keys like:
        avg_malicious_removal_rate, avg_benign_removal_rate, avg_kept_malicious, ...
    If no aggregation records found, returns empty dict.
    """
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
            if key in ["defense-detect-precision", "defense-detect-recall", "defense-detect-fpr"]:
                metrics[key.replace("defense-detect-", "")] = float(val) if pd.notna(val) else None
            elif key in ["defense-detect-tp", "defense-detect-fp", "defense-detect-fn", "defense-detect-tn"]:
                metrics[key.replace("defense-detect-", "")] = float(val) if pd.notna(val) else None

        # Check if there are any aggregation records
        agg_rows = df_metrics[df_metrics["component"] == "aggregation"]
        if not agg_rows.empty:
            has_aggregation = True
            # Compute averages
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
            for key, new_key in key_mapping.items():
                values = agg_rows[agg_rows["key"] == key]["value"]
                if not values.empty:
                    numeric = pd.to_numeric(values, errors='coerce').dropna()
                    metrics[new_key] = numeric.mean() if not numeric.empty else 0.0
                else:
                    metrics[new_key] = 0.0

    # If no aggregation metrics found (baseline), compute from clients CSV
    if not has_aggregation:
        # Compute average malicious per round from clients CSV
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


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory {RESULTS_DIR} not found.")
        return

    exp_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not exp_dirs:
        print("No experiment subdirectories found.")
        return

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

    csv_path = RESULTS_DIR / "summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    latex_path = RESULTS_DIR / "summary_table.tex"
    with open(latex_path, "w") as f:
        f.write(df_summary.to_latex(index=False, float_format="%.4f"))
    print(f"LaTeX table saved to {latex_path}")

    print("\nSummary Table:\n", df_summary.to_string())


if __name__ == "__main__":
    main()