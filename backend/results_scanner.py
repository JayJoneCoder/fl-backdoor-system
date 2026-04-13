"""
Results scanner for reading experiment outputs.
Reuses logic from scripts/summarize_results.py but returns JSON-friendly dicts.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Helper functions (copied/adapted from summarize_results.py)
# ----------------------------------------------------------------------
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
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2.0
        auc_value += width * height
    return auc_value


def compute_auc_from_clients_csv(exp_dir: Path) -> Optional[float]:
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


def find_main_csv(exp_dir: Path) -> Optional[Path]:
    for f in exp_dir.glob("*.csv"):
        if "_metrics" not in f.name and "_clients" not in f.name:
            return f
    return None


def extract_experiment_metrics(exp_dir: Path) -> Optional[dict[str, Any]]:
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
    last_round = int(last_row.get("round", -1))

    metrics_csv = exp_dir / f"{main_csv.stem}_metrics.csv"
    metrics = {}
    has_aggregation = False

    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)

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

        # Aggregation averages
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
                    numeric = pd.to_numeric(values, errors="coerce").dropna()
                    metrics[new_key] = numeric.mean() if not numeric.empty else 0.0
                else:
                    metrics[new_key] = 0.0

    # If no aggregation metrics, compute from clients CSV
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

    auc = compute_auc_from_clients_csv(exp_dir)
    if auc is not None:
        metrics["auc"] = auc

    # Main curve data (for frontend plotting)
    curve_data = df_main[["round", "accuracy", "asr", "loss"]].to_dict(orient="list")

    # 提取数据集名称
    dataset = None
    snapshot_path = exp_dir / "config_snapshot.toml"
    if snapshot_path.exists():
        try:
            import tomli
            with open(snapshot_path, "rb") as f:
                config = tomli.load(f)
            tool_config = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
            dataset = tool_config.get("dataset")
        except Exception:
            pass

    return {
        "accuracy": acc,
        "asr": asr,
        "loss": loss,
        "last_round": last_round,
        "curve_data": curve_data,
        "dataset": dataset or "",
        **metrics,
    }


def list_experiments(results_dir: Path) -> list[dict[str, Any]]:
    """Return summary list of all experiments."""
    import os
    from datetime import datetime

    experiments = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        metrics = extract_experiment_metrics(exp_dir)
        if metrics is None:
            continue

        # 获取目录创建时间
        created_timestamp = os.path.getctime(exp_dir)
        created = datetime.fromtimestamp(created_timestamp).isoformat()

        # 尝试获取数据集名称
        dataset = None
        snapshot_path = exp_dir / "config_snapshot.toml"
        if snapshot_path.exists():
            try:
                import tomli
                with open(snapshot_path, "rb") as f:
                    config = tomli.load(f)
                tool_config = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
                dataset = tool_config.get("dataset")
            except Exception:
                pass

        # 如果快照没有，尝试从主 CSV 文件名或实验名推测
        if dataset is None:
            main_csv = find_main_csv(exp_dir)
            if main_csv:
                name_lower = main_csv.stem.lower()
                if "mnist" in name_lower:
                    dataset = "mnist"
                elif "cifar10" in name_lower or "cifar" in name_lower:
                    dataset = "cifar10"
                elif "cifar100" in name_lower:
                    dataset = "cifar100"
        if dataset is None:
            exp_name_lower = exp_dir.name.lower()
            if "mnist" in exp_name_lower:
                dataset = "mnist"
            elif "cifar10" in exp_name_lower or "cifar" in exp_name_lower:
                dataset = "cifar10"
            elif "cifar100" in exp_name_lower:
                dataset = "cifar100"

        experiments.append({
            "name": exp_dir.name,
            "accuracy": metrics.get("accuracy"),
            "asr": metrics.get("asr"),
            "loss": metrics.get("loss"),
            "last_round": metrics.get("last_round"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "fpr": metrics.get("fpr"),
            "auc": metrics.get("auc"),
            "created": created,
            "dataset": dataset or "",
        })
    # 按创建时间倒序排列（最新的在前）
    experiments.sort(key=lambda x: x["created"] or "", reverse=True)
    return experiments


def get_experiment_detail(exp_dir: Path) -> Optional[dict[str, Any]]:
    """Return detailed metrics plus file list for a specific experiment."""
    metrics = extract_experiment_metrics(exp_dir)
    if metrics is None:
        return None

    # Gather available image files
    images = [f.name for f in exp_dir.glob("*.png")]
    csv_files = [f.name for f in exp_dir.glob("*.csv")]

    return {
        "name": exp_dir.name,
        "metrics": metrics,
        "images": images,
        "csv_files": csv_files,
    }