"""
Results scanner for reading experiment outputs.
Reuses logic from scripts/summarize_results.py but returns JSON-friendly dicts.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import math

from fl_backdoor.utils.metrics import auc as _auc, roc_curve as _roc_curve
from fl_backdoor.utils.paths import child_path, direct_files, experiment_path
from fl_backdoor.utils.run_artifacts import read_run_metadata

# ----------------------------------------------------------------------
# CSV readers; ROC/AUC is shared with the command-line reports and plots.
# ----------------------------------------------------------------------
def compute_auc_from_clients_csv(exp_dir: Path) -> Optional[float]:
    clients_csv = None
    for f in direct_files(exp_dir, "*_clients.csv"):
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


def find_main_csv(exp_dir: Path) -> Optional[Path]:
    for f in direct_files(exp_dir, "*.csv"):
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

    metrics_csv = child_path(exp_dir, f"{main_csv.stem}_metrics.csv")
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
    try:
        snapshot_path = child_path(exp_dir, "config_snapshot.toml")
        if snapshot_path.exists():
            import tomli
            with open(snapshot_path, "rb") as f:
                config = tomli.load(f)
            tool_config = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
            dataset = tool_config.get("dataset")
    except (ValueError, OSError):
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
    if not results_dir.exists():
        return experiments
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        try:
            exp_dir = experiment_path(results_dir, exp_dir.name)
        except ValueError:
            continue
        metrics = extract_experiment_metrics(exp_dir)
        if metrics is None:
            continue

        # 获取目录创建时间
        created_timestamp = os.path.getctime(exp_dir)
        created = datetime.fromtimestamp(created_timestamp).isoformat()

        # 尝试获取数据集名称
        dataset = None
        try:
            snapshot_path = child_path(exp_dir, "config_snapshot.toml")
            if snapshot_path.exists():
                import tomli
                with open(snapshot_path, "rb") as f:
                    config = tomli.load(f)
                tool_config = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
                dataset = tool_config.get("dataset")
        except (ValueError, OSError):
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

        try:
            metadata = read_run_metadata(exp_dir)
        except (ValueError, OSError):
            metadata = {}  # A broken manifest must not hide other experiments.
        experiments.append({
            "name": exp_dir.name,
            "run_id": metadata.get("run_id"),
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
    images = [f.name for f in direct_files(exp_dir, "*.png")]
    csv_files = [f.name for f in direct_files(exp_dir, "*.csv")]

    return {
        "name": exp_dir.name,
        "run_id": read_run_metadata(exp_dir).get("run_id"),
        "model_file": "final_model.pt" if child_path(exp_dir, "final_model.pt").is_file() else None,
        "metrics": metrics,
        "images": images,
        "csv_files": csv_files,
    }

def _clean_for_json(obj):
    """递归地将对象中的 NaN、Infinity 转换为 None，确保 JSON 兼容。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    else:
        return obj

def get_summary_detail(summary_dir: Path) -> Optional[dict[str, Any]]:
    """Return detailed data for a summary: summary table, and curve data for all experiments."""
    summary_csv = child_path(summary_dir, "summary.csv")
    if not summary_csv.exists():
        return None

    df_summary = pd.read_csv(summary_csv)
    if df_summary.empty:
        return None

    # 将 DataFrame 中的 NaN 替换为 None，以便 JSON 序列化
    df_summary = df_summary.replace({np.nan: None})

    # 收集每个实验的曲线数据
    experiments_curve_data = {}
    for exp_name in df_summary["experiment"]:
        # Summary lives at results/summaries/name, not results/name.
        try:
            exp_dir = experiment_path(summary_dir.parent.parent, exp_name)
        except ValueError:
            continue
        if exp_dir.exists():
            main_csv = find_main_csv(exp_dir)
            if main_csv:
                df_main = pd.read_csv(main_csv)
                if not df_main.empty and "round" in df_main.columns:
                    # 提取曲线数据，并替换 NaN
                    curve_data = df_main[["round", "accuracy", "asr", "loss"]].replace({np.nan: None}).to_dict(orient="list")
                    experiments_curve_data[exp_name] = curve_data

    # 汇总表格数据
    summary_rows = df_summary.to_dict(orient="records")

    # 图片列表
    images = [f.name for f in direct_files(summary_dir, "*.png")]

    result = {
        "name": summary_dir.name,
        "summary_table": summary_rows,
        "experiments_curve_data": experiments_curve_data,
        "images": images,
    }

    # 最终再清理一次，确保万无一失
    return _clean_for_json(result)
