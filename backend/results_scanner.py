"""Read result artifacts independently so one damaged file cannot hide a run."""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tomli

from fl_backdoor.utils.metrics import auc as _auc, roc_curve as _roc_curve
from fl_backdoor.utils.paths import child_path, direct_files, experiment_path
from fl_backdoor.utils.run_artifacts import read_run_metadata

CURVE_COLUMNS = ("round", "accuracy", "asr", "loss")
READ_ERRORS = (OSError, ValueError, UnicodeError, pd.errors.ParserError)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _read_csv(path: Path, warnings: list[str]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except READ_ERRORS:
        warnings.append(f"{path.name} 无法读取或格式损坏，已跳过该文件。")
        return None


def find_main_csv(exp_dir: Path) -> Optional[Path]:
    files = direct_files(exp_dir, "*.csv")
    expected = exp_dir / f"{exp_dir.name}.csv"
    if expected in files:
        return expected
    return next((f for f in sorted(files) if not f.stem.endswith(("_metrics", "_clients"))), None)


def _curves(frame: pd.DataFrame, warnings: list[str]) -> dict:
    missing = [key for key in CURVE_COLUMNS if key not in frame.columns]
    if missing:
        warnings.append(f"主指标 CSV 缺少列：{', '.join(missing)}；缺失指标显示为未记录。")
    numeric = frame.reindex(columns=CURVE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    rounds = numeric["round"]
    numeric = numeric[rounds.notna() & (rounds >= 0) & (rounds % 1 == 0)]
    if len(numeric) != len(frame):
        warnings.append("主指标 CSV 中轮次无效的记录已跳过。")
    if any(frame[key].notna().sum() > pd.to_numeric(frame[key], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).notna().sum() for key in CURVE_COLUMNS if key in frame):
        warnings.append("主指标 CSV 含无效数值，已显示为未记录。")
    numeric = numeric.drop_duplicates("round", keep="last").sort_values("round")
    return {key: [int(value) if key == "round" else _number(value) for value in numeric[key]]
            for key in CURVE_COLUMNS}


def can_plot_experiment(detail: dict) -> bool:
    """The plot script requires at least one complete main-metric row."""
    curves = detail.get("metrics", {}).get("curve_data", {})
    return any(all(value is not None for value in row)
               for row in zip(*(curves.get(key, []) for key in CURVE_COLUMNS)))


def compute_auc_from_clients_csv(exp_dir: Path) -> Optional[float]:
    for path in direct_files(exp_dir, "*_clients.csv"):
        frame = _read_csv(path, [])
        if frame is None or not {"score", "is_malicious"}.issubset(frame.columns):
            continue
        labels = pd.to_numeric(frame["is_malicious"], errors="coerce")
        scores = pd.to_numeric(frame["score"], errors="coerce")
        valid = labels.isin([0, 1]) & np.isfinite(scores)
        fpr, tpr, _ = _roc_curve(labels[valid].to_numpy(), scores[valid].to_numpy())
        return _number(_auc(fpr, tpr))
    return None


def extract_experiment_metrics(exp_dir: Path) -> Optional[dict[str, Any]]:
    """Keep valid main metrics even when optional client/detection files are broken."""
    main_csv = find_main_csv(exp_dir)
    if main_csv is None:
        return None
    warnings: list[str] = []
    frame = _read_csv(main_csv, warnings)
    if frame is not None and frame.empty:
        warnings.append("主指标 CSV 尚无数据，可查看运行状态和日志。")
    curves = {key: [] for key in CURVE_COLUMNS} if frame is None else _curves(frame, warnings)
    metrics = {key: values[-1] if values else None for key, values in curves.items() if key != "round"}
    last_round = curves["round"][-1] if curves["round"] else None
    metrics.update(last_round=last_round, curve_data=curves, warnings=warnings)
    has_aggregation = False
    path = child_path(exp_dir, f"{main_csv.stem}_metrics.csv")
    if path.is_file():
        extra = _read_csv(path, warnings)
        required = {"round", "component", "key", "value"}
        if extra is not None and not required.issubset(extra.columns):
            warnings.append(f"{path.name} 缺少必要列，检测与聚合统计未读取。")
        elif extra is not None:
            detection = extra[(pd.to_numeric(extra["round"], errors="coerce") == last_round)
                              & (extra["component"] == "detection")]
            for _, row in detection.iterrows():
                if row["key"] in {"detect_precision", "detect_recall", "detect_fpr", "detect_tp", "detect_fp", "detect_fn", "detect_tn"}:
                    metrics[row["key"].removeprefix("detect_")] = _number(row["value"])
            aggregation = extra[extra["component"] == "aggregation"]
            has_aggregation = not aggregation.empty
            for key in ("malicious_removal_rate", "benign_removal_rate", "kept_malicious", "removed_malicious",
                        "removed_benign", "kept_benign", "total_malicious", "total_benign", "selected_clients"):
                if has_aggregation:
                    values = pd.to_numeric(aggregation.loc[aggregation["key"] == f"agg_{key}", "value"], errors="coerce")
                    metrics[f"avg_{key}"] = _number(values.replace([np.inf, -np.inf], np.nan).mean())

    for path in direct_files(exp_dir, "*_clients.csv"):
        clients = _read_csv(path, warnings)
        if clients is None:
            continue
        if not {"round", "is_malicious"}.issubset(clients.columns):
            warnings.append(f"{path.name} 缺少必要列，客户端统计未读取。")
            continue
        labels = pd.to_numeric(clients["is_malicious"], errors="coerce")
        rounds = pd.to_numeric(clients["round"], errors="coerce")
        valid = labels.isin([0, 1]) & np.isfinite(rounds)
        counts = pd.DataFrame({"round": rounds[valid], "is_malicious": labels[valid]})
        if not has_aggregation and not counts.empty:
            malicious = counts.groupby("round")["is_malicious"].sum().mean()
            benign = counts.groupby("round").size().mean() - malicious
            metrics.update(avg_total_malicious=float(malicious), avg_total_benign=float(benign),
                           avg_kept_malicious=float(malicious), avg_kept_benign=float(benign),
                           avg_malicious_removal_rate=0.0, avg_benign_removal_rate=0.0,
                           avg_removed_malicious=0.0, avg_removed_benign=0.0, avg_selected_clients=0.0)
        if "score" in clients:
            scores = pd.to_numeric(clients["score"], errors="coerce")
            valid = labels.isin([0, 1]) & np.isfinite(scores)
            fpr, tpr, _ = _roc_curve(labels[valid].to_numpy(), scores[valid].to_numpy())
            metrics["auc"] = _number(_auc(fpr, tpr))
        break
    return _clean_for_json(metrics)


def _dataset(exp_dir: Path) -> str:
    try:
        snapshot = child_path(exp_dir, "config_snapshot.toml")
        if snapshot.is_file():
            config = tomli.loads(snapshot.read_text(encoding="utf-8"))
            value = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("dataset")
            if isinstance(value, str):
                return value
    except (OSError, ValueError, UnicodeError, AttributeError):
        pass
    name = exp_dir.name.lower()
    return next((dataset for dataset in ("cifar100", "cifar10", "mnist") if dataset in name), "")


def get_experiment_detail(exp_dir: Path) -> Optional[dict[str, Any]]:
    if not exp_dir.is_dir():
        return None
    warnings: list[str] = []
    try:
        metadata = read_run_metadata(exp_dir)
    except (OSError, ValueError, UnicodeError):
        metadata = {}
        warnings.append("运行记录无法读取，部分状态信息未记录。")
    try:
        metrics = extract_experiment_metrics(exp_dir)
    except (*READ_ERRORS, KeyError, TypeError, AttributeError):
        metrics = None
        warnings.append("指标文件无法完整解析，运行记录与其他文件仍可查看。")
    if metrics is None and not metadata and not child_path(exp_dir, "run.log").is_file():
        return None
    if metrics is None:
        metrics = {"accuracy": None, "asr": None, "loss": None, "last_round": None,
                   "curve_data": {key: [] for key in CURVE_COLUMNS}}
        warnings.append("尚无指标 CSV，可查看运行状态和日志。")
    warnings.extend(metrics.pop("warnings", []))
    metrics["dataset"] = _dataset(exp_dir)
    has_metrics = any(metrics.get(key) is not None for key in ("accuracy", "asr", "loss"))
    detail = {
        "name": exp_dir.name, "run_id": metadata.get("run_id"),
        "status": metadata.get("status", "unknown"), "error": metadata.get("error"),
        "exit_code": metadata.get("exit_code"),
        "created": metadata.get("created_at") or datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
        "has_metrics": has_metrics, "warnings": warnings,
        "model_file": "final_model.pt" if child_path(exp_dir, "final_model.pt").is_file() else None,
        "metrics": metrics, "images": [f.name for f in direct_files(exp_dir, "*.png")],
        "csv_files": [f.name for f in direct_files(exp_dir, "*.csv")],
    }
    detail["can_plot"] = can_plot_experiment(detail)
    return detail


def list_experiments(results_dir: Path) -> list[dict[str, Any]]:
    experiments = []
    if not results_dir.exists():
        return experiments
    for directory in results_dir.iterdir():
        try:
            detail = get_experiment_detail(experiment_path(results_dir, directory.name))
            if detail is None:
                continue
            experiments.append({**{key: detail[key] for key in (
                "name", "run_id", "status", "error", "exit_code", "created", "has_metrics", "can_plot", "warnings")},
                **{key: detail["metrics"].get(key) for key in (
                    "accuracy", "asr", "loss", "last_round", "precision", "recall", "fpr", "auc", "dataset")}})
        except (*READ_ERRORS, KeyError, TypeError, AttributeError):
            # A disappearing/unreadable directory must not hide other runs.
            continue
    experiments.sort(key=lambda item: str(item["created"] or ""), reverse=True)
    return experiments


def _clean_for_json(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _clean_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(value) for value in obj]
    return obj


def get_summary_detail(summary_dir: Path) -> Optional[dict[str, Any]]:
    summary_csv = child_path(summary_dir, "summary.csv")
    if not summary_csv.is_file():
        return None
    frame = _read_csv(summary_csv, [])
    if frame is None or frame.empty or "experiment" not in frame:
        return None
    curves = {}
    for name in frame["experiment"]:
        try:
            metrics = extract_experiment_metrics(experiment_path(summary_dir.parent.parent, name))
            if metrics and metrics["curve_data"]["round"]:
                curves[name] = metrics["curve_data"]
        except (*READ_ERRORS, TypeError):
            continue
    return _clean_for_json({"name": summary_dir.name, "summary_table": frame.to_dict(orient="records"),
                            "experiments_curve_data": curves,
                            "images": [f.name for f in direct_files(summary_dir, "*.png")]})