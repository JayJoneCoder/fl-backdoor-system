"""
Plot experiment curves and detection diagnostics from CSV logs.

Supported CSV types:
1) Main experiment log:
   columns = round, accuracy, asr, loss

2) Detection / aggregation metrics log:
   columns = round, component, key, value

3) Per-client diagnostics log:
   columns = round, client_id, is_malicious, score, norm, norm_z, cosine, suspicious, kept

Usage:
    python plot.py results/*.csv
    python plot.py results/baseline.csv results/badnets.csv
    python plot.py results/*_clients.csv
    python plot.py results/*_metrics.csv
    python plot.py results/                # 自动生成多实验对比图
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize

# 图表面向中文用户：优先使用系统中常见的中文字体，避免中文显示为方框。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 指标键名保留英文以对应 CSV，图例统一显示中文名称。
METRIC_LABELS = {
    "detect_tp": "真阳性 (TP)",
    "detect_fp": "假阳性 (FP)",
    "detect_fn": "假阴性 (FN)",
    "detect_tn": "真阴性 (TN)",
    "detect_precision": "精确率",
    "detect_recall": "召回率",
    "detect_fpr": "假正例率",
    "detect_total_clients": "总客户端数",
    "detect_kept_clients": "保留客户端数",
    "detect_filtered_clients": "过滤客户端数",
    "detect_suspicious_clients": "可疑客户端数",
    "detect_raw_suspicious_count": "原始可疑数",
    "detect_skip_count": "跳过轮次数",
    "detect_mean_score": "平均分数",
    "detect_max_score": "最高分数",
    "detect_min_score": "最低分数",
    "detect_mean_cosine": "平均余弦相似度",
    "detect_min_cosine": "最低余弦相似度",
    "detect_mean_update_norm": "平均更新范数",
    "detect_max_update_norm": "最大更新范数",
    "agg_total_clients": "总客户端数",
    "agg_clipped_clients": "裁剪客户端数",
    "agg_clipped_ratio": "裁剪比例",
    "agg_avg_update_norm": "平均更新范数",
    "agg_max_update_norm": "最大更新范数",
    "agg_avg_post_norm": "聚合后平均范数",
    "agg_selected_clients": "选中客户端数",
    "agg_malicious_removal_rate": "恶意客户端移除率",
    "agg_benign_removal_rate": "良性客户端移除率",
    "agg_removed_malicious": "移除恶意客户端数",
    "agg_removed_benign": "移除良性客户端数",
    "agg_removed_clients": "移除客户端数",
    "agg_kept_malicious": "保留恶意客户端数",
    "agg_kept_benign": "保留良性客户端数",
    "agg_kept_clients": "保留客户端数",
    "client_defense_applied": "客户端防御启用",
    "client_defense_filtered_samples": "过滤样本数",
    "client_defense_kept_samples": "保留样本数",
    "client_defense_filter_ratio": "过滤比例",
    "client_defense_mean_score": "平均分数",
    "client_defense_max_score": "最高分数",
    "client_defense_label_aware": "标签感知",
    "client_defense_label_blend_alpha": "标签混合系数",
    "precision": "精确率",
    "recall": "召回率",
    "fpr": "假正例率",
    "auc": "AUC",
}


def _metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fl_backdoor.utils.metrics import auc as _auc, roc_curve as _roc_curve
RESULTS_DIR = PROJECT_ROOT / "results"

# -----------------------------
# Helpers
# -----------------------------
def _safe_read_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[SKIP] {csv_path}: failed to read CSV ({e})")
        return None


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _force_integer_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _legend_outside(ax: plt.Axes, *, title: str | None = None, ncol: int = 1) -> None:
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        title=title,
        ncol=ncol,
    )


# -----------------------------
# Data conversion helpers
# -----------------------------
def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _as_bool_series(s: pd.Series) -> pd.Series:
    """Normalize common CSV encodings to bool-like ints."""
    if s.dtype == bool:
        return s.astype(int)

    out = pd.to_numeric(s, errors="coerce")
    if out.notna().any():
        return out.fillna(0).astype(int)

    lowered = s.astype(str).str.lower().str.strip()
    return lowered.isin({"1", "true", "yes", "y", "t"}).astype(int)


def _round_label(r: float | int) -> str:
    try:
        rr = float(r)
        if rr.is_integer():
            return f"第 {int(rr)} 轮"
        return f"第 {rr:.2f} 轮"
    except Exception:
        return f"第 {r} 轮"


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").dropna().mean()) if len(series) else 0.0


# -----------------------------
# Main log plots
# -----------------------------
def plot_main_log(csv_path: Path, df: pd.DataFrame) -> None:
    required = {"round", "accuracy", "asr", "loss"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {csv_path}: missing main-log columns {sorted(required - set(df.columns))}")
        return

    df = _to_numeric(df, ["round", "accuracy", "asr", "loss"]).dropna(
        subset=["round", "accuracy", "asr", "loss"]
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["round", "accuracy", "asr", "loss"]
    )
    df = df[(df["round"] >= 0) & (df["round"] % 1 == 0)]
    df = df.drop_duplicates("round", keep="last").sort_values("round")
    if df.empty:
        print(f"[SKIP] {csv_path}: no valid rows after cleaning")
        return

    out_acc = csv_path.with_name(csv_path.stem + "_acc.png")
    out_asr = csv_path.with_name(csv_path.stem + "_asr.png")
    out_acc_vs_asr = csv_path.with_name(csv_path.stem + "_acc_vs_asr.png")
    out_loss = csv_path.with_name(csv_path.stem + "_loss.png")

    # Accuracy
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["accuracy"], marker="o")
    ax.set_xlabel("轮次")
    ax.set_ylabel("准确率")
    ax.set_title(f"准确率曲线（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _savefig(out_acc)

    # ASR
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["asr"], marker="o")
    ax.set_xlabel("轮次")
    ax.set_ylabel("攻击成功率（ASR）")
    ax.set_title(f"攻击成功率曲线（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _savefig(out_asr)

    # ACC vs ASR
    fig, ax = plt.subplots()
    ax.plot(df["asr"], df["accuracy"], marker="o")
    ax.set_xlabel("攻击成功率（ASR）")
    ax.set_ylabel("准确率")
    ax.set_title(f"准确率与攻击成功率（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    _savefig(out_acc_vs_asr)

    # Loss
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["loss"], marker="o")
    ax.set_xlabel("轮次")
    ax.set_ylabel("损失")
    ax.set_title(f"损失曲线（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _savefig(out_loss)

    print(f"[OK] main log plotted: {csv_path}")


# -----------------------------
# Client diagnostics plots
# -----------------------------
def _plot_grouped_scatter_by_round(
    csv_path: Path,
    df: pd.DataFrame,
    *,
    ycol: str,
    ylabel: str,
    title: str,
    outfile_suffix: str,
    legend_round_threshold: int = 8,
    color_by: str | None = None,
) -> None:
    unique_rounds = sorted(df["round"].dropna().unique().tolist())
    out_path = csv_path.with_name(csv_path.stem + outfile_suffix)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.grid(True, alpha=0.3)

    if len(unique_rounds) <= legend_round_threshold and color_by is None:
        for r, g in df.groupby("round", sort=True):
            ax.scatter(
                g["client_id"],
                g[ycol],
                s=28,
                alpha=0.85,
                label=_round_label(r),
            )
        _legend_outside(ax, title="轮次")
    else:
        rounds = df["round"].to_numpy(dtype=float)

        if color_by is not None and color_by in df.columns:
            cvals = pd.to_numeric(df[color_by], errors="coerce").fillna(0).to_numpy(dtype=float)
            sc = ax.scatter(
                df["client_id"],
                df[ycol],
                c=cvals,
                cmap="viridis",
                s=28,
                alpha=0.85,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(_metric_label(color_by))
        else:
            norm = Normalize(
                vmin=float(pd.Series(rounds).min()),
                vmax=float(pd.Series(rounds).max()),
            )
            cmap = plt.get_cmap("viridis")
            sc = ax.scatter(
                df["client_id"],
                df[ycol],
                c=rounds,
                cmap=cmap,
                norm=norm,
                s=28,
                alpha=0.85,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("轮次")
            cbar.locator = mticker.MaxNLocator(integer=True)
            cbar.update_ticks()

    ax.set_xlabel("客户端 ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _savefig(out_path)


def plot_client_log(csv_path: Path, df: pd.DataFrame) -> None:
    required = {"round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {csv_path}: missing client-log columns {sorted(required - set(df.columns))}")
        return

    df = df.copy()
    df = _to_numeric(
        df,
        ["round", "client_id", "is_malicious", "score", "norm", "norm_z", "cosine", "suspicious", "kept"],
    ).dropna(subset=["round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"])

    if "is_malicious" not in df.columns:
        df["is_malicious"] = -1
    else:
        df["is_malicious"] = pd.to_numeric(df["is_malicious"], errors="coerce").fillna(-1).astype(int)

    if df.empty:
        print(f"[SKIP] {csv_path}: no valid rows after cleaning")
        return

    df["round"] = df["round"].astype(int)
    df["client_id"] = df["client_id"].astype(int)
    df["suspicious"] = df["suspicious"].astype(int)
    df["kept"] = df["kept"].astype(int)
    df["is_malicious"] = df["is_malicious"].astype(int)
    df = df.sort_values(["round", "client_id"])

    out_score_hist = csv_path.with_name(csv_path.stem + "_score_hist.png")
    out_score_box = csv_path.with_name(csv_path.stem + "_score_box.png")
    out_score_norm_scatter = csv_path.with_name(csv_path.stem + "_score_vs_norm.png")
    out_score_cosine_scatter = csv_path.with_name(csv_path.stem + "_score_vs_cosine.png")
    out_label_scatter = csv_path.with_name(csv_path.stem + "_score_by_gt.png")

    # 1) Score by round
    _plot_grouped_scatter_by_round(
        csv_path,
        df,
        ycol="score",
        ylabel="分数",
        title=f"客户端分数随轮次变化（{csv_path.stem}）",
        outfile_suffix="_score_by_round.png",
    )

    # 2) Norm by round
    _plot_grouped_scatter_by_round(
        csv_path,
        df,
        ycol="norm",
        ylabel="更新范数",
        title=f"客户端更新范数随轮次变化（{csv_path.stem}）",
        outfile_suffix="_norm_by_round.png",
    )

    # 3) Cosine by round
    _plot_grouped_scatter_by_round(
        csv_path,
        df,
        ycol="cosine",
        ylabel="余弦相似度",
        title=f"客户端余弦相似度随轮次变化（{csv_path.stem}）",
        outfile_suffix="_cosine_by_round.png",
    )

    # 4) Score histogram: suspicious vs benign
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    benign_scores = df.loc[df["suspicious"] == 0, "score"]
    suspicious_scores = df.loc[df["suspicious"] == 1, "score"]

    if len(benign_scores) > 0:
        ax.hist(benign_scores, bins=20, alpha=0.65, label="良性")
    if len(suspicious_scores) > 0:
        ax.hist(suspicious_scores, bins=20, alpha=0.65, label="可疑")
    ax.set_xlabel("分数")
    ax.set_ylabel("数量")
    ax.set_title(f"分数分布（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    if len(benign_scores) > 0 or len(suspicious_scores) > 0:
        _legend_outside(ax)
    _savefig(out_score_hist)

    # 5) Suspicious vs Malicious count by round (merged)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    suspicious_by_round = df.groupby("round", as_index=False)["suspicious"].sum()
    malicious_by_round = (
        df.loc[df["is_malicious"] >= 0]
        .groupby("round", as_index=False)["is_malicious"]
        .sum()
    )

    ax.plot(suspicious_by_round["round"], suspicious_by_round["suspicious"],
            marker="o", linestyle="-", linewidth=2, markersize=6,
            label="可疑客户端（检测结果）", color="red")

    if not malicious_by_round.empty:
        ax.plot(malicious_by_round["round"], malicious_by_round["is_malicious"],
                marker="s", linestyle="--", linewidth=2, markersize=6,
                label="恶意客户端（真实标签）", color="blue")

    ax.set_xlabel("轮次")
    ax.set_ylabel("数量")
    ax.set_title(f"每轮可疑客户端与恶意客户端数量（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _force_integer_axis(ax)
    _savefig(csv_path.with_name(csv_path.stem + "_suspicious_vs_malicious.png"))

    # 6) Boxplot: score grouped by suspicious label
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    benign_scores = df.loc[df["suspicious"] == 0, "score"].to_list()
    suspicious_scores = df.loc[df["suspicious"] == 1, "score"].to_list()
    groups = []
    labels = []
    if benign_scores:
        groups.append(benign_scores)
        labels.append("良性")
    if suspicious_scores:
        groups.append(suspicious_scores)
        labels.append("可疑")
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_ylabel("分数")
        ax.set_title(f"按检测标签分组的分数箱线图（{csv_path.stem}）")
        ax.grid(True, axis="y", alpha=0.3)
        _savefig(out_score_box)
    else:
        plt.close(fig)

    # 7) Score vs norm scatter
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(
        df["norm"],
        df["score"],
        c=df["suspicious"],
        cmap="coolwarm",
        alpha=0.85,
        s=28,
    )
    ax.set_xlabel("更新范数")
    ax.set_ylabel("分数")
    ax.set_title(f"分数与更新范数（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("可疑标记（0=良性，1=可疑）")
    _savefig(out_score_norm_scatter)

    # 8) Score vs cosine scatter
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(
        df["cosine"],
        df["score"],
        c=df["suspicious"],
        cmap="coolwarm",
        alpha=0.85,
        s=28,
    )
    ax.set_xlabel("余弦相似度")
    ax.set_ylabel("分数")
    ax.set_title(f"分数与余弦相似度（{csv_path.stem}）")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("可疑标记（0=良性，1=可疑）")
    _savefig(out_score_cosine_scatter)

    # 9) Score colored by GT malicious label
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    plot_df = df.loc[df["is_malicious"] >= 0].copy()
    if not plot_df.empty:
        sc = ax.scatter(
            plot_df["client_id"],
            plot_df["score"],
            c=plot_df["is_malicious"],
            cmap="coolwarm",
            alpha=0.85,
            s=28,
        )
        ax.set_xlabel("客户端 ID")
        ax.set_ylabel("分数")
        ax.set_title(f"按真实标签着色的客户端分数（{csv_path.stem}）")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("是否为恶意客户端（0=良性，1=恶意）")
        _force_integer_axis(ax)
        _savefig(out_label_scatter)
    else:
        plt.close(fig)

    print(f"[OK] client log plotted: {csv_path}")


def plot_roc_from_clients_csv(csv_path: Path, df: pd.DataFrame) -> None:
    """Plot ROC curve and compute AUC from client-level predictions."""
    required = {"suspicious", "is_malicious", "score"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {csv_path}: missing columns for ROC {sorted(required - set(df.columns))}")
        return

    df_valid = df[df["is_malicious"].isin([0, 1])].copy()
    df_valid["score"] = pd.to_numeric(df_valid["score"], errors="coerce")
    df_valid = df_valid[np.isfinite(df_valid["score"])]
    if df_valid.empty:
        print(f"[SKIP] {csv_path}: no valid ground truth for ROC")
        return

    y_true = df_valid["is_malicious"].astype(int).values
    y_score = df_valid["score"].values

    fpr, tpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _auc(fpr, tpr)
    if not np.isfinite(roc_auc):
        print(f"[SKIP] {csv_path}: ROC requires both benign and malicious clients")
        return

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(fpr, tpr, lw=2, label=f'ROC 曲线（AUC = {roc_auc:.3f}）')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测（AUC=0.5）')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正例率（FPR）')
    ax.set_ylabel('真正例率（TPR）')
    ax.set_title(f'ROC 曲线 - {csv_path.stem}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out_path = csv_path.with_name(csv_path.stem + "_roc.png")
    _savefig(out_path)
    print(f"[OK] ROC curve saved: {out_path} (AUC={roc_auc:.3f})")


# -----------------------------
# Metrics log plots
# -----------------------------
def _metrics_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert [round, component, key, value] rows into wide format:
    index = round, columns = key, values = numeric value
    """
    out = df.copy()
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out = out.dropna(subset=["round", "key"])
    if out.empty:
        return pd.DataFrame()

    out["value_num"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value_num"])
    if out.empty:
        return pd.DataFrame()

    wide = (
        out.pivot_table(index="round", columns="key", values="value_num", aggfunc="last")
        .reset_index()
        .sort_values("round")
    )
    return wide


def _plot_metric_lines(
    csv_path: Path,
    wide: pd.DataFrame,
    keys: list[str],
    *,
    title: str,
    ylabel: str,
    outfile_suffix: str,
) -> None:
    if wide.empty:
        return

    present = [k for k in keys if k in wide.columns]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k in present:
        ax.plot(wide["round"], wide[k], marker="o", label=_metric_label(k))

    ax.set_xlabel("轮次")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _legend_outside(ax, ncol=1)
    _savefig(csv_path.with_name(csv_path.stem + outfile_suffix))


def plot_metrics_log(csv_path: Path, df: pd.DataFrame) -> None:
    required = {"round", "component", "key", "value"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {csv_path}: missing metrics-log columns {sorted(required - set(df.columns))}")
        return

    df = df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round", "component", "key"])
    if df.empty:
        print(f"[SKIP] {csv_path}: no valid rows after cleaning")
        return

    wide = _metrics_pivot(df)
    if wide.empty:
        print(f"[SKIP] {csv_path}: no numeric metric rows after pivot")
        return

    # 1) Detection quality metrics
    detection_keys = [
        "detect_tp",
        "detect_fp",
        "detect_fn",
        "detect_tn",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        detection_keys,
        title=f"检测混淆计数（{csv_path.stem}）",
        ylabel="数量",
        outfile_suffix="_detection_confusion.png",
    )

    detection_rate_keys = [
        "detect_precision",
        "detect_recall",
        "detect_fpr",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        detection_rate_keys,
        title=f"检测比率（{csv_path.stem}）",
        ylabel="比率",
        outfile_suffix="_detection_rates.png",
    )

    # 2) Detection summary metrics
    summary_keys = [
        "detect_total_clients",
        "detect_kept_clients",
        "detect_filtered_clients",
        "detect_suspicious_clients",
        "detect_raw_suspicious_count",
        "detect_skip_count",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        summary_keys,
        title=f"检测汇总（{csv_path.stem}）",
        ylabel="数量",
        outfile_suffix="_detection_summary.png",
    )

    # 3) Threshold / score related metrics
    score_keys = [
        "detect_mean_score",
        "detect_max_score",
        "detect_min_score",
        "detect_mean_cosine",
        "detect_min_cosine",
        "detect_mean_update_norm",
        "detect_max_update_norm",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        score_keys,
        title=f"检测分数统计（{csv_path.stem}）",
        ylabel="数值",
        outfile_suffix="_detection_scores.png",
    )

    # 4) Aggregation metrics
    agg_keys = [
        "agg_total_clients",
        "agg_clipped_clients",
        "agg_clipped_ratio",
        "agg_avg_update_norm",
        "agg_max_update_norm",
        "agg_avg_post_norm",
        "agg_selected_clients",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        agg_keys,
        title=f"聚合指标（{csv_path.stem}）",
        ylabel="数值",
        outfile_suffix="_aggregation_metrics.png",
    )

    # 5) Client-defense metrics
    client_keys = [
        "client_defense_applied",
        "client_defense_filtered_samples",
        "client_defense_kept_samples",
        "client_defense_filter_ratio",
        "client_defense_mean_score",
        "client_defense_max_score",
        "client_defense_label_aware",
        "client_defense_label_blend_alpha",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        client_keys,
        title=f"客户端防御指标（{csv_path.stem}）",
        ylabel="数值",
        outfile_suffix="_client_defense_metrics.png",
    )

    # 6) Compact all-in-one detection snapshot
    wanted_keys = [
        "detect_tp",
        "detect_fp",
        "detect_fn",
        "detect_precision",
        "detect_recall",
        "detect_fpr",
        "detect_suspicious_clients",
        "detect_kept_clients",
        "detect_filtered_clients",
    ]
    present = [k for k in wanted_keys if k in wide.columns]
    if present:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        for k in present:
            ax.plot(wide["round"], wide[k], marker="o", label=_metric_label(k))
        ax.set_xlabel("轮次")
        ax.set_title(f"检测总览（{csv_path.stem}）")
        ax.grid(True, alpha=0.3)
        _force_integer_axis(ax)
        _legend_outside(ax, ncol=2)
        _savefig(csv_path.with_name(csv_path.stem + "_detection_overview.png"))

    # 7) Aggregation filtering metrics (removal rates, removed counts, kept counts)
    agg_removal_rate_keys = [
        "agg_malicious_removal_rate",
        "agg_benign_removal_rate",
    ]
    if any(k in wide.columns for k in agg_removal_rate_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_removal_rate_keys,
            title=f"聚合移除率（{csv_path.stem}）",
            ylabel="比率",
            outfile_suffix="_aggregation_removal_rates.png",
        )

    agg_removed_count_keys = [
        "agg_removed_malicious",
        "agg_removed_benign",
        "agg_removed_clients",
    ]
    if any(k in wide.columns for k in agg_removed_count_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_removed_count_keys,
            title=f"聚合移除客户端数（{csv_path.stem}）",
            ylabel="数量",
            outfile_suffix="_aggregation_removed_counts.png",
        )

    agg_kept_count_keys = [
        "agg_kept_malicious",
        "agg_kept_benign",
        "agg_kept_clients",
    ]
    if any(k in wide.columns for k in agg_kept_count_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_kept_count_keys,
            title=f"聚合保留客户端数（{csv_path.stem}）",
            ylabel="数量",
            outfile_suffix="_aggregation_kept_counts.png",
        )

    print(f"[OK] metrics log plotted: {csv_path}")


# -----------------------------
# Multi-experiment comparison plots
# -----------------------------
def plot_multi_experiment_curves(results_dir: Path, metric: str = "accuracy", exp_filter: list[str] = None, output_dir: Path = None) -> None:
    exp_data = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "summaries":
            continue
        if exp_filter is not None and exp_dir.name not in exp_filter:
            continue
        main_csv = None
        for f in exp_dir.glob("*.csv"):
            if "_metrics" not in f.name and "_clients" not in f.name:
                main_csv = f
                break
        if main_csv is None:
            continue
        df = pd.read_csv(main_csv)
        if df.empty or "round" not in df.columns or metric not in df.columns:
            continue
        exp_data.append((exp_dir.name, df))

    if not exp_data:
        print(f"No valid experiment data found in {results_dir} for metric {metric}.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for exp_name, df in exp_data:
        ax.plot(df["round"], df[metric], marker='o', label=exp_name)

    ax.set_xlabel("轮次")
    ylabel = "准确率" if metric == "accuracy" else "攻击成功率（ASR）"
    ax.set_ylabel(ylabel)
    ax.set_title(f"各实验{ylabel}对比")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _legend_outside(ax, ncol=1)
    out_path = (output_dir or results_dir) / f"comparison_{metric}.png"
    _savefig(out_path)
    print(f"Saved {metric} comparison plot to {out_path}")


def plot_multi_experiment_aggregation_metric(results_dir: Path, metric_key: str, ylabel: str, filename: str, exp_filter: list[str] = None, output_dir: Path = None):
    """
    Scan experiment subdirectories, extract a specific aggregation metric from metrics CSV,
    and plot the curves across experiments.
    """
    exp_data = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "summaries":
            continue
        if exp_filter is not None and exp_dir.name not in exp_filter:
            continue
        metrics_csv = None
        for f in exp_dir.glob("*_metrics.csv"):
            metrics_csv = f
            break
        if metrics_csv is None:
            continue
        df = _safe_read_csv(metrics_csv)
        if df is None or df.empty or not {"round", "component", "key", "value"}.issubset(df.columns):
            continue
        rows = df[(df["component"] == "aggregation") & (df["key"] == metric_key)]
        if rows.empty:
            continue
        df_metric = rows[["round", "value"]].copy()
        df_metric = _to_numeric(df_metric, ["round", "value"])
        df_metric = df_metric.replace([np.inf, -np.inf], np.nan).dropna()
        if df_metric.empty:
            continue
        exp_data.append((exp_dir.name, df_metric))

    if not exp_data:
        print(f"No experiments found with metric {metric_key}.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for exp_name, df_metric in exp_data:
        ax.plot(df_metric["round"], df_metric["value"], marker='o', label=exp_name)
    ax.set_xlabel("轮次")
    ax.set_ylabel(ylabel)
    ax.set_title(f"各实验{ylabel}对比")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _legend_outside(ax, ncol=1)
    out_path = (output_dir or results_dir) / filename
    _savefig(out_path)
    print(f"Saved {metric_key} comparison plot to {out_path}")

def plot_multi_experiment_summary(summary_csv: Path) -> None:
    """Generate summary comparison plots from summary.csv."""
    if not summary_csv.exists():
        print(f"[SKIP] {summary_csv} not found, cannot generate summary plots.")
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    
    # 1) ACC vs ASR scatter plot
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    # 区分是否有检测
    has_detection = df["precision"].notna() if "precision" in df.columns else [False]*len(df)
    colors = ["red" if d else "blue" for d in has_detection]
    ax.scatter(df["asr"], df["accuracy"], c=colors, s=80, alpha=0.7)
        # 添加标签
    for _, row in df.iterrows():
        ax.annotate(row["experiment"], (row["asr"], row["accuracy"]), fontsize=8)
    ax.set_xlabel("攻击成功率（最终轮次）")
    ax.set_ylabel("准确率（最终轮次）")
    ax.set_title("防御效果：准确率与攻击成功率")
    ax.grid(alpha=0.3)
    # 添加理想区域标记
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    out_path = summary_csv.parent / "summary_acc_vs_asr.png"
    _savefig(out_path)
    print(f"Saved ACC vs ASR summary plot to {out_path}")
    
    # 2) Detection metrics bar chart (if precision column exists)
    if "precision" in df.columns and df["precision"].notna().any():
        det_df = df[df["precision"].notna()].copy()
        if not det_df.empty:
            metrics = ["precision", "recall", "fpr", "auc"]
            present = [m for m in metrics if m in det_df.columns]
            if present:
                fig, ax = plt.subplots(figsize=(8.0, 5.0))
                x = np.arange(len(det_df))
                width = 0.2
                for i, m in enumerate(present):
                    ax.bar(x + i*width, det_df[m], width, label=_metric_label(m))
                ax.set_xticks(x + width*(len(present)-1)/2)
                ax.set_xticklabels(det_df["experiment"], rotation=45, ha="right")
                ax.set_ylabel("分数")
                ax.set_title("检测性能对比")
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                out_path = summary_csv.parent / "summary_detection_metrics.png"
                _savefig(out_path)
                print(f"Saved detection metrics bar chart to {out_path}")
    
    # 3) Aggregation metrics bar chart (average removal rates, kept malicious, etc.)
    agg_avg_columns = [
        "avg_malicious_removal_rate",
        "avg_benign_removal_rate",
        "avg_kept_malicious",
        "avg_selected_clients",
    ]
    # 找出实际存在的聚合平均列
    present_agg = [c for c in agg_avg_columns if c in df.columns and df[c].notna().any()]
    if present_agg:
        # 只保留有聚合数据的行（至少一个聚合列非空）
        agg_df = df.dropna(subset=present_agg, how='all').copy()
        if not agg_df.empty:
            # 重命名列名以便绘图显示更友好
            rename_map = {
                "avg_malicious_removal_rate": "恶意移除率",
                "avg_benign_removal_rate": "良性移除率",
                "avg_kept_malicious": "平均保留恶意数",
                "avg_selected_clients": "平均选中客户端数",
            }
            fig, ax = plt.subplots(figsize=(max(8, len(agg_df)*0.8), 5.0))
            x = np.arange(len(agg_df))
            width = 0.8 / len(present_agg) if len(present_agg) > 1 else 0.6
            for i, col in enumerate(present_agg):
                offset = (i - (len(present_agg)-1)/2) * width
                ax.bar(x + offset, agg_df[col], width, label=rename_map.get(col, col))
            ax.set_xticks(x)
            ax.set_xticklabels(agg_df["experiment"], rotation=45, ha="right")
            ax.set_ylabel("数值")
            ax.set_title("聚合性能对比（平均值）")
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            out_path = summary_csv.parent / "summary_aggregation_metrics.png"
            _savefig(out_path)
            print(f"Saved aggregation metrics bar chart to {out_path}")

# -----------------------------
# Dispatcher
# -----------------------------
def plot_csv(csv_path: Path) -> None:
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        print(f"[SKIP] {csv_path}: empty or unreadable")
        return

    cols = set(df.columns)

    if {"round", "accuracy", "asr", "loss"}.issubset(cols):
        plot_main_log(csv_path, df)
        return

    if {"round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"}.issubset(cols):
        plot_client_log(csv_path, df)
        plot_roc_from_clients_csv(csv_path, df)
        return

    if {"round", "component", "key", "value"}.issubset(cols):
        plot_metrics_log(csv_path, df)
        return

    print(f"[SKIP] {csv_path}: unknown CSV schema -> columns={sorted(cols)}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python plot.py <csv-or-glob> [more_paths...]")
        return 1

    paths: list[Path] = []
    for arg in argv[1:]:
        p = Path(arg)
        if any(ch in arg for ch in ["*", "?", "["]):
            paths.extend(sorted(Path().glob(arg)))
        elif p.is_dir():
            summary_csv = p / "summary.csv"
            if summary_csv.exists():
                print(f"[INFO] Detected summary directory: {p}")
                df = pd.read_csv(summary_csv)
                if "experiment" in df.columns:
                    # A summary is nested at <configured-root>/summaries/<name>.
                    results_root = p.resolve().parent.parent
                    exp_names = df["experiment"].tolist()
                    print(f"[INFO] Found experiments: {exp_names}")
                    # 生成多实验对比曲线（ACC/ASR）
                    plot_multi_experiment_curves(results_root, metric="accuracy", exp_filter=exp_names, output_dir=p)
                    plot_multi_experiment_curves(results_root, metric="asr", exp_filter=exp_names, output_dir=p)
                    # 生成聚合指标对比图
                    plot_multi_experiment_aggregation_metric(
                        results_root, exp_filter=exp_names, output_dir=p,
                        metric_key="agg_malicious_removal_rate",
                        ylabel="恶意客户端移除率",
                        filename="comparison_malicious_removal_rate.png"
                    )
                    plot_multi_experiment_aggregation_metric(
                        results_root, exp_filter=exp_names, output_dir=p,
                        metric_key="agg_benign_removal_rate",
                        ylabel="良性客户端移除率",
                        filename="comparison_benign_removal_rate.png"
                    )
                # 生成总结图表（散点图、条形图）
                plot_multi_experiment_summary(summary_csv)
            # 继续收集该目录下的 CSV（用于单实验图）
            paths.extend(sorted(p.rglob("*.csv")))
        else:
            paths.append(p)

    # 处理收集到的所有 CSV 文件（生成单实验图）
    if paths:
        seen = set()
        unique_paths = []
        for p in paths:
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                unique_paths.append(p)

        for csv_path in unique_paths:
            if csv_path.suffix.lower() != ".csv":
                continue
            if not csv_path.exists():
                print(f"[SKIP] {csv_path}: not found")
                continue
            plot_csv(csv_path)

    # 如果第一个参数是 results 目录（非总结目录），也生成全局对比图
    first_arg = Path(argv[1])
    if first_arg.is_dir() and first_arg.name == "results":
        subdirs = [d for d in first_arg.iterdir() if d.is_dir() and d.name != "summaries"]
        if len(subdirs) >= 2:
            plot_multi_experiment_curves(first_arg, metric="accuracy")
            plot_multi_experiment_curves(first_arg, metric="asr")
            summary_csv = first_arg / "summary.csv"
            if summary_csv.exists():
                plot_multi_experiment_summary(summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
