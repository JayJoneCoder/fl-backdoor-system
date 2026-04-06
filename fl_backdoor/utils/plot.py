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
# ROC and AUC (pure numpy, no sklearn)
# -----------------------------
def _roc_curve(y_true, y_score):
    """Compute ROC curve using numpy.

    Args:
        y_true: 1D array of true labels (0 or 1)
        y_score: 1D array of scores (higher = more suspicious)

    Returns:
        fpr, tpr, thresholds
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by score descending
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]

    # Unique thresholds (scores)
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
    thresholds = np.array([np.inf] + list(thresholds) + [-np.inf])

    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    return fpr, tpr, thresholds


def _auc(fpr, tpr):
    """Compute AUC using trapezoidal rule (pure numpy, no trapz/trapezoid)."""
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    auc_value = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2.0
        auc_value += width * height
    return auc_value


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
            return f"round {int(rr)}"
        return f"round {rr:.2f}"
    except Exception:
        return f"round {r}"


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

    df = df.sort_values("round")
    df = _to_numeric(df, ["round", "accuracy", "asr", "loss"]).dropna(
        subset=["round", "accuracy", "asr", "loss"]
    )
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
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Curve ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _savefig(out_acc)

    # ASR
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["asr"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("ASR")
    ax.set_title(f"ASR Curve ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _savefig(out_asr)

    # ACC vs ASR
    fig, ax = plt.subplots()
    ax.plot(df["asr"], df["accuracy"], marker="o")
    ax.set_xlabel("ASR")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ACC vs ASR ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    _savefig(out_acc_vs_asr)

    # Loss
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["loss"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve ({csv_path.stem})")
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
        _legend_outside(ax, title="Round")
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
            cbar.set_label(color_by)
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
            cbar.set_label("Round")
            cbar.locator = mticker.MaxNLocator(integer=True)
            cbar.update_ticks()

    ax.set_xlabel("Client ID")
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
        ylabel="Score",
        title=f"Client Score by Round ({csv_path.stem})",
        outfile_suffix="_score_by_round.png",
    )

    # 2) Norm by round
    _plot_grouped_scatter_by_round(
        csv_path,
        df,
        ycol="norm",
        ylabel="Norm",
        title=f"Client Norm by Round ({csv_path.stem})",
        outfile_suffix="_norm_by_round.png",
    )

    # 3) Cosine by round
    _plot_grouped_scatter_by_round(
        csv_path,
        df,
        ycol="cosine",
        ylabel="Cosine",
        title=f"Client Cosine by Round ({csv_path.stem})",
        outfile_suffix="_cosine_by_round.png",
    )

    # 4) Score histogram: suspicious vs benign
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    benign_scores = df.loc[df["suspicious"] == 0, "score"]
    suspicious_scores = df.loc[df["suspicious"] == 1, "score"]

    if len(benign_scores) > 0:
        ax.hist(benign_scores, bins=20, alpha=0.65, label="benign")
    if len(suspicious_scores) > 0:
        ax.hist(suspicious_scores, bins=20, alpha=0.65, label="suspicious")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution ({csv_path.stem})")
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
            label="Suspicious (detected)", color="red")

    if not malicious_by_round.empty:
        ax.plot(malicious_by_round["round"], malicious_by_round["is_malicious"],
                marker="s", linestyle="--", linewidth=2, markersize=6,
                label="Malicious (ground truth)", color="blue")

    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_title(f"Suspicious vs Malicious Clients by Round ({csv_path.stem})")
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
        labels.append("benign")
    if suspicious_scores:
        groups.append(suspicious_scores)
        labels.append("suspicious")
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_ylabel("Score")
        ax.set_title(f"Score Boxplot by Detection Label ({csv_path.stem})")
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
    ax.set_xlabel("Norm")
    ax.set_ylabel("Score")
    ax.set_title(f"Score vs Norm ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Suspicious (0=benign, 1=suspicious)")
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
    ax.set_xlabel("Cosine")
    ax.set_ylabel("Score")
    ax.set_title(f"Score vs Cosine ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Suspicious (0=benign, 1=suspicious)")
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
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Score")
        ax.set_title(f"Score by Ground Truth ({csv_path.stem})")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("is_malicious (0=benign, 1=malicious)")
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

    df_valid = df[df["is_malicious"] >= 0].copy()
    if df_valid.empty:
        print(f"[SKIP] {csv_path}: no valid ground truth for ROC")
        return

    y_true = df_valid["is_malicious"].astype(int).values
    y_score = df_valid["score"].values

    fpr, tpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'ROC Curve - {csv_path.stem}')
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
        ax.plot(wide["round"], wide[k], marker="o", label=k)

    ax.set_xlabel("Round")
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
        "defense-detect-tp",
        "defense-detect-fp",
        "defense-detect-fn",
        "defense-detect-tn",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        detection_keys,
        title=f"Detection Confusion Counts ({csv_path.stem})",
        ylabel="Count",
        outfile_suffix="_detection_confusion.png",
    )

    detection_rate_keys = [
        "defense-detect-precision",
        "defense-detect-recall",
        "defense-detect-fpr",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        detection_rate_keys,
        title=f"Detection Rates ({csv_path.stem})",
        ylabel="Rate",
        outfile_suffix="_detection_rates.png",
    )

    # 2) Detection summary metrics
    summary_keys = [
        "defense-detect-total-clients",
        "defense-detect-kept-clients",
        "defense-detect-filtered-clients",
        "defense-detect-suspicious-clients",
        "defense-detect-raw-suspicious-count",
        "defense-detect-skip-count",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        summary_keys,
        title=f"Detection Summary ({csv_path.stem})",
        ylabel="Count",
        outfile_suffix="_detection_summary.png",
    )

    # 3) Threshold / score related metrics
    score_keys = [
        "defense-detect-mean-score",
        "defense-detect-max-score",
        "defense-detect-min-score",
        "defense-detect-mean-cosine",
        "defense-detect-min-cosine",
        "defense-detect-mean-update-norm",
        "defense-detect-max-update-norm",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        score_keys,
        title=f"Detection Score Statistics ({csv_path.stem})",
        ylabel="Value",
        outfile_suffix="_detection_scores.png",
    )

    # 4) Aggregation metrics
    agg_keys = [
        "defense-agg-total-clients",
        "defense-agg-clipped-clients",
        "defense-agg-clipped-ratio",
        "defense-agg-avg-update-norm",
        "defense-agg-max-update-norm",
        "defense-agg-avg-post-norm",
        "defense-agg-selected-clients",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        agg_keys,
        title=f"Aggregation Metrics ({csv_path.stem})",
        ylabel="Value",
        outfile_suffix="_aggregation_metrics.png",
    )

    # 5) Client-defense metrics
    client_keys = [
        "client-defense-applied",
        "client-defense-filtered-samples",
        "client-defense-kept-samples",
        "client-defense-filter-ratio",
        "client-defense-mean-score",
        "client-defense-max-score",
        "client-defense-label-aware",
        "client-defense-label-blend-alpha",
    ]
    _plot_metric_lines(
        csv_path,
        wide,
        client_keys,
        title=f"Client Defense Metrics ({csv_path.stem})",
        ylabel="Value",
        outfile_suffix="_client_defense_metrics.png",
    )

    # 6) Compact all-in-one detection snapshot
    wanted_keys = [
        "defense-detect-tp",
        "defense-detect-fp",
        "defense-detect-fn",
        "defense-detect-precision",
        "defense-detect-recall",
        "defense-detect-fpr",
        "defense-detect-suspicious-clients",
        "defense-detect-kept-clients",
        "defense-detect-filtered-clients",
    ]
    present = [k for k in wanted_keys if k in wide.columns]
    if present:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        for k in present:
            ax.plot(wide["round"], wide[k], marker="o", label=k)
        ax.set_xlabel("Round")
        ax.set_title(f"Detection Overview ({csv_path.stem})")
        ax.grid(True, alpha=0.3)
        _force_integer_axis(ax)
        _legend_outside(ax, ncol=2)
        _savefig(csv_path.with_name(csv_path.stem + "_detection_overview.png"))

    # 7) Aggregation filtering metrics (removal rates, removed counts, kept counts)
    agg_removal_rate_keys = [
        "defense-agg-malicious-removal-rate",
        "defense-agg-benign-removal-rate",
    ]
    if any(k in wide.columns for k in agg_removal_rate_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_removal_rate_keys,
            title=f"Aggregation Removal Rates ({csv_path.stem})",
            ylabel="Rate",
            outfile_suffix="_aggregation_removal_rates.png",
        )

    agg_removed_count_keys = [
        "defense-agg-removed-malicious",
        "defense-agg-removed-benign",
        "defense-agg-removed-clients",
    ]
    if any(k in wide.columns for k in agg_removed_count_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_removed_count_keys,
            title=f"Aggregation Removed Clients ({csv_path.stem})",
            ylabel="Count",
            outfile_suffix="_aggregation_removed_counts.png",
        )

    agg_kept_count_keys = [
        "defense-agg-kept-malicious",
        "defense-agg-kept-benign",
        "defense-agg-kept-clients",
    ]
    if any(k in wide.columns for k in agg_kept_count_keys):
        _plot_metric_lines(
            csv_path,
            wide,
            agg_kept_count_keys,
            title=f"Aggregation Kept Clients ({csv_path.stem})",
            ylabel="Count",
            outfile_suffix="_aggregation_kept_counts.png",
        )

    print(f"[OK] metrics log plotted: {csv_path}")


# -----------------------------
# Multi-experiment comparison plots
# -----------------------------
def plot_multi_experiment_curves(results_dir: Path, metric: str = "accuracy") -> None:
    """
    Scan results_dir for experiment subdirectories, find main CSV in each,
    and plot the specified metric (accuracy or asr) across all experiments.
    """
    exp_data = []  # list of (exp_name, df)
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        # Find main CSV (does not contain '_metrics' or '_clients')
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

    ax.set_xlabel("Round")
    ylabel = "Accuracy" if metric == "accuracy" else "Attack Success Rate (ASR)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Comparison Across Experiments")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _legend_outside(ax, ncol=1)
    out_path = results_dir / f"comparison_{metric}.png"
    _savefig(out_path)
    print(f"Saved {metric} comparison plot to {out_path}")

def plot_multi_experiment_aggregation_metric(results_dir: Path, metric_key: str, ylabel: str, filename: str):
    """
    Scan experiment subdirectories, extract a specific aggregation metric from metrics CSV,
    and plot the curves across experiments.
    """
    exp_data = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        # Find metrics CSV (ends with _metrics.csv)
        metrics_csv = None
        for f in exp_dir.glob("*_metrics.csv"):
            metrics_csv = f
            break
        if metrics_csv is None:
            continue
        df = pd.read_csv(metrics_csv)
        if df.empty:
            continue
        # Filter rows for component 'aggregation' and key == metric_key
        rows = df[(df["component"] == "aggregation") & (df["key"] == metric_key)]
        if rows.empty:
            continue
        # Pivot to get round vs value
        df_metric = rows[["round", "value"]].copy()
        df_metric["value"] = pd.to_numeric(df_metric["value"], errors="coerce")
        df_metric = df_metric.dropna()
        if df_metric.empty:
            continue
        exp_data.append((exp_dir.name, df_metric))

    if not exp_data:
        print(f"No experiments found with metric {metric_key}.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for exp_name, df_metric in exp_data:
        ax.plot(df_metric["round"], df_metric["value"], marker='o', label=exp_name)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Comparison Across Experiments")
    ax.grid(True, alpha=0.3)
    _force_integer_axis(ax)
    _legend_outside(ax, ncol=1)
    out_path = results_dir / filename
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
    ax.set_xlabel("ASR (final round)")
    ax.set_ylabel("Accuracy (final round)")
    ax.set_title("Defense Effectiveness: ACC vs ASR")
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
                    ax.bar(x + i*width, det_df[m], width, label=m)
                ax.set_xticks(x + width*(len(present)-1)/2)
                ax.set_xticklabels(det_df["experiment"], rotation=45, ha="right")
                ax.set_ylabel("Score")
                ax.set_title("Detection Performance Comparison")
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                out_path = summary_csv.parent / "summary_detection_metrics.png"
                _savefig(out_path)
                print(f"Saved detection metrics bar chart to {out_path}")

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
            # If argument is a directory, collect all CSV files inside it (recursively)
            paths.extend(sorted(p.rglob("*.csv")))
        else:
            paths.append(p)

    if not paths:
        print("[SKIP] no CSV files found")
        return 0

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

    # Multi-experiment comparison: if the first argument is a directory named "results"
    # and it contains multiple subdirectories with main CSV files.
    first_arg = Path(argv[1])
    if first_arg.is_dir() and first_arg.name == "results":
        # Check if there are at least two experiment subdirectories
        subdirs = [d for d in first_arg.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            plot_multi_experiment_curves(first_arg, metric="accuracy")
            plot_multi_experiment_curves(first_arg, metric="asr")
            summary_csv = first_arg / "summary.csv"
            if summary_csv.exists():
                plot_multi_experiment_summary(summary_csv)
        else:
            print("[INFO] Not enough experiment subdirectories for multi-experiment plots (need at least 2).")

    # Multi-experiment comparison for aggregation metrics
    if first_arg.is_dir() and first_arg.name == "results":
        subdirs = [d for d in first_arg.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            # ... existing calls ...
            plot_multi_experiment_aggregation_metric(
                first_arg,
                "defense-agg-malicious-removal-rate",
                "Malicious Removal Rate",
                "comparison_malicious_removal_rate.png"
            )
            plot_multi_experiment_aggregation_metric(
                first_arg,
                "defense-agg-benign-removal-rate",
                "Benign Removal Rate",
                "comparison_benign_removal_rate.png"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))