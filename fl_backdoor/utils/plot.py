"""
Plot experiment curves and detection diagnostics from CSV logs.

Supported CSV types:
1) Main experiment log:
   columns = round, accuracy, asr, loss

2) Detection / aggregation metrics log:
   columns = round, component, key, value

3) Per-client diagnostics log:
   columns = round, client_id, score, norm, norm_z, cosine, suspicious, kept

Usage:
    python plot.py results/*.csv
    python plot.py results/baseline.csv results/badnets.csv
    python plot.py results/*_clients.csv
    python plot.py results/*_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.cm import ScalarMappable
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


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


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
) -> None:
    unique_rounds = sorted(df["round"].dropna().unique().tolist())
    out_path = csv_path.with_name(csv_path.stem + outfile_suffix)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.grid(True, alpha=0.3)

    if len(unique_rounds) <= legend_round_threshold:
        # Few rounds: show explicit legend outside
        for r, g in df.groupby("round", sort=True):
            ax.scatter(
                g["client_id"],
                g[ycol],
                s=28,
                alpha=0.85,
                label=f"round {int(r)}" if float(r).is_integer() else f"round {r}",
            )
        _legend_outside(ax, title="Round")
    else:
        # Many rounds: use colorbar instead of huge legend
        rounds = df["round"].to_numpy(dtype=float)
        norm = Normalize(vmin=float(pd.Series(rounds).min()), vmax=float(pd.Series(rounds).max()))
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
        ["round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"],
    ).dropna(subset=["round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"])
    if df.empty:
        print(f"[SKIP] {csv_path}: no valid rows after cleaning")
        return

    df["round"] = df["round"].astype(int)
    df["client_id"] = df["client_id"].astype(int)
    df["suspicious"] = df["suspicious"].astype(int)
    df["kept"] = df["kept"].astype(int)

    # Optional: sort for cleaner plots
    df = df.sort_values(["round", "client_id"])

    out_score_round = csv_path.with_name(csv_path.stem + "_score_by_round.png")
    out_norm_round = csv_path.with_name(csv_path.stem + "_norm_by_round.png")
    out_cosine_round = csv_path.with_name(csv_path.stem + "_cosine_by_round.png")
    out_score_hist = csv_path.with_name(csv_path.stem + "_score_hist.png")
    out_suspicious_count = csv_path.with_name(csv_path.stem + "_suspicious_count.png")
    out_score_box = csv_path.with_name(csv_path.stem + "_score_box.png")
    out_score_norm_scatter = csv_path.with_name(csv_path.stem + "_score_vs_norm.png")
    out_score_cosine_scatter = csv_path.with_name(csv_path.stem + "_score_vs_cosine.png")

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

    # 4) Score histogram: benign vs suspicious
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

    # 5) Suspicious count by round
    suspicious_by_round = df.groupby("round", as_index=False)["suspicious"].sum()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(suspicious_by_round["round"], suspicious_by_round["suspicious"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Suspicious Clients")
    ax.set_title(f"Suspicious Count by Round ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _savefig(out_suspicious_count)

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
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_ylabel("Score")
        ax.set_title(f"Score Boxplot by Label ({csv_path.stem})")
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

    print(f"[OK] client log plotted: {csv_path}")


# -----------------------------
# Metrics log plots
# -----------------------------
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

    out_path = csv_path.with_name(csv_path.stem + "_metrics_summary.png")

    # We keep a compact but informative summary plot.
    wanted_keys = [
        "defense-detect-mean-score",
        "defense-detect-max-score",
        "defense-detect-mean-cosine",
        "defense-detect-min-cosine",
        "defense-detect-mean-update-norm",
        "defense-detect-max-update-norm",
        "defense-agg-total-clients",
        "defense-agg-clipped-clients",
        "defense-agg-selected-clients",
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    plotted_any = False

    for key in wanted_keys:
        sub = df[df["key"] == key].copy()
        if sub.empty:
            continue
        sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["value_num"]).sort_values("round")
        if sub.empty:
            continue
        ax.plot(sub["round"], sub["value_num"], marker="o", label=key)
        plotted_any = True

    if not plotted_any:
        print(f"[SKIP] {csv_path}: no known keys found for summary plot")
        return

    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.set_title(f"Metrics Summary ({csv_path.stem})")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend_outside(ax)
    _savefig(out_path)

    print(f"[OK] metrics log plotted: {csv_path}")


# -----------------------------
# Dispatch
# -----------------------------
def plot_single(csv_path: Path) -> None:
    df = _safe_read_csv(csv_path)
    if df is None:
        return

    columns = set(df.columns)

    if {"round", "accuracy", "asr", "loss"}.issubset(columns):
        plot_main_log(csv_path, df)
        return

    if {"round", "client_id", "score", "norm", "norm_z", "cosine", "suspicious", "kept"}.issubset(columns):
        plot_client_log(csv_path, df)
        return

    if {"round", "component", "key", "value"}.issubset(columns):
        plot_metrics_log(csv_path, df)
        return

    print(f"[SKIP] {csv_path}: unrecognized CSV schema {sorted(columns)}")


def plot_compare(csv_paths: list[Path]) -> None:
    out_dir = csv_paths[0].parent

    main_dfs: list[tuple[Path, pd.DataFrame]] = []
    for path in csv_paths:
        df = _safe_read_csv(path)
        if df is None:
            continue
        if {"round", "accuracy", "asr", "loss"}.issubset(df.columns):
            main_dfs.append((path, df))

    if not main_dfs:
        return

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for path, df in main_dfs:
        df = df.copy()
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df = df.dropna(subset=["round", "accuracy"]).sort_values("round")
        if df.empty:
            continue
        ax.plot(df["round"], df["accuracy"], marker="o", label=path.stem)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend_outside(ax)
    _savefig(out_dir / "compare_accuracy.png")

    # ASR comparison
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for path, df in main_dfs:
        df = df.copy()
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
        df = df.dropna(subset=["round", "asr"]).sort_values("round")
        if df.empty:
            continue
        ax.plot(df["round"], df["asr"], marker="o", label=path.stem)
    ax.set_xlabel("Round")
    ax.set_ylabel("ASR")
    ax.set_title("ASR Comparison")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend_outside(ax)
    _savefig(out_dir / "compare_asr.png")

    # Loss comparison
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for path, df in main_dfs:
        df = df.copy()
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna(subset=["round", "loss"]).sort_values("round")
        if df.empty:
            continue
        ax.plot(df["round"], df["loss"], marker="o", label=path.stem)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Comparison")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend_outside(ax)
    _savefig(out_dir / "compare_loss.png")

    print(f"[OK] comparison plots saved in: {out_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot.py results/*.csv")
        return

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        matched = list(Path().glob(arg))
        if matched:
            paths.extend(matched)
        else:
            p = Path(arg)
            if p.exists():
                paths.append(p)

    paths = sorted(set(paths))
    if not paths:
        print("No CSV files found.")
        return

    for path in paths:
        plot_single(path)

    if len(paths) > 1:
        plot_compare(paths)


if __name__ == "__main__":
    main()