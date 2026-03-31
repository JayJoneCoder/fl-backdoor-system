"""
Plot training curves from CSV logs.

Usage:
    python plot.py results/*.csv
    python plot.py results/baseline.csv results/badnets.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_single(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    required = {"round", "accuracy", "asr", "loss"}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] {csv_path}: missing columns {sorted(missing)}")
        return

    rounds = df["round"]
    acc = df["accuracy"]
    asr = df["asr"]
    loss = df["loss"]

    out_acc = csv_path.with_name(csv_path.stem + "_acc.png")
    out_asr = csv_path.with_name(csv_path.stem + "_asr.png")
    out_acc_vs_asr = csv_path.with_name(csv_path.stem + "_acc_vs_asr.png")
    out_loss = csv_path.with_name(csv_path.stem + "_loss.png")

    # ===== Figure 1: Accuracy =====
    plt.figure()
    plt.plot(rounds, acc)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({csv_path.stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_acc, dpi=300)
    plt.close()

    # ===== Figure 2: ASR =====
    plt.figure()
    plt.plot(rounds, asr)
    plt.xlabel("Round")
    plt.ylabel("ASR")
    plt.title(f"ASR Curve ({csv_path.stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_asr, dpi=300)
    plt.close()

    # ===== Figure 3: ACC vs ASR =====
    plt.figure()
    plt.plot(asr, acc)
    plt.xlabel("ASR")
    plt.ylabel("Accuracy")
    plt.title(f"ACC vs ASR ({csv_path.stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_acc_vs_asr, dpi=300)
    plt.close()

    # ===== Figure 4: Loss =====
    plt.figure()
    plt.plot(rounds, loss)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({csv_path.stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_loss, dpi=300)
    plt.close()

    print(f"[OK] Plotted: {csv_path}")


def plot_compare(csv_paths: list[Path]) -> None:
    out_dir = csv_paths[0].parent

    # ===== Accuracy Compare =====
    plt.figure()
    for path in csv_paths:
        df = pd.read_csv(path)
        if {"round", "accuracy"} <= set(df.columns):
            plt.plot(df["round"], df["accuracy"], label=path.stem)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_accuracy.png", dpi=300)
    plt.close()

    # ===== ASR Compare =====
    plt.figure()
    for path in csv_paths:
        df = pd.read_csv(path)
        if {"round", "asr"} <= set(df.columns):
            plt.plot(df["round"], df["asr"], label=path.stem)
    plt.xlabel("Round")
    plt.ylabel("ASR")
    plt.title("ASR Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_asr.png", dpi=300)
    plt.close()

    print(f"[OK] Comparison plots saved in: {out_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot.py results/*.csv")
        return

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        paths.extend(Path().glob(arg))

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