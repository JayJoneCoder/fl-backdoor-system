"""Simple CSV logger for experiment metrics."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class CSVLogger:
    """Log round / accuracy / ASR / loss into a CSV file."""

    def __init__(self, save_dir: str = "results", filename: str | None = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}.csv"

        self.filepath = self.save_dir / filename

        with self.filepath.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "asr", "loss"])

        print(f">>> [LOGGER] Saving results to: {self.filepath}")

    def log(self, round: int, accuracy: float, asr: float, loss: float) -> None:
        with self.filepath.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([round, accuracy, asr, loss])