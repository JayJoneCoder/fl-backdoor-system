"""Simple CSV logger for experiment metrics."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class CSVLogger:
    """Log round / accuracy / ASR / loss into a CSV file.

    Also writes a separate diagnostics CSV for arbitrary per-round metrics.
    """

    def __init__(self, save_dir: str = "results", filename: str | None = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}.csv"

        self.filepath = self.save_dir / filename
        self.metrics_filepath = self.save_dir / f"{self.filepath.stem}_metrics.csv"

        with self.filepath.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "asr", "loss"])

        with self.metrics_filepath.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "component", "key", "value"])

        print(f">>> [LOGGER] Saving results to: {self.filepath}")
        print(f">>> [LOGGER] Saving metrics to: {self.metrics_filepath}")

    @staticmethod
    def _to_csv_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False, default=str)

    def log(self, round: int, accuracy: float, asr: float, loss: float) -> None:
        with self.filepath.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([round, accuracy, asr, loss])

    def log_metrics(
        self,
        round: int,
        component: str,
        metrics: Mapping[str, Any] | None,
    ) -> None:
        if not metrics:
            return

        with self.metrics_filepath.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, value in metrics.items():
                writer.writerow(
                    [
                        round,
                        component,
                        key,
                        self._to_csv_value(value),
                    ]
                )