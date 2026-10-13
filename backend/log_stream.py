"""Independent file cursors for experiment logs and complete CSV metric rows."""

import csv
import io
import math
import os
from pathlib import Path
from typing import Any

import tomli


def read_total_rounds(snapshot_path: Path) -> int | None:
    try:
        with snapshot_path.open("rb") as f:
            config = tomli.load(f)["tool"]["flwr"]["app"]["config"]
        total = int(config["num-server-rounds"])
        return total if total > 0 else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


class LogFileReader:
    """Replay a log, then follow it without keeping a stale file handle open."""

    def __init__(self, path: Path):
        self.path = path
        self._identity = None
        self._generation = None
        self._offset = 0
        self._pending = b""

    def read_lines(self, generation: Any = None) -> list[str]:
        try:
            with self.path.open("rb") as f:
                stat = os.fstat(f.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if (
                    identity != self._identity
                    or generation != self._generation
                    or stat.st_size < self._offset
                ):
                    self._offset = 0
                    self._pending = b""
                self._identity = identity
                self._generation = generation
                f.seek(self._offset)
                chunk = f.read(64 * 1024)
                self._offset = f.tell()
        except FileNotFoundError:
            self._identity = None
            self._offset = 0
            self._pending = b""
            return []

        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()
        return [line.rstrip(b"\r").decode("utf-8", errors="replace") for line in lines]


class MetricFileReader:
    """Read only completed, finite metric rows; reset on a new experiment run."""

    def __init__(self, path: Path):
        self.path = path
        self._signature = None
        self._generation = None
        self._last_round = -1

    def read_rows(
        self, generation: Any = None, minimum_mtime_ns: int = 0
    ) -> list[dict[str, int | float]]:
        if generation != self._generation:
            self._generation = generation
            self._signature = None
            self._last_round = -1

        try:
            with self.path.open("rb") as f:
                stat = os.fstat(f.fileno())
                if stat.st_mtime_ns < minimum_mtime_ns:
                    return []  # The previous run's CSV may still exist during startup.
                signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if signature == self._signature:
                    return []
                content = f.read()
        except FileNotFoundError:
            self._signature = None
            self._last_round = -1
            return []

        if self._signature and (
            signature[:2] != self._signature[:2]
            or signature[2] < self._signature[2]
        ):
            self._last_round = -1
        self._signature = signature

        # Ignore a partially written last row, including a split UTF-8 character.
        complete = content[: content.rfind(b"\n") + 1].decode("utf-8-sig", errors="replace")
        rows = []
        for row in csv.DictReader(io.StringIO(complete)):
            try:
                values = {key: float(row[key]) for key in ("round", "accuracy", "asr", "loss")}
                if not all(math.isfinite(value) for value in values.values()):
                    continue
                if values["round"] < 0 or not values["round"].is_integer():
                    continue
                values["round"] = int(values["round"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(values)

        if rows and rows[-1]["round"] < self._last_round:
            self._last_round = -1
        updates = []
        for row in rows:
            if row["round"] > self._last_round:
                updates.append(row)
                self._last_round = row["round"]
        return updates