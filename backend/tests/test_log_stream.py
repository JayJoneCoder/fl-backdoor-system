"""Regression tests for log truncation, independent cursors, and CSV replay."""

import os
import tempfile
import unittest
from pathlib import Path

from backend.log_stream import LogFileReader, MetricFileReader, read_total_rounds


class LogFileReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run.log"
        self.reader = LogFileReader(self.path)

    def test_waits_for_file_and_replays_history_once(self):
        self.assertEqual(self.reader.read_lines(), [])
        self.path.write_bytes(b"starting\r\n[ROUND 1/10]\n")
        self.assertEqual(self.reader.read_lines(), ["starting", "[ROUND 1/10]"])
        self.assertEqual(self.reader.read_lines(), [])
        with self.path.open("ab") as f:
            f.write(b"finished\n")
        self.assertEqual(self.reader.read_lines(), ["finished"])

    def test_buffers_incomplete_utf8_line(self):
        text = "训练完成".encode("utf-8")
        self.path.write_bytes(b"first\n" + text[:4])
        self.assertEqual(self.reader.read_lines(), ["first"])
        with self.path.open("ab") as f:
            f.write(text[4:] + b"\r\n")
        self.assertEqual(self.reader.read_lines(), ["训练完成"])

    def test_truncation_resets_cursor_and_discards_old_partial_line(self):
        self.path.write_bytes(b"old complete line\nold partial")
        self.assertEqual(self.reader.read_lines(), ["old complete line"])
        self.path.write_bytes(b"new\n")
        self.assertEqual(self.reader.read_lines(), ["new"])

    def test_generation_resets_even_when_new_file_is_larger(self):
        self.path.write_bytes(b"old log\n")
        self.assertEqual(self.reader.read_lines(1), ["old log"])
        self.path.write_bytes(b"Starting local SuperLink\n[ROUND 1/10]\n")
        self.assertEqual(
            self.reader.read_lines(2), ["Starting local SuperLink", "[ROUND 1/10]"]
        )
        self.assertEqual(self.reader.read_lines(2), [])

    def test_file_replacement_resets_cursor(self):
        self.path.write_bytes(b"old\n")
        self.assertEqual(self.reader.read_lines(), ["old"])
        replacement = self.path.with_suffix(".new")
        replacement.write_bytes(b"replacement line\n")
        os.replace(replacement, self.path)
        self.assertEqual(self.reader.read_lines(), ["replacement line"])

    def test_clients_have_independent_cursors(self):
        self.path.write_bytes(b"history\n")
        other = LogFileReader(self.path)
        self.assertEqual(self.reader.read_lines(), ["history"])
        self.assertEqual(other.read_lines(), ["history"])
        with self.path.open("ab") as f:
            f.write(b"live\n")
        self.assertEqual(other.read_lines(), ["live"])
        self.assertEqual(self.reader.read_lines(), ["live"])


class MetricFileReaderTests(unittest.TestCase):
    HEADER = b"round,accuracy,asr,loss\r\n"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "experiment.csv"
        self.reader = MetricFileReader(self.path)

    def test_missing_file_then_full_history_and_incremental_updates(self):
        self.assertEqual(self.reader.read_rows(), [])
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\r\n1,0.7,0.6,0.4\r\n")
        rows = self.reader.read_rows()
        self.assertEqual([row["round"] for row in rows], [0, 1])
        self.assertEqual(rows[1], {"round": 1, "accuracy": 0.7, "asr": 0.6, "loss": 0.4})
        self.assertEqual(self.reader.read_rows(), [])
        with self.path.open("ab") as f:
            f.write(b"2,0.8,0.5,0.3\r\n3,0.9,0.4,0.2\r\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [2, 3])

    def test_partial_csv_row_is_not_sent_until_complete(self):
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\n1,0.7,0.6,")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [0])
        with self.path.open("ab") as f:
            f.write(b"0.4\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [1])

    def test_malformed_and_nonfinite_values_are_ignored(self):
        self.path.write_bytes(
            self.HEADER
            + b"0,0.1,0.2,2.5\n1,nan,0.2,1.0\n2,0.8,inf,1.0\n"
            + b"3,missing,0.1,1.0\n-1,0.8,0.1,1.0\n4.5,0.8,0.1,1.0\n"
            + b"5,0.8,0.1,1.0\n"
        )
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [0, 5])

    def test_previous_run_csv_is_ignored_until_new_run_writes(self):
        self.path.write_bytes(self.HEADER + b"10,0.9,0.1,0.1\n")
        old_mtime = self.path.stat().st_mtime_ns
        start = old_mtime + 1_000_000_000
        self.assertEqual(self.reader.read_rows(2, start), [])
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\n")
        os.utime(self.path, ns=(start + 1_000_000_000, start + 1_000_000_000))
        self.assertEqual([row["round"] for row in self.reader.read_rows(2, start)], [0])

    def test_new_run_resets_identical_sized_csv_and_rounds(self):
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\n1,0.7,0.6,0.4\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows(1)], [0, 1])
        self.assertEqual([row["round"] for row in self.reader.read_rows(2)], [0, 1])
        self.assertEqual(self.reader.read_rows(2), [])

    def test_truncation_resets_rounds(self):
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\n10,0.9,0.1,0.1\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [0, 10])
        self.path.write_bytes(self.HEADER + b"0,0.1,0.2,2.5\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [0])

    def test_reads_bom_csv(self):
        self.path.write_bytes(b"\xef\xbb\xbf" + self.HEADER + b"0,0.1,0.2,2.5\n")
        self.assertEqual([row["round"] for row in self.reader.read_rows()], [0])


class TotalRoundTests(unittest.TestCase):
    def test_total_comes_from_snapshot_and_invalid_values_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config_snapshot.toml"
            self.assertIsNone(read_total_rounds(path))
            for value, expected in [("10", 10), ("0", None), ("-1", None), ('"invalid"', None)]:
                with self.subTest(value=value):
                    path.write_text(
                        f"[tool.flwr.app.config]\nnum-server-rounds = {value}\n", encoding="utf-8"
                    )
                    self.assertEqual(read_total_rounds(path), expected)
            path.write_text("not valid TOML [", encoding="utf-8")
            self.assertIsNone(read_total_rounds(path))


if __name__ == "__main__":
    unittest.main()