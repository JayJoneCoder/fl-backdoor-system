"""Tests for extracting the root cause from a failed experiment worker log."""

import tempfile
import unittest
from pathlib import Path

from backend.experiment_runner import _extract_failure_reason, _worker_failure_message


class ExtractFailureReasonTest(unittest.TestCase):
    def _write_log(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_prefers_the_server_fatal_error(self):
        log = self._write_log(
            "ValueError: inner error\n"
            "!!! FATAL ERROR in server main: Krum requires n > 2*f + 2 after detection, got n=7, f=3.\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Exception in ServerApp thread\n"
        )
        self.assertEqual(
            _extract_failure_reason(log),
            "Krum requires n > 2*f + 2 after detection, got n=7, f=3.",
        )

    def test_falls_back_to_the_last_error_line(self):
        log = self._write_log(
            "Traceback (most recent call last):\n"
            "ValueError: first failure\n"
            "RuntimeError: second failure\n"
        )
        self.assertEqual(_extract_failure_reason(log), "second failure")

    def test_strips_ansi_sequences(self):
        log = self._write_log("\x1b[91mValueError\x1b[0m: invalid configuration\n")
        self.assertEqual(_extract_failure_reason(log), "invalid configuration")

    def test_missing_log_returns_none(self):
        self.assertIsNone(_extract_failure_reason(Path("does-not-exist.log")))
        self.assertIsNone(_extract_failure_reason(None))

    def test_log_without_recognizable_error_returns_none(self):
        log = self._write_log("INFO: everything looked fine\nERROR : Exit Code: 700\n")
        self.assertIsNone(_extract_failure_reason(log))


class WorkerFailureMessageTest(unittest.TestCase):
    def _write_log(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_reason_is_embedded_in_the_worker_message(self):
        log = self._write_log("ValueError: Krum requires n > 2*f + 2 after detection, got n=7, f=3.\n")
        self.assertEqual(
            _worker_failure_message(1, log),
            "Experiment worker exited with code 1: "
            "Krum requires n > 2*f + 2 after detection, got n=7, f=3. See run.log for details.",
        )

    def test_missing_punctuation_is_added_once(self):
        log = self._write_log("ValueError: invalid configuration\n")
        self.assertEqual(
            _worker_failure_message(2, log),
            "Experiment worker exited with code 2: invalid configuration. See run.log for details.",
        )

    def test_unknown_reason_keeps_the_generic_message(self):
        log = self._write_log("INFO: nothing wrong\n")
        self.assertEqual(
            _worker_failure_message(3, log),
            "Experiment worker exited with code 3. See run.log for details.",
        )


if __name__ == "__main__":
    unittest.main()
