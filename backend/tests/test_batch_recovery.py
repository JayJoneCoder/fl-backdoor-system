"""Historical batch snapshots must not masquerade as active tasks or executions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import main
from backend.batch_manager import BatchManager, RESTART_ERROR, UNRECORDED_ERROR


class BatchRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_record(self, **updates):
        data = {"batch_id": "old", "status": "running", "current_index": 0,
                "experiments": [{"name": "one"}, {"name": "two"}]}
        data.update(updates)
        path = self.directory / f"{data['batch_id']}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    async def test_current_endpoint_is_idle_with_historical_tasks_still_queryable(self):
        self.write_record(status="failed", error=RESTART_ERROR)
        manager = BatchManager(self.directory)
        with patch.object(main, "batch_manager", manager):
            self.assertEqual(await main.get_batch_status(), {"status": "idle"})
            historical = await main.get_batch_task_status("old")
        self.assertEqual(historical["status"], "failed")
        self.assertEqual(historical["batch_id"], "old")

    async def test_current_endpoint_still_returns_active_batch_not_newer_history(self):
        self.write_record(batch_id="zzzz", status="failed")
        manager = BatchManager(self.directory)
        active = manager.create_task([{"name": "active"}])
        for state in ("pending", "running", "stopping"):
            active.status = state
            with patch.object(main, "batch_manager", manager):
                self.assertEqual((await main.get_batch_status())["batch_id"], active.batch_id)

    async def test_legacy_missing_details_are_unknown_not_cancelled(self):
        for state in ("running", "completed", "failed"):
            with self.subTest(state=state):
                path = self.write_record(status=state)
                original = path.read_bytes()
                task = BatchManager(self.directory).get_task("old")
                self.assertEqual([item["status"] for item in task.results], ["unknown", "unknown"])
                self.assertTrue(all(item["error"] == UNRECORDED_ERROR for item in task.results))
                data = task.to_dict()
                self.assertEqual(data["completed_count"], 0)
                self.assertEqual(data["cancelled_count"], 0)
                self.assertEqual(data["unknown_count"], 2)
                self.assertEqual(path.read_bytes(), original, "Reading must not rewrite historical files")
                self.assertIsNone(BatchManager(self.directory).get_active_task())

    async def test_previous_loader_synthetic_cancellations_are_uncertain(self):
        path = self.write_record(status="failed", error=RESTART_ERROR, results=[
            {"name": name, "status": "cancelled", "error": "Not started before backend restart.",
             "start_time": None, "end_time": None, "exit_code": None}
            for name in ("one", "two")
        ])
        original = path.read_bytes()
        task = BatchManager(self.directory).get_task("old")
        self.assertEqual(task.to_dict()["unknown_count"], 2)
        self.assertEqual(task.to_dict()["cancelled_count"], 0)
        self.assertEqual(path.read_bytes(), original)

    async def test_versioned_real_cancellation_remains_cancelled(self):
        self.write_record(record_version=2, status="failed", error=RESTART_ERROR, results=[
            {"name": "one", "status": "cancelled", "error": "Not started before backend restart."},
            {"name": "two", "status": "completed", "exit_code": 0},
        ])
        task = BatchManager(self.directory).get_task("old")
        self.assertEqual(task.to_dict()["cancelled_count"], 1)
        self.assertEqual(task.to_dict()["success_count"], 1)
        self.assertEqual(task.to_dict()["unknown_count"], 0)

    async def test_modern_interrupted_record_keeps_real_execution_details(self):
        manager = BatchManager(self.directory)
        task = manager.create_task([{"name": "done"}, {"name": "running"}, {"name": "waiting"}])
        task.status = "running"
        task.results[0].update(status="completed", exit_code=0)
        task.results[1].update(status="running", start_time="2026-10-14T12:00:00")
        manager.update_task(task)
        restored = BatchManager(self.directory).get_task(task.batch_id)
        self.assertEqual(restored.status, "failed")
        self.assertEqual([item["status"] for item in restored.results], ["completed", "failed", "cancelled"])
        self.assertEqual(restored.to_dict()["unknown_count"], 0)
        self.assertIsNotNone(restored.results[1]["end_time"])

    async def test_status_revision_increases_on_updates_and_after_restart(self):
        manager = BatchManager(self.directory)
        task = manager.create_task([{"name": "one"}])
        self.assertEqual(task.to_dict()["revision"], 0)
        manager.update_task(task)
        first = task.to_dict()["revision"]
        task.pause_requested = True
        manager.update_task(task)
        self.assertGreater(task.to_dict()["revision"], first)
        saved_revision = json.loads((self.directory / f"{task.batch_id}.json").read_text(encoding="utf-8"))["revision"]
        restored = BatchManager(self.directory).get_task(task.batch_id)
        self.assertGreater(restored.to_dict()["revision"], saved_revision)

    async def test_missing_or_mismatched_detail_is_not_assigned_to_another_experiment(self):
        for saved in (None, [], [{"name": "another", "status": "completed"}]):
            self.write_record(status="completed", results=saved)
            task = BatchManager(self.directory).get_task("old")
            self.assertEqual([item["status"] for item in task.results], ["unknown", "unknown"])


if __name__ == "__main__":
    unittest.main()