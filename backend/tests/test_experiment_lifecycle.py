"""Lifecycle tests use temporary configs and fake workers, never real training."""

import asyncio
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import tomli
import tomli_w
from fastapi import HTTPException

from backend import config_manager as cm, experiment_runner as er, main
from backend.batch_manager import BatchManager, BatchTask
from backend.experiment_worker import verified_exit_code


def configuration(**updates):
    values = {"run-name": "baseline", "dataset": "mnist", "attack": "none", "defense": "none",
              "detection": "none", "num-clients": 10, "num-server-rounds": 2, "batch-size": 32}
    values.update(updates)
    return tomli_w.dumps({"project": {"name": "test-project", "version": "1.0"},
                         "tool": {"flwr": {"app": {"config": values, "components": {
                             "serverapp": "fl_backdoor.server.server:app", "clientapp": "fl_backdoor.client.client:app"}}}}}).encode()


class FakeOwner:
    def __init__(self, exit_code=None):
        self.code = exit_code
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None
        self.calls = 0
        self.process = Mock()
        self.process.stdout = io.StringIO("[ROUND 1/2]\r\nlast log")
        self.process.poll.side_effect = lambda: self.code

    def spawn(self, *args, **kwargs):
        return self.process

    def observe(self):
        pass

    def cleanup(self):
        self.calls += 1
        self.started.set()
        if not self.release.wait(3):
            raise RuntimeError("Test cleanup gate timed out")
        if self.failure:
            raise self.failure
        if self.code is None:
            self.code = 1


class RunnerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "pyproject.toml"
        self.original = configuration()
        self.config.write_bytes(self.original)
        for obj, name, value in ((cm, "TOML_PATH", self.config), (cm, "BACKUP_PATH", self.root / "backup.bak"),
                                 (er, "RESULTS_DIR", self.root / "results")):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.runner = er.ExperimentRunner()

    async def start(self, owner, **kwargs):
        with patch.object(er, "OwnedProcessTree", return_value=owner):
            await self.runner.start("first", **kwargs)

    async def test_stop_keeps_busy_and_event_loop_responsive_until_cleanup_done(self):
        owner = FakeOwner()
        owner.release.clear()
        await self.start(owner)
        stop = asyncio.create_task(self.runner.stop())
        try:
            await asyncio.wait_for(asyncio.to_thread(owner.started.wait), 2)
            self.assertEqual((await self.runner.get_status())["status"], "stopping")
            self.assertTrue(self.runner.is_busy())
            self.assertFalse(stop.done())
            with self.assertRaises(RuntimeError):
                await self.runner.start("second")
            await asyncio.wait_for(asyncio.sleep(0.01), 0.1)
        finally:
            owner.release.set()
            result = await stop
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(self.runner.is_busy())
        self.assertEqual(self.config.read_bytes(), self.original)

    async def test_duplicate_stops_share_cleanup_and_do_not_reset_identity(self):
        owner = FakeOwner()
        await self.start(owner)
        results = await asyncio.gather(self.runner.stop(), self.runner.stop())
        self.assertEqual(owner.calls, 1)
        self.assertTrue(all(result["status"] == "cancelled" for result in results))
        self.assertEqual(self.runner.exp_name, "first")

    async def test_natural_success_is_not_finished_until_cleanup_done(self):
        owner = FakeOwner(0)
        owner.release.clear()
        await self.start(owner)
        try:
            await asyncio.wait_for(asyncio.to_thread(owner.started.wait), 2)
            self.assertEqual((await self.runner.get_status())["status"], "cleaning")
            self.assertTrue(self.runner.is_busy())
        finally:
            owner.release.set()
        result = await self.runner.wait()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["exit_code"], 0)
        self.assertIsNone(self.runner._temp_home)
        self.assertIn("last log", self.runner.log_file.read_text())

    async def test_worker_failure_is_never_success(self):
        await self.start(FakeOwner(23))
        result = await self.runner.wait()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 23)
        self.assertIn("23", result["error"])

    async def test_cleanup_failure_blocks_next_run_and_can_be_retried(self):
        owner = FakeOwner(0)
        owner.failure = OSError("cannot clean yet")
        await self.start(owner)
        self.assertEqual((await self.runner.wait())["status"], "cleanup_failed")
        self.assertTrue(self.runner.is_busy())
        with self.assertRaises(RuntimeError):
            await self.runner.start("second")
        owner.failure = None
        self.assertEqual((await self.runner.stop())["status"], "finished")
        self.assertFalse(self.runner.is_busy())

    async def test_startup_failure_cleans_and_restores_only_owned_config(self):
        owner = FakeOwner()
        owner.spawn = Mock(side_effect=OSError("spawn failed"))
        with self.assertRaises(OSError):
            await self.start(owner)
        self.assertEqual((await self.runner.get_status())["status"], "failed")
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertFalse(self.runner.is_busy())
        self.assertEqual(owner.calls, 1)

    async def test_cleanup_does_not_overwrite_config_saved_during_running(self):
        await self.start(FakeOwner())
        cm.update_config({"batch-size": 77})
        saved = self.config.read_bytes()
        await self.runner.stop()
        self.assertEqual(self.config.read_bytes(), saved)

    async def test_batch_run_config_never_modifies_global_or_inherits_previous(self):
        baseline = configuration()
        owner = FakeOwner(0)
        await self.start(owner, config_overrides={"batch-size": 80}, base_config=baseline, backup_config=False)
        first = (self.runner.log_file.parent / "config_snapshot.toml").read_bytes()
        await self.runner.wait()
        with patch.object(er, "OwnedProcessTree", return_value=FakeOwner(0)):
            await self.runner.start("second", base_config=baseline)
        second = (self.runner.log_file.parent / "config_snapshot.toml").read_bytes()
        await self.runner.wait()
        section = lambda content: tomli.loads(content.decode())["tool"]["flwr"]["app"]["config"]
        self.assertEqual(section(first)["batch-size"], 80)
        self.assertEqual(section(second)["batch-size"], 32)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertFalse(cm.BACKUP_PATH.exists())

    async def test_new_runtime_is_private_and_has_explicit_non_global_connection(self):
        owner = FakeOwner()
        owner.spawn = Mock(wraps=owner.spawn)
        with patch.dict(os.environ, {"PYTHONPATH": str(self.root / "inherited-modules")}):
            await self.start(owner)
        home = self.runner._temp_home
        doc = tomli.loads((home / "flower" / "config.toml").read_text())
        conn = doc["superlink"]["project-runtime"]
        self.assertTrue(conn["address"].startswith("127.0.0.1:"))
        self.assertTrue(conn["insecure"])
        self.assertTrue((home / "app" / "fl_backdoor" / "task.py").exists())
        self.assertFalse((home / "app" / "fl_backdoor" / "tests").exists())
        env = owner.spawn.call_args.kwargs["env"]
        self.assertEqual(owner.spawn.call_args.kwargs["cwd"], str(home / "app"))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(home / "app"))
        self.assertIn(str(self.root / "inherited-modules"), env["PYTHONPATH"].split(os.pathsep))
        self.assertEqual(env["PYTHONHASHSEED"], "42")
        self.assertEqual(env["FL_BACKDOOR_RUNTIME_APP"], str(home / "app"))
        self.assertEqual(env["RAY_RUNTIME_ENV_HOOK"], "fl_backdoor.utils.ray_runtime.runtime_env_hook")
        self.assertEqual(env["FL_BACKDOOR_DATA_DIR"], str(er.PROJECT_ROOT / "data"))
        await self.runner.stop()
        self.assertFalse(home.exists())

    async def test_cancellation_during_preparation_waits_for_thread_then_restores(self):
        started, release = threading.Event(), threading.Event()
        prepare = self.runner._prepare_runtime

        def blocked(content):
            started.set()
            release.wait(3)
            return prepare(content)

        with patch.object(self.runner, "_prepare_runtime", side_effect=blocked), \
             patch.object(er, "OwnedProcessTree") as owner:
            task = asyncio.create_task(self.runner.start("cancelled_start"))
            try:
                await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                self.assertTrue(self.runner.is_busy())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            owner.assert_not_called()
        self.assertFalse(self.runner.is_busy())
        self.assertEqual(self.config.read_bytes(), self.original)


class FlowerOutcomeTests(unittest.TestCase):
    def test_cli_zero_failed_run_is_failure(self):
        run = SimpleNamespace(status=SimpleNamespace(status="finished", sub_status="failed", details="training error"))
        self.assertEqual(verified_exit_code(0, [run]), 1)

    def test_only_finished_completed_is_success(self):
        for status, sub_status in [("finished", "completed"), ("finished", "stopped"), ("running", ""), ("pending", "")]:
            run = SimpleNamespace(status=SimpleNamespace(status=status, sub_status=sub_status, details=""))
            self.assertEqual(verified_exit_code(0, [run]), 0 if sub_status == "completed" else 1)
        self.assertEqual(verified_exit_code(4, []), 4)

    def test_missing_or_ambiguous_outcome_cannot_report_success(self):
        for runs in ([], [None, None]):
            with self.assertRaises(RuntimeError):
                verified_exit_code(0, runs)


class BatchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manager = BatchManager(self.root / "batches")
        self.runner = Mock()
        self.runner.run_id = "test-run-id"
        self.runner._on_log_line = None
        self.runner.start = AsyncMock()
        self.runner.wait = AsyncMock(return_value={"status": "finished", "exit_code": 0})
        self.runner.stop = AsyncMock(return_value={"status": "cancelled"})
        self.runner.is_busy.return_value = False
        p = patch.multiple(main, runner=self.runner, batch_manager=self.manager, execution_lock=asyncio.Lock())
        p.start()
        self.addCleanup(p.stop)
        self.config = self.root / "pyproject.toml"
        self.config.write_bytes(configuration())
        p = patch.object(cm, "TOML_PATH", self.config)
        p.start()
        self.addCleanup(p.stop)

    def task(self):
        return self.manager.create_task([{"name": "one", "batch-size": 80}, {"name": "two"}], configuration())

    async def test_all_success_means_completed_and_per_item_exit_codes_saved(self):
        task = self.task()
        await main._run_batch_sequence(task)
        self.assertEqual(task.status, "completed")
        self.assertEqual([item["status"] for item in task.results], ["completed", "completed"])
        self.assertEqual(task.to_dict()["success_count"], 2)
        self.assertEqual(self.runner.start.await_args_list[1].kwargs["base_config"], configuration())

    async def test_nonzero_failure_continues_but_batch_is_failed(self):
        task = self.task()
        self.runner.wait.side_effect = [{"status": "failed", "exit_code": 3, "error": "bad run"}, {"status": "finished", "exit_code": 0}]
        await main._run_batch_sequence(task)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.results[0]["exit_code"], 3)
        self.assertEqual(task.results[1]["status"], "completed")
        self.assertEqual(task.to_dict()["failed_count"], 1)

    async def test_cancelled_worker_is_not_completed_and_does_not_start_next(self):
        task = self.task()
        self.runner.wait.return_value = {"status": "cancelled", "exit_code": 1}
        await main._run_batch_sequence(task)
        self.assertEqual(task.status, "cancelled")
        self.assertEqual(self.runner.start.await_count, 1)

    async def test_sequence_waits_for_previous_cleanup_before_next_start(self):
        task = self.task()
        waiting, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def wait():
            nonlocal calls
            calls += 1
            if calls == 1:
                waiting.set()
                await release.wait()
            return {"status": "finished", "exit_code": 0}

        self.runner.wait.side_effect = wait
        task.job = asyncio.create_task(main._run_batch_sequence(task))
        await waiting.wait()
        self.assertEqual(self.runner.start.await_count, 1)
        await asyncio.sleep(0.01)
        self.assertEqual(self.runner.start.await_count, 1)
        release.set()
        await task.job
        self.assertEqual(self.runner.start.await_count, 2)

    async def test_stop_timeout_does_not_release_runner_or_cancel_cleanup(self):
        from unittest.mock import AsyncMock
        self.runner.is_busy.return_value = True
        slow = asyncio.Event()

        async def stop():
            await slow.wait()
            return {"status": "cancelled"}

        self.runner.stop.side_effect = stop
        with patch.object(main.asyncio, "wait_for", new_callable=AsyncMock, side_effect=asyncio.TimeoutError):
            response = await main.stop_experiment()
        self.assertEqual(response.status_code, 202)
        self.assertTrue(self.runner.is_busy())
        slow.set()
        await asyncio.sleep(0)

    async def test_start_failure_is_recorded_not_completed(self):
        task = self.task()
        self.runner.start.side_effect = [OSError("spawn failed"), None]
        await main._run_batch_sequence(task)
        self.assertEqual(task.status, "failed")
        self.assertIn("spawn failed", task.results[0]["error"])
        self.assertIsNone(task.results[0]["exit_code"])

    async def test_cleanup_failure_stops_sequence_and_preserves_active_reservation(self):
        task = self.task()
        self.runner.wait.return_value = {"status": "cleanup_failed", "exit_code": 1, "error": "cleanup error"}
        self.runner.is_busy.return_value = True
        await main._run_batch_sequence(task)
        self.assertEqual(task.status, "stopping")
        self.assertIs(self.manager.get_active_task(), task)
        self.assertEqual(self.runner.start.await_count, 1)
        self.assertEqual(task.results[1]["status"], "pending")

    async def test_cancel_before_first_run_skips_everything(self):
        task = self.task()
        task.cancel_requested = True
        await main._run_batch_sequence(task)
        self.runner.start.assert_not_awaited()
        self.assertEqual(task.status, "cancelled")
        self.assertEqual(task.to_dict()["cancelled_count"], 2)

    async def test_cancel_running_item_prevents_next_item(self):
        task = self.task()
        started, stopped = asyncio.Event(), asyncio.Event()

        async def wait():
            started.set()
            await stopped.wait()
            return {"status": "cancelled", "exit_code": 1}

        async def stop():
            stopped.set()
            return {"status": "cancelled"}

        self.runner.wait.side_effect = wait
        self.runner.stop.side_effect = stop
        task.job = asyncio.create_task(main._run_batch_sequence(task))
        await started.wait()
        result = await main.cancel_batch_task(task.batch_id)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.runner.start.await_count, 1)
        self.assertEqual([item["status"] for item in task.results], ["cancelled", "cancelled"])

    async def test_single_start_is_blocked_while_batch_pending_or_stopping(self):
        task = self.task()
        for status in ("pending", "stopping"):
            task.status = status
            with self.assertRaises(HTTPException) as exc:
                await main.start_experiment({"name": "single"})
            self.assertEqual(exc.exception.status_code, 409)
        self.runner.start.assert_not_awaited()

    async def test_batch_start_blocked_when_runner_cleaning(self):
        self.runner.is_busy.return_value = True
        with self.assertRaises(HTTPException) as exc:
            await main.run_batch_experiments({"experiments": [{"name": "one"}]})
        self.assertEqual(exc.exception.status_code, 409)
        self.assertIsNone(self.manager.get_active_task())

    async def test_invalid_batch_is_validated_before_task_created(self):
        for experiments in ([{"name": "one"}, {"name": "bad", "batch-size": -1}], [None], "not a list"):
            with self.assertRaises(HTTPException) as exc:
                await main.run_batch_experiments({"experiments": experiments})
            self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(self.manager._tasks, {})

    async def test_reloaded_records_mark_interrupted_not_success_and_keep_finished(self):
        interrupted = self.task()
        interrupted.status = "running"
        interrupted.results[0]["status"] = "running"
        self.manager.update_task(interrupted)
        finished = self.manager.create_task([{"name": "old"}])
        finished.status = "completed"
        finished.results[0].update(status="completed", exit_code=0)
        self.manager.update_task(finished)
        reloaded = BatchManager(self.manager._save_dir)
        self.assertEqual(reloaded.get_task(interrupted.batch_id).status, "failed")
        self.assertEqual(reloaded.get_task(finished.batch_id).status, "completed")
        self.assertIsNone(reloaded.get_active_task())

    async def test_corrupt_record_does_not_prevent_loading_others(self):
        task = self.task()
        (self.manager._save_dir / "corrupt.json").write_text("{")
        reloaded = BatchManager(self.manager._save_dir)
        self.assertIsNotNone(reloaded.get_task(task.batch_id))

    async def test_logs_survive_manager_restart(self):
        task = self.task()
        task.add_log("first line")
        task.add_log("last line")
        task.status = "completed"
        self.manager.update_task(task)
        reloaded = BatchManager(self.manager._save_dir)
        self.assertEqual(reloaded.get_task(task.batch_id).logs, ["first line", "last line"])

    async def test_cancel_cleanup_failure_does_not_release_task_or_mark_cancelled(self):
        task = self.task()
        task.status = "running"
        self.runner.stop.return_value = {"status": "cleanup_failed", "error": "owned cleanup failed"}
        with self.assertRaises(HTTPException) as exc:
            await main.cancel_batch_task(task.batch_id)
        self.assertEqual(exc.exception.status_code, 503)
        self.assertEqual(task.status, "stopping")
        self.assertIs(self.manager.get_active_task(), task)

    async def test_backend_shutdown_waits_for_batch_and_owned_cleanup(self):
        task = self.task()
        task.status = "running"

        async def complete():
            await asyncio.sleep(0.01)
            task.status = "cancelled"

        task.job = asyncio.create_task(complete())
        async with main.lifespan(main.app):
            pass
        self.runner.stop.assert_awaited_once()
        self.assertTrue(task.cancel_requested)
        self.assertTrue(task.job.done())


if __name__ == "__main__":
    unittest.main()