"""History and report regressions use disposable files, never real run artifacts."""

import asyncio
import io
import json
import subprocess
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
from fastapi.testclient import TestClient
from backend.tests.auth_support import authenticated_client, authenticated_session

from backend import main, results_scanner as scanner
from backend.batch_manager import BatchTask
from backend.catalog import Catalog
from backend.experiment_runner import ExperimentRunner
from fl_backdoor.utils.run_artifacts import reserve_experiment, write_run_metadata


class TemporaryHistory:
    def prepare(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        replacement = patch.object(main, "get_catalog", return_value=Catalog(self.root / "catalog.sqlite3"))
        replacement.start()
        self.addCleanup(replacement.stop)

    def experiment(self, name="one", content="round,accuracy,asr,loss\n1,0.9,0.1,0.2\n", status="finished"):
        directory, metadata = reserve_experiment(self.root, name)
        metadata.update(status=status, exit_code=0 if status == "finished" else 1)
        if status == "failed":
            metadata["error"] = "训练失败，请查看日志。"
        write_run_metadata(directory, metadata)
        if content is not None:
            (directory / f"{name}.csv").write_text(content, encoding="utf-8")
        return directory


class HistoryReaderTests(TemporaryHistory, unittest.TestCase):
    def setUp(self):
        self.prepare()

    def test_damaged_files_and_missing_columns_do_not_hide_healthy_runs(self):
        self.experiment("healthy")
        for name, data in (("partial", b"round,accuracy\n1,0.8\n"), ("empty", b""),
                           ("truncated", b'round,accuracy,asr,loss\n1,"unterminated'),
                           ("bad_encoding", b"\xff\xfe\x81")):
            directory = self.experiment(name, content=None)
            (directory / f"{name}.csv").write_bytes(data)
        items = {item["name"]: item for item in scanner.list_experiments(self.root)}
        self.assertEqual(set(items), {"healthy", "partial", "empty", "truncated", "bad_encoding"})
        self.assertEqual(items["healthy"]["accuracy"], 0.9)
        self.assertEqual(items["partial"]["accuracy"], 0.8)
        self.assertIsNone(items["partial"]["asr"])
        self.assertFalse(items["partial"]["can_plot"])
        for name in ("empty", "truncated", "bad_encoding"):
            self.assertIsNone(items[name]["accuracy"])
            self.assertTrue(items[name]["warnings"])
        json.dumps(items, allow_nan=False)

    def test_failed_cancelled_and_starting_runs_without_csv_are_visible(self):
        for status in ("failed", "cancelled", "starting"):
            directory = self.experiment(status, content=None, status=status)
            detail = scanner.get_experiment_detail(directory)
            self.assertEqual(detail["status"], status)
            self.assertFalse(detail["has_metrics"])
            self.assertFalse(detail["can_plot"])
            for field in ("accuracy", "asr", "loss", "last_round"):
                self.assertIsNone(detail["metrics"][field])
            self.assertEqual(detail["metrics"]["curve_data"]["round"], [])
        self.assertEqual(len(scanner.list_experiments(self.root)), 3)

    def test_old_log_only_run_is_visible_but_unrelated_directories_are_not(self):
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "run.log").write_text("old run failed", encoding="utf-8")
        (self.root / "empty").mkdir()
        (self.root / "batches").mkdir()
        (self.root / "summaries").mkdir()
        self.assertEqual([item["name"] for item in scanner.list_experiments(self.root)], ["legacy"])
        self.assertEqual(scanner.get_experiment_detail(legacy)["status"], "unknown")

    def test_broken_optional_files_do_not_discard_main_metrics(self):
        directory = self.experiment()
        (directory / "one_metrics.csv").write_text("key,value\nwrong,NaN\n", encoding="utf-8")
        (directory / "one_clients.csv").write_bytes(b'round,is_malicious,score\n1,"unfinished')
        detail = scanner.get_experiment_detail(directory)
        self.assertEqual(detail["metrics"]["accuracy"], 0.9)
        self.assertTrue(detail["can_plot"])
        self.assertEqual(len(detail["warnings"]), 2)
        self.assertIsNone(scanner.compute_auc_from_clients_csv(directory))

    def test_invalid_numbers_rounds_and_duplicates_are_json_safe(self):
        directory = self.experiment(content=(
            "round,accuracy,asr,loss\n2,0.5,0.2,0.4\nwrong,0.3,0.4,0.5\n"
            "1.5,0.3,0.4,0.5\n-1,0.3,0.4,0.5\n1,0.6,0.1,0.3\n2,inf,NaN,bad\n"))
        detail = scanner.get_experiment_detail(directory)
        curves = detail["metrics"]["curve_data"]
        self.assertEqual(curves["round"], [1, 2])
        self.assertEqual(curves["accuracy"], [0.6, None])
        self.assertIsNone(detail["metrics"]["loss"])
        self.assertTrue(detail["can_plot"])
        json.dumps(detail, allow_nan=False)

    def test_valid_detection_aggregation_and_auc_are_preserved(self):
        directory = self.experiment()
        (directory / "one_metrics.csv").write_text(
            "round,component,key,value\n1,detection,detect_precision,0.75\n"
            "1,aggregation,agg_selected_clients,5\n2,aggregation,agg_selected_clients,7\n",
            encoding="utf-8")
        (directory / "one_clients.csv").write_text(
            "round,is_malicious,score\n1,0,0.1\n1,1,0.9\n1,-1,invalid\n", encoding="utf-8")
        metrics = scanner.get_experiment_detail(directory)["metrics"]
        self.assertEqual(metrics["precision"], 0.75)
        self.assertEqual(metrics["avg_selected_clients"], 6)
        self.assertEqual(metrics["auc"], 1)

    def test_broken_snapshot_or_metadata_do_not_hide_existing_metrics(self):
        directory = self.experiment()
        (directory / "run.json").write_text("broken", encoding="utf-8")
        (directory / "config_snapshot.toml").write_text("[unfinished", encoding="utf-8")
        detail = scanner.get_experiment_detail(directory)
        self.assertEqual(detail["metrics"]["accuracy"], 0.9)
        self.assertIsNone(detail["run_id"])
        self.assertTrue(detail["warnings"])

    def test_summary_curves_skip_broken_run_and_keep_partial_run(self):
        self.experiment("healthy")
        self.experiment("partial", "round,accuracy\n1,0.8\n")
        self.experiment("broken", 'round,accuracy\n1,"unfinished')
        summary = self.root / "summaries" / "report"
        summary.mkdir(parents=True)
        (summary / "summary.csv").write_text(
            "experiment,accuracy\nhealthy,0.9\npartial,0.8\nbroken,\n../outside,\n", encoding="utf-8")
        detail = scanner.get_summary_detail(summary)
        self.assertEqual(set(detail["experiments_curve_data"]), {"healthy", "partial"})
        self.assertEqual(detail["experiments_curve_data"]["partial"]["asr"], [None])
        json.dumps(detail, allow_nan=False)


class HistoryApiTests(TemporaryHistory, unittest.TestCase):
    def setUp(self):
        self.prepare()
        replacement = patch.object(main, "_results_root", return_value=self.root)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.client = authenticated_client(self, main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_history_and_failed_detail_and_log_download(self):
        self.experiment("healthy")
        directory = self.experiment("failed", content=None, status="failed")
        (directory / "run.log").write_bytes(b"failure details")
        response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()), 2)
        response = self.client.get("/api/experiments/failed")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["metrics"]["accuracy"])
        response = self.client.get("/api/experiments/failed/all/download")
        self.assertEqual(response.status_code, 200, response.text)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(archive.read("run.log"), b"failure details")

    def test_report_validation_preserves_http_status_and_never_starts_worker(self):
        self.experiment("partial", "round,accuracy\n1,0.8\n")
        self.experiment("failed", content=None, status="failed")
        with patch.object(main.subprocess, "run") as worker:
            for url in ("/api/experiments/missing/plot", "/api/summaries/missing/plot"):
                self.assertEqual(self.client.post(url).status_code, 404)
            for name in ("partial", "failed"):
                response = self.client.post(f"/api/experiments/{name}/plot")
                self.assertEqual(response.status_code, 422, response.text)
            for payload, status in (({"experiments": ["missing"]}, 404),
                                    ({"experiments": ["failed"]}, 422),
                                    ({"experiments": "partial"}, 400),
                                    ({"experiments": ["../outside"]}, 400)):
                response = self.client.post("/api/summarize", json=payload)
                self.assertEqual(response.status_code, status, response.text)
            worker.assert_not_called()

    def test_invalid_summary_is_422_and_existing_summary_is_409(self):
        self.experiment()
        summary = self.root / "summaries" / "report"
        summary.mkdir(parents=True)
        (summary / "summary.csv").write_text('experiment\n"unfinished', encoding="utf-8")
        self.assertEqual(self.client.post("/api/summaries/report/plot").status_code, 422)
        self.assertEqual(self.client.post("/api/summarize", json={
            "experiments": ["one"], "output_name": "report"}).status_code, 409)

    def test_delete_removes_only_selected_run_and_leaves_summary_and_neighbor(self):
        directory = self.experiment()
        run_id = json.loads((directory / "run.json").read_text())["run_id"]
        (directory / "final_model.pt").write_bytes(b"model")
        (directory / "plots").mkdir()
        (directory / "plots" / "figure.png").write_bytes(b"image")
        neighbor = self.experiment("neighbor")
        summary = self.root / "summaries" / "report"
        summary.mkdir(parents=True)
        (summary / "summary.csv").write_bytes(b"experiment\none\n")
        response = self.client.delete("/api/experiments/one", params={"run_id": run_id})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertTrue((neighbor / "neighbor.csv").is_file())
        self.assertEqual((summary / "summary.csv").read_bytes(), b"experiment\none\n")
        self.assertEqual(self.client.get("/api/experiments/one").status_code, 404)
        self.assertEqual(self.client.delete("/api/experiments/one").status_code, 404)

    def test_delete_rejects_stale_uuid_invalid_name_and_link(self):
        directory = self.experiment()
        response = self.client.delete("/api/experiments/one", params={"run_id": "stale"})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(directory.exists())
        for name in ("NUL", "batches", "summaries", "..%5Coutside"):
            self.assertEqual(self.client.delete(f"/api/experiments/{name}").status_code, 400)
        from fl_backdoor.tests.test_run_artifacts import directory_link
        alias = self.root / "alias"
        directory_link(alias, directory)
        self.addCleanup(lambda: alias.unlink() if alias.is_symlink() else alias.rmdir())
        self.assertEqual(self.client.delete("/api/experiments/alias").status_code, 400)
        self.assertTrue((directory / "one.csv").is_file())

    def test_delete_failed_run_without_csv_and_legacy_run(self):
        directory = self.experiment("failed", content=None, status="failed")
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "run.log").write_bytes(b"old log")
        for name in ("failed", "legacy"):
            response = self.client.delete(f"/api/experiments/{name}")
            self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertFalse(legacy.exists())

    def test_delete_rejects_owned_running_and_cleanup_states_not_unrelated_run(self):
        directory = self.experiment()
        runner = ExperimentRunner()
        runner.exp_name, runner._experiment_dir = "one", directory
        with patch.object(main, "runner", runner):
            for status in ("starting", "running", "stopping", "cleaning", "cleanup_failed"):
                runner._status = status
                response = self.client.delete("/api/experiments/one")
                self.assertEqual(response.status_code, 409, response.text)
                self.assertTrue(directory.exists())
            other = self.root / "elsewhere" / "one"
            other.mkdir(parents=True)
            runner._experiment_dir = other
            self.assertEqual(self.client.delete("/api/experiments/one").status_code, 200)
            self.assertTrue(other.exists())

    def test_delete_rejects_run_referenced_by_active_batch(self):
        directory = self.experiment()
        batch = BatchTask("test", [{"name": "one"}])
        with patch.object(main.batch_manager, "get_active_task", return_value=batch):
            response = self.client.delete("/api/experiments/one")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(directory.exists())

    def test_delete_io_failure_keeps_error_chinese_and_releases_reservation(self):
        directory = self.experiment()
        with patch.object(main.shutil, "rmtree", side_effect=PermissionError("private diagnostic")):
            response = self.client.delete("/api/experiments/one")
        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("删除实验失败", response.json()["detail"])
        self.assertNotIn("private", response.text)
        self.assertNotIn(directory.resolve(), main._report_directories)
        self.assertTrue(directory.exists())


    def test_delete_summary_removes_only_selected_summary(self):
        summaries = self.root / "summaries"
        keep = summaries / "keep"
        keep.mkdir(parents=True)
        (keep / "summary.csv").write_bytes(b"experiment\none\n")
        removed = summaries / "remove"
        removed.mkdir()
        (removed / "summary.csv").write_bytes(b"experiment\none\n")
        (removed / "chart.png").write_bytes(b"image")
        response = self.client.delete("/api/summaries/remove")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"status": "success", "name": "remove"})
        self.assertFalse(removed.exists())
        self.assertEqual((keep / "summary.csv").read_bytes(), b"experiment\none\n")
        self.assertEqual(self.client.delete("/api/summaries/remove").status_code, 404)
        for name in ("NUL", "..%5Coutside"):
            self.assertEqual(self.client.delete(f"/api/summaries/{name}").status_code, 400)

    def test_delete_summary_rejects_active_report_without_removing_files(self):
        summary = self.root / "summaries" / "busy"
        summary.mkdir(parents=True)
        (summary / "summary.csv").write_bytes(b"experiment\none\n")
        main._report_directories.add(summary.resolve())
        try:
            response = self.client.delete("/api/summaries/busy")
        finally:
            main._report_directories.discard(summary.resolve())
        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(summary.exists())

    def test_delete_summary_io_failure_keeps_error_chinese_and_releases_reservation(self):
        summary = self.root / "summaries" / "report"
        summary.mkdir(parents=True)
        (summary / "summary.csv").write_bytes(b"experiment\none\n")
        with patch.object(main.shutil, "rmtree", side_effect=PermissionError("private diagnostic")):
            response = self.client.delete("/api/summaries/report")
        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("删除总结失败", response.json()["detail"])
        self.assertNotIn("private", response.text)
        self.assertNotIn(summary.resolve(), main._report_directories)
        self.assertTrue(summary.exists())


class ReportWorkerTests(TemporaryHistory, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prepare()
        self.token, self.csrf = authenticated_session(self)

    async def test_slow_report_does_not_block_status_and_duplicate_is_409(self):
        self.experiment()
        entered, release = threading.Event(), threading.Event()
        def execute(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release report worker")
            return subprocess.CompletedProcess(args[0], 0, "generated", "")

        with patch.object(main, "_results_root", return_value=self.root), patch.object(
                main.subprocess, "run", side_effect=execute):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test",
                                         headers={"X-CSRF-Token": self.csrf, "X-Lianan-Request": "1"},
                                         cookies={"lianan_session": self.token}) as client:
                job = asyncio.create_task(client.post("/api/experiments/one/plot"))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    response = await asyncio.wait_for(client.get("/api/experiment/status"), 1)
                    self.assertEqual(response.status_code, 200)
                    response = await asyncio.wait_for(client.post("/api/experiments/one/plot"), 1)
                    self.assertEqual(response.status_code, 409, response.text)
                    response = await client.delete("/api/experiments/one")
                    self.assertEqual(response.status_code, 409, response.text)
                    self.assertTrue((self.root / "one").exists())
                finally:
                    release.set()
                    response = await job
                self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn((self.root / "one").resolve(), main._report_directories)

    async def test_report_worker_timeout_failure_and_success(self):
        for outcome, status in ((subprocess.TimeoutExpired("plot", 300), 504),
                                (OSError("missing interpreter"), 500),
                                (subprocess.CompletedProcess([], 1, "", "private stack"), 500)):
            with self.subTest(outcome=outcome), patch.object(main.subprocess, "run", side_effect=(
                    outcome if isinstance(outcome, Exception) else None), return_value=outcome):
                with self.assertRaises(main.HTTPException) as caught:
                    await main._run_report(["python", "plot.py"], "图表生成")
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn("private stack", caught.exception.detail)
        with patch.object(main.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "")) as worker:
            result = await main._run_report(["python", "plot.py"], "图表生成")
            self.assertEqual(result.stdout, "ok")
            self.assertEqual(worker.call_args.kwargs["timeout"], main.REPORT_TIMEOUT)
            self.assertEqual(worker.call_args.kwargs["env"]["MPLBACKEND"], "Agg")

    async def test_cancelled_request_holds_slot_until_worker_exits(self):
        entered, release = threading.Event(), threading.Event()
        def execute(*args, **kwargs):
            entered.set()
            release.wait(5)
            return subprocess.CompletedProcess([], 0, "", "")
        async def generate():
            async with main._report_slot(self.root):
                await main._run_report(["python"], "图表生成")
        with patch.object(main.subprocess, "run", side_effect=execute):
            job = asyncio.create_task(generate())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                job.cancel()
                await asyncio.sleep(0)
                with self.assertRaises(main.HTTPException) as caught:
                    async with main._report_slot(self.root):
                        self.fail("worker still owns the output")
                self.assertEqual(caught.exception.status_code, 409)
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
        self.assertNotIn(self.root.resolve(), main._report_directories)

    async def test_deletion_is_nonblocking_and_blocks_concurrent_plot_summary_and_delete(self):
        directory = self.experiment()
        entered, release = threading.Event(), threading.Event()
        def remove(path):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release deletion")
        with patch.object(main, "_results_root", return_value=self.root), patch.object(main.shutil, "rmtree", side_effect=remove):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test",
                                         headers={"X-CSRF-Token": self.csrf, "X-Lianan-Request": "1"},
                                         cookies={"lianan_session": self.token}) as client:
                job = asyncio.create_task(client.delete("/api/experiments/one"))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    response = await asyncio.wait_for(client.get("/api/experiment/status"), 1)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual((await client.post("/api/experiments/one/plot")).status_code, 409)
                    response = await client.post("/api/summarize", json={"experiments": ["one"]})
                    self.assertEqual(response.status_code, 409, response.text)
                finally:
                    release.set()
                    self.assertEqual((await job).status_code, 200)
        self.assertNotIn(directory.resolve(), main._report_directories)

    async def test_summary_inputs_are_reserved_and_released_even_on_error(self):
        directory = self.experiment()
        with patch.object(main, "_results_root", return_value=self.root):
            async with main._report_slot(self.root / "summaries" / "report", sources=[directory]):
                with self.assertRaises(main.HTTPException) as caught:
                    await main.delete_experiment("one")
                self.assertEqual(caught.exception.status_code, 409)
                self.assertTrue(directory.exists())
            self.assertNotIn(directory.resolve(), main._report_inputs)
            with patch.object(main.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "error")):
                with self.assertRaises(main.HTTPException):
                    await main.create_summary({"experiments": ["one"], "output_name": "fail-report"})
            self.assertNotIn(directory.resolve(), main._report_inputs)

    async def test_cancelled_deletion_keeps_slot_until_removal_finishes(self):
        directory = self.experiment()
        entered, release = threading.Event(), threading.Event()
        def remove(path):
            entered.set()
            release.wait(5)
        with patch.object(main, "_results_root", return_value=self.root), patch.object(main.shutil, "rmtree", side_effect=remove):
            job = asyncio.create_task(main.delete_experiment("one"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                job.cancel()
                await asyncio.sleep(0)
                self.assertIn(directory.resolve(), main._report_directories)
                self.assertTrue(main.execution_lock.locked())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
        self.assertNotIn(directory.resolve(), main._report_directories)
        self.assertFalse(main.execution_lock.locked())


class SummaryScriptTests(TemporaryHistory, unittest.TestCase):
    def setUp(self):
        self.prepare()

    def test_shared_reader_keeps_partial_rows_and_skips_unusable_runs(self):
        from scripts import summarize_results as script
        self.experiment("healthy")
        self.experiment("partial", "round,accuracy\n1,0.8\n")
        self.experiment("failed", content=None, status="failed")
        self.experiment("damaged", 'round,accuracy\n1,"unfinished')
        with patch.object(script, "RESULTS_DIR", self.root), patch.object(script, "SUMMARIES_DIR", self.root / "summaries"), patch(
                "sys.argv", ["summarize_results.py", "--results-dir", str(self.root), "--output-name", "report", "--no-plots"]):
            script.main()
        frame = pd.read_csv(self.root / "summaries" / "report" / "summary.csv")
        self.assertEqual(set(frame["experiment"]), {"healthy", "partial"})
        partial = frame.set_index("experiment").loc["partial"]
        self.assertEqual(partial["accuracy"], 0.8)
        self.assertTrue(pd.isna(partial["asr"]))
        self.assertTrue(pd.isna(partial["loss"]))

    def test_no_usable_metrics_creates_no_report_directory(self):
        from scripts import summarize_results as script
        self.experiment("failed", content=None, status="failed")
        with patch.object(script, "RESULTS_DIR", self.root), patch.object(script, "SUMMARIES_DIR", self.root / "summaries"), patch(
                "sys.argv", ["summarize_results.py", "--results-dir", str(self.root), "--output-name", "report", "--no-plots"]):
            with self.assertRaises(SystemExit) as caught:
                script.main()
        self.assertNotEqual(caught.exception.code, 0)
        self.assertFalse((self.root / "summaries").exists())


if __name__ == "__main__":
    unittest.main()