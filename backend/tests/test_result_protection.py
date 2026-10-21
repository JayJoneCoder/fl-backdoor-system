"""Result and backup APIs never operate on real configurations or experiments."""

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import tomli
import tomli_w
from fastapi.testclient import TestClient
from backend.tests.auth_support import authenticated_client

from backend import config_manager as cm, experiment_runner as er, main
from backend.batch_manager import BatchManager
from backend.catalog import Catalog
from backend.results_scanner import get_summary_detail, list_experiments
from fl_backdoor.utils.run_artifacts import read_run_metadata, reserve_experiment


def configuration(**updates):
    values = {"run-name": "baseline", "dataset": "mnist", "attack": "none", "detection": "none",
              "defense": "none", "num-clients": 10, "num-server-rounds": 2, "batch-size": 32}
    values.update(updates)
    return tomli_w.dumps({"project": {"name": "test-project", "version": "1.0"}, "tool": {"flwr": {"app": {
        "config": values, "components": {"serverapp": "fl_backdoor.server.server:app", "clientapp": "fl_backdoor.client.client:app"},
    }}}}).encode("utf-8")


class TemporaryResultFiles:
    def prepare_files(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        replacement = patch.object(main, "get_catalog", return_value=Catalog(self.root / "catalog.sqlite3"))
        replacement.start()
        self.addCleanup(replacement.stop)
        self.results = self.root / "results"
        self.results.mkdir()
        self.config = self.root / "pyproject.toml"
        self.original = configuration()
        self.config.write_bytes(self.original)
        for obj, name, value in ((cm, "PROJECT_ROOT", self.root), (cm, "TOML_PATH", self.config),
                                 (cm, "BACKUP_PATH", self.root / "pyproject.toml.bak"),
                                 (er, "RESULTS_DIR", self.results), (main, "RESULTS_DIR", self.results)):
            replacement = patch.object(obj, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def experiment(self, name="one"):
        directory, metadata = reserve_experiment(self.results, name)
        (directory / f"{name}.csv").write_text("round,accuracy,asr,loss\n1,0.9,0.1,0.2\n", encoding="utf-8")
        (directory / "config_snapshot.toml").write_bytes(self.original)
        return directory, metadata


class BackupBoundaryTests(TemporaryResultFiles, unittest.TestCase):
    def setUp(self):
        self.prepare_files()

    def test_new_backups_live_in_separate_directory(self):
        path = cm.create_backup("before")
        self.assertEqual(path, self.root / "config_backups" / "before.toml")
        self.assertEqual(path.read_bytes(), self.original)
        self.assertFalse((self.root / "before.toml").exists())
        self.assertEqual(cm.list_backups()[0]["filename"], "before.toml")

    def test_main_configuration_cannot_be_read_edited_deleted_or_restored_as_backup(self):
        for filename in ("pyproject.toml", "PYPROJECT.TOML", "PyProject.toml"):
            for action in (cm.read_backup_content, cm.delete_backup,
                           lambda name: cm.restore_from_backup(name, backup_current=False),
                           lambda name: cm.update_backup_content(name, configuration(**{"batch-size": 64}).decode())):
                with self.subTest(filename=filename, action=action), self.assertRaises(ValueError):
                    action(filename)
                self.assertEqual(self.config.read_bytes(), self.original)

    def test_backup_name_traversal_and_device_names_are_rejected(self):
        for name in ("../outside", "..\\outside", "C:\\outside", "other:name", "NUL", "CON.toml", "--option", ".hidden", "trailing."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cm.create_backup(name, "overwrite")
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_legacy_backups_remain_readable_editable_and_restorable(self):
        legacy = self.root / "legacy.toml"
        legacy.write_bytes(configuration(**{"batch-size": 8}))
        self.assertIn("batch-size = 8", cm.read_backup_content(legacy.name))
        cm.update_backup_content(legacy.name, configuration(**{"batch-size": 16}).decode())
        cm.restore_from_backup(legacy.name, backup_current=False)
        self.assertIn(b"batch-size = 16", self.config.read_bytes())
        cm.delete_backup(legacy.name)
        self.assertFalse(legacy.exists())

    def test_unrelated_root_toml_is_not_a_deletable_backup(self):
        path = self.root / "tool-settings.toml"
        path.write_text("[tool]\nsetting = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            cm.delete_backup(path.name)
        self.assertTrue(path.is_file())
        self.assertEqual(cm.list_backups(), [])

    def test_new_backup_overwrite_does_not_modify_legacy_file(self):
        legacy = self.root / "before.toml"
        legacy.write_bytes(self.original)
        self.assertIsNone(cm.create_backup("before", "ask"))
        path = cm.create_backup_from_content(configuration(**{"batch-size": 16}).decode(), "before", "overwrite")
        self.assertEqual(path.parent, self.root / "config_backups")
        self.assertEqual(legacy.read_bytes(), self.original)
        self.assertIn("batch-size = 16", cm.read_backup_content("before.toml"))
        self.assertEqual(len(cm.list_backups()), 1)

    def test_conflicting_legacy_name_auto_suffix_uses_new_directory(self):
        (self.root / "before.toml").write_bytes(self.original)
        (self.root / "before(1).toml").write_bytes(self.original)
        path = cm.create_backup("before", "auto")
        self.assertEqual(path, self.root / "config_backups" / "before(2).toml")

    def test_backup_directory_link_is_rejected(self):
        from fl_backdoor.tests.test_run_artifacts import directory_link

        target = self.root / "outside"
        target.mkdir()
        link = self.root / "config_backups"
        directory_link(link, target)
        self.addCleanup(lambda: link.unlink() if link.is_symlink() else link.rmdir())
        with self.assertRaises(ValueError):
            cm.create_backup("before")
        self.assertEqual(list(target.iterdir()), [])


class ResultApiTests(TemporaryResultFiles, unittest.TestCase):
    def setUp(self):
        self.prepare_files()
        self.runner = er.ExperimentRunner()
        self.manager = BatchManager(self.results / "batches")
        replacement = patch.multiple(main, runner=self.runner, batch_manager=self.manager, execution_lock=asyncio.Lock())
        replacement.start()
        self.addCleanup(replacement.stop)
        self.client = authenticated_client(self, main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_new_result_directory_lists_as_empty_without_creating_it(self):
        cm.update_config({"results-dir": "future-results"})
        response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [])
        self.assertFalse((self.root / "future-results").exists())

    def test_history_stays_in_selected_root_even_when_same_name_is_running(self):
        old_dir, old = self.experiment("same")
        self.runner.exp_name = "same"
        self.runner.run_id = old["run_id"]
        self.runner._experiment_dir = old_dir
        self.runner._status = "running"
        other_root = self.root / "other-results"
        new_dir, new = reserve_experiment(other_root, "same")
        (new_dir / "same.csv").write_text("round,accuracy,asr,loss\n1,0.2,0.3,0.4\n", encoding="utf-8")
        cm.update_config({"results-dir": str(other_root)})
        detail = self.client.get("/api/experiments/same").json()
        self.assertEqual(detail["run_id"], new["run_id"])
        self.assertEqual(detail["metrics"]["accuracy"], 0.2)
        self.assertEqual(main._monitor_directory("same", old["run_id"]), old_dir)
        self.assertFalse(main._runner_owns_directory(new_dir))
        self.assertTrue(main._runner_owns_directory(old_dir))

    def test_websocket_run_id_mismatch_is_closed(self):
        self.experiment("one")
        from starlette.websockets import WebSocketDisconnect
        with self.assertRaises(WebSocketDisconnect) as rejected:
            with self.client.websocket_connect("/ws/logs/one?run_id=not-this-run", headers={"Origin": "http://testserver"}) as socket:
                socket.receive_json()
        self.assertEqual(rejected.exception.code, 1008)

    def test_repeated_name_returns_the_effective_uuid_name(self):
        async def start(name, config, **kwargs):
            self.runner.exp_name = f"{name}-12345678"
            self.runner.run_id = uuid.uuid4().hex

        self.runner.start = AsyncMock(side_effect=start)
        response = self.client.post("/api/experiment/start", json={"name": "one"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["name"], "one-12345678")
        self.assertEqual(response.json()["run_id"], self.runner.run_id)

    def test_invalid_start_name_returns_400_without_creating_directory(self):
        for name in ("../outside", "..\\outside", "C:\\outside", "NUL", "batches", "name.", "--option", ["not", "text"]):
            response = self.client.post("/api/experiment/start", json={"name": name})
            self.assertEqual(response.status_code, 400, (name, response.text))
        self.assertEqual(list(self.results.iterdir()), [self.results / "batches"])
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_batch_repeated_or_existing_names_are_allowed(self):
        self.experiment("one")
        experiments = [{"name": "one"}, {"name": "NEW"}]
        with patch.object(main, "_run_batch_sequence", new=AsyncMock()):
            response = self.client.post("/api/batch/run", json={"experiments": experiments})
            self.assertEqual(response.status_code, 200, response.text)
        task = self.manager.get_latest_task()
        self.assertIsNotNone(task)
        self.assertEqual([item["name"] for item in task.experiments], ["one", "NEW"])

    def test_delete_blocks_active_batch_effective_name(self):
        directory, _ = self.experiment("one-12345678")
        task = self.manager.create_task([{"name": "one"}], configuration())
        task.results[0].update(name="one-12345678", status="running")
        self.manager.update_task(task)
        response = self.client.delete("/api/experiments/one-12345678")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(directory.is_dir())

    def test_model_manifest_csv_and_snapshot_in_experiment_download(self):
        directory, metadata = self.experiment("中文实验")
        (directory / "final_model.pt").write_bytes(b"synthetic model")
        (directory / "run.log").write_bytes(b"logs")
        (directory / ".partial.tmp").write_bytes(b"not an artifact")
        response = self.client.get("/api/experiments/中文实验/all/download")
        self.assertEqual(response.status_code, 200, response.text[:100])
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {"run.json", "final_model.pt", "中文实验.csv", "config_snapshot.toml", "run.log"})
            self.assertEqual(archive.read("final_model.pt"), b"synthetic model")
        detail = self.client.get("/api/experiments/中文实验").json()
        self.assertEqual(detail["run_id"], metadata["run_id"])
        self.assertEqual(detail["model_file"], "final_model.pt")
        self.assertIn("filename*=UTF-8''", response.headers["content-disposition"])

    def test_result_list_skips_linked_experiments(self):
        from fl_backdoor.tests.test_run_artifacts import directory_link

        target, _ = self.experiment("one")
        link = self.results / "alias"
        directory_link(link, target)
        self.addCleanup(lambda: link.unlink() if link.is_symlink() else link.rmdir())
        self.assertEqual([item["name"] for item in list_experiments(self.results)], ["one"])
        self.assertEqual(self.client.get("/api/experiments/alias/all/download").status_code, 400)

    def test_invalid_names_for_result_and_summary_apis_return_400(self):
        for path in ("/api/experiments/NUL", "/api/experiments/..%5Coutside/all/download",
                     "/api/experiments/batches/images/download", "/api/summaries/..%5Coutside"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 400, (path, response.text))

    def test_missing_experiment_is_404_not_500(self):
        self.assertEqual(self.client.get("/api/experiments/missing").status_code, 404)

    def test_summary_creation_refuses_existing_output_or_invalid_input(self):
        self.experiment("one")
        output = self.results / "summaries" / "old"
        output.mkdir(parents=True)
        (output / "summary.csv").write_bytes(b"original report")
        with patch.object(main.subprocess, "run") as execute:
            for data, code in (({"experiments": ["one"], "output_name": "old"}, 409),
                               ({"experiments": ["../outside"]}, 400),
                               ({"experiments": ["one"], "output_name": "../outside"}, 400),
                               ({"experiments": ["--option"]}, 400)):
                response = self.client.post("/api/summarize", json=data)
                self.assertEqual(response.status_code, code, response.text)
            execute.assert_not_called()
        self.assertEqual((output / "summary.csv").read_bytes(), b"original report")

    def test_summary_curves_use_result_root_and_ignore_paths_from_csv(self):
        self.experiment("one")
        output = self.results / "summaries" / "comparison"
        output.mkdir(parents=True)
        (output / "summary.csv").write_text("experiment,accuracy\none,0.9\n../outside,0.1\n", encoding="utf-8")
        detail = get_summary_detail(output)
        self.assertEqual(set(detail["experiments_curve_data"]), {"one"})
        self.assertEqual(detail["experiments_curve_data"]["one"]["round"], [1])

    def test_backup_api_rejects_main_config_and_preserves_it(self):
        for method in ("get", "delete", "put"):
            kwargs = {"json": {"content": self.original.decode()}} if method == "put" else {}
            response = getattr(self.client, method)("/api/config/backups/pyproject.toml", **kwargs)
            self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/api/config/restore", json={"filename": "pyproject.toml"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.config.read_bytes(), self.original)


class RunIdentityTests(TemporaryResultFiles, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prepare_files()
        self.runner = er.ExperimentRunner()

    async def test_successful_run_persists_id_and_separates_flower_id(self):
        owner = Mock()
        owner.spawn.return_value.stdout = io.StringIO("🎊 Successfully started run 123456\n")
        owner.spawn.return_value.poll.return_value = 0
        with patch.object(er, "OwnedProcessTree", return_value=owner):
            await self.runner.start("one")
            result = await self.runner.wait()
        directory = self.results / "one"
        metadata = read_run_metadata(directory)
        uuid.UUID(metadata["run_id"])
        self.assertEqual(result["run_id"], metadata["run_id"])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["flower_run_id"], "123456")
        self.assertEqual(metadata["flower_run_id"], "123456")
        self.assertEqual(metadata["status"], "finished")
        self.assertEqual(metadata["exit_code"], 0)
        env = owner.spawn.call_args.kwargs["env"]
        self.assertEqual(env["FL_BACKDOOR_RUN_ID"], metadata["run_id"])
        snapshot = tomli.loads((directory / "config_snapshot.toml").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["tool"]["flwr"]["app"]["config"]["results-dir"], str(self.results.resolve()))
        self.assertEqual(self.config.read_bytes(), self.original)

    async def test_failed_start_records_failure_and_keeps_old_directory(self):
        with patch.object(self.runner, "_prepare_runtime", side_effect=RuntimeError("startup failed")):
            with self.assertRaises(RuntimeError):
                await self.runner.start("one")
        metadata = read_run_metadata(self.results / "one")
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("startup failed", metadata["error"])
        self.assertFalse(self.runner.is_busy())
        # A retry must reserve a new UUID directory instead of reusing the failed one.
        new_dir, new_metadata = reserve_experiment(self.results, "one")
        self.assertNotEqual(new_dir, self.results / "one")
        self.assertEqual(new_metadata["requested_name"], "one")
        self.assertEqual(read_run_metadata(self.results / "one")["run_id"], metadata["run_id"])
        self.assertEqual(self.config.read_bytes(), self.original)

    async def test_api_returns_run_id_after_successful_reservation(self):
        fake = Mock()
        fake.run_id = uuid.uuid4().hex
        fake.start = AsyncMock()
        manager = BatchManager(self.results / "batches")
        with patch.multiple(main, runner=fake, batch_manager=manager, execution_lock=asyncio.Lock()):
            response = await main.start_experiment({"name": "one"})
        self.assertEqual(response["run_id"], fake.run_id)


if __name__ == "__main__":
    unittest.main()