"""SQLite catalog tests only touch disposable files and databases."""

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.tests.auth_support import authenticated_client

from backend import main
from backend.catalog import APPLICATION_ID, Catalog
from fl_backdoor.utils.run_artifacts import reserve_experiment, write_run_metadata


class CatalogFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "results"
        self.root.mkdir()
        self.catalog = Catalog(self.base / "catalog.sqlite3")

    def run_files(self, name="one", *, legacy=False, status="finished", metrics=True):
        if legacy:
            directory = self.root / name
            directory.mkdir()
        else:
            directory, metadata = reserve_experiment(self.root, name)
            metadata.update(status=status)
            write_run_metadata(directory, metadata)
        (directory / "run.log").write_text("训练日志 <plain>\n", encoding="utf-8")
        (directory / "config_snapshot.toml").write_text(
            '[tool.flwr.app.config]\ndataset = "mnist"\nnum-clients = 3\n', encoding="utf-8")
        if metrics:
            (directory / f"{name}.csv").write_text("round,accuracy,asr,loss\n1,0.8,0.1,0.2\n", encoding="utf-8")
            (directory / f"{name}_clients.csv").write_text("round,client_id,is_malicious,score\n1,0,0,0.1\n1,1,1,0.8\n", encoding="utf-8")
        return directory


class CatalogTests(CatalogFixture):
    def test_initialize_schema_and_integrity(self):
        stats = self.catalog.stats()
        self.assertEqual(stats["schema_version"], 1)
        self.assertEqual(stats["integrity"], "ok")
        self.assertEqual(stats["counts"]["runs"], 0)
        with self.catalog.connection() as con:
            self.assertEqual(con.execute("PRAGMA application_id").fetchone()[0], APPLICATION_ID)
            self.assertEqual(con.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_import_is_idempotent_and_preserves_uuid_curves_and_source_files(self):
        directory = self.run_files()
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.assertEqual(self.catalog.sync_root(self.root)["imported"], 1)
        detail = self.catalog.detail(self.root, "one")
        self.assertEqual(detail["run_id"], json.loads(before["run.json"])["run_id"])
        self.assertEqual(detail["metrics"]["curve_data"]["accuracy"], [0.8])
        self.assertEqual(self.catalog.sync_root(self.root)["unchanged"], 1)
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 1)
        self.assertEqual(self.catalog.stats()["counts"]["metric_records"], 2)
        with self.catalog.connection() as con:
            digest = con.execute("SELECT sha256 FROM artifacts WHERE name='run.log'").fetchone()[0]
        self.assertEqual(digest, hashlib.sha256(before["run.log"]).hexdigest())
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_legacy_identity_is_stable_without_fabricating_source_uuid(self):
        self.run_files(legacy=True)
        self.catalog.sync_root(self.root)
        first = self.catalog.detail(self.root, "one")
        self.catalog.sync_root(self.root, force=True)
        self.assertIsNone(first["run_id"])
        self.assertEqual(first["catalog_id"], self.catalog.detail(self.root, "one")["catalog_id"])
        self.assertFalse((self.root / "one" / "run.json").exists())

    def test_incremental_metrics_replace_instead_of_duplicate(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        with (directory / "one.csv").open("a", encoding="utf-8") as f:
            f.write("2,0.9,0.2,0.1\n")
        self.catalog.sync_root(self.root)
        self.assertEqual(self.catalog.detail(self.root, "one")["metrics"]["last_round"], 2)
        self.assertEqual(self.catalog.stats()["counts"]["round_metrics"], 2)

    def test_broken_and_failed_records_do_not_hide_healthy_run(self):
        self.run_files("healthy")
        bad = self.run_files("broken")
        (bad / "broken.csv").write_bytes(b'round,accuracy\n1,"unfinished')
        self.run_files("failed", status="failed", metrics=False)
        self.run_files("cancelled", status="cancelled", metrics=False)
        report = self.catalog.sync_root(self.root)
        self.assertEqual(report["imported"], 4)
        self.assertEqual(self.catalog.query(self.root)["total"], 4)
        self.assertIsNone(self.catalog.detail(self.root, "failed")["metrics"]["accuracy"])
        self.assertTrue(self.catalog.detail(self.root, "broken")["warnings"])

    def test_filters_pagination_and_root_isolation(self):
        self.run_files("实验A")
        self.run_files("实验B", status="failed", metrics=False)
        self.catalog.sync_root(self.root)
        self.assertEqual(self.catalog.query(self.root, search="实验", limit=1)["total"], 2)
        self.assertEqual(len(self.catalog.query(self.root, limit=1, offset=1)["items"]), 1)
        self.assertEqual(self.catalog.query(self.root, status="failed")["items"][0]["name"], "实验B")
        self.assertEqual(self.catalog.query(self.root, dataset="cifar10")["total"], 0)
        self.assertEqual(self.catalog.query(self.base / "other")["total"], 0)
        for kwargs in ({"limit": 0}, {"limit": 201}, {"offset": -1}):
            with self.assertRaises(ValueError):
                self.catalog.query(self.root, **kwargs)

    def test_search_matches_name_or_run_uuid(self):
        self.run_files("实验A")
        directory = self.run_files("实验B")
        self.catalog.sync_root(self.root)
        run_id = json.loads((directory / "run.json").read_text(encoding="utf-8"))["run_id"]
        self.assertEqual([item["name"] for item in self.catalog.query(self.root, search=run_id)["items"]],
                         ["实验B"])
        self.assertEqual(self.catalog.query(self.root, search=run_id.upper())["total"], 1)
        self.assertEqual(self.catalog.query(self.root, search=run_id[:8])["total"], 1)
        self.assertEqual([item["name"] for item in self.catalog.query(self.root, search="实验A")["items"]],
                         ["实验A"])

    def test_replacement_at_same_path_retains_distinct_run_ids(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        old = self.catalog.detail(self.root, "one")["catalog_id"]
        shutil.rmtree(directory)
        self.run_files()
        self.catalog.sync_root(self.root)
        self.assertNotEqual(old, self.catalog.detail(self.root, "one")["catalog_id"])
        self.assertEqual(self.catalog.query(self.root)["total"], 1)
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 2)

    def test_explicit_removal_cascades_without_deleting_other_run(self):
        self.run_files("one")
        self.run_files("two")
        self.catalog.sync_root(self.root)
        self.catalog.mark_removed(self.root, "one")
        self.assertIsNone(self.catalog.detail(self.root, "one"))
        self.assertEqual(self.catalog.stats()["counts"]["plans"], 1)
        with self.catalog.connection() as con:
            self.assertEqual(con.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_external_removal_keeps_history_metrics_config_and_identity(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")
        counts = self.catalog.stats()["counts"]
        shutil.rmtree(directory)
        self.assertEqual(self.catalog.sync_root(self.root)["missing"], 1)
        self.assertEqual(self.catalog.query(self.root)["total"], 1)
        after = self.catalog.detail(self.root, "one")
        for key in ("name", "run_id", "catalog_id", "status", "metrics", "config_toml", "created"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(self.catalog.stats()["counts"], counts)
        self.assertFalse(after["files_available"])
        self.assertTrue(after["attachments_missing"])
        self.assertFalse(after["can_plot"])
        self.assertEqual(after["available_files"], [])
        self.assertEqual(after["images"], [])
        self.assertEqual(after["csv_files"], [])
        self.assertIn("run.log", after["missing_files"])

    def test_missing_root_and_empty_recreated_root_keep_history_after_restart(self):
        self.run_files()
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")
        shutil.rmtree(self.root)
        for recreated in (False, True):
            with self.subTest(recreated=recreated):
                if recreated:
                    self.root.mkdir()
                reopened = Catalog(self.catalog.path)
                self.assertEqual(reopened.sync_root(self.root)["missing"], 1)
                item = reopened.query(self.root)["items"][0]
                self.assertEqual(item["name"], "one")
                self.assertEqual(item["run_id"], before["run_id"])
                self.assertEqual(item["accuracy"], 0.8)
                self.assertFalse(item["files_available"])
                self.assertEqual(reopened.detail(self.root, "one")["config_toml"], before["config_toml"])

    def test_old_absent_rows_are_queryable_without_restoring_files(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")
        shutil.rmtree(directory)
        with self.catalog.connection() as con:
            con.execute("UPDATE runs SET present=0")
        self.catalog.sync_root(self.root)
        self.assertEqual(self.catalog.query(self.root)["total"], 1)
        self.assertEqual(self.catalog.detail(self.root, "one")["run_id"], before["run_id"])

    def test_partial_file_loss_keeps_saved_data_and_only_disables_missing_files(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")
        for name in ("run.json", "config_snapshot.toml", "one.csv", "one_clients.csv"):
            (directory / name).unlink()
        self.catalog.sync_root(self.root, force=True)
        after = self.catalog.detail(self.root, "one")
        for key in ("run_id", "catalog_id", "status", "metrics", "config_toml", "created"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 1)
        self.assertEqual(self.catalog.stats()["counts"]["metric_records"], 2)
        self.assertEqual(after["available_files"], ["run.log"])
        self.assertTrue(after["files_available"])
        self.assertTrue(after["attachments_missing"])
        self.assertFalse(after["can_plot"])
        (directory / "one.csv").write_text("round,accuracy,asr,loss\n1,0.8,0.1,0.2\n", encoding="utf-8")
        self.catalog.sync_root(self.root)
        self.assertEqual(self.catalog.detail(self.root, "one")["csv_files"], ["one.csv"])

    def test_truncated_metrics_do_not_erase_already_imported_rounds(self):
        directory = self.run_files()
        with (directory / "one.csv").open("a", encoding="utf-8") as stream:
            stream.write("2,0.9,0.2,0.1\n")
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")["metrics"]
        (directory / "one.csv").write_text("round,accuracy\n1,0.8\n", encoding="utf-8")
        self.catalog.sync_root(self.root)
        self.assertEqual(self.catalog.detail(self.root, "one")["metrics"], before)
        self.assertEqual(self.catalog.stats()["counts"]["round_metrics"], 2)

    def test_failed_import_keeps_last_atomic_snapshot(self):
        directory = self.run_files()
        self.catalog.sync_root(self.root)
        before = self.catalog.detail(self.root, "one")
        with patch("backend.catalog._hash", side_effect=OSError("reading failed")):
            report = self.catalog.sync_root(self.root, force=True)
        self.assertTrue(report["warnings"])
        self.assertEqual(before, self.catalog.detail(self.root, "one"))

    def test_concurrent_imports_do_not_duplicate(self):
        self.run_files()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: self.catalog.sync_root(self.root), range(8)))
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 1)

    def test_imports_batches_without_changing_restart_state(self):
        batches = self.root / "batches"
        batches.mkdir()
        data = {"batch_id": "test", "status": "running", "experiments": []}
        path = batches / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        (batches / "bad.json").write_text("broken", encoding="utf-8")
        report = self.catalog.sync_root(self.root)
        self.assertTrue(report["warnings"])
        self.assertEqual(self.catalog.list_batches(self.root), [data])
        self.assertEqual(json.loads(path.read_text()), data)

    def test_backup_restore_round_trip_and_existing_target_protection(self):
        self.run_files()
        self.catalog.sync_root(self.root)
        backup = self.base / "backup.sqlite3"
        self.catalog.backup(backup)
        restored = self.base / "restored.sqlite3"
        Catalog.restore(backup, restored)
        self.assertEqual(Catalog(restored).detail(self.root, "one"), self.catalog.detail(self.root, "one"))
        self.assertEqual(Catalog(restored).stats()["integrity"], "ok")
        before = restored.read_bytes()
        with self.assertRaises(FileExistsError):
            Catalog.restore(backup, restored)
        self.assertEqual(restored.read_bytes(), before)

    def test_refuses_foreign_and_newer_database_without_schema_changes(self):
        for name, version in (("foreign", 0), ("future", 99)):
            path = self.base / f"{name}.sqlite3"
            with closing(sqlite3.connect(path)) as con, con:
                con.execute("CREATE TABLE original (value TEXT)")
                con.execute(f"PRAGMA user_version={version}")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                Catalog(path).initialize()
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                Catalog.restore(path, self.base / f"copy-{name}.sqlite3")


class CatalogApiTests(CatalogFixture):
    def setUp(self):
        super().setUp()
        for replacement in (patch.object(main, "get_catalog", return_value=self.catalog),
                            patch.object(main, "_results_root", return_value=self.root)):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.client = authenticated_client(self, main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_legacy_and_paginated_api_and_detail_use_imported_records(self):
        self.run_files()
        self.run_files("failed", status="failed", metrics=False)
        self.assertEqual(len(self.client.get("/api/experiments").json()), 2)
        response = self.client.get("/api/experiments", params={"paginated": True, "limit": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(len(response.json()["items"]), 1)
        detail = self.client.get("/api/experiments/one").json()
        self.assertEqual(detail["metrics"]["accuracy"], 0.8)
        self.assertTrue(detail["catalog_id"])
        response = self.client.get("/api/experiments",
                                   params={"search": detail["run_id"], "paginated": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["items"][0]["name"], "one")
        self.assertEqual(self.client.get("/api/experiments", params={"limit": 0}).status_code, 422)
        self.assertEqual(self.client.get("/api/catalog/status").json()["integrity"], "ok")

    def test_database_failure_is_503_not_empty_history_or_source_modification(self):
        directory = self.run_files()
        original = (directory / "run.json").read_bytes()
        with patch.object(self.catalog, "sync_root", side_effect=sqlite3.OperationalError("locked")):
            response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 503)
        self.assertIn("数据库", response.json()["detail"])
        self.assertEqual((directory / "run.json").read_bytes(), original)

    def test_deleted_source_is_removed_from_database_and_not_reimported(self):
        directory = self.run_files()
        self.client.get("/api/experiments")
        self.assertEqual(self.client.delete("/api/experiments/one").status_code, 200)
        self.assertFalse(directory.exists())
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 0)
        self.assertEqual(self.client.get("/api/experiments").json(), [])

    def test_history_and_detail_survive_removing_results_and_allow_explicit_deletion(self):
        directory = self.run_files()
        self.client.get("/api/experiments")
        before = self.client.get("/api/experiments/one").json()
        shutil.rmtree(self.root)
        for recreated in (False, True):
            if recreated:
                self.root.mkdir()
            response = self.client.get("/api/experiments")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()[0]["run_id"], before["run_id"])
            response = self.client.get("/api/experiments/one")
            self.assertEqual(response.status_code, 200, response.text)
            after = response.json()
            self.assertEqual(after["metrics"], before["metrics"])
            self.assertEqual(after["config_toml"], before["config_toml"])
            self.assertFalse(after["files_available"])
            self.assertFalse(directory.exists())
        for suffix in ("download", "download-all", "files/run.log"):
            self.assertEqual(self.client.get(f"/api/experiments/one/{suffix}").status_code, 404)
        self.assertEqual(self.client.delete("/api/experiments/one", params={"run_id": "wrong"}).status_code, 409)
        response = self.client.delete("/api/experiments/one", params={"run_id": before["run_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/experiments").json(), [])
        self.assertEqual(self.client.get("/api/experiments/one").status_code, 404)
        self.assertEqual(self.catalog.stats()["counts"]["round_metrics"], 0)

    def test_missing_metadata_file_uses_stored_uuid_for_explicit_deletion(self):
        directory = self.run_files()
        before = self.client.get("/api/experiments/one").json()
        (directory / "run.json").unlink()
        response = self.client.delete("/api/experiments/one", params={"run_id": before["run_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertEqual(self.catalog.stats()["counts"]["runs"], 0)


if __name__ == "__main__":
    unittest.main()