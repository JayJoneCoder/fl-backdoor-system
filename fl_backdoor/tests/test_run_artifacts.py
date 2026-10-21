"""Result protection uses temporary files and synthetic model state only."""

import io
import json
import os
import subprocess
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import tomli

from fl_backdoor.utils import CSVLogger
from fl_backdoor.utils.paths import child_path, direct_files, experiment_path, summary_path, validate_name
from fl_backdoor.utils.run_artifacts import (
    build_config_snapshot, ensure_experiment, read_run_metadata, reserve_experiment, save_final_model,
    write_run_metadata,
)


def directory_link(link, target):
    """Use a junction if Windows does not grant symlink privileges."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            raise unittest.SkipTest("Directory links are not permitted in this environment")


class RunArtifactTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "results"
        self.root.mkdir()

    def test_names_reject_paths_and_windows_aliases_on_all_platforms(self):
        for name in (None, 42, "", ".", "..", "../outside", "..\\outside", "/absolute", "C:\\outside",
                     "a/b", "a\\b", "a:b", "a\x00b", "a\nb", "a?b", "a*b", "a|b", 'a"b', "<name>",
                     " spaced", "spaced ", "trailing.", ".hidden", "--option", "CON", "nul.txt", "AUX.csv",
                     "COM1", "lpt9.txt", "COM¹.txt", "CONIN$", "a" * 121):
            with self.subTest(name=name), self.assertRaises(ValueError):
                experiment_path(self.root, name)
        for name in ("summaries", "BATCHES", "config_backups"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                experiment_path(self.root, name)

    def test_unicode_names_and_summary_namespace_remain_usable(self):
        self.assertEqual(experiment_path(self.root, "实验一 (FFT)"), self.root / "实验一 (FFT)")
        self.assertEqual(summary_path(self.root, "对照组"), self.root / "summaries" / "对照组")
        self.assertEqual(validate_name("baseline-v2.1"), "baseline-v2.1")

    def test_directory_links_cannot_escape_or_alias_other_runs(self):
        target = self.root.parent / "outside"
        target.mkdir()
        link = self.root / "linked"
        directory_link(link, target)
        self.addCleanup(lambda: link.unlink() if link.is_symlink() else link.rmdir())
        with self.assertRaises(ValueError):
            experiment_path(self.root, "linked")
        with self.assertRaises(ValueError):
            reserve_experiment(self.root, "linked")

    def test_direct_file_listing_skips_links_directories_and_hidden_helpers(self):
        (self.root / "normal.csv").write_text("normal", encoding="utf-8")
        (self.root / ".partial.tmp").write_text("temporary", encoding="utf-8")
        (self.root / "subdirectory").mkdir()
        self.assertEqual([p.name for p in direct_files(self.root)], ["normal.csv"])

    def test_existing_directories_are_never_reused_and_get_a_run_id_suffix(self):
        for name in ("empty", "legacy"):
            with self.subTest(name=name):
                directory = self.root / name
                directory.mkdir()
                if name == "legacy":
                    (directory / "old.csv").write_bytes(b"old result")
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                new_dir, metadata = reserve_experiment(self.root, name)
                self.assertNotEqual(new_dir, directory)
                self.assertTrue(new_dir.is_dir())
                self.assertEqual(metadata["name"], new_dir.name)
                self.assertEqual(metadata["requested_name"], name)
                self.assertTrue(new_dir.name.startswith(f"{name}-"))
                self.assertIn(metadata["run_id"][:8], new_dir.name)
                self.assertEqual(read_run_metadata(new_dir), metadata)
                self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, before)

    def test_concurrent_reservation_creates_distinct_runs(self):
        def reserve(_):
            return reserve_experiment(self.root, "same")[1]

        with ThreadPoolExecutor(max_workers=8) as pool:
            metadata = list(pool.map(reserve, range(8)))
        self.assertEqual(len({item["run_id"] for item in metadata}), 8)
        self.assertEqual(len({item["name"] for item in metadata}), 8)
        self.assertEqual(sum(item["name"] == "same" for item in metadata), 1)
        for item in metadata:
            self.assertEqual(read_run_metadata(self.root / item["name"])["run_id"], item["run_id"])

    def test_run_id_is_persistent_and_independent_of_name_and_flower_id(self):
        first, before = reserve_experiment(self.root, "one")
        _, other = reserve_experiment(self.root, "two")
        self.assertNotEqual(before["run_id"], other["run_id"])
        uuid.UUID(before["run_id"])
        updated = {**before, "status": "finished", "flower_run_id": "123", "exit_code": 0}
        write_run_metadata(first, updated)
        self.assertEqual(read_run_metadata(first), updated)

    def test_managed_server_must_present_matching_active_run_id(self):
        directory, metadata = reserve_experiment(self.root, "one")
        # Without a managed UUID the call reserves a new run instead of reusing.
        new_dir, new_metadata = ensure_experiment(self.root, "one")
        self.assertNotEqual(new_dir, directory)
        self.assertEqual(new_metadata["requested_name"], "one")
        self.assertEqual(read_run_metadata(directory), metadata)
        with self.assertRaises(FileExistsError):
            ensure_experiment(self.root, "one", run_id="different")
        self.assertEqual(ensure_experiment(self.root, "one", run_id=metadata["run_id"])[0], directory)
        write_run_metadata(directory, {**metadata, "status": "finished"})
        with self.assertRaises(FileExistsError):
            ensure_experiment(self.root, "one", run_id=metadata["run_id"])
        with self.assertRaises(ValueError):
            ensure_experiment(self.root, "missing", run_id=metadata["run_id"])
        self.assertFalse((self.root / "missing").exists())

    def test_corrupt_manifests_are_not_accepted_as_owned_runs(self):
        directory, metadata = reserve_experiment(self.root, "one")
        for data in ([], {**metadata, "name": "another"}, {**metadata, "run_id": "123"}):
            (directory / "run.json").write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_run_metadata(directory)

    def test_failed_metadata_replace_keeps_existing_manifest(self):
        directory, metadata = reserve_experiment(self.root, "one")
        original = (directory / "run.json").read_bytes()
        with patch("fl_backdoor.utils.run_artifacts.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                write_run_metadata(directory, {**metadata, "status": "finished"})
        self.assertEqual((directory / "run.json").read_bytes(), original)
        self.assertEqual(list(directory.glob(".run-*.tmp")), [])

    def test_csv_logger_never_truncates_existing_metric_or_diagnostic_file(self):
        for suffix in (".csv", "_metrics.csv", "_clients.csv"):
            with self.subTest(suffix=suffix):
                directory = self.root / suffix.strip(".")
                directory.mkdir()
                existing = directory / ("one" + suffix)
                existing.write_bytes(b"retained")
                with self.assertRaises(FileExistsError):
                    CSVLogger(str(directory), "one.csv")
                self.assertEqual(list(directory.iterdir()), [existing])
                self.assertEqual(existing.read_bytes(), b"retained")

    def test_csv_logger_rejects_filename_traversal_and_logs_normal_rounds(self):
        with self.assertRaises(ValueError):
            CSVLogger(str(self.root), "../outside.csv")
        with patch("sys.stdout", new=io.StringIO()):
            logger = CSVLogger(str(self.root), "one.csv")
        logger.log(1, 0.9, 0.1, 0.2)
        self.assertIn("1,0.9,0.1,0.2", logger.filepath.read_text(encoding="utf-8"))

    def test_final_model_is_per_experiment_and_cannot_be_overwritten(self):
        state = {"weight": torch.tensor([1.0, 2.0])}
        directory, _ = reserve_experiment(self.root, "one")
        path = save_final_model(state, directory)
        torch.testing.assert_close(torch.load(path, weights_only=True)["weight"], state["weight"])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_final_model({"weight": torch.zeros(2)}, directory)
        self.assertEqual(path.read_bytes(), before)

    def test_config_snapshot_records_requested_and_effective_partition_settings(self):
        document = {"tool": {"flwr": {"app": {"config": {
            "partition-mode": "dirichlet", "partition-alpha": 0.2,
            "partition-seed": 99, "seed": 42, "num-clients": 7,
        }}}}}
        partition = tomli.loads(build_config_snapshot(document).decode("utf-8"))["tool"]["flwr"]["app"]["partition"]
        self.assertEqual(partition["requested-mode"], "dirichlet")
        self.assertEqual(partition["requested-alpha"], 0.2)
        self.assertEqual(partition["requested-seed"], 99)
        self.assertEqual(partition["requested-seed-source"], "partition-seed")
        self.assertEqual(partition["effective-mode"], "dirichlet")
        self.assertEqual(partition["effective-seed"], 99)
        self.assertEqual(partition["effective-num-clients"], 7)

    def test_config_snapshot_records_global_seed_fallback(self):
        document = {"tool": {"flwr": {"app": {"config": {
            "partition-mode": "dirichlet", "partition-alpha": 0.3,
            "seed": 17, "num-clients": 4,
        }}}}}
        partition = tomli.loads(build_config_snapshot(document).decode("utf-8"))["tool"]["flwr"]["app"]["partition"]
        self.assertNotIn("requested-seed", partition)
        self.assertEqual(partition["requested-seed-source"], "global-seed")
        self.assertEqual(partition["effective-seed"], 17)

    def test_failed_model_save_removes_partial_file(self):
        directory, _ = reserve_experiment(self.root, "one")
        with patch.object(torch, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_final_model({}, directory)
        self.assertFalse((directory / "final_model.pt").exists())


class ServerModelArchiveTests(unittest.TestCase):
    def test_real_server_entrypoint_archives_model_for_cli_and_managed_runs(self):
        from fl_backdoor.server import server

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {"weight": torch.tensor([3.0])}
            for managed in (False, True):
                name = "managed" if managed else "standalone"
                metadata = reserve_experiment(root, name)[1] if managed else None
                run_config = {"run-name": name, "results-dir": str(root), "dataset": "mnist",
                              "attack": "none", "defense": "none", "detection": "none",
                              "partition-mode": "dirichlet", "partition-alpha": 0.2,
                              "partition-seed": 88, "seed": 42, "num-clients": 4}
                strategy = Mock()
                strategy.start.return_value.arrays.to_torch_state_dict.return_value = state
                pipeline = Mock()
                pipeline.apply.return_value = strategy
                with patch.dict(os.environ, {"FL_BACKDOOR_RUN_ID": metadata["run_id"] if managed else ""}), \
                     patch.object(server, "get_dataset", return_value=SimpleNamespace(meta=SimpleNamespace(input_shape=(1, 28, 28), num_classes=10))), \
                     patch.object(server, "build_attack"), patch.object(server, "build_defense_pipeline", return_value=pipeline), \
                     patch.object(server, "Net"), patch.object(server, "ArrayRecord"), patch.object(server, "FedAvg"), \
                     patch.object(server, "get_global_evaluate_fn"), patch("sys.stdout", new=io.StringIO()):
                    server.main(None, SimpleNamespace(run_config=run_config))
                output = torch.load(root / name / "final_model.pt", weights_only=True)
                torch.testing.assert_close(output["weight"], state["weight"])
                self.assertTrue((root / name / f"{name}.csv").exists())
                partition = tomli.loads((root / name / "config_snapshot.toml").read_text())["tool"]["flwr"]["app"]["partition"]
                self.assertEqual(partition["effective-seed"], 88)
                self.assertEqual(partition["effective-num-clients"], 4)
                self.assertEqual(read_run_metadata(root / name)["status"], "starting" if managed else "finished")
            self.assertFalse((root / "final_model.pt").exists())


if __name__ == "__main__":
    unittest.main()