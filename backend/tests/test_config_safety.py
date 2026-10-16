"""Configuration writes and HTTP failures never touch the real project config."""

import tempfile
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import tomli
import tomli_w
from fastapi.testclient import TestClient

from backend import config_manager as cm, main, experiment_runner


def document(**overrides):
    config = {"attack": "none", "defense": "none", "detection": "none",
              "dataset": "mnist", "batch-size": 32, "num-clients": 10, "run-name": "original"}
    config.update(overrides)
    return tomli_w.dumps({
        "project": {"name": "test-project", "version": "1.0"},
        "tool": {"flwr": {"app": {
            "components": {"serverapp": "fl_backdoor.server.server:app", "clientapp": "fl_backdoor.client.client:app"},
            "config": config,
        }}},
    })


class TemporaryConfiguration(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "pyproject.toml"
        self.backup = self.root / "pyproject.toml.bak"
        for name, value in (("PROJECT_ROOT", self.root), ("TOML_PATH", self.path), ("BACKUP_PATH", self.backup)):
            replacement = patch.object(cm, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.original = document().encode("utf-8")
        self.path.write_bytes(self.original)
        self.backup.write_bytes(b"previous backup bytes")
        self.original_backup = self.backup.read_bytes()

    def assert_unchanged(self):
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), self.original_backup)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def section(self):
        return tomli.loads(self.path.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]


class ConfigurationSafetyTests(TemporaryConfiguration):
    def test_invalid_result_paths_are_rejected_before_save_or_backup(self):
        paths = ["", " ", " results ", "bad\x00path", "bad\npath"]
        import os
        if os.name == "nt":
            paths += ["C:relative", "\\rooted", "\\\\server\\share", "bad?name", "NUL", "bad."]
        for value in paths:
            with self.subTest(value=value):
                for save in (
                    lambda: cm.update_config({"results-dir": value}, backup_current=True),
                    lambda: cm.replace_config(document(**{"results-dir": value}), backup_current=True),
                ):
                    with self.assertRaises(ValueError):
                        save()
                    self.assert_unchanged()

    def test_result_path_below_a_file_is_rejected(self):
        parent = self.root / "not-a-directory"
        parent.write_bytes(b"retain")
        with self.assertRaises(ValueError):
            cm.update_config({"results-dir": str(parent / "results")}, backup_current=True)
        self.assert_unchanged()

    def test_new_result_root_is_resolved_without_creating_it(self):
        cm.update_config({"results-dir": "new-results/nested"})
        expected = self.root / "new-results" / "nested"
        self.assertEqual(cm.get_results_dir(), expected)
        self.assertFalse(expected.exists())

    def test_valid_update_can_repair_invalid_legacy_value(self):
        self.path.write_text(document(**{"batch-size": -1}), encoding="utf-8")
        cm.update_config({"batch-size": 32})
        self.assertEqual(self.section()["batch-size"], 32)

    def test_valid_form_conversions_and_false_string(self):
        cm.update_config({"batch-size": "64", "learning-rate": "0.02", "detection-enable-filter": "false"})
        self.assertEqual(self.section()["batch-size"], 64)
        self.assertEqual(self.section()["learning-rate"], 0.02)
        self.assertIs(self.section()["detection-enable-filter"], False)

    def test_optional_null_fields_are_removed(self):
        cm.update_config({"dba-sub-pattern-size": 2, "fcba-sub-block-size": 2, "defense-trim-ratio": 0.1})
        cm.update_config({"dba-sub-pattern-size": None, "fcba-sub-block-size": "", "defense-trim-ratio": None})
        for key in ("dba-sub-pattern-size", "fcba-sub-block-size", "defense-trim-ratio"):
            self.assertNotIn(key, self.section())

    def test_required_empty_is_rejected_without_backup_changes(self):
        for key in ("batch-size", "learning-rate", "dataset", "detection-enable-filter"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cm.update_config({key: None}, backup_current=True)
            self.assert_unchanged()

    def test_invalid_values_leave_active_and_backup_unchanged(self):
        for updates in (
            {"batch-size": -1}, {"batch-size": 0}, {"batch-size": 1.5}, {"batch-size": True},
            {"batch-size": "1.5"}, {"learning-rate": 0}, {"learning-rate": float("nan")},
            {"learning-rate": float("inf")}, {"detection-enable-filter": "not false"},
            {"detection-enable-filter": 0}, {"poison-rate": 1.1}, {"seed": -1},
            {"detection-percentile": 101}, {"detection-cosine-floor": -2},
            {"defense-trim-ratio": 0.5}, {"defense": "not_a_defense"}, {"run-name": ""},
            {"partition-mode": "shards"}, {"partition-alpha": 0}, {"partition-seed": -1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                cm.update_config(updates, backup_current=True)
            self.assert_unchanged()

    def test_current_project_document_and_form_can_roundtrip_in_temp_directory(self):
        source = Path(__file__).resolve().parents[2] / "pyproject.toml"
        content = source.read_text(encoding="utf-8")
        cm.replace_config(content)
        values = cm.get_ui_config()
        cm.update_config(values, backup_current=True)
        self.assertEqual(self.backup.read_bytes(), content.encode("utf-8"))
        self.assertEqual(self.section()['batch-size'], values['batch-size'])
        self.assertEqual(self.section()['attack'], values['attack'])

    def test_cross_constraints_leave_files_unchanged(self):
        for updates in (
            {"target-label": 10}, {"dataset": "cifar100"},
            {"attack": "dba", "dba-global-trigger-location": "[28,28]"},
            {"attack": "dba", "dba-num-sub-patterns": 3},
            {"attack": "fcba", "fcba-sub-block-size": 10},
            {"defense": "krum", "num-clients": 7, "defense-num-malicious": 3},
            {"defense": "trimmed_mean", "num-clients": 4, "defense-trim-k": 2},
            {"attack": "badnets", "attack-malicious-mode": "fixed", "attack-fixed-clients": "10"},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                cm.update_config(updates, backup_current=True)
            self.assert_unchanged()

    def test_all_schema_defaults_can_be_saved_with_optional_nulls(self):
        values = {key: field["default"] for key, field in cm.CONFIG_SCHEMA.items() if not field.get("ui_only")}
        cm.update_config(values, backup_current=True)
        self.assertEqual(self.section()["batch-size"], 32)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertNotIn("defense-trim-ratio", self.section())

    def test_ui_client_count_uses_current_total_not_hardcoded_ten(self):
        cm.update_config({"num-clients": 20})
        cm.update_config({"malicious-count": 3})
        self.assertEqual(self.section()["malicious-ratio"], 0.15)
        self.assertNotIn("malicious-count", self.section())
        with self.assertRaises(ValueError):
            cm.update_config({"malicious-count": 21})

    def test_serialization_failure_does_not_create_partial_file(self):
        with patch.object(cm.tomli_w, "dumps", side_effect=TypeError("serialization failed")):
            with self.assertRaises(TypeError):
                cm.update_config({"batch-size": 64}, backup_current=True)
        self.assert_unchanged()

    def test_low_level_none_serialization_preserves_file(self):
        with self.assertRaises(TypeError):
            cm._write_toml({"invalid": None})
        self.assert_unchanged()

    def test_fsync_failure_keeps_old_file_and_cleans_temp(self):
        with patch.object(cm.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                cm.update_config({"batch-size": 64})
        self.assert_unchanged()

    def test_atomic_replace_failure_keeps_old_file_and_cleans_temp(self):
        with patch.object(cm.os, "replace", side_effect=PermissionError("locked destination")):
            with self.assertRaises(PermissionError):
                cm.update_config({"batch-size": 64})
        self.assert_unchanged()

    def test_named_backup_and_bak_contain_previous_bytes(self):
        cm.update_config({"batch-size": 64}, backup_current=True, backup_name="before-save")
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual((self.root / "config_backups" / "before-save.toml").read_bytes(), self.original)
        self.assertEqual(self.section()["batch-size"], 64)

    def test_backup_failure_aborts_active_save(self):
        with patch.object(cm, "backup_config", side_effect=OSError("backup write failed")):
            with self.assertRaises(OSError):
                cm.update_config({"batch-size": 64}, backup_current=True)
        self.assert_unchanged()

    def test_backup_conflict_aborts_active_save(self):
        (self.root / "existing.toml").write_text("retain", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            cm.update_config({"batch-size": 64}, backup_current=True, backup_name="existing")
        self.assert_unchanged()
        self.assertEqual((self.root / "existing.toml").read_text(encoding="utf-8"), "retain")

    def test_raw_invalid_syntax_or_types_never_replace_files(self):
        invalid = ["invalid [", "", "[project]\nname='incomplete'", document(**{"batch-size": "64"}),
                   document(**{"detection-enable-filter": "false"}), document(**{"batch-size": -1})]
        for content in invalid:
            with self.subTest(content=content[:50]), self.assertRaises(ValueError):
                cm.replace_config(content, backup_current=True)
            self.assert_unchanged()

    def test_raw_success_preserves_comments_and_unicode(self):
        content = "# 测试配置\r\n" + document(**{"run-name": "中文实验", "batch-size": 64}).replace("\n", "\r\n")
        cm.replace_config("\ufeff" + content, backup_current=True)
        self.assertEqual(self.path.read_bytes(), content.replace("\r\n", "\n").encode("utf-8"))
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual(self.section()["run-name"], "中文实验")

    def test_concurrent_partial_updates_do_not_lose_each_other(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(cm.update_config, ({"batch-size": 64}, {"num-server-rounds": 7})))
        self.assertEqual(results, [None, None])
        self.assertEqual(self.section()["batch-size"], 64)
        self.assertEqual(self.section()["num-server-rounds"], 7)

    def test_bad_restore_keeps_both_files(self):
        path = self.root / "bad.toml"
        path.write_text("invalid [", encoding="utf-8")
        with self.assertRaises(ValueError):
            cm.restore_from_backup("bad.toml")
        self.assert_unchanged()

    def test_restore_from_legacy_bak_reads_before_overwriting_source(self):
        restored = document(**{"batch-size": 8}).encode("utf-8")
        self.backup.write_bytes(restored)
        cm.restore_from_backup("pyproject.toml.bak", backup_current=True, backup_name="before-restore")
        self.assertEqual(self.path.read_bytes(), restored)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual((self.root / "config_backups" / "before-restore.toml").read_bytes(), self.original)

    def test_invalid_draft_does_not_overwrite_existing_backup(self):
        backup = self.root / "draft.toml"
        backup.write_bytes(self.original)
        with self.assertRaises(ValueError):
            cm.create_backup_from_content("invalid [", "draft", "overwrite")
        self.assertEqual(backup.read_bytes(), self.original)
        self.assert_unchanged()

    def test_reserved_backup_name_cannot_overwrite_active_config(self):
        for name in ("pyproject", "PYPROJECT.TOML", "PyProject.toml"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cm.create_backup_from_content(document(**{"batch-size": 8}), name, "overwrite")
            self.assert_unchanged()

    def test_raw_optional_numbers_must_be_omitted_not_empty_text(self):
        for key in ("dba-sub-pattern-size", "fcba-sub-block-size", "defense-trim-ratio", "frequency-window-size"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cm.replace_config(document(**{key: ""}), backup_current=True)
            self.assert_unchanged()

    def test_runtime_defaults_are_not_replaced_by_ui_algorithm_defaults(self):
        content = document()
        data = tomli.loads(content)
        section = data['tool']['flwr']['app']['config']
        for key in ('attack', 'defense', 'detection'):
            section.pop(key)
        section['num-clients'] = 2
        cm.replace_config(tomli_w.dumps(data))
        self.assertEqual(self.section()['num-clients'], 2)
        self.assertNotIn('defense', self.section())

    def test_optional_trigger_location_null_is_auto_position(self):
        cm.update_config({'dba-global-trigger-location': None, 'fcba-global-trigger-location': None})
        self.assertEqual(self.section()['dba-global-trigger-location'], '')
        self.assertEqual(self.section()['fcba-global-trigger-location'], '')

    def test_frequency_window_none_removes_key(self):
        cm.update_config({"frequency-window-size": 6})
        cm.update_config({"frequency-window-size": None})
        self.assertNotIn("frequency-window-size", self.section())

    def test_noninteger_fixed_client_ids_do_not_silently_truncate(self):
        for value in ("1.5,2", [1.5, 2], [True, 2], "-1,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cm.update_config({"attack-fixed-clients": value}, backup_current=True)
            self.assert_unchanged()

    def test_invalid_backup_policy_does_not_overwrite_existing_file(self):
        path = self.root / "existing.toml"
        path.write_bytes(self.original)
        with self.assertRaises(ValueError):
            cm.create_backup("existing", "not_a_policy")
        self.assertEqual(path.read_bytes(), self.original)
        self.assert_unchanged()


class ConfigurationApiTests(TemporaryConfiguration):
    def setUp(self):
        super().setUp()
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_invalid_result_directory_returns_400_on_form_raw_and_upload(self):
        paths = [" results ", "bad\x00path", "bad\npath"]
        import os
        if os.name == "nt":
            paths.extend(["C:relative", "\\rooted", "\\\\server\\share", "NUL"])
        for value in paths:
            with self.subTest(value=value):
                content = document(**{"results-dir": value})
                responses = [
                    self.client.post("/api/config", json={"results-dir": value}),
                    self.client.post("/api/config/raw", json={"content": content, "backup_current": True}),
                    self.client.post("/api/config/upload", files={"file": ("config.toml", content.encode())}),
                ]
                for response in responses:
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("结果目录", response.json()["detail"])
                self.assert_unchanged()

    def test_json_invalid_config_returns_400_without_writes(self):
        response = self.client.post("/api/config", json={"batch-size": -1})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("batch-size", response.json()["detail"])
        self.assert_unchanged()

    def test_json_optional_null_and_false_are_safely_saved(self):
        response = self.client.post("/api/config", json={"dba-sub-pattern-size": None, "detection-enable-filter": "false"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("dba-sub-pattern-size", self.section())
        self.assertIs(self.section()["detection-enable-filter"], False)
        self.assertEqual(self.backup.read_bytes(), self.original)

    def test_json_named_backup_in_same_save_operation(self):
        response = self.client.post("/api/config?backup_name=before", json={"batch-size": 64})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((self.root / "config_backups" / "before.toml").read_bytes(), self.original)

    def test_raw_invalid_document_returns_400_not_success(self):
        response = self.client.post("/api/config/raw", json={"content": "bad [", "backup_current": True})
        self.assertEqual(response.status_code, 400, response.text)
        self.assert_unchanged()

    def test_raw_valid_document_backups_old_configuration(self):
        response = self.client.post("/api/config/raw", json={
            "content": document(**{"batch-size": 64}), "backup_current": True, "backup_name": "before",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((self.root / "config_backups" / "before.toml").read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual(self.section()["batch-size"], 64)

    def test_raw_backup_false_string_is_not_truthy(self):
        response = self.client.post("/api/config/raw", json={"content": document(**{"batch-size": 64}), "backup_current": "false"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.backup.read_bytes(), self.original_backup)

    def test_raw_non_string_content_is_rejected(self):
        response = self.client.post("/api/config/raw", json={"content": {"batch-size": 64}})
        self.assertEqual(response.status_code, 400, response.text)
        self.assert_unchanged()

    def test_upload_invalid_or_non_utf8_keeps_old_config_and_backup(self):
        for content in (b"bad [", b"\xff\xfe", document(**{"batch-size": -1}).encode()):
            with self.subTest(content=content[:30]):
                response = self.client.post("/api/config/upload", files={"file": ("config.toml", content, "application/octet-stream")})
                self.assertEqual(response.status_code, 400, response.text)
                self.assert_unchanged()

    def test_upload_valid_document_is_installed_and_old_bytes_backed_up(self):
        response = self.client.post("/api/config/upload", files={"file": ("config.TOML", document(**{"batch-size": 8}).encode(), "application/octet-stream")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.section()["batch-size"], 8)
        self.assertEqual(self.backup.read_bytes(), self.original)

    def test_named_backup_conflict_returns_409_and_preserves_configuration(self):
        (self.root / "existing.toml").write_bytes(self.original)
        response = self.client.post("/api/config?backup_name=existing", json={"batch-size": 64})
        self.assertEqual(response.status_code, 409, response.text)
        self.assert_unchanged()

    def test_raw_replace_failure_returns_error_without_partial_configuration(self):
        with patch.object(cm.os, "replace", side_effect=OSError("disk error")):
            response = self.client.post("/api/config/raw", json={"content": document(**{"batch-size": 64})})
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_unchanged()

    def test_backup_editor_rejects_invalid_content_and_updates_valid_content(self):
        path = self.root / "saved.toml"
        path.write_bytes(self.original)
        response = self.client.put("/api/config/backups/saved.toml", json={"content": "bad ["})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(path.read_bytes(), self.original)
        response = self.client.put("/api/config/backups/saved.toml", json={"content": document(**{"batch-size": 8})})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('batch-size = 8', path.read_text(encoding="utf-8"))
        self.assert_unchanged()

    def test_invalid_backup_restore_is_rejected_without_modifying_active(self):
        (self.root / "saved.toml").write_text("bad [", encoding="utf-8")
        response = self.client.post("/api/config/restore", json={"filename": "saved.toml", "backup_current": True})
        self.assertEqual(response.status_code, 400, response.text)
        self.assert_unchanged()

    def test_draft_invalid_document_returns_400_and_does_not_create_file(self):
        response = self.client.post("/api/config/backups/content", json={"name": "draft", "content": "bad ["})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse((self.root / "draft.toml").exists())
        self.assert_unchanged()

    def test_json_disk_failure_is_500_not_input_validation_error(self):
        with patch.object(cm.os, "replace", side_effect=OSError("disk error")):
            response = self.client.post("/api/config", json={"batch-size": 64})
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_unchanged()

    def test_start_accepts_omitted_or_null_overrides(self):
        from unittest.mock import AsyncMock
        for request in ({"name": "requested"}, {"name": "requested", "config": None}):
            with self.subTest(request=request), patch.object(main.runner, "start", new_callable=AsyncMock) as start:
                response = self.client.post("/api/experiment/start", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                start.assert_awaited_once_with("requested", {"run-name": "requested"}, backup_config=True)
        self.assert_unchanged()

    def test_start_rejects_invalid_override_type(self):
        from unittest.mock import AsyncMock
        with patch.object(main.runner, "start", new_callable=AsyncMock) as start:
            response = self.client.post("/api/experiment/start", json={"name": "requested", "config": []})
            self.assertEqual(response.status_code, 400, response.text)
            start.assert_not_awaited()
        self.assert_unchanged()

    def test_start_invalid_configuration_never_launches_process_or_modifies_backup(self):
        runner = experiment_runner.ExperimentRunner()
        with patch.object(main, "runner", runner), \
             patch.object(experiment_runner, "RESULTS_DIR", self.root / "results"), \
             patch.object(experiment_runner.subprocess, "Popen") as spawn:
            response = self.client.post("/api/experiment/start", json={"name": "requested", "config": {"batch-size": -1}})
            self.assertEqual(response.status_code, 400, response.text)
            spawn.assert_not_called()
        self.assert_unchanged()


class RunnerConfigSafetyTests(TemporaryConfiguration, unittest.IsolatedAsyncioTestCase):
    async def test_valid_runner_start_backups_previous_config_without_mutating_caller(self):
        from unittest.mock import Mock
        process = Mock(returncode=0)
        process.poll.return_value = 0
        process.stdout = io.StringIO("")
        runner = experiment_runner.ExperimentRunner()
        caller = {"run-name": "caller-name", "batch-size": 64}
        owner = Mock()
        owner.spawn.return_value = process
        with patch.object(experiment_runner, "RESULTS_DIR", self.root / "results"), \
             patch.object(experiment_runner, "OwnedProcessTree", return_value=owner):
            await runner.start("requested", caller)
            self.assertEqual(self.backup.read_bytes(), self.original)
            self.assertEqual(self.section()["run-name"], "requested")
            self.assertEqual(self.section()["batch-size"], 64)
            await runner.wait()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(caller["run-name"], "caller-name")


if __name__ == "__main__":
    unittest.main()