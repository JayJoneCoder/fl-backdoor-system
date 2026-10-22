"""Bootstrap tests never download wheels or modify the developer interpreter."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import time
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import Mock, patch

from desktop import runtime
from desktop.setup_server import SetupState
from fl_backdoor.utils import app_paths, device
from backend.experiment_worker import simulation_overrides


class AppPathsTests(unittest.TestCase):
    def test_development_defaults_are_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(app_paths.user_root(), app_paths.SOURCE_ROOT)
            self.assertEqual(app_paths.data_root(), app_paths.SOURCE_ROOT / "data")

    def test_desktop_resources_and_state_are_separate_and_config_is_seeded_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources, user = root / "installed", root / "user"
            resources.mkdir()
            (resources / "pyproject.toml").write_bytes(b"original config")
            with patch.dict(os.environ, {"FL_BACKDOOR_DESKTOP": "1", "FL_BACKDOOR_RESOURCE_DIR": str(resources),
                                         "FL_BACKDOOR_USER_DIR": str(user)}, clear=True):
                app_paths.initialize_desktop()
                self.assertEqual(app_paths.data_root(), resources / "data")
                self.assertEqual(app_paths.results_root(), user / "results")
                self.assertEqual(Path(os.environ["FL_IDENTITY_DATABASE"]), user / "platform_state" / "identity.sqlite3")
                (user / "pyproject.toml").write_bytes(b"private edits")
                app_paths.initialize_desktop()
                self.assertEqual((user / "pyproject.toml").read_bytes(), b"private edits")
                self.assertEqual((resources / "pyproject.toml").read_bytes(), b"original config")
                self.assertFalse((user / "platform_state" / "identity.sqlite3").exists())

    def test_explicit_paths_are_respected(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"FL_BACKDOOR_DATA_DIR": temp}):
            self.assertEqual(app_paths.data_root(), Path(temp).resolve())


class DeviceProbeTests(unittest.TestCase):
    def fake_torch(self, *, cuda_build="12.8", available=True):
        cuda = SimpleNamespace(is_available=lambda: available, get_arch_list=lambda: ["sm_120"],
                               get_device_properties=lambda index: SimpleNamespace(name="GPU", major=12, minor=0,
                                                                                     total_memory=8 * 1024**3))
        return SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda=cuda_build),
                               cuda=cuda, set_num_threads=Mock())

    def run_probe(self, torch, failure=None):
        with patch.dict(sys.modules, {"torch": torch, "torchvision": SimpleNamespace(__version__="0.23.0")}), \
                patch.object(device, "nvidia_status", return_value={"detected": True, "devices": []}), \
                patch.object(device, "compute_probe", side_effect=failure) as compute:
            return device.probe_environment(), compute

    def test_cpu_only_build_never_claims_gpu_even_with_driver(self):
        report, compute = self.run_probe(self.fake_torch(cuda_build=None))
        self.assertTrue(report["cpu_available"])
        self.assertFalse(report["gpu_available"])
        compute.assert_called_once()
        self.assertIn("CPU", report["warnings"][0])

    def test_cuda_build_without_driver_or_visibility_is_not_gpu_ready(self):
        report, _ = self.run_probe(self.fake_torch(available=False))
        self.assertTrue(report["cpu_available"])
        self.assertFalse(report["gpu_available"])
        self.assertTrue(report["guidance"])

    def test_architecture_or_cuda_kernel_failure_is_reported_without_losing_cpu(self):
        report, _ = self.run_probe(self.fake_torch(), [None, RuntimeError("no kernel image")])
        self.assertTrue(report["cpu_available"])
        self.assertFalse(report["gpu_available"])
        self.assertIn("no kernel image", report["warnings"][0])
        self.assertEqual(report["devices"][0]["capability"], "12.0")

    def test_success_requires_cpu_and_gpu_actual_operations(self):
        report, compute = self.run_probe(self.fake_torch())
        self.assertTrue(report["gpu_available"])
        self.assertEqual([c.args[1] for c in compute.call_args_list], ["cpu", "cuda:0"])

    def test_broken_or_missing_install_is_unavailable(self):
        report, _ = self.run_probe(self.fake_torch(), RuntimeError("missing DLL"))
        self.assertFalse(report["cpu_available"])
        self.assertFalse(report["gpu_available"])

    def test_forced_cuda_blocks_and_auto_decides_before_training(self):
        report = {"cpu_available": True, "gpu_available": False, "warnings": ["CPU wheel"], "guidance": ["换 CUDA 版"]}
        self.assertEqual(device.select_device("auto", report), "cpu")
        with self.assertRaisesRegex(RuntimeError, "强制 GPU"):
            device.select_device("cuda", report)
        report["gpu_available"] = True
        self.assertEqual(device.select_device("auto", report), "cuda")
        self.assertEqual(device.select_device("cpu", report), "cpu")
        with self.assertRaises(ValueError):
            device.select_device("bogus", report)

    def test_training_process_cannot_silently_fallback_from_cuda(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch(available=False)}), \
                patch.dict(os.environ, {"FL_BACKDOOR_DEVICE": "cuda"}):
            with self.assertRaisesRegex(RuntimeError, "禁止静默回退"):
                device.training_device("test")


class RuntimeInstallationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {"FL_BACKDOOR_USER_DIR": str(self.root)}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)

    def test_install_command_only_uses_local_wheels(self):
        wheels = [self.root / runtime.wheel_filename("torch", "2.8.0", "cpu"),
                  self.root / runtime.wheel_filename("torchvision", "0.23.0", "cpu")]
        command = runtime.install_command(self.root / "torch", wheels)
        self.assertIn("--isolated", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--no-index", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertNotIn("--trusted-host", command)
        self.assertNotIn("--index-url", command)
        for wheel in wheels:
            self.assertIn(str(wheel), command)
        with self.assertRaises(ValueError):
            runtime.install_command(self.root / "torch", [])

    def test_all_sources_use_the_flat_wheel_layout(self):
        for source in runtime.SOURCES:
            url = runtime.wheel_url(source, "cpu", "torch", "2.8.0")
            self.assertTrue(url.endswith("/cpu/torch-2.8.0+cpu-cp311-cp311-win_amd64.whl"))

    def test_runtime_environment_never_writes_bytecode_into_install_dir(self):
        env = runtime.runtime_environment()
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[-1], str(app_paths.resource_root()))

    def test_active_record_cannot_escape_private_runtime_root(self):
        state = self.root / "platform_state"
        state.mkdir()
        for name in ("../escape", str(self.root), "torch-../outside"):
            (state / "runtime.json").write_text(json.dumps({"directory": name}), encoding="utf-8")
            with self.assertRaises(ValueError):
                runtime.active_runtime()

    def fake_owner(self, returncode=0):
        process = Mock(stdout=io.StringIO("install progress\n"))
        process.wait.return_value = returncode
        owner = Mock()
        owner.spawn.return_value = process
        return owner

    @staticmethod
    def fake_download(url, destination, expected, progress, timeout=60):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wheel")
        progress(5)
        return 5

    @staticmethod
    def successful_probe(target):
        for name in runtime.VERSIONS:
            (target / name).mkdir(parents=True)
            (target / name / "__init__.py").touch()
        return {"cpu_available": True, "gpu_available": False, "torch": "2.8.0+cpu", "torchvision": "0.23.0+cpu"}

    def install_patches(self, owner, probe=None):
        return (
            patch.object(runtime, "check_source", return_value={name: 5 for name in runtime.VERSIONS}),
            patch.object(runtime, "download_wheel", side_effect=self.fake_download),
            patch("backend.owned_processes.OwnedProcessTree", return_value=owner),
            patch.object(runtime, "probe", side_effect=probe or self.successful_probe),
            patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)),
        )

    def test_failed_download_preserves_previous_activation(self):
        state = self.root / "platform_state"
        state.mkdir()
        marker = state / "runtime.json"
        marker.write_bytes(b"previous runtime")
        owner = self.fake_owner(1)
        with patch.object(runtime, "check_source", return_value={name: 5 for name in runtime.VERSIONS}), \
                patch.object(runtime, "download_wheel", side_effect=self.fake_download), \
                patch("backend.owned_processes.OwnedProcessTree", return_value=owner), \
                patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, "安装失败"):
                runtime.install_runtime("official", "cpu", lambda line: None)
        self.assertEqual(marker.read_bytes(), b"previous runtime")
        owner.cleanup.assert_called_once()

    def test_success_is_activated_only_after_exact_version_and_compute_validation(self):
        owner = self.fake_owner()
        check, download, tree, probe, disk = self.install_patches(owner)
        with check, download, tree, probe, disk:
            runtime.install_runtime("official", "cpu", lambda line: None)
        self.assertTrue(runtime.active_runtime().is_dir())
        arguments = owner.spawn.call_args.args[1]
        self.assertEqual(arguments[0], "--isolated")
        self.assertIn("--no-deps", arguments)
        self.assertIn("--no-index", arguments)
        self.assertTrue(any("cache" in argument and "wheels" in argument for argument in arguments))

    def test_install_reports_download_progress_and_done_phase(self):
        events = []
        owner = self.fake_owner()
        check, download, tree, probe, disk = self.install_patches(owner)
        with check, download, tree, probe, disk:
            runtime.install_runtime("official", "cpu", lambda line: None,
                                    progress=lambda phase, done, total: events.append((phase, done, total)))
        phases = [event[0] for event in events]
        self.assertIn("download", phases)
        self.assertIn("install", phases)
        self.assertEqual(phases[-1], "done")
        last_download = [event for event in events if event[0] == "download"][-1]
        self.assertEqual(last_download[1], 10)
        self.assertEqual(last_download[2], 10)

    def test_cuda_wheel_that_cannot_compute_is_not_activated(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA 安装未通过"):
            target = self.root / "stage"
            for name in runtime.VERSIONS:
                (target / name).mkdir(parents=True)
                (target / name / "__init__.py").touch()
            runtime.validate_install({"cpu_available": True, "gpu_available": False, "torch": "2.8.0+cu128",
                                      "torchvision": "0.23.0+cu128", "warnings": ["no driver"]}, target, "cu128")

    def test_source_probe_checks_exact_windows_wheels_and_reads_sizes(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 0-0/5"}

        responses = [Response(b"\x00") for _ in runtime.VERSIONS]
        with patch.object(runtime.urllib.request, "urlopen", side_effect=responses) as request:
            sizes = runtime.check_source("official", "cpu")
        self.assertEqual(request.call_count, 2)
        requested = [call.args[0].full_url for call in request.call_args_list]
        for name, version in runtime.VERSIONS.items():
            self.assertTrue(any(f"{name}-{version}+cpu-cp311-cp311-win_amd64.whl" in url for url in requested))
            self.assertEqual(sizes[name], 5)
        missing = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None)
        with patch.object(runtime.urllib.request, "urlopen", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "未提供"):
                runtime.check_source("sjtu", "cpu")

    def test_speed_test_ranks_sources_and_reports_failures(self):
        class Response(io.BytesIO):
            pass

        def fake_urlopen(request, timeout=0):
            if "download.pytorch.org" in request.full_url:
                raise OSError("blocked")
            return Response(b"x" * 1024)

        with patch.object(runtime.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = runtime.speed_test("cpu")
        self.assertTrue(result["fastest"])
        self.assertTrue(result["results"][result["fastest"]]["ok"])
        self.assertFalse(result["results"]["official"]["ok"])
        self.assertIn("blocked", result["results"]["official"]["error"])
        self.assertIn("latency_ms", result["results"][result["fastest"]])

    def test_runtime_environment_never_writes_bytecode_into_install_dir(self):
        env = runtime.runtime_environment()
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[-1], str(app_paths.resource_root()))


class RayRuntimeTests(unittest.TestCase):
    def test_ray_environment_uses_ascii_temp_and_relaxed_health_checks(self):
        from fl_backdoor.utils import ray_runtime

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"PUBLIC": temp}, clear=False):
            env = {"TEMP": "C:\\Users\\张三\\AppData\\Local\\Temp",
                   "TMP": "C:\\Users\\张三\\AppData\\Local\\Temp"}
            path = ray_runtime.configure_ray_environment(env)
        self.assertTrue(str(path).isascii())
        self.assertEqual(env["TEMP"], str(path))
        self.assertEqual(env["RAY_TEMP_DIR"], str(path))
        self.assertEqual(env["RAY_health_check_timeout_ms"], "120000")
        self.assertIn("LianAnZhiDun", str(path))

    def test_probe_report_round_trips_through_the_user_directory(self):
        from fl_backdoor.utils import ray_runtime

        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"FL_BACKDOOR_USER_DIR": temp}, clear=False):
            ray_runtime._write_probe_report({"ok": False, "checked_at": time.time(),
                                             "error": "timeout", "ray": "2.51.1"})
            report = ray_runtime.read_probe_report()
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"], "timeout")


class SetupAndResourceTests(unittest.TestCase):
    def test_regenerated_flower_entry_point_executes_with_current_interpreter(self):
        from fl_backdoor.utils.flower_runtime import create_entry_points

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            create_entry_points(directory)
            entry = directory / ("flower-superexec.exe" if os.name == "nt" else "flower-superexec")
            self.assertTrue(entry.is_file())
            result = subprocess.run([str(entry), "--help"], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    def test_generated_simulation_entry_registers_the_local_backend(self):
        from fl_backdoor.utils.flower_runtime import create_entry_points

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            create_entry_points(directory)
            entry = directory / ("flwr-simulation.exe" if os.name == "nt" else "flwr-simulation")
            self.assertTrue(entry.is_file())
            env = os.environ.copy()
            env["FL_BACKDOOR_SIM_BACKEND"] = "local"
            result = subprocess.run([str(entry), "--help"], capture_output=True, timeout=90, env=env,
                                    text=True, encoding="utf-8", errors="replace")
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("registered backend 'local'", output)
            self.assertIn("usage", output.lower())

    def test_setup_http_requires_token_and_rejects_foreign_origin(self):
        from desktop.setup_server import setup_window

        def browser(url, **kwargs):
            parsed = urllib.parse.urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            token = urllib.parse.parse_qs(parsed.query)["token"][0]

            def request(path, data=None, **headers):
                return urllib.request.urlopen(urllib.request.Request(base + path,
                    data=json.dumps(data).encode() if data is not None else None,
                    headers={"X-Setup-Token": token, **headers}), timeout=10)

            try:
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(base + "/status", timeout=10)
                self.assertEqual(denied.exception.code, 403)
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    request("/install", {"source": "official", "build": "cpu"}, Origin="https://untrusted.invalid")
                self.assertEqual(denied.exception.code, 403)
                with request("/status") as response:
                    self.assertIn("official", json.load(response)["sources"])
                with urllib.request.urlopen(base + "/favicon.svg", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b"<svg", response.read())
                with request("/continue", {"mode": "cpu", "acknowledge": True}) as response:
                    self.assertEqual(response.status, 200)
            finally:
                # On assertion failure cancel as well, so no test server lingers.
                try:
                    request("/cancel", {}).close()
                except urllib.error.URLError:
                    pass

        with patch("desktop.setup_server.webbrowser.open", side_effect=browser):
            mode = setup_window({"cpu_available": True, "gpu_available": False, "warnings": [], "guidance": []})
        self.assertEqual(mode, "cpu")

    def test_live_owned_api_binds_port_serves_frontend_and_cleans_up(self):
        from backend.owned_processes import OwnedProcessTree

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"FL_BACKDOOR_DESKTOP": "1", "FL_BACKDOOR_USER_DIR": str(root)}, clear=False):
                app_paths.initialize_desktop()
                env = runtime.runtime_environment()
                env["FL_BACKDOOR_STARTUP_ID"] = "desktop-integration-test"
            owner = OwnedProcessTree()
            ready = root / "ready.json"
            with (root / "server.log").open("w", encoding="utf-8") as logfile:
                try:
                    process = owner.spawn(app_paths.resource_root() / "desktop" / "api_entry.py", ["--ready-file", str(ready)],
                                          cwd=str(app_paths.resource_root()), env=env, stdout=logfile,
                                          stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), (root / "server.log").read_text(encoding="utf-8"))
                        try:
                            port = json.loads(ready.read_text(encoding="utf-8"))["port"]
                            url = f"http://127.0.0.1:{port}"
                            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                                self.assertEqual(json.load(response)["startup_id"], "desktop-integration-test")
                            break
                        except (OSError, ValueError):
                            time.sleep(0.1)
                    else:
                        self.fail("API health check timed out")
                    self.assertTrue((root / "platform_state" / "identity.sqlite3").is_file())
                    with self.assertRaises(urllib.error.HTTPError) as denied:
                        urllib.request.urlopen(url + "/api/runtime", timeout=5)
                    self.assertEqual(denied.exception.code, 401)
                    if (app_paths.resource_root() / "frontend" / "dist" / "index.html").is_file():
                        with urllib.request.urlopen(url + "/", timeout=5) as response:
                            self.assertIn(b"<html", response.read())
                finally:
                    owner.cleanup()
                self.assertIsNotNone(process.poll())

    def test_offline_loaders_never_request_downloads(self):
        from fl_backdoor.task import load_data, load_centralized_dataset

        dataset = Mock()
        with patch("fl_backdoor.task.get_dataset", return_value=dataset), \
                patch.dict(os.environ, {"FL_BACKDOOR_OFFLINE": "1"}):
            load_data(0, 1, 8, "mnist")
            load_centralized_dataset("mnist")
        self.assertFalse(dataset.load_partition.call_args.kwargs["download"])
        self.assertFalse(dataset.load_centralized_test.call_args.kwargs["download"])

    def test_dependency_manifest_keeps_pip_and_ml_transitive_dependencies_bundled(self):
        from desktop.dependencies import bundled_requirements

        names = {entry.split("==")[0] for entry in bundled_requirements()}
        self.assertTrue({"pip", "numpy", "pillow", "sympy", "flwr", "ray", "fastapi"} <= names)
        self.assertFalse({"torch", "torchvision"} & names)

    def test_dependency_manifest_can_verify_a_bundle_without_torch(self):
        from desktop import dependencies

        real = dependencies.distribution

        def fake(name):
            if name in dependencies.EXTERNAL:
                raise dependencies.PackageNotFoundError(name)
            return real(name)

        with patch.object(dependencies, "distribution", side_effect=fake):
            names = {entry.split("==")[0] for entry in dependencies.bundled_requirements(allow_missing_external=True)}
            # The two first-run wheels are absent, but their Windows runtime
            # requirements must still be proven to exist in the bundle.
            self.assertTrue({"pip", "numpy", "pillow", "sympy", "filelock", "networkx", "flwr", "ray"} <= names)
            self.assertFalse({"torch", "torchvision"} & names)
            with self.assertRaisesRegex(RuntimeError, "发行依赖缺失"):
                dependencies.bundled_requirements()

    def test_setup_progress_reports_download_percentage(self):
        from desktop.setup_server import SetupState

        state = SetupState({})
        state.report_progress("download", 50, 100)
        self.assertEqual(state.progress["phase"], "download")
        self.assertEqual(state.progress["done"], 50)
        self.assertEqual(state.progress["total"], 100)
        self.assertEqual(state.progress["percent"], 50)
        state.report_progress("download", 50, None)
        self.assertIsNone(state.progress["percent"])

    def test_gui_entry_writes_logs_into_the_user_directory(self):
        from desktop import start_gui

        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"FL_BACKDOOR_USER_DIR": temp}, clear=False):
                self.assertEqual(start_gui.log_directory(), Path(temp).resolve() / "logs")
        self.assertEqual(start_gui.application_root(), app_paths.resource_root())

    def test_abandoned_setup_window_cancels_instead_of_hanging(self):
        from desktop.setup_server import SetupState, setup_window

        original = SetupState.IDLE_TIMEOUT
        SetupState.IDLE_TIMEOUT = 0.5
        try:
            with patch("desktop.setup_server.webbrowser.open", side_effect=lambda *args, **kwargs: None):
                with self.assertRaisesRegex(RuntimeError, "已关闭或长时间无响应"):
                    setup_window({"cpu_available": True, "gpu_available": False, "warnings": [], "guidance": []})
        finally:
            SetupState.IDLE_TIMEOUT = original

    def test_cpu_fallback_must_be_acknowledged_and_forced_gpu_is_blocked(self):
        state = SetupState({"cpu_available": True, "gpu_available": False, "warnings": [], "guidance": []})
        with self.assertRaises(ValueError):
            state.confirm("auto", False)
        with self.assertRaises(RuntimeError):
            state.confirm("cuda", True)
        state.confirm("cpu", True)
        self.assertTrue(state.ready.is_set())

    def test_setup_rejects_concurrent_install_or_invalid_source(self):
        state = SetupState({})
        with self.assertRaises(ValueError):
            state.install("arbitrary", "cpu")
        state.installing = True
        with self.assertRaises(ValueError):
            state.install("official", "cpu")
        with self.assertRaises(ValueError):
            state.confirm("cpu", True)

    def test_flower_accepts_explicit_cpu_gpu_quotas(self):
        from flwr.common.config import parse_config_args
        from flwr.supercore.utils import simulation_config_from_json

        config = {"tool": {"flwr": {"app": {"config": {"num-clients": 16}}}}}
        for mode in ("cpu", "cuda"):
            with patch.dict(os.environ, {"FL_BACKDOOR_SIM_BACKEND": "local"}):
                overrides = simulation_overrides(config, mode)
            values = {k.replace("-", "_"): v for k, v in parse_config_args([overrides]).items()}
            parsed = simulation_config_from_json(values)
            self.assertEqual(parsed.num_supernodes, 16)
            self.assertEqual(parsed.backend, "local")
            self.assertEqual(parsed.client_resources_num_cpus, 1)
            self.assertEqual(parsed.client_resources_num_gpus, 1 if mode == "cuda" else 0)
            self.assertLessEqual(parsed.init_args_num_cpus, 2)
            self.assertEqual(parsed.init_args_num_gpus, 1 if mode == "cuda" else 0)

    def test_bootstrap_imports_without_site_packages_or_torch(self):
        result = subprocess.run([sys.executable, "-B", "-S", "-c",
                                 "import sys, desktop.launcher; assert 'torch' not in sys.modules; assert 'fastapi' not in sys.modules"],
                                cwd=app_paths.resource_root(), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


class ConsoleWindowTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows-only subprocess flag")
    def test_desktop_sitecustomize_suppresses_console_windows(self):
        root = str(app_paths.resource_root())
        code = "import subprocess; print(subprocess.Popen.__name__)"
        env = {**os.environ, "PYTHONPATH": root}
        env.pop("FL_BACKDOOR_DESKTOP", None)
        normal = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(normal.stdout.strip(), "Popen", normal.stderr)
        env["FL_BACKDOOR_DESKTOP"] = "1"
        hidden = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(hidden.stdout.strip(), "_NoWindowPopen", hidden.stderr)


if __name__ == "__main__":
    unittest.main()