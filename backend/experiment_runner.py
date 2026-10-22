"""Serialize experiment lifecycles and clean up only the owned worker tree."""

import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import tomli
import tomli_w

from backend import config_manager
from backend.identity import claim_resource
from backend.owned_processes import OwnedProcessTree
from backend.runtime_status import environment_status
from fl_backdoor.utils.device import select_device
from fl_backdoor.utils.run_artifacts import build_config_snapshot, reserve_experiment, write_run_metadata
from fl_backdoor.utils.app_paths import data_root, desktop_mode, user_root

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BUSY_STATUSES = {"starting", "running", "stopping", "cleaning", "cleanup_failed"}

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_ERROR_LINE = re.compile(r"^(?:[A-Za-z_][\w.]*Error|Exception):\s*(.+)$")
_FATAL_PREFIX = "!!! FATAL ERROR in server main:"
_MAX_FAILURE_REASON_CHARS = 300


def _extract_failure_reason(log_file: Optional[Path], max_lines: int = 500) -> Optional[str]:
    """Return the server-side root cause recorded in a failed worker log."""
    if log_file is None or not log_file.exists():
        return None
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return None

    cleaned_lines = [_ANSI_ESCAPE.sub("", line).strip() for line in lines]
    for line in cleaned_lines:
        if line.startswith(_FATAL_PREFIX):
            reason = line[len(_FATAL_PREFIX):].strip()
            if reason:
                return reason[:_MAX_FAILURE_REASON_CHARS]
    for line in reversed(cleaned_lines):
        match = _ERROR_LINE.match(line)
        if match:
            return match.group(1).strip()[:_MAX_FAILURE_REASON_CHARS]
    return None


def _worker_failure_message(code: int, log_file: Optional[Path]) -> str:
    reason = _extract_failure_reason(log_file)
    if not reason:
        return f"Experiment worker exited with code {code}. See run.log for details."
    separator = "" if reason.endswith((".", "!", "?")) else "."
    return f"Experiment worker exited with code {code}: {reason}{separator} See run.log for details."


class ExperimentRunner:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.exp_name: Optional[str] = None
        self.log_file: Optional[Path] = None
        self._status = "idle"
        self._stdout_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._on_log_line: Optional[Callable[[str], Awaitable[None]]] = None
        self._temp_home: Optional[Path] = None
        self._owned: Optional[OwnedProcessTree] = None
        self._lock = asyncio.Lock()
        self._cancel_requested = False
        self._original_config: Optional[bytes] = None
        self._applied_config: Optional[bytes] = None
        self._config_path: Optional[Path] = None
        self._exit_code: Optional[int] = None
        self._error: Optional[str] = None
        self._run_error: Optional[str] = None
        self._terminal_status = "failed"
        self.run_id: Optional[str] = None
        self.flower_run_id: Optional[str] = None
        self._run_metadata: Optional[dict] = None
        self._experiment_dir: Optional[Path] = None
        self._log_generation = 0
        self._log_runs: dict[str, tuple[int, int]] = {}
        self._log_directories: dict[str, Path] = {}
        self.total_rounds: Optional[int] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_busy(self) -> bool:
        return self._status in BUSY_STATUSES or self.is_running()

    def set_log_callback(self, callback: Optional[Callable[[str], Awaitable[None]]]) -> None:
        self._on_log_line = callback

    def get_log_context(self, exp_name: str, directory: Path | None = None) -> tuple[int, int]:
        if directory is not None:
            owned = self._log_directories.get(exp_name, self._experiment_dir)
            if owned is not None and owned.resolve() != directory.resolve():
                return (0, 0)
        return self._log_runs.get(exp_name, (0, 0))

    async def start(
        self, exp_name: str, config_overrides: Optional[dict] = None,
        backup_config: bool = True, *, base_config: Optional[bytes] = None,
    ) -> None:
        async with self._lock:
            if self.is_busy():
                raise RuntimeError("An experiment is running or cleanup is not finished")
            overrides = dict(config_overrides or {})
            overrides["run-name"] = exp_name
            baseline = base_config if base_config is not None else config_manager.config_path().read_bytes()
            # No status/file/backup mutation until input validation succeeds.
            content = config_manager.build_run_config(baseline, overrides)
            section = tomli.loads(content.decode("utf-8"))["tool"]["flwr"]["app"]["config"]
            mode = section.get("runtime-device", "auto")
            if mode == "auto":
                mode = os.environ.get("FL_BACKDOOR_DEVICE", "auto")
            device_report = await environment_status()
            selected_device = select_device(mode, device_report)
            results_root = config_manager.get_results_dir(section, default=RESULTS_DIR)
            # Reserve atomically BEFORE touching config/backup/logs. A repeated
            # name gets a run-ID suffix instead of rejecting the launch.
            exp_dir, metadata = reserve_experiment(results_root, exp_name)
            metadata["runtime_device"] = {"requested": mode, "selected": selected_device,
                                           "warnings": device_report["warnings"]}
            claim_resource("experiment", exp_dir)
            requested_name = exp_name
            exp_name = metadata["name"]
            if exp_name != requested_name:
                # The directory name is this run's identity: the worker writes
                # <run-name>.csv there, so the snapshot must use the same name.
                overrides["run-name"] = exp_name
                content = config_manager.build_run_config(baseline, overrides)
            self._status = "starting"
            self.exp_name = exp_name
            self._cancel_requested = False
            self._exit_code = None
            self._error = None
            self._run_error = None
            self._original_config = None
            self._applied_config = None
            self._owned = None
            self.process = None
            self.run_id = metadata["run_id"]
            self.flower_run_id = None
            self._run_metadata = metadata
            self._experiment_dir = exp_dir
            self._config_path = config_manager.config_path()
            self._monitor_task = None
            self._stdout_task = None
            try:
                if base_config is None:
                    # Keep the single-run backup behavior, but never restore a
                    # shared backup during cleanup. The FAB isolates training.
                    with config_manager._CONFIG_LOCK:
                        self._original_config = config_manager.config_path().read_bytes()
                        config_manager.update_config(overrides, backup_current=backup_config)
                        self._applied_config = config_manager.config_path().read_bytes()
                        content = self._applied_config
                document = tomli.loads(content.decode("utf-8"))
                section = document["tool"]["flwr"]["app"]["config"]
                # Use the same absolute artifact root in the isolated Flower app
                # as in the API; its working directory must not redirect output.
                section["results-dir"] = str(results_root)
                content = tomli_w.dumps(document).encode("utf-8")
                snapshot_content = build_config_snapshot(document)
                self.total_rounds = int(section.get("num-server-rounds", config_manager.CONFIG_SCHEMA["num-server-rounds"]["default"]))
                self.log_file = exp_dir / "run.log"
                self._log_generation += 1
                self._log_runs[exp_name] = (self._log_generation, time.time_ns())
                self._log_directories[exp_name] = exp_dir
                with self.log_file.open("x", encoding="utf-8"):
                    pass
                with (exp_dir / "config_snapshot.toml").open("xb") as snapshot:
                    snapshot.write(snapshot_content)
                self._temp_home = Path(tempfile.mkdtemp(prefix="fl-experiment-", dir=user_root() / "tmp" if desktop_mode() else None))
                preparing = asyncio.create_task(asyncio.to_thread(self._prepare_runtime, content))
                try:
                    app_dir, address = await asyncio.shield(preparing)
                except asyncio.CancelledError:
                    # The thread cannot be cancelled; don't remove its directory
                    # while it is still copying the immutable app.
                    await preparing
                    raise
                self._owned = OwnedProcessTree()
                env = os.environ.copy()
                env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                           FL_BACKDOOR_DEVICE=selected_device,
                           FLWR_TELEMETRY_ENABLED="0", HF_DATASETS_OFFLINE="1",
                           FLWR_HOME=str(self._temp_home / "flower"), RAY_ADDRESS="local",
                           FL_BACKDOOR_RUN_ID=self.run_id,
                           FL_BACKDOOR_DATA_DIR=str(data_root()))
                # Ray deserializes ClientApp handlers by module name in fresh
                # processes. Prefer this run's snapshot over an editable install
                # pointing at the live workspace (including its RNG/data code).
                env["PYTHONPATH"] = str(app_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                from fl_backdoor.utils.ray_runtime import configure_worker_environment
                configure_worker_environment(env, app_dir)
                env["PYTHONHASHSEED"] = str(int(section.get("seed", 42)) % (2**32))
                # Flower's own executors still use console entry points.
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                scripts = self._temp_home / "scripts"
                env["PATH"] = str(scripts) + os.pathsep + env["PATH"]
                self.process = self._owned.spawn(
                    PROJECT_ROOT / "backend" / "experiment_worker.py",
                    ["--project", str(app_dir), "--address", address],
                    cwd=str(app_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", bufsize=1, env=env,
                )
                self._stdout_task = asyncio.create_task(self._read_stdout())
                self._status = "running"
                self._save_metadata()
                self._monitor_task = asyncio.create_task(self._monitor())
            except BaseException as exc:
                self._error = f"Experiment startup failed: {exc}"
                self._run_error = self._error
                self._terminal_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                await self._cleanup()
                raise

    def _prepare_runtime(self, content: bytes) -> tuple[Path, str]:
        app_dir = self._temp_home / "app"
        app_dir.mkdir()
        (app_dir / "pyproject.toml").write_bytes(content)
        for name in ("README.md", "LICENSE"):
            if (PROJECT_ROOT / name).is_file():
                shutil.copy2(PROJECT_ROOT / name, app_dir / name)
        shutil.copytree(PROJECT_ROOT / "fl_backdoor", app_dir / "fl_backdoor", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
        # This runs inside the same shielded preparation task as the snapshot:
        # cancellation must not delete a directory while its launchers are built.
        from fl_backdoor.utils.flower_runtime import create_entry_points
        create_entry_points(self._temp_home / "scripts")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{sock.getsockname()[1]}"
        home = self._temp_home / "flower"
        home.mkdir()
        (home / "config.toml").write_text(tomli_w.dumps({"superlink": {
            "default": "project-runtime", "project-runtime": {"address": address, "insecure": True},
        }}), encoding="utf-8")
        return app_dir, address

    async def _read_stdout(self) -> None:
        if not self.process or not self.log_file or not self.process.stdout:
            return
        process, log_file = self.process, self.log_file
        with log_file.open("a", encoding="utf-8") as f:
            while True:
                line = await asyncio.to_thread(process.stdout.readline)
                if line == "":
                    break
                clean_line = line.rstrip("\r\n")
                f.write(clean_line + "\n")
                f.flush()
                if "Successfully started run" in line:
                    match = re.search(r"run (\d+)", line)
                    if match:
                        self.flower_run_id = match.group(1)
                if self._on_log_line:
                    try:
                        await self._on_log_line(clean_line)
                    except Exception as exc:
                        print(f"[Runner] Failed to send log line: {exc}", flush=True)
        process.stdout.close()

    async def _monitor(self) -> None:
        try:
            while self.process.poll() is None and not self._cancel_requested:
                self._owned.observe()
                await asyncio.sleep(0.1)
            code = self.process.poll()
            self._exit_code = code
            self._terminal_status = "cancelled" if self._cancel_requested else ("finished" if code == 0 else "failed")
            if self._terminal_status == "failed":
                self._error = _worker_failure_message(code, self.log_file)
        except Exception as exc:
            self._terminal_status = "failed"
            self._error = f"Experiment monitor failed: {exc}"
        self._run_error = self._error
        await self._cleanup()

    async def _cleanup(self) -> None:
        self._status = "stopping" if self._cancel_requested else "cleaning"
        try:
            if self._owned:
                await asyncio.to_thread(self._owned.cleanup)
                self._owned = None
            if self.process:
                self._exit_code = self.process.poll()
            if self._stdout_task:
                await asyncio.wait_for(asyncio.shield(self._stdout_task), timeout=5)
            if (
                self._terminal_status == "failed"
                and self._exit_code not in (None, 0)
                and (self._run_error or "").startswith("Experiment worker exited with code")
            ):
                # The monitor may have read the log before buffered lines landed.
                self._run_error = _worker_failure_message(self._exit_code, self.log_file)
            if self._original_config is not None and self._applied_config is not None:
                await asyncio.to_thread(config_manager.restore_run_config, self._original_config, self._applied_config, path=self._config_path)
                self._original_config = self._applied_config = None
            if self._temp_home:
                await asyncio.to_thread(shutil.rmtree, self._temp_home)
                self._temp_home = None
            self._status = self._terminal_status
            self._error = self._run_error
            self._save_metadata()
        except Exception as exc:
            self._status = "cleanup_failed"
            self._error = f"Cleanup failed: {exc}"
            try:
                self._save_metadata()
            except Exception:
                pass  # Keep the original cleanup error; retry remains available.

    def _save_metadata(self) -> None:
        if self._run_metadata is not None and self._experiment_dir is not None:
            self._run_metadata.update(status=self._status, exit_code=self._exit_code,
                                      error=self._error, flower_run_id=self.flower_run_id)
            write_run_metadata(self._experiment_dir, self._run_metadata)

    async def wait(self) -> dict:
        if self._monitor_task:
            await asyncio.shield(self._monitor_task)
        return await self.get_status()

    async def stop(self) -> dict:
        async with self._lock:
            if self._status == "cleanup_failed":
                self._status = "stopping" if self._cancel_requested else "cleaning"
                self._monitor_task = asyncio.create_task(self._cleanup())
            elif self._status in {"starting", "running"}:
                self._cancel_requested = True
                self._status = "stopping"
        return await self.wait()

    async def get_status(self) -> dict:
        status = self._status
        # Allow status observation of externally supplied processes in tests.
        if status == "idle" and self.process:
            code = self.process.poll()
            status = "running" if code is None else ("finished" if code == 0 else "failed")
        return {"status": status, "exp_name": self.exp_name, "total_rounds": self.total_rounds,
                "exit_code": self._exit_code, "error": self._error, "run_id": self.run_id,
                "flower_run_id": self.flower_run_id}

    def get_log_tail(self, lines: int = 50) -> list[str]:
        if not self.log_file or not self.log_file.exists():
            return []
        return self.log_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:]