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
from backend.owned_processes import OwnedProcessTree

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BUSY_STATUSES = {"starting", "running", "stopping", "cleaning", "cleanup_failed"}


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
        self._exit_code: Optional[int] = None
        self._error: Optional[str] = None
        self._run_error: Optional[str] = None
        self._terminal_status = "failed"
        self.run_id: Optional[str] = None
        self._log_generation = 0
        self._log_runs: dict[str, tuple[int, int]] = {}
        self.total_rounds: Optional[int] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_busy(self) -> bool:
        return self._status in BUSY_STATUSES or self.is_running()

    def set_log_callback(self, callback: Optional[Callable[[str], Awaitable[None]]]) -> None:
        self._on_log_line = callback

    def get_log_context(self, exp_name: str) -> tuple[int, int]:
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
            baseline = base_config if base_config is not None else config_manager.TOML_PATH.read_bytes()
            # No status/file/backup mutation until input validation succeeds.
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
            self.run_id = None
            self._monitor_task = None
            self._stdout_task = None
            try:
                if base_config is None:
                    # Keep the single-run backup behavior, but never restore a
                    # shared backup during cleanup. The FAB isolates training.
                    with config_manager._CONFIG_LOCK:
                        self._original_config = config_manager.TOML_PATH.read_bytes()
                        config_manager.update_config(overrides, backup_current=backup_config)
                        self._applied_config = config_manager.TOML_PATH.read_bytes()
                        content = self._applied_config
                section = tomli.loads(content.decode("utf-8"))["tool"]["flwr"]["app"]["config"]
                self.total_rounds = int(section.get("num-server-rounds", config_manager.CONFIG_SCHEMA["num-server-rounds"]["default"]))
                exp_dir = RESULTS_DIR / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                self.log_file = exp_dir / "run.log"
                self._log_generation += 1
                self._log_runs[exp_name] = (self._log_generation, time.time_ns())
                self.log_file.write_text("", encoding="utf-8")
                (exp_dir / "config_snapshot.toml").write_bytes(content)
                self._temp_home = Path(tempfile.mkdtemp(prefix="fl-experiment-"))
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
                           FLWR_TELEMETRY_ENABLED="0", HF_DATASETS_OFFLINE="1",
                           FLWR_HOME=str(self._temp_home / "flower"), RAY_ADDRESS="local")
                # Flower's own executors still use console entry points.
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                self.process = self._owned.spawn(
                    PROJECT_ROOT / "backend" / "experiment_worker.py",
                    ["--project", str(app_dir), "--address", address],
                    cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", bufsize=1, env=env,
                )
                self._stdout_task = asyncio.create_task(self._read_stdout())
                self._status = "running"
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
                        self.run_id = match.group(1)
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
                self._error = f"Experiment worker exited with code {code}. See run.log for details."
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
            if self._original_config is not None and self._applied_config is not None:
                await asyncio.to_thread(config_manager.restore_run_config, self._original_config, self._applied_config)
                self._original_config = self._applied_config = None
            if self._temp_home:
                await asyncio.to_thread(shutil.rmtree, self._temp_home)
                self._temp_home = None
            self._status = self._terminal_status
            self._error = self._run_error
        except Exception as exc:
            self._status = "cleanup_failed"
            self._error = f"Cleanup failed: {exc}"

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
                "exit_code": self._exit_code, "error": self._error, "run_id": self.run_id}

    def get_log_tail(self, lines: int = 50) -> list[str]:
        if not self.log_file or not self.log_file.exists():
            return []
        return self.log_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:]