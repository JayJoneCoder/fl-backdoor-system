"""
Experiment runner: manages subprocess for `flwr run`.
"""

import asyncio
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable, Awaitable

import tomli_w

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class ExperimentRunner:
    """Manages a single running experiment subprocess."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.exp_name: Optional[str] = None
        self.log_file: Optional[Path] = None
        self._status = "idle"
        self._stdout_task: Optional[asyncio.Task] = None
        self._on_log_line: Optional[Callable[[str], Awaitable[None]]] = None
        self._temp_home: Optional[Path] = None   # 用于清理临时目录

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def set_log_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_log_line = callback

    async def start(
        self,
        exp_name: str,
        config_overrides: Optional[dict] = None,
        backup_config: bool = True,
    ) -> None:
        if self.is_running():
            raise RuntimeError("An experiment is already running")

        from backend import config_manager

        # 应用配置覆盖（不涉及 num-clients）
        if config_overrides:
            config_manager.update_config(config_overrides)
        else:
            config_manager.update_config({"run-name": exp_name})

        if backup_config:
            config_manager.backup_config()

        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = exp_dir / "run.log"
        self.exp_name = exp_name

        # 直接运行命令，不修改环境
        cmd = ["flwr", "run", str(PROJECT_ROOT), "--stream"]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        self._status = "starting"

        self.process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )

        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._status = "running"

    async def _read_stdout(self) -> None:
        if not self.process or not self.log_file:
            return

        with open(self.log_file, "w", encoding="utf-8") as f:
            assert self.process.stdout
            for line in iter(self.process.stdout.readline, ""):
                f.write(line)
                f.flush()
                if self._on_log_line:
                    try:
                        await self._on_log_line(line.rstrip())
                    except Exception as e:
                        print(f"[Runner] Failed to send log line: {e}")
                else:
                    print("[Runner] _on_log_line is None")

        self.process.stdout.close()

    async def stop(self) -> None:
        if self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._stdout_task and not self._stdout_task.done():
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass

        self.process = None
        self.exp_name = None
        self.log_file = None
        self._status = "idle"

        # 清理临时目录
        if self._temp_home and self._temp_home.exists():
            shutil.rmtree(self._temp_home, ignore_errors=True)
            print(f"[Runner] Cleaned up temporary config at {self._temp_home}")
            self._temp_home = None

        from backend import config_manager
        config_manager.restore_config()

    async def get_status(self) -> dict:
        if self.process is None:
            return {"status": "idle", "exp_name": None}
        if self.is_running():
            return {"status": "running", "exp_name": self.exp_name}
        return {
            "status": "finished",
            "exp_name": self.exp_name,
            "exit_code": self.process.returncode,
        }

    def get_log_tail(self, lines: int = 50) -> list[str]:
        if not self.log_file or not self.log_file.exists():
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return all_lines[-lines:]