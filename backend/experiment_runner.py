"""
Experiment runner: manages subprocess for `flwr run`.
"""

import asyncio
import psutil
import time
import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable, Awaitable
from backend import config_manager

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
        self._temp_home: Optional[Path] = None
        self.run_id: Optional[str] = None
        self._log_generation = 0
        self._log_runs: dict[str, tuple[int, int]] = {}
        self.total_rounds: Optional[int] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def set_log_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_log_line = callback

    def get_log_context(self, exp_name: str) -> tuple[int, int]:
        """Keep each run's identity even after stop clears the active process."""
        return self._log_runs.get(exp_name, (0, 0))

    async def start(
        self,
        exp_name: str,
        config_overrides: Optional[dict] = None,
        backup_config: bool = True,
    ) -> None:
        if self.is_running():
            raise RuntimeError("An experiment is already running")

        # Save the original configuration before applying this experiment's overrides.
        if backup_config:
            config_manager.backup_config()
        overrides = dict(config_overrides or {})
        overrides["run-name"] = exp_name
        config_manager.update_config(overrides)

        # 保存配置快照到实验目录
        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = exp_dir / "run.log"
        self.exp_name = exp_name
        self.total_rounds = int(config_manager.get_config()["num-server-rounds"])
        self.run_id = None

        # Truncate before returning from start, so a reconnect cannot replay old logs.
        self._log_generation += 1
        self._log_runs[exp_name] = (self._log_generation, time.time_ns())
        self.log_file.write_text("", encoding="utf-8")

        snapshot_path = exp_dir / "config_snapshot.toml"
        try:
            shutil.copy2(config_manager.TOML_PATH, snapshot_path)
            print(f"[Runner] Config snapshot saved to {snapshot_path}")
        except Exception as e:
            print(f"[Runner] Warning: Failed to save config snapshot: {e}")

        # 直接运行命令，不修改环境
        cmd = ["flwr", "run", str(PROJECT_ROOT), "--stream"]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["FLWR_TELEMETRY_ENABLED"] = "0"     # 关闭遥测
        env["HF_DATASETS_OFFLINE"] = "1"        # 强制 Hugging Face 离线模式

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
        if not self.process or not self.log_file or not self.process.stdout:
            return

        # Bind this reader to its own process/file, not a later run's mutable fields.
        process = self.process
        log_file = self.log_file
        with open(log_file, "a", encoding="utf-8") as f:
            while True:
                try:
                    line = await asyncio.to_thread(process.stdout.readline)
                except Exception as e:
                    print(f"[Runner] stdout read failed: {e}", flush=True)
                    break

                if line == "":
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                    continue

                # text=True already normalizes CR/CRLF; never hold a line until EOF.
                clean_line = line.rstrip("\r\n")
                f.write(clean_line + "\n")
                f.flush()

                if "Successfully started run" in line:
                    match = re.search(r"run (\d+)", line)
                    if match:
                        self.run_id = match.group(1)
                        print(f"[Runner] Captured run_id: {self.run_id}", flush=True)

                if self._on_log_line:
                    try:
                        await self._on_log_line(clean_line)
                    except Exception as e:
                        print(f"[Runner] Failed to send log line: {e}", flush=True)

        try:
            process.stdout.close()
        except Exception:
            pass

    async def stop(self) -> None:
        print("[Runner] >>> Stop method called <<<", flush=True)

        self._status = "idle"

        process = self.process
        run_id = self.run_id
        log_file = self.log_file
        temp_home = self._temp_home

        self.process = None
        self.exp_name = None
        self.log_file = None
        self.run_id = None
        self._temp_home = None

        if process and process.poll() is None:
            print("[Runner] Terminating main process...", flush=True)
            process.terminate()

        asyncio.create_task(self._background_cleanup(process, run_id, log_file, temp_home))
        print("[Runner] Stop request processed, background cleanup started", flush=True)

    async def _background_cleanup(self, process, run_id, log_file, temp_home):
        import time
        try:
            if process:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"[Runner] Error waiting process: {e}", flush=True)

            if run_id:
                print(f"[Runner] Stopping Flower run {run_id}", flush=True)
                subprocess.run(f'flwr stop {run_id}', shell=True, check=False, timeout=5)

            print("[Runner] Stopping Ray", flush=True)
            subprocess.run('ray stop', shell=True, check=False, timeout=5)

            try:
                import psutil
                target_names = {'raylet.exe', 'gcs_server.exe'}
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        name = proc.info['name'].lower() if proc.info['name'] else ''
                        if name in target_names:
                            proc.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                time.sleep(2)
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        name = proc.info['name'].lower() if proc.info['name'] else ''
                        if name in target_names:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except ImportError:
                if os.name == 'nt':
                    subprocess.run('taskkill /F /IM raylet.exe /T', shell=True, check=False)
                    subprocess.run('taskkill /F /IM gcs_server.exe /T', shell=True, check=False)

            from backend import config_manager
            if config_manager.BACKUP_PATH.exists():
                config_manager.restore_config()
                print("[Runner] Config restored from backup", flush=True)

            if temp_home and temp_home.exists():
                shutil.rmtree(temp_home, ignore_errors=True)
                print(f"[Runner] Cleaned up temp: {temp_home}", flush=True)

            print("[Runner] Background cleanup finished", flush=True)
        except Exception as e:
            print(f"[Runner] Background cleanup error: {e}", flush=True)

    async def get_status(self) -> dict:
        if self.process is None:
            return {"status": "idle", "exp_name": None}
        if self.is_running():
            return {"status": "running", "exp_name": self.exp_name, "total_rounds": self.total_rounds}
        return {
            "status": "finished",
            "exp_name": self.exp_name,
            "exit_code": self.process.returncode,
            "total_rounds": self.total_rounds,
        }

    def get_log_tail(self, lines: int = 50) -> list[str]:
        if not self.log_file or not self.log_file.exists():
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return all_lines[-lines:]
