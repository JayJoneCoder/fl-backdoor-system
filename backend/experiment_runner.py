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
        self.run_id: Optional[str] = None   # 保存当前运行的 run_id

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

        # 保存配置快照到实验目录
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
        env["FLWR_TELEMETRY_ENABLED"] = "0"      # 关闭遥测
        env["HF_DATASETS_OFFLINE"] = "1"         # 强制 Hugging Face 离线模式

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

        with open(self.log_file, "w", encoding="utf-8") as f:
            while True:
                try:
                    # 把阻塞式 readline 放到线程里
                    line = await asyncio.to_thread(self.process.stdout.readline)
                except Exception as e:
                    print(f"[Runner] stdout read failed: {e}", flush=True)
                    break

                if line == "":
                    # 进程结束后，退出读取循环
                    if self.process.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                    continue

                f.write(line)
                f.flush()

                if "Successfully started run" in line:
                    match = re.search(r"run (\d+)", line)
                    if match:
                        self.run_id = match.group(1)
                        print(f"[Runner] Captured run_id: {self.run_id}", flush=True)

                callback = self._on_log_line
                if callback is not None:
                    try:
                        await callback(line.rstrip())
                    except Exception as e:
                        print(f"[Runner] Failed to send log line: {e}", flush=True)
                else:
                    print("[Runner] _on_log_line is None", flush=True)

        try:
            self.process.stdout.close()
        except Exception:
            pass

    async def stop(self) -> None:
        print("[Runner] >>> Stop method called <<<", flush=True)

        # 立即更新状态，让前端知道已停止
        self._status = "idle"

        # 保存需要清理的数据
        process = self.process
        run_id = self.run_id
        log_file = self.log_file
        temp_home = self._temp_home

        # 清理实例属性，避免重复操作
        self.process = None
        self.exp_name = None
        self.log_file = None
        self.run_id = None
        self._temp_home = None

        # 终止主进程（不等待，快速返回）
        if process and process.poll() is None:
            print("[Runner] Terminating main process...", flush=True)
            process.terminate()
            # 不等待，直接返回，让操作系统异步终止

        # 启动后台清理任务（不阻塞当前请求）
        asyncio.create_task(self._background_cleanup(process, run_id, log_file, temp_home))
        print("[Runner] Stop request processed, background cleanup started", flush=True)

    async def _background_cleanup(self, process, run_id, log_file, temp_home):
        """后台清理任务，不阻塞 HTTP 响应"""
        import time
        try:
            # 等待主进程结束（最多5秒）
            if process:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"[Runner] Error waiting process: {e}", flush=True)

            # 清理 Flower 运行
            if run_id:
                print(f"[Runner] Stopping Flower run {run_id}", flush=True)
                subprocess.run(f'flwr stop {run_id}', shell=True, check=False, timeout=5)

            # 停止 Ray
            print("[Runner] Stopping Ray", flush=True)
            subprocess.run('ray stop', shell=True, check=False, timeout=5)

            # 强制清理残留进程
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

            # 恢复配置
            from backend import config_manager
            if config_manager.BACKUP_PATH.exists():
                config_manager.restore_config()
                print("[Runner] Config restored from backup", flush=True)

            # 清理临时目录
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