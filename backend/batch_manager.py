# backend/batch_manager.py
"""
Batch experiment manager for tracking status and aggregating logs.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

from backend.config_manager import _atomic_write

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESTART_ERROR = "Backend restarted; task was interrupted and was not automatically resumed."
UNRECORDED_ERROR = "Execution details were not recorded by this older version."


class BatchTask:
    """Represents a single batch run."""

    def __init__(self, batch_id: str, experiments: list[dict[str, Any]], base_config: bytes = b""):
        self.batch_id = batch_id
        self.experiments = experiments  # list of {name, config}
        self.status = "pending"
        self.current_index = -1
        self.current_exp_name: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.log_subscribers: list[asyncio.Queue] = []
        self.logs: list[str] = []
        self._log_file: Optional[Path] = None
        self.base_config = base_config
        self.cancel_requested = False
        self.job: Optional[asyncio.Task] = None
        self.results = [{"name": exp.get("name"), "status": "pending", "exit_code": None,
                         "error": None, "start_time": None, "end_time": None} for exp in experiments]

    def to_dict(self) -> dict:
        return {
            "record_version": 2,
            "batch_id": self.batch_id,
            "status": self.status,
            "current_index": self.current_index,
            "current_exp_name": self.current_exp_name,
            "total": len(self.experiments),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "experiments": self.experiments,
            "results": self.results,
            "completed_count": sum(item["status"] in {"completed", "failed", "cancelled"} for item in self.results),
            "success_count": sum(item["status"] == "completed" for item in self.results),
            "failed_count": sum(item["status"] == "failed" for item in self.results),
            "cancelled_count": sum(item["status"] == "cancelled" for item in self.results),
            "unknown_count": sum(item["status"] == "unknown" for item in self.results),
        }

    def add_log(self, line: str):
        """Push a log line to all subscribers and keep a history buffer."""
        self.logs.append(line)
        if self._log_file:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        for queue in self.log_subscribers:
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                pass

    async def subscribe_logs(self) -> tuple[asyncio.Queue, list[str]]:
        """Create a new subscriber queue and return current log history."""
        queue = asyncio.Queue(maxsize=1000)
        self.log_subscribers.append(queue)
        return queue, list(self.logs)

    def unsubscribe_logs(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        if queue in self.log_subscribers:
            self.log_subscribers.remove(queue)


class BatchManager:
    """Manages active batch tasks."""

    def __init__(self, save_dir: Optional[Path] = None):
        self._tasks: dict[str, BatchTask] = {}
        self._save_dir = save_dir if save_dir is not None else RESULTS_DIR / "batches"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._load_tasks()

    def _load_tasks(self):
        """Restore records, not processes. An interrupted run is never resumed blindly."""
        for path in self._save_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                task = BatchTask(data["batch_id"], data["experiments"])
                task.status = data["status"]
                task.current_index = data.get("current_index", -1)
                task.current_exp_name = data.get("current_exp_name")
                task.error = data.get("error")
                for key in ("start_time", "end_time"):
                    value = data.get(key)
                    setattr(task, key, datetime.fromisoformat(value) if value else None)
                saved_results = data.get("results")
                for index, item in enumerate(task.results):
                    saved = saved_results[index] if isinstance(saved_results, list) and index < len(saved_results) else None
                    if isinstance(saved, dict) and saved.get("name") == item["name"] and saved.get("status") in {
                        "pending", "running", "completed", "failed", "cancelled", "unknown",
                    }:
                        item.update(saved)
                        # Earlier loaders synthesized cancellations for old records
                        # without execution details. Do not treat those as evidence
                        # that an experiment was never started.
                        if (data.get("record_version") != 2 and data.get("error") == RESTART_ERROR
                                and item["status"] == "cancelled"
                                and item.get("error") == "Not started before backend restart."
                                and all(item.get(key) is None for key in ("start_time", "end_time", "exit_code"))):
                            item.update(status="unknown", error=UNRECORDED_ERROR)
                    else:
                        item.update(status="unknown", error=UNRECORDED_ERROR)
                task._log_file = self._save_dir / f"{task.batch_id}.log"
                if task._log_file.exists():
                    task.logs = task._log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-10000:]
                if task.status in {"pending", "running", "stopping"}:
                    task.status = "failed"
                    task.error = RESTART_ERROR
                    task.end_time = datetime.now()
                    task.current_exp_name = None
                    for item in task.results:
                        if item["status"] == "running":
                            item.update(status="failed", error=task.error, end_time=task.end_time.isoformat())
                        elif item["status"] == "pending":
                            item.update(status="cancelled", error="Not started before backend restart.")
                self._tasks[task.batch_id] = task
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"[BatchManager] Cannot load {path.name}: {exc}", flush=True)

    def create_task(self, experiments: list[dict[str, Any]], base_config: bytes = b"") -> BatchTask:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        task = BatchTask(batch_id, experiments, base_config)
        task._log_file = self._save_dir / f"{batch_id}.log"
        self._save_task(task)
        self._tasks[batch_id] = task
        return task

    def get_task(self, batch_id: str) -> Optional[BatchTask]:
        return self._tasks.get(batch_id)

    def get_active_task(self) -> Optional[BatchTask]:
        for task in self._tasks.values():
            if task.status in ("pending", "running", "stopping"):
                return task
        return None

    def get_latest_task(self) -> Optional[BatchTask]:
        return self._tasks[max(self._tasks)] if self._tasks else None

    def _save_task(self, task: BatchTask):
        """Persist task info to JSON."""
        self._save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        file_path = self._save_dir / f"{task.batch_id}.json"
        try:
            content = json.dumps(task.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
            _atomic_write(file_path, content)
        except Exception as e:
            print(f"[BatchManager] Failed to save task {task.batch_id}: {e}", flush=True)
            raise

    def update_task(self, task: BatchTask):
        self._save_task(task)


batch_manager = BatchManager()