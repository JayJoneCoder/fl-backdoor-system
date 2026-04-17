# backend/batch_manager.py
"""
Batch experiment manager for tracking status and aggregating logs.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class BatchTask:
    """Represents a single batch run."""

    def __init__(self, batch_id: str, experiments: list[dict[str, Any]]):
        self.batch_id = batch_id
        self.experiments = experiments  # list of {name, config}
        self.status = "pending"  # pending, running, completed, failed
        self.current_index = -1
        self.current_exp_name: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.log_subscribers: list[asyncio.Queue] = []
        self.logs: list[str] = []
        self._log_file: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "current_index": self.current_index,
            "current_exp_name": self.current_exp_name,
            "total": len(self.experiments),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "experiments": self.experiments,
        }

    def add_log(self, line: str):
        """Push a log line to all subscribers and keep a history buffer."""
        self.logs.append(line)

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

    def __init__(self):
        self._tasks: dict[str, BatchTask] = {}
        self._save_dir = RESULTS_DIR / "batches"
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, experiments: list[dict[str, Any]]) -> BatchTask:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        task = BatchTask(batch_id, experiments)
        self._tasks[batch_id] = task
        self._save_task(task)
        return task

    def get_task(self, batch_id: str) -> Optional[BatchTask]:
        return self._tasks.get(batch_id)

    def get_active_task(self) -> Optional[BatchTask]:
        for task in self._tasks.values():
            if task.status in ("pending", "running"):
                return task
        return None

    def _save_task(self, task: BatchTask):
        """Persist task info to JSON."""
        self._save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        file_path = self._save_dir / f"{task.batch_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[BatchManager] Failed to save task {task.batch_id}: {e}", flush=True)

    def update_task(self, task: BatchTask):
        self._save_task(task)


batch_manager = BatchManager()