"""
FastAPI backend for FL backdoor experiment platform.
"""

import asyncio
from contextlib import asynccontextmanager
import subprocess
import shutil
from datetime import datetime
import os
import sys
from pathlib import Path
from urllib.parse import quote
from typing import Any
import zipfile
import io
import re
import uuid
import pandas as pd
import tomli

from fastapi import FastAPI, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse


# Add project root to sys.path for importing fl_backdoor modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config_manager
from backend.catalog import get_catalog
from backend.config_validation import parse_boolean
from backend.experiment_runner import ExperimentRunner
from backend.log_stream import LogFileReader, MetricFileReader, read_total_rounds
from backend.results_scanner import (
    get_experiment_detail,
    list_experiments,
    get_summary_detail,
)
from backend.batch_manager import batch_manager, BatchTask
from fl_backdoor.utils.paths import child_path, direct_files, experiment_path, summary_path, summary_root
from fl_backdoor.utils.run_artifacts import read_run_metadata

# ------------------------------
# FastAPI app setup
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _sync_catalog(_results_root())
    except HTTPException:
        # Keep diagnostics and the existing runner reachable; history requests
        # report 503 until the database is repaired and synchronization retries.
        pass
    catalog_job = asyncio.create_task(_refresh_catalog())
    try:
        yield
    finally:
        # Normal backend shutdown must not abandon an owned worker/cleanup task.
        task = batch_manager.get_active_task()
        if task:
            task.cancel_requested = True
            task.resume_event.set()
        await runner.stop()
        if task and task.job and not task.job.done():
            await asyncio.shield(task.job)
        catalog_job.cancel()
        await asyncio.gather(catalog_job, return_exceptions=True)
        try:
            await _sync_catalog(_results_root())
        except HTTPException:
            pass


app = FastAPI(title="FL Backdoor Platform API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.requests import Request
from starlette.websockets import WebSocket as StarletteWebSocket

from starlette.requests import Request
from starlette.websockets import WebSocket as StarletteWebSocket

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    # 跳过 WebSocket 升级和 OPTIONS 预检请求的日志（但正常处理）
    if request.method == "OPTIONS" or request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)
    response = await call_next(request)
    return response

# Static files for results
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _results_root() -> Path:
    return config_manager.get_results_dir(default=RESULTS_DIR)


async def _sync_catalog(root: Path, *, force: bool = False) -> dict:
    """Only importers write the SQLite read model; never change run artifacts."""
    try:
        catalog = get_catalog()
        report = await asyncio.to_thread(catalog.sync_root, root, force=force)
        catalog.record_sync(root, report)
        return report
    except Exception as exc:
        print(f"[Catalog] Synchronization failed: {exc}", flush=True)
        raise HTTPException(503, "实验数据库暂时不可用，原始结果文件未改变，请检查磁盘、权限或数据库备份。") from exc


async def _refresh_catalog():
    while True:
        await asyncio.sleep(5)
        try:
            await _sync_catalog(_results_root())
        except Exception as exc:
            print(f"[Catalog] Background refresh will retry: {exc}", flush=True)


@app.get("/api/catalog/status")
async def catalog_status():
    try:
        return await asyncio.to_thread(get_catalog().stats)
    except Exception as exc:
        raise HTTPException(503, "无法读取实验数据库状态，请检查后端日志。") from exc


@app.post("/api/catalog/sync")
async def synchronize_catalog():
    return await _sync_catalog(_results_root(), force=True)


class ConfiguredResults:
    """Resolve the same configured root as training, without restarting the API."""

    async def __call__(self, scope, receive, send):
        root = _results_root()
        if not root.is_dir():
            await JSONResponse(status_code=404, content={"detail": "结果目录尚不存在。"})(scope, receive, send)
            return
        await StaticFiles(directory=str(root))(scope, receive, send)


app.mount("/results", ConfiguredResults(), name="results")

# ------------------------------
# Global experiment runner instance
# ------------------------------
runner = ExperimentRunner()
execution_lock = asyncio.Lock()


def _experiment_directory(name: str) -> Path:
    try:
        # History and its assets always belong to the currently selected root.
        return experiment_path(_results_root(), name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _monitor_directory(name: str, run_id: str | None) -> Path:
    if run_id and runner.run_id == run_id and runner.exp_name == name and runner._experiment_dir is not None:
        directory = experiment_path(runner._experiment_dir.parent, name)
    else:
        directory = _experiment_directory(name)
    if run_id and read_run_metadata(directory).get("run_id") != run_id:
        raise ValueError("实验运行标识与结果目录不一致，请刷新后重新连接。")
    return directory


def _runner_owns_directory(directory: Path) -> bool:
    if runner.exp_name != directory.name:
        return False
    owned = runner._experiment_dir
    if owned is None or owned.resolve() != directory.resolve():
        return False
    if runner.run_id:
        return read_run_metadata(directory).get("run_id") == runner.run_id
    return True


def _summary_directory(name: str) -> Path:
    try:
        return summary_path(_results_root(), name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _download_header(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}"}


# ------------------------------
# WebSocket for real-time logs
# ------------------------------
@app.websocket("/ws/logs/{exp_name}")
async def websocket_logs(websocket: WebSocket, exp_name: str):
    await websocket.accept()

    try:
        run_id = getattr(websocket, "query_params", {}).get("run_id")
        exp_dir = _monitor_directory(exp_name, run_id)
        log_path = child_path(exp_dir, "run.log")
        child_path(exp_dir, f"{exp_name}.csv")
        child_path(exp_dir, "config_snapshot.toml")
    except (ValueError, OSError, HTTPException):
        await websocket.close(code=1008, reason="实验名称、路径或运行标识无效")
        return
    if not log_path.exists():
        await websocket.send_text("[System] Waiting for log file...")

    # One log source per connection. Never replace the runner's batch callback.
    send_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(_stream_log_file(websocket, log_path, send_lock)),
        asyncio.create_task(_watch_csv_files(websocket, exp_dir, send_lock)),
        asyncio.create_task(_stream_experiment_status(websocket, exp_dir, send_lock)),
        asyncio.create_task(_wait_for_disconnect(websocket)),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, OSError, RuntimeError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

@app.websocket("/ws/batch/{batch_id}")
async def websocket_batch_logs(websocket: WebSocket, batch_id: str):
    try:
        await websocket.accept()

        task = batch_manager.get_task(batch_id)
        if task is None:
            await websocket.send_text("[System] Batch task not found.")
            await websocket.close()
            return

        queue, history = await task.subscribe_logs()

        # 先发历史日志，避免晚连时丢日志
        for line in history:
            await websocket.send_text(line)

        # 再发当前状态
        await websocket.send_json({"type": "status", "data": task.to_dict()})

        try:
            next_status = asyncio.get_running_loop().time() + 1
            while True:
                try:
                    timeout = max(0.001, next_status - asyncio.get_running_loop().time())
                    line = await asyncio.wait_for(queue.get(), timeout=timeout)
                    await websocket.send_text(line)
                except asyncio.TimeoutError:
                    pass
                # Continuous worker output must not starve status updates.
                if asyncio.get_running_loop().time() >= next_status:
                    await websocket.send_json({"type": "status", "data": task.to_dict()})
                    next_status = asyncio.get_running_loop().time() + 1
                    if task.status in ("completed", "failed", "cancelled"):
                        await asyncio.sleep(1)
                        break
        except WebSocketDisconnect:
            print(f"[WS-Batch] Client disconnected for {batch_id}", flush=True)
        finally:
            task.unsubscribe_logs(queue)

    except Exception as e:
        print(f"[WS-Batch] Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass

async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        await websocket.receive_text()


async def _stream_log_file(
    websocket: WebSocket, log_path: Path, send_lock: asyncio.Lock | None = None
) -> None:
    """Replay complete lines and follow new runs, truncations, and replacements."""
    reader = LogFileReader(log_path)
    lock = send_lock or asyncio.Lock()
    while True:
        generation, _ = runner.get_log_context(log_path.parent.name, log_path.parent)
        lines = await asyncio.to_thread(reader.read_lines, generation)
        for line in lines:
            async with lock:
                await websocket.send_text(line)
        await asyncio.sleep(0.1)


async def _watch_csv_files(
    websocket: WebSocket, exp_dir: Path, send_lock: asyncio.Lock | None = None
) -> None:
    """Replay metrics on connect; polling also works before the directory exists."""
    main_csv = exp_dir / f"{exp_dir.name}.csv"
    reader = MetricFileReader(main_csv)
    lock = send_lock or asyncio.Lock()
    while True:
        generation, started_at_ns = runner.get_log_context(exp_dir.name, exp_dir)
        rows = await asyncio.to_thread(reader.read_rows, generation, started_at_ns)
        for row in rows:
            async with lock:
                await websocket.send_json({"type": "csv_update", "file": main_csv.name, "data": row})
        await asyncio.sleep(0.5)


async def _stream_experiment_status(
    websocket: WebSocket, exp_dir: Path, send_lock: asyncio.Lock
) -> None:
    previous = None
    while True:
        status = await runner.get_status()
        if not _runner_owns_directory(exp_dir):
            status = {"status": "idle", "exp_name": exp_dir.name}
            try:
                metadata = await asyncio.to_thread(read_run_metadata, exp_dir)
                status.update({key: metadata[key] for key in ("status", "run_id", "flower_run_id", "exit_code", "error") if key in metadata})
            except (ValueError, OSError, KeyError):
                pass
        if not status.get("total_rounds"):
            try:
                snapshot = child_path(exp_dir, "config_snapshot.toml")
                status["total_rounds"] = await asyncio.to_thread(read_total_rounds, snapshot)
            except ValueError:
                status["total_rounds"] = None
        status["log_generation"] = runner.get_log_context(exp_dir.name, exp_dir)[0]
        if status != previous:
            async with send_lock:
                await websocket.send_json({"type": "status", "data": status})
            previous = status
        await asyncio.sleep(0.5)

# ------------------------------
# REST API Endpoints
# ------------------------------
@app.get("/api/config/schema")
async def get_config_schema():
    """Return configuration schema for frontend form generation."""
    return config_manager.get_config_schema()

@app.get("/api/config")
async def get_current_config():
    """Return current configuration (including UI computed fields)."""
    try:
        return config_manager.get_ui_config()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/config")
async def update_current_config(updates: dict[str, Any], backup_name: str | None = None):
    """Update configuration in pyproject.toml."""
    try:
        config_manager.update_config(updates, backup_current=True, backup_name=backup_name)
        return {"status": "success"}
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/config/upload")
async def upload_config_file(file: UploadFile):
    """Replace entire pyproject.toml with uploaded file."""
    if not file.filename or not file.filename.lower().endswith(".toml"):
        raise HTTPException(400, "Only .toml files are accepted")
    try:
        content = await file.read()
        # 将 Windows 换行符 \r\n 统一转换为 \n
        text_content = content.decode('utf-8').replace('\r\n', '\n')
        config_manager.replace_config(text_content, backup_current=True)
        return {"status": "success", "message": "Configuration uploaded"}
    except (ValueError, UnicodeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/attack/preview")
async def get_attack_preview(request: dict[str, Any]):
    """Preview an unsaved attack draft without running or modifying an experiment."""
    from backend.attack_preview import preview_attack
    try:
        return await asyncio.to_thread(preview_attack, request)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        print(f"[Preview] Failed to render sample: {exc}", file=sys.stderr)
        raise HTTPException(500, "生成预览失败，请查看后端日志后重试。")


@app.post("/api/experiment/start")
async def start_experiment(request: dict[str, Any]):
    """Start a new experiment."""
    exp_name = request.get("name")
    if not exp_name:
        raise HTTPException(400, "Experiment name is required")

    config_overrides = request.get("config")
    if config_overrides is None:
        config_overrides = {}
    if not isinstance(config_overrides, dict):
        raise HTTPException(400, "Experiment config must be an object")
    config_overrides = dict(config_overrides)
    config_overrides["run-name"] = exp_name

    try:
        async with execution_lock:
            if batch_manager.get_active_task():
                raise HTTPException(409, "A batch task is running or stopping")
            await runner.start(exp_name, config_overrides, backup_config=True)
        return {"status": "started", "name": exp_name, "run_id": runner.run_id}
    except HTTPException:
        raise
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/experiment/stop")
async def stop_experiment():
    """Await owned cleanup; a timeout never releases the execution slot."""
    batch = batch_manager.get_active_task()
    if batch:
        return await cancel_batch_task(batch.batch_id)
    try:
        result = await asyncio.wait_for(asyncio.shield(runner.stop()), timeout=15)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=202, content={"status": "stopping", "message": "Cleanup is still in progress; new experiments remain blocked."})
    except Exception as e:
        raise HTTPException(500, str(e))
    if result["status"] == "cleanup_failed":
        raise HTTPException(503, result["error"])
    return result


@app.get("/api/experiment/status")
async def get_experiment_status():
    """Get current experiment runner status."""
    return await runner.get_status()


@app.get("/api/experiments")
async def list_all_experiments(
    search: str = "", status: str | None = None, dataset: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=200), offset: int = Query(default=0, ge=0),
    paginated: bool = False,
):
    """Keep the legacy array response; optional pagination includes a total."""
    try:
        root = _results_root()
        await _sync_catalog(root)
        page = await asyncio.to_thread(get_catalog().query, root, search=search, status=status,
                                       dataset=dataset, limit=limit, offset=offset)
        return page if paginated else page["items"]
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Catalog] Query failed: {e}", flush=True)
        raise HTTPException(503, "读取实验数据库失败，请稍后重试或查看后端日志。") from e


@app.get("/api/experiments/{exp_name}")
async def get_experiment_details(exp_name: str):
    """Return detailed metrics and file list for a specific experiment."""
    try:
        directory = _experiment_directory(exp_name)
        await _sync_catalog(directory.parent)
        detail = await asyncio.to_thread(get_catalog().detail, directory.parent, exp_name)
        if detail is None:
            raise HTTPException(404, "实验不存在或未找到运行记录。")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Catalog] Detail query failed: {e}", flush=True)
        raise HTTPException(503, "读取实验详情失败，请稍后重试或查看后端日志。") from e

@app.get("/api/experiments/{exp_name}/csv/latest")
async def get_latest_csv_row(exp_name: str):
    """返回主日志 CSV 的最新一行（用于前端图表实时更新）"""
    exp_dir = _experiment_directory(exp_name)
    try:
        csv_path = child_path(exp_dir, f"{exp_name}.csv")
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    if not csv_path.exists():
        raise HTTPException(404, "CSV file not found yet")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"round": None, "accuracy": None, "asr": None, "loss": None}
    
    last_row = df.iloc[-1].to_dict()
    return {
        "round": int(last_row.get("round", 0)),
        "accuracy": float(last_row.get("accuracy", 0)),
        "asr": float(last_row.get("asr", 0)),
        "loss": float(last_row.get("loss", 0))
    }

@app.post("/api/batch/parse")
async def parse_batch_config(file: UploadFile):
    """Parse uploaded JSON batch experiment configuration."""
    try:
        content = await file.read()
        experiments = config_manager.parse_batch_experiments(content)
        return {"count": len(experiments), "experiments": experiments}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/batch/run")
async def run_batch_experiments(request: dict[str, Any]):
    """Run a batch of experiments sequentially."""
    try:
        experiments = request.get("experiments", [])
        if not isinstance(experiments, list) or not experiments:
            raise HTTPException(400, "No experiments provided")
        async with execution_lock:
            if runner.is_busy():
                raise HTTPException(409, "An experiment is running or cleanup is not finished")
            if batch_manager.get_active_task():
                raise HTTPException(409, "A batch task is already running or stopping")
            baseline = config_manager.TOML_PATH.read_bytes()
            names = set()
            for idx, exp in enumerate(experiments):
                if not isinstance(exp, dict) or not isinstance(exp.get("name"), str) or not exp["name"].strip():
                    raise ValueError(f"Experiment {idx + 1}: name is required")
                config = {k: v for k, v in exp.items() if k != "name"}
                config["run-name"] = exp["name"]
                content = config_manager.build_run_config(baseline, config)
                section = tomli.loads(content.decode("utf-8"))["tool"]["flwr"]["app"]["config"]
                directory = experiment_path(config_manager.get_results_dir(section, default=RESULTS_DIR), exp["name"])
                key = exp["name"].casefold()
                if key in names or directory.exists():
                    raise FileExistsError(f'Experiment "{exp["name"]}" already exists or is repeated in this batch. Choose a new name.')
                names.add(key)
            task = batch_manager.create_task(experiments, baseline)
            task.job = asyncio.create_task(_run_batch_sequence(task))
        return {"status": "batch_started", "batch_id": task.batch_id, "total": len(experiments)}
    except HTTPException:
        raise
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")


async def _run_batch_sequence(task: BatchTask) -> None:
    """Finish owned cleanup before each next item; ordinary failures may continue."""
    try:
        task.start_time = datetime.now()
        task.status = "stopping" if task.cancel_requested else "running"
        batch_manager.update_task(task)
        task.add_log(f"[Batch] Started {task.batch_id} with {len(task.experiments)} experiments.")
        for idx, exp in enumerate(task.experiments):
            if task.cancel_requested:
                break
            exp_name = exp["name"]
            item = task.results[idx]
            config = {k: v for k, v in exp.items() if k != "name"}
            config["run-name"] = exp_name
            original_callback = runner._on_log_line

            async def forward_log(line: str):
                task.add_log(f"[{exp_name}] {line}")
                if original_callback:
                    await original_callback(line)

            runner.set_log_callback(forward_log)
            try:
                while not task.cancel_requested:
                    async with execution_lock:
                        if task.cancel_requested:
                            break
                        # Pause and start must be serialized. A pause request
                        # queued before this lock must prevent this item starting.
                        if not task.pause_requested:
                            task.status = "running"
                            task.current_index = idx
                            task.current_exp_name = exp_name
                            item.update(status="running", start_time=datetime.now().isoformat())
                            batch_manager.update_task(task)
                            task.add_log(f"[Batch] Starting {idx + 1}/{len(task.experiments)}: {exp_name}")
                            await runner.start(exp_name, config, backup_config=False, base_config=task.base_config)
                            item["run_id"] = runner.run_id
                            break
                        task.status = "paused"
                        task.current_exp_name = None
                        batch_manager.update_task(task)
                    await task.resume_event.wait()
                if task.cancel_requested:
                    item["status"] = "cancelled"
                    break
                result = await runner.wait()
                item.update(exit_code=result.get("exit_code"), error=result.get("error"),
                            flower_run_id=result.get("flower_run_id"))
                item["status"] = "completed" if result["status"] == "finished" else (
                    "cancelled" if result["status"] == "cancelled" else "failed")
                if item["status"] == "cancelled" and task.stop_current_index == idx:
                    item["error"] = "已手动结束当前实验。"
                elif item["status"] == "cancelled":
                    task.cancel_requested = True
                if item["status"] == "failed":
                    task.error = item["error"] or f"Experiment {exp_name} failed."
                if runner.is_busy():
                    task.status = "stopping"
                    task.error = result.get("error") or "Owned cleanup is not finished. Retry cancellation."
                    batch_manager.update_task(task)
                    return
            except Exception as exc:
                item.update(status="failed", error=str(exc))
                task.error = f"Experiment {exp_name} failed: {exc}"
                if runner.is_busy():
                    task.status = "stopping"
                    batch_manager.update_task(task)
                    return
            finally:
                runner.set_log_callback(original_callback)
                item["end_time"] = datetime.now().isoformat()
                task.stop_current_index = None
            task.current_exp_name = None
            task.add_log(f"[Batch] {exp_name}: {item['status']}; exit_code={item['exit_code']}")
            batch_manager.update_task(task)
        _finish_batch(task)
    except asyncio.CancelledError:
        task.cancel_requested = True
        await asyncio.shield(runner.stop())
        if runner.is_busy():
            task.status = "stopping"
            batch_manager.update_task(task)
        else:
            _finish_batch(task)
        raise
    except Exception as exc:
        task.error = f"Batch controller failed: {exc}"
        task.cancel_requested = True
        await runner.stop()
        if runner.is_busy():
            task.status = "stopping"
            batch_manager.update_task(task)
        else:
            _finish_batch(task, failed=True)


def _finish_batch(task: BatchTask, *, failed: bool = False) -> None:
    for item in task.results:
        if item["status"] in {"pending", "running"}:
            item.update(status="cancelled", error="Batch cancelled before completion.", end_time=datetime.now().isoformat())
    task.status = "failed" if failed else ("cancelled" if task.cancel_requested else (
        "failed" if any(item["status"] in {"failed", "unknown"} for item in task.results) else (
            "cancelled" if any(item["status"] == "cancelled" for item in task.results) else "completed")))
    task.pause_requested = False
    task.resume_event.set()
    task.end_time = datetime.now()
    task.current_index = len(task.experiments)
    task.current_exp_name = None
    task.add_log(f"[Batch] Batch {task.batch_id}: {task.status}.")
    batch_manager.update_task(task)


@app.post("/api/batch/{batch_id}/cancel")
async def cancel_batch_task(batch_id: str):
    task = batch_manager.get_task(batch_id)
    if task is None:
        raise HTTPException(404, "Batch task not found")
    async with execution_lock:
        if task.status not in {"pending", "running", "paused", "stopping"}:
            return task.to_dict()
        task.cancel_requested = True
        task.resume_event.set()
        task.status = "stopping"
        batch_manager.update_task(task)
    try:
        result = await asyncio.wait_for(asyncio.shield(runner.stop()), timeout=15)
        if result["status"] == "cleanup_failed":
            task.error = result["error"]
            batch_manager.update_task(task)
            raise HTTPException(503, task.error)
        if task.job and not task.job.done():
            await asyncio.wait_for(asyncio.shield(task.job), timeout=15)
        if task.status == "stopping":
            _finish_batch(task)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=202, content=task.to_dict())
    return task.to_dict()


@app.post("/api/batch/{batch_id}/pause")
async def pause_batch_task(batch_id: str):
    task = batch_manager.get_task(batch_id)
    if task is None:
        raise HTTPException(404, "Batch task not found")
    async with execution_lock:
        if task.status not in {"pending", "running", "paused"} or task.cancel_requested:
            raise HTTPException(409, "该批次已结束或正在停止，无法暂停队列。")
        task.pause_requested = True
        task.resume_event.clear()
        if not task.current_exp_name:
            task.status = "paused"
        task.add_log("[Batch] 已请求暂停队列；当前实验结束后不会启动下一项。")
        batch_manager.update_task(task)
    return task.to_dict()


@app.post("/api/batch/{batch_id}/resume")
async def resume_batch_task(batch_id: str):
    task = batch_manager.get_task(batch_id)
    if task is None:
        raise HTTPException(404, "Batch task not found")
    async with execution_lock:
        if task.status not in {"pending", "running", "paused"} or task.cancel_requested:
            raise HTTPException(409, "该批次已结束或正在停止，无法继续队列。")
        task.pause_requested = False
        task.status = "running"
        task.resume_event.set()
        task.add_log("[Batch] 队列已继续。")
        batch_manager.update_task(task)
    return task.to_dict()


@app.post("/api/batch/{batch_id}/stop-current")
async def stop_current_batch_experiment(batch_id: str, request: dict[str, Any]):
    """Bind the stop to an observed run, never accidentally stop the next item."""
    task = batch_manager.get_task(batch_id)
    if task is None:
        raise HTTPException(404, "Batch task not found")
    async with execution_lock:
        if task.status != "running" or not task.current_exp_name or task.cancel_requested:
            raise HTTPException(409, "当前没有可结束的实验，请刷新状态。")
        item = task.results[task.current_index]
        if not request.get("run_id") or request["run_id"] != item.get("run_id"):
            raise HTTPException(409, "当前实验已切换，请确认最新状态后重试。")
        task.stop_current_index = task.current_index
        task.add_log(f"[Batch] 仅结束当前实验：{task.current_exp_name}，保留后续队列。")
        # Keep the execution lock until this stop has captured/finished the owned
        # run. The controller cannot start another worker while it is held.
        try:
            result = await asyncio.wait_for(asyncio.shield(runner.stop()), timeout=15)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=202, content=task.to_dict())
        if result["status"] == "cleanup_failed":
            task.status = "stopping"
            task.error = result.get("error")
            batch_manager.update_task(task)
            raise HTTPException(503, task.error)
    return task.to_dict()

@app.get("/api/batch/status")
async def get_batch_status():
    """Get only an active batch; historical records remain queryable by ID."""
    task = batch_manager.get_active_task()
    if task is None:
        return {"status": "idle"}
    return task.to_dict()


@app.get("/api/batch/{batch_id}/status")
async def get_batch_task_status(batch_id: str):
    """Get status of a specific batch task."""
    task = batch_manager.get_task(batch_id)
    if task is None:
        raise HTTPException(404, "Batch task not found")
    return task.to_dict()

REPORT_TIMEOUT = 300
_report_directories: set[Path] = set()
_report_inputs: dict[Path, int] = {}


@asynccontextmanager
async def _report_slot(directory: Path, *, sources=()):
    key = directory.resolve()
    inputs = {source.resolve() for source in sources}
    if key in _report_directories or any(source in _report_directories for source in inputs):
        raise HTTPException(409, "该目录的图表或总结正在生成，请等待完成后再试。")
    _report_directories.add(key)
    for source in inputs:
        _report_inputs[source] = _report_inputs.get(source, 0) + 1
    try:
        yield
    finally:
        _report_directories.discard(key)
        for source in inputs:
            count = _report_inputs[source] - 1
            if count:
                _report_inputs[source] = count
            else:
                del _report_inputs[source]


@app.delete("/api/experiments/{exp_name}")
async def delete_experiment(exp_name: str, run_id: str | None = None):
    """Explicitly delete a stored run, with or without its remaining attachments."""
    directory = _experiment_directory(exp_name)
    async with execution_lock:
        owned = runner._experiment_dir
        if runner.is_busy() and owned is not None and owned.resolve() == directory.resolve():
            raise HTTPException(409, "该实验仍在运行或清理中，请等待结束后再删除。")
        batch = batch_manager.get_active_task()
        if batch and any(item.get("name") == exp_name for item in batch.experiments):
            raise HTTPException(409, "活动批量任务仍引用该实验，请等待批次结束后再删除。")
        if directory.resolve() in _report_inputs:
            raise HTTPException(409, "该实验正在用于生成总结，请等待完成后再删除。")
        async with _report_slot(directory):
            # Keep both start/report reservations until removal really finishes,
            # including when the requesting browser disconnects.
            catalog = get_catalog()
            await asyncio.to_thread(catalog.initialize)
            stored = await asyncio.to_thread(catalog.detail, directory.parent, exp_name)
            if not directory.is_dir() and stored is None:
                raise HTTPException(404, "实验不存在，请刷新列表后重试。")
            # A lost run.json must not prevent deleting a retained database
            # record. A different on-disk UUID, however, must still block it.
            metadata = read_run_metadata(directory) if directory.is_dir() else {}
            current_id = metadata.get("run_id") or (stored or {}).get("run_id")
            if run_id is not None and current_id != run_id:
                raise HTTPException(409, "实验运行标识已变化，请刷新列表后重新确认删除。")

            def remove():
                if directory.is_dir():
                    shutil.rmtree(directory)
                catalog.mark_removed(directory.parent, exp_name)

            job = asyncio.create_task(asyncio.to_thread(remove))
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                try:
                    await job
                except OSError as exc:
                    print(f"[History] Cancelled deletion failed: {exc}")
                raise
            except OSError as exc:
                raise HTTPException(500, "删除实验失败，文件可能正在被占用，请刷新列表后重试。") from exc
    return {"status": "success", "name": exp_name}


async def _run_report(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    """Wait outside the event loop; subprocess.run also reaps timed-out children."""
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    try:
        job = asyncio.create_task(asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(PROJECT_ROOT), env=env, timeout=REPORT_TIMEOUT,
        ))
        try:
            result = await asyncio.shield(job)
        except asyncio.CancelledError:
            # Keep the directory reserved until the worker exits or times out.
            # Cancelling to_thread alone does not stop its subprocess.
            try:
                await job
            except Exception as exc:
                print(f"[Report] Cancelled request worker ended: {exc}")
            raise
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, f"{label}超时，请缩小数据范围后重试。") from exc
    except OSError as exc:
        raise HTTPException(500, f"无法启动{label}程序，请检查后端运行环境。") from exc
    if result.returncode != 0:
        print(f"[Report] {label} failed ({result.returncode}): {result.stderr}")
        raise HTTPException(500, f"{label}失败，请检查实验数据或查看后端日志。")
    return result


@app.post("/api/experiments/{exp_name}/plot")
async def generate_plots_for_experiment(exp_name: str):
    """为指定实验运行 plot.py 生成图表"""
    exp_dir = _experiment_directory(exp_name)
    async with _report_slot(exp_dir):
        if not exp_dir.is_dir():
            raise HTTPException(404, "实验不存在，请刷新列表后重试。")
        detail = await asyncio.to_thread(get_experiment_detail, exp_dir)
        if detail is None or not detail["can_plot"]:
            raise HTTPException(422, "该实验缺少完整有效的轮次、ACC、ASR 或 Loss 指标，不能生成图表；仍可查看和下载运行日志。")
        result = await _run_report([sys.executable, str(PROJECT_ROOT / "scripts" / "plot.py"), str(exp_dir)], "图表生成")
    return {"status": "success", "output": result.stdout}

@app.post("/api/experiments/summarize")
async def generate_summary_report():
    """运行 summarize_results.py 生成总结报告"""
    experiments = await asyncio.to_thread(list_experiments, _results_root())
    names = [item["name"] for item in experiments if item["has_metrics"]]
    if not names:
        raise HTTPException(422, "没有可用于总结的实验指标。")
    result = await create_summary({"experiments": names})
    return {**result, "count": len(names)}
    
@app.post("/api/summarize")
async def create_summary(request: dict[str, Any]):
    """Create a new summary from selected experiments."""
    experiments = request.get("experiments", [])
    if not experiments:
        raise HTTPException(400, "No experiments selected")
    output_name = request.get("output_name")
    if not isinstance(experiments, list):
        raise HTTPException(400, "Experiments must be a list of names")
    root = _results_root()
    sources = [_experiment_directory(name) for name in experiments]
    if output_name is None:
        output_name = datetime.now().strftime("summary_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = _summary_directory(output_name)
    if output_dir.exists():
        raise HTTPException(409, "总结名称已存在，请换一个名称。")
    
    script_path = PROJECT_ROOT / "scripts" / "summarize_results.py"
    cmd = [sys.executable, str(script_path), "--results-dir", str(root), "--experiments"]
    for exp in experiments:
        cmd.append(exp)
    if output_name:
        cmd += ["--output-name", output_name]
    cmd.append("--no-plots")
    # Run plots as a separately tracked child, not an orphanable grandchild of
    # summarize_results.py when the outer subprocess hits its timeout.
    async with _report_slot(output_dir, sources=sources):
        if output_dir.exists():
            raise HTTPException(409, "总结名称已存在，请换一个名称。")
        for directory in sources:
            if not directory.is_dir():
                raise HTTPException(404, f"实验 {directory.name} 不存在。")
            detail = await asyncio.to_thread(get_experiment_detail, directory)
            if detail is None or not detail["has_metrics"]:
                raise HTTPException(422, f"实验 {directory.name} 尚无可用指标，不能加入总结。")
        result = await _run_report(cmd, "总结生成")
        await _run_report([sys.executable, str(PROJECT_ROOT / "scripts" / "plot.py"), str(output_dir)], "总结图表生成")
    return {"status": "success", "output": result.stdout, "name": output_name}

@app.get("/api/summaries")
async def list_summaries():
    """List all summary directories under results/summaries."""
    try:
        summaries_dir = summary_root(_results_root())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not summaries_dir.exists():
        return []
    summaries = []
    for d in summaries_dir.iterdir():
        try:
            d = summary_path(_results_root(), d.name)
        except ValueError:
            continue
        if d.is_dir():
            # 读取 summary.csv 获取基本信息
            try:
                csv_path = child_path(d, "summary.csv")
            except ValueError:
                continue
            if csv_path.exists():
                try:
                    df = await asyncio.to_thread(pd.read_csv, csv_path)
                except (OSError, ValueError, pd.errors.ParserError):
                    summaries.append({"name": d.name, "experiments": [], "created": None,
                                      "warnings": ["总结 CSV 无法读取，请检查文件。"]})
                    continue
                summaries.append({
                    "name": d.name,
                    "experiments": df["experiment"].tolist() if "experiment" in df.columns else [],
                    "created": datetime.fromtimestamp(os.path.getctime(d)).isoformat(),
                })
            else:
                summaries.append({"name": d.name, "experiments": [], "created": None})
    return summaries

@app.post("/api/summaries/{summary_name}/plot")
async def generate_summary_plots(summary_name: str):
    """为指定总结目录运行 plot.py 生成多实验对比图"""
    summary_dir = _summary_directory(summary_name)
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    
    detail = await asyncio.to_thread(get_summary_detail, summary_dir)
    if detail is None:
        raise HTTPException(422, "总结数据无效，无法生成图表。")
    async with _report_slot(summary_dir):
        result = await _run_report([sys.executable, str(PROJECT_ROOT / "scripts" / "plot.py"), str(summary_dir)], "总结图表生成")
    return {"status": "success", "output": result.stdout}

@app.get("/api/summaries/{summary_name}/images")
async def list_summary_images(summary_name: str):
    """列出总结目录下的所有 PNG 图片"""
    summary_dir = _summary_directory(summary_name)
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    images = [f.name for f in direct_files(summary_dir, "*.png")]
    return {"images": images}

@app.get("/api/summaries/{summary_name}/download")
async def download_summary_images(summary_name: str):
    """将总结目录下的所有图片打包为 ZIP 下载"""
    summary_dir = _summary_directory(summary_name)
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")

    png_files = direct_files(summary_dir, "*.png")
    if not png_files:
        raise HTTPException(404, "No images found in this summary")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in png_files:
            zf.write(png, arcname=png.name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers=_download_header(f"{summary_name}_images.zip")
    )

@app.get("/api/experiments/{exp_name}/images/download")
async def download_experiment_images(exp_name: str):
    """下载实验目录下所有 PNG 图片的 ZIP 压缩包"""
    exp_dir = _experiment_directory(exp_name)
    if not exp_dir.exists():
        raise HTTPException(404, "Experiment not found")

    png_files = direct_files(exp_dir, "*.png")
    if not png_files:
        raise HTTPException(404, "No images found in this experiment")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in png_files:
            zf.write(png, arcname=png.name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers=_download_header(f"{exp_name}_images.zip")
    )

@app.get("/api/experiments/{exp_name}/all/download")
async def download_experiment_all_files(exp_name: str):
    """下载实验目录下所有文件（不含子目录）的 ZIP 压缩包"""
    exp_dir = _experiment_directory(exp_name)
    if not exp_dir.exists():
        raise HTTPException(404, "Experiment not found")

    all_files = direct_files(exp_dir)
    if not all_files:
        raise HTTPException(404, "No files found in this experiment")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in all_files:
            zf.write(file, arcname=file.name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers=_download_header(f"{exp_name}_all_files.zip")
    )

@app.get("/api/summaries/{summary_name}")
async def get_summary_detail_endpoint(summary_name: str):
    """获取单个总结的详细数据"""
    summary_dir = _summary_directory(summary_name)
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    detail = await asyncio.to_thread(get_summary_detail, summary_dir)
    if detail is None:
        raise HTTPException(404, "Invalid summary data")
    return detail

@app.get("/api/config/raw")
async def get_raw_config():
    """返回 pyproject.toml 的原始文本内容"""
    try:
        with open(config_manager.TOML_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/config/raw")
async def update_raw_config(data: dict[str, Any]):
    """Validate and atomically save raw TOML; optional backup precedes replace."""
    content = data.get("content")
    if content is None:
        raise HTTPException(400, "Missing 'content' field")
    try:
        backup = parse_boolean(data.get("backup_current", False), "backup_current")
        config_manager.replace_config(content, backup_current=backup, backup_name=data.get("backup_name"))
        return {"status": "success"}
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.get("/api/config/backups")
async def list_backup_files():
    """列出所有备份文件"""
    try:
        backups = config_manager.list_backups()
        return {"backups": backups}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/config/backups")
async def create_backup_file(data: dict[str, Any] | None = None):
    """创建一份新备份，可选自定义名称和冲突处理策略"""
    custom_name = data.get("name") if data else None
    on_conflict = data.get("on_conflict", "ask") if data else "ask"
    
    try:
        backup_path = config_manager.create_backup(custom_name, on_conflict)
        if backup_path is None:
            # 重名冲突，返回特殊状态码让前端处理
            return JSONResponse(
                status_code=409,
                content={"status": "conflict", "message": "文件已存在"}
            )
        return {"status": "success", "filename": backup_path.name}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/config/backups/{filename}")
async def get_backup_content(filename: str):
    """获取指定备份文件的内容"""
    try:
        content = config_manager.read_backup_content(filename)
        return {"content": content}
    except FileNotFoundError:
        raise HTTPException(404, "Backup not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/config/backups/{filename}")
async def delete_backup_file(filename: str):
    """删除指定备份文件"""
    try:
        config_manager.delete_backup(filename)
        return {"status": "success"}
    except FileNotFoundError:
        raise HTTPException(404, "Backup not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.put("/api/config/backups/{filename}")
async def update_backup_file(filename: str, data: dict[str, Any]):
    content = data.get("content")
    if content is None:
        raise HTTPException(400, "Missing content")
    try:
        config_manager.update_backup_content(filename, content)
        return {"status": "success"}
    except FileNotFoundError:
        raise HTTPException(404, "Backup not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/config/restore")
async def restore_config_from_backup(data: dict[str, Any]):
    """从指定备份恢复配置"""
    filename = data.get("filename")
    if not filename:
        raise HTTPException(400, "Missing filename")
    backup_name = data.get("backup_name")
    try:
        backup_current = parse_boolean(data.get("backup_current", True), "backup_current")
        config_manager.restore_from_backup(filename, backup_current, backup_name)
        return {"status": "success"}
    except FileNotFoundError:
        raise HTTPException(404, "Backup not found")
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.post("/api/config/backups/content")
async def create_backup_from_content(data: dict[str, Any]):
    """将指定内容保存为备份文件（不修改 pyproject.toml）"""
    content = data.get("content")
    if content is None:
        raise HTTPException(400, "Missing 'content' field")
    custom_name = data.get("name")
    on_conflict = data.get("on_conflict", "ask")
    
    try:
        backup_path = config_manager.create_backup_from_content(content, custom_name, on_conflict)
        if backup_path is None:
            return JSONResponse(
                status_code=409,
                content={"status": "conflict", "message": "文件已存在"}
            )
        return {"status": "success", "filename": backup_path.name}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

# ------------------------------
# Main entry point
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)