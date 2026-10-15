"""
FastAPI backend for FL backdoor experiment platform.
"""

import asyncio
from contextlib import asynccontextmanager
import subprocess
from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Any
import zipfile
import io
import re
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse


# Add project root to sys.path for importing fl_backdoor modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config_manager
from backend.config_validation import parse_boolean
from backend.experiment_runner import ExperimentRunner
from backend.log_stream import LogFileReader, MetricFileReader, read_total_rounds
from backend.results_scanner import (
    get_experiment_detail,
    list_experiments,
    get_summary_detail,
)
from backend.batch_manager import batch_manager, BatchTask

# ------------------------------
# FastAPI app setup
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        # Normal backend shutdown must not abandon an owned worker/cleanup task.
        task = batch_manager.get_active_task()
        if task:
            task.cancel_requested = True
        await runner.stop()
        if task and task.job and not task.job.done():
            await asyncio.shield(task.job)


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
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ------------------------------
# Global experiment runner instance
# ------------------------------
runner = ExperimentRunner()
execution_lock = asyncio.Lock()


# ------------------------------
# WebSocket for real-time logs
# ------------------------------
@app.websocket("/ws/logs/{exp_name}")
async def websocket_logs(websocket: WebSocket, exp_name: str):
    await websocket.accept()

    exp_dir = RESULTS_DIR / exp_name
    if not (exp_dir / "run.log").exists():
        await websocket.send_text("[System] Waiting for log file...")

    # One log source per connection. Never replace the runner's batch callback.
    send_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(_stream_log_file(websocket, exp_dir / "run.log", send_lock)),
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
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await websocket.send_text(line)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "status", "data": task.to_dict()})
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
        generation, _ = runner.get_log_context(log_path.parent.name)
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
        generation, started_at_ns = runner.get_log_context(exp_dir.name)
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
        if status.get("exp_name") != exp_dir.name:
            status = {"status": "idle", "exp_name": exp_dir.name}
        if not status.get("total_rounds"):
            status["total_rounds"] = await asyncio.to_thread(
                read_total_rounds, exp_dir / "config_snapshot.toml"
            )
        status["log_generation"] = runner.get_log_context(exp_dir.name)[0]
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
        return {"status": "started", "name": exp_name}
    except HTTPException:
        raise
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
async def list_all_experiments():
    """Return list of all experiments with summary metrics."""
    try:
        experiments = list_experiments(RESULTS_DIR)
        return experiments
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/experiments/{exp_name}")
async def get_experiment_details(exp_name: str):
    """Return detailed metrics and file list for a specific experiment."""
    try:
        detail = get_experiment_detail(RESULTS_DIR / exp_name)
        if detail is None:
            raise HTTPException(404, "Experiment not found or no valid data")
        return detail
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/experiments/{exp_name}/csv/latest")
async def get_latest_csv_row(exp_name: str):
    """返回主日志 CSV 的最新一行（用于前端图表实时更新）"""
    exp_dir = RESULTS_DIR / exp_name
    csv_path = exp_dir / f"{exp_name}.csv"
    
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
            for idx, exp in enumerate(experiments):
                if not isinstance(exp, dict) or not isinstance(exp.get("name"), str) or not exp["name"].strip():
                    raise ValueError(f"Experiment {idx + 1}: name is required")
                config = {k: v for k, v in exp.items() if k != "name"}
                config["run-name"] = exp["name"]
                config_manager.build_run_config(baseline, config)
            task = batch_manager.create_task(experiments, baseline)
            task.job = asyncio.create_task(_run_batch_sequence(task))
        return {"status": "batch_started", "batch_id": task.batch_id, "total": len(experiments)}
    except HTTPException:
        raise
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
            task.current_index = idx
            task.current_exp_name = exp_name
            item.update(status="running", start_time=datetime.now().isoformat())
            batch_manager.update_task(task)
            config = {k: v for k, v in exp.items() if k != "name"}
            config["run-name"] = exp_name
            original_callback = runner._on_log_line

            async def forward_log(line: str):
                task.add_log(f"[{exp_name}] {line}")
                if original_callback:
                    await original_callback(line)

            runner.set_log_callback(forward_log)
            try:
                task.add_log(f"[Batch] Starting {idx + 1}/{len(task.experiments)}: {exp_name}")
                async with execution_lock:
                    if task.cancel_requested:
                        item["status"] = "cancelled"
                        break
                    await runner.start(exp_name, config, backup_config=False, base_config=task.base_config)
                result = await runner.wait()
                item.update(exit_code=result.get("exit_code"), error=result.get("error"))
                item["status"] = "completed" if result["status"] == "finished" else (
                    "cancelled" if result["status"] == "cancelled" else "failed")
                if item["status"] == "cancelled":
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
        "completed" if all(item["status"] == "completed" for item in task.results) else "failed"))
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
        if task.status not in {"pending", "running", "stopping"}:
            return task.to_dict()
        task.cancel_requested = True
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

@app.get("/api/batch/status")
async def get_batch_status():
    """Get current batch task status."""
    task = batch_manager.get_active_task() or batch_manager.get_latest_task()
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

@app.post("/api/experiments/{exp_name}/plot")
async def generate_plots_for_experiment(exp_name: str):
    """为指定实验运行 plot.py 生成图表"""
    exp_dir = RESULTS_DIR / exp_name
    if not exp_dir.exists():
        raise HTTPException(404, "Experiment not found")
    
    script_path = PROJECT_ROOT / "scripts" / "plot.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(exp_dir)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Plot generation failed: {result.stderr}")
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/experiments/summarize")
async def generate_summary_report():
    """运行 summarize_results.py 生成总结报告"""
    script_path = PROJECT_ROOT / "scripts" / "summarize_results.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Summary generation failed: {result.stderr}")
        # 读取生成的 summary.csv 行数
        summary_csv = RESULTS_DIR / "summary.csv"
        count = 0
        if summary_csv.exists():
            import pandas as pd
            df = pd.read_csv(summary_csv)
            count = len(df)
        return {"status": "success", "output": result.stdout, "count": count}
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.post("/api/summarize")
async def create_summary(request: dict[str, Any]):
    """Create a new summary from selected experiments."""
    experiments = request.get("experiments", [])
    if not experiments:
        raise HTTPException(400, "No experiments selected")
    output_name = request.get("output_name")
    
    script_path = PROJECT_ROOT / "scripts" / "summarize_results.py"
    cmd = [sys.executable, str(script_path), "--experiments"]
    for exp in experiments:
        cmd.append(exp)
    if output_name:
        cmd += ["--output-name", output_name]
    
    print(f"[DEBUG] Summarize command: {cmd}")  # 便于调试
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Summary failed: {result.stderr}")
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/summaries")
async def list_summaries():
    """List all summary directories under results/summaries."""
    summaries_dir = RESULTS_DIR / "summaries"
    if not summaries_dir.exists():
        return []
    summaries = []
    for d in summaries_dir.iterdir():
        if d.is_dir():
            # 读取 summary.csv 获取基本信息
            csv_path = d / "summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
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
    summary_dir = RESULTS_DIR / "summaries" / summary_name
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    
    script_path = PROJECT_ROOT / "scripts" / "plot.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(summary_dir)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Plot generation failed: {result.stderr}")
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/summaries/{summary_name}/images")
async def list_summary_images(summary_name: str):
    """列出总结目录下的所有 PNG 图片"""
    summary_dir = RESULTS_DIR / "summaries" / summary_name
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    images = [f.name for f in summary_dir.glob("*.png")]
    return {"images": images}

@app.get("/api/summaries/{summary_name}/download")
async def download_summary_images(summary_name: str):
    """将总结目录下的所有图片打包为 ZIP 下载"""
    summary_dir = RESULTS_DIR / "summaries" / summary_name
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")

    png_files = list(summary_dir.glob("*.png"))
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
        headers={"Content-Disposition": f"attachment; filename={summary_name}_images.zip"}
    )

@app.get("/api/experiments/{exp_name}/images/download")
async def download_experiment_images(exp_name: str):
    """下载实验目录下所有 PNG 图片的 ZIP 压缩包"""
    exp_dir = RESULTS_DIR / exp_name
    if not exp_dir.exists():
        raise HTTPException(404, "Experiment not found")

    png_files = list(exp_dir.glob("*.png"))
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
        headers={"Content-Disposition": f"attachment; filename={exp_name}_images.zip"}
    )

@app.get("/api/experiments/{exp_name}/all/download")
async def download_experiment_all_files(exp_name: str):
    """下载实验目录下所有文件（不含子目录）的 ZIP 压缩包"""
    exp_dir = RESULTS_DIR / exp_name
    if not exp_dir.exists():
        raise HTTPException(404, "Experiment not found")

    all_files = [f for f in exp_dir.iterdir() if f.is_file()]
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
        headers={"Content-Disposition": f"attachment; filename={exp_name}_all_files.zip"}
    )

@app.get("/api/summaries/{summary_name}")
async def get_summary_detail_endpoint(summary_name: str):
    """获取单个总结的详细数据"""
    summary_dir = RESULTS_DIR / "summaries" / summary_name
    if not summary_dir.exists():
        raise HTTPException(404, "Summary not found")
    detail = get_summary_detail(summary_dir)
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