"""
FastAPI backend for FL backdoor experiment platform.
"""

import asyncio
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
from watchfiles import awatch
from fastapi.responses import StreamingResponse, JSONResponse


# Add project root to sys.path for importing fl_backdoor modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config_manager
from backend.experiment_runner import ExperimentRunner
from backend.results_scanner import (
    get_experiment_detail,
    list_experiments,
    get_summary_detail,
)

# ------------------------------
# FastAPI app setup
# ------------------------------
app = FastAPI(title="FL Backdoor Platform API", version="1.0.0")

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

# Static files for results
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ------------------------------
# Global experiment runner instance
# ------------------------------
runner = ExperimentRunner()


# ------------------------------
# WebSocket for real-time logs
# ------------------------------
@app.websocket("/ws/logs/{exp_name}")
async def websocket_logs(websocket: WebSocket, exp_name: str):
    await websocket.accept()

    exp_dir = RESULTS_DIR / exp_name
    log_file = exp_dir / "run.log"

    # Set callback to forward subprocess stdout to WebSocket
    async def send_log_line(line: str):
        print(f"[WS] Sending line: {line[:80]}")
        await websocket.send_text(line)

    runner.set_log_callback(send_log_line)

    # Send initial status
    await websocket.send_json({"type": "status", "data": await runner.get_status()})

    # Watch for CSV file changes (metrics updates)
    watch_task = asyncio.create_task(_watch_csv_files(websocket, exp_dir))

    # If log file exists, start streaming its new lines
    if log_file.exists():
        await _stream_log_file(websocket, log_file)
    else:
        await websocket.send_text("[System] Waiting for log file...")

    # Keep connection alive until client disconnects
    try:
        while True:
            await asyncio.sleep(1)
            # Could also send periodic status updates here
    except WebSocketDisconnect:
        watch_task.cancel()
        runner.set_log_callback(None)  # Clear callback
        print(f"WebSocket disconnected for {exp_name}")


async def _stream_log_file(websocket: WebSocket, log_path: Path) -> None:
    """Stream new lines appended to log file."""
    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(0, 2)  # Seek to end
        while True:
            line = f.readline()
            if line:
                await websocket.send_text(line.rstrip())
            else:
                await asyncio.sleep(0.1)


async def _watch_csv_files(websocket: WebSocket, exp_dir: Path) -> None:
    """Watch CSV files for changes and send parsed updates."""
    async for changes in awatch(exp_dir):
        for change_type, path_str in changes:
            if not path_str.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(path_str)
                if df.empty:
                    continue
                last_row = df.iloc[-1].to_dict()
                await websocket.send_json({
                    "type": "csv_update",
                    "file": Path(path_str).name,
                    "data": last_row,
                })
            except Exception as e:
                print(f"Error reading CSV {path_str}: {e}")


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
async def update_current_config(updates: dict[str, Any]):
    """Update configuration in pyproject.toml."""
    try:
        config_manager.backup_config()
        config_manager.update_config(updates)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/config/upload")
async def upload_config_file(file: UploadFile):
    """Replace entire pyproject.toml with uploaded file."""
    if not file.filename.endswith(".toml"):
        raise HTTPException(400, "Only .toml files are accepted")
    try:
        content = await file.read()
        # 将 Windows 换行符 \r\n 统一转换为 \n
        text_content = content.decode('utf-8').replace('\r\n', '\n')
        config_manager.backup_config()
        with open(config_manager.TOML_PATH, "w", encoding='utf-8', newline='') as f:
            f.write(text_content)
        return {"status": "success", "message": "Configuration uploaded"}
    except Exception as e:
        config_manager.restore_config()
        raise HTTPException(500, str(e))


@app.post("/api/experiment/start")
async def start_experiment(request: dict[str, Any]):
    """Start a new experiment."""
    exp_name = request.get("name")
    if not exp_name:
        raise HTTPException(400, "Experiment name is required")

    config_overrides = request.get("config", {})
    config_overrides["run-name"] = exp_name

    try:
        await runner.start(exp_name, config_overrides, backup_config=True)
        return {"status": "started", "name": exp_name}
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/experiment/stop")
async def stop_experiment():
    """Stop the currently running experiment."""
    await runner.stop()
    return {"status": "stopped"}


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
    experiments = request.get("experiments", [])
    if not experiments:
        raise HTTPException(400, "No experiments provided")

    if runner.is_running():
        raise HTTPException(409, "An experiment is already running")

    asyncio.create_task(_run_batch_sequence(experiments))
    return {"status": "batch_started", "total": len(experiments)}


async def _run_batch_sequence(experiments: list[dict[str, Any]]) -> None:
    """Internal: run experiments sequentially."""
    for exp in experiments:
        exp_name = exp.pop("name")
        try:
            await runner.start(exp_name, exp, backup_config=True)
            while runner.is_running():
                await asyncio.sleep(2)
        except Exception as e:
            print(f"Batch experiment {exp_name} failed: {e}")
        finally:
            await runner.stop()
    print("Batch experiments finished")



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
async def update_raw_config(data: dict[str, str]):
    """用原始文本覆盖 pyproject.toml（不自动备份，备份由前端显式触发）"""
    content = data.get("content")
    if content is None:
        raise HTTPException(400, "Missing 'content' field")
    try:
        # 统一换行符
        content = content.replace('\r\n', '\n')
        with open(config_manager.TOML_PATH, "w", encoding="utf-8", newline='') as f:
            f.write(content)
        return {"status": "success"}
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
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/config/restore")
async def restore_config_from_backup(data: dict[str, Any]):
    """从指定备份恢复配置"""
    filename = data.get("filename")
    if not filename:
        raise HTTPException(400, "Missing filename")
    backup_current = data.get("backup_current", True)
    backup_name = data.get("backup_name")
    try:
        config_manager.restore_from_backup(filename, backup_current, backup_name)
        return {"status": "success"}
    except FileNotFoundError:
        raise HTTPException(404, "Backup not found")
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
    except Exception as e:
        raise HTTPException(500, str(e))

# ------------------------------
# Main entry point
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)