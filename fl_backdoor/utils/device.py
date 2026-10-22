"""CPU/CUDA diagnostics. Importing this module never imports PyTorch."""

import csv
import io
import json
import os
import shutil
import subprocess
import sys


def nvidia_status() -> dict:
    command = shutil.which("nvidia-smi")
    if not command and os.name == "nt":
        candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        command = candidate if os.path.isfile(candidate) else None
    if not command:
        return {"detected": False, "devices": [], "detail": "未找到 nvidia-smi；无法仅据此排除 NVIDIA 硬件。"}
    try:
        result = subprocess.run([command, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "NVIDIA 驱动查询失败")
        devices = [{"name": row[0].strip(), "driver": row[1].strip()}
                   for row in csv.reader(io.StringIO(result.stdout)) if len(row) >= 2]
        return {"detected": bool(devices), "devices": devices, "detail": "驱动查询成功"}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"detected": False, "devices": [], "detail": str(exc)}


def compute_probe(torch, device: str) -> None:
    """Exercise convolution, cuDNN, backward, NumPy bridge and torchvision ops."""
    import numpy as np
    from torchvision.ops import nms

    layer = torch.nn.Conv2d(1, 2, 3).to(device)
    x = torch.from_numpy(np.ones((2, 1, 8, 8), dtype=np.float32)).to(device)
    loss = layer(x).square().mean()
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError("计算探针返回非有限值")
    nms(torch.tensor([[0., 0., 2., 2.]], device=device), torch.tensor([1.], device=device), 0.5)
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def probe_environment() -> dict:
    result = {"python": sys.version.split()[0], "executable": sys.executable,
              "torch": None, "torchvision": None, "cuda_build": None,
              "cpu_available": False, "gpu_available": False, "devices": [],
              "nvidia": nvidia_status(), "warnings": [], "guidance": []}
    try:
        import torch
        import torchvision

        result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_build=torch.version.cuda)
        torch.set_num_threads(1)
        compute_probe(torch, "cpu")
        result["cpu_available"] = True
    except Exception as exc:
        result["warnings"].append(f"PyTorch/torchvision 缺失或 CPU 计算自检失败：{exc}")
        result["guidance"].append("在启动器的运行环境窗口重新安装配套的 PyTorch 2.8.0 / torchvision 0.23.0；检查磁盘空间和 VC++ 运行库。")
        return result
    if not torch.version.cuda:
        result["warnings"].append("当前安装的是 CPU 版 PyTorch，不能使用 CUDA。")
        result["guidance"].append("如需 NVIDIA GPU，请关闭平台，在启动器选择 CUDA 版 PyTorch；只安装系统 CUDA Toolkit 不能启用 CPU wheel 的 GPU 功能。")
        return result
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() 返回 False")
        # Ray is deliberately limited to the first visible GPU in desktop runs.
        props = torch.cuda.get_device_properties(0)
        result["devices"].append({"index": 0, "name": props.name,
                                  "capability": f"{props.major}.{props.minor}",
                                  "memory_mib": props.total_memory // (1024 * 1024)})
        result["arch_list"] = torch.cuda.get_arch_list()
        compute_probe(torch, "cuda:0")
        result["gpu_available"] = True
    except Exception as exc:
        result["warnings"].append(f"CUDA 实际运算自检失败：{exc}")
        result["guidance"].extend([
            "更新 NVIDIA 驱动并重启，检查显卡是否被禁用或被 CUDA_VISIBLE_DEVICES 屏蔽。",
            "若提示 no kernel image / 不支持的架构，请换用兼容的 CUDA 构建；较新显卡可尝试 CUDA 12.8。",
            "不需要单独安装 CUDA Toolkit；仍不兼容可明确选择 CPU 模式。",
        ])
    return result


def select_device(mode: str, report: dict) -> str:
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("运行设备必须是 auto、cpu 或 cuda。")
    if not report["cpu_available"]:
        raise RuntimeError("PyTorch 运行环境未通过自检，请从启动器修复环境。")
    if mode == "cuda" and not report["gpu_available"]:
        raise RuntimeError("强制 GPU 模式不可用：" + "；".join(report["warnings"] + report["guidance"]))
    return "cuda" if mode != "cpu" and report["gpu_available"] else "cpu"


def training_device(role: str):
    """Managed runs never silently switch away from the preflight decision."""
    import torch

    mode = os.environ.get("FL_BACKDOOR_DEVICE", "auto")
    if mode not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"非法设备策略：{mode}")
    threads = int(os.environ.get("FL_BACKDOOR_CPU_THREADS", "1"))
    torch.set_num_threads(max(1, threads))
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已选择 GPU，但训练进程不可见 CUDA；请检查 Ray GPU 配额，禁止静默回退。")
    device = torch.device("cuda:0" if mode == "cuda" or (mode == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {role}: actual={device}, torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    return device


if __name__ == "__main__":
    print(json.dumps(probe_environment(), ensure_ascii=False))