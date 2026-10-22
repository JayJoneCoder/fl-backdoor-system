"""Download only the two large ML wheels; keep bundled dependencies offline."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request

from fl_backdoor.utils.app_paths import resource_root, state_root, user_root


# Every source exposes the same flat wheel layout:
#   <base>/<build>/<name>-<version>+<build>-cp311-cp311-win_amd64.whl
# Aliyun has no /<package>/ index page, so the installer downloads the two
# exact wheels itself (with byte progress) and then pip-installs local files.
SOURCES = {
    "sjtu": {"label": "国内：上海交通大学 PyTorch 镜像（第三方）",
             "url": "https://mirror.sjtu.edu.cn/pytorch-wheels"},
    "sjtug": {"label": "国内：上海交大思源镜像（备用线路）",
              "url": "https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels"},
    "aliyun": {"label": "国内：阿里云 PyTorch 镜像（第三方）",
               "url": "https://mirrors.aliyun.com/pytorch-wheels"},
    "official": {"label": "国外：PyTorch 官方源",
                 "url": "https://download.pytorch.org/whl"},
    "official-cdn": {"label": "国外：PyTorch 官方 CDN",
                     "url": "https://download-r2.pytorch.org/whl"},
}
BUILDS = {"cpu": "CPU 精简版", "cu126": "NVIDIA GPU / CUDA 12.6", "cu128": "NVIDIA GPU / CUDA 12.8（新显卡优先尝试）"}
VERSIONS = {"torch": "2.8.0", "torchvision": "0.23.0"}
SPEED_SAMPLE_BYTES = 512 * 1024


def runtime_environment(target: Path | None = None) -> dict:
    env = os.environ.copy()
    # Never accidentally import another user's globally installed packages, and
    # never write __pycache__ into the (possibly read-only) installation folder.
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    if target:
        env["FL_BACKDOOR_TORCH_DIR"] = str(target)
    else:
        env.pop("FL_BACKDOOR_TORCH_DIR", None)
    paths = ([str(target)] if target else []) + [str(resource_root())]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def active_runtime() -> Path | None:
    path = state_root() / "runtime.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("运行环境记录无效，请重新配置 PyTorch。")
    # Only accept a direct versioned child of the private runtime directory.
    root = (user_root() / "runtimes").resolve()
    name = record.get("directory")
    if not isinstance(name, str) or not name.startswith("torch-") or Path(name).name != name:
        raise ValueError("运行环境记录无效，请重新配置 PyTorch。")
    candidate = root / name
    target = candidate.resolve()
    if target.parent != root or candidate.is_symlink() or not target.is_dir():
        raise ValueError("运行环境记录无效，请重新配置 PyTorch。")
    return target


def probe(target: Path | None = None, timeout: int = 90) -> dict:
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "fl_backdoor.utils.device"],
                                cwd=resource_root(), env=runtime_environment(target), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:] or "探测进程启动失败")
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired, RuntimeError) as exc:
        return {"cpu_available": False, "gpu_available": False, "warnings": [str(exc)],
                "guidance": ["请重新配置运行环境，检查启动器日志及 VC++ 运行库。"]}


def wheel_filename(name: str, version: str, build: str) -> str:
    return f"{name}-{version}+{build}-cp311-cp311-win_amd64.whl"


def wheel_url(source: str, build: str, name: str, version: str) -> str:
    base = SOURCES[source]["url"].rstrip("/")
    return f"{base}/{build}/{wheel_filename(name, version, build)}"


def install_command(target: Path, wheels: list[Path]) -> list[str]:
    """Install the two already downloaded wheels without any network access."""
    if not wheels:
        raise ValueError("缺少已下载的 PyTorch wheel。")
    return [sys.executable, "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
            "--no-input", "--only-binary=:all:", "--no-deps", "--no-warn-script-location",
            "--progress-bar", "off", "--no-index", "--target", str(target),
            *[str(path) for path in wheels]]


def _response_size(response) -> int | None:
    headers = getattr(response, "headers", None)
    if headers:
        content_range = headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total)
        content_length = headers.get("Content-Length")
        if content_length and content_length.isdigit():
            return int(content_length)
    return None


def check_source(source: str, build: str) -> dict[str, int | None]:
    """Probe the exact Windows wheels and return their byte sizes."""
    if source not in SOURCES or build not in BUILDS:
        raise ValueError("请选择内置的下载源和计算平台。")
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 11):
        raise RuntimeError("当前发行适配支持 Windows x64 / Python 3.11；其他平台需重新验证并提供配套发行环境。")

    sizes: dict[str, int | None] = {}
    for name, version in VERSIONS.items():
        url = wheel_url(source, build, name, version)
        request = urllib.request.Request(url, headers={
            "Range": "bytes=0-0", "User-Agent": "LianAnZhiDun/1.0 (PyTorch runtime setup)"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status not in (200, 206):
                    raise RuntimeError(f"所选源返回 HTTP {response.status}，请换源重试。")
                sizes[name] = _response_size(response)
                response.read(1)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise RuntimeError(
                    f"所选源未提供 {name}-{version}+{build} 的 Windows wheel（HTTP {exc.code}），"
                    "可能尚未同步。请切换其它源，不能用其他构建冒充。") from exc
            raise RuntimeError(f"访问所选 PyTorch 源失败：HTTP {exc.code}。请检查网络或换源重试。") from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"无法访问所选 PyTorch 源：{exc}。请检查网络或换源重试。") from exc
    return sizes


def download_wheel(url: str, destination: Path, expected: int | None, progress, timeout: int = 60) -> int:
    """Stream one wheel to disk, reporting bytes so the setup page can show progress."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={
        "User-Agent": "LianAnZhiDun/1.0 (PyTorch runtime download)"})
    done = 0
    last = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 0.2:
                progress(done)
                last = now
    if expected is not None and done != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"下载大小不匹配：期望 {expected} 字节，实际 {done} 字节。请换源重试。")
    progress(done)
    part.replace(destination)
    return done



def measure_source(source: str, build: str, *, timeout: float = 12) -> dict:
    """Latency + rough throughput from a 512 KiB ranged download."""
    if source not in SOURCES or build not in BUILDS:
        raise ValueError("请选择内置的下载源和计算平台。")
    url = wheel_url(source, build, "torchvision", VERSIONS["torchvision"])
    request = urllib.request.Request(url, headers={
        "Range": f"bytes=0-{SPEED_SAMPLE_BYTES - 1}",
        "User-Agent": "LianAnZhiDun/1.0 (PyTorch speed test)"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            latency = time.monotonic() - started
            data = response.read(SPEED_SAMPLE_BYTES)
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the UI
        return {"ok": False, "latency_ms": None, "mbps": None, "bytes": 0,
                "error": f"{type(exc).__name__}: {exc}"}
    elapsed = max(time.monotonic() - started, 0.001)
    if not data:
        return {"ok": False, "latency_ms": None, "mbps": None, "bytes": 0, "error": "没有收到数据"}
    return {"ok": True, "latency_ms": round(latency * 1000),
            "mbps": round(len(data) * 8 / elapsed / 1_000_000, 2),
            "bytes": len(data), "error": ""}


def speed_test(build: str, *, timeout: float = 12) -> dict:
    """Measure every built-in source concurrently and rank the fastest one."""
    if build not in BUILDS:
        raise ValueError("请选择内置的计算平台。")
    with ThreadPoolExecutor(max_workers=max(1, len(SOURCES))) as pool:
        futures = {name: pool.submit(measure_source, name, build, timeout=timeout) for name in SOURCES}
        results = {name: future.result() for name, future in futures.items()}

    def rank(item):
        name, value = item
        return (0 if value["ok"] else 1, value.get("latency_ms") or 10**9, -(value.get("mbps") or 0))

    ordered = [name for name, _ in sorted(results.items(), key=rank)]
    fastest = ordered[0] if ordered and results[ordered[0]]["ok"] else ""
    return {"build": build, "results": results, "fastest": fastest,
            "tested_at": time.strftime("%H:%M:%S")}



def validate_install(report: dict, target: Path, build: str) -> None:
    if not report.get("cpu_available"):
        raise RuntimeError("安装后 CPU 运算自检失败：" + "；".join(report.get("warnings", [])))
    for name, version in VERSIONS.items():
        if report.get(name) != f"{version}+{build}" or not (target / name / "__init__.py").is_file():
            raise RuntimeError(f"下载的 {name} 版本或安装位置不正确，拒绝激活。")
    if build != "cpu" and not report.get("gpu_available"):
        raise RuntimeError("CUDA 安装未通过显卡实际运算检查：" + "；".join(report.get("warnings", []) + report.get("guidance", [])))


def install_runtime(source: str, build: str, log=print, progress=None) -> dict:
    """Transactionally download the two wheels, install them and activate the runtime.

    ``progress(phase, downloaded_bytes, total_bytes)`` is called from the install
    thread; ``phase`` is one of check/download/install/verify/done/failed.
    """
    observer = log
    logfile = user_root() / "logs" / "runtime-install.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    def log(line):
        observer(line)
        with logfile.open("a", encoding="utf-8") as stream:
            stream.write(str(line) + "\n")

    def report(phase, done, total):
        if progress is not None:
            try:
                progress(phase, done, total)
            except Exception:  # never let UI bookkeeping break the install
                pass

    root = user_root() / "runtimes"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"torch-{build}-{uuid.uuid4().hex}"
    if shutil.disk_usage(root).free < (5 if build == "cpu" else 16) * 1024**3:
        raise RuntimeError("磁盘空间不足：CPU 安装至少预留 5 GiB，CUDA 安装至少预留 16 GiB（含下载缓存；缓存在用户目录 cache\\wheels，成功后可清理）。")
    log("安装到私有用户目录；仅下载 torch 和 torchvision，不修改系统 Python。")
    report("check", 0, None)
    log("正在检查所选源是否具有精确匹配的 Windows wheel…")
    sizes = check_source(source, build)
    total = sum(size for size in sizes.values() if size) or None
    for name, version in VERSIONS.items():
        size = sizes.get(name)
        log(f"源已提供 {name}-{version}+{build}" + (f"（{size / 1024**2:.1f} MB）" if size else ""))

    cache = user_root() / "cache" / "wheels" / build
    cache.mkdir(parents=True, exist_ok=True)
    wheels: list[Path] = []
    downloaded = 0
    report("download", 0, total)
    for name, version in VERSIONS.items():
        path = cache / wheel_filename(name, version, build)
        expected = sizes.get(name)
        if path.is_file() and path.stat().st_size > 0 and (expected is None or path.stat().st_size == expected):
            log(f"复用已下载的 {path.name}")
            downloaded += path.stat().st_size
            report("download", downloaded, total)
        else:
            url = wheel_url(source, build, name, version)
            log(f"下载 {path.name} …")
            base = downloaded
            size = download_wheel(url, path, expected,
                                  lambda count, base=base: report("download", base + count, total))
            downloaded += size
            log(f"已下载 {path.name}（{size / 1024**2:.1f} MB）")
        wheels.append(path)

    report("install", downloaded, total)
    command = install_command(target, wheels)
    from backend.owned_processes import OwnedProcessTree

    try:
        owner = OwnedProcessTree()
        try:
            process = owner.spawn(resource_root() / "desktop" / "pip_entry.py", command[3:],
                                  cwd=str(resource_root()), env=runtime_environment(),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace",
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            for line in process.stdout:
                log(line.rstrip())
            if process.wait():
                raise RuntimeError("安装失败。请检查磁盘空间和安全软件；已下载的 wheel 会复用，无需重新下载。")
        finally:
            owner.cleanup()
        report("verify", downloaded, total)
        report_verified = probe(target)
        validate_install(report_verified, target, build)
        state_root().mkdir(parents=True, exist_ok=True)
        temporary = state_root() / f".runtime-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps({"directory": target.name, "source": source, "build": build}), encoding="utf-8")
            os.replace(temporary, state_root() / "runtime.json")
        finally:
            temporary.unlink(missing_ok=True)
        report("done", downloaded, total)
        log("安装及实际运算检查通过，运行环境已激活。")
        return report_verified
    except BaseException as exc:
        log(f"安装失败：{exc}")
        report("failed", downloaded, total)
        shutil.rmtree(target, ignore_errors=True)
        raise