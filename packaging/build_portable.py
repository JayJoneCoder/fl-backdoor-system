"""Build a relocatable Windows application directory from pinned inputs.

Inputs (all offline after one preparation run):
* ``packaging/python-runtime.json`` - pinned python-build-standalone interpreter
* ``packaging/requirements-lock.txt`` - 95 bundled packages (no torch)
* ``build/wheelhouse`` - matching wheels (``--download-wheels`` fills it)
* ``frontend/dist`` and ``data`` in the repository

Usage:
    python packaging/build_portable.py --download-wheels --build-frontend --zip
    python packaging/build_portable.py --reuse          # iterate on verification

Output: ``build/stage`` (the installed layout) and optional ``dist/*.zip``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build hosts
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = Path(__file__).resolve().parent
APP_DIRS = ("backend", "desktop", "fl_backdoor", "scripts", "docs")
APP_FILES = ("pyproject.toml", "README.md", "LICENSE", "sitecustomize.py")
REQUIRED_DATA = (
    "MNIST/raw/train-images-idx3-ubyte",
    "MNIST/raw/t10k-images-idx3-ubyte",
    "cifar-10-batches-py/data_batch_1",
    "cifar-10-batches-py/test_batch",
)
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache")


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(urls, target: Path, expected: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected:
        log(f"复用已校验的 {target.name}")
        return target
    temporary = target.with_name(target.name + ".part")
    problems = []
    for url in urls:
        log(f"下载 {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            actual = sha256(temporary)
            if actual != expected:
                problems.append(f"{url} -> SHA256 {actual}")
                continue
        except Exception as exc:  # network errors must fall back to the next mirror
            problems.append(f"{url} -> {type(exc).__name__}: {exc}")
            continue
        finally:
            if temporary.exists() and sha256(temporary) != expected:
                temporary.unlink(missing_ok=True)
        temporary.replace(target)
        return target
    raise SystemExit("私有解释器下载失败或校验不通过：\n" + "\n".join(problems))


def extract_python(archive: Path, stage: Path) -> Path:
    log("解压私有解释器")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(stage, filter="data")
        except TypeError:  # Python < 3.11.4
            tar.extractall(stage)
    python_dir = stage / "python"
    if not (python_dir / "python.exe").is_file() or not (python_dir / "pythonw.exe").is_file():
        raise SystemExit("解压后找不到 python/python.exe 与 pythonw.exe，解释器布局可能已变化。")
    return python_dir


def run(command: list[str], *, cwd: Path = ROOT, env=None) -> None:
    log("执行：" + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def download_wheels(python_dir: Path, wheelhouse: Path, lock: Path, index_url: str) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    run([str(python_dir / "python.exe"), "-m", "pip", "download", "-r", str(lock), "-d", str(wheelhouse),
         "--only-binary=:all:", "--no-deps", "--platform", "win_amd64", "--implementation", "cp",
         "--python-version", "311", "--abi", "cp311", "--index-url", index_url,
         "--timeout", "60", "--retries", "3",
         "--disable-pip-version-check", "--progress-bar", "off"])


def install_dependencies(python_dir: Path, wheelhouse: Path, lock: Path) -> None:
    log("离线安装随包依赖")
    run([str(python_dir / "python.exe"), "-m", "pip", "--isolated", "install", "--no-index",
         "--find-links", str(wheelhouse), "--disable-pip-version-check", "--no-input",
         "--no-warn-script-location", "-r", str(lock)])


def remove_console_launchers(python_dir: Path) -> int:
    """Remove files that embed the build path: .pdb symbols and Scripts launchers."""
    removed = 0
    for pdb in python_dir.rglob("*.pdb"):
        pdb.unlink()
        removed += 1
    scripts = python_dir / "Scripts"
    if scripts.is_dir():
        for entry in scripts.iterdir():
            if entry.is_file() and entry.name != ".empty":
                entry.unlink()
                removed += 1
    return removed


def build_frontend() -> None:
    log("构建前端（npm run build）")
    command = ["cmd", "/c", "npm", "run", "build"] if os.name == "nt" else ["npm", "run", "build"]
    run(command, cwd=ROOT / "frontend")


def copy_sources(stage: Path) -> None:
    log("复制后端、桌面入口、核心源码与文档")
    for name in APP_DIRS:
        target = stage / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / name, target, ignore=IGNORE)
    for name in APP_FILES:
        shutil.copy2(ROOT / name, stage / name)
    frontend = ROOT / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        raise SystemExit("frontend/dist 缺失，请加 --build-frontend 或先运行 npm run build。")
    target = stage / "frontend" / "dist"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(frontend, target)
    log("复制离线数据集（排除可重新生成的 .gz 归档）")
    target = stage / "data"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / "data", target, ignore=shutil.ignore_patterns("__pycache__", "*.gz"))
    for relative in REQUIRED_DATA:
        if not (stage / "data" / relative).is_file():
            raise SystemExit(f"数据文件缺失：data/{relative}")


def lock_entries(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def verify(stage: Path, lock: Path, python_dir: Path) -> dict:
    log("校验依赖闭包与可迁移性")
    run([sys.executable, str(PACKAGING / "verify_runtime.py"), "--stage", str(stage), "--lock", str(lock)])
    bundled = stage / "build-lock.txt"
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    run([str(python_dir / "python.exe"), "-B", "-m", "desktop.dependencies",
         "--allow-missing-external", "--output", str(bundled)], cwd=stage, env=env)
    if set(lock_entries(bundled)) != set(lock_entries(lock)):
        raise SystemExit("发行环境的依赖与锁定清单不一致。")
    bundled.unlink()

    log("冒烟测试：无 PyTorch 时应给出明确提示而不是崩溃")
    result = subprocess.run([str(python_dir / "python.exe"), "-B", "-m", "desktop.launcher", "--diagnose"],
                            cwd=str(stage), capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=300)
    if result.returncode:
        raise SystemExit(f"启动器自检失败：{result.stderr[-3000:]}")
    report = json.loads(result.stdout)
    if report.get("cpu_available") is not False or "PyTorch" not in " ".join(report.get("warnings", [])):
        raise SystemExit(f"未安装 PyTorch 时的诊断结果不符合预期：{report}")

    relocated = stage.with_name(f"{stage.name}-relocated-{uuid.uuid4().hex}")
    log("迁移性测试：整体改名后重新自检")
    stage.rename(relocated)
    try:
        moved = subprocess.run([str(relocated / "python" / "python.exe"), "-B", "-m",
                                "desktop.launcher", "--diagnose"],
                               cwd=str(relocated), capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=300)
        if moved.returncode or json.loads(moved.stdout).get("cpu_available") is not False:
            raise SystemExit(f"迁移后自检失败：{moved.stderr[-3000:]}")
    finally:
        relocated.rename(stage)
    return report


def write_info(stage: Path, lock: Path) -> dict:
    info = {
        "application": "LianAnZhiDun",
        "version": project_version(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "bundled_packages": len(lock_entries(lock)),
        "lock_sha256": sha256(lock),
        "first_run_downloads": ["torch==2.8.0", "torchvision==0.23.0"],
    }
    info["stage_bytes"] = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
    info["stage_files"] = sum(1 for path in stage.rglob("*") if path.is_file())
    payload = json.dumps(info, ensure_ascii=False, indent=2)
    (ROOT / "build" / "build-info.json").write_text(payload, encoding="utf-8")
    (stage / "build-info.json").write_text(payload, encoding="utf-8")
    return info


def make_zip(stage: Path, output: Path) -> Path:
    log("生成便携 ZIP（耗时较长）")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.make_archive(str(output), "zip", root_dir=str(stage.parent), base_dir=stage.name))


def main() -> int:
    parser = argparse.ArgumentParser(description="构建可迁移的联安智盾应用目录")
    parser.add_argument("--stage", type=Path, default=ROOT / "build" / "stage")
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "build" / "wheelhouse")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "build" / "downloads")
    parser.add_argument("--lock", type=Path, default=PACKAGING / "requirements-lock.txt")
    parser.add_argument("--reuse", action="store_true", help="复用已解压的解释器与已装依赖")
    parser.add_argument("--download-wheels", action="store_true", help="从 PyPI 补齐 wheelhouse")
    parser.add_argument("--build-frontend", action="store_true", help="运行 npm run build")
    parser.add_argument("--zip", action="store_true", help="额外生成 dist/*-portable-*.zip")
    parser.add_argument("--index-url", default="https://pypi.tuna.tsinghua.edu.cn/simple",
                        help="仅用于 --download-wheels；可改为 https://pypi.org/simple")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    stage = args.stage.resolve()
    lock = args.lock.resolve()
    if not lock.is_file():
        raise SystemExit(f"缺少锁定清单：{lock}")

    meta = json.loads((PACKAGING / "python-runtime.json").read_text(encoding="utf-8"))
    if args.build_frontend:
        build_frontend()
    if args.reuse and (stage / "python" / "python.exe").is_file():
        python_dir = stage / "python"
        log("复用现有私有解释器")
    else:
        if stage.exists():
            log(f"清理 {stage}")
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)
        archive = download([*meta.get("mirrors", []), meta["url"]],
                           args.cache_dir / meta["asset"], meta["sha256"])
        python_dir = extract_python(archive, stage)

    if args.download_wheels:
        download_wheels(python_dir, args.wheelhouse.resolve(), lock, args.index_url)
    install_dependencies(python_dir, args.wheelhouse.resolve(), lock)
    removed = remove_console_launchers(python_dir)
    log(f"移除 {removed} 个调试符号与内嵌构建路径的启动器（Flower 每次运行重新生成）")
    copy_sources(stage)
    if not args.skip_verify:
        verify(stage, lock, python_dir)
    info = write_info(stage, lock)
    log(f"完成：{info['stage_files']} 个文件，{info['stage_bytes'] / 1024**3:.2f} GiB")
    if args.zip:
        output = ROOT / "dist" / f"LianAnZhiDun-{info['version']}-portable-win64"
        archive = make_zip(stage, output)
        log(f"便携包：{archive} ({archive.stat().st_size / 1024**3:.2f} GiB)")
    log("安装器：使用 Inno Setup 编译 packaging/liananzhidun.iss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
