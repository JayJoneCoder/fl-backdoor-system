"""Prove that the staged application is relocatable and reproducible offline.

Run with any Python 3.10+; it inspects the staged private interpreter only:

    python packaging/verify_runtime.py --stage build/stage --lock packaging/requirements-lock.txt

Checks:
* private interpreter version and the presence of pythonw.exe
* every pinned package (95) is installed at the exact version
* torch/torchvision are absent (first-run downloads only)
* key runtime imports work, including pip and Flower
* no .pth / pyvenv.cfg / console launcher embeds a build-machine absolute path
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


KEY_IMPORTS = ("pip", "fastapi", "uvicorn", "flwr", "ray", "numpy", "pandas", "matplotlib", "PIL", "tomli")
ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]|^\\\\|^/")
CHECK_SCRIPT = r"""
import importlib.metadata as md
import importlib.util
import json
import sys

lock = {}
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            lock[name] = version

report = {"python": sys.version.split()[0], "mismatched": [], "missing": [], "extra_torch": []}
for name, version in lock.items():
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        report["missing"].append(name)
        continue
    if installed != version:
        report["mismatched"].append(f"{name}: {installed} != {version}")
for name in ("torch", "torchvision"):
    if importlib.util.find_spec(name) is not None:
        report["extra_torch"].append(name)
try:
    import pip, fastapi, uvicorn, flwr, ray, numpy, pandas, matplotlib, PIL, tomli  # noqa: F401
    report["imports"] = "ok"
except Exception as exc:  # pragma: no cover - reported to the build log
    report["imports"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, ensure_ascii=False))
"""


def read_lock(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            entries[name] = version
    return entries


def scan_absolute_paths(stage: Path, python_dir: Path) -> list[str]:
    problems = []
    for pth in (python_dir / "Lib" / "site-packages").glob("*.pth"):
        for line in pth.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith(("import ", "#")) and ABSOLUTE_PATH.match(line):
                problems.append(f"{pth.relative_to(stage)} -> {line}")
    cfg = python_dir / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith(("home =", "executable =", "command =")):
                problems.append(f"pyvenv.cfg -> {line.strip()}")
    for launcher in (python_dir / "Scripts").iterdir():
        if launcher.is_file() and launcher.name != ".empty":
            problems.append(f"console launcher left behind: {launcher.relative_to(stage)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="校验可迁移的私有 Python 运行环境")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("requirements-lock.txt"))
    args = parser.parse_args()
    stage = args.stage.resolve()
    python_dir = stage / "python"
    python = python_dir / "python.exe"
    if not python.is_file() or not (python_dir / "pythonw.exe").is_file():
        raise SystemExit(f"缺少私有解释器：{python} 或 pythonw.exe")
    result = subprocess.run([str(python), "-I", "-B", "-c", CHECK_SCRIPT, str(args.lock.resolve())],
                            cwd=str(stage), capture_output=True, text=True, encoding="utf-8", timeout=300)
    if result.returncode:
        raise SystemExit(f"依赖校验进程失败：{result.stderr[-3000:]}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    problems = []
    if not report["python"].startswith("3.11."):
        problems.append(f"解释器版本不是 3.11.x：{report['python']}")
    if report["missing"]:
        problems.append(f"缺少锁定依赖：{', '.join(report['missing'])}")
    if report["mismatched"]:
        problems.append("版本不一致：" + "; ".join(report["mismatched"]))
    if report["extra_torch"]:
        problems.append(f"发行包中不应包含：{', '.join(report['extra_torch'])}")
    if report["imports"] != "ok":
        problems.append(f"关键模块导入失败：{report['imports']}")
    problems.extend(scan_absolute_paths(stage, python_dir))
    summary = {"stage": str(stage), "python": report["python"], "packages": len(read_lock(args.lock)),
               "problems": problems}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        return 1
    print("可迁移性与依赖闭包校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
