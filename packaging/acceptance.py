"""End-to-end acceptance for the packaged application (no clean machine required).

It uses only the private interpreter and the user-data directory an installed
copy would use:

1. download + activate the CPU PyTorch wheels through ``desktop.runtime``
2. start the packaged API on a random loopback port
3. log in, start a 1-round / 2-client MNIST run, wait for it to finish
4. assert the result artifacts and the recorded device

Usage:
    python packaging/acceptance.py                       # build/stage, CPU, SJTU mirror
    python packaging/acceptance.py --app build/installed --skip-install
"""

import argparse
import http.cookiejar
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "num-clients": 2,
    "num-server-rounds": 1,
    "local-epochs": 1,
    "batch-size": 8,
    "dataset": "mnist",
    "fraction-evaluate": 1.0,
    "partition-mode": "iid",
    "attack": "none",
    "client-defense": "none",
    "detection": "none",
    "defense": "none",
    "runtime-device": "cpu",
    "seed": 42,
}


def log(message: str) -> None:
    print(f"[acceptance] {message}", flush=True)


def private_python(app: Path) -> Path:
    python = app / "python" / "python.exe"
    if not python.is_file():
        raise SystemExit(f"找不到私有解释器：{python}")
    return python


def base_env(app: Path, user: Path, target: Path | None) -> dict:
    env = os.environ.copy()
    env.update(
        FL_BACKDOOR_DESKTOP="1",
        FL_BACKDOOR_USER_DIR=str(user),
        FL_BACKDOOR_CONFIG=str(user / "pyproject.toml"),
        FL_BACKDOOR_DATA_DIR=str(app / "data"),
        FL_BACKDOOR_RESULTS_DIR=str(user / "results"),
        FL_IDENTITY_DATABASE=str(user / "platform_state" / "identity.sqlite3"),
        FL_BACKDOOR_DATABASE=str(user / "platform_state" / "catalog.sqlite3"),
        MPLCONFIGDIR=str(user / "cache" / "matplotlib"),
        FL_BACKDOOR_OFFLINE="1",
        FL_BACKDOOR_DEVICE="cpu",
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    paths = ([str(target)] if target else []) + [str(app)]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if target:
        env["FL_BACKDOOR_TORCH_DIR"] = str(target)
    return env


def prepare_user_dir(app: Path, user: Path) -> None:
    for directory in (user, user / "platform_state", user / "logs", user / "cache", user / "tmp", user / "results"):
        directory.mkdir(parents=True, exist_ok=True)
    config = user / "pyproject.toml"
    if not config.exists():
        shutil.copyfile(app / "pyproject.toml", config)




class Client:
    def __init__(self, base: str):
        self.base = base
        self.csrf = ""
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(self, method: str, path: str, payload=None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        request.add_header("X-Lianan-Request", "1")
        if self.csrf:
            request.add_header("X-CSRF-Token", self.csrf)
        with self.opener.open(request, timeout=120) as response:
            body = response.read()
        return json.loads(body) if body else {}


def start_api(app: Path, user: Path, target: Path, log_path: Path):
    env = base_env(app, user, target)
    startup = uuid.uuid4().hex
    env["FL_BACKDOOR_STARTUP_ID"] = startup
    ready = user / "tmp" / f"api-{startup}.json"
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen([str(private_python(app)), "-u", str(app / "desktop" / "api_entry.py"),
                                "--ready-file", str(ready)], cwd=str(app), env=env, stdin=subprocess.PIPE,
                               stdout=stream, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    process.log_stream = stream
    process.stdin.write("start\n")
    process.stdin.flush()
    process.stdin.close()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"API 启动失败，见 {log_path}：\n"
                             f"{log_path.read_text(encoding='utf-8', errors='replace')[-4000:]}")
        try:
            port = json.loads(ready.read_text(encoding="utf-8"))["port"]
            url = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(url + "/api/health", timeout=2) as response:
                if json.load(response).get("startup_id") == startup:
                    return process, url
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.3)
    raise SystemExit("API 健康检查超时")


def stop_process(process) -> None:
    if process is None:
        return
    if process.poll() is None and os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
    stream = getattr(process, "log_stream", None)
    if stream:
        stream.close()


def run_experiment(client: Client, name: str, timeout: int, config: dict) -> dict:
    session = client.request("POST", "/api/auth/login",
                             {"username": "admin", "password": "admin123", "remember": False})
    client.csrf = session["csrf_token"]
    log(f"登录成功：{session['user']['username']} ({session['user']['role']})")
    started = client.request("POST", "/api/experiment/start", {"name": name, "config": config})
    log(f"实验已启动：{started}")
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        status = client.request("GET", "/api/experiment/status")
        if status.get("status") != previous:
            log(f"状态：{status.get('status')} {status.get('error') or ''}".rstrip())
            previous = status.get("status")
        if previous in {"finished", "failed", "cancelled"}:
            return status
        time.sleep(3)
    raise SystemExit("实验超时，未在限定时间内结束。")


def verify_results(user: Path, name: str, status: dict) -> dict:
    directory = user / "results" / name
    files = sorted(path.name for path in directory.glob("*")) if directory.is_dir() else []
    if status.get("status") != "finished":
        log(f"结果目录：{directory} -> {files}")
        raise SystemExit(f"实验未成功完成：{status}")
    for required in ("run.log", "run.json", "device_runtime.json"):
        if not (directory / required).is_file():
            raise SystemExit(f"缺少结果文件 {required}，目录内容：{files}")
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"没有生成指标 CSV，目录内容：{files}")
    device = json.loads((directory / "device_runtime.json").read_text(encoding="utf-8"))
    if device.get("selected_device") != "cpu" or not str(device.get("torch", "")).endswith("+cpu"):
        raise SystemExit(f"设备记录不符合预期：{device}")
    rows = csv_files[0].read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(rows) < 2:
        raise SystemExit(f"指标 CSV 没有数据行：{csv_files[0]}")
    log(f"训练完成：{csv_files[0].name} 共 {len(rows) - 1} 行指标，torch={device.get('torch')}")
    return {"files": files, "csv": csv_files[0].name, "rows": len(rows) - 1,
            "torch": device.get("torch"), "device": device.get("selected_device")}


def main() -> int:
    parser = argparse.ArgumentParser(description="打包产物本地全链路验收")
    parser.add_argument("--app", type=Path, default=ROOT / "build" / "stage")
    parser.add_argument("--user-dir", type=Path, default=ROOT / "build" / "acceptance-user")
    parser.add_argument("--source", choices=("sjtu", "official"), default="sjtu")
    parser.add_argument("--build", choices=("cpu", "cu126", "cu128"), default="cpu")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--reuse-runtime", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--name", default="acceptance-smoke")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset", default="mnist", choices=("mnist", "cifar10"))
    parser.add_argument("--attack", default="none")
    parser.add_argument("--client-defense", default="none")
    parser.add_argument("--detection", default="none")
    parser.add_argument("--defense", default="none")
    args = parser.parse_args()
    app = args.app.resolve()
    user = args.user_dir.resolve()
    prepare_user_dir(app, user)
    record = user / "platform_state" / "runtime.json"
    if args.skip_install or (args.reuse_runtime and record.is_file()):
        configured = json.loads(record.read_text(encoding="utf-8"))
        target = user / "runtimes" / configured["directory"]
        log(f"复用已激活环境：{target}")
        if not target.is_dir():
            raise SystemExit(f"运行环境不存在：{target}")
    else:
        target = install_runtime(app, user, args.source, args.build)
    log(f"运行环境：{target}")

    api_log = user / "logs" / "acceptance-api.log"
    process = None
    try:
        process, url = start_api(app, user, target, api_log)
        log(f"API 已启动：{url}")
        client = Client(url)
        health = client.request("GET", "/api/health")
        if health.get("status") != "ok":
            raise SystemExit(f"健康检查失败：{health}")
        with urllib.request.urlopen(url + "/", timeout=10) as response:
            if b"<html" not in response.read().lower():
                raise SystemExit("根路径没有返回前端页面。")
        log("前端静态页面可用")
        config = dict(CONFIG)
        config.update({
            "num-clients": args.clients, "num-server-rounds": args.rounds,
            "local-epochs": args.local_epochs, "batch-size": args.batch_size,
            "dataset": args.dataset, "attack": args.attack,
            "client-defense": args.client_defense, "detection": args.detection,
            "defense": args.defense,
        })
        status = run_experiment(client, args.name, args.timeout, config)
        summary = verify_results(user, args.name, status)
    finally:
        stop_process(process)
    print(json.dumps({"status": "passed", "app": str(app), "user_dir": str(user), **summary},
                     ensure_ascii=False, indent=2))
    return 0


def install_runtime(app: Path, user: Path, source: str, build: str) -> Path:
    log(f"通过 {source}/{build} 下载并激活 PyTorch（这会产生约 1 GB 网络流量）")
    env = base_env(app, user, None)
    code = ("import json; from desktop.runtime import install_runtime; "
            f"report = install_runtime({source!r}, {build!r}, lambda line: print(line, flush=True)); "
            "print('REPORT=' + json.dumps(report, ensure_ascii=False))")
    result = subprocess.run([str(private_python(app)), "-B", "-c", code], cwd=str(app), env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    if result.returncode:
        raise SystemExit(f"PyTorch 安装失败：\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    log(result.stdout.strip().splitlines()[-1])
    record = json.loads((user / "platform_state" / "runtime.json").read_text(encoding="utf-8"))
    target = user / "runtimes" / record["directory"]
    if not target.is_dir():
        raise SystemExit(f"激活记录指向不存在的目录：{target}")
    return target


if __name__ == "__main__":
    raise SystemExit(main())

