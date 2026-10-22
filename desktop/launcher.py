"""Desktop entry: bootstrap ML runtime, then start a loopback-only private API."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
import uuid

from fl_backdoor.utils.app_paths import initialize_desktop, resource_root, state_root, user_root
from desktop.runtime import active_runtime, probe, runtime_environment
from desktop.setup_server import setup_window


class InstanceLock:
    """OS-owned advisory lock is automatically released after a launcher crash."""

    def __enter__(self):
        self.file = (state_root() / "desktop.lock").open("a+b")
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt

            if self.file.read(1) == b"":
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            try:
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self.file.close()
                raise RuntimeError("当前用户已启动平台或配置窗口，请切换到现有窗口。") from None
        else:
            import fcntl

            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.file.close()
                raise RuntimeError("平台已启动。") from None
        return self

    def __exit__(self, *args):
        self.file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="联安智盾桌面启动器（发行适配入口）")
    parser.add_argument("--setup", action="store_true", help="重新选择 PyTorch 来源/版本和运行设备")
    parser.add_argument("--diagnose", action="store_true", help="只输出环境自检，不安装或启动后端")
    args = parser.parse_args()
    if args.diagnose:
        # No writes; useful for development and portable-runtime acceptance.
        print(json.dumps(probe(active_runtime()), ensure_ascii=False, indent=2))
        return 0
    os.environ["FL_BACKDOOR_DESKTOP"] = "1"
    state_root().mkdir(parents=True, exist_ok=True)
    with InstanceLock():
        initialize_desktop()
        settings = state_root() / "desktop.json"
        try:
            target = active_runtime()
        except (ValueError, KeyError, OSError):
            target = None
        report = probe(target)
        mode = "auto"
        if settings.exists():
            try:
                mode = json.loads(settings.read_text(encoding="utf-8"))["mode"]
            except (ValueError, KeyError, OSError):
                pass
        if (args.setup or not settings.exists() or mode not in {"auto", "cpu", "cuda"}
                or not report.get("cpu_available") or (mode != "cpu" and not report.get("gpu_available"))):
            mode = setup_window(report)
            settings.write_text(json.dumps({"mode": mode}), encoding="utf-8")
            try:
                target = active_runtime()
            except (ValueError, KeyError, OSError):
                # Reusing a valid bundled runtime is safe even if an old overlay
                # marker is corrupt. A successful install replaces that marker.
                target = None
        if mode not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("运行设备记录无效，请使用 --setup 重新配置。")
        env = runtime_environment(target)
        env["FL_BACKDOOR_DEVICE"] = mode
        if target:
            env["FL_BACKDOOR_TORCH_DIR"] = str(target)
        if not (resource_root() / "frontend" / "dist" / "index.html").is_file():
            raise RuntimeError("发行资源中缺少 frontend/dist，请先构建前端；评委无需安装 Node.js。")
        from backend.owned_processes import OwnedProcessTree

        startup_id = uuid.uuid4().hex
        env["FL_BACKDOOR_STARTUP_ID"] = startup_id
        ready = user_root() / "tmp" / f"api-{startup_id}.json"
        owner = OwnedProcessTree()
        logfile = user_root() / "logs" / "backend.log"
        with logfile.open("a", encoding="utf-8") as logs:
            try:
                child = owner.spawn(resource_root() / "desktop" / "api_entry.py", ["--ready-file", str(ready)],
                                    cwd=str(resource_root()), env=env, stdout=logs,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                deadline = time.monotonic() + 60
                url = ""
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        raise RuntimeError(f"后端启动失败，请检查 {logfile}")
                    try:
                        port = json.loads(ready.read_text(encoding="utf-8"))["port"]
                        url = f"http://127.0.0.1:{port}"
                        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                            if json.load(response).get("startup_id") == startup_id:
                                break
                    except (OSError, ValueError, KeyError):
                        pass
                    time.sleep(0.3)
                else:
                    raise RuntimeError(f"后端健康检查超时，请检查 {logfile}")
                print(f"联安智盾已启动：{url}\n关闭启动器将结束所属训练进程。日志：{logfile}", flush=True)
                webbrowser.open(url)
                child.wait()
                return child.returncode
            except KeyboardInterrupt:
                return 0
            finally:
                owner.cleanup()
                ready.unlink(missing_ok=True)
                ready.with_suffix(".tmp").unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)