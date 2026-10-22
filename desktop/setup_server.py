"""A loopback-only, token-protected bootstrap UI requiring no installed web stack."""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import webbrowser

from desktop.runtime import BUILDS, SOURCES, active_runtime, install_runtime, probe
from fl_backdoor.utils.device import select_device


class SetupState:
    def __init__(self, report: dict):
        self.report = report
        self.logs: list[str] = []
        self.error = ""
        self.installing = False
        self.mode = "auto"
        self.cancelled = False
        self.ready = threading.Event()
        self.lock = threading.Lock()

    def install(self, source: str, build: str) -> None:
        with self.lock:
            if self.installing:
                raise ValueError("已有安装或检测正在进行，请等待。")
            if source not in SOURCES or build not in BUILDS:
                raise ValueError("无效的下载源或构建版本。")
            self.installing, self.error, self.logs = True, "", []

        def run():
            try:
                self.report = install_runtime(source, build, lambda line: self.logs.append(line))
            except Exception as exc:
                self.error = str(exc)
                self.logs.append(self.error)
            finally:
                self.installing = False

        threading.Thread(target=run, daemon=True).start()

    def confirm(self, mode: str, acknowledge: bool) -> None:
        with self.lock:
            if self.installing:
                raise ValueError("请等待安装和自检完成。")
            selected = select_device(mode, self.report)
            if selected == "cpu" and not acknowledge:
                raise ValueError("请先阅读警告并勾选允许 CPU 运行。")
            self.mode = mode
            self.ready.set()

    def cancel(self) -> None:
        with self.lock:
            if self.installing:
                raise ValueError("安装过程中请等待完成；需要中断时关闭启动器，私有进程将随之终止。")
            self.cancelled = True
            self.ready.set()


def setup_window(report: dict) -> str:
    state = SetupState(report)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, dict) else data
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return (self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"
                    and hmac.compare_digest(self.headers.get("X-Setup-Token", ""), token)
                    and self.headers.get("Origin", f"http://127.0.0.1:{self.server.server_port}") == f"http://127.0.0.1:{self.server.server_port}")

        def do_GET(self):
            if self.path.startswith("/?token="):
                self.reply(200, (Path(__file__).parent / "setup.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/status" and self.authorized():
                self.reply(200, {"sources": SOURCES, "builds": BUILDS, "report": state.report,
                                 "logs": state.logs[-300:], "installing": state.installing, "error": state.error})
            else:
                self.reply(403, {"detail": "无效的本地配置会话。"})

        def do_POST(self):
            if not self.authorized() or state.ready.is_set():
                return self.reply(403, {"detail": "无效或已结束的本地配置会话。"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("请求大小无效。")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("请求必须是对象。")
                if self.path == "/install":
                    state.install(data.get("source"), data.get("build"))
                elif self.path == "/continue":
                    state.confirm(data.get("mode"), data.get("acknowledge") is True)
                elif self.path == "/cancel":
                    state.cancel()
                elif self.path == "/probe":
                    with state.lock:
                        if state.installing:
                            raise ValueError("请等待安装完成。")
                        state.installing = True
                    try:
                        state.report = probe(active_runtime())
                    finally:
                        state.installing = False
                else:
                    return self.reply(404, {"detail": "接口不存在。"})
                self.reply(200, {"status": "ok"})
            except (ValueError, RuntimeError, TypeError, KeyError, OSError) as exc:
                self.reply(400, {"detail": str(exc)})

        def log_message(self, *args):
            pass  # Do not write the secret URL to HTTP access logs.

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/?token={token}"
        print(f"运行环境配置窗口：{url}", flush=True)
        try:
            webbrowser.open(url, new=1)
            state.ready.wait()
        finally:
            server.shutdown()
            thread.join()
    if state.cancelled:
        raise RuntimeError("已取消启动，未开始训练。")
    return state.mode