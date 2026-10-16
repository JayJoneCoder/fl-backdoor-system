#!/usr/bin/env python3
"""Submit batches to the same backend scheduler as the web interface.

This client never edits pyproject.toml or starts Flower itself. Start the backend
first, then run: python scripts/batch_runner.py --config experiments.json
"""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 30):
        parts = urlsplit(base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.query
                or parts.fragment or parts.username or parts.password):
            raise ValueError("后端地址必须是有效的 HTTP/HTTPS 地址，且不能包含凭据、查询或片段。")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Local API traffic must not be redirected through system HTTP proxies.
        self.opener = build_opener(ProxyHandler({})) if parts.hostname in {
            "localhost", "127.0.0.1", "::1",
        } else build_opener()

    def request(self, path, *, data=None, content_type="application/json"):
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = content_type
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (ValueError, UnicodeError, AttributeError):
                detail = None
            raise RuntimeError(f"后端拒绝请求（HTTP {exc.code}）：{detail or '请检查配置与后端日志。'}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("无法连接后端或请求超时，请先启动后端并检查 --backend-url。") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("后端返回的内容不是有效 JSON，请检查后端地址和版本。") from exc

    def parse_file(self, path: Path):
        boundary = "fl-batch-" + uuid.uuid4().hex
        content = path.read_bytes()
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="experiments.json"\r\nContent-Type: application/json\r\n\r\n').encode()
        body += content + f"\r\n--{boundary}--\r\n".encode()
        return self.request("/api/batch/parse", data=body,
                            content_type=f"multipart/form-data; boundary={boundary}")["experiments"]

    def submit(self, experiments):
        return self.request("/api/batch/run", data=json.dumps(
            {"experiments": experiments}, ensure_ascii=False).encode("utf-8"))

    def status(self, batch_id):
        return self.request(f"/api/batch/{quote(batch_id, safe='')}/status")

    def cancel(self, batch_id):
        return self.request(f"/api/batch/{quote(batch_id, safe='')}/cancel", data=b"{}")


def positive_number(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须为有限正数。") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数。")
    return number


def wait_for_batch(client, batch_id, interval):
    previous = None
    while True:
        state = client.status(batch_id)
        progress = (state["status"], state.get("completed_count", 0), state.get("current_exp_name"))
        if progress != previous:
            print(f"状态：{progress[0]}；已结束 {progress[1]}/{state.get('total', '?')}；"
                  f"当前实验：{progress[2] or '无'}", flush=True)
            previous = progress
        if state["status"] in {"completed", "failed", "cancelled"}:
            for item in state.get("results", []):
                print(f"  {item['name']}：{item['status']}；退出码：{item.get('exit_code')}"
                      f"{'; ' + item['error'] if item.get('error') else ''}", flush=True)
            if state.get("error"):
                print(f"批次说明：{state['error']}", flush=True)
            return {"completed": 0, "failed": 1, "cancelled": 3}[state["status"]]
        if state["status"] not in {"pending", "running", "stopping"}:
            raise RuntimeError(f"无法识别的批次状态：{state['status']}；请在网页查看任务。")
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="复用已启动后端的批量调度，不直接修改配置或启动训练。")
    parser.add_argument("--config", type=Path, required=True, help="批量实验 JSON（与网页上传格式相同）")
    parser.add_argument("--backend-url", default=os.environ.get("FL_BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--poll-interval", type=positive_number, default=1.0)
    parser.add_argument("--request-timeout", type=positive_number, default=30.0)
    parser.add_argument("--no-wait", action="store_true", help="提交成功后退出；批次继续由后端运行")
    args = parser.parse_args(argv)
    batch_id = None
    client = None
    try:
        client = BackendClient(args.backend_url, args.request_timeout)
        experiments = client.parse_file(args.config)
        state = client.submit(experiments)
        batch_id = state["batch_id"]
        print(f"批次已提交：{batch_id}；可在网页批量实验页面查看。", flush=True)
        return 0 if args.no_wait else wait_for_batch(client, batch_id, args.poll_interval)
    except KeyboardInterrupt:
        if batch_id and client:
            try:
                state = client.cancel(batch_id)
                print(f"已请求取消批次 {batch_id}，后端状态：{state['status']}；请等待所属进程清理。", flush=True)
            except RuntimeError as exc:
                print(f"取消请求未确认：{exc} 请在网页检查批次 {batch_id}。", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"批量执行失败：{exc}", file=sys.stderr)
        if batch_id:
            print(f"批次 {batch_id} 已提交，连接错误不代表训练已停止；请在网页查看或取消。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())