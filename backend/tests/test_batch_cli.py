"""Exercise the CLI's actual HTTP contract without starting training."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from backend.config_manager import parse_batch_experiments
from scripts import batch_runner as cli


class BatchCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "batch.json"
        self.payload = {"common": {"num-clients": 3}, "experiments": [{"name": "示例实验"}]}
        self.path.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
        self.requests = []
        self.status = "completed"
        self.rejection = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                owner.requests.append(("POST", self.path, body))
                code = 200
                if self.path == "/api/batch/parse":
                    content = body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
                    experiments = parse_batch_experiments(content)
                    data = {"experiments": experiments, "count": len(experiments)}
                elif self.path == "/api/batch/run":
                    code, data = owner.rejection or (200, {"batch_id": "batch-1", "status": "batch_started"})
                else:
                    data = {"status": "stopping"}
                self.respond(code, data)

            def do_GET(self):
                owner.requests.append(("GET", self.path, None))
                self.respond(200, {"status": owner.status, "total": 1, "completed_count": 1,
                                   "results": [{"name": "示例实验", "status": owner.status, "exit_code": 0}]})

            def respond(self, code, value):
                data = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def invoke(self, *extra):
        self.output = io.StringIO()
        with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output):
            return cli.main(["--config", str(self.path), "--backend-url", self.url, *extra])

    def test_upload_uses_backend_parser_and_submits_merged_config(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual([item[:2] for item in self.requests], [
            ("POST", "/api/batch/parse"), ("POST", "/api/batch/run"), ("GET", "/api/batch/batch-1/status")])
        self.assertEqual(json.loads(self.requests[1][2]), {
            "experiments": [{"name": "示例实验", "num-clients": 3}]})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.payload)

    def test_no_wait_submits_without_polling(self):
        self.assertEqual(self.invoke("--no-wait"), 0)
        self.assertEqual(len(self.requests), 2)

    def test_failure_and_cancelled_status_return_nonzero(self):
        for status, expected in (("failed", 1), ("cancelled", 3)):
            self.status = status
            self.assertEqual(self.invoke(), expected)

    def test_paused_queue_remains_pollable_until_resumed(self):
        client = Mock()
        client.status.side_effect = [
            {"status": "paused", "total": 2, "completed_count": 1},
            {"status": "running", "total": 2, "completed_count": 1},
            {"status": "completed", "total": 2, "completed_count": 2},
        ]
        with patch.object(cli.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.wait_for_batch(client, "batch-1", .01), 0)
        self.assertEqual(client.status.call_count, 3)

    def test_http_conflict_keeps_detail_and_does_not_poll(self):
        self.rejection = (409, {"detail": "实验名称已存在，请换一个名称。"})
        self.assertEqual(self.invoke(), 1)
        self.assertIn("HTTP 409", self.output.getvalue())
        self.assertIn("实验名称已存在", self.output.getvalue())
        self.assertEqual(len(self.requests), 2)

    def test_keyboard_interrupt_cancels_submitted_batch(self):
        with patch.object(cli, "wait_for_batch", side_effect=KeyboardInterrupt):
            self.assertEqual(self.invoke(), 130)
        self.assertEqual(self.requests[-1][:2], ("POST", "/api/batch/batch-1/cancel"))
        self.assertIn("已请求取消", self.output.getvalue())

    def test_connection_error_after_submission_reports_batch_id(self):
        with patch.object(cli.BackendClient, "status", side_effect=RuntimeError("连接失败")):
            self.assertEqual(self.invoke(), 1)
        self.assertIn("batch-1 已提交", self.output.getvalue())
        self.assertIn("连接错误不代表训练已停止", self.output.getvalue())
        self.assertFalse(any(item[1].endswith("cancel") for item in self.requests))

    def test_bad_file_or_url_never_submits(self):
        self.path.unlink()
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.requests, [])
        for url in ("file:///tmp", "http://user:secret@localhost:8000", "http://localhost:8000?token=x"):
            with self.assertRaises(ValueError):
                cli.BackendClient(url)
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.assertRaises(cli.argparse.ArgumentTypeError):
                cli.positive_number(value)


if __name__ == "__main__":
    unittest.main()