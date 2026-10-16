"""Exercise the actual websocket handler without a server or training process."""

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import WebSocketDisconnect

from backend import main
from backend.experiment_runner import ExperimentRunner
from fl_backdoor.utils.run_artifacts import reserve_experiment


class MemoryWebSocket:
    def __init__(self):
        self.accepted = False
        self.messages = asyncio.Queue()
        self.disconnected = asyncio.Event()
        self.history = []
        self.query_params = {}
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def send_text(self, text):
        self.history.append(text)
        await self.messages.put(text)

    async def send_json(self, data):
        self.history.append(data)
        await self.messages.put(data)

    async def receive_text(self):
        await self.disconnected.wait()
        raise WebSocketDisconnect(code=1000)

    async def close(self, code=1000, reason=""):
        self.close_code = code
        self.disconnected.set()

    async def until(self, predicate):
        for message in self.history:
            if predicate(message):
                return message

        async def read():
            while True:
                message = await self.messages.get()
                if predicate(message):
                    return message

        return await asyncio.wait_for(read(), timeout=3)


class ExperimentWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.exp_dir = self.root / "same_name"
        self.exp_dir.mkdir()
        self.log = self.exp_dir / "run.log"
        self.csv = self.exp_dir / "same_name.csv"
        self.snapshot = self.exp_dir / "config_snapshot.toml"
        self.snapshot.write_text(
            "[tool.flwr.app.config]\nnum-server-rounds = 10\n", encoding="utf-8"
        )
        self.runner = ExperimentRunner()
        self.runner.exp_name = "same_name"
        self.runner._experiment_dir = self.exp_dir
        self.runner.total_rounds = 10
        self.runner._log_runs["same_name"] = (1, 0)
        self.runner.process = Mock(returncode=None)
        self.runner.process.poll.return_value = None
        self.connections = []
        self.results_patch = patch.object(main, "_results_root", lambda: self.root)
        self.runner_patch = patch.object(main, "runner", self.runner)
        self.results_patch.start()
        self.runner_patch.start()

    async def asyncTearDown(self):
        try:
            for socket, task in self.connections:
                socket.disconnected.set()
            for socket, task in self.connections:
                try:
                    await asyncio.wait_for(task, timeout=3)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        finally:
            self.results_patch.stop()
            self.runner_patch.stop()

    def connect(self, name="same_name", run_id=None):
        socket = MemoryWebSocket()
        if run_id is not None:
            socket.query_params["run_id"] = run_id
        task = asyncio.create_task(main.websocket_logs(socket, name))
        self.connections.append((socket, task))
        return socket, task

    async def test_uuid_pins_status_log_and_metrics_after_result_root_switch(self):
        old_dir, old = reserve_experiment(self.root / "old-results", "same_name")
        (old_dir / "run.log").write_text("owned run log\n", encoding="utf-8")
        (old_dir / "same_name.csv").write_text("round,accuracy,asr,loss\n0,0.8,0.1,0.2\n", encoding="utf-8")
        self.runner._experiment_dir = old_dir
        self.runner.run_id = old["run_id"]
        self.runner._status = "running"
        self.runner._log_directories["same_name"] = old_dir
        new_id = uuid.uuid4().hex
        (self.exp_dir / "run.json").write_text(json.dumps({
            "name": "same_name", "run_id": new_id, "status": "finished",
        }), encoding="utf-8")
        self.log.write_text("other directory log\n", encoding="utf-8")
        self.csv.write_text("round,accuracy,asr,loss\n0,0.2,0.3,0.4\n", encoding="utf-8")
        for state in ("running", "finished"):
            self.runner._status = state
            socket, _ = self.connect(run_id=old["run_id"])
            status = await socket.until(lambda m: isinstance(m, dict) and m.get("type") == "status")
            metric = await socket.until(lambda m: isinstance(m, dict) and m.get("type") == "csv_update")
            await socket.until(lambda m: m == "owned run log")
            self.assertEqual(status["data"]["run_id"], old["run_id"])
            self.assertEqual(status["data"]["status"], state)
            self.assertEqual(float(metric["data"]["accuracy"]), 0.8)
            self.assertNotIn("other directory log", socket.history)

    async def test_wrong_uuid_is_rejected_without_sending_logs_or_metrics(self):
        self.log.write_text("private log\n", encoding="utf-8")
        socket, task = self.connect(run_id=uuid.uuid4().hex)
        await asyncio.wait_for(task, 2)
        self.assertEqual(socket.close_code, 1008)
        self.assertEqual(socket.history, [])

    async def test_finished_same_name_in_another_root_does_not_supply_status_or_generation(self):
        other_dir = self.root / "old-results" / "same_name"
        other_dir.mkdir(parents=True)
        self.runner._experiment_dir = other_dir
        self.runner.run_id = uuid.uuid4().hex
        self.runner._status = "finished"
        self.runner.process.poll.return_value = 0
        new_run_id = uuid.uuid4().hex
        (self.exp_dir / "run.json").write_text(json.dumps({
            "name": "same_name", "run_id": new_run_id, "status": "finished",
        }), encoding="utf-8")
        self.log.write_text("new directory log\n", encoding="utf-8")
        self.csv.write_text("round,accuracy,asr,loss\n0,0.5,0.1,0.2\n", encoding="utf-8")
        self.runner._log_runs["same_name"] = (9, 2**63)
        socket, _ = self.connect()
        state = await socket.until(lambda m: isinstance(m, dict) and m.get("type") == "status")
        self.assertEqual(state["data"]["run_id"], new_run_id)
        self.assertEqual(state["data"]["log_generation"], 0)
        await socket.until(lambda m: isinstance(m, dict) and m.get("type") == "csv_update")
        await socket.until(lambda m: m == "new directory log")

    async def test_two_clients_and_disconnect_leave_batch_callback_intact(self):
        async def batch_callback(line):
            return None

        self.runner.set_log_callback(batch_callback)
        self.log.write_text("history\n", encoding="utf-8")
        first, first_task = self.connect()
        second, _ = self.connect()
        await first.until(lambda message: message == "history")
        await second.until(lambda message: message == "history")
        self.assertIs(self.runner._on_log_line, batch_callback)

        first.disconnected.set()
        await asyncio.wait_for(first_task, timeout=3)
        self.assertIs(self.runner._on_log_line, batch_callback)
        with self.log.open("a", encoding="utf-8") as f:
            f.write("new line\n")
        await second.until(lambda message: message == "new line")
        self.assertEqual(second.history.count("history"), 1)
        self.assertEqual(second.history.count("new line"), 1)

    async def test_same_name_rerun_replays_new_log_from_beginning(self):
        self.log.write_text("old\n", encoding="utf-8")
        socket, _ = self.connect()
        await socket.until(lambda message: message == "old")
        self.log.write_text("Starting local SuperLink\n[ROUND 1/10]\n", encoding="utf-8")
        self.runner._log_runs["same_name"] = (2, 0)
        await socket.until(lambda message: message == "Starting local SuperLink")
        await socket.until(lambda message: message == "[ROUND 1/10]")
        self.assertEqual(socket.history.count("old"), 1)

    async def test_connect_before_directory_exists_then_receive_csv_and_status(self):
        socket, _ = self.connect("later")
        await socket.until(lambda message: message == "[System] Waiting for log file...")
        exp_dir = self.root / "later"
        exp_dir.mkdir()
        (exp_dir / "config_snapshot.toml").write_text(
            "[tool.flwr.app.config]\nnum-server-rounds = 7\n", encoding="utf-8"
        )
        (exp_dir / "run.log").write_text("first line\n", encoding="utf-8")
        (exp_dir / "later.csv").write_text(
            "round,accuracy,asr,loss\n0,0.1,0.2,2.5\n3,0.7,0.3,0.4\n", encoding="utf-8"
        )
        status = await socket.until(
            lambda message: isinstance(message, dict)
            and message.get("type") == "status"
            and message["data"].get("total_rounds") == 7
        )
        self.assertEqual(status["data"]["total_rounds"], 7)
        # Other tasks can send rows while waiting for status; inspect the replay.
        if not any(isinstance(message, dict) and message.get("type") == "csv_update"
                   and message["data"]["round"] == 3 for message in socket.history):
            await socket.until(
                lambda message: isinstance(message, dict)
                and message.get("type") == "csv_update" and message["data"]["round"] == 3
            )
        rows = [message["data"]["round"] for message in socket.history
                if isinstance(message, dict) and message.get("type") == "csv_update"]
        self.assertEqual(rows, [0, 3])

    async def test_reconnect_replays_full_history_then_follows_new_rows(self):
        self.log.write_text("starting\nround one\n", encoding="utf-8")
        self.csv.write_text(
            "round,accuracy,asr,loss\n0,0.1,0.2,2.0\n1,0.5,0.4,1.0\n",
            encoding="utf-8",
        )
        first, first_task = self.connect()
        await first.until(lambda message: message == "round one")
        await first.until(
            lambda message: isinstance(message, dict)
            and message.get("type") == "csv_update" and message["data"]["round"] == 1
        )
        first.disconnected.set()
        await asyncio.wait_for(first_task, timeout=3)

        # Work keeps progressing while the page is unmounted or being refreshed.
        with self.log.open("a", encoding="utf-8") as f:
            f.write("round two\n")
        with self.csv.open("a", encoding="utf-8") as f:
            f.write("2,0.6,0.3,0.9\n")
        restored, _ = self.connect()
        await restored.until(lambda message: message == "round two")
        await restored.until(
            lambda message: isinstance(message, dict)
            and message.get("type") == "csv_update" and message["data"]["round"] == 2
        )
        with self.log.open("a", encoding="utf-8") as f:
            f.write("round three\n")
        with self.csv.open("a", encoding="utf-8") as f:
            f.write("3,0.7,0.2,0.8\n")
        await restored.until(lambda message: message == "round three")
        await restored.until(
            lambda message: isinstance(message, dict)
            and message.get("type") == "csv_update" and message["data"]["round"] == 3
        )
        self.assertEqual(
            [message for message in restored.history if isinstance(message, str)],
            ["starting", "round one", "round two", "round three"],
        )
        self.assertEqual(
            [message["data"]["round"] for message in restored.history
             if isinstance(message, dict) and message.get("type") == "csv_update"],
            [0, 1, 2, 3],
        )

    async def test_reconnect_after_completion_restores_history_without_next_round(self):
        self.runner.process.poll.return_value = 0
        self.runner.process.returncode = 0
        self.log.write_text("starting\nfinal summary\n", encoding="utf-8")
        self.csv.write_text(
            "round,accuracy,asr,loss\n0,0.1,0.2,2.0\n10,0.9,0.1,0.3\n",
            encoding="utf-8",
        )
        socket, _ = self.connect()
        await socket.until(lambda message: message == "final summary")
        await socket.until(
            lambda message: isinstance(message, dict)
            and message.get("type") == "csv_update" and message["data"]["round"] == 10
        )
        status = await socket.until(
            lambda message: isinstance(message, dict) and message.get("type") == "status"
        )
        self.assertEqual(status["data"]["status"], "finished")
        self.assertEqual(status["data"]["total_rounds"], 10)
        self.assertEqual(
            [message["data"]["round"] for message in socket.history
             if isinstance(message, dict) and message.get("type") == "csv_update"],
            [0, 10],
        )

    async def test_finished_status_does_not_cut_off_final_log_line(self):
        self.log.write_text("first\n", encoding="utf-8")
        socket, _ = self.connect()
        await socket.until(lambda message: message == "first")
        self.runner.process.poll.return_value = 0
        self.runner.process.returncode = 0
        await socket.until(
            lambda message: isinstance(message, dict) and message.get("type") == "status"
            and message["data"]["status"] == "finished"
        )
        with self.log.open("a", encoding="utf-8") as f:
            f.write("final metrics summary\n")
        await socket.until(lambda message: message == "final metrics summary")


class RunnerLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_crlf_and_final_unterminated_line_reach_file_and_batch_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner()
            runner.log_file = Path(directory) / "run.log"
            runner.process = Mock()
            runner.process.poll.return_value = 0
            runner.process.stdout = io.StringIO(
                "Starting local SuperLink\r\n[ROUND 1/10]\r\nfinal summary"
            )
            lines = []

            async def collect(line):
                lines.append(line)

            runner.set_log_callback(collect)
            await runner._read_stdout()
            self.assertEqual(lines, ["Starting local SuperLink", "[ROUND 1/10]", "final summary"])
            self.assertEqual(runner.log_file.read_text(encoding="utf-8").splitlines(), lines)

    async def test_start_rejects_old_directory_and_new_run_exposes_total_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp_dir = root / "same_name"
            exp_dir.mkdir()
            log = exp_dir / "run.log"
            log.write_text("previous experiment log\n", encoding="utf-8")
            snapshot = root / "source.toml"
            snapshot.write_text(
                '[project]\nname = "test-project"\nversion = "1.0"\n'
                '[tool.flwr.app.components]\nserverapp = "fl_backdoor.server.server:app"\n'
                'clientapp = "fl_backdoor.client.client:app"\n'
                '[tool.flwr.app.config]\nnum-server-rounds = 7\n', encoding="utf-8")
            process = Mock(returncode=0)
            process.stdout = io.StringIO("")
            process.poll.return_value = 0
            runner = ExperimentRunner()
            owner = Mock()
            owner.spawn.return_value = process
            with patch("backend.experiment_runner.RESULTS_DIR", root), \
                 patch("backend.experiment_runner.config_manager.TOML_PATH", snapshot), \
                 patch("backend.experiment_runner.config_manager.backup_config"), \
                 patch("backend.experiment_runner.OwnedProcessTree", return_value=owner):
                with self.assertRaises(FileExistsError):
                    await runner.start("same_name")
                self.assertEqual(log.read_text(encoding="utf-8"), "previous experiment log\n")
                owner.spawn.assert_not_called()
                await runner.start("new_name", backup_config=False)
                self.assertEqual((await runner.get_status())["total_rounds"], 7)
                self.assertEqual(runner.get_log_context("new_name")[0], 1)
                await runner.wait()


if __name__ == "__main__":
    unittest.main()