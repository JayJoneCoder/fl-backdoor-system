"""Real lightweight worker/child processes; no Flower, Ray or training."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import psutil

from backend.owned_processes import OwnedProcessTree


class OwnedProcessTests(unittest.TestCase):
    def test_cleanup_kills_only_owned_tree_including_children(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text(
                'import sys,subprocess,time\n'
                'if sys.stdin.readline().strip() != "start": raise SystemExit(1)\n'
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
                'print(child.pid,flush=True)\n'
                'time.sleep(30)\n', encoding="utf-8")
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tree = OwnedProcessTree()
            process = None
            try:
                process = tree.spawn(worker, [], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                child = psutil.Process(int(process.stdout.readline().strip()))
                tree.observe()
                tree.cleanup()
                self.assertIsNotNone(process.poll())
                self.assertFalse(child.is_running())
                self.assertIsNone(unrelated.poll(), "Unrelated Python process must survive owned cleanup")
                tree.cleanup()  # Idempotent: no killpg fallback on Windows.
            finally:
                tree.cleanup()
                if process and process.stdout:
                    process.stdout.close()
                unrelated.terminate()
                unrelated.wait(timeout=5)

    @unittest.skipUnless(os.name == "nt", "Windows Job owns descendants even after root exits")
    def test_child_remains_owned_after_worker_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text(
                'import sys,subprocess\n'
                'if sys.stdin.readline().strip() != "start": raise SystemExit(1)\n'
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
                'print(child.pid,flush=True)\n', encoding="utf-8")
            tree = OwnedProcessTree()
            process = tree.spawn(worker, [], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                child = psutil.Process(int(process.stdout.readline().strip()))
                process.wait(timeout=5)
                self.assertTrue(child.is_running())
                tree.cleanup()
                self.assertFalse(child.is_running())
            finally:
                tree.cleanup()
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()