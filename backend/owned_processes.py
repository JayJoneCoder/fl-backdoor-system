"""Own one worker tree, never terminate processes by name or global Ray commands."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from ctypes import wintypes

import psutil


class WindowsJob:
    def __init__(self):
        kernel = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD

        class BasicLimit(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("basic", BasicLimit), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimit()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, no breakaway.
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_existing_children(self, process):
        # Windows venv's python.exe is a redirector: it may have already created
        # the real interpreter before root assignment. That interpreter is still
        # waiting at the stdin gate, so assign it explicitly before releasing it.
        for child in psutil.Process(process.pid).children(recursive=True):
            handle = self.kernel.OpenProcess(0x0100 | 0x0001 | 0x1000, False, child.pid)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                member = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not member.value and not self.kernel.AssignProcessToJobObject(self.handle, handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(handle)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def active_count(self):
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, wintypes.DWORD) for name in ("faults", "total", "active", "terminated")]
        info = Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.active

    def process_handles(self):
        capacity = 64
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("ids", ctypes.c_size_t * capacity)]
            info = ProcessIds()
            ok = self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None)
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            if ok and info.count == info.assigned:
                break
            capacity = max(capacity * 2, info.assigned)
        handles = []
        try:
            for pid in info.ids[:info.count]:
                handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE, QUERY
                if not handle:
                    if ctypes.get_last_error() == 87:  # Already exited.
                        continue
                    raise ctypes.WinError(ctypes.get_last_error())
                member = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                    self.kernel.CloseHandle(handle)
                    raise ctypes.WinError(ctypes.get_last_error())
                if member.value:
                    handles.append(handle)
                else:
                    self.kernel.CloseHandle(handle)  # PID was reused outside this job.
            return handles
        except BaseException:
            for handle in handles:
                self.kernel.CloseHandle(handle)
            raise

    def terminate_and_wait(self):
        handles = self.process_handles()
        try:
            self.terminate()
            deadline = time.monotonic() + 8
            while self.active_count():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned processes are still exiting; cleanup must be retried.")
                time.sleep(0.05)
            # Job accounting can reach zero before the process object is signalled.
            for handle in handles:
                timeout = max(0, int((deadline - time.monotonic()) * 1000))
                result = self.kernel.WaitForSingleObject(handle, timeout)
                if result == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
                if result != 0:
                    raise RuntimeError("Owned process termination is not complete.")
        finally:
            for handle in handles:
                self.kernel.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class OwnedProcessTree:
    def __init__(self):
        self.job = WindowsJob() if os.name == "nt" else None
        self.process = None
        self.descendants = {}

    def spawn(self, worker_path, arguments, **kwargs):
        # The worker reads this gate BEFORE importing Flower or spawning anything.
        # Assigning it to a Job first eliminates the child-before-assignment race.
        process = subprocess.Popen(
            [sys.executable, "-u", str(worker_path), *arguments],
            stdin=subprocess.PIPE, start_new_session=os.name != "nt", **kwargs,
        )
        self.process = process
        try:
            if self.job:
                self.job.assign(process)
                self.job.assign_existing_children(process)
            process.stdin.write("start\n")
            process.stdin.flush()
            process.stdin.close()
            return process
        except BaseException:
            if not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait(timeout=5)
            if self.job:
                self.job.close()
                self.job = None
            raise

    def observe(self):
        if self.job or self.process is None:
            return
        try:
            root = psutil.Process(self.process.pid)
            for child in root.children(recursive=True):
                self.descendants[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            pass

    def cleanup(self):
        """Blocking work: runner always calls this through asyncio.to_thread."""
        if self.job and self.job.handle:
            self.job.terminate_and_wait()
            self.job.close()
            self.job = None
        elif self.process and os.name != "nt":
            self.observe()
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            children = list(self.descendants.values())
            for child in children:
                try:
                    child.terminate()  # psutil also guards against PID reuse.
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=3)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, alive = psutil.wait_procs(alive, timeout=3)
            if alive:
                raise RuntimeError("Some owned child processes could not be stopped.")
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.descendants.clear()