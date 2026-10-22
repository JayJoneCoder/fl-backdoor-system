"""Console-less entry point for Start Menu shortcuts (``pythonw.exe``).

``pythonw.exe`` has no standard streams. Redirect them to the per-user log
before anything else runs, so bootstrap messages survive, and report fatal
errors with a native message box instead of exiting silently.

The installer invokes this file by absolute path:
``{app}\\python\\pythonw.exe "{app}\\desktop\\start_gui.py"``.
"""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys


MUTEX_NAME = "LianAnZhiDunPlatform"
MESSAGE_TITLE = "联安智盾"


def application_root() -> Path:
    return Path(__file__).resolve().parents[1]


def log_directory() -> Path:
    override = os.environ.get("FL_BACKDOOR_USER_DIR")
    if override:
        return Path(override).expanduser().resolve() / "logs"
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / "AppData" / "Local") / "LianAnZhiDun" / "logs"


def redirect_streams() -> None:
    """Give pythonw.exe a real stdout/stderr so logging and tracebacks work."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        directory = log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        stream = (directory / "launcher.log").open("a", encoding="utf-8", buffering=1)
    except OSError:
        return
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _report(text: str) -> None:
    if sys.stderr is not None:
        try:
            print(text, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
    if os.name != "nt":
        return
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
        user32.MessageBoxW.restype = ctypes.c_int
        user32.MessageBoxW(None, text, MESSAGE_TITLE, 0x10)  # MB_ICONERROR
    except Exception:
        pass


def main() -> int:
    root = application_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Keep the installation directory free of runtime __pycache__ (uninstall clean).
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    redirect_streams()
    try:
        from desktop.launcher import main as launch
        return launch()
    except SystemExit:
        raise
    except BaseException as exc:
        _report(f"{MESSAGE_TITLE}启动失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
