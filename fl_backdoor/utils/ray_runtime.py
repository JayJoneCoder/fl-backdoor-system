"""Keep local Flower/Ray workers on the immutable experiment snapshot.

Flower 1.29 builds the Ray PYTHONPATH using ':' even on Windows. Use Ray's
runtime-environment hook in owned experiments to supply native path separators;
otherwise cloudpickle can import ClientApp handlers from an editable install.
The runtime driver must also start in the snapshot, since Ray prepends its cwd
to worker sys.path ahead of PYTHONPATH.

Ray is only experimental on Windows: raylet/GCS can fail to start on machines
with antivirus filtering, a missing MSVC runtime, or a non-ASCII %TEMP%. These
helpers use an ASCII temp root, relax the startup health checks and surface a
diagnosable report instead of Flower's opaque "No heartbeat received" error.
"""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


HOOK = "fl_backdoor.utils.ray_runtime.runtime_env_hook"
PROBE_CACHE = "ray-probe.json"
PROBE_TTL_SECONDS = 12 * 60 * 60
FAILURE_TTL_SECONDS = 5 * 60

# Ray reads these before the raylet/GCS processes start. The values tolerate
# slow Windows hosts and antivirus scanners.
RAY_ENVIRONMENT = {
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    "RAY_USAGE_STATS_ENABLED": "0",
    "RAY_DISABLE_USAGE_STATS": "1",
    "RAY_LOG_TO_STDERR": "1",
    "RAY_health_check_initial_delay_ms": "5000",
    "RAY_health_check_timeout_ms": "120000",
    "RAY_health_check_period_ms": "10000",
    "RAY_health_check_failure_threshold": "10",
    "RAY_num_heartbeat_timeout_milliseconds": "120000",
}


def _is_ascii(value: str | None) -> bool:
    if not value:
        return True
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def ray_temp_root() -> Path:
    """Return an ASCII-only directory; raylet cannot handle non-ASCII paths."""
    public = os.environ.get("PUBLIC")
    if public and _is_ascii(public):
        return Path(public) / "LianAnZhiDun" / "ray"
    drive = os.environ.get("SystemDrive", "C:")
    return Path(f"{drive}\\Users\\Public\\LianAnZhiDun\\ray")


def configure_ray_environment(env: dict[str, str]) -> Path:
    """Apply Windows-safe Ray settings to a child-process environment."""
    temp = Path(env["RAY_TEMP_DIR"]) if env.get("RAY_TEMP_DIR") else ray_temp_root()
    if not _is_ascii(str(temp)):
        temp = ray_temp_root()
    try:
        temp.mkdir(parents=True, exist_ok=True)
    except OSError:
        temp = Path(tempfile.gettempdir()) / "LianAnZhiDun-ray"
        temp.mkdir(parents=True, exist_ok=True)
    env.setdefault("RAY_TEMP_DIR", str(temp))
    if not _is_ascii(env.get("TEMP")) or not _is_ascii(env.get("TMP")):
        # Chinese user names commonly land here; Ray's raylet fails on them.
        env["TEMP"] = env["TMP"] = str(temp)
    for key, value in RAY_ENVIRONMENT.items():
        env.setdefault(key, value)
    return temp


def configure_worker_environment(env: dict[str, str], app_dir: Path) -> None:
    """Only mutate the environment passed to this experiment's process tree."""
    inherited = env.get("RAY_RUNTIME_ENV_HOOK")
    if inherited and inherited != HOOK:
        env["FL_BACKDOOR_UPSTREAM_RAY_HOOK"] = inherited
    env["FL_BACKDOOR_RUNTIME_APP"] = str(app_dir)
    env["RAY_RUNTIME_ENV_HOOK"] = HOOK
    configure_ray_environment(env)


def probe_report_path() -> Path:
    from fl_backdoor.utils.app_paths import state_root

    return state_root() / PROBE_CACHE


def read_probe_report() -> dict | None:
    try:
        report = json.loads(probe_report_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None


def _write_probe_report(report: dict) -> None:
    path = probe_report_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass


def probe_ray(timeout: int = 180, *, force: bool = False) -> dict:
    """Start Ray once in a child process and prove that a remote task executes."""
    cached = read_probe_report()
    if cached and not force:
        age = time.time() - float(cached.get("checked_at", 0))
        ttl = PROBE_TTL_SECONDS if cached.get("ok") else FAILURE_TTL_SECONDS
        if 0 <= age < ttl:
            return cached

    env = os.environ.copy()
    configure_ray_environment(env)
    report: dict = {
        "ok": False,
        "checked_at": time.time(),
        "checked_at_text": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ray": None,
        "seconds": None,
        "error": "",
        "ray_temp_dir": env.get("RAY_TEMP_DIR"),
    }
    try:
        report["ray"] = importlib.metadata.version("ray")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "fl_backdoor.utils.ray_probe"],
                                cwd=str(Path(__file__).resolve().parents[2]), env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}
        for key in ("ok", "ray", "seconds", "error", "ray_temp_dir"):
            if key in payload:
                report[key] = payload[key]
        if not report["ok"] and not report["error"]:
            report["error"] = (result.stderr or result.stdout or "Ray 自检进程未返回结果")[-1200:]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    _write_probe_report(report)
    return report


def runtime_env_hook(runtime_env: dict | None) -> dict:
    """Preserve Ray options while correcting the local worker import path."""
    result = dict(runtime_env or {})
    upstream = os.environ.get("FL_BACKDOOR_UPSTREAM_RAY_HOOK")
    if upstream and upstream != HOOK:
        module, function = upstream.rsplit(".", 1)
        result = dict(getattr(importlib.import_module(module), function)(result))
    env_vars = dict(result.get("env_vars") or {})
    snapshot = os.environ.get("FL_BACKDOOR_RUNTIME_APP", str(Path(__file__).resolve().parents[2]))
    paths = []
    seen = set()
    for entry in (snapshot, *sys.path):
        path = os.fspath(entry) if entry else os.getcwd()
        key = os.path.normcase(path)
        if key not in seen:
            paths.append(path)
            seen.add(key)
    env_vars["PYTHONPATH"] = os.pathsep.join(paths)
    result["env_vars"] = env_vars
    return result