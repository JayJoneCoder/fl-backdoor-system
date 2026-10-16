"""Keep local Flower/Ray workers on the immutable experiment snapshot.

Flower 1.29 builds the Ray PYTHONPATH using ':' even on Windows. Use Ray's
runtime-environment hook in owned experiments to supply native path separators;
otherwise cloudpickle can import ClientApp handlers from an editable install.
The runtime driver must also start in the snapshot, since Ray prepends its cwd
to worker sys.path ahead of PYTHONPATH.
"""

import importlib
import os
import sys
from pathlib import Path


HOOK = "fl_backdoor.utils.ray_runtime.runtime_env_hook"


def configure_worker_environment(env: dict[str, str], app_dir: Path) -> None:
    """Only mutate the environment passed to this experiment's process tree."""
    inherited = env.get("RAY_RUNTIME_ENV_HOOK")
    if inherited and inherited != HOOK:
        env["FL_BACKDOOR_UPSTREAM_RAY_HOOK"] = inherited
    env["FL_BACKDOOR_RUNTIME_APP"] = str(app_dir)
    env["RAY_RUNTIME_ENV_HOOK"] = HOOK


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