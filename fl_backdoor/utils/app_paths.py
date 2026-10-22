"""Installation resources are immutable; desktop state belongs to the current user."""

import os
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    return Path(os.environ.get("FL_BACKDOOR_RESOURCE_DIR", SOURCE_ROOT)).expanduser().resolve()


def desktop_mode() -> bool:
    return os.environ.get("FL_BACKDOOR_DESKTOP") == "1"


def user_root() -> Path:
    override = os.environ.get("FL_BACKDOOR_USER_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if desktop_mode():
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "LianAnZhiDun"
    return resource_root()


def state_root() -> Path:
    return user_root() / "platform_state"


def data_root() -> Path:
    return Path(os.environ.get("FL_BACKDOOR_DATA_DIR", resource_root() / "data")).expanduser().resolve()


def results_root() -> Path:
    return Path(os.environ.get("FL_BACKDOOR_RESULTS_DIR", user_root() / "results")).expanduser().resolve()


def initialize_desktop() -> None:
    """Seed a fresh writable configuration; never copy accounts or old results."""
    import shutil

    root = user_root()
    for path in (root, state_root(), root / "logs", root / "cache", root / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    config = root / "pyproject.toml"
    if not config.exists():
        # The launcher holds the per-user instance lock during initialization.
        shutil.copyfile(resource_root() / "pyproject.toml", config)
    os.environ.setdefault("FL_BACKDOOR_CONFIG", str(config))
    os.environ.setdefault("FL_BACKDOOR_DATA_DIR", str(data_root()))
    os.environ.setdefault("FL_BACKDOOR_RESULTS_DIR", str(results_root()))
    os.environ.setdefault("FL_IDENTITY_DATABASE", str(state_root() / "identity.sqlite3"))
    os.environ.setdefault("FL_BACKDOOR_DATABASE", str(state_root() / "catalog.sqlite3"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
    os.environ.setdefault("FL_BACKDOOR_OFFLINE", "1")