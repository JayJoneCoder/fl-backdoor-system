"""Exclusive experiment directories, persistent run identities and model archives."""

import json
import os
import tempfile
import uuid
from copy import deepcopy
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .paths import child_path, experiment_path


def build_config_snapshot(document: Mapping) -> bytes:
    """Serialize a run document with the requested and effective partition settings."""
    import tomli_w

    snapshot = deepcopy(dict(document))
    app = snapshot.setdefault("tool", {}).setdefault("flwr", {}).setdefault("app", {})
    config = app.setdefault("config", {})
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be a TOML table.")

    requested_seed = config.get("partition-seed")
    effective_seed = requested_seed if requested_seed is not None else config.get("seed", 42)
    app["partition"] = {
        "requested-mode": config.get("partition-mode", "iid"),
        "requested-alpha": float(config.get("partition-alpha", 0.5)),
        "requested-seed-source": "partition-seed" if requested_seed is not None else "global-seed",
        "effective-mode": config.get("partition-mode", "iid"),
        "effective-alpha": float(config.get("partition-alpha", 0.5)),
        "effective-seed": int(effective_seed),
        "effective-num-clients": int(config.get("num-clients", 10)),
    }
    if requested_seed is not None:
        app["partition"]["requested-seed"] = int(requested_seed)
    return tomli_w.dumps(snapshot).encode("utf-8")


def write_config_snapshot(directory: Path, document: Mapping) -> Path:
    """Create an exclusive immutable configuration snapshot in an experiment directory."""
    path = child_path(directory, "config_snapshot.toml")
    with path.open("xb") as stream:
        stream.write(build_config_snapshot(document))
    return path


def write_run_metadata(directory: Path, data: dict) -> None:
    path = child_path(directory, "run.json")
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".run-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_run_metadata(directory: Path) -> dict:
    path = child_path(directory, "run.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("name") != directory.name:
        raise ValueError("Invalid experiment run metadata.")
    try:
        uuid.UUID(data["run_id"])
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid experiment run ID.") from exc
    return data


NAME_LENGTH_LIMIT = 120  # Matches validate_name's default experiment-name limit.


def _reserve_directory(root: Path, name: str, run_id: str) -> tuple[Path, str]:
    """Create a new directory, adding this run's UUID when the name is already used.

    The UUID suffix keeps every run distinct without rejecting a repeated name:
    an existing (even empty or legacy) directory is never reused or modified.
    """
    directory = experiment_path(root, name)
    try:
        directory.mkdir()
    except FileExistsError:
        pass
    else:
        return directory, name
    for length in (8, 12, len(run_id)):
        suffix = f"-{run_id[:length]}"
        stem = name[: NAME_LENGTH_LIMIT - len(suffix)].rstrip(" .-") or "experiment"
        candidate = f"{stem}{suffix}"
        directory = experiment_path(root, candidate)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory, candidate
    raise FileExistsError(f'Experiment "{name}" could not be given a unique result directory. Choose a new name.')


def reserve_experiment(root: Path, name: str) -> tuple[Path, dict]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    directory, effective_name = _reserve_directory(root, name, run_id)
    metadata = {"name": effective_name, "requested_name": name, "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "starting"}
    try:
        write_run_metadata(directory, metadata)
    except BaseException:
        directory.rmdir()
        raise
    return directory, metadata


def ensure_experiment(root: Path, name: str, *, run_id: str | None = None) -> tuple[Path, dict]:
    directory = experiment_path(root, name)
    if not run_id:
        # Unmanaged runs may repeat a name; reservation adds the UUID suffix.
        return reserve_experiment(root, name)
    if not directory.exists():
        raise ValueError("The managed experiment directory is missing; refusing to create a different run.")
    metadata = read_run_metadata(directory)
    if metadata.get("run_id") != run_id or metadata.get("status") not in {"starting", "running"}:
        raise FileExistsError(f'Experiment "{name}" already exists and cannot be overwritten.')
    return directory, metadata


def save_final_model(state_dict, directory: Path) -> Path:
    import torch

    path = child_path(directory, "final_model.pt")
    # Exclusive creation also protects direct CLI runs from accidental overwrites.
    with path.open("xb") as stream:
        try:
            torch.save(state_dict, stream)
        except BaseException:
            stream.close()
            path.unlink(missing_ok=True)
            raise
    return path