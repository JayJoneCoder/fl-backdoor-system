"""Regenerate Flower launchers for the actual private interpreter, not a dev venv."""

from importlib.metadata import distribution
import os
from pathlib import Path
import sys


def create_entry_points(directory: Path) -> None:
    from pip._vendor.distlib.scripts import ScriptMaker

    directory.mkdir(parents=True, exist_ok=True)
    maker = ScriptMaker(None, str(directory))
    maker.executable = sys.executable
    maker.clobber = True
    maker.variants = {""}
    for entry in distribution("flwr").entry_points:
        if entry.group == "console_scripts":
            maker.make(f"{entry.name} = {entry.value}")
    if os.name != "nt":
        for path in directory.iterdir():
            path.chmod(path.stat().st_mode | 0o111)