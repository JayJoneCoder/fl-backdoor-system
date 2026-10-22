"""Run potentially broken CUDA imports out of the API process, with a deadline."""

import asyncio
from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path
import sys
import time
from desktop.runtime import probe


@lru_cache(maxsize=8)
def _cached_probe(executable: str, target: str, visibility: str, minute: int) -> dict:
    return probe(Path(target) if target else None)


async def environment_status(*, refresh: bool = False) -> dict:
    # Short-lived cache avoids reinitializing CUDA for every item of a batch.
    # Each owned worker also checks again immediately before starting Flower.
    if refresh:
        _cached_probe.cache_clear()
    result = await asyncio.to_thread(_cached_probe, sys.executable, os.environ.get("FL_BACKDOOR_TORCH_DIR", ""),
                                    os.environ.get("CUDA_VISIBLE_DEVICES", ""), int(time.monotonic() // 60))
    result = deepcopy(result)
    result["ray"] = ray_status()
    return result


def ray_status() -> dict:
    """Report the cached Ray self-check; the worker refreshes it before a run."""
    from fl_backdoor.utils.ray_runtime import read_probe_report

    report = read_probe_report()
    if not report:
        return {"checked": False, "ok": None, "error": "", "checked_at_text": "",
                "ray": None, "seconds": None, "ray_temp_dir": ""}
    return {
        "checked": True,
        "ok": bool(report.get("ok")),
        "error": report.get("error", ""),
        "checked_at_text": report.get("checked_at_text", ""),
        "ray": report.get("ray"),
        "seconds": report.get("seconds"),
        "ray_temp_dir": report.get("ray_temp_dir", ""),
    }
