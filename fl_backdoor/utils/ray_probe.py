"""Liveness probe for the Ray simulation backend (always run as a child process)."""

import json
import os
import time


def main() -> int:
    report = {
        "ok": False,
        "ray": None,
        "seconds": None,
        "error": "",
        "ray_temp_dir": os.environ.get("RAY_TEMP_DIR"),
    }
    ray = None
    try:
        import ray

        report["ray"] = ray.__version__
        started = time.monotonic()
        ray.init(
            num_cpus=1,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=os.environ.get("RAY_TEMP_DIR"),
            ignore_reinit_error=True,
        )

        @ray.remote
        def _ping():
            return 1

        value = ray.get(_ping.remote(), timeout=60)
        report["seconds"] = round(time.monotonic() - started, 1)
        report["ok"] = value == 1
        if not report["ok"]:
            report["error"] = f"远程任务返回 {value!r}"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if ray is not None:
            try:
                ray.shutdown()
            except Exception:
                pass
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
