"""Gate pip startup until the launcher owns its complete process tree."""

import sys

if __name__ == "__main__":
    if sys.stdin.readline().strip() != "start":
        raise SystemExit(1)
    import runpy

    sys.argv[0] = "pip"
    runpy.run_module("pip", run_name="__main__")