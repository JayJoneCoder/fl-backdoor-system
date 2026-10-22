"""Owned API child: bind its own socket before publishing the selected port."""

import sys


def main() -> None:
    if sys.stdin.readline().strip() != "start":
        raise SystemExit(1)
    import argparse
    import json
    import os
    from pathlib import Path
    import socket

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        ready = Path(args.ready_file)
        temporary = ready.with_suffix(".tmp")
        temporary.write_text(json.dumps({"port": listener.getsockname()[1]}), encoding="utf-8")
        os.replace(temporary, ready)
        config = uvicorn.Config("backend.main:app", log_level="info")
        uvicorn.Server(config).run(sockets=[listener])


if __name__ == "__main__":
    main()