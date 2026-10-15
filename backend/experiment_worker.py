"""Private Flower runtime entry point, executed only inside an owned process tree."""

import sys


def verified_exit_code(cli_exit: int, runs) -> int:
    """A log-stream exit of zero is not proof that training completed."""
    if cli_exit:
        return cli_exit
    if len(runs) != 1:
        raise RuntimeError("Cannot verify Flower run outcome; refusing to report success.")
    run = runs[0]
    print(f"[Runner] Flower result: {run.status.status}:{run.status.sub_status} {run.status.details}", flush=True)
    return 0 if run.status.status == "finished" and run.status.sub_status == "completed" else 1


def main() -> int:
    # Do not import heavy libraries or create child processes before ownership.
    if sys.stdin.readline().strip() != "start":
        return 1
    import argparse
    import os
    import subprocess
    import time
    from importlib.metadata import distribution
    from pathlib import Path

    import grpc
    import tomli
    from flwr.proto.control_pb2 import ListRunsRequest
    from flwr.proto.control_pb2_grpc import ControlStub

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--address", required=True)
    args = parser.parse_args()
    entries = {entry.name: entry.value for entry in distribution("flwr").entry_points}
    superlink_entry = entries["flower-superlink"]
    module, function = superlink_entry.split(":", 1)
    boot = f"from {module} import {function}; {function}()"
    output = Path(os.environ["FLWR_HOME"]) / "superlink.log"
    with output.open("w", encoding="utf-8") as logs:
        superlink = subprocess.Popen([
            sys.executable, "-u", "-c", boot, "--insecure", "--simulation",
            "--isolation", "subprocess", "--control-api-address", args.address,
            "--serverappio-api-address", "127.0.0.1:0",
        ], stdout=logs, stderr=subprocess.STDOUT, env=os.environ.copy())
        print(f"[Runner] Starting isolated SuperLink on {args.address}...", flush=True)
        with grpc.insecure_channel(args.address) as channel:
            stub = ControlStub(channel)
            deadline = time.monotonic() + 45
            while True:
                if superlink.poll() is not None:
                    print(output.read_text(encoding="utf-8", errors="replace"), flush=True)
                    raise RuntimeError(f"SuperLink exited with code {superlink.returncode} before becoming ready.")
                try:
                    stub.ListRuns(ListRunsRequest(), timeout=1)
                    break
                except grpc.RpcError:
                    if time.monotonic() >= deadline:
                        print(output.read_text(encoding="utf-8", errors="replace"), flush=True)
                        raise RuntimeError("Isolated SuperLink startup timed out.")
                    time.sleep(0.1)
            cli = next(entry for entry in distribution("flwr").entry_points if entry.name == "flwr").load()
            config = tomli.loads((Path(args.project) / "pyproject.toml").read_text(encoding="utf-8"))
            clients = config["tool"]["flwr"]["app"]["config"].get("num-clients", 10)
            sys.argv = ["flwr", "run", args.project, "project-runtime", "--stream",
                        "--federation-config", f"num-supernodes={clients}"]
            cli_exit = 0
            try:
                cli()
            except SystemExit as exc:
                cli_exit = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
            if cli_exit:
                return cli_exit
            response = stub.ListRuns(ListRunsRequest(limit=1), timeout=5)
            runs = list(response.run_dict.values())
            result = verified_exit_code(cli_exit, runs)
            if result:
                print(output.read_text(encoding="utf-8", errors="replace"), flush=True)
            return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Runner] Worker failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)