"""``flwr-simulation`` entry point with the Ray-free backend registered."""

ENTRY_POINT = "fl_backdoor.simulation.entry:main"


def main() -> None:
    from fl_backdoor.simulation.local_backend import register

    register()
    from flwr.simulation.app import flwr_simulation

    flwr_simulation()


if __name__ == "__main__":
    main()
