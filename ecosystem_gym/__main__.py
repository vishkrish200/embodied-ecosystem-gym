"""Entry points for the evolving Gym and its explicitly named experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .trajectory import replay_and_validate
from .viewer import LocalViewerServer, ViewerSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Embodied Ecosystem Gym")
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay", help="replay and validate a JSONL trajectory")
    replay.add_argument("trajectory", type=Path)

    viewer = commands.add_parser("viewer", help="start the live Gym viewer")
    viewer.add_argument("--trace", type=Path)
    viewer.add_argument("--port", type=int, default=8765)

    experiment = commands.add_parser("experiment", help="run a frozen milestone experiment")
    experiment.add_argument("args", nargs=argparse.REMAINDER, help="experiment command and its arguments")

    args = parser.parse_args()
    if args.command == "replay":
        print(replay_and_validate(args.trajectory))
    elif args.command == "viewer":
        with ViewerSession(trace_path=args.trace) as session:
            server = LocalViewerServer(session, port=args.port)
            print(f"Viewer running at {server.url}; press Ctrl-C to stop.")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.close()
    else:
        # Keep the active Gym CLI independent of every frozen experiment until one is requested.
        from .experiments.cli import main as run_experiment

        run_experiment(args.args)


if __name__ == "__main__":
    main()
