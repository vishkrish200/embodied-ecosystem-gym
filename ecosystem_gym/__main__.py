"""Entry points for the evolving Gym and its explicitly named experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .trajectory import replay_and_validate
from .viewer import LocalViewerServer, ViewerSession


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embodied Ecosystem Gym")
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay", help="replay and validate a JSONL trajectory")
    replay.add_argument("trajectory", type=Path)

    viewer = commands.add_parser("viewer", help="start the live Gym viewer")
    viewer.add_argument("--trace", type=Path)
    viewer.add_argument("--port", type=int, default=8765)

    experiment = commands.add_parser("experiment", help="run a frozen milestone experiment")
    experiment.add_argument("args", nargs=argparse.REMAINDER, help="experiment command and its arguments")

    manifest = commands.add_parser(
        "maintenance-policy-manifest",
        help="write the unopened M13.13 policy-family protocol manifest without running an environment",
    )
    manifest.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
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
    elif args.command == "experiment":
        # Keep the active Gym CLI independent of every frozen experiment until one is requested.
        from .experiments.cli import main as run_experiment

        run_experiment(args.args)
    else:
        from .maintenance.policy_protocol import write_policy_family_manifest

        payload = write_policy_family_manifest(args.output)
        print(
            json.dumps(
                {
                    "protocol_fingerprint": payload["protocol_fingerprint"],
                    "status": payload["status"],
                }
            )
        )


if __name__ == "__main__":
    main()
