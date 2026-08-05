from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import write_benchmark_report
from .learned_rgb import learned_rgb_gate
from .m3 import collect_rgb_behavior_cloning_data, write_perception_report
from .m4 import write_drive_report
from .m6 import run_m6_viewer_demo, write_m6_report
from .m7 import run_m7_viewer_demo, write_m7_report
from .m8 import run_m8_viewer_demo, write_m8_report
from .m81 import run_m81_viewer_demo, write_m81_report
from .m82 import run_m82_viewer_demo, write_m82_report
from .policies import evaluate_scripted_policy
from .trajectory import replay_and_validate
from .video import write_find_and_eat_regression_video
from .viewer import LocalViewerServer, ViewerSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Embodied Ecosystem Gym tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="run the fixed-seed scripted baseline")
    evaluate.add_argument("--trajectory-dir", type=Path)
    replay = subparsers.add_parser("replay", help="replay and validate a JSONL trajectory")
    replay.add_argument("trajectory", type=Path)
    video = subparsers.add_parser("record-video", help="record the seed-7 Find and eat regression video")
    video.add_argument("output", type=Path)
    benchmark = subparsers.add_parser("benchmark", help="compare raw-random and learned oracle baselines")
    benchmark.add_argument("--task", default="find_and_eat")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--training-episodes", type=int, default=400)
    m3_benchmark = subparsers.add_parser("perception-benchmark", help="report oracle, hybrid, and RGB results")
    m3_benchmark.add_argument("--output", type=Path, required=True)
    m3_benchmark.add_argument("--training-episodes", type=int, default=400)
    collect_bc = subparsers.add_parser("collect-bc", help="write RGB behavior-cloning demonstrations")
    collect_bc.add_argument("--output", type=Path, required=True)
    learned_rgb = subparsers.add_parser("learned-rgb-gate", help="fit and evaluate RGB behavior cloning by appearance split")
    learned_rgb.add_argument("--output", type=Path, required=True)
    learned_rgb.add_argument("--dataset", type=Path)
    m4_benchmark = subparsers.add_parser("m4-benchmark", help="report M4 drive-task robustness")
    m4_benchmark.add_argument("--output", type=Path, required=True)
    m6_benchmark = subparsers.add_parser("m6-benchmark", help="train and evaluate learned M4 drive arbitration")
    m6_benchmark.add_argument("--output", type=Path, required=True)
    m6_benchmark.add_argument("--training-episodes", type=int, default=1200)
    m6_demo = subparsers.add_parser("m6-demo", help="record and replay one learned M6 viewer rollout")
    m6_demo.add_argument("--trace", type=Path, required=True)
    m6_demo.add_argument("--task", choices=("play_when_bored", "competing_drives"), default="competing_drives")
    m6_demo.add_argument("--training-episodes", type=int, default=1200)
    m7_benchmark = subparsers.add_parser("m7-benchmark", help="train and evaluate RGB drive arbitration")
    m7_benchmark.add_argument("--output", type=Path, required=True)
    m7_benchmark.add_argument("--training-episodes", type=int, default=400)
    m7_demo = subparsers.add_parser("m7-demo", help="record and replay one learned RGB drive rollout")
    m7_demo.add_argument("--trace", type=Path, required=True)
    m7_demo.add_argument("--task", choices=("play_when_bored", "competing_drives"), default="competing_drives")
    m7_demo.add_argument("--training-episodes", type=int, default=400)
    m8_benchmark = subparsers.add_parser("m8-benchmark", help="run the frozen sequential RGB recovery protocol")
    m8_benchmark.add_argument("--output", type=Path, required=True)
    m8_demo = subparsers.add_parser("m8-demo", help="record and replay one M8 RGB recovery trace")
    m8_demo.add_argument("--trace", type=Path, required=True)
    m8_demo.add_argument("--seed", type=int, default=7)
    m81_benchmark = subparsers.add_parser("m81-benchmark", help="compare feed-forward and recurrent RGB recovery policies")
    m81_benchmark.add_argument("--output", type=Path, required=True)
    m81_demo = subparsers.add_parser("m81-demo", help="record and replay one recurrent M8.1 rollout")
    m81_demo.add_argument("--trace", type=Path, required=True)
    m82_benchmark = subparsers.add_parser("m82-benchmark", help="run the sealed M8.2 external-validity audit")
    m82_benchmark.add_argument("--output", type=Path, required=True)
    m82_demo = subparsers.add_parser("m82-demo", help="record and replay one frozen recurrent M8.2 rollout")
    m82_demo.add_argument("--trace", type=Path, required=True)
    viewer = subparsers.add_parser("viewer", help="start the thin local live Gym viewer")
    viewer.add_argument("--trace", type=Path)
    viewer.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_scripted_policy(trajectory_dir=args.trajectory_dir)
        print(json.dumps({"success_rate": result.success_rate, "successes": result.successes, "mean_reward": result.mean_reward}))
    elif args.command == "replay":
        print(replay_and_validate(args.trajectory))
    elif args.command == "record-video":
        print(write_find_and_eat_regression_video(args.output))
    elif args.command == "benchmark":
        report = write_benchmark_report(args.output, task_name=args.task, training_episodes=args.training_episodes)
        print(json.dumps(report["comparison"]))
    elif args.command == "perception-benchmark":
        report = write_perception_report(args.output, training_episodes=args.training_episodes)
        print(json.dumps({name: value["heldout"] for name, value in report["results"].items()}))
    elif args.command == "collect-bc":
        print(json.dumps(collect_rgb_behavior_cloning_data(args.output)))
    elif args.command == "learned-rgb-gate":
        report = learned_rgb_gate(args.output, dataset_path=args.dataset)
        print(json.dumps({"heldout_appearance": report["heldout_appearance"], "passed": report["passed"]}))
    elif args.command == "m4-benchmark":
        report = write_drive_report(args.output)
        print(json.dumps(report["robustness"]))
    elif args.command == "m6-benchmark":
        report = write_m6_report(args.output, training_episodes=args.training_episodes)
        print(json.dumps(report["gate"]))
    elif args.command == "m6-demo":
        print(run_m6_viewer_demo(args.trace, task_id=args.task, training_episodes=args.training_episodes))
    elif args.command == "m7-benchmark":
        report = write_m7_report(args.output, training_episodes=args.training_episodes)
        print(json.dumps(report["gate"]))
    elif args.command == "m7-demo":
        print(run_m7_viewer_demo(args.trace, task_id=args.task, training_episodes=args.training_episodes))
    elif args.command == "m8-benchmark":
        report = write_m8_report(args.output)
        print(json.dumps({condition: result["fixed_rgb_scan_recovery_baseline"] for condition, result in report["results"].items()}))
    elif args.command == "m8-demo":
        print(run_m8_viewer_demo(args.trace, seed=args.seed))
    elif args.command == "m81-benchmark":
        report = write_m81_report(args.output)
        print(json.dumps(report["recurrence_verdict"]))
    elif args.command == "m81-demo":
        print(run_m81_viewer_demo(args.trace))
    elif args.command == "m82-benchmark":
        report = write_m82_report(args.output)
        print(json.dumps(report["transfer_verdict"]))
    elif args.command == "m82-demo":
        print(run_m82_viewer_demo(args.trace))
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


if __name__ == "__main__":
    main()
