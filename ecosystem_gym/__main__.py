from __future__ import annotations

import os

# Parallel M13.5 workers must not each create a nested BLAS thread pool.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

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
from .m83 import write_m83_report
from .m84 import run_m84_viewer_demo, write_m84_report
from .m85 import run_m85_viewer_demo, write_m85_report
from .m86 import run_m86_viewer_demo, write_m86_report
from .m87 import run_m87_viewer_demo, write_m87_report
from .m88 import run_m88_viewer_demo, write_m88_report
from .m89 import run_m89_viewer_demo, write_m89_report
from .m9 import run_m9_viewer_demo, write_m9_report
from .m91 import run_m91_viewer_demo, write_m91_report
from .m10 import run_m10_viewer_demo, write_m10_report
from .m11 import write_m11_report
from .m12 import run_m12_viewer_demo, write_m12_report
from .m13 import write_m13_audit_report, write_m13_report, write_m13_training_report
from .m131 import write_m131_report, write_m131_training_report
from .m132 import write_m132_training_report
from .m133 import write_m133_training_report
from .m134 import write_m134_training_report
from .m135 import write_m135_training_report
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
    m83_diagnostics = subparsers.add_parser("m83-diagnostics", help="run one-factor diagnostics on frozen M8.1 policies")
    m83_diagnostics.add_argument("--output", type=Path, required=True)
    m84_validation = subparsers.add_parser("m84-validation", help="validate learned RGB heatmap grounding before any M8.2 audit")
    m84_validation.add_argument("--output", type=Path, required=True)
    m84_demo = subparsers.add_parser("m84-demo", help="record and replay one learned M8.4 heatmap rollout")
    m84_demo.add_argument("--trace", type=Path, required=True)
    m85_benchmark = subparsers.add_parser("m85-benchmark", help="run the sealed M8.5 scan_v2 external audit")
    m85_benchmark.add_argument("--output", type=Path, required=True)
    m85_demo = subparsers.add_parser("m85-demo", help="record and replay one frozen M8.5 heatmap rollout")
    m85_demo.add_argument("--trace", type=Path, required=True)
    m86_diagnostics = subparsers.add_parser("m86-diagnostics", help="run development-only M8.6 factor diagnostics")
    m86_diagnostics.add_argument("--output", type=Path, required=True)
    m86_demo = subparsers.add_parser("m86-demo", help="record and replay one M8.6 blue-grounding diagnostic")
    m86_demo.add_argument("--trace", type=Path, required=True)
    m87_validation = subparsers.add_parser("m87-validation", help="validate the larger-context M8.7 RGB grounder")
    m87_validation.add_argument("--output", type=Path, required=True)
    m87_demo = subparsers.add_parser("m87-demo", help="record and replay one M8.7 blue-grounding rollout")
    m87_demo.add_argument("--trace", type=Path, required=True)
    m88_benchmark = subparsers.add_parser("m88-benchmark", help="run the sealed M8.8 scan_v2 external audit")
    m88_benchmark.add_argument("--output", type=Path, required=True)
    m88_demo = subparsers.add_parser("m88-demo", help="record and replay one frozen M8.8 rollout")
    m88_demo.add_argument("--trace", type=Path, required=True)
    m89_benchmark = subparsers.add_parser("m89-benchmark", help="run the sealed M8.9 visual-morphology audit")
    m89_benchmark.add_argument("--output", type=Path, required=True)
    m89_demo = subparsers.add_parser("m89-demo", help="record and replay one frozen M8.9 rollout")
    m89_demo.add_argument("--trace", type=Path, required=True)
    m9_benchmark = subparsers.add_parser("m9-benchmark", help="run the frozen M9 integrated RGB maintenance protocol")
    m9_benchmark.add_argument("--output", type=Path, required=True)
    m9_benchmark.add_argument("--audit", action="store_true", help="score the one-shot sealed M9 audit rather than validation")
    m9_demo = subparsers.add_parser("m9-demo", help="record and replay one M9 integrated RGB rollout")
    m9_demo.add_argument("--trace", type=Path, required=True)
    m91_diagnosis = subparsers.add_parser("m91-diagnosis", help="separate M9 distractor choice from forced recovery without retraining")
    m91_diagnosis.add_argument("--output", type=Path, required=True)
    m91_demo = subparsers.add_parser("m91-demo", help="record and replay one M9.1 forced-recovery rollout")
    m91_demo.add_argument("--trace", type=Path, required=True)
    m10_benchmark = subparsers.add_parser("m10-validation", help="validate frozen persistent-maintenance mechanics and observability")
    m10_benchmark.add_argument("--output", type=Path, required=True)
    m10_demo = subparsers.add_parser("m10-demo", help="record and replay one privileged M10 maintenance trace")
    m10_demo.add_argument("--trace", type=Path, required=True)
    m11_baseline = subparsers.add_parser("m11-baseline", help="run the frozen M9 baseline and write its persistent failure atlas")
    m11_baseline.add_argument("--output", type=Path, required=True)
    m12_validation = subparsers.add_parser("m12-validation", help="validate the learned RGB persistent-maintenance successor")
    m12_validation.add_argument("--output", type=Path, required=True)
    m12_demo = subparsers.add_parser("m12-demo", help="record and replay one learned M12 maintenance trace")
    m12_demo.add_argument("--trace", type=Path, required=True)
    m13_validation = subparsers.add_parser("m13-validation", help="train and evaluate the frozen M13 state-oracle RL baseline")
    m13_validation.add_argument("--output", type=Path, required=True)
    m13_train = subparsers.add_parser("m13-train", help="write frozen M13 development-only policy artifacts")
    m13_train.add_argument("--output", type=Path, required=True)
    m13_audit = subparsers.add_parser("m13-audit", help="run the one-shot sealed M13 audit after a passing validation")
    m13_audit.add_argument("--validation-report", type=Path, required=True)
    m13_audit.add_argument("--output", type=Path, required=True)
    m131_validation = subparsers.add_parser("m131-validation", help="run the one-shot M13.1 cycle-reward validation")
    m131_validation.add_argument("--output", type=Path, required=True)
    m131_train = subparsers.add_parser("m131-train", help="write M13.1 development-only policy artifacts")
    m131_train.add_argument("--output", type=Path, required=True)
    m132_train = subparsers.add_parser("m132-train", help="write M13.2 development-only Double-DQN policy artifacts")
    m132_train.add_argument("--output", type=Path, required=True)
    m133_train = subparsers.add_parser("m133-train", help="train and gate the recovery-safe M13.3 development baseline")
    m133_train.add_argument("--output", type=Path, required=True)
    m134_train = subparsers.add_parser("m134-train", help="train and gate the public macro-precondition M13.4 development baseline")
    m134_train.add_argument("--output", type=Path, required=True)
    m135_train = subparsers.add_parser("m135-train", help="run the parallel M13.5 update-ratio development diagnostic")
    m135_train.add_argument("--output", type=Path, required=True)
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
    elif args.command == "m83-diagnostics":
        report = write_m83_report(args.output)
        print(json.dumps(report["initial_frame_diagnostics"]))
    elif args.command == "m84-validation":
        report = write_m84_report(args.output)
        print(json.dumps(report["validation_gate"]))
    elif args.command == "m84-demo":
        print(run_m84_viewer_demo(args.trace))
    elif args.command == "m85-benchmark":
        report = write_m85_report(args.output)
        print(json.dumps(report["external_validity_gate"]))
    elif args.command == "m85-demo":
        print(run_m85_viewer_demo(args.trace))
    elif args.command == "m86-diagnostics":
        report = write_m86_report(args.output)
        print(json.dumps({condition: result["frozen_m84_rgb_heatmap"] for condition, result in report["results"].items()}))
    elif args.command == "m86-demo":
        print(run_m86_viewer_demo(args.trace))
    elif args.command == "m87-validation":
        report = write_m87_report(args.output)
        print(json.dumps(report["validation_gate"]))
    elif args.command == "m87-demo":
        print(run_m87_viewer_demo(args.trace))
    elif args.command == "m88-benchmark":
        report = write_m88_report(args.output)
        print(json.dumps(report["external_validity_gate"]))
    elif args.command == "m88-demo":
        print(run_m88_viewer_demo(args.trace))
    elif args.command == "m89-benchmark":
        report = write_m89_report(args.output)
        print(json.dumps(report["external_validity_gate"]))
    elif args.command == "m89-demo":
        print(run_m89_viewer_demo(args.trace))
    elif args.command == "m9-benchmark":
        report = write_m9_report(args.output, audit=args.audit)
        print(json.dumps(report["gate"]))
    elif args.command == "m9-demo":
        print(run_m9_viewer_demo(args.trace))
    elif args.command == "m91-diagnosis":
        report = write_m91_report(args.output)
        print(json.dumps(report["verdict"]))
    elif args.command == "m91-demo":
        print(run_m91_viewer_demo(args.trace))
    elif args.command == "m10-validation":
        report = write_m10_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m10-demo":
        print(run_m10_viewer_demo(args.trace))
    elif args.command == "m11-baseline":
        report = write_m11_report(args.output)
        print(json.dumps({"diagnostic_complete": report["diagnostic_complete"]}))
    elif args.command == "m12-validation":
        report = write_m12_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m12-demo":
        print(run_m12_viewer_demo(args.trace))
    elif args.command == "m13-validation":
        report = write_m13_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m13-train":
        report = write_m13_training_report(args.output)
        print(json.dumps(report["policy_artifacts"]))
    elif args.command == "m13-audit":
        report = write_m13_audit_report(args.output, validation_report_path=args.validation_report)
        print(json.dumps(report["gate"]))
    elif args.command == "m131-validation":
        report = write_m131_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m131-train":
        report = write_m131_training_report(args.output)
        print(json.dumps(report["policy_artifacts"]))
    elif args.command == "m132-train":
        report = write_m132_training_report(args.output)
        print(json.dumps(report["policy_artifacts"]))
    elif args.command == "m133-train":
        report = write_m133_training_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m134-train":
        report = write_m134_training_report(args.output)
        print(json.dumps(report["gate"]))
    elif args.command == "m135-train":
        report = write_m135_training_report(args.output)
        print(json.dumps(report["gate"]))
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
