from __future__ import annotations

import numpy as np
import json
from gymnasium.utils.env_checker import check_env

from ecosystem_gym import EcosystemConfig, EcosystemEnv
from ecosystem_gym.actions import ActionKind
from ecosystem_gym.benchmark import benchmark, write_benchmark_report
from ecosystem_gym.m3 import collect_rgb_behavior_cloning_data, perception_benchmark
from ecosystem_gym.learned_rgb import learned_rgb_gate
from ecosystem_gym.policies import FIXED_EVALUATION_SEEDS, evaluate_scripted_policy, scripted_find_and_eat
from ecosystem_gym.trajectory import replay_and_validate


def action(kind: ActionKind, target: np.ndarray = np.zeros(2, dtype=np.float32), duration: float = 1.0) -> dict:
    return {"kind": int(kind), "target": target, "duration": np.asarray(duration, dtype=np.float32)}


def test_gymnasium_contract() -> None:
    check_env(EcosystemEnv(), skip_render_check=True)


def test_seeded_resets_are_reproducible() -> None:
    first, _ = EcosystemEnv().reset(seed=7)
    second, _ = EcosystemEnv().reset(seed=7)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_find_eat_transition_is_explicit() -> None:
    env = EcosystemEnv()
    observation, _ = env.reset(seed=7)
    observation, _, terminated, _, info = env.step(action(ActionKind.WALK_TO, observation["food_xy"], duration=5.0))
    assert not terminated
    observation, _, terminated, _, info = env.step(action(ActionKind.PICK_UP))
    assert info["outcome"] == "success"
    observation, reward, terminated, _, info = env.step(action(ActionKind.CONSUME))
    assert terminated
    assert info["task_success"]
    assert reward > 0


def test_mujoco_world_renders_a_room() -> None:
    env = EcosystemEnv(render_mode="rgb_array")
    env.reset(seed=7)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert env.model.ngeom >= 7  # floor, four walls, creature, and food
    env.close()


def test_scripted_policy_clears_fixed_seed_suite_and_replays(tmp_path) -> None:
    result = evaluate_scripted_policy(trajectory_dir=tmp_path)
    assert result.seeds == FIXED_EVALUATION_SEEDS
    assert result.success_rate >= 0.95
    for seed in FIXED_EVALUATION_SEEDS:
        replay = replay_and_validate(tmp_path / f"find-eat_seed-{seed:04d}.jsonl")
        assert replay.steps == 3
        assert replay.task_success


def test_single_trajectory_has_an_explicit_success_transition(tmp_path) -> None:
    path = tmp_path / "find-eat.jsonl"
    episode = scripted_find_and_eat(7, trajectory_path=path)
    assert episode.task_success
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"outcome":"success"' in lines[-1]


def test_legacy_step_only_trace_replays_after_the_metadata_schema_upgrade(tmp_path) -> None:
    current = tmp_path / "current.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    scripted_find_and_eat(7, trajectory_path=current)
    records = [json.loads(line) for line in current.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["schema_version"] = "0.1"
        record["observation"].pop("toy_xy", None)
    legacy.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    assert replay_and_validate(legacy).task_success


def test_registered_layout_reset_is_seeded_and_reported() -> None:
    env = EcosystemEnv()
    first, first_info = env.reset(seed=11, options={"layout_id": "heldout_northwest"})
    second, second_info = env.reset(seed=11, options={"layout_id": "heldout_northwest"})
    assert first_info["layout_id"] == "heldout_northwest"
    assert first_info == second_info
    assert np.array_equal(first["agent_xy"], second["agent_xy"])
    assert np.array_equal(first["food_xy"], second["food_xy"])
    env.close()


def test_benchmark_report_compares_learned_oracle_with_raw_random(tmp_path) -> None:
    path = tmp_path / "benchmark.json"
    report = write_benchmark_report(path, training_episodes=40)
    assert path.is_file()
    assert report["comparison"]["learned_beats_random"]
    baselines = report["baselines"]
    assert baselines["state_oracle_q_learning"]["heldout"]["success_rate"] > baselines["raw_random"]["heldout"]["success_rate"]


def test_hybrid_and_rgb_observations_are_space_valid_without_oracle_coordinates() -> None:
    for mode in ("hybrid", "rgb"):
        env = EcosystemEnv(EcosystemConfig(observation_mode=mode))
        observation, _ = env.reset(seed=7, options={"layout_id": "m3_train_center"})
        assert env.observation_space.contains(observation)
        assert "agent_xy" not in observation
        assert "food_xy" not in observation
        if mode == "hybrid":
            assert observation["food_detection"][0] == 1.0
        env.close()


def test_m3_perception_report_separates_modes_and_recovery() -> None:
    report = perception_benchmark(training_episodes=40)
    for name in ("state_oracle_q_learning", "hybrid_visual_servo", "rgb_visual_servo"):
        heldout = report["results"][name]["heldout"]
        assert heldout["clean"]["success_rate"] == 1.0
        assert heldout["disturbed"]["disturbed_success_rate"] == 1.0
        assert heldout["disturbed"]["recovery_actions"] == heldout["disturbed"]["episodes"]
    assert report["results"]["rgb_visual_servo"]["heldout"]["disturbed"]["failure_reasons"]


def test_rgb_behavior_cloning_data_contains_frames_and_teacher_actions(tmp_path) -> None:
    path = tmp_path / "rgb-bc.npz"
    summary = collect_rgb_behavior_cloning_data(path, seeds=(7,))
    dataset = np.load(path)
    assert summary["samples"] == len(dataset["rgb"])
    assert dataset["rgb"].shape[1:] == (64, 64, 3)
    assert set(dataset["action_kind"]) <= {int(kind) for kind in ActionKind}
    assert set(dataset["food_variant"]) == {"red", "orange"}


def test_learned_rgb_gate_is_rgb_only_and_reports_heldout_appearance(tmp_path) -> None:
    report = learned_rgb_gate(tmp_path / "learned-rgb.json", dataset_path=tmp_path / "learned-rgb.npz")
    assert report["policy"] == "numpy_ridge_rgb_behavior_cloning"
    assert report["heldout_appearance"]["episodes"] == 40
    assert report["heldout_random_rgb"]["episodes"] == 40
    assert report["passed"]
