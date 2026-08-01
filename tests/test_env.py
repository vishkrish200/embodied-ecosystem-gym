from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from ecosystem_gym import EcosystemEnv
from ecosystem_gym.actions import ActionKind


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
