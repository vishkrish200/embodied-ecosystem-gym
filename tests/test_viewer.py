from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from ecosystem_gym.actions import ActionKind
from ecosystem_gym import EcosystemConfig, EcosystemEnv
from ecosystem_gym.viewer import LocalViewerServer, ViewerSession


def action(kind: ActionKind, target: np.ndarray = np.zeros(2, dtype=np.float32), duration: float = 0.1) -> dict:
    return {"kind": int(kind), "target": target, "duration": np.asarray(duration, dtype=np.float32)}


def test_live_session_records_then_replays_the_same_gym_transitions(tmp_path) -> None:
    trace = tmp_path / "viewer.jsonl"
    with ViewerSession(trace_path=trace, episode_id="live-view") as live:
        observation, _ = live.reset(seed=7)
        live.step(action(ActionKind.WALK_TO, observation["food_xy"], duration=5.0))
        live.step(action(ActionKind.PICK_UP))
        _, _, terminated, _, info = live.step(action(ActionKind.CONSUME))
        assert terminated
        assert info["task_success"]
        assert live.snapshot().frame is not None

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert records[0]["record_type"] == "episode_metadata"
    assert records[0]["reset_options"] == {}
    assert records[0]["config"]["observation_mode"] == "state_oracle"
    step_records = records[1:]
    assert [record["step"] for record in step_records] == [1, 2, 3]
    assert {record["episode_id"] for record in step_records} == {"live-view"}

    original_trace = trace.read_text(encoding="utf-8")
    with ViewerSession(trace_path=trace) as replay:
        result = replay.replay(trace)
        assert result.steps == 3
        assert result.task_success
        assert replay.snapshot().terminated
    assert trace.read_text(encoding="utf-8") == original_trace


def test_session_forwards_reset_step_and_render_to_its_environment() -> None:
    class SpyEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def reset(self, *, seed: int | None, options: dict | None):
            self.calls.append(("reset", (seed, options)))
            return {"state": np.asarray([seed])}, {"seed": seed}

        def step(self, received_action: dict):
            self.calls.append(("step", received_action))
            return {"state": np.asarray([2])}, 0.5, False, False, {
                "outcome": "success",
                "task_success": False,
                "environment_version": "test",
            }

        def render(self):
            self.calls.append(("render", None))
            return np.zeros((2, 3, 3), dtype=np.uint8)

        def close(self):
            self.calls.append(("close", None))

    env = SpyEnv()
    session = ViewerSession(env=env)  # type: ignore[arg-type]
    session.reset(seed=12, options={"layout_id": "default"})
    submitted = action(ActionKind.IDLE)
    _, reward, _, _, _ = session.step(submitted)
    frame = session.render()
    session.close()

    assert reward == 0.5
    assert frame is not None and frame.shape == (2, 3, 3)
    assert env.calls[0] == ("reset", (12, {"layout_id": "default"}))
    assert env.calls[1] == ("render", None)
    assert env.calls[2] == ("step", submitted)
    assert [call[0] for call in env.calls] == ["reset", "render", "step", "render", "render", "close"]


def test_local_viewer_server_uses_the_same_session_api() -> None:
    class FastEnv:
        def reset(self, *, seed: int | None, options: dict | None):
            return {"state": np.asarray([seed])}, {"seed": seed}

        def step(self, action: dict):
            return {"state": np.asarray([2])}, 0.0, False, False, {
                "outcome": "success",
                "task_success": False,
                "environment_version": "test",
            }

        def render(self):
            return np.zeros((2, 3, 3), dtype=np.uint8)

        def close(self):
            return None

    with ViewerSession(env=FastEnv()) as session:  # type: ignore[arg-type]
        server = LocalViewerServer(session, port=0)
        try:
            url = server.start()
            with urlopen(f"{url}/api/state") as response:  # noqa: S310 -- loopback test server only.
                assert json.loads(response.read())["step"] == 0
            request = Request(
                f"{url}/api/reset",
                data=json.dumps({"seed": 7}).encode(),
                headers={"Content-Type": "application/json", "X-Viewer-Token": server._control_token},
                method="POST",
            )
            with urlopen(request) as response:  # noqa: S310 -- loopback test server only.
                assert json.loads(response.read())["reset_info"]["seed"] == 7
            with urlopen(f"{url}/api/frame") as response:  # noqa: S310 -- loopback test server only.
                assert response.headers["Content-Type"] == "image/png"
                assert response.read().startswith(b"\x89PNG")
        finally:
            server.close()


def test_local_viewer_rejects_cross_origin_style_control_requests() -> None:
    class FastEnv:
        def reset(self, *, seed: int | None, options: dict | None):
            return {"state": np.asarray([seed])}, {"seed": seed}

        def render(self):
            return np.zeros((2, 3, 3), dtype=np.uint8)

        def close(self):
            return None

    with ViewerSession(env=FastEnv()) as session:  # type: ignore[arg-type]
        server = LocalViewerServer(session, port=0)
        try:
            url = server.start()
            request = Request(
                f"{url}/api/reset",
                data=b'{"seed": 7}',
                headers={"Content-Type": "text/plain"},
                method="POST",
            )
            try:
                urlopen(request)  # noqa: S310 -- loopback test server only.
            except HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("viewer controls must require the per-server token")
            assert session.snapshot().step == 0
        finally:
            server.close()


def test_local_viewer_rejects_non_loopback_hosts() -> None:
    class FastEnv:
        def close(self):
            return None

    with ViewerSession(env=FastEnv()) as session:  # type: ignore[arg-type]
        try:
            LocalViewerServer(session, host="0.0.0.0")
        except ValueError as error:
            assert "loopback" in str(error)
        else:
            raise AssertionError("non-loopback viewer binding must be rejected")


def test_replay_reconstructs_the_recorded_rgb_observation_mode(tmp_path) -> None:
    trace = tmp_path / "rgb-viewer.jsonl"
    with ViewerSession(EcosystemEnv(EcosystemConfig(observation_mode="rgb"), render_mode="rgb_array"), trace_path=trace) as live:
        live.reset(seed=7, options={"layout_id": "m3_train_center", "food_variant": "orange"})
        live.step(action(ActionKind.IDLE))
    with ViewerSession() as replay:
        result = replay.replay(trace)
    assert result.steps == 1


def test_replay_reconstructs_a_non_default_environment_config(tmp_path) -> None:
    trace = tmp_path / "custom-config.jsonl"
    config = EcosystemConfig(max_episode_steps=1, step_penalty_per_second=0.25)
    with ViewerSession(EcosystemEnv(config, render_mode="rgb_array"), trace_path=trace) as live:
        live.reset(seed=7)
        _, _, _, truncated, _ = live.step(action(ActionKind.IDLE))
        assert truncated
    with ViewerSession() as replay:
        assert replay.replay(trace).steps == 1
