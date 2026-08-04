"""Small reproducible regression-video helper for the M1 vertical slice."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .actions import ActionKind
from .env import EcosystemEnv
from .policies import skill_action


def write_find_and_eat_regression_video(path: str | Path, *, seed: int = 7) -> Path:
    """Record a portable MP4 by exercising the public skill interface."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = EcosystemEnv(render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)
    frames = [env.render()]
    try:
        while float(np.linalg.norm(observation["food_xy"] - observation["agent_xy"])) > env.config.pickup_radius / 2:
            observation, _, terminated, truncated, _ = env.step(
                skill_action(ActionKind.WALK_TO, observation["food_xy"], 0.1)
            )
            if terminated or truncated:
                raise RuntimeError("walk unexpectedly ended the regression episode")
            frames.append(env.render())
        for action in (
            skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1),
            skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1),
        ):
            observation, _, _, _, _ = env.step(action)
            frames.append(env.render())
    finally:
        env.close()
    first = frames[0]
    height, width, channels = first.shape
    if channels != 3:
        raise RuntimeError("MuJoCo renderer did not produce RGB frames")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "10",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to write regression video: {stderr}")
    return output
