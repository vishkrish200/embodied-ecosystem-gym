from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .actions import ActionKind, ActionOutcome, SkillAction
from .config import EcosystemConfig
from .drives import Drives
from .perception import food_detection_from_rgb
from .tasks import LayoutSpec, TaskSpec, get_layout, get_task


FOOD_VARIANTS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.92, 0.20, 0.16, 1.0),
    "orange": (0.95, 0.48, 0.08, 1.0),
    "blue": (0.16, 0.38, 0.92, 1.0),
    "purple": (0.60, 0.24, 0.78, 1.0),
}
_TOY_VARIANTS = {
    "ball": "toy_ball_geom",
    "cube": "toy_cube_geom",
    "capsule": "toy_capsule_geom",
}
_AGENT_SHAPE_VARIANTS = {
    "sphere": "agent_geom",
    "capsule": "agent_capsule_geom",
    "box": "agent_box_geom",
}
_FOOD_SHAPE_VARIANTS = {
    "sphere": "food_geom",
    "capsule": "food_capsule_geom",
    "box": "food_box_geom",
}
_DYNAMICS_VARIANTS = {
    "nominal": (1.0, 0.9),
    "grippy": (0.90, 1.2),
    "slippery": (1.10, 0.25),
}
_LIGHTING_VARIANTS = {
    "nominal": ((0.10, 0.10, 0.10), (0.40, 0.40, 0.40)),
    # This reduces illumination without changing the cyan agent's visibility.
    "dim": ((0.06, 0.06, 0.06), (0.28, 0.28, 0.28)),
}


_MODEL_XML = """
<mujoco model="embodied_ecosystem_m1">
  <option timestep="0.02" gravity="0 0 -9.81" integrator="Euler"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default>
    <geom friction="0.9 0.1 0.1" rgba="0.65 0.65 0.65 1"/>
    <joint damping="2"/>
  </default>
  <worldbody>
    <light pos="0 0 3" directional="false" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.18 0.21 0.18 1"/>
    <geom name="wall_north" type="box" pos="0 1.05 0.18" size="1.05 0.05 0.18" rgba="0.35 0.35 0.38 1"/>
    <geom name="wall_south" type="box" pos="0 -1.05 0.18" size="1.05 0.05 0.18" rgba="0.35 0.35 0.38 1"/>
    <geom name="wall_east" type="box" pos="1.05 0 0.18" size="0.05 1.05 0.18" rgba="0.35 0.35 0.38 1"/>
    <geom name="wall_west" type="box" pos="-1.05 0 0.18" size="0.05 1.05 0.18" rgba="0.35 0.35 0.38 1"/>
    <body name="agent" pos="0 0 0.12">
      <joint name="agent_x" type="slide" axis="1 0 0" range="-0.9 0.9"/>
      <joint name="agent_y" type="slide" axis="0 1 0" range="-0.9 0.9"/>
      <geom name="agent_geom" type="sphere" size="0.12" rgba="0.18 0.55 0.95 1"/>
      <geom name="agent_capsule_geom" type="capsule" size="0.065 0.070" euler="0 90 0" contype="0" conaffinity="0" rgba="0.18 0.55 0.95 0"/>
      <geom name="agent_box_geom" type="box" size="0.115 0.075 0.100" contype="0" conaffinity="0" rgba="0.18 0.55 0.95 0"/>
      <camera name="agent_cam_center" pos="0 0 1.5" fovy="75"/>
      <camera name="agent_cam_left" pos="-0.08 0.04 1.5" fovy="75"/>
      <camera name="agent_cam_right" pos="0.08 -0.04 1.5" fovy="75"/>
      <camera name="agent_cam_scan_north" pos="0 0.52 1.15" fovy="48" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_east" pos="0.52 0 1.15" fovy="48" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_south" pos="0 -0.52 1.15" fovy="48" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_west" pos="-0.52 0 1.15" fovy="48" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_v2_north" pos="0 0.52 1.15" fovy="90" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_v2_east" pos="0.52 0 1.15" fovy="90" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_v2_south" pos="0 -0.52 1.15" fovy="90" xyaxes="1 0 0 0 1 0"/>
      <camera name="agent_cam_scan_v2_west" pos="-0.52 0 1.15" fovy="90" xyaxes="1 0 0 0 1 0"/>
    </body>
    <body name="food" pos="0 0 0.06">
      <joint name="food_x" type="slide" axis="1 0 0" range="-0.9 0.9"/>
      <joint name="food_y" type="slide" axis="0 1 0" range="-0.9 0.9"/>
      <geom name="food_geom" type="sphere" size="0.06" contype="0" conaffinity="0" rgba="0.92 0.20 0.16 1"/>
      <geom name="food_capsule_geom" type="capsule" size="0.040 0.045" euler="0 90 0" contype="0" conaffinity="0" rgba="0.92 0.20 0.16 0"/>
      <geom name="food_box_geom" type="box" size="0.065 0.040 0.040" contype="0" conaffinity="0" rgba="0.92 0.20 0.16 0"/>
    </body>
    <body name="toy" pos="0 0 0.08">
      <joint name="toy_x" type="slide" axis="1 0 0" range="-0.9 0.9"/>
      <joint name="toy_y" type="slide" axis="0 1 0" range="-0.9 0.9"/>
      <geom name="toy_ball_geom" type="sphere" size="0.08" contype="0" conaffinity="0" rgba="0.95 0.78 0.10 1"/>
      <geom name="toy_cube_geom" type="box" size="0.075 0.075 0.075" contype="0" conaffinity="0" rgba="0.95 0.78 0.10 0"/>
      <geom name="toy_capsule_geom" type="capsule" size="0.055 0.10" contype="0" conaffinity="0" rgba="0.95 0.78 0.10 0"/>
    </body>
    <body name="rest" pos="0 0 0.04">
      <joint name="rest_x" type="slide" axis="1 0 0" range="-0.9 0.9"/>
      <joint name="rest_y" type="slide" axis="0 1 0" range="-0.9 0.9"/>
      <geom name="rest_geom" type="box" size="0.13 0.10 0.04" contype="0" conaffinity="0" rgba="0.28 0.82 0.48 1"/>
    </body>
    <body name="distractor" pos="0 0 0.06">
      <joint name="distractor_x" type="slide" axis="1 0 0" range="-0.9 0.9"/>
      <joint name="distractor_y" type="slide" axis="0 1 0" range="-0.9 0.9"/>
      <geom name="distractor_geom" type="sphere" size="0.06" contype="0" conaffinity="0" rgba="0.92 0.20 0.16 0"/>
    </body>
    <geom name="m8_geometry_geom" type="box" pos="0 0 0.18" size="0.18 0.18 0.18" contype="0" conaffinity="0" rgba="0.32 0.32 0.34 0"/>
    <geom name="m81_landmark_geom" type="capsule" pos="-0.56 0.46 0.16" size="0.06 0.16" contype="0" conaffinity="0" rgba="0.20 0.70 0.34 0"/>
    <geom name="m82_landmark_geom" type="box" pos="0.62 -0.56 0.12" size="0.08 0.08 0.12" contype="0" conaffinity="0" rgba="0.24 0.76 0.48 0"/>
    <camera name="overview" pos="0 -2.8 2.8" xyaxes="1 0 0 0 0.7 0.7"/>
  </worldbody>
</mujoco>
"""


@dataclass(slots=True)
class WorldState:
    holding_food: bool
    food_consumed: bool
    drives: Drives
    step_count: int
    task_id: str = "find_and_eat"
    toy_played: bool = False
    boundary_contacts: int = 0
    feed_cycles: int = 0
    play_cycles: int = 0
    rest_cycles: int = 0
    food_available: bool = True
    food_respawn_remaining: float = 0.0


class EcosystemEnv(gym.Env[dict[str, Any], dict[str, np.ndarray | int]]):
    """A deterministic MuJoCo environment with state, hybrid, and RGB modes.

    The policy-facing action and state-oracle observation contracts intentionally
    match M0. MuJoCo now owns the room geometry and body transforms, while this
    class supplies the task loop, drives, rewards, explicit outcomes, and replay
    boundary. Hybrid detections are computed from pixels, while RGB exposes only
    the camera frame, drives, held state, and prior outcome.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    _SCAN_SECTORS = ("north", "east", "south", "west")

    def __init__(
        self,
        config: EcosystemConfig | None = None,
        render_mode: Literal["rgb_array"] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EcosystemConfig()
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        radius = self.config.world_radius
        self.action_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(ActionKind)),
                "target": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "duration": spaces.Box(
                    low=np.asarray(0.1, dtype=np.float32),
                    high=np.asarray(5.0, dtype=np.float32),
                    shape=(),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Dict(self._observation_spaces(radius))
        self.model = mujoco.MjModel.from_xml_string(_MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self._agent_qpos = self._qpos_indices("agent_x", "agent_y")
        self._food_qpos = self._qpos_indices("food_x", "food_y")
        self._toy_qpos = self._qpos_indices("toy_x", "toy_y")
        self._rest_qpos = self._qpos_indices("rest_x", "rest_y")
        self._distractor_qpos = self._qpos_indices("distractor_x", "distractor_y")
        self._state: WorldState | None = None
        self._prior_outcome = ActionOutcome.SUCCESS
        self._active_layout: LayoutSpec = get_layout("default")
        self._renderer: mujoco.Renderer | None = None
        self._observation_renderer: mujoco.Renderer | None = None
        self._camera_variant = "center"
        self._disturbance_step: int | None = None
        self._disturbance_occurred = False
        self._recovery_pending = False
        self._post_disturbance_pending = False
        self._active_task: TaskSpec = get_task("find_and_eat")
        self._food_variant = "red"
        self._toy_variant = "ball"
        self._dynamics_variant = "nominal"
        self._lighting_variant = "nominal"
        self._movement_speed_scale = 1.0
        self._camera_control = "fixed"
        self._scan_sector = "north"
        self._blocked_distractor = False
        self._geometry_variant = "default"
        self._agent_shape_variant = "sphere"
        self._food_shape_variant = "sphere"
        self._event_relocation_on_first_pickup = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        reset_options = options or {}
        task_name = str(reset_options.get("task_id", reset_options.get("task", "find_and_eat")))
        self._active_task = get_task(task_name)
        explicit_task = "task_id" in reset_options or "task" in reset_options
        layout_name = reset_options.get(
            "layout_id", self._active_task.train_layout_ids[0] if explicit_task else "default"
        )
        self._active_layout = get_layout(str(layout_name))
        self._camera_variant = str(reset_options.get("camera_variant", "center"))
        if self._camera_variant not in {"center", "left", "right"}:
            raise ValueError("camera_variant must be center, left, or right")
        disturbance_step = reset_options.get("disturbance_step")
        self._disturbance_step = int(disturbance_step) if disturbance_step is not None else None
        self._disturbance_occurred = False
        self._recovery_pending = False
        self._post_disturbance_pending = False
        self._camera_control = str(reset_options.get("camera_control", "fixed"))
        if self._camera_control not in {"fixed", "scan", "scan_v2"}:
            raise ValueError("camera_control must be fixed, scan, or scan_v2")
        self._scan_sector = str(reset_options.get("initial_scan_sector", "north"))
        if self._scan_sector not in self._SCAN_SECTORS:
            raise ValueError(f"initial_scan_sector must be one of {self._SCAN_SECTORS}")
        self._blocked_distractor = bool(reset_options.get("blocked_distractor", False))
        self._event_relocation_on_first_pickup = bool(reset_options.get("event_relocation_on_first_pickup", False))
        self._geometry_variant = str(reset_options.get("geometry_variant", "default"))
        if self._geometry_variant not in {"default", "unseen_block", "m81_landmark", "m82_landmark"}:
            raise ValueError("geometry_variant must be default, unseen_block, m81_landmark, or m82_landmark")
        mujoco.mj_resetData(self.model, self.data)
        self._configure_variants(
            food_variant=str(reset_options.get("food_variant", "red")),
            toy_variant=str(reset_options.get("toy_variant", "ball")),
            dynamics_variant=str(reset_options.get("dynamics_variant", "nominal")),
            lighting_variant=str(reset_options.get("lighting_variant", "nominal")),
            agent_shape_variant=str(reset_options.get("agent_shape_variant", "sphere")),
            food_shape_variant=str(reset_options.get("food_shape_variant", "sphere")),
        )
        self._set_agent_xy(np.asarray(self._active_layout.agent_xy, dtype=np.float32))
        self._set_food_xy(self._active_layout.sample_food_xy(self.np_random))
        self._set_toy_xy(np.asarray(self._active_layout.toy_xy, dtype=np.float32))
        self._set_rest_xy(np.asarray(self._active_layout.rest_xy, dtype=np.float32))
        distractor_xy = np.asarray(reset_options.get("distractor_xy", (-0.55, 0.0)), dtype=np.float32)
        if distractor_xy.shape != (2,):
            raise ValueError("distractor_xy must have shape (2,)")
        self._set_distractor_xy(distractor_xy)
        self._configure_m8_scene()
        self._set_rest_visible(self._is_persistent_task())
        initial_drives = reset_options.get("initial_drives")
        if initial_drives is None:
            drives = Drives(
                satiety=self._active_task.initial_satiety,
                energy=self._active_task.initial_energy,
                boredom=self._active_task.initial_boredom,
            )
        else:
            values = np.asarray(initial_drives, dtype=np.float32)
            if values.shape != (3,) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("initial_drives must contain three values in [0, 1]")
            drives = Drives(satiety=float(values[0]), energy=float(values[1]), boredom=float(values[2]))
        self._state = WorldState(
            holding_food=False,
            food_consumed=False,
            drives=drives,
            step_count=0,
            task_id=self._active_task.name,
        )
        self._prior_outcome = ActionOutcome.SUCCESS
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {
            "environment_version": self._environment_version(),
            "seed": seed,
            "layout_id": self._active_layout.name,
            "camera_variant": self._camera_variant,
            "task_id": self._active_task.name,
            "food_variant": self._food_variant,
            "toy_variant": self._toy_variant,
            "dynamics_variant": self._dynamics_variant,
            "lighting_variant": self._lighting_variant,
            "camera_control": self._camera_control,
            "camera_sector": self._scan_sector if self._camera_control in {"scan", "scan_v2"} else None,
            "blocked_distractor": self._blocked_distractor,
            "event_relocation_on_first_pickup": self._event_relocation_on_first_pickup,
            "geometry_variant": self._geometry_variant,
        }

    def step(
        self, action: dict[str, np.ndarray | int]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        skill = SkillAction.from_gym(action)
        state.drives = state.drives.evolve(self.config, skill.duration_seconds)
        resource_event = self._advance_persistent_resources(skill.duration_seconds)
        event_disturbance = self._apply_event_disturbance_if_due(state, skill)
        cycle_counts_before = (state.feed_cycles, state.play_cycles, state.rest_cycles)
        outcome, task_success = self._execute(state, skill)
        state.step_count += 1
        recovery_action = self._recovery_pending and skill.kind in {ActionKind.WALK_TO, ActionKind.WALK_RELATIVE} and outcome is ActionOutcome.SUCCESS
        if recovery_action:
            self._recovery_pending = False
        post_disturbance_completion = self._post_disturbance_pending and (
            task_success
            or (
                self._is_persistent_task()
                and skill.kind is ActionKind.CONSUME
                and outcome is ActionOutcome.SUCCESS
            )
        )
        if post_disturbance_completion:
            self._post_disturbance_pending = False
        disturbance = event_disturbance or self._apply_disturbance_if_due(state)
        self._prior_outcome = outcome
        survived = state.drives.satiety > 0.0 and state.drives.energy > 0.0
        terminated = (task_success and not self._is_persistent_task()) or not survived
        truncated = state.step_count >= self.config.max_episode_steps and not terminated
        if self._is_persistent_task():
            task_success = truncated and survived and self._persistent_requirements_met(state)
        reward = -self.config.step_penalty_per_second * skill.duration_seconds
        if outcome not in {ActionOutcome.SUCCESS, ActionOutcome.BLOCKED}:
            reward -= self.config.invalid_action_penalty
        if task_success:
            reward += self.config.play_success_reward if self._active_task.success_condition == "relieve_boredom" else self.config.task_success_reward
        if recovery_action or post_disturbance_completion:
            reward += self.config.disturbance_recovery_reward
        if self._is_persistent_task():
            feed_before, play_before, rest_before = cycle_counts_before
            if state.feed_cycles > feed_before:
                reward += self.config.persistent_feed_cycle_reward
            if state.play_cycles > play_before:
                reward += self.config.persistent_play_cycle_reward
            if state.rest_cycles > rest_before:
                reward += self.config.persistent_rest_cycle_reward
        if state.drives.satiety <= 0.0 or state.drives.energy <= 0.0:
            reward -= self.config.task_success_reward
        return self._observation(), float(reward), terminated, truncated, self._info(
            outcome,
            task_success,
            disturbance=disturbance,
            recovery_action=recovery_action,
            post_disturbance_completion=post_disturbance_completion,
            resource_event=resource_event,
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._observation_renderer is not None:
            self._observation_renderer.close()
            self._observation_renderer = None

    def _execute(self, state: WorldState, action: SkillAction) -> tuple[ActionOutcome, bool]:
        if action.kind is ActionKind.WALK_TO:
            self._walk_toward(action.target_xy, action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.WALK_RELATIVE:
            self._walk_toward(self._agent_xy() + action.target_xy, action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.SCAN:
            if self._camera_control in {"scan", "scan_v2"}:
                current = self._SCAN_SECTORS.index(self._scan_sector)
                self._scan_sector = self._SCAN_SECTORS[(current + 1) % len(self._SCAN_SECTORS)]
            self._advance_physics(action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.PICK_UP:
            if not state.food_available or state.food_consumed or self._distance_to_food() > self.config.pickup_radius:
                self._advance_physics(action.duration_seconds)
                return ActionOutcome.BLOCKED, False
            state.holding_food = True
            self._set_food_xy(self._agent_xy())
            self._advance_physics(action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.PICK_UP_RELATIVE:
            candidate = self._nearest_pickup_candidate(self._agent_xy() + action.target_xy)
            if candidate != "food" or self._distance_to_food() > self.config.pickup_radius:
                self._advance_physics(action.duration_seconds)
                return ActionOutcome.BLOCKED, False
            state.holding_food = True
            self._set_food_xy(self._agent_xy())
            self._advance_physics(action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.CONSUME:
            self._advance_physics(action.duration_seconds)
            if not state.holding_food:
                return ActionOutcome.NOT_HOLDING_OBJECT, False
            state.holding_food = False
            state.food_consumed = True
            state.drives = Drives(
                satiety=min(1.0, state.drives.satiety + self.config.eat_satiety_gain),
                energy=state.drives.energy,
                boredom=state.drives.boredom,
            )
            if self._is_persistent_task():
                state.feed_cycles += 1
                state.food_available = False
                state.food_respawn_remaining = self.config.food_respawn_seconds
                self._set_food_visible(False)
                return ActionOutcome.SUCCESS, False
            task_success = self._active_task.success_condition == "consume_food"
            if self._active_task.name == "competing_drives" and state.toy_played:
                task_success = False
            if self._active_task.success_condition == "maintain_needs":
                task_success = state.toy_played
            return ActionOutcome.SUCCESS, task_success
        if action.kind is ActionKind.PLACE:
            self._advance_physics(action.duration_seconds)
            if not state.holding_food:
                return ActionOutcome.NOT_HOLDING_OBJECT, False
            self._set_food_xy(action.target_xy)
            state.holding_food = False
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.RUN_AROUND:
            if self._supports_play() and self._distance_to_toy() > self.config.toy_interaction_radius:
                self._advance_physics(action.duration_seconds)
                return ActionOutcome.BLOCKED, False
            play_cycle_ready = state.drives.boredom >= self.config.play_success_boredom_threshold
            self._run_around(action.duration_seconds)
            if self._supports_play():
                state.toy_played = True
                state.drives = state.drives.relieve_boredom(self.config.play_boredom_reduction)
                if self._is_persistent_task() and play_cycle_ready:
                    state.play_cycles += 1
                    return ActionOutcome.SUCCESS, False
                return (
                    ActionOutcome.SUCCESS,
                    (
                        self._active_task.success_condition == "relieve_boredom"
                        and state.drives.boredom <= self.config.play_success_boredom_threshold
                    )
                    or (
                        self._active_task.success_condition == "maintain_needs" and state.food_consumed
                    ),
                )
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.REST:
            if not self._is_persistent_task() or self._distance_to_rest() > self.config.rest_interaction_radius:
                self._advance_physics(action.duration_seconds)
                return ActionOutcome.BLOCKED, False
            self._advance_physics(action.duration_seconds)
            if state.drives.energy > self.config.rest_cycle_energy_threshold:
                return ActionOutcome.BLOCKED, False
            state.drives = state.drives.restore_energy(self.config.rest_energy_gain)
            state.rest_cycles += 1
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.IDLE:
            self._advance_physics(action.duration_seconds)
            return ActionOutcome.SUCCESS, False
        raise AssertionError(f"unhandled action kind: {action.kind}")

    def _walk_toward(self, target_xy: np.ndarray, duration_seconds: float) -> None:
        requested_target = np.asarray(target_xy, dtype=np.float32)
        target = self._bounded_xy(requested_target)
        if not np.array_equal(requested_target, target):
            self._require_state().boundary_contacts += 1
        start = self._agent_xy()
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance == 0.0:
            self._advance_physics(duration_seconds)
            return
        travel = min(distance, self.config.walk_speed_per_second * self._movement_speed_scale * duration_seconds)
        destination = start + delta / distance * travel
        steps = max(1, ceil(duration_seconds / self.model.opt.timestep))
        for fraction in np.linspace(1 / steps, 1.0, steps, dtype=np.float32):
            self._set_agent_xy(start + (destination - start) * fraction)
            self._sync_held_food()
            self._step_physics_once()

    def _run_around(self, duration_seconds: float) -> None:
        start = self._agent_xy()
        radius = min(0.2, self.config.world_radius - float(np.max(np.abs(start))))
        steps = max(1, ceil(duration_seconds / self.model.opt.timestep))
        for phase in np.linspace(0.0, 2 * np.pi, steps, endpoint=False, dtype=np.float32):
            offset = radius * np.asarray([np.cos(phase), np.sin(phase)], dtype=np.float32)
            self._set_agent_xy(start + offset)
            self._sync_held_food()
            self._step_physics_once()

    def _advance_physics(self, duration_seconds: float) -> None:
        for _ in range(max(1, ceil(duration_seconds / self.model.opt.timestep))):
            self._sync_held_food()
            self._step_physics_once()

    def _step_physics_once(self) -> None:
        self.data.qvel[:] = 0.0
        mujoco.mj_step(self.model, self.data)

    def _sync_held_food(self) -> None:
        if self._require_state().holding_food:
            self._set_food_xy(self._agent_xy())

    def _observation(self) -> dict[str, Any]:
        state = self._require_state()
        common = {
            "drives": state.drives.as_array(),
            "holding_food": int(state.holding_food),
            "prior_outcome": list(ActionOutcome).index(self._prior_outcome),
        }
        if self.config.observation_mode == "state_oracle":
            return {
                "agent_xy": self._agent_xy().copy(),
                "food_xy": self._food_xy().copy(),
                "toy_xy": self._toy_xy().copy(),
                "rest_xy": self._rest_xy().copy(),
                **common,
            }
        rgb = self._rgb_observation()
        if self.config.observation_mode == "hybrid":
            return {"rgb": rgb, "food_detection": food_detection_from_rgb(rgb), **common}
        return {"rgb": rgb, **common}

    def _info(
        self,
        outcome: ActionOutcome,
        task_success: bool,
        *,
        disturbance: str | None,
        recovery_action: bool,
        post_disturbance_completion: bool,
        resource_event: str | None,
    ) -> dict[str, Any]:
        state = self._require_state()
        return {
            "outcome": outcome.value,
            "task_success": task_success,
            "step_count": state.step_count,
            "environment_version": self._environment_version(),
            "sim_time_seconds": float(self.data.time),
            "layout_id": self._active_layout.name,
            "camera_variant": self._camera_variant,
            "disturbance": disturbance,
            "recovery_action": recovery_action,
            "post_disturbance_completion": post_disturbance_completion,
            "resource_event": resource_event,
            "failure_reason": self._failure_reason(outcome),
            "task_id": self._active_task.name,
            "food_variant": self._food_variant,
            "toy_variant": self._toy_variant,
            "dynamics_variant": self._dynamics_variant,
            "toy_played": state.toy_played,
            "feed_cycles": state.feed_cycles,
            "play_cycles": state.play_cycles,
            "rest_cycles": state.rest_cycles,
            "food_available": state.food_available,
            "maintenance_complete": self._persistent_requirements_met(state) if self._is_persistent_task() else False,
            "boundary_contacts": state.boundary_contacts,
            "survived": state.drives.satiety > 0.0 and state.drives.energy > 0.0,
            "camera_control": self._camera_control,
            "camera_sector": self._scan_sector if self._camera_control in {"scan", "scan_v2"} else None,
            "blocked_distractor": self._blocked_distractor,
            "geometry_variant": self._geometry_variant,
        }

    def _observation_spaces(self, radius: float) -> dict[str, spaces.Space[Any]]:
        common: dict[str, spaces.Space[Any]] = {
            "drives": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "holding_food": spaces.Discrete(2),
            "prior_outcome": spaces.Discrete(len(ActionOutcome)),
        }
        if self.config.observation_mode == "state_oracle":
            return {
                "agent_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "food_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "toy_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "rest_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                **common,
            }
        image = spaces.Box(
            low=0, high=255, shape=(self.config.rgb_height, self.config.rgb_width, 3), dtype=np.uint8
        )
        if self.config.observation_mode == "hybrid":
            return {"rgb": image, "food_detection": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32), **common}
        return {"rgb": image, **common}

    def _rgb_observation(self) -> np.ndarray:
        if self._observation_renderer is None:
            self._observation_renderer = mujoco.Renderer(
                self.model, height=self.config.rgb_height, width=self.config.rgb_width
            )
        camera = (
            f"agent_cam_scan_v2_{self._scan_sector}"
            if self._camera_control == "scan_v2"
            else f"agent_cam_scan_{self._scan_sector}"
            if self._camera_control == "scan"
            else f"agent_cam_{self._camera_variant}"
        )
        self._observation_renderer.update_scene(self.data, camera=camera)
        return self._observation_renderer.render().copy()

    def _apply_disturbance_if_due(self, state: WorldState) -> str | None:
        if (
            self._disturbance_step is None
            or self._disturbance_occurred
            or state.holding_food
            or state.food_consumed
            or state.step_count != self._disturbance_step
        ):
            return None
        previous = self._food_xy()
        candidate = self._active_layout.sample_food_xy(self.np_random)
        for _ in range(8):
            if float(np.linalg.norm(candidate - previous)) > 0.25:
                break
            candidate = self._active_layout.sample_food_xy(self.np_random)
        self._set_food_xy(candidate)
        self._disturbance_occurred = True
        self._recovery_pending = True
        self._post_disturbance_pending = True
        return "food_relocated"

    @staticmethod
    def _failure_reason(outcome: ActionOutcome) -> str | None:
        if outcome is ActionOutcome.BLOCKED:
            return "food_not_within_pickup_radius"
        if outcome is ActionOutcome.NOT_HOLDING_OBJECT:
            return "food_not_held"
        if outcome is ActionOutcome.TARGET_NOT_VISIBLE:
            return "target_not_visible"
        if outcome is ActionOutcome.NOT_EDIBLE:
            return "held_object_not_edible"
        return None

    def _agent_xy(self) -> np.ndarray:
        return self.data.qpos[self._agent_qpos].astype(np.float32)

    def _food_xy(self) -> np.ndarray:
        return self.data.qpos[self._food_qpos].astype(np.float32)

    def _set_agent_xy(self, xy: np.ndarray) -> None:
        requested = np.asarray(xy, dtype=np.float32)
        bounded = self._bounded_xy(requested)
        if self._state is not None and not np.array_equal(requested, bounded):
            self._state.boundary_contacts += 1
        self.data.qpos[self._agent_qpos] = bounded
        mujoco.mj_forward(self.model, self.data)

    def _set_food_xy(self, xy: np.ndarray) -> None:
        self.data.qpos[self._food_qpos] = self._bounded_xy(xy)
        mujoco.mj_forward(self.model, self.data)

    def _toy_xy(self) -> np.ndarray:
        return self.data.qpos[self._toy_qpos].astype(np.float32)

    def _set_toy_xy(self, xy: np.ndarray) -> None:
        self.data.qpos[self._toy_qpos] = self._bounded_xy(xy)
        mujoco.mj_forward(self.model, self.data)

    def _rest_xy(self) -> np.ndarray:
        return self.data.qpos[self._rest_qpos].astype(np.float32)

    def _set_rest_xy(self, xy: np.ndarray) -> None:
        self.data.qpos[self._rest_qpos] = self._bounded_xy(xy)
        mujoco.mj_forward(self.model, self.data)

    def _set_rest_visible(self, visible: bool) -> None:
        self.model.geom_rgba[self.model.geom("rest_geom").id, 3] = 1.0 if visible else 0.0

    def _distractor_xy(self) -> np.ndarray:
        return self.data.qpos[self._distractor_qpos].astype(np.float32)

    def _set_distractor_xy(self, xy: np.ndarray) -> None:
        self.data.qpos[self._distractor_qpos] = self._bounded_xy(xy)
        mujoco.mj_forward(self.model, self.data)

    def _distance_to_toy(self) -> float:
        return float(np.linalg.norm(self._agent_xy() - self._toy_xy()))

    def _distance_to_rest(self) -> float:
        return float(np.linalg.norm(self._agent_xy() - self._rest_xy()))

    def _is_m4_task(self) -> bool:
        return self._active_task.name in {"play_when_bored", "competing_drives"}

    def _supports_play(self) -> bool:
        return self._is_m4_task() or self._active_task.success_condition in {"maintain_needs", "persistent_maintenance"}

    def _is_persistent_task(self) -> bool:
        return self._active_task.success_condition == "persistent_maintenance"

    def _persistent_requirements_met(self, state: WorldState) -> bool:
        return (
            state.feed_cycles >= self.config.persistent_min_feed_cycles
            and state.play_cycles >= self.config.persistent_min_play_cycles
            and state.rest_cycles >= self.config.persistent_min_rest_cycles
        )

    def _environment_version(self) -> str:
        if self._is_persistent_task():
            return "0.8.0"
        if self._active_task.success_condition == "maintain_needs":
            return "0.7.0"
        if self._camera_control == "scan_v2":
            return "0.6.0"
        return "0.5.0" if self._camera_control == "scan" else ("0.4.0" if self._is_m4_task() else "0.3.0")

    def _configure_variants(
        self,
        *,
        food_variant: str,
        toy_variant: str,
        dynamics_variant: str,
        lighting_variant: str,
        agent_shape_variant: str,
        food_shape_variant: str,
    ) -> None:
        try:
            food_rgba = FOOD_VARIANTS[food_variant]
        except KeyError as exc:
            raise ValueError(f"unknown food_variant '{food_variant}'; expected one of {sorted(FOOD_VARIANTS)}") from exc
        try:
            active_toy_geom = _TOY_VARIANTS[toy_variant]
        except KeyError as exc:
            raise ValueError(f"unknown toy_variant '{toy_variant}'; expected one of {sorted(_TOY_VARIANTS)}") from exc
        try:
            movement_scale, floor_friction = _DYNAMICS_VARIANTS[dynamics_variant]
        except KeyError as exc:
            raise ValueError(f"unknown dynamics_variant '{dynamics_variant}'; expected one of {sorted(_DYNAMICS_VARIANTS)}") from exc
        try:
            ambient, diffuse = _LIGHTING_VARIANTS[lighting_variant]
        except KeyError as exc:
            raise ValueError(f"unknown lighting_variant '{lighting_variant}'; expected one of {sorted(_LIGHTING_VARIANTS)}") from exc
        try:
            active_agent_geom = _AGENT_SHAPE_VARIANTS[agent_shape_variant]
        except KeyError as exc:
            raise ValueError(f"unknown agent_shape_variant '{agent_shape_variant}'; expected one of {sorted(_AGENT_SHAPE_VARIANTS)}") from exc
        try:
            active_food_geom = _FOOD_SHAPE_VARIANTS[food_shape_variant]
        except KeyError as exc:
            raise ValueError(f"unknown food_shape_variant '{food_shape_variant}'; expected one of {sorted(_FOOD_SHAPE_VARIANTS)}") from exc
        for geom_name in _AGENT_SHAPE_VARIANTS.values():
            geom = self.model.geom(geom_name).id
            self.model.geom_rgba[geom, 3] = 1.0 if geom_name == active_agent_geom else 0.0
        for geom_name in _FOOD_SHAPE_VARIANTS.values():
            geom = self.model.geom(geom_name).id
            self.model.geom_rgba[geom] = food_rgba
            self.model.geom_rgba[geom, 3] = 1.0 if geom_name == active_food_geom else 0.0
        for variant, geom_name in _TOY_VARIANTS.items():
            self.model.geom_rgba[self.model.geom(geom_name).id, 3] = 1.0 if geom_name == active_toy_geom else 0.0
        self.model.geom_friction[self.model.geom("floor").id, 0] = floor_friction
        self.model.vis.headlight.ambient[:] = ambient
        self.model.vis.headlight.diffuse[:] = diffuse
        self._food_variant = food_variant
        self._toy_variant = toy_variant
        self._dynamics_variant = dynamics_variant
        self._lighting_variant = lighting_variant
        self._agent_shape_variant = agent_shape_variant
        self._food_shape_variant = food_shape_variant
        self._movement_speed_scale = movement_scale

    def _active_agent_geom_names(self) -> tuple[str, ...]:
        return (_AGENT_SHAPE_VARIANTS[self._agent_shape_variant],)

    def _active_food_geom_names(self) -> tuple[str, ...]:
        return (_FOOD_SHAPE_VARIANTS[self._food_shape_variant],)

    def _set_food_visible(self, visible: bool) -> None:
        active = _FOOD_SHAPE_VARIANTS[self._food_shape_variant]
        for geom_name in _FOOD_SHAPE_VARIANTS.values():
            self.model.geom_rgba[self.model.geom(geom_name).id, 3] = 1.0 if visible and geom_name == active else 0.0

    def _configure_m8_scene(self) -> None:
        distractor_geom = self.model.geom("distractor_geom").id
        geometry_geom = self.model.geom("m8_geometry_geom").id
        landmark_geom = self.model.geom("m81_landmark_geom").id
        m82_landmark_geom = self.model.geom("m82_landmark_geom").id
        self.model.geom_rgba[distractor_geom, :3] = self.model.geom_rgba[self.model.geom("food_geom").id, :3]
        self.model.geom_rgba[distractor_geom, 3] = 1.0 if self._blocked_distractor else 0.0
        self.model.geom_rgba[geometry_geom, 3] = 1.0 if self._geometry_variant == "unseen_block" else 0.0
        self.model.geom_rgba[landmark_geom, 3] = 1.0 if self._geometry_variant == "m81_landmark" else 0.0
        self.model.geom_rgba[m82_landmark_geom, 3] = 1.0 if self._geometry_variant == "m82_landmark" else 0.0

    def _nearest_pickup_candidate(self, target_xy: np.ndarray) -> str | None:
        candidates: list[tuple[str, np.ndarray]] = []
        if self._require_state().food_available:
            candidates.append(("food", self._food_xy()))
        if self._blocked_distractor:
            candidates.append(("distractor", self._distractor_xy()))
        if not candidates:
            return None
        name, candidate_xy = min(candidates, key=lambda item: float(np.linalg.norm(item[1] - target_xy)))
        if float(np.linalg.norm(candidate_xy - target_xy)) > self.config.pickup_radius:
            return None
        return name

    def _bounded_xy(self, xy: np.ndarray) -> np.ndarray:
        limit = self.config.world_radius - self.config.body_clearance
        return np.clip(np.asarray(xy, dtype=np.float32), -limit, limit)

    def _distance_to_food(self) -> float:
        return float(np.linalg.norm(self._agent_xy() - self._food_xy()))

    def _advance_persistent_resources(self, elapsed_seconds: float) -> str | None:
        if not self._is_persistent_task():
            return None
        state = self._require_state()
        if state.food_available:
            return None
        state.food_respawn_remaining = max(0.0, state.food_respawn_remaining - elapsed_seconds)
        if state.food_respawn_remaining > 0.0:
            return None
        previous = self._food_xy()
        candidate = self._active_layout.sample_food_xy(self.np_random)
        for _ in range(8):
            if float(np.linalg.norm(candidate - previous)) > 0.25 and float(np.linalg.norm(candidate - self._agent_xy())) > 0.25:
                break
            candidate = self._active_layout.sample_food_xy(self.np_random)
        self._set_food_xy(candidate)
        self._set_food_visible(True)
        state.food_available = True
        state.food_consumed = False
        return "food_replenished"

    def _apply_event_disturbance_if_due(self, state: WorldState, action: SkillAction) -> str | None:
        if (
            not self._event_relocation_on_first_pickup
            or self._disturbance_occurred
            or not state.food_available
            or state.holding_food
            or action.kind not in {ActionKind.PICK_UP, ActionKind.PICK_UP_RELATIVE}
        ):
            return None
        previous = self._food_xy()
        candidate = self._active_layout.sample_food_xy(self.np_random)
        for _ in range(8):
            if float(np.linalg.norm(candidate - previous)) > 0.25 and float(np.linalg.norm(candidate - self._agent_xy())) > self.config.pickup_radius:
                break
            candidate = self._active_layout.sample_food_xy(self.np_random)
        if float(np.linalg.norm(candidate - previous)) <= 0.25 or float(np.linalg.norm(candidate - self._agent_xy())) <= self.config.pickup_radius:
            low = np.asarray(self._active_layout.food_low, dtype=np.float32)
            high = np.asarray(self._active_layout.food_high, dtype=np.float32)
            corners = (
                low,
                high,
                np.asarray((low[0], high[1]), dtype=np.float32),
                np.asarray((high[0], low[1]), dtype=np.float32),
            )
            candidate = max(corners, key=lambda item: float(np.linalg.norm(item - previous)))
        self._set_food_xy(candidate)
        self._disturbance_occurred = True
        self._recovery_pending = True
        self._post_disturbance_pending = True
        return "food_relocated"

    def _qpos_indices(self, *joint_names: str) -> np.ndarray:
        return np.asarray([self.model.jnt_qposadr[self.model.joint(name).id] for name in joint_names])

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise gym.error.ResetNeeded("Call reset() before requesting observations or stepping the environment")
        return self._state
