"""Registered task definitions and deterministic spawn-layout splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """A named creature/food spawn distribution within the MuJoCo room."""

    name: str
    agent_xy: tuple[float, float]
    food_low: tuple[float, float]
    food_high: tuple[float, float]
    toy_xy: tuple[float, float] = (0.45, 0.0)

    def sample_food_xy(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.food_low, self.food_high, size=2).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    description: str
    train_layout_ids: tuple[str, ...]
    heldout_layout_ids: tuple[str, ...]
    initial_satiety: float = 0.45
    initial_energy: float = 1.0
    initial_boredom: float = 0.0
    success_condition: str = "consume_food"


LAYOUTS: dict[str, LayoutSpec] = {
    # Keep this exact distribution for M0/M1 callers that omit reset options.
    "default": LayoutSpec("default", (0.0, 0.0), (-0.8, -0.8), (0.8, 0.8)),
    "train_center": LayoutSpec("train_center", (0.0, 0.0), (-0.55, -0.55), (0.55, 0.55)),
    "train_east": LayoutSpec("train_east", (-0.45, 0.0), (0.10, -0.65), (0.75, 0.65)),
    "heldout_northwest": LayoutSpec(
        "heldout_northwest", (0.45, -0.45), (-0.82, 0.25), (-0.25, 0.82)
    ),
    "heldout_southeast": LayoutSpec(
        "heldout_southeast", (-0.45, 0.45), (0.25, -0.82), (0.82, -0.25)
    ),
    "m3_train_center": LayoutSpec("m3_train_center", (0.0, 0.0), (-0.55, -0.55), (0.55, 0.55)),
    "m3_train_east": LayoutSpec("m3_train_east", (0.0, 0.0), (0.10, -0.65), (0.75, 0.65)),
    "m3_heldout_northwest": LayoutSpec("m3_heldout_northwest", (0.0, 0.0), (-0.82, 0.25), (-0.25, 0.82)),
    "m3_heldout_southeast": LayoutSpec("m3_heldout_southeast", (0.0, 0.0), (0.25, -0.82), (0.82, -0.25)),
    "m4_train_play": LayoutSpec("m4_train_play", (0.0, 0.0), (-0.55, -0.55), (0.55, 0.55), (0.48, 0.0)),
    "m4_train_competing": LayoutSpec("m4_train_competing", (0.0, 0.0), (0.35, -0.30), (0.65, 0.30), (-0.46, 0.0)),
    "m4_heldout_play": LayoutSpec("m4_heldout_play", (0.38, -0.42), (-0.82, 0.25), (-0.30, 0.80), (-0.42, 0.42)),
    "m4_heldout_competing": LayoutSpec("m4_heldout_competing", (-0.38, 0.42), (0.30, -0.80), (0.82, -0.25), (0.42, -0.42)),
    # M8 uses this fixed-distribution protocol layout so the relocation event
    # always happens before a food pickup can succeed.
    "m8_protocol": LayoutSpec("m8_protocol", (0.0, 0.0), (0.25, 0.50), (0.50, 0.70)),
    # M8.2's post-result suite uses separate agent/food quadrants.  These are
    # deliberately not part of M8.1 training or its original held-out matrix.
    "m82_northwest": LayoutSpec("m82_northwest", (0.38, -0.42), (-0.82, 0.25), (-0.40, 0.72)),
    "m82_southeast": LayoutSpec("m82_southeast", (-0.38, 0.42), (0.30, -0.82), (0.82, -0.25)),
    # M8.3 separates agent pose from food position so transfer failures can be
    # attributed to one spatial change at a time rather than a compound row.
    "m83_agent_northwest": LayoutSpec("m83_agent_northwest", (0.38, -0.42), (0.25, 0.50), (0.50, 0.70)),
    "m83_food_northwest": LayoutSpec("m83_food_northwest", (0.0, 0.0), (-0.82, 0.25), (-0.40, 0.72)),
    # M8.4 uses its own spatial development and validation distributions; it
    # never reuses M8.2 audit layouts or M8.3 diagnosis rows for selection.
    "m84_train_west": LayoutSpec("m84_train_west", (0.18, -0.20), (-0.72, -0.08), (-0.32, 0.36)),
    "m84_train_southeast": LayoutSpec("m84_train_southeast", (-0.18, 0.16), (0.18, -0.72), (0.64, -0.30)),
    "m84_train_southwest": LayoutSpec("m84_train_southwest", (0.24, 0.26), (-0.74, -0.76), (-0.34, -0.36)),
    "m84_validation_northwest": LayoutSpec("m84_validation_northwest", (0.30, -0.26), (-0.78, 0.18), (-0.48, 0.58)),
    "m84_validation_southeast": LayoutSpec("m84_validation_southeast", (-0.28, 0.30), (0.38, -0.78), (0.76, -0.42)),
    # M8.5 is a post-result audit of the frozen M8.4 grounder.  These layouts
    # are disjoint from M8.4 development/validation and every legacy M8 suite.
    "m85_northeast": LayoutSpec("m85_northeast", (-0.56, -0.46), (0.30, 0.28), (0.70, 0.68)),
    "m85_southwest": LayoutSpec("m85_southwest", (0.58, 0.46), (-0.72, -0.72), (-0.32, -0.32)),
    "m85_northwest": LayoutSpec("m85_northwest", (0.56, -0.48), (-0.76, 0.26), (-0.36, 0.66)),
    "m85_southeast": LayoutSpec("m85_southeast", (-0.58, 0.48), (0.32, -0.72), (0.72, -0.32)),
    # M8.6 is a development-only factorial diagnosis. It deliberately avoids
    # M8.5's held-out coordinates while testing a new northeast relation.
    "m86_northeast": LayoutSpec("m86_northeast", (-0.46, -0.34), (0.18, 0.14), (0.54, 0.50)),
    # M8.7 has separate development and validation layouts after M8.6 showed
    # that blue-food grounding, not grippy dynamics, is the weak component.
    "m87_train_northeast": LayoutSpec("m87_train_northeast", (-0.42, -0.30), (0.12, 0.10), (0.50, 0.48)),
    "m87_train_northwest": LayoutSpec("m87_train_northwest", (0.42, -0.30), (-0.50, 0.10), (-0.12, 0.48)),
    "m87_validation_northwest": LayoutSpec("m87_validation_northwest", (0.50, -0.40), (-0.70, 0.18), (-0.32, 0.56)),
    "m87_validation_southeast": LayoutSpec("m87_validation_southeast", (-0.50, 0.40), (0.32, -0.70), (0.70, -0.32)),
    # M8.8 is the fresh external audit for M8.7 and shares no M8.7 or M8.5 layouts.
    "m88_northeast": LayoutSpec("m88_northeast", (-0.62, -0.50), (0.24, 0.30), (0.66, 0.70)),
    "m88_southwest": LayoutSpec("m88_southwest", (0.62, 0.50), (-0.70, -0.70), (-0.28, -0.28)),
    "m88_northwest": LayoutSpec("m88_northwest", (0.56, -0.44), (-0.66, 0.24), (-0.30, 0.62)),
    "m88_southeast": LayoutSpec("m88_southeast", (-0.62, 0.50), (0.30, -0.74), (0.72, -0.32)),
    # M8.9 is a fresh visual-morphology audit for the frozen M8.7 grounder.
    # It does not reuse M8.5, M8.7, or M8.8 layouts.
    "m89_northeast": LayoutSpec("m89_northeast", (-0.68, -0.56), (0.18, 0.34), (0.56, 0.72)),
    "m89_southwest": LayoutSpec("m89_southwest", (0.68, 0.56), (-0.76, -0.68), (-0.38, -0.30)),
    "m89_northwest": LayoutSpec("m89_northwest", (0.62, -0.54), (-0.74, 0.20), (-0.38, 0.58)),
    "m89_southeast": LayoutSpec("m89_southeast", (-0.68, 0.56), (0.30, -0.68), (0.62, -0.34)),
}

TASKS: dict[str, TaskSpec] = {
    "find_and_eat": TaskSpec(
        name="find_and_eat",
        description="Reach the food, pick it up from within the guard radius, and consume it.",
        train_layout_ids=("train_center", "train_east"),
        heldout_layout_ids=("heldout_northwest", "heldout_southeast"),
    ),
    "find_and_eat_perception": TaskSpec(
        name="find_and_eat_perception",
        description="Find and eat with an agent-mounted RGB camera and optional visual food detection.",
        train_layout_ids=("m3_train_center", "m3_train_east"),
        heldout_layout_ids=("m3_heldout_northwest", "m3_heldout_southeast"),
    ),
    "play_when_bored": TaskSpec(
        name="play_when_bored",
        description="Reach the toy and run around it to reduce boredom below the task threshold.",
        train_layout_ids=("m4_train_play",),
        heldout_layout_ids=("m4_heldout_play",),
        initial_satiety=0.75,
        initial_boredom=0.85,
        success_condition="relieve_boredom",
    ),
    "competing_drives": TaskSpec(
        name="competing_drives",
        description="Choose food before play when low satiety and high boredom are both active.",
        train_layout_ids=("m4_train_competing",),
        heldout_layout_ids=("m4_heldout_competing",),
        initial_satiety=0.20,
        initial_boredom=0.85,
        success_condition="consume_food",
    ),
}


def get_layout(name: str) -> LayoutSpec:
    try:
        return LAYOUTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown layout '{name}'; expected one of {sorted(LAYOUTS)}") from exc


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown task '{name}'; expected one of {sorted(TASKS)}") from exc
