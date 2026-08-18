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
    rest_xy: tuple[float, float] = (-0.45, 0.0)

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
    # M9 integrates food, toy, scanning, and recovery in one longer-lived
    # episode.  These layouts are disjoint from all M8 development, validation,
    # and sealed-audit coordinates.
    "m9_train_northeast": LayoutSpec("m9_train_northeast", (-0.34, -0.28), (0.16, 0.22), (0.46, 0.56), (0.60, -0.42)),
    "m9_train_southwest": LayoutSpec("m9_train_southwest", (0.34, 0.28), (-0.58, -0.60), (-0.24, -0.26), (-0.56, 0.42)),
    "m9_train_northwest": LayoutSpec("m9_train_northwest", (0.36, -0.30), (-0.62, 0.20), (-0.28, 0.54), (0.52, 0.46)),
    "m9_train_southeast": LayoutSpec("m9_train_southeast", (-0.36, 0.30), (0.22, -0.62), (0.56, -0.28), (-0.50, -0.46)),
    "m9_validation_northeast": LayoutSpec("m9_validation_northeast", (-0.48, -0.38), (0.20, 0.30), (0.58, 0.66), (0.60, -0.50)),
    "m9_validation_southwest": LayoutSpec("m9_validation_southwest", (0.48, 0.38), (-0.68, -0.66), (-0.30, -0.30), (-0.62, 0.46)),
    "m9_validation_northwest": LayoutSpec("m9_validation_northwest", (0.50, -0.38), (-0.70, 0.22), (-0.32, 0.62), (0.56, 0.50)),
    "m9_validation_southeast": LayoutSpec("m9_validation_southeast", (-0.50, 0.38), (0.24, -0.70), (0.64, -0.32), (-0.54, -0.50)),
    "m9_audit_northeast": LayoutSpec("m9_audit_northeast", (-0.60, -0.50), (0.18, 0.32), (0.54, 0.70), (0.70, -0.46)),
    "m9_audit_southwest": LayoutSpec("m9_audit_southwest", (0.60, 0.50), (-0.74, -0.70), (-0.36, -0.32), (-0.66, 0.52)),
    "m9_audit_northwest": LayoutSpec("m9_audit_northwest", (0.62, -0.50), (-0.76, 0.20), (-0.36, 0.60), (0.62, 0.52)),
    "m9_audit_southeast": LayoutSpec("m9_audit_southeast", (-0.62, 0.50), (0.26, -0.76), (0.66, -0.36), (-0.58, -0.54)),
    # M9.1 is a fresh development-only protocol repair. It does not touch M9's
    # unscored audit layouts and isolates distractor choice from forced recovery.
    "m91_choice_northwest": LayoutSpec("m91_choice_northwest", (0.54, -0.46), (-0.74, 0.22), (-0.38, 0.60), (0.60, 0.46)),
    "m91_recovery_southeast": LayoutSpec("m91_recovery_southeast", (-0.54, 0.46), (0.24, -0.74), (0.62, -0.38), (-0.58, -0.50)),
    # M10 freezes separate development, validation, and audit layouts for
    # persistent food, play, and rest cycles. These do not reuse M9.1.
    "m10_train_northeast": LayoutSpec("m10_train_northeast", (-0.32, -0.26), (0.18, 0.20), (0.48, 0.54), (0.58, -0.42), (-0.54, 0.40)),
    "m10_train_southwest": LayoutSpec("m10_train_southwest", (0.32, 0.26), (-0.56, -0.58), (-0.22, -0.24), (-0.54, 0.40), (0.56, -0.42)),
    "m10_validation_northeast": LayoutSpec("m10_validation_northeast", (-0.46, -0.36), (0.22, 0.28), (0.58, 0.64), (0.62, -0.48), (-0.62, 0.46)),
    "m10_validation_southwest": LayoutSpec("m10_validation_southwest", (0.46, 0.36), (-0.66, -0.64), (-0.30, -0.28), (-0.62, 0.48), (0.62, -0.46)),
    "m10_validation_northwest": LayoutSpec("m10_validation_northwest", (0.48, -0.36), (-0.68, 0.22), (-0.34, 0.60), (0.58, 0.50), (-0.58, -0.48)),
    "m10_validation_southeast": LayoutSpec("m10_validation_southeast", (-0.48, 0.36), (0.24, -0.68), (0.62, -0.32), (-0.56, -0.50), (0.58, 0.48)),
    "m10_audit_northeast": LayoutSpec("m10_audit_northeast", (-0.58, -0.48), (0.20, 0.34), (0.56, 0.70), (0.68, -0.44), (-0.66, 0.50)),
    "m10_audit_southwest": LayoutSpec("m10_audit_southwest", (0.58, 0.48), (-0.72, -0.68), (-0.34, -0.30), (-0.64, 0.52), (0.68, -0.48)),
    "m10_audit_northwest": LayoutSpec("m10_audit_northwest", (0.60, -0.48), (-0.74, 0.20), (-0.38, 0.60), (0.64, 0.50), (-0.64, -0.50)),
    "m10_audit_southeast": LayoutSpec("m10_audit_southeast", (-0.60, 0.48), (0.26, -0.74), (0.66, -0.36), (-0.60, -0.52), (0.64, 0.50)),
    # M13 deliberately retires M10's 1700–1919 splits.  These independent
    # layouts support the state-oracle persistent-maintenance RL protocol.
    "m13_dev_northeast": LayoutSpec("m13_dev_northeast", (-0.26, -0.22), (0.24, 0.26), (0.52, 0.58), (0.62, -0.44), (-0.60, 0.44)),
    "m13_dev_southwest": LayoutSpec("m13_dev_southwest", (0.26, 0.22), (-0.60, -0.62), (-0.26, -0.28), (-0.60, 0.44), (0.60, -0.44)),
    "m13_dev_northwest": LayoutSpec("m13_dev_northwest", (0.30, -0.24), (-0.64, 0.24), (-0.30, 0.58), (0.58, 0.48), (-0.58, -0.46)),
    "m13_dev_southeast": LayoutSpec("m13_dev_southeast", (-0.30, 0.24), (0.26, -0.64), (0.60, -0.30), (-0.58, -0.48), (0.58, 0.46)),
    "m13_validation_northeast": LayoutSpec("m13_validation_northeast", (-0.50, -0.40), (0.24, 0.32), (0.60, 0.68), (0.66, -0.50), (-0.66, 0.50)),
    "m13_validation_southwest": LayoutSpec("m13_validation_southwest", (0.50, 0.40), (-0.70, -0.68), (-0.34, -0.32), (-0.66, 0.50), (0.66, -0.50)),
    "m13_validation_northwest": LayoutSpec("m13_validation_northwest", (0.52, -0.40), (-0.72, 0.24), (-0.36, 0.64), (0.62, 0.52), (-0.62, -0.50)),
    "m13_validation_southeast": LayoutSpec("m13_validation_southeast", (-0.52, 0.40), (0.28, -0.72), (0.64, -0.36), (-0.62, -0.52), (0.62, 0.50)),
    "m13_audit_northeast": LayoutSpec("m13_audit_northeast", (-0.64, -0.52), (0.22, 0.34), (0.58, 0.72), (0.72, -0.48), (-0.70, 0.54)),
    "m13_audit_southwest": LayoutSpec("m13_audit_southwest", (0.64, 0.52), (-0.74, -0.72), (-0.38, -0.34), (-0.70, 0.54), (0.70, -0.54)),
    "m13_audit_northwest": LayoutSpec("m13_audit_northwest", (0.66, -0.52), (-0.76, 0.22), (-0.40, 0.62), (0.68, 0.54), (-0.68, -0.54)),
    "m13_audit_southeast": LayoutSpec("m13_audit_southeast", (-0.66, 0.52), (0.30, -0.76), (0.66, -0.40), (-0.68, -0.54), (0.68, 0.54)),
    # M13.1 changes only the transition reward and therefore uses an entirely
    # new split family; its coordinates never overlap the original M13 suite.
    "m131_dev_northeast": LayoutSpec("m131_dev_northeast", (-0.24, -0.20), (0.22, 0.30), (0.50, 0.62), (0.60, -0.40), (-0.58, 0.42)),
    "m131_dev_southwest": LayoutSpec("m131_dev_southwest", (0.24, 0.20), (-0.62, -0.64), (-0.28, -0.30), (-0.58, 0.42), (0.58, -0.40)),
    "m131_dev_northwest": LayoutSpec("m131_dev_northwest", (0.28, -0.22), (-0.66, 0.20), (-0.32, 0.60), (0.56, 0.46), (-0.56, -0.44)),
    "m131_dev_southeast": LayoutSpec("m131_dev_southeast", (-0.28, 0.22), (0.24, -0.66), (0.62, -0.32), (-0.56, -0.46), (0.56, 0.44)),
    "m131_validation_northeast": LayoutSpec("m131_validation_northeast", (-0.48, -0.38), (0.26, 0.34), (0.62, 0.70), (0.68, -0.48), (-0.68, 0.52)),
    "m131_validation_southwest": LayoutSpec("m131_validation_southwest", (0.48, 0.38), (-0.72, -0.70), (-0.36, -0.34), (-0.68, 0.52), (0.68, -0.48)),
    "m131_validation_northwest": LayoutSpec("m131_validation_northwest", (0.54, -0.42), (-0.74, 0.22), (-0.38, 0.66), (0.64, 0.54), (-0.64, -0.52)),
    "m131_validation_southeast": LayoutSpec("m131_validation_southeast", (-0.54, 0.42), (0.30, -0.74), (0.66, -0.38), (-0.64, -0.54), (0.64, 0.52)),
    "m131_audit_northeast": LayoutSpec("m131_audit_northeast", (-0.66, -0.54), (0.24, 0.36), (0.60, 0.74), (0.74, -0.50), (-0.72, 0.56)),
    "m131_audit_southwest": LayoutSpec("m131_audit_southwest", (0.66, 0.54), (-0.76, -0.74), (-0.40, -0.36), (-0.72, 0.56), (0.72, -0.50)),
    "m131_audit_northwest": LayoutSpec("m131_audit_northwest", (0.68, -0.54), (-0.78, 0.20), (-0.42, 0.64), (0.70, 0.56), (-0.70, -0.54)),
    "m131_audit_southeast": LayoutSpec("m131_audit_southeast", (-0.68, 0.54), (0.32, -0.78), (0.68, -0.42), (-0.70, -0.56), (0.70, 0.54)),
    # M13.2 keeps the M13.1 task/reward but has a new function-approximation
    # split family, disjoint from every M13/M13.1 layout.
    "m132_dev_northeast": LayoutSpec("m132_dev_northeast", (-0.22, -0.18), (0.20, 0.32), (0.48, 0.64), (0.58, -0.38), (-0.56, 0.40)),
    "m132_dev_southwest": LayoutSpec("m132_dev_southwest", (0.22, 0.18), (-0.64, -0.66), (-0.30, -0.32), (-0.56, 0.40), (0.56, -0.38)),
    "m132_dev_northwest": LayoutSpec("m132_dev_northwest", (0.26, -0.20), (-0.68, 0.18), (-0.34, 0.62), (0.54, 0.44), (-0.54, -0.42)),
    "m132_dev_southeast": LayoutSpec("m132_dev_southeast", (-0.26, 0.20), (0.22, -0.68), (0.64, -0.34), (-0.54, -0.44), (0.54, 0.42)),
    "m132_validation_northeast": LayoutSpec("m132_validation_northeast", (-0.46, -0.36), (0.28, 0.36), (0.64, 0.72), (0.70, -0.46), (-0.70, 0.54)),
    "m132_validation_southwest": LayoutSpec("m132_validation_southwest", (0.46, 0.36), (-0.74, -0.72), (-0.38, -0.36), (-0.70, 0.54), (0.70, -0.46)),
    "m132_validation_northwest": LayoutSpec("m132_validation_northwest", (0.56, -0.44), (-0.76, 0.20), (-0.40, 0.68), (0.66, 0.56), (-0.66, -0.54)),
    "m132_validation_southeast": LayoutSpec("m132_validation_southeast", (-0.56, 0.44), (0.32, -0.76), (0.68, -0.40), (-0.66, -0.56), (0.66, 0.54)),
    "m132_audit_northeast": LayoutSpec("m132_audit_northeast", (-0.68, -0.56), (0.26, 0.38), (0.62, 0.76), (0.76, -0.52), (-0.74, 0.58)),
    "m132_audit_southwest": LayoutSpec("m132_audit_southwest", (0.68, 0.56), (-0.78, -0.76), (-0.42, -0.38), (-0.74, 0.58), (0.74, -0.52)),
    "m132_audit_northwest": LayoutSpec("m132_audit_northwest", (0.70, -0.56), (-0.80, 0.18), (-0.44, 0.66), (0.72, 0.58), (-0.72, -0.56)),
    "m132_audit_southeast": LayoutSpec("m132_audit_southeast", (-0.70, 0.56), (0.34, -0.80), (0.70, -0.44), (-0.72, -0.58), (0.72, 0.56)),
    # M13.3 is a reward-contract experiment with a new disjoint split family.
    "m133_dev_northeast": LayoutSpec("m133_dev_northeast", (-0.20, -0.16), (0.18, 0.34), (0.46, 0.66), (0.56, -0.36), (-0.54, 0.38)),
    "m133_dev_southwest": LayoutSpec("m133_dev_southwest", (0.20, 0.16), (-0.66, -0.68), (-0.32, -0.34), (-0.54, 0.38), (0.54, -0.36)),
    "m133_dev_northwest": LayoutSpec("m133_dev_northwest", (0.24, -0.18), (-0.70, 0.16), (-0.36, 0.64), (0.52, 0.42), (-0.52, -0.40)),
    "m133_dev_southeast": LayoutSpec("m133_dev_southeast", (-0.24, 0.18), (0.20, -0.70), (0.66, -0.36), (-0.52, -0.42), (0.52, 0.40)),
    "m133_validation_northeast": LayoutSpec("m133_validation_northeast", (-0.44, -0.34), (0.30, 0.38), (0.66, 0.74), (0.72, -0.44), (-0.72, 0.56)),
    "m133_validation_southwest": LayoutSpec("m133_validation_southwest", (0.44, 0.34), (-0.76, -0.74), (-0.40, -0.38), (-0.72, 0.56), (0.72, -0.44)),
    "m133_validation_northwest": LayoutSpec("m133_validation_northwest", (0.58, -0.46), (-0.78, 0.18), (-0.42, 0.70), (0.68, 0.58), (-0.68, -0.56)),
    "m133_validation_southeast": LayoutSpec("m133_validation_southeast", (-0.58, 0.46), (0.34, -0.78), (0.70, -0.42), (-0.68, -0.58), (0.68, 0.56)),
    "m133_audit_northeast": LayoutSpec("m133_audit_northeast", (-0.70, -0.58), (0.28, 0.40), (0.64, 0.78), (0.78, -0.54), (-0.76, 0.60)),
    "m133_audit_southwest": LayoutSpec("m133_audit_southwest", (0.70, 0.58), (-0.80, -0.78), (-0.44, -0.40), (-0.76, 0.60), (0.76, -0.54)),
    "m133_audit_northwest": LayoutSpec("m133_audit_northwest", (0.72, -0.58), (-0.82, 0.16), (-0.46, 0.68), (0.74, 0.60), (-0.74, -0.58)),
    "m133_audit_southeast": LayoutSpec("m133_audit_southeast", (-0.72, 0.58), (0.36, -0.82), (0.72, -0.46), (-0.74, -0.60), (0.74, 0.58)),
    # M13.4 is a public macro-precondition experiment and uses a wholly fresh split.
    "m134_dev_northeast": LayoutSpec("m134_dev_northeast", (-0.18, -0.22), (0.20, 0.30), (0.50, 0.62), (0.58, -0.34), (-0.56, 0.36)),
    "m134_dev_southwest": LayoutSpec("m134_dev_southwest", (0.18, 0.22), (-0.64, -0.70), (-0.30, -0.36), (-0.56, 0.36), (0.58, -0.34)),
    "m134_dev_northwest": LayoutSpec("m134_dev_northwest", (0.22, -0.22), (-0.72, 0.14), (-0.38, 0.60), (0.50, 0.40), (-0.54, -0.42)),
    "m134_dev_southeast": LayoutSpec("m134_dev_southeast", (-0.22, 0.22), (0.18, -0.72), (0.64, -0.38), (-0.54, -0.42), (0.50, 0.40)),
    "m134_validation_northeast": LayoutSpec("m134_validation_northeast", (-0.42, -0.32), (0.32, 0.36), (0.68, 0.72), (0.74, -0.42), (-0.74, 0.54)),
    "m134_validation_southwest": LayoutSpec("m134_validation_southwest", (0.42, 0.32), (-0.78, -0.72), (-0.42, -0.36), (-0.74, 0.54), (0.74, -0.42)),
    "m134_validation_northwest": LayoutSpec("m134_validation_northwest", (0.60, -0.42), (-0.80, 0.16), (-0.44, 0.68), (0.70, 0.56), (-0.70, -0.54)),
    "m134_validation_southeast": LayoutSpec("m134_validation_southeast", (-0.60, 0.42), (0.36, -0.80), (0.72, -0.44), (-0.70, -0.56), (0.70, 0.54)),
    "m134_audit_northeast": LayoutSpec("m134_audit_northeast", (-0.74, -0.60), (0.30, 0.42), (0.66, 0.80), (0.80, -0.56), (-0.78, 0.62)),
    "m134_audit_southwest": LayoutSpec("m134_audit_southwest", (0.74, 0.60), (-0.82, -0.80), (-0.46, -0.42), (-0.78, 0.62), (0.80, -0.56)),
    "m134_audit_northwest": LayoutSpec("m134_audit_northwest", (0.76, -0.60), (-0.84, 0.14), (-0.48, 0.70), (0.76, 0.62), (-0.76, -0.60)),
    "m134_audit_southeast": LayoutSpec("m134_audit_southeast", (-0.76, 0.60), (0.38, -0.84), (0.74, -0.48), (-0.76, -0.62), (0.76, 0.60)),
    # M13.5 is a training-dynamics diagnosis with a fresh split family.
    "m135_dev_northeast": LayoutSpec("m135_dev_northeast", (-0.16, -0.24), (0.22, 0.28), (0.52, 0.60), (0.60, -0.32), (-0.58, 0.34)),
    "m135_dev_southwest": LayoutSpec("m135_dev_southwest", (0.16, 0.24), (-0.62, -0.72), (-0.28, -0.38), (-0.58, 0.34), (0.60, -0.32)),
    "m135_dev_northwest": LayoutSpec("m135_dev_northwest", (0.20, -0.24), (-0.74, 0.12), (-0.40, 0.58), (0.48, 0.38), (-0.56, -0.44)),
    "m135_dev_southeast": LayoutSpec("m135_dev_southeast", (-0.20, 0.24), (0.16, -0.74), (0.62, -0.40), (-0.56, -0.44), (0.48, 0.38)),
    "m135_validation_northeast": LayoutSpec("m135_validation_northeast", (-0.40, -0.30), (0.34, 0.34), (0.70, 0.70), (0.76, -0.40), (-0.76, 0.52)),
    "m135_validation_southwest": LayoutSpec("m135_validation_southwest", (0.40, 0.30), (-0.80, -0.70), (-0.44, -0.34), (-0.76, 0.52), (0.76, -0.40)),
    "m135_validation_northwest": LayoutSpec("m135_validation_northwest", (0.62, -0.40), (-0.82, 0.14), (-0.46, 0.66), (0.72, 0.54), (-0.72, -0.52)),
    "m135_validation_southeast": LayoutSpec("m135_validation_southeast", (-0.62, 0.40), (0.38, -0.82), (0.74, -0.46), (-0.72, -0.54), (0.72, 0.52)),
    "m135_audit_northeast": LayoutSpec("m135_audit_northeast", (-0.78, -0.62), (0.32, 0.44), (0.68, 0.82), (0.82, -0.58), (-0.80, 0.64)),
    "m135_audit_southwest": LayoutSpec("m135_audit_southwest", (0.78, 0.62), (-0.84, -0.82), (-0.48, -0.44), (-0.80, 0.64), (0.82, -0.58)),
    "m135_audit_northwest": LayoutSpec("m135_audit_northwest", (0.80, -0.62), (-0.86, 0.12), (-0.50, 0.68), (0.78, 0.64), (-0.78, -0.62)),
    "m135_audit_southeast": LayoutSpec("m135_audit_southeast", (-0.80, 0.62), (0.40, -0.86), (0.76, -0.50), (-0.78, -0.64), (0.78, 0.62)),
    # M13.6 is a fresh seed-stability replication; no M13.5 layouts are reused.
    "m136_dev_northeast": LayoutSpec("m136_dev_northeast", (-0.12, -0.18), (0.36, 0.22), (0.64, 0.54), (0.56, -0.46), (-0.50, 0.40)),
    "m136_dev_southwest": LayoutSpec("m136_dev_southwest", (0.12, 0.18), (-0.56, -0.68), (-0.22, -0.30), (-0.50, 0.40), (0.56, -0.46)),
    "m136_dev_northwest": LayoutSpec("m136_dev_northwest", (0.26, -0.18), (-0.68, 0.20), (-0.34, 0.52), (0.44, 0.44), (-0.52, -0.38)),
    "m136_dev_southeast": LayoutSpec("m136_dev_southeast", (-0.26, 0.18), (0.20, -0.68), (0.58, -0.34), (-0.52, -0.38), (0.44, 0.44)),
    "m136_validation_northeast": LayoutSpec("m136_validation_northeast", (-0.46, -0.34), (0.40, 0.30), (0.74, 0.62), (0.70, -0.46), (-0.70, 0.48)),
    "m136_validation_southwest": LayoutSpec("m136_validation_southwest", (0.46, 0.34), (-0.76, -0.66), (-0.38, -0.28), (-0.70, 0.48), (0.70, -0.46)),
    "m136_validation_northwest": LayoutSpec("m136_validation_northwest", (0.58, -0.34), (-0.78, 0.18), (-0.42, 0.62), (0.68, 0.50), (-0.68, -0.48)),
    "m136_validation_southeast": LayoutSpec("m136_validation_southeast", (-0.58, 0.34), (0.34, -0.78), (0.70, -0.42), (-0.68, -0.50), (0.68, 0.48)),
    "m136_audit_northeast": LayoutSpec("m136_audit_northeast", (-0.74, -0.56), (0.38, 0.48), (0.72, 0.78), (0.80, -0.54), (-0.78, 0.60)),
    "m136_audit_southwest": LayoutSpec("m136_audit_southwest", (0.74, 0.56), (-0.82, -0.78), (-0.44, -0.40), (-0.78, 0.60), (0.80, -0.54)),
    "m136_audit_northwest": LayoutSpec("m136_audit_northwest", (0.76, -0.56), (-0.84, 0.10), (-0.46, 0.66), (0.74, 0.60), (-0.76, -0.58)),
    "m136_audit_southeast": LayoutSpec("m136_audit_southeast", (-0.76, 0.56), (0.36, -0.84), (0.74, -0.46), (-0.76, -0.60), (0.74, 0.58)),
    "m137_dev_northeast": LayoutSpec("m137_dev_northeast", (-0.10, -0.20), (0.30, 0.20), (0.60, 0.56), (0.54, -0.44), (-0.48, 0.38)),
    "m137_dev_southwest": LayoutSpec("m137_dev_southwest", (0.10, 0.20), (-0.54, -0.66), (-0.20, -0.28), (-0.48, 0.38), (0.54, -0.44)),
    "m137_dev_northwest": LayoutSpec("m137_dev_northwest", (0.24, -0.20), (-0.66, 0.18), (-0.32, 0.50), (0.42, 0.42), (-0.50, -0.36)),
    "m137_dev_southeast": LayoutSpec("m137_dev_southeast", (-0.24, 0.20), (0.18, -0.66), (0.56, -0.32), (-0.50, -0.36), (0.42, 0.42)),
    "m137_validation_northeast": LayoutSpec("m137_validation_northeast", (-0.42, -0.32), (0.38, 0.28), (0.70, 0.60), (0.66, -0.44), (-0.66, 0.46)),
    "m137_validation_southwest": LayoutSpec("m137_validation_southwest", (0.42, 0.32), (-0.72, -0.64), (-0.36, -0.26), (-0.66, 0.46), (0.66, -0.44)),
    "m137_validation_northwest": LayoutSpec("m137_validation_northwest", (0.56, -0.32), (-0.74, 0.16), (-0.40, 0.60), (0.64, 0.48), (-0.64, -0.46)),
    "m137_validation_southeast": LayoutSpec("m137_validation_southeast", (-0.56, 0.32), (0.32, -0.74), (0.68, -0.40), (-0.64, -0.48), (0.64, 0.46)),
    # M13.8 predeclares five mutually disjoint reward-study partitions.  The
    # suffix fixes the food quadrant; agent, toy, and rest positions preserve
    # the corresponding M13 condition geometry.  Three-decimal coordinates
    # make every new point auditable against the retired M13--M13.7 layouts.
    "m138_screen_northeast": LayoutSpec(
        "m138_screen_northeast",
        (-0.115, -0.185),
        (0.315, 0.215),
        (0.615, 0.565),
        (0.545, -0.435),
        (-0.485, 0.385),
    ),
    "m138_screen_southwest": LayoutSpec(
        "m138_screen_southwest",
        (0.125, 0.195),
        (-0.565, -0.675),
        (-0.225, -0.295),
        (-0.495, 0.405),
        (0.555, -0.455),
    ),
    "m138_screen_northwest": LayoutSpec(
        "m138_screen_northwest",
        (0.255, -0.195),
        (-0.675, 0.175),
        (-0.335, 0.515),
        (0.435, 0.425),
        (-0.515, -0.375),
    ),
    "m138_screen_southeast": LayoutSpec(
        "m138_screen_southeast",
        (-0.245, 0.205),
        (0.185, -0.685),
        (0.575, -0.345),
        (-0.525, -0.365),
        (0.445, 0.415),
    ),
    "m138_probe_northeast": LayoutSpec(
        "m138_probe_northeast",
        (-0.365, -0.295),
        (0.405, 0.275),
        (0.725, 0.615),
        (0.675, -0.455),
        (-0.655, 0.475),
    ),
    "m138_probe_southwest": LayoutSpec(
        "m138_probe_southwest",
        (0.375, 0.305),
        (-0.745, -0.655),
        (-0.395, -0.275),
        (-0.665, 0.485),
        (0.685, -0.465),
    ),
    "m138_probe_northwest": LayoutSpec(
        "m138_probe_northwest",
        (0.535, -0.315),
        (-0.765, 0.155),
        (-0.415, 0.605),
        (0.635, 0.495),
        (-0.625, -0.475),
    ),
    "m138_probe_southeast": LayoutSpec(
        "m138_probe_southeast",
        (-0.545, 0.325),
        (0.335, -0.775),
        (0.695, -0.415),
        (-0.635, -0.485),
        (0.645, 0.505),
    ),
    "m138_confirm_fit_northeast": LayoutSpec(
        "m138_confirm_fit_northeast",
        (-0.085, -0.225),
        (0.285, 0.235),
        (0.585, 0.595),
        (0.525, -0.475),
        (-0.465, 0.415),
    ),
    "m138_confirm_fit_southwest": LayoutSpec(
        "m138_confirm_fit_southwest",
        (0.095, 0.235),
        (-0.585, -0.705),
        (-0.245, -0.325),
        (-0.475, 0.435),
        (0.535, -0.485),
    ),
    "m138_confirm_fit_northwest": LayoutSpec(
        "m138_confirm_fit_northwest",
        (0.275, -0.225),
        (-0.705, 0.195),
        (-0.365, 0.545),
        (0.455, 0.455),
        (-0.535, -0.405),
    ),
    "m138_confirm_fit_southeast": LayoutSpec(
        "m138_confirm_fit_southeast",
        (-0.265, 0.235),
        (0.215, -0.715),
        (0.595, -0.365),
        (-0.545, -0.395),
        (0.465, 0.445),
    ),
    "m138_confirm_eval_northeast": LayoutSpec(
        "m138_confirm_eval_northeast",
        (-0.435, -0.345),
        (0.425, 0.325),
        (0.755, 0.665),
        (0.705, -0.495),
        (-0.685, 0.525),
    ),
    "m138_confirm_eval_southwest": LayoutSpec(
        "m138_confirm_eval_southwest",
        (0.445, 0.355),
        (-0.775, -0.695),
        (-0.425, -0.315),
        (-0.695, 0.535),
        (0.715, -0.505),
    ),
    "m138_confirm_eval_northwest": LayoutSpec(
        "m138_confirm_eval_northwest",
        (0.585, -0.355),
        (-0.795, 0.145),
        (-0.445, 0.645),
        (0.665, 0.545),
        (-0.655, -0.515),
    ),
    "m138_confirm_eval_southeast": LayoutSpec(
        "m138_confirm_eval_southeast",
        (-0.595, 0.365),
        (0.365, -0.805),
        (0.725, -0.445),
        (-0.665, -0.525),
        (0.675, 0.555),
    ),
    "m138_audit_northeast": LayoutSpec(
        "m138_audit_northeast",
        (-0.705, -0.575),
        (0.355, 0.465),
        (0.695, 0.805),
        (0.815, -0.565),
        (-0.785, 0.635),
    ),
    "m138_audit_southwest": LayoutSpec(
        "m138_audit_southwest",
        (0.715, 0.585),
        (-0.835, -0.805),
        (-0.475, -0.425),
        (-0.795, 0.645),
        (0.825, -0.575),
    ),
    "m138_audit_northwest": LayoutSpec(
        "m138_audit_northwest",
        (0.745, -0.585),
        (-0.855, 0.095),
        (-0.485, 0.675),
        (0.765, 0.655),
        (-0.755, -0.605),
    ),
    "m138_audit_southeast": LayoutSpec(
        "m138_audit_southeast",
        (-0.755, 0.595),
        (0.385, -0.865),
        (0.765, -0.485),
        (-0.765, -0.615),
        (0.775, 0.665),
    ),
    # M13.9 is a fresh semi-Markov reward study. Every complete tuple and each
    # constituent point is distinct from M10--M13.8 protocol geometry.
    "m139_screen_northeast": LayoutSpec(
        "m139_screen_northeast",
        (-0.102, -0.202),
        (0.328, 0.198),
        (0.628, 0.548),
        (0.558, -0.452),
        (-0.472, 0.368),
    ),
    "m139_screen_southwest": LayoutSpec(
        "m139_screen_southwest",
        (0.138, 0.178),
        (-0.552, -0.692),
        (-0.212, -0.312),
        (-0.482, 0.388),
        (0.568, -0.472),
    ),
    "m139_screen_northwest": LayoutSpec(
        "m139_screen_northwest",
        (0.268, -0.212),
        (-0.662, 0.158),
        (-0.322, 0.498),
        (0.448, 0.408),
        (-0.502, -0.392),
    ),
    "m139_screen_southeast": LayoutSpec(
        "m139_screen_southeast",
        (-0.232, 0.188),
        (0.198, -0.702),
        (0.588, -0.362),
        (-0.512, -0.382),
        (0.458, 0.398),
    ),
    "m139_probe_northeast": LayoutSpec(
        "m139_probe_northeast",
        (-0.382, -0.282),
        (0.388, 0.288),
        (0.708, 0.628),
        (0.658, -0.442),
        (-0.672, 0.488),
    ),
    "m139_probe_southwest": LayoutSpec(
        "m139_probe_southwest",
        (0.358, 0.318),
        (-0.762, -0.642),
        (-0.412, -0.262),
        (-0.682, 0.498),
        (0.668, -0.452),
    ),
    "m139_probe_northwest": LayoutSpec(
        "m139_probe_northwest",
        (0.518, -0.302),
        (-0.782, 0.168),
        (-0.432, 0.618),
        (0.618, 0.508),
        (-0.642, -0.462),
    ),
    "m139_probe_southeast": LayoutSpec(
        "m139_probe_southeast",
        (-0.562, 0.338),
        (0.318, -0.762),
        (0.678, -0.402),
        (-0.652, -0.472),
        (0.628, 0.518),
    ),
    "m139_confirm_fit_northeast": LayoutSpec(
        "m139_confirm_fit_northeast",
        (-0.066, -0.214),
        (0.304, 0.246),
        (0.604, 0.606),
        (0.544, -0.464),
        (-0.446, 0.426),
    ),
    "m139_confirm_fit_southwest": LayoutSpec(
        "m139_confirm_fit_southwest",
        (0.114, 0.246),
        (-0.566, -0.694),
        (-0.226, -0.314),
        (-0.456, 0.446),
        (0.554, -0.474),
    ),
    "m139_confirm_fit_northwest": LayoutSpec(
        "m139_confirm_fit_northwest",
        (0.294, -0.214),
        (-0.686, 0.206),
        (-0.346, 0.556),
        (0.474, 0.466),
        (-0.516, -0.394),
    ),
    "m139_confirm_fit_southeast": LayoutSpec(
        "m139_confirm_fit_southeast",
        (-0.246, 0.246),
        (0.234, -0.704),
        (0.614, -0.354),
        (-0.526, -0.384),
        (0.484, 0.456),
    ),
    "m139_confirm_eval_northeast": LayoutSpec(
        "m139_confirm_eval_northeast",
        (-0.448, -0.364),
        (0.412, 0.306),
        (0.742, 0.646),
        (0.692, -0.514),
        (-0.698, 0.506),
    ),
    "m139_confirm_eval_southwest": LayoutSpec(
        "m139_confirm_eval_southwest",
        (0.432, 0.336),
        (-0.788, -0.714),
        (-0.438, -0.334),
        (-0.708, 0.516),
        (0.702, -0.524),
    ),
    "m139_confirm_eval_northwest": LayoutSpec(
        "m139_confirm_eval_northwest",
        (0.572, -0.374),
        (-0.808, 0.126),
        (-0.458, 0.626),
        (0.652, 0.526),
        (-0.668, -0.534),
    ),
    "m139_confirm_eval_southeast": LayoutSpec(
        "m139_confirm_eval_southeast",
        (-0.608, 0.346),
        (0.352, -0.824),
        (0.712, -0.464),
        (-0.678, -0.544),
        (0.662, 0.536),
    ),
    "m139_audit_northeast": LayoutSpec(
        "m139_audit_northeast",
        (-0.414, -0.322),
        (0.446, 0.348),
        (0.776, 0.688),
        (0.726, -0.472),
        (-0.664, 0.548),
    ),
    "m139_audit_southwest": LayoutSpec(
        "m139_audit_southwest",
        (0.466, 0.378),
        (-0.754, -0.672),
        (-0.404, -0.292),
        (-0.674, 0.558),
        (0.736, -0.482),
    ),
    "m139_audit_northwest": LayoutSpec(
        "m139_audit_northwest",
        (0.606, -0.332),
        (-0.774, 0.168),
        (-0.424, 0.668),
        (0.686, 0.568),
        (-0.634, -0.492),
    ),
    "m139_audit_southeast": LayoutSpec(
        "m139_audit_southeast",
        (-0.574, 0.388),
        (0.386, -0.782),
        (0.746, -0.422),
        (-0.644, -0.502),
        (0.696, 0.578),
    ),
    # M13.10 tests imitation-initialized RL on a fresh split family.  Every
    # tuple and every constituent point is disjoint from M10--M13.9.
    "m1310_screen_northeast": LayoutSpec(
        "m1310_screen_northeast", (-0.095, -0.211), (0.335, 0.189), (0.635, 0.539), (0.565, -0.461), (-0.465, 0.359)
    ),
    "m1310_screen_southwest": LayoutSpec(
        "m1310_screen_southwest", (0.145, 0.169), (-0.545, -0.701), (-0.205, -0.321), (-0.475, 0.379), (0.575, -0.481)
    ),
    "m1310_screen_northwest": LayoutSpec(
        "m1310_screen_northwest", (0.275, -0.221), (-0.655, 0.149), (-0.315, 0.489), (0.455, 0.399), (-0.495, -0.401)
    ),
    "m1310_screen_southeast": LayoutSpec(
        "m1310_screen_southeast", (-0.225, 0.179), (0.205, -0.711), (0.595, -0.371), (-0.505, -0.391), (0.465, 0.389)
    ),
    "m1310_probe_northeast": LayoutSpec(
        "m1310_probe_northeast", (-0.391, -0.271), (0.379, 0.299), (0.699, 0.639), (0.649, -0.431), (-0.681, 0.499)
    ),
    "m1310_probe_southwest": LayoutSpec(
        "m1310_probe_southwest", (0.349, 0.329), (-0.771, -0.631), (-0.421, -0.251), (-0.691, 0.509), (0.659, -0.441)
    ),
    "m1310_probe_northwest": LayoutSpec(
        "m1310_probe_northwest", (0.509, -0.291), (-0.791, 0.179), (-0.441, 0.629), (0.609, 0.519), (-0.651, -0.451)
    ),
    "m1310_probe_southeast": LayoutSpec(
        "m1310_probe_southeast", (-0.571, 0.349), (0.309, -0.751), (0.669, -0.391), (-0.661, -0.461), (0.619, 0.529)
    ),
    "m1310_confirm_fit_northeast": LayoutSpec(
        "m1310_confirm_fit_northeast", (-0.055, -0.207), (0.315, 0.253), (0.615, 0.613), (0.555, -0.457), (-0.435, 0.433)
    ),
    "m1310_confirm_fit_southwest": LayoutSpec(
        "m1310_confirm_fit_southwest", (0.125, 0.253), (-0.555, -0.687), (-0.215, -0.307), (-0.445, 0.453), (0.565, -0.467)
    ),
    "m1310_confirm_fit_northwest": LayoutSpec(
        "m1310_confirm_fit_northwest", (0.305, -0.207), (-0.675, 0.213), (-0.335, 0.563), (0.485, 0.473), (-0.505, -0.387)
    ),
    "m1310_confirm_fit_southeast": LayoutSpec(
        "m1310_confirm_fit_southeast", (-0.235, 0.253), (0.245, -0.697), (0.625, -0.347), (-0.515, -0.377), (0.495, 0.463)
    ),
    "m1310_confirm_eval_northeast": LayoutSpec(
        "m1310_confirm_eval_northeast", (-0.461, -0.371), (0.399, 0.299), (0.729, 0.639), (0.679, -0.521), (-0.711, 0.499)
    ),
    "m1310_confirm_eval_southwest": LayoutSpec(
        "m1310_confirm_eval_southwest", (0.419, 0.329), (-0.801, -0.721), (-0.451, -0.341), (-0.721, 0.509), (0.689, -0.531)
    ),
    "m1310_confirm_eval_northwest": LayoutSpec(
        "m1310_confirm_eval_northwest", (0.559, -0.381), (-0.821, 0.119), (-0.471, 0.619), (0.639, 0.519), (-0.681, -0.541)
    ),
    "m1310_confirm_eval_southeast": LayoutSpec(
        "m1310_confirm_eval_southeast", (-0.621, 0.339), (0.339, -0.831), (0.699, -0.471), (-0.691, -0.551), (0.649, 0.529)
    ),
    "m1310_audit_northeast": LayoutSpec(
        "m1310_audit_northeast", (-0.405, -0.335), (0.455, 0.335), (0.785, 0.675), (0.735, -0.485), (-0.655, 0.535)
    ),
    "m1310_audit_southwest": LayoutSpec(
        "m1310_audit_southwest", (0.475, 0.365), (-0.745, -0.685), (-0.395, -0.305), (-0.665, 0.545), (0.745, -0.495)
    ),
    "m1310_audit_northwest": LayoutSpec(
        "m1310_audit_northwest", (0.615, -0.345), (-0.763, 0.157), (-0.415, 0.655), (0.695, 0.555), (-0.625, -0.505)
    ),
    "m1310_audit_southeast": LayoutSpec(
        "m1310_audit_southeast", (-0.565, 0.375), (0.395, -0.795), (0.755, -0.435), (-0.635, -0.515), (0.705, 0.565)
    ),
    # M13.11 is a development-only PPO/DQN comparison.  Its eight layouts are
    # fresh against every earlier M10/M13 coordinate and split into fit/check.
    "m1311_fit_northeast": LayoutSpec("m1311_fit_northeast", (-0.117, -0.193), (0.347, 0.271), (0.657, 0.581), (0.587, -0.439), (-0.487, 0.371)),
    "m1311_fit_southwest": LayoutSpec("m1311_fit_southwest", (0.157, 0.243), (-0.573, -0.683), (-0.243, -0.353), (-0.463, 0.413), (0.587, -0.453)),
    "m1311_fit_northwest": LayoutSpec("m1311_fit_northwest", (0.287, -0.253), (-0.673, 0.173), (-0.343, 0.513), (0.477, 0.423), (-0.517, -0.413)),
    "m1311_fit_southeast": LayoutSpec("m1311_fit_southeast", (-0.247, 0.213), (0.233, -0.713), (0.623, -0.333), (-0.527, -0.403), (0.507, 0.403)),
    "m1311_check_northeast": LayoutSpec("m1311_check_northeast", (-0.413, -0.293), (0.397, 0.307), (0.717, 0.647), (0.667, -0.423), (-0.697, 0.507)),
    "m1311_check_southwest": LayoutSpec("m1311_check_southwest", (0.367, 0.347), (-0.783, -0.643), (-0.433, -0.263), (-0.703, 0.517), (0.677, -0.433)),
    "m1311_check_northwest": LayoutSpec("m1311_check_northwest", (0.527, -0.273), (-0.803, 0.193), (-0.453, 0.643), (0.627, 0.537), (-0.667, -0.433)),
    "m1311_check_southeast": LayoutSpec("m1311_check_southeast", (-0.593, 0.367), (0.327, -0.773), (0.687, -0.413), (-0.677, -0.473), (0.637, 0.547)),
    # M13.11-r2 retains M13.11's mechanics but owns fresh fit/check geometry.
    "m1311r2_fit_northeast": LayoutSpec("m1311r2_fit_northeast", (-0.131, -0.211), (0.361, 0.289), (0.681, 0.609), (0.611, -0.421), (-0.461, 0.389)),
    "m1311r2_fit_southwest": LayoutSpec("m1311r2_fit_southwest", (0.181, 0.267), (-0.601, -0.711), (-0.271, -0.381), (-0.491, 0.441), (0.611, -0.481)),
    "m1311r2_fit_northwest": LayoutSpec("m1311r2_fit_northwest", (0.311, -0.281), (-0.701, 0.201), (-0.371, 0.541), (0.501, 0.451), (-0.541, -0.441)),
    "m1311r2_fit_southeast": LayoutSpec("m1311r2_fit_southeast", (-0.271, 0.241), (0.261, -0.741), (0.651, -0.361), (-0.551, -0.431), (0.531, 0.431)),
    "m1311r2_check_northeast": LayoutSpec("m1311r2_check_northeast", (-0.439, -0.321), (0.421, 0.331), (0.751, 0.671), (0.701, -0.451), (-0.671, 0.531)),
    "m1311r2_check_southwest": LayoutSpec("m1311r2_check_southwest", (0.391, 0.371), (-0.811, -0.671), (-0.461, -0.291), (-0.731, 0.541), (0.701, -0.461)),
    "m1311r2_check_northwest": LayoutSpec("m1311r2_check_northwest", (0.551, -0.301), (-0.831, 0.221), (-0.481, 0.671), (0.651, 0.561), (-0.691, -0.461)),
    "m1311r2_check_southeast": LayoutSpec("m1311r2_check_southeast", (-0.621, 0.391), (0.361, -0.801), (0.721, -0.441), (-0.701, -0.501), (0.661, 0.571)),
    # M13.12 owns fresh geometry for its one-factor PPO learning-rate test.
    "m1312_fit_northeast": LayoutSpec("m1312_fit_northeast", (-0.149, -0.229), (0.379, 0.311), (0.699, 0.631), (0.629, -0.403), (-0.443, 0.407)),
    "m1312_fit_southwest": LayoutSpec("m1312_fit_southwest", (0.199, 0.283), (-0.619, -0.729), (-0.289, -0.399), (-0.509, 0.459), (0.629, -0.499)),
    "m1312_fit_northwest": LayoutSpec("m1312_fit_northwest", (0.329, -0.299), (-0.719, 0.219), (-0.389, 0.559), (0.519, 0.469), (-0.559, -0.459)),
    "m1312_fit_southeast": LayoutSpec("m1312_fit_southeast", (-0.289, 0.259), (0.279, -0.759), (0.669, -0.379), (-0.569, -0.449), (0.549, 0.449)),
    "m1312_check_northeast": LayoutSpec("m1312_check_northeast", (-0.457, -0.339), (0.439, 0.349), (0.769, 0.689), (0.719, -0.469), (-0.653, 0.549)),
    "m1312_check_southwest": LayoutSpec("m1312_check_southwest", (0.409, 0.389), (-0.829, -0.689), (-0.479, -0.309), (-0.749, 0.559), (0.719, -0.479)),
    "m1312_check_northwest": LayoutSpec("m1312_check_northwest", (0.569, -0.319), (-0.849, 0.239), (-0.499, 0.689), (0.669, 0.579), (-0.709, -0.479)),
    "m1312_check_southeast": LayoutSpec("m1312_check_southeast", (-0.639, 0.409), (0.379, -0.819), (0.739, -0.459), (-0.719, -0.519), (0.679, 0.589)),
    # M13.13 freezes five fresh partitions before any policy-family run.  The
    # four-decimal points are disjoint from every earlier M10--M13.12 layout.
    "m1313_fit_northeast": LayoutSpec("m1313_fit_northeast", (-0.1731, -0.2511), (0.4011, 0.3291), (0.7211, 0.6491), (0.6411, -0.3911), (-0.4211, 0.4191)),
    "m1313_fit_southwest": LayoutSpec("m1313_fit_southwest", (0.2232, 0.3012), (-0.6412, -0.7472), (-0.3112, -0.4172), (-0.5312, 0.4772), (0.6472, -0.5172)),
    "m1313_fit_northwest": LayoutSpec("m1313_fit_northwest", (0.3473, -0.3173), (-0.7373, 0.2373), (-0.4073, 0.5773), (0.5373, 0.4873), (-0.5773, -0.4773)),
    "m1313_fit_southeast": LayoutSpec("m1313_fit_southeast", (-0.3074, 0.2774), (0.2974, -0.7774), (0.6874, -0.3974), (-0.5874, -0.4674), (0.5674, 0.4674)),
    "m1313_check_northeast": LayoutSpec("m1313_check_northeast", (-0.4751, -0.3571), (0.4571, 0.3671), (0.7871, 0.7071), (0.7371, -0.4871), (-0.6351, 0.5671)),
    "m1313_check_southwest": LayoutSpec("m1313_check_southwest", (0.4272, 0.4072), (-0.8472, -0.7072), (-0.4972, -0.3272), (-0.7672, 0.5772), (0.7372, -0.4972)),
    "m1313_check_northwest": LayoutSpec("m1313_check_northwest", (0.5873, -0.3373), (-0.8673, 0.2573), (-0.5173, 0.7073), (0.6873, 0.5973), (-0.7273, -0.4973)),
    "m1313_check_southeast": LayoutSpec("m1313_check_southeast", (-0.6574, 0.4274), (0.3974, -0.8374), (0.7574, -0.4774), (-0.7374, -0.5374), (0.6974, 0.6074)),
    "m1313_confirm_fit_northeast": LayoutSpec("m1313_confirm_fit_northeast", (-0.1571, -0.2631), (0.4131, 0.3431), (0.7331, 0.6631), (0.6531, -0.3731), (-0.4091, 0.4331)),
    "m1313_confirm_fit_southwest": LayoutSpec("m1313_confirm_fit_southwest", (0.2372, 0.3152), (-0.6572, -0.7332), (-0.3272, -0.4032), (-0.5472, 0.4912), (0.6632, -0.5032)),
    "m1313_confirm_fit_northwest": LayoutSpec("m1313_confirm_fit_northwest", (0.3633, -0.3033), (-0.7533, 0.2513), (-0.4233, 0.5913), (0.5533, 0.5013), (-0.5933, -0.4633)),
    "m1313_confirm_fit_southeast": LayoutSpec("m1313_confirm_fit_southeast", (-0.3234, 0.2914), (0.3134, -0.7634), (0.7034, -0.3834), (-0.6034, -0.4534), (0.5834, 0.4814)),
    "m1313_confirm_eval_northeast": LayoutSpec("m1313_confirm_eval_northeast", (-0.4891, -0.3711), (0.4711, 0.3811), (0.8011, 0.7211), (0.7511, -0.5011), (-0.6211, 0.5811)),
    "m1313_confirm_eval_southwest": LayoutSpec("m1313_confirm_eval_southwest", (0.4412, 0.4212), (-0.8332, -0.7212), (-0.4832, -0.3412), (-0.7532, 0.5912), (0.7512, -0.5112)),
    "m1313_confirm_eval_northwest": LayoutSpec("m1313_confirm_eval_northwest", (0.6013, -0.3513), (-0.8533, 0.2713), (-0.5033, 0.7213), (0.7013, 0.6113), (-0.7413, -0.5113)),
    "m1313_confirm_eval_southeast": LayoutSpec("m1313_confirm_eval_southeast", (-0.6714, 0.4414), (0.4114, -0.8234), (0.7714, -0.4634), (-0.7514, -0.5514), (0.7114, 0.6214)),
    "m1313_audit_northeast": LayoutSpec("m1313_audit_northeast", (-0.4331, -0.3091), (0.4911, 0.3951), (0.8211, 0.7351), (0.7711, -0.5151), (-0.6071, 0.5951)),
    "m1313_audit_southwest": LayoutSpec("m1313_audit_southwest", (0.4552, 0.4352), (-0.8192, -0.7352), (-0.4692, -0.3552), (-0.7392, 0.6052), (0.7652, -0.5252)),
    "m1313_audit_northwest": LayoutSpec("m1313_audit_northwest", (0.6153, -0.3653), (-0.8393, 0.2853), (-0.4893, 0.7353), (0.7153, 0.6253), (-0.7553, -0.5253)),
    "m1313_audit_southeast": LayoutSpec("m1313_audit_southeast", (-0.6854, 0.4554), (0.4254, -0.8094), (0.7854, -0.4494), (-0.7654, -0.5654), (0.7254, 0.6354)),
    # M13.14 keeps M13.13 frozen and declares 20 fresh layouts across its
    # preflighted fit/check/confirmation/audit partitions.
    "m1314_fit_northeast": LayoutSpec("m1314_fit_northeast", (-0.18105, -0.24105), (0.38905, 0.31705), (0.70905, 0.63705), (0.62905, -0.37905), (-0.40905, 0.43105)),
    "m1314_fit_southwest": LayoutSpec("m1314_fit_southwest", (0.21515, 0.29315), (-0.62915, -0.73515), (-0.29915, -0.40515), (-0.51915, 0.48915), (0.65915, -0.50515)),
    "m1314_fit_northwest": LayoutSpec("m1314_fit_northwest", (0.33925, -0.32525), (-0.72525, 0.22525), (-0.39525, 0.56525), (0.54925, 0.49925), (-0.56525, -0.46525)),
    "m1314_fit_southeast": LayoutSpec("m1314_fit_southeast", (-0.31535, 0.28535), (0.28535, -0.76535), (0.67535, -0.38535), (-0.57535, -0.45535), (0.57935, 0.47935)),
    "m1314_check_northeast": LayoutSpec("m1314_check_northeast", (-0.48306, -0.34906), (0.44506, 0.35506), (0.77506, 0.69506), (0.72506, -0.47506), (-0.62306, 0.57906)),
    "m1314_check_southwest": LayoutSpec("m1314_check_southwest", (0.41916, 0.41516), (-0.83516, -0.69516), (-0.48516, -0.31516), (-0.75516, 0.58916), (0.74916, -0.48516)),
    "m1314_check_northwest": LayoutSpec("m1314_check_northwest", (0.57926, -0.32926), (-0.85526, 0.24526), (-0.50526, 0.69526), (0.69926, 0.60926), (-0.71526, -0.48526)),
    "m1314_check_southeast": LayoutSpec("m1314_check_southeast", (-0.64936, 0.43536), (0.38536, -0.82536), (0.74536, -0.46536), (-0.72536, -0.52536), (0.72336, 0.61936)),
    "m1314_confirm_fit_northeast": LayoutSpec("m1314_confirm_fit_northeast", (-0.14907, -0.25507), (0.42107, 0.33107), (0.74107, 0.65107), (0.66107, -0.36107), (-0.39707, 0.44107)),
    "m1314_confirm_fit_southwest": LayoutSpec("m1314_confirm_fit_southwest", (0.24517, 0.32317), (-0.64917, -0.72517), (-0.31917, -0.39517), (-0.53917, 0.49917), (0.67117, -0.49517)),
    "m1314_confirm_fit_northwest": LayoutSpec("m1314_confirm_fit_northwest", (0.37127, -0.29527), (-0.74527, 0.25927), (-0.41527, 0.59927), (0.56127, 0.50927), (-0.58527, -0.45527)),
    "m1314_confirm_fit_southeast": LayoutSpec("m1314_confirm_fit_southeast", (-0.31537, 0.29937), (0.32137, -0.75537), (0.71137, -0.37537), (-0.59537, -0.44537), (0.59137, 0.48937)),
    "m1314_confirm_eval_northeast": LayoutSpec("m1314_confirm_eval_northeast", (-0.48108, -0.36308), (0.47908, 0.36908), (0.80908, 0.70908), (0.75908, -0.48908), (-0.60908, 0.58908)),
    "m1314_confirm_eval_southwest": LayoutSpec("m1314_confirm_eval_southwest", (0.44918, 0.42918), (-0.82518, -0.71118), (-0.47518, -0.33118), (-0.74518, 0.59918), (0.75918, -0.49918)),
    "m1314_confirm_eval_northwest": LayoutSpec("m1314_confirm_eval_northwest", (0.60928, -0.34328), (-0.84528, 0.27928), (-0.49528, 0.72928), (0.70928, 0.61928), (-0.72928, -0.49928)),
    "m1314_confirm_eval_southeast": LayoutSpec("m1314_confirm_eval_southeast", (-0.66338, 0.44938), (0.41938, -0.81538), (0.77938, -0.45538), (-0.73938, -0.53938), (0.73738, 0.62938)),
    "m1314_audit_northeast": LayoutSpec("m1314_audit_northeast", (-0.42509, -0.30109), (0.49909, 0.40309), (0.82909, 0.72309), (0.77909, -0.50309), (-0.59509, 0.60309)),
    "m1314_audit_southwest": LayoutSpec("m1314_audit_southwest", (0.46319, 0.44319), (-0.81119, -0.72919), (-0.46119, -0.34919), (-0.73119, 0.61319), (0.77319, -0.51319)),
    "m1314_audit_northwest": LayoutSpec("m1314_audit_northwest", (0.62329, -0.35729), (-0.83129, 0.29329), (-0.48129, 0.74329), (0.72329, 0.63329), (-0.74329, -0.51329)),
    "m1314_audit_southeast": LayoutSpec("m1314_audit_southeast", (-0.67739, 0.46339), (0.43339, -0.80139), (0.79339, -0.44139), (-0.75339, -0.55339), (0.75139, 0.64339)),
    # M13.14-r2 keeps M13.14 frozen and declares a fresh additive split family
    # for the corrected teacher/artifact protocol lane.
    "m1314r2_fit_northeast": LayoutSpec("m1314r2_fit_northeast", (-0.18346, -0.24346), (0.39146, 0.31946), (0.71146, 0.63946), (0.63146, -0.38146), (-0.41146, 0.43346)),
    "m1314r2_fit_southwest": LayoutSpec("m1314r2_fit_southwest", (0.21756, 0.29556), (-0.63156, -0.73756), (-0.30156, -0.40756), (-0.52156, 0.49156), (0.66156, -0.50756)),
    "m1314r2_fit_northwest": LayoutSpec("m1314r2_fit_northwest", (0.34166, -0.32766), (-0.72766, 0.22766), (-0.39766, 0.56766), (0.55166, 0.50166), (-0.56766, -0.46766)),
    "m1314r2_fit_southeast": LayoutSpec("m1314r2_fit_southeast", (-0.31776, 0.28776), (0.28776, -0.76776), (0.67776, -0.38776), (-0.57776, -0.45776), (0.58176, 0.48176)),
    "m1314r2_check_northeast": LayoutSpec("m1314r2_check_northeast", (-0.48547, -0.35147), (0.44747, 0.35747), (0.77747, 0.69747), (0.72747, -0.47747), (-0.62547, 0.58147)),
    "m1314r2_check_southwest": LayoutSpec("m1314r2_check_southwest", (0.42157, 0.41757), (-0.83757, -0.69757), (-0.48757, -0.31757), (-0.75757, 0.59157), (0.75157, -0.48757)),
    "m1314r2_check_northwest": LayoutSpec("m1314r2_check_northwest", (0.58167, -0.33167), (-0.85767, 0.24767), (-0.50767, 0.69767), (0.70167, 0.61167), (-0.71767, -0.48767)),
    "m1314r2_check_southeast": LayoutSpec("m1314r2_check_southeast", (-0.65177, 0.43777), (0.38777, -0.82777), (0.74777, -0.46777), (-0.72777, -0.52777), (0.72577, 0.62177)),
    "m1314r2_confirm_fit_northeast": LayoutSpec("m1314r2_confirm_fit_northeast", (-0.15148, -0.25748), (0.42348, 0.33348), (0.74348, 0.65348), (0.66348, -0.36348), (-0.39948, 0.44348)),
    "m1314r2_confirm_fit_southwest": LayoutSpec("m1314r2_confirm_fit_southwest", (0.24758, 0.32558), (-0.65158, -0.72758), (-0.32158, -0.39758), (-0.54158, 0.50158), (0.67358, -0.49758)),
    "m1314r2_confirm_fit_northwest": LayoutSpec("m1314r2_confirm_fit_northwest", (0.37368, -0.29768), (-0.74768, 0.26168), (-0.41768, 0.60168), (0.56368, 0.51168), (-0.58768, -0.45768)),
    "m1314r2_confirm_fit_southeast": LayoutSpec("m1314r2_confirm_fit_southeast", (-0.31778, 0.30178), (0.32378, -0.75778), (0.71378, -0.37778), (-0.59778, -0.44778), (0.59378, 0.49178)),
    "m1314r2_confirm_eval_northeast": LayoutSpec("m1314r2_confirm_eval_northeast", (-0.48349, -0.36549), (0.48149, 0.37149), (0.81149, 0.71149), (0.76149, -0.49149), (-0.61149, 0.59149)),
    "m1314r2_confirm_eval_southwest": LayoutSpec("m1314r2_confirm_eval_southwest", (0.45159, 0.43159), (-0.82759, -0.71359), (-0.47759, -0.33359), (-0.74759, 0.60159), (0.76159, -0.50159)),
    "m1314r2_confirm_eval_northwest": LayoutSpec("m1314r2_confirm_eval_northwest", (0.61169, -0.34569), (-0.84769, 0.28169), (-0.49769, 0.73169), (0.71169, 0.62169), (-0.73169, -0.50169)),
    "m1314r2_confirm_eval_southeast": LayoutSpec("m1314r2_confirm_eval_southeast", (-0.66579, 0.45179), (0.42179, -0.81779), (0.78179, -0.45779), (-0.74179, -0.54179), (0.73979, 0.63179)),
    "m1314r2_audit_northeast": LayoutSpec("m1314r2_audit_northeast", (-0.42750, -0.30350), (0.50150, 0.40550), (0.83150, 0.72550), (0.78150, -0.50550), (-0.59750, 0.60550)),
    "m1314r2_audit_southwest": LayoutSpec("m1314r2_audit_southwest", (0.46560, 0.44560), (-0.81360, -0.73160), (-0.46360, -0.35160), (-0.73360, 0.61560), (0.77560, -0.51560)),
    "m1314r2_audit_northwest": LayoutSpec("m1314r2_audit_northwest", (0.62570, -0.35970), (-0.83370, 0.29570), (-0.48370, 0.74570), (0.72570, 0.63570), (-0.74570, -0.51570)),
    "m1314r2_audit_southeast": LayoutSpec("m1314r2_audit_southeast", (-0.67980, 0.46580), (0.43580, -0.80380), (0.79580, -0.44380), (-0.75580, -0.55580), (0.75380, 0.64580)),
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
    "maintain_needs": TaskSpec(
        name="maintain_needs",
        description="Feed and play in one episode while maintaining satiety and relieving boredom.",
        train_layout_ids=("m9_train_northeast", "m9_train_southwest"),
        heldout_layout_ids=("m9_validation_northeast", "m9_validation_southwest"),
        initial_satiety=0.25,
        initial_boredom=0.85,
        success_condition="maintain_needs",
    ),
    "persistent_maintenance": TaskSpec(
        name="persistent_maintenance",
        description="Maintain satiety, energy, and boredom across repeated feed, play, and rest cycles.",
        train_layout_ids=("m10_train_northeast", "m10_train_southwest"),
        heldout_layout_ids=("m10_validation_northeast", "m10_validation_southwest"),
        initial_satiety=0.45,
        initial_energy=0.65,
        initial_boredom=0.65,
        success_condition="persistent_maintenance",
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
