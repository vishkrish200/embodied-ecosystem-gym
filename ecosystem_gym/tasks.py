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
