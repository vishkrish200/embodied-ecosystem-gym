from __future__ import annotations

import numpy as np

from ecosystem_gym.actions import ActionKind, ActionOutcome
from ecosystem_gym.m12 import M12Macro, M12Memory, M12Selector, _advance_memory, _food_target, _macro_action, _mandatory_macro, _reset_no_memory, _select_macro, m12_config


def _observation(*, holding_food: bool = False, outcome: ActionOutcome = ActionOutcome.SUCCESS) -> dict[str, object]:
    return {
        "holding_food": holding_food,
        "prior_outcome": list(ActionOutcome).index(outcome),
        "drives": np.asarray((0.8, 0.8, 0.2), dtype=np.float32),
    }


def test_m12_scan_cycle_is_four_views_before_need_selection() -> None:
    memory = M12Memory()
    config = m12_config()

    for expected_views in range(1, 5):
        assert _mandatory_macro(_observation(), memory) is M12Macro.SCAN
        action = _macro_action(M12Macro.SCAN, memory, config)
        assert ActionKind(int(action["kind"])) is ActionKind.SCAN
        assert memory.scans_in_cycle == expected_views

    assert not memory.scanning


def test_m12_walk_transitions_interact_then_discard_stale_offsets() -> None:
    memory = M12Memory(scanning=False, toy_candidates=[np.asarray((0.4, 0.2), dtype=np.float32)])
    memory.last_macro = M12Macro.WALK_TOY
    _advance_memory(memory, _observation())

    assert memory.pending_play
    assert not memory.toy_candidates
    assert _mandatory_macro(_observation(), memory) is M12Macro.PLAY
    assert ActionKind(int(_macro_action(M12Macro.PLAY, memory, m12_config())["kind"])) is ActionKind.RUN_AROUND

    memory.last_macro = M12Macro.PLAY
    _advance_memory(memory, _observation())
    assert memory.play_done
    assert memory.scanning
    assert memory.scans_in_cycle == 0

    memory.scanning = False
    memory.rest_candidates.append(np.asarray((0.2, -0.3), dtype=np.float32))
    memory.last_macro = M12Macro.WALK_REST
    _advance_memory(memory, _observation())
    assert memory.pending_rest
    assert not memory.rest_candidates
    assert _mandatory_macro(_observation(), memory) is M12Macro.REST
    assert ActionKind(int(_macro_action(M12Macro.REST, memory, m12_config())["kind"])) is ActionKind.REST


def test_m12_blocked_pickup_restarts_a_clean_full_scan() -> None:
    memory = M12Memory(
        scanning=False,
        food_candidates=[np.asarray((0.5, 0.0), dtype=np.float32)],
        toy_candidates=[np.asarray((0.0, 0.5), dtype=np.float32)],
        rest_candidates=[np.asarray((0.0, -0.5), dtype=np.float32)],
        last_macro=M12Macro.PICK_UP,
    )
    _advance_memory(memory, _observation(outcome=ActionOutcome.BLOCKED))

    assert memory.blocked_attempts == 1
    assert memory.scanning
    assert not memory.food_candidates and not memory.toy_candidates and not memory.rest_candidates


def test_m12_selector_cannot_emit_an_interaction_without_its_public_transition() -> None:
    selector = M12Selector(
        hidden=np.zeros((2, 1), dtype=np.float64),
        hidden_bias=np.ones(1, dtype=np.float64),
        output=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        output_bias=np.zeros(len(M12Macro), dtype=np.float64),
    )

    assert selector.choose(np.zeros(2), allowed=(M12Macro.SCAN, M12Macro.WALK_TOY)) is M12Macro.SCAN
    assert _select_macro(M12Macro.SCAN, selector, np.zeros(2)) is M12Macro.SCAN


def test_m12_no_memory_ablation_keeps_only_the_current_observation() -> None:
    memory = M12Memory(
        scans_in_cycle=3,
        scanning=True,
        pending_pickup=True,
        food_candidates=[np.asarray((0.5, 0.0), dtype=np.float32)],
        completed_cycles=2,
    )
    _reset_no_memory(memory)

    assert not memory.scanning
    assert not memory.pending_pickup
    assert not memory.food_candidates
    assert memory.completed_cycles == 0


def test_m12_food_retry_prefers_the_nearby_fresh_candidate_after_relocation() -> None:
    memory = M12Memory(
        food_candidates=[
            np.asarray((0.4, -0.3), dtype=np.float32),
            np.asarray((1.0, -0.8), dtype=np.float32),
            np.asarray((0.2, 0.1), dtype=np.float32),
        ]
    )
    assert np.allclose(_food_target(memory), np.asarray((1.0, -0.8), dtype=np.float32))
    memory.blocked_attempts = 1
    assert np.allclose(_food_target(memory), np.asarray((0.2, 0.1), dtype=np.float32))
    memory.blocked_attempts = 2
    assert np.allclose(_food_target(memory), np.asarray((0.4, -0.3), dtype=np.float32))
