"""Stable public maintenance-policy primitives for successor learners.

Frozen M13 experiment modules deliberately do not import this package.
"""

from .contract import MaintenanceMacro, MaintenanceMemory, advance_memory, compile_macro, eligible_macro_mask, encode_features
from .model_based import ModelBasedSchedulerConfig, ModelBasedSchedulerPolicy
from .modular_q import (
    MODULAR_Q_ARM_SPECS,
    DurationReplayBuffer,
    ModularQArm,
    ModularQArmSpec,
    ModularQConfig,
    ModularQPolicy,
    QNetworkKind,
    ShieldedModularQPolicy,
    TeacherBatch,
    TeacherReplayBuffer,
    TransitionBatch,
    dqfd_teacher_margin_loss,
    duration_aware_double_dqn_targets,
    masked_softmax,
    modular_q_evaluation_spec,
)
from .modular_q_artifacts import (
    MODULAR_Q_ARTIFACT_SCHEMA,
    load_modular_q_policy,
    modular_q_artifact_payload,
    modular_q_policy_fingerprint,
    modular_q_weight_fingerprint,
    save_modular_q_policy,
)
from .policy_artifacts import load_policy, policy_fingerprint, save_policy
from .policy_state import MaintenanceGoal, PolicyMemory, SafetyLimits, memory_snapshot, restore_memory
from .reward import MaintenanceRewardState, duration_discount, learning_reward
from .shielded import PublicMLPActor, ShieldConfig, ShieldedLearnedPolicy
from .teacher_data import (
    PUBLIC_OBSERVATION_KEYS,
    TeacherDataset,
    load_teacher_dataset,
    save_teacher_dataset,
    teacher_dataset_fingerprint,
    teacher_example,
    validate_public_observation,
)
from .urgency import UrgencySchedulerConfig, UrgencySchedulerPolicy

__all__ = [
    "DurationReplayBuffer", "MODULAR_Q_ARM_SPECS", "MODULAR_Q_ARTIFACT_SCHEMA", "MaintenanceGoal",
    "MaintenanceMacro", "MaintenanceMemory", "MaintenanceRewardState",
    "ModelBasedSchedulerConfig", "ModelBasedSchedulerPolicy", "ModularQArm", "ModularQArmSpec",
    "ModularQConfig", "ModularQPolicy", "PUBLIC_OBSERVATION_KEYS", "PolicyMemory", "PublicMLPActor",
    "QNetworkKind", "SafetyLimits", "ShieldConfig", "ShieldedLearnedPolicy", "ShieldedModularQPolicy",
    "TeacherBatch", "TeacherDataset", "TeacherReplayBuffer", "TransitionBatch", "UrgencySchedulerConfig",
    "UrgencySchedulerPolicy", "advance_memory", "compile_macro", "dqfd_teacher_margin_loss",
    "duration_aware_double_dqn_targets", "duration_discount", "eligible_macro_mask", "encode_features",
    "learning_reward", "load_modular_q_policy", "load_policy", "load_teacher_dataset", "masked_softmax",
    "memory_snapshot", "modular_q_artifact_payload", "modular_q_evaluation_spec", "modular_q_policy_fingerprint",
    "modular_q_weight_fingerprint", "policy_fingerprint", "restore_memory", "save_modular_q_policy", "save_policy",
    "save_teacher_dataset", "teacher_dataset_fingerprint", "teacher_example", "validate_public_observation",
]
