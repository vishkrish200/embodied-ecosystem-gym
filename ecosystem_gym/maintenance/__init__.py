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
from .teacher_data_m1314r2 import (
    PUBLIC_OBSERVATION_KEYS as PUBLIC_OBSERVATION_KEYS_M1314R2,
    TEACHER_DATASET_SCHEMA as TEACHER_DATASET_SCHEMA_M1314R2,
    TeacherDataset as TeacherDatasetM1314R2,
    TeacherMinibatch as TeacherMinibatchM1314R2,
    load_teacher_dataset as load_teacher_dataset_m1314r2,
    save_teacher_dataset as save_teacher_dataset_m1314r2,
    teacher_dataset_fingerprint as teacher_dataset_fingerprint_m1314r2,
    teacher_example as teacher_example_m1314r2,
    validate_public_observation as validate_public_observation_m1314r2,
)
from .modular_q_m1314r2 import (
    MODULAR_Q_ARM_SPECS as MODULAR_Q_ARM_SPECS_M1314R2,
    DurationReplayBuffer as DurationReplayBufferM1314R2,
    ModularQArm as ModularQArmM1314R2,
    ModularQArmSpec as ModularQArmSpecM1314R2,
    ModularQConfig as ModularQConfigM1314R2,
    ModularQPolicy as ModularQPolicyM1314R2,
    QNetworkKind as QNetworkKindM1314R2,
    ShieldedModularQPolicy as ShieldedModularQPolicyM1314R2,
    TeacherReplayBuffer as TeacherReplayBufferM1314R2,
    TransitionBatch as TransitionBatchM1314R2,
    dqfd_teacher_margin_loss as dqfd_teacher_margin_loss_m1314r2,
    duration_aware_double_dqn_targets as duration_aware_double_dqn_targets_m1314r2,
    masked_softmax as masked_softmax_m1314r2,
    modular_q_evaluation_spec as modular_q_evaluation_spec_m1314r2,
)
from .modular_q_artifacts_m1314r2 import (
    MODULAR_Q_ARTIFACT_SCHEMA as MODULAR_Q_ARTIFACT_SCHEMA_M1314R2,
    load_modular_q_policy as load_modular_q_policy_m1314r2,
    modular_q_artifact_payload as modular_q_artifact_payload_m1314r2,
    modular_q_policy_fingerprint as modular_q_policy_fingerprint_m1314r2,
    modular_q_weight_fingerprint as modular_q_weight_fingerprint_m1314r2,
    save_modular_q_policy as save_modular_q_policy_m1314r2,
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
    "DurationReplayBufferM1314R2", "MODULAR_Q_ARM_SPECS_M1314R2", "MODULAR_Q_ARTIFACT_SCHEMA_M1314R2",
    "ModularQArmM1314R2", "ModularQArmSpecM1314R2", "ModularQConfigM1314R2", "ModularQPolicyM1314R2",
    "PUBLIC_OBSERVATION_KEYS_M1314R2", "QNetworkKindM1314R2", "ShieldedModularQPolicyM1314R2",
    "TEACHER_DATASET_SCHEMA_M1314R2", "TeacherDatasetM1314R2", "TeacherMinibatchM1314R2",
    "TeacherReplayBufferM1314R2", "TransitionBatchM1314R2", "dqfd_teacher_margin_loss_m1314r2",
    "duration_aware_double_dqn_targets_m1314r2", "load_modular_q_policy_m1314r2", "load_teacher_dataset_m1314r2",
    "masked_softmax_m1314r2", "modular_q_artifact_payload_m1314r2", "modular_q_evaluation_spec_m1314r2",
    "modular_q_policy_fingerprint_m1314r2", "modular_q_weight_fingerprint_m1314r2", "save_modular_q_policy_m1314r2",
    "save_teacher_dataset_m1314r2", "teacher_dataset_fingerprint_m1314r2", "teacher_example_m1314r2",
    "validate_public_observation_m1314r2",
]
