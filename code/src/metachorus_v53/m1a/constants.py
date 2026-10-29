"""Frozen constants for the M1A ACM gate."""

from __future__ import annotations

from typing import Final


SCHEMA_PREFIX: Final = "metachorus.v53.m1a"
DATASET: Final = "acm"
MODEL_SEEDS: Final[tuple[int, ...]] = (
    383130895,
    2009113329,
    448070258,
)
CHECKPOINT_SCHEDULE: Final[tuple[int, ...]] = (
    100,
    300,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
)
PATH_SET_ROLES: Final[tuple[str, ...]] = ("Manual", "Automatic")
PHYSICAL_TRAINING_COUNT: Final = 6
EVALUATION_KIND_COUNT: Final = 11
EVALUATION_ONLY_ROW_COUNT: Final = 66

DIAGNOSTIC_TEMPERATURE: Final = 0.2
DIAGNOSTIC_EPSILON: Final = 1e-12
PROBABILITY_CLAMP_MIN: Final = 1e-7
PROBABILITY_CLAMP_MAX: Final = 1.0 - PROBABILITY_CLAMP_MIN
ANCHOR_CAP: Final = 1024
NEGATIVE_CAP_PER_ANCHOR: Final = 64
STRUCTURE_PAIR_CAP_PER_PATH: Final = 4096
MASTER_BANK_SIZE: Final = 16
BANK_PREFIX_SIZES: Final[tuple[int, ...]] = (4, 8, 16)
INITIAL_WINDOW_LENGTHS: Final[tuple[int, ...]] = (1, 3, 5)
MIDRANK_TIE_ATOL: Final = 1e-12
MIDRANK_TIE_RTOL: Final = 1e-9

DIAGNOSTIC_BRANCHES: Final[tuple[str, ...]] = (
    "D_str",
    "D_cn",
    "D_cv",
    "D_ng",
)
REPRESENTATION_LEVELS: Final[tuple[str, ...]] = ("encoder", "projector")

RESOURCE_LIMITS: Final[dict[str, float | int]] = {
    "max_new_physical_trainings": 6,
    "max_training_gpu_hours": 1.25,
    "max_offline_diagnostic_gpu_hours": 2.50,
    "max_total_gpu_hours": 4.00,
    "max_storage_gb": 5.0,
    "max_failed_training_attempts_before_pause": 2,
    "max_failed_evaluation_jobs_before_pause": 5,
    "max_checkpoints_per_run": 8,
}

EVALUATION_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("E00", "Fixed-End"),
    ("E01", "Raw-O0-Min"),
    ("E02", "RankMe-Only"),
    ("E03", "UTS-L"),
    ("E04", "UTS-P"),
    ("E05", "UTS-LP"),
    ("E06", "UTS-F hard-guard"),
    ("E07", "UTS-F rank-aux"),
    ("E08", "window candidates"),
    ("E09", "bank-size/subsample audit"),
    ("E10", "encoder/projector diagnostic audit"),
)

FORBIDDEN_SELECTOR_TOKENS: Final[tuple[str, ...]] = (
    "label",
    "labels",
    "label_cache",
    "class_count",
    "num_classes",
    "svc",
    "kmeans",
    "downstream",
    "oracle",
)
