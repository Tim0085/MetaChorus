"""Frozen identities and limits for the M7F-H1 execution protocol."""

from __future__ import annotations

from metachorus_v53.m6f_m7f_fast.constants import (
    CLASSIFICATION_SPLIT_SEEDS,
    PRELIM_KMEANS_SEEDS,
    SEARCH_KMEANS_SEEDS,
)


TASKBOOK_FILENAME = (
    'Stage status is recorded in the structured output.\n'
    'protocol_overview.md'
)
TASKBOOK_SHA256 = "3457b6f36794af960d6727d41d555864e67a02d06dfa9ab0c863a46d7c5f1eff"

PARENT_PACKAGE_FILENAME = (
    "20260814_0608_metachorus_v53_dblp_imdb_aminer_3seed_fast_MIN_a8902e0d.zip"
)
PARENT_PACKAGE_SHA256 = "8f7c159c8eaaf63050e88a0499ae0ed23b2a58f8986161436778fb6426084e1f"
PARENT_PACKAGE_SIZE_BYTES = 55_750
PARENT_SOURCE_COMMIT = "a8902e0d1e7ecac6a7a7c271f543232ed52b136f"
PARENT_SOURCE_TREE = "d0f71adaa438fbcbda007318ec60f08ae7cd8703"
PARENT_TERMINAL_STATUS = "FAST_3DATASET_PARTIAL"

DATASET_NAMES = ("dblp", "imdb", "aminer")
MANUAL_DATASETS = ("dblp", "aminer")
EVALUATION_MODEL_SEEDS = (3276665818, 1910071238, 610349320)
SEARCH_MODEL_SEEDS = (3139726158, 3297290585, 2204985712)
BUDGET_MODEL_SEEDS = SEARCH_MODEL_SEEDS[:2]

TRAJECTORY_UPDATES = 3000
TRAJECTORY_CHECKPOINT_SCHEDULE = tuple(range(500, 3001, 500))
TRAJECTORY_REPRESENTATION_SCHEDULE = tuple(range(50, 3001, 50))
MANUAL_CHECKPOINT_SCHEDULE = TRAJECTORY_CHECKPOINT_SCHEDULE
MANUAL_REPRESENTATION_SCHEDULE = (3000,)
BUDGET_CANARY_UPDATES = 300
BUDGET_CANARY_CHECKPOINT_SCHEDULE = (300,)
BUDGET_CANARY_REPRESENTATION_SCHEDULE = (300,)

FULL_SEARCH_ENDPOINTS = (300, 1000, 2000, 3000)
FULL_SEARCH_PRELIMINARY_ENDPOINT = 3000
FULL_SEARCH_STAGE_NAMES = (
    "FULL_SEARCH_R1",
    "FULL_SEARCH_R2",
    "FULL_SEARCH_R3",
    "FULL_SEARCH_R4",
    "FULL_SEARCH_PRELIMINARY",
)

IMDB_FINAL_SET = {
    "candidate_set_id": "fast_imdb_014_a91a43294eadd6bc",
    "candidate_set_hash": "a91a43294eadd6bc44b9f18af021be977f7edd7a3698a4eafd0a70b0b83c2280",
    "candidate_set_ordinal": 14,
}
IMDB_MANUAL_SET = {
    "candidate_set_id": "fast_imdb_000_1461aea830c29417",
    "candidate_set_hash": "1461aea830c29417d7ef79320e43d337e61efb99865a0e29bbc497c4b19ccb1c",
    "candidate_set_ordinal": 0,
}
MANUAL_SET_IDENTITIES = {
    "dblp": {
        "candidate_set_id": "fast_dblp_010_e7c3ad780037d0be",
        "candidate_set_hash": "e7c3ad780037d0be26082d7e7903e6bbff736a9b220bb9795627c00fac45aa06",
        "candidate_set_ordinal": 10,
    },
    "aminer": {
        "candidate_set_id": "fast_aminer_000_ed3a05ce6bb8f245",
        "candidate_set_hash": "ed3a05ce6bb8f24554c9fd3cde6eee27802ade3aae0b29523e8459ff2ad8410b",
        "candidate_set_ordinal": 0,
    },
}

CHECKPOINT_SELECTORS = (
    "MICRO_BEST",
    "NMI_BEST",
    "JOINT_BEST",
    "COMMON_JOINT_UPDATE",
    "FIXED_END_3000",
)

RESOURCE_LIMITS = {
    "hard_gpu_hours": 28.0,
    "hard_critical_path_hours": 16.0,
    "hard_remote_storage_bytes": 28 * 1024**3,
    "retry_reserve_fraction": 0.10,
    "target_package_bytes": 20 * 1024**2,
    "hard_package_bytes": 50 * 1024**2,
}

IMDB_TRAJECTORY_SOFT_GPU_HOURS = 5.0
IMDB_TRAJECTORY_HARD_GPU_HOURS = 8.0
MANUAL_CANARY_SOFT_GPU_HOURS = 8.0
MANUAL_CANARY_HARD_GPU_HOURS = 12.0

REQUIRED_TEST_FILES = (
    "test_parent_package_hash.py",
    "test_imdb_set_ids_hashes.py",
    "test_imdb_eval_seed_list.py",
    "test_existing_metric_source_identity.py",
    "test_no_locked_report_checkpoint_selection.py",
    "test_metric_best_not_spliced.py",
    "test_common_update_rule.py",
    "test_imdb_six_run_matrix_if_needed.py",
    "test_manual_baseline_six_runs.py",
    "test_budget_canary_exact_sets.py",
    "test_budget_canary_not_scientific_selection.py",
    "test_measured_budget_gate.py",
    "test_training_no_labels.py",
    "test_minimal_package_excludes_large_files.py",
    "test_checksums.py",
)

MINIMAL_MEMBER_NAMES = (
    "IMDB_EXISTING_METRIC_SOURCE_INVENTORY.csv",
    "IMDB_EXISTING_TRAJECTORY_METRICS.csv",
    "IMDB_TRAJECTORY_METRICS_LONG.csv",
    "IMDB_GLOBAL_BEST_BY_METRIC_AND_UPDATE.csv",
    "IMDB_BEST_BY_METRIC.csv",
    "IMDB_CHECKPOINT_SELECTION_TABLE.csv",
    "IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv",
    "IMDB_SELECTOR_SUMMARY.csv",
    "IMDB_SELECTED_VS_MANUAL_DELTAS.csv",
    "IMDB_TRAJECTORY_DECISION.json",
    "IMDB_TRAJECTORY_DECISION.md",
    "DBLP_AMINER_MANUAL_METRICS_BY_SEED.csv",
    "DBLP_AMINER_MANUAL_SUMMARY.csv",
    "DBLP_AMINER_BUDGET_CANARY_RUNS.csv",
    "DBLP_AMINER_MEASURED_BUDGET.csv",
    "DBLP_AMINER_SEARCH_AUTHORIZATION.json",
    "M7F_H1_FULL_SEARCH_AUTHORIZATION_BRIDGE.json",
    "M7F_H1_FULL_SEARCH_STATUS.json",
    "DBLP_AMINER_FULL_SEARCH_SELECTION_REGISTRY.csv",
    "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_METRICS_BY_SEED.csv",
    "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_SUMMARY.csv",
    "DBLP_AMINER_FULL_SEARCH_COMPARISON.csv",
    "UPDATED_FIVE_DATASET_STATUS_TABLE.csv",
    "UPDATED_FIVE_DATASET_STATUS_TABLE.md",
    "M7F_H1_RUN_REGISTRY_MINIMAL.csv",
    "M7F_H1_RESOURCE_SUMMARY.csv",
    "M7F_H1_ERROR_SUMMARY.csv",
    "REMOTE_M7F_H1_ARTIFACT_RETENTION_MANIFEST.csv",
    "M7F_H1_GIT_AND_CONFIG_IDENTITY.json",
    "TEST_RECEIPT_M7F_H1.json",
    "TEST_WARNING_REGISTRY_M7F_H1.csv",
    "M7F_H1_VALIDATION.json",
    "M7F_H1_COMPLETE.json",
)

__all__ = [name for name in globals() if name.isupper()]
