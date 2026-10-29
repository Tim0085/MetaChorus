"""Frozen identities and protocol constants for M8-F3R-RERUN-2G."""

from __future__ import annotations

TASKBOOK_FILENAME = 'protocol_m8-f3r-rerun-2g.md'
TASKBOOK_SHA256 = "586174c3e1a89ccbe037700f16df1c208de5f600b24eee56b1e66a870127168f"
TASKBOOK_STAGE = "M8-F3R-RERUN-2G"
SOURCE_TASKBOOK_FILENAME = 'protocol_m8-f3r.md'
SOURCE_TASKBOOK_SHA256 = "20d9a2002bad750d8a2d60a6871c8cdc062714d2a5fe82051c8d694dbc454ea6"

ABLATION_DATASETS = ("freebase", "dblp")
EXISTING_EVIDENCE_ONLY_DATASETS = ("acm", "imdb", "aminer")
ABLATION_VARIANTS = (
    "FULL_FROZEN",
    "SINGLE_SEED_RANK_S0",
    "SINGLE_SEED_RANK_S1",
    "SINGLE_SEED_RANK_S2",
    "NO_RANDOM_EXPLORATION",
    "MEDIAN_ONLY_NO_WORST_CASE",
    "BALANCED_SCALAR_ONLY_NO_PARETO",
    "CLASSIFICATION_ONLY",
    "CLUSTERING_ONLY",
    "NO_MANUAL_SAFETY",
)
REFERENCE_VARIANTS = ("FULL_FROZEN", "MANUAL")
FREEBASE_SEARCH_MODEL_SEEDS = (1985123271, 508898167, 1950017805)
DBLP_SEARCH_MODEL_SEEDS = (3139726158, 3297290585, 2204985712)
SEARCH_MODEL_SEEDS = {
    "freebase": FREEBASE_SEARCH_MODEL_SEEDS,
    "dblp": DBLP_SEARCH_MODEL_SEEDS,
}
EVALUATION_MODEL_SEEDS = (3276665818, 1910071238, 610349320)

EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
TRAINER = "O0-CodeFaithful"
VIEW_PIPELINE = "S0"
ENDPOINT_UPDATE = 3000
CANARY_ENDPOINT_UPDATE = 20
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

RUN_CAPS = {
    "maximum_variant_rows_per_dataset": 10,
    "maximum_unique_sets_per_dataset_including_manual": 11,
    "maximum_unique_sets_total": 22,
    "maximum_new_full_training_runs": 66,
    "maximum_search_continuation_runs": 6,
}
RESOURCE_LIMITS = {
    "soft_gpu_hours": 18.0,
    "hard_gpu_hours": 24.0,
    "soft_critical_path_hours": 10.0,
    "hard_critical_path_hours": 14.0,
    "soft_remote_storage_bytes": 10 * 1024**3,
    "hard_remote_storage_bytes": 15 * 1024**3,
    "soft_failed_attempts": 4,
    "hard_failed_attempts": 8,
    "hard_unrecovered_identities": 2,
    "target_package_bytes": 20 * 1024**2,
    "hard_package_bytes": 50 * 1024**2,
    "hard_member_bytes": 20 * 1024**2,
}

PYTEST_WARNING_ALLOWLIST = (
    (
        "DeprecationWarning",
        "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
    ),
)
REQUIRED_SNAPSHOT_PREFIXES = (
    "SNAPSHOT_0_SELECTION",
    "SNAPSHOT_1_FREEBASE",
    "SNAPSHOT_2_DBLP",
)
HISTORICAL_FULL_RUN_SECONDS = {"freebase": 359.1289792, "dblp": 1265.1445267}




DATASET_IDENTITIES = {
    "freebase": "6124093b9acae82ca28fd7b1e262a6e5fe62ebd658890b93e12929e1ef64dc9d",
    "dblp": "ac4dcc0c6bcb735a0646ae75aac1a41ff343758cb4cb01fdea40a3df43ff78c7",
}
DBLP_PUBLIC_ROLE_DATASET_IDENTITY = (
    "30aad38c6d6dcad7220aa85e7cb95b9699ed3d2d97970498750d1a0a3810a84b"
)
CACHE_IDENTITIES = {
    "freebase": "5c7ef95ae2fa68bab116d36aa81599709ee347889bf5da036230a86e04901c1f",
    "dblp": "69b3ff44a9025373e1fecc16f8c59731b8f645d1c02363c1df7cab663863a6b7",
}
PROJECTION_IDENTITIES = {
    "freebase": "cc0c66778a30d75df64765f205d5f28adfaf9f014f4c34e7bea8e4e2ba9799ce",
    
    
    "dblp": "",
}
CANDIDATE_BANK_IDENTITIES = {
    "freebase": "987b24fccd2b0ac276509f3e176723b8216def8ff39a366be06c0d7a9836bdb8",
    "dblp": "fe7f38c77c389eb4ab79b52584904df7da6720d731de6555370bb53c3723b80c",
}
PATH_LIBRARY_IDENTITIES = {
    "freebase": "b927cdc86a58d3213104b1dfec77d7bdd23aadaa33c764bceba4709864e80c8b",
    "dblp": "4bfcb4cf63351c8b7752441c0d2a33827b62281da85f33d27b073886cdf1cc1e",
}

FREEBASE_H8_KMEANS_SEED_IDENTITY = "2ed73366d5833956ae23f734657c45f2eb85e753c2ca7297bea6657d31d776a3"
DBLP_PRELIM_KMEANS_CONTRACT_IDENTITY = "3a7f6af830dd77dbee3455a094dab054a590298f4a07551cdf5241f18b6c044d"
EVALUATION_CONTRACTS = {
    "freebase": {
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_metric_role": "H5_LOCKED_REPORT",
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "H5_LOCKED_REPORT",
        "kmeans_random_state_identity": FREEBASE_H8_KMEANS_SEED_IDENTITY,
        "kmeans_n_init": 10,
        "kmeans_algorithm": "lloyd",
    },
    "dblp": {
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_metric_role": "PRELIM_LOCKED_REPORT",
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "PRELIM_LOCKED_REPORT",
        "kmeans_random_state_identity": DBLP_PRELIM_KMEANS_CONTRACT_IDENTITY,
        "kmeans_n_init": 10,
        "kmeans_algorithm": "lloyd",
    },
}

LOGICAL_STAGES = ("CANARY", "EVALUATION", "LOCKED_EVAL", "FINALIZE")

PARENT_PACKAGES = {
    "FREEBASE_H5": (
        "20260811_073500_metachorus_v53_m4a_h5_global_search_pool_r3_gate_41b6949e.zip",
        "0bcf99aab6ee053c06cb096f46e3a057c9a8ca9ffc071fa0a76c4c0b0d238e67",
    ),
    "FREEBASE_H6": (
        "20260811_2320_metachorus_v53_m4a_h6_third_seed_r3_audit_68e61037.zip",
        "3d2c837ecb9137dc3d6ece61fac46d0e7b26a4577e08b35e67abdb3d2a150c81",
    ),
    "FREEBASE_H7_ACM_SAFETY": (
        "20260812_1915_metachorus_v53_m4a_h7_robust_shortlist_r4_0c98cbef.zip",
        "204ea4f2ab3123644a7448e0d2742417f78b2083898a56e3747465ca5b1d1346",
    ),
    "FREEBASE_H8": (
        "20260813_0703_metachorus_v53_m4a_h8_s_eval_locked_MIN_ce83ac06.zip",
        "896d3a9970ec6897b76d9300f6c4e44fa8747dd7e9ff918a251bd3e694490235",
    ),
    "DBLP_H3": (
        "20260816_0945_metachorus_v53_m7f_h3_aminer_fix_dblp_budgeted_MIN_7b6e17f2.zip",
        "83190385d89c75001106a8fe71de788489cdb3639bcbadc43327308f9c1c1ed1",
    ),
    "DBLP_H4": (
        "20260816_2013_metachorus_v53_m7f_h4_dblp_10seed_confirm_MIN_fac2db4c.zip",
        "e4411afa89416910746f529f521ebe186c7dda8da166cde9905564c84a9b1dbe",
    ),
    "IMDB_SAFETY": (
        "20260814_1900_metachorus_v53_m7f_h1_imdb_trajectory_budget_MIN_d2941bd8.zip",
        "d4720d5c13851ec1af012a845bb432e946a22e349dfb12020ec8892064cf190e",
    ),
    "AMINER_SAFETY": (
        "20260815_0944_metachorus_v53_m7f_h2_aminer_4gpu_full_search_MIN_c438245e.zip",
        "877bae95729c9c2d2debc03d33871c5ce867ddc98fe1b6e7118bc27beccd4e03",
    ),
    "REMOTE_NOT_FOUND": (
        "20260818_1947_metachorus_v53_m8_f3r_remote_output_not_found_MIN_ce7d5981.zip",
        "10964d51f3b9e8539b2a990b3216938880270cb972a75993358211cf38dd1645",
    ),
}

FROZEN_SETS = {
    "freebase": {
        "FULL_FROZEN": {
            "set_id": "fbs_d76f77491499bc567e544ca9",
            "set_hash": "d76f77491499bc567e544ca9e0d088eca75ab5a53df5aecdd8f957bdd8930442",
            "path_ids": (
                "fbp_16118f46f333e3809c6b88d4",
                "fbp_1debb04338dd1dc59fddeddd",
                "fbp_4f9feffc3af2fe4bba58bac9",
            ),
        },
        "MANUAL": {
            "set_id": "fbs_165bb53a4b3f67ae5d3310ea",
            "set_hash": "165bb53a4b3f67ae5d3310ea13397c05c954a51410a3ddcf632569ce74358e13",
            "path_ids": (
                "fbp_1debb04338dd1dc59fddeddd",
                "fbp_3519c22f254da970c0120b14",
                "fbp_38b886ff4b3395533130c873",
            ),
        },
    },
    "dblp": {
        "FULL_FROZEN": {
            "set_id": "fast_dblp_057_81d0a8a9e8b5b789",
            "set_hash": "81d0a8a9e8b5b789451c30ac69b16b8479f548ed72aad426e9f1220a2a1b1c91",
            "path_ids": (
                "author_to_paper::paper_to_author::author_to_paper::paper_to_author",
                "author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
                "author_to_paper::paper_to_author::author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
            ),
        },
        "MANUAL": {
            "set_id": "fast_dblp_010_e7c3ad780037d0be",
            "set_hash": "e7c3ad780037d0be26082d7e7903e6bbff736a9b220bb9795627c00fac45aa06",
            "path_ids": (
                "author_to_paper::paper_to_author",
                "author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
                "author_to_paper::paper_to_term::term_to_paper::paper_to_author",
            ),
        },
    },
}

SELECTION_FIELDS = (
    "dataset", "ablation_variant", "selection_seed_role", "set_id", "set_hash", "path_ids",
    "selection_pool_identity", "manual_safety_used", "random_exploration_used", "Pareto_used",
    "median_objectives_used", "worst_case_objectives_used", "classification_used", "clustering_used",
    "search_metric_source", "status",
)
METRIC_FIELDS = (
    "dataset", "ablation_variant", "set_id", "set_hash", "model_seed", "Micro_F1", "Macro_F1",
    "NMI", "ARI", "F_cls", "F_clu", "F_joint", "assignment_stability", "GPU_UUID",
    "runtime_domain", "training_seconds", "result_source", "evaluated_at_utc",
)

MINIMAL_MEMBER_NAMES = (
    "ABLATION_PARENT_PACKAGE_AUDIT.csv",
    "M8_F3R_RERUN_2G_GPU_ATTESTATION.json",
    "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json",
    "M8_F3R_RERUN_2G_LAUNCH_RECEIPT.json",
    "ABLATION_SELECTIONS_FREEBASE.csv",
    "ABLATION_SELECTIONS_DBLP.csv",
    "ABLATION_SELECTIONS_ALL.csv",
    "ABLATION_SELECTION_COMPONENTS.csv",
    "ABLATION_SELECTION_SEAL.json",
    "ABLATION_EXACT_REUSE_REGISTRY.csv",
    "ABLATION_EVALUATION_MATRIX.csv",
    "ABLATION_EVALUATION_MATRIX_SEAL.json",
    "ABLATION_RESOURCE_FORECAST.json",
    "ABLATION_TWO_GPU_GROUP_SCHEDULE.csv",
    "ABLATION_RUN_REGISTRY_MINIMAL.csv",
    "ABLATION_RESOURCE_SUMMARY.csv",
    "ABLATION_ERROR_SUMMARY.csv",
    "ABLATION_METRICS_BY_SEED.csv",
    "ABLATION_SUMMARY.csv",
    "ABLATION_PAIRED_DELTAS_VS_FULL.csv",
    "ABLATION_PAIRED_DELTAS_VS_MANUAL.csv",
    "SAFETY_GATE_ABLATION_EXISTING_EVIDENCE.csv",
    "PROGRESSIVE_FIDELITY_RETENTION.csv",
    "PROGRESSIVE_FIDELITY_COST.csv",
    "ABLATION_SOURCE_LEDGER.csv",
    "ABLATION_DATA_DICTIONARY.csv",
    "ABLATION_EXECUTION_STATUS.json",
    "ABLATION_EXECUTION_STATUS.md",
    "REMOTE_M8_F3R_RERUN_2G_ARTIFACT_RETENTION_MANIFEST.csv",
    "TEST_RECEIPT_M8_F3R_RERUN_2G.json",
    "TEST_WARNING_REGISTRY_M8_F3R_RERUN_2G.csv",
)
MANIFEST_NAME = "M8_F3R_RERUN_2G_MINIMAL_PACKAGE_MEMBER_MANIFEST.json"
CHECKSUMS_NAME = "M8_F3R_RERUN_2G_MINIMAL_PACKAGE_CHECKSUMS.sha256"
POST_PACKAGE_RECEIPT_NAME = "M8_F3R_RERUN_2G_POST_PACKAGE_COMPLETION_RECEIPT.json"

REQUIRED_TEST_FILES = (
    "test_taskbook_identity.py", "test_stage_started_before_gpu.py", "test_parent_package_hashes.py",
    "test_two_4090_attestation.py", "test_exact_ablation_variants.py", "test_freebase_search_seed_list.py",
    "test_dblp_search_seed_list.py", "test_evaluation_seed_list.py", "test_selection_uses_search_metrics_only.py",
    "test_selection_seal_before_locked_eval.py", "test_single_seed_uses_full_safe_mask.py",
    "test_no_exploration_provenance.py", "test_median_only_formula.py",
    "test_scalar_only_no_pareto_formula.py", "test_classification_only_formula.py",
    "test_clustering_only_formula.py", "test_no_manual_safety_formula.py", "test_unique_set_caps.py",
    "test_exact_reuse_identity.py", "test_matrix_seal.py", "test_same_dataset_seed_group_same_gpu.py",
    "test_new_runs_from_zero.py", "test_training_no_labels.py", "test_freebase_report_role.py",
    "test_dblp_report_role.py", "test_metric_formula.py", "test_summary_aggregation.py",
    "test_paired_delta_alignment.py", "test_existing_safety_evidence_no_new_training.py",
    "test_progressive_cost_no_fabrication.py", "test_no_external_benchmark_runs.py",
    "test_no_paper_outputs.py", "test_no_figure_outputs.py", "test_no_full_backup.py",
    "test_internal_checksums.py", "test_clean_extract.py",
)

TERMINAL_STATUSES = (
    "M8_F3R_RERUN_2G_COMPLETE",
    "M8_F3R_RERUN_2G_PARTIAL",
    "M8_F3R_RERUN_2G_BLOCKED_SOURCE",
    "M8_F3R_RERUN_2G_FAILED_PROTOCOL",
    "PARTIAL_RESOURCE_PAUSE",
)
