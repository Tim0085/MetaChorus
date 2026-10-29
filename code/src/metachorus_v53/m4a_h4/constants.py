"""Frozen scientific identities and budgets for M4A-H4."""

from __future__ import annotations

TASKBOOK_FILENAME = (
    'protocol_m4a-h4.md'
)
TASKBOOK_SHA256 = "74e0626df307072d6078a57b62f53e759aa35d77dfb320c2d5f6a5241035d7ae"
PARENT_PACKAGE_FILENAME = (
    "20260807_1337_metachorus_v53_m4a_h3_tc_lpmso_7a2efb90.zip"
)
PARENT_PACKAGE_SHA256 = "07972b8a101e9aeb72d3ef4abf448cc4cebac7339003ad65a6b0e7c252adb681"
PARENT_SCIENTIFIC_SOURCE_COMMIT = "1b11377a28f795af10da0bac4ddd44c8924caf59"
PARENT_SCIENTIFIC_SOURCE_TREE = "58be274369b748d89535aee58340776b1e2a4182"
PARENT_TERMINAL_STATUS = "TC_LPMSO_DEVELOPMENT_FAIL"

PARENT_SPLIT_MANIFEST_SHA256 = "8f3e37470e272db7f4b693f5fab6cce7ef902815079d3fd19628b228b0b1f3bd"
PARENT_SPLIT_SEAL_SHA256 = "849ed8ce7108708e1544fa46544634b67d28331bad55debe5fb570bea6a267a6"
PARENT_EVALUATOR_CONTRACT_SHA256 = "2168b5355ae2e1ce372e01e3bb7733e032b9a8338f6be375e2c612e26b421512"

SEARCH_MODEL_SEEDS = (1985123271, 508898167)
RESERVED_MODEL_SEED = 1950017805
FUTURE_S_EVAL_MODEL_SEEDS = (
    1803939233,
    1780645700,
    1630573285,
    446298291,
    890281712,
    1939954495,
    1925711465,
    716773178,
    1430108399,
    1364270667,
)
FIDELITY_UPDATES = (300, 1000, 2000, 3000)
CHECKPOINT_SCHEDULE = (100, 300, 1000, 2000, 3000)

CLASSIFICATION_SPLIT_COUNT = 5
CLUSTER_SEARCH_SEED_COUNT = 20
CLUSTER_EVAL_SPLIT_INDEX = 0
CLUSTER_SEED_NAMESPACE = "TC_LPMSO_H4_CLUSTER_SEARCH"
R2_EXPLORATION_NAMESPACE = "TC_LPMSO_H4_R2_EXPLORATION"

CLASSIFIER_CONTRACT = {
    "class": "sklearn.linear_model.LogisticRegression",
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 5000,
    "fit_intercept": True,
    "class_weight": None,
    "feature_preprocessing": "none",
}
KMEANS_CONTRACT = {
    "class": "sklearn.cluster.KMeans",
    "algorithm": "lloyd",
    "n_init": 10,
    "fit_nodes": "all_target_nodes_without_labels",
    "metric_nodes": "SEARCH_PROBE_TRAIN_union_SEARCH_PROBE_VALIDATION",
    "canonical_search_split_index": CLUSTER_EVAL_SPLIT_INDEX,
}
TARGET_NODE_COUNTS = {"acm": 4025, "freebase": 3492}

RELIABILITY_RHO_MIN = 0.30
RELIABILITY_SNR_MIN = 0.50

CONTROLS = {
    "acm": {
        "manual": (
            "set_000__00-01",
            "8291c32669df0184a9ca1af7161850218f19347130b10a52f18e7885db2ff0fd",
        ),
        "current_automatic": (
            "set_002__00-03",
            "fb76bd2a3315421c76757bb1e5a9c215666ffa9e9289839f26e8958af6f84c67",
        ),
    },
    "freebase": {
        "manual": (
            "fbs_165bb53a4b3f67ae5d3310ea",
            "165bb53a4b3f67ae5d3310ea13397c05c954a51410a3ddcf632569ce74358e13",
        ),
        "current_automatic": (
            "fbs_8a6e01ba8154f9df2ddf25ea",
            "8a6e01ba8154f9df2ddf25eac73510459e2aef156f197743309b9d0b43182b8d",
        ),
    },
}

R1_PROMOTION_COUNTS = {"acm": 6, "freebase": 80}
R2_PROMOTION_COUNTS = {"acm": 3, "freebase": 24}
R2_DEFERRED_FREEBASE_COUNT = 30
R2_DEFERRED_EXPLORE_COUNT = 6
R3_PROMOTION_COUNTS = {"acm": 2, "freebase": 8}

RESOURCE_LIMITS = {
    "soft_gpu_hours": 10.0,
    "hard_gpu_hours": 12.0,
    "soft_critical_path_hours": 6.0,
    "hard_critical_path_hours": 7.0,
    "soft_storage_bytes": 15 * 1024**3,
    "hard_storage_bytes": 18 * 1024**3,
    "soft_failed_attempts": 5,
    "hard_failed_attempts": 8,
    "engineering_retries_per_identity": 1,
}

H1_RESULT_RELATIVE = (
    "02_run_results/m4a_h1_lpmso_r1/"
    "20260806_214500_m4a_h1_a22ee8a_opfix1"
)
H2_RESULT_RELATIVE = (
    "02_run_results/m4a_h2_srp_r1/"
    "20260807_034706_m4a_h2_88bddd3"
)
H3_RESULT_RELATIVE = (
    "02_run_results/m4a_h3_tc_lpmso/"
    "20260807_132802_m4a_h3_e2ae696"
)

R1_RECALL_THRESHOLDS = {
    "acm_classification_top3": 3,
    "acm_clustering_top3": 3,
    "acm_joint_top3": 3,
    "freebase_classification_top20": 16,
    "freebase_joint_top20": 14,
    "freebase_joint_top10": 7,
}

R4_PER_SEED_DROP_LIMIT = 0.02
RESERVED_TASK_DROP_LIMIT = 0.01
RESERVED_JOINT_DROP_LIMIT = 0.005
RESERVED_CATASTROPHIC_DROP_LIMIT = 0.02
