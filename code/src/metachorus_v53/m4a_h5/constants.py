"""Frozen identities, thresholds, counts, and budgets for M4A-H5."""

from __future__ import annotations

TASKBOOK_FILENAME = (
    'protocol_m4a-h5.md'
)
TASKBOOK_SHA256 = "d8c7e7cdb98777df8c39c70b5a1e31f288b392d13ce9dd4d3ee5c30b313d4340"

PARENT_PACKAGE_DECLARED_FILENAME = (
    "20260808_012025_metachorus_v53_m4a_h4_reliability_pareto_lpmso_79f71adb(1).zip"
)
PARENT_PACKAGE_CANONICAL_FILENAME = (
    "20260808_012025_metachorus_v53_m4a_h4_reliability_pareto_lpmso_79f71adb.zip"
)
PARENT_PACKAGE_SHA256 = "1e829576f3c56403cfd444c797f8e3e5c26b2e8006b5fb8727fd6770e9c87e60"
PARENT_SOURCE_COMMIT = "79f71adb39cba3f44d41cd39c4375d57dc566f37"
PARENT_SOURCE_TREE = "408a04e1c92a14e79258d2244216844212628233"
PARENT_TERMINAL_STATUS = "TC_LPMSO_H4_DEVELOPMENT_FAIL"
PARENT_FAILURE_CODE = "FREEBASE_R2_CLASSIFICATION_RELIABILITY_FAIL"
PARENT_TASKBOOK_SHA256 = "74e0626df307072d6078a57b62f53e759aa35d77dfb320c2d5f6a5241035d7ae"

SEARCH_MODEL_SEEDS = (1985123271, 508898167)
THIRD_DEVELOPMENT_MODEL_SEED = 1950017805
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
ALL_DEVELOPMENT_MODEL_SEEDS = SEARCH_MODEL_SEEDS + (THIRD_DEVELOPMENT_MODEL_SEED,)

GLOBAL_POOL_NAMESPACE = "TC_LPMSO_H5_GLOBAL_SEARCH_POOL"
FREEBASE_EXPLORATION_NAMESPACE = "TC_LPMSO_H5_FREEBASE_EXPLORATION"
CLASSIFICATION_SPLIT_COUNT = 5
CLUSTER_SEARCH_SEED_COUNT = 20
GLOBAL_SEARCH_POOL_FRACTION = 0.20
WITHIN_POOL_TRAIN_FRACTION = 0.50

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
    "metric_nodes": "entire_GLOBAL_SEARCH_POOL",
}

TARGET_NODE_COUNTS = {"acm": 4025, "freebase": 3492}
EXPECTED_CLASS_COUNTS = {
    "acm": {0: 1061, 1: 1994, 2: 970},
    "freebase": {0: 1327, 1: 618, 2: 1547},
}
EXPECTED_CANDIDATE_COUNTS = {"acm": 10, "freebase": 220}

CLASSIFICATION_MODEL_RHO_MIN = 0.30
CLASSIFICATION_LOO_RHO_MEDIAN_MIN = 0.85
CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN = 0.75
R2_CLASSIFICATION_TRUTH_RHO_MIN = 0.30
R2_CLASSIFICATION_TRUTH_KENDALL_MIN_EXCLUSIVE = 0.0
R3_CLUSTER_MODEL_RHO_MIN = 0.30
R3_ASSIGNMENT_STABILITY_MEDIAN_MIN = 0.90
R3_CLUSTER_TRUTH_RHO_MIN = 0.20
R3_JOINT_TRUTH_RHO_MIN = 0.25
R3_TRUTH_KENDALL_MIN_EXCLUSIVE = 0.0

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

FREEBASE_EXPLOITATION_COUNT = 80
FREEBASE_EXPLORATION_BANDS = ((81, 115), (116, 150), (151, 185), (186, 220))
FREEBASE_SAMPLES_PER_BAND = 10
R3_AUDIT_COUNTS = {"acm": 3, "freebase": 24}
R4_PROMOTION_COUNTS = {"acm": 2, "freebase": 8}

R1_RECALL_THRESHOLDS = {
    "freebase_classification_top20": 16,
    "freebase_joint_top20": 14,
    "freebase_joint_top10": 7,
    "freebase_clustering_top20": 7,
}

FIDELITY_UPDATES = (300, 1000, 2000, 3000)
CHECKPOINT_SCHEDULE = (100, 300, 1000, 2000, 3000)
MAX_R2_IDENTITIES = 264
MAX_R3_IDENTITIES = 264
MAX_THIRD_SEED_RUNS = 31
MAX_R4_CONTINUATIONS = 42
MAX_GPUS = 2

RESOURCE_LIMITS = {
    "soft_gpu_hours": 18.0,
    "hard_gpu_hours": 22.0,
    "soft_critical_path_hours": 10.0,
    "hard_critical_path_hours": 12.0,
    "soft_storage_bytes": 22 * 1024**3,
    "hard_storage_bytes": 25 * 1024**3,
    "soft_failed_attempts": 10,
    "hard_failed_attempts": 14,
    "engineering_retries_per_identity": 1,
}

R4_PER_SEED_DROP_LIMIT = 0.02

H1_RESULT_RELATIVE = (
    "02_run_results/m4a_h1_lpmso_r1/20260806_214500_m4a_h1_a22ee8a_opfix1"
)
H2_RESULT_RELATIVE = "02_run_results/m4a_h2_srp_r1/20260807_034706_m4a_h2_88bddd3"
H4_RESULT_RELATIVE = (
    "02_run_results/m4a_h4_reliability_pareto/20260807_182251_m4a_h4_d76b20c"
)

H5_TERMINAL_STATES = (
    "TC_LPMSO_H5_PASS",
    "TC_LPMSO_H5_NONINFERIOR",
    "TC_LPMSO_H5_SELECTED_MANUAL",
    "TC_LPMSO_H5_DEVELOPMENT_FAIL",
    "PARTIAL_RESOURCE_PAUSE",
)
