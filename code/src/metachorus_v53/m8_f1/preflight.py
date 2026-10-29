"""Append-only local preflight for the M8-F1 final ten-seed completion."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import stat
import statistics
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .common import (
    canonical_json_hash,
    envelope,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    CONFIRMATION_MODEL_SEEDS,
    DATASETS,
    DATASET_SPECS,
    FINAL10_ENDPOINT,
    FINAL_METHOD_ROLES,
    FROZEN_PROTOCOL_RELATIVE_PATH,
    FROZEN_PROTOCOL_SHA256,
    FROZEN_SEED_NAMESPACE_IDENTITY,
    LABEL_ROLE_MANIFEST_IDENTITY,
    LABEL_ROLE_SEAL_IDENTITY,
    METHOD_FREEZE_IDENTITY,
    METHOD_NAME,
    PARENT_FIVE_TABLE_IDENTITY,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TERMINAL_STATUS,
    SEED_ROLE_PUBLIC_ARTIFACTS,
    SUPPORT_PACKAGES,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
    TRAINER_ID,
    VIEW_PIPELINE,
)
from .planning import (
    build_canary_plan,
    build_final10_plan,
    load_cache_map,
    resolve_manual_set,
)


FINAL_METRICS = ("Micro_F1", "Macro_F1", "NMI", "ARI", "F_cls", "F_clu", "F_joint")
FIVE_DATASETS = ("acm", "freebase", "imdb", "dblp", "aminer")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("M8-F1 preflight requires a clean source worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or re.fullmatch(
        r"[0-9a-f]{40}", tree
    ) is None:
        raise ValueError("M8-F1 source commit/tree identity is malformed")
    return commit, tree


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise ValueError("unsafe ZIP member")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("unsafe ZIP member")
    return path


class SealedZip:
    """Read-only exact-identity ZIP view for public MIN packages."""

    def __init__(self, path: str | Path, *, filename: str, digest: str, size: int) -> None:
        raw = Path(path).absolute()
        if raw.is_symlink():
            raise ValueError("governed package path is a symlink")
        value = raw.resolve(strict=True)
        if not value.is_file() or value.stat().st_nlink != 1:
            raise ValueError("governed package is not a plain regular file")
        if (
            value.name != filename
            or value.stat().st_size != int(size)
            or sha256_file(value) != digest
        ):
            raise ValueError("governed package identity mismatch")
        self.path = value
        self.archive = zipfile.ZipFile(value)
        infos = self.archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if (
            self.archive.testzip() is not None
            or len(names) != len(set(names))
            or len({name.casefold() for name in names}) != len(names)
            or any(_safe_member(name) is None for name in names)
            or any(stat.S_ISLNK(info.external_attr >> 16) for info in infos)
        ):
            raise ValueError("governed package ZIP structure is unsafe")
        self.names = set(names)

    def bytes(self, name: str) -> bytes:
        if name not in self.names:
            raise FileNotFoundError(f"required package member is missing: {name}")
        return self.archive.read(name)

    def json(self, name: str) -> dict[str, Any]:
        value = json.loads(self.bytes(name).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"package member is not a JSON object: {name}")
        return value

    def csv(self, name: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.bytes(name).decode("utf-8-sig"))))

    def close(self) -> None:
        self.archive.close()


def _validate_envelope(
    value: Mapping[str, Any], *, status: str | None = None
) -> dict[str, Any]:
    basis = value.get("basis", {})
    if (
        not isinstance(basis, dict)
        or value.get("identity") != canonical_json_hash(basis)
        or (status is not None and value.get("status") != status)
    ):
        raise ValueError("governed JSON envelope identity/status is invalid")
    return dict(basis)


def _copy_public_roles(source_root: Path, result: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    for name in SEED_ROLE_PUBLIC_ARTIFACTS:
        source = source_root / name
        if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
            raise FileNotFoundError(f"missing plain public seed/label artifact: {name}")
        value = read_json(source)
        basis = _validate_envelope(value)
        if (
            name == "FAST_SEED_NAMESPACE_MANIFEST.json"
            and value.get("identity") != FROZEN_SEED_NAMESPACE_IDENTITY
        ):
            raise ValueError("seed namespace identity changed")
        if name == "FAST_LABEL_ROLE_MANIFEST.json" and (
            value.get("identity") != LABEL_ROLE_MANIFEST_IDENTITY
            or basis.get("training_process_mounts_labels") is not False
            or basis.get("raw_labels_embedded") is not False
        ):
            raise ValueError("label role manifest identity/boundary changed")
        if (
            name == "FAST_LABEL_ROLE_SEAL.json"
            and value.get("identity") != LABEL_ROLE_SEAL_IDENTITY
        ):
            raise ValueError("label role seal identity changed")
        if name == "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json" and (
            value.get("status") != "PASS"
            or basis.get("contains_full_labels") is not False
            or basis.get("contains_prelim_locked_report") is not False
        ):
            raise ValueError("private search mount audit boundary changed")
        write_json_once(result / name, value)
        identities[name] = str(value["identity"])
    return identities


def _markdown_table(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    if not rows:
        return f"# {title}\n"
    fields = list(rows[0])
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _metric_row(
    row: Mapping[str, Any], *, dataset: str, evidence_source: str
) -> dict[str, Any]:
    values = {metric: float(row[metric]) for metric in FINAL_METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("public metric evidence contains NaN or infinity")
    return {
        "dataset": dataset,
        "candidate_set_id": str(row.get("candidate_set_id", row.get("set_id", ""))),
        "candidate_set_hash": str(row.get("candidate_set_hash", row.get("set_hash", ""))),
        "model_seed": int(row["model_seed"]),
        **values,
        "evidence_source": evidence_source,
        "status": "PASS",
    }


def _exact_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    count: int,
    predicate: Any,
    evidence_source: str,
) -> list[dict[str, Any]]:
    matches = [row for row in rows if predicate(row)]
    normalized = [
        _metric_row(row, dataset=dataset, evidence_source=evidence_source) for row in matches
    ]
    if (
        len(normalized) != int(count)
        or len({int(row["model_seed"]) for row in normalized}) != int(count)
    ):
        raise ValueError(f"{dataset} public metric evidence is not exact")
    return normalized


def _summary_row(rows: Sequence[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("sample-SD evidence requires at least two rows")
    result: dict[str, Any] = {
        "dataset": dataset,
        "model_seed_count": len(rows),
        "source_rows_identity": canonical_json_hash(list(rows)),
        "status": "PASS",
    }
    for metric in FINAL_METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_sample_sd"] = statistics.stdev(values)
    return result


def _parent_and_support_evidence(
    *, parent: SealedZip, h8: SealedZip, h1: SealedZip, h2: SealedZip
) -> tuple[
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, str, Sequence[Mapping[str, Any]]]],
    dict[str, str],
]:
    complete = parent.json("M7F_H4_COMPLETE.json")
    complete_basis = _validate_envelope(complete, status=PARENT_TERMINAL_STATUS)
    parent_git = parent.json("M7F_H4_GIT_AND_CONFIG_IDENTITY.json")
    git_basis = _validate_envelope(parent_git, status="PASS")
    parent_method = parent.json("M7F_H4_METHOD_FREEZE.json")
    method_basis = _validate_envelope(parent_method, status="PASS")
    parent_rows = parent.csv("FIVE_DATASET_FINAL_STATUS_TABLE.csv")
    if (
        len(parent_rows) != 5
        or tuple(row.get("dataset") for row in parent_rows) != FIVE_DATASETS
        or canonical_json_hash(parent_rows) != PARENT_FIVE_TABLE_IDENTITY
        or complete_basis.get("final_five_dataset_table_identity") != PARENT_FIVE_TABLE_IDENTITY
        or complete_basis.get("all_required_scientific_runs_complete") is not True
        or complete_basis.get("algorithm_retuning") is not False
        or complete_basis.get("additional_model_seeds") is not False
        or git_basis.get("source_commit") != PARENT_SOURCE_COMMIT
        or git_basis.get("source_tree") != PARENT_SOURCE_TREE
        or method_basis.get("method_name") != METHOD_NAME
        or method_basis.get("trainer") != TRAINER_ID
        or method_basis.get("view_pipeline") != VIEW_PIPELINE
        or int(method_basis.get("endpoint_update", -1)) != FINAL10_ENDPOINT
    ):
        raise ValueError("M8-F1 parent scientific identity is invalid")

    h8_rows = h8.csv("TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv")
    acm = _exact_metric_rows(
        h8_rows,
        dataset="acm",
        count=10,
        predicate=lambda row: row.get("dataset") == "acm"
        and row.get("role") == "H7_SELECTED"
        and row.get("status") == "PASS",
        evidence_source="H8:TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv:H7_SELECTED",
    )
    freebase = _exact_metric_rows(
        h8_rows,
        dataset="freebase",
        count=10,
        predicate=lambda row: row.get("dataset") == "freebase"
        and row.get("role") == "H7_SELECTED"
        and row.get("status") == "PASS",
        evidence_source="H8:TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv:H7_SELECTED",
    )
    h4_metrics = parent.csv("DBLP_H4_LOCKED_METRICS_BY_MODEL_SEED.csv")
    dblp = _exact_metric_rows(
        h4_metrics,
        dataset="dblp",
        count=10,
        predicate=lambda row: row.get("dataset") == "dblp"
        and row.get("logical_role") == "DBLP_SELECTED"
        and row.get("status") == "PASS",
        evidence_source="H4:DBLP_H4_LOCKED_METRICS_BY_MODEL_SEED.csv:DBLP_SELECTED",
    )
    parent_stats = [_summary_row(acm, "acm"), _summary_row(freebase, "freebase"), _summary_row(dblp, "dblp")]
    parent_by_dataset = {row["dataset"]: row for row in parent_rows}
    for summary in parent_stats:
        source = parent_by_dataset[str(summary["dataset"])]
        for metric in FINAL_METRICS:
            if abs(float(source[metric]) - float(summary[f"{metric}_mean"])) > 1e-12:
                raise ValueError("parent five-table mean does not match public ten-seed evidence")

    h1_locked = h1.csv("IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv")
    imdb_prior3 = _exact_metric_rows(
        h1_locked,
        dataset="imdb",
        count=3,
        predicate=lambda row: row.get("dataset") == "imdb"
        and row.get("selector") == "FIXED_END_3000"
        and row.get("path_set_role") == "MANUAL"
        and row.get("candidate_set_hash") == DATASET_SPECS["imdb"]["manual_set_hash"]
        and row.get("metric_role") == "PRELIM_LOCKED_REPORT"
        and row.get("status") == "PASS",
        evidence_source="H1:IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv:MANUAL:FIXED_END_3000",
    )
    h1_manual = h1.csv("DBLP_AMINER_MANUAL_METRICS_BY_SEED.csv")
    aminer_prior3 = _exact_metric_rows(
        h1_manual,
        dataset="aminer",
        count=3,
        predicate=lambda row: row.get("dataset") == "aminer"
        and row.get("candidate_set_hash") == DATASET_SPECS["aminer"]["manual_set_hash"]
        and str(row.get("update")) == str(FINAL10_ENDPOINT)
        and row.get("classification_fit_role") == "GLOBAL_SEARCH_POOL"
        and row.get("classification_metric_role") == "PRELIM_LOCKED_REPORT"
        and row.get("status") == "PASS",
        evidence_source="H1:DBLP_AMINER_MANUAL_METRICS_BY_SEED.csv:AMINER_MANUAL",
    )
    prior3 = imdb_prior3 + aminer_prior3

    imdb_rejected_rows = _exact_metric_rows(
        h1_locked,
        dataset="imdb",
        count=3,
        predicate=lambda row: row.get("dataset") == "imdb"
        and row.get("selector") == "FIXED_END_3000"
        and row.get("candidate_set_hash") == DATASET_SPECS["imdb"]["rejected_set_hash"]
        and row.get("metric_role") == "PRELIM_LOCKED_REPORT"
        and row.get("status") == "PASS",
        evidence_source="H1:IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv:REJECTED_FIXED_END_3000",
    )
    h2_prelim = h2.csv("AMINER_H2_PRELIM_METRICS_BY_SEED.csv")
    aminer_rejected_rows = _exact_metric_rows(
        h2_prelim,
        dataset="aminer",
        count=3,
        predicate=lambda row: row.get("dataset") == "aminer"
        and row.get("logical_role") == "FINAL_FROZEN_METHOD"
        and row.get("candidate_set_hash") == DATASET_SPECS["aminer"]["rejected_set_hash"]
        and int(row.get("endpoint_update", -1)) == FINAL10_ENDPOINT
        and row.get("classification_metric_role") == "PRELIM_LOCKED_REPORT"
        and row.get("status") == "PASS",
        evidence_source="H2:AMINER_H2_PRELIM_METRICS_BY_SEED.csv:REJECTED_SELECTED_SET",
    )
    rejected: list[dict[str, Any]] = []
    for dataset, rows in (("imdb", imdb_rejected_rows), ("aminer", aminer_rejected_rows)):
        summary = _summary_row(rows, dataset)
        rejected.append(
            {
                "dataset": dataset,
                "candidate_set_id": DATASET_SPECS[dataset]["rejected_set_id"],
                "candidate_set_hash": DATASET_SPECS[dataset]["rejected_set_hash"],
                "model_seed_count": 3,
                **{metric: summary[f"{metric}_mean"] for metric in FINAL_METRICS},
                "source_rows_identity": summary["source_rows_identity"],
                "evidence_level": "3_SEED_DEVELOPMENT_OR_PRELIMINARY_ABLATION",
                "final_evidence_eligibility": "NOT_10_SEED_FINAL_EVIDENCE",
                "status": "PASS",
            }
        )

    seed_sources: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [
        ("H8", "TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv", h8_rows),
        ("H1", "M7F_H1_RUN_REGISTRY_MINIMAL.csv", h1.csv("M7F_H1_RUN_REGISTRY_MINIMAL.csv")),
        ("H2", "AMINER_H2_RUN_REGISTRY_MINIMAL.csv", h2.csv("AMINER_H2_RUN_REGISTRY_MINIMAL.csv")),
        ("H4", "DBLP_H4_RUN_REGISTRY_MINIMAL.csv", parent.csv("DBLP_H4_RUN_REGISTRY_MINIMAL.csv")),
    ]
    identities = {
        "parent_complete_identity": str(complete["identity"]),
        "parent_git_identity": str(parent_git["identity"]),
        "parent_method_identity": str(parent_method["identity"]),
    }
    return parent_rows, parent_stats, prior3, rejected, seed_sources, identities


def _seed_usage_audit(
    sources: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for ordinal, seed in enumerate(CONFIRMATION_MODEL_SEEDS):
            hits: list[str] = []
            for package_name, member_name, source_rows in sources:
                if any(
                    str(row.get("dataset", "")).casefold() == dataset
                    and str(row.get("model_seed", "")) == str(seed)
                    for row in source_rows
                ):
                    hits.append(f"{package_name}:{member_name}")
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_identity": DATASET_SPECS[dataset]["dataset_identity"],
                    "model_seed_ordinal": ordinal,
                    "model_seed": int(seed),
                    "historical_scientific_use_count": len(hits),
                    "historical_source_members": hits,
                    "same_integer_reuse_across_datasets_allowed": True,
                    "eligible_for_m8_final10": len(hits) == 0,
                    "status": "PASS" if not hits else "BLOCKED_HISTORICAL_USE",
                }
            )
    if len(rows) != 20 or any(
        int(row["historical_scientific_use_count"]) != 0
        or row["eligible_for_m8_final10"] is not True
        for row in rows
    ):
        raise RuntimeError("M8-F1 per-dataset confirmation seed usage audit failed")
    return rows


def _pre_final10_table(parent_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(parent_rows) != 5 or tuple(row.get("dataset") for row in parent_rows) != FIVE_DATASETS:
        raise ValueError("parent five-dataset table is incomplete")
    output: list[dict[str, Any]] = []
    for raw in parent_rows:
        row = dict(raw)
        dataset = str(row["dataset"])
        if dataset in DATASETS:
            row.update(
                {
                    "final_method_role": FINAL_METHOD_ROLES[dataset],
                    "confirmation_evidence": "PRIOR3_DESCRIPTIVE_ONLY_PENDING_FINAL10",
                    "execution_status": "PENDING_M8_F1_FINAL10",
                    "selection_status": "PARENT_ROLE_FROZEN_NO_REOPEN",
                    "final_freeze_status": "PENDING_M8_F1_FINAL10",
                    "claim_boundary": "PENDING_10_SEED_FINAL_FIXED_METHOD_RESULT",
                }
            )
        output.append(row)
    return output


def _path_resolution_rows(caches: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manual = resolve_manual_set(caches[dataset], dataset)
        spec = DATASET_SPECS[dataset]
        rows.append(
            {
                "dataset": dataset,
                "dataset_identity": spec["dataset_identity"],
                "cache_identity": spec["cache_identity"],
                "path_library_identity": spec["path_library_identity"],
                "all_sets_identity": spec["all_sets_identity"],
                "final_method_role": FINAL_METHOD_ROLES[dataset],
                "candidate_set_id": manual["candidate_set_id"],
                "candidate_set_hash": manual["candidate_set_hash"],
                "candidate_set_ordinal": int(manual["candidate_set_ordinal"]),
                "path_ids": list(manual["path_ids"]),
                "set_cardinality": len(manual["path_ids"]),
                "status": "FROZEN_EXACT",
            }
        )
    return rows


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_package: str | Path,
    h8_support_package: str | Path,
    h1_support_package: str | Path,
    h2_support_package: str | Path,
    cache_parent: str | Path,
    seed_role_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    task = Path(taskbook).resolve(strict=True)
    result = Path(result_root).absolute()
    if result.exists():
        raise FileExistsError("M8-F1 preflight requires a fresh result root")
    if task.name != TASKBOOK_FILENAME or sha256_file(task) != TASKBOOK_SHA256:
        raise ValueError("M8-F1 taskbook identity mismatch")
    commit, tree = git_identity(repo)
    protocol_path = repo / FROZEN_PROTOCOL_RELATIVE_PATH
    if protocol_path.is_symlink() or sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("M8-F1 frozen protocol source changed")

    parent = SealedZip(
        parent_package,
        filename=PARENT_PACKAGE_FILENAME,
        digest=PARENT_PACKAGE_SHA256,
        size=PARENT_PACKAGE_SIZE_BYTES,
    )
    support_archives: dict[str, SealedZip] = {}
    support_paths = {
        "h8_acm_freebase": h8_support_package,
        "h1_imdb_aminer_prior3": h1_support_package,
        "h2_aminer_rejected": h2_support_package,
    }
    try:
        for role, path in support_paths.items():
            spec = SUPPORT_PACKAGES[role]
            support_archives[role] = SealedZip(
                path,
                filename=str(spec["filename"]),
                digest=str(spec["sha256"]),
                size=int(spec["size_bytes"]),
            )
        (
            parent_rows,
            parent_stats,
            prior3_rows,
            rejected_rows,
            seed_sources,
            parent_member_identities,
        ) = _parent_and_support_evidence(
            parent=parent,
            h8=support_archives["h8_acm_freebase"],
            h1=support_archives["h1_imdb_aminer_prior3"],
            h2=support_archives["h2_aminer_rejected"],
        )
        caches = load_cache_map(cache_parent)
        path_rows = _path_resolution_rows(caches)
        usage_rows = _seed_usage_audit(seed_sources)
        pre_table = _pre_final10_table(parent_rows)
        canary_rows, canary_seal = build_canary_plan(caches)
        final10_rows, final10_seal = build_final10_plan(caches)
        if len(final10_rows) != 20:
            raise RuntimeError("M8-F1 FINAL10 matrix is not exact")

        result.mkdir(parents=True)
        public_identities = _copy_public_roles(Path(seed_role_root).resolve(strict=True), result)
        parent_identity = envelope(
            "metachorus.v53.m8-f1.parent-identity.v1",
            "PASS",
            {
                "parent_package_filename": PARENT_PACKAGE_FILENAME,
                "parent_package_sha256": PARENT_PACKAGE_SHA256,
                "parent_package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
                "parent_source_commit": PARENT_SOURCE_COMMIT,
                "parent_source_tree": PARENT_SOURCE_TREE,
                "parent_terminal_status": PARENT_TERMINAL_STATUS,
                "parent_five_table_identity": PARENT_FIVE_TABLE_IDENTITY,
                **parent_member_identities,
                "support_packages": SUPPORT_PACKAGES,
                "support_package_roles": {
                    "h8_acm_freebase": "PARENT_10SEED_SAMPLE_SD_ONLY",
                    "h1_imdb_aminer_prior3": "PRIOR3_DESCRIPTIVE_AND_IMDB_REJECTED_ONLY",
                    "h2_aminer_rejected": "AMINER_REJECTED_3SEED_ABLATION_ONLY",
                },
                "prior3_merged_into_final10": False,
                "new_gpu_runs_for_rejected_sets": 0,
            },
        )
        method_freeze = envelope(
            "metachorus.v53.m8-f1.method-freeze.v1",
            "PASS",
            {
                "method_name": METHOD_NAME,
                "trainer": TRAINER_ID,
                "view_pipeline": VIEW_PIPELINE,
                "endpoint_update": FINAL10_ENDPOINT,
                "final_method_roles": FINAL_METHOD_ROLES,
                "manual_set_hashes": {
                    dataset: DATASET_SPECS[dataset]["manual_set_hash"] for dataset in DATASETS
                },
                "algorithm_retuning": False,
                "additional_path_search": False,
                "additional_model_seeds": False,
                "method_freeze_identity": METHOD_FREEZE_IDENTITY,
                "protocol_relative_path": FROZEN_PROTOCOL_RELATIVE_PATH,
                "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "labels_visible_during_training": False,
                "prior3_descriptive_only": True,
                "rejected_sets_existing_evidence_only": True,
            },
        )
        git_config = envelope(
            "metachorus.v53.m8-f1.git-config-identity.v1",
            "PASS",
            {
                "source_commit": commit,
                "source_tree": tree,
                "worktree_clean": True,
                "taskbook_filename": TASKBOOK_FILENAME,
                "taskbook_sha256": TASKBOOK_SHA256,
                "parent_identity": parent_identity["identity"],
                "method_freeze_identity": method_freeze["identity"],
                "dataset_specs": DATASET_SPECS,
                "confirmation_model_seeds": list(CONFIRMATION_MODEL_SEEDS),
                "public_seed_role_identities": public_identities,
                "labels_mounted_during_training": False,
                "algorithm_retuning": False,
                "additional_path_search": False,
            },
        )

        write_json_once(result / "M8_F1_PARENT_IDENTITY.json", parent_identity)
        write_json_once(result / "M8_F1_METHOD_FREEZE.json", method_freeze)
        write_csv_once(result / "M8_F1_PATH_SET_RESOLUTION.csv", path_rows)
        write_csv_once(result / "FINAL10_COMPLETION_SEED_USAGE_AUDIT.csv", usage_rows)
        write_csv_once(result / "FINAL10_COMPLETION_MATRIX.csv", final10_rows)
        write_json_once(result / "FINAL10_COMPLETION_MATRIX_SEAL.json", final10_seal)
        write_csv_once(result / "M8_F1_CANARY_PLAN.csv", canary_rows)
        write_json_once(result / "M8_F1_CANARY_PLAN_SEAL.json", canary_seal)
        write_csv_once(result / "M8_F1_PRIOR3_METRICS_BY_SEED.csv", prior3_rows)
        write_csv_once(result / "M8_F1_REJECTED_SET_EXISTING_METRICS.csv", rejected_rows)
        write_csv_once(result / "M8_F1_PARENT_FINAL_METRIC_STATS.csv", parent_stats)
        write_csv_once(result / "FIVE_DATASET_STATUS_TABLE_PRE_FINAL10.csv", pre_table)
        write_text_once(
            result / "FIVE_DATASET_STATUS_TABLE_PRE_FINAL10.md",
            _markdown_table(pre_table, "Five-Dataset Status Before M8-F1 FINAL10"),
        )
        write_json_once(result / "M8_F1_GIT_AND_CONFIG_IDENTITY.json", git_config)
        complete = envelope(
            "metachorus.v53.m8-f1.preflight-complete.v1",
            "READY_FOR_M8_F1_4GPU_COMPLETION",
            {
                "parent_identity": parent_identity["identity"],
                "method_freeze_identity": method_freeze["identity"],
                "git_and_config_identity": git_config["identity"],
                "canary_plan_seal_identity": canary_seal["identity"],
                "final10_matrix_seal_identity": final10_seal["identity"],
                "canary_row_count": len(canary_rows),
                "final10_row_count": len(final10_rows),
                "model_seed_count_per_dataset": len(CONFIRMATION_MODEL_SEEDS),
                "seed_usage_audit_identity": canonical_json_hash(usage_rows),
                "pre_final10_table_identity": canonical_json_hash(pre_table),
                "pre_final10_table_sha256": sha256_file(
                    result / "FIVE_DATASET_STATUS_TABLE_PRE_FINAL10.csv"
                ),
                "parent_final_metric_stats_identity": canonical_json_hash(parent_stats),
                "parent_final_metric_stats_sha256": sha256_file(
                    result / "M8_F1_PARENT_FINAL_METRIC_STATS.csv"
                ),
                "prior3_evidence_identity": canonical_json_hash(prior3_rows),
                "rejected_set_evidence_identity": canonical_json_hash(rejected_rows),
                "scientific_training_started": False,
                "locked_evaluation_started": False,
                "labels_mounted_during_training": False,
                "full_backup_forbidden": True,
                "minimal_download_only": True,
            },
        )
        write_json_once(result / "M8_F1_PREFLIGHT_COMPLETE.json", complete)
        write_text_once(
            result / "plan.md",
            "# M8-F1 Execution Plan\n\n"
            "1. Run the four sealed 20-update engineering canaries on the fixed GPU mapping.\n"
            "2. If and only if all canaries pass, run the sealed 20-row IMDb/AMiner Manual FINAL10 matrix from update 0 to 3000.\n"
            "3. Seal the training registry before opening labels; then evaluate GLOBAL_SEARCH_POOL/PRELIM_LOCKED_REPORT exactly once per dataset.\n"
            "4. Keep the prior three-seed and rejected-set results descriptive-only; never merge them into the final ten seeds.\n"
            "5. Freeze the five-dataset paper table and publish only the canonical minimal ZIP plus its SHA-256 sidecar.\n",
        )
        return complete
    finally:
        parent.close()
        for archive in support_archives.values():
            archive.close()


__all__ = ["SealedZip", "git_identity", "build_preflight"]
