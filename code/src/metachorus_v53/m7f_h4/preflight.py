"""Append-only local preflight for M7F-H4."""

from __future__ import annotations

import csv
import io
import json
import re
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

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
    ALL_SETS_IDENTITY,
    CACHE_IDENTITY,
    CONFIRMATION_MODEL_SEEDS,
    CURRENT_AUTOMATIC_STATUS,
    DATASET,
    DATASET_IDENTITY,
    FROZEN_PROTOCOL_RELATIVE_PATH,
    FROZEN_PROTOCOL_SHA256,
    H8_BOOTSTRAP_NAMESPACE,
    H8_BOOTSTRAP_RESAMPLES,
    H8_STATISTICS_FUNCTION,
    H8_STATISTICS_RELATIVE_PATH,
    H8_STATISTICS_SHA256,
    HISTORICAL_PRELIMINARY_MODEL_SEEDS,
    HISTORICAL_SEARCH_MODEL_SEEDS,
    IMDB_SOURCE_MEMBER,
    IMDB_SOURCE_PACKAGE_FILENAME,
    IMDB_SOURCE_PACKAGE_SHA256,
    IMDB_SOURCE_PACKAGE_SIZE_BYTES,
    LABEL_ROLE_MANIFEST_IDENTITY,
    LABEL_ROLE_SEAL_IDENTITY,
    MANUAL_PATH_IDS,
    MANUAL_SET_HASH,
    MANUAL_SET_ID,
    METHOD_FREEZE_IDENTITY,
    METHOD_NAME,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TERMINAL_STATUS,
    PATH_LIBRARY_IDENTITY,
    SEED_ROLE_PUBLIC_ARTIFACTS,
    SELECTED_PATH_IDS,
    SELECTED_SET_HASH,
    SELECTED_SET_ID,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
    TRAINER_ID,
    VIEW_PIPELINE,
)
from .planning import (
    build_canary_plan,
    build_confirmation_plan,
    load_dblp_cache,
    resolve_all_sets,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("M7F-H4 preflight requires a clean source worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise ValueError("M7F-H4 source commit/tree identity is malformed")
    return commit, tree


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise ValueError("unsafe ZIP member")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe ZIP member")
    return path


class SealedZip:
    def __init__(self, path: str | Path, *, filename: str, digest: str, size: int) -> None:
        value = Path(path).resolve(strict=True)
        if value.is_symlink() or not value.is_file():
            raise ValueError("governed package is not a regular file")
        if value.name != filename or value.stat().st_size != size or sha256_file(value) != digest:
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


def _validate_envelope(value: Mapping[str, Any], *, status: str | None = None) -> dict[str, Any]:
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
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"missing public seed/label artifact: {name}")
        value = read_json(source)
        basis = _validate_envelope(value)
        if name == "FAST_SEED_NAMESPACE_MANIFEST.json" and value.get("identity") != "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6":
            raise ValueError("seed namespace identity changed")
        if name == "FAST_LABEL_ROLE_MANIFEST.json" and (
            value.get("identity") != LABEL_ROLE_MANIFEST_IDENTITY
            or basis.get("training_process_mounts_labels") is not False
            or basis.get("raw_labels_embedded") is not False
        ):
            raise ValueError("label role manifest identity/boundary changed")
        if name == "FAST_LABEL_ROLE_SEAL.json" and value.get("identity") != LABEL_ROLE_SEAL_IDENTITY:
            raise ValueError("label role seal identity changed")
        destination = result / name
        write_json_once(destination, value)
        identities[name] = value["identity"]
    return identities


def _markdown_table(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"# {title}\n"
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _imdb_manual_row(package: SealedZip) -> dict[str, str]:
    rows = package.csv(IMDB_SOURCE_MEMBER)
    matches = [
        row
        for row in rows
        if row.get("dataset") == "imdb" and row.get("logical_role") == "MANUAL" and row.get("status") == "PASS"
    ]
    if len(matches) != 1 or int(matches[0].get("model_seed_count", 0)) != 3:
        raise ValueError("formal M6F IMDb Manual Fixed-End 3000 source row is missing")
    return matches[0]


def _pre_confirmation_table(parent_rows: list[dict[str, str]], imdb: Mapping[str, str]) -> list[dict[str, Any]]:
    if len(parent_rows) != 5 or {row.get("dataset") for row in parent_rows} != {"acm", "freebase", "imdb", "dblp", "aminer"}:
        raise ValueError("parent five-dataset table is incomplete")
    output: list[dict[str, Any]] = []
    for row in parent_rows:
        value: dict[str, Any] = dict(row)
        dataset = str(value["dataset"])
        if dataset == "dblp":
            value.update(
                {
                    "method_role": "BUDGETED_SELECTED_SET",
                    "execution_status": "DBLP_H3_FULL_SEARCH_COMPLETE",
                    "search_selection_status": "DBLP_BUDGETED_NEW_SET_SELECTED",
                    "selection_status": "DBLP_BUDGETED_NEW_SET_SELECTED",
                    "preliminary_decision": "DBLP_BUDGETED_PRELIMINARY_PROMISING",
                    "final_freeze_status": "PENDING_10SEED_CONFIRMATION",
                    "model_seed_count": 3,
                    "evidence_level": "BUDGET_LIMITED_SEARCH_PLUS_3_SEED_PRELIMINARY",
                    "confirmatory_claim_allowed": False,
                    "status": "PENDING_10SEED_CONFIRMATION",
                }
            )
        elif dataset == "imdb":
            value.update(
                {
                    "method_role": "MANUAL_FIXED_END_3000",
                    "Micro_F1": imdb["Micro_F1_mean"],
                    "Macro_F1": imdb["Macro_F1_mean"],
                    "NMI": imdb["NMI_mean"],
                    "ARI": imdb["ARI_mean"],
                    "F_cls": imdb["F_cls_mean"],
                    "F_clu": imdb["F_clu_mean"],
                    "F_joint": imdb["F_joint_mean"],
                    "model_seed_count": 3,
                    "evidence_level": "3_SEED_PRELIMINARY_MANUAL_FIXED_END_3000",
                    "imdb_main_table_source_package": IMDB_SOURCE_PACKAGE_FILENAME,
                    "imdb_main_table_source_member": IMDB_SOURCE_MEMBER,
                    "trajectory_rerun_replicate_only": True,
                }
            )
        elif dataset == "aminer":
            if (
                value.get("method_role") != "MANUAL"
                or value.get("preliminary_decision") != "AMINER_FAST_UNSAFE"
                or value.get("final_freeze_status") != "UNSAFE_NEW_SET_FALLBACK_MANUAL"
            ):
                raise ValueError("AMiner corrected Manual fallback was not preserved")
        output.append(value)
    return output


def _seed_usage_audit(parent: SealedZip) -> list[dict[str, Any]]:
    registry = parent.csv("M7F_H3_RUN_REGISTRY_MINIMAL.csv")
    pending = parent.csv("DBLP_10SEED_CONFIRMATION_MATRIX_PENDING.csv")
    rows = []
    for ordinal, seed in enumerate(CONFIRMATION_MODEL_SEEDS):
        scientific = [
            row
            for row in registry
            if str(row.get("dataset", "")).casefold() == DATASET
            and str(row.get("model_seed", "")) == str(seed)
        ]
        pending_rows = [row for row in pending if str(row.get("model_seed", "")) == str(seed)]
        rows.append(
            {
                "dataset": DATASET,
                "model_seed_ordinal": ordinal,
                "model_seed": int(seed),
                "historical_scientific_use_count": len(scientific),
                "parent_pending_plan_row_count": len(pending_rows),
                "overlaps_historical_search_seeds": seed in HISTORICAL_SEARCH_MODEL_SEEDS,
                "overlaps_historical_preliminary_seeds": seed in HISTORICAL_PRELIMINARY_MODEL_SEEDS,
                "eligible_for_h4_confirmation": (
                    not scientific
                    and seed not in HISTORICAL_SEARCH_MODEL_SEEDS
                    and seed not in HISTORICAL_PRELIMINARY_MODEL_SEEDS
                ),
                "status": "PASS",
            }
        )
    if any(
        int(row["historical_scientific_use_count"]) != 0
        or row["overlaps_historical_search_seeds"]
        or row["overlaps_historical_preliminary_seeds"]
        or row["eligible_for_h4_confirmation"] is not True
        for row in rows
    ):
        raise RuntimeError("DBLP confirmation seed usage audit failed")
    return rows


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_package: str | Path,
    imdb_source_package: str | Path,
    cache_root: str | Path,
    seed_role_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    task = Path(taskbook).resolve(strict=True)
    result = Path(result_root).absolute()
    if result.exists():
        raise FileExistsError("M7F-H4 preflight requires a fresh result root")
    if task.name != TASKBOOK_FILENAME or sha256_file(task) != TASKBOOK_SHA256:
        raise ValueError("M7F-H4 taskbook identity mismatch")
    commit, tree = git_identity(repo)
    protocol_path = repo / FROZEN_PROTOCOL_RELATIVE_PATH
    statistics_path = repo / H8_STATISTICS_RELATIVE_PATH
    if sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256 or sha256_file(statistics_path) != H8_STATISTICS_SHA256:
        raise ValueError("M7F-H4 frozen protocol/statistics source changed")
    parent = SealedZip(
        parent_package,
        filename=PARENT_PACKAGE_FILENAME,
        digest=PARENT_PACKAGE_SHA256,
        size=PARENT_PACKAGE_SIZE_BYTES,
    )
    imdb_package = SealedZip(
        imdb_source_package,
        filename=IMDB_SOURCE_PACKAGE_FILENAME,
        digest=IMDB_SOURCE_PACKAGE_SHA256,
        size=IMDB_SOURCE_PACKAGE_SIZE_BYTES,
    )
    try:
        parent_complete = parent.json("M7F_H3_COMPLETE.json")
        complete_basis = _validate_envelope(parent_complete, status=PARENT_TERMINAL_STATUS)
        parent_git = parent.json("M7F_H3_GIT_AND_CONFIG_IDENTITY.json")
        git_basis = _validate_envelope(parent_git, status="PASS")
        parent_method = parent.json("M7F_H3_METHOD_FREEZE.json")
        method_basis = _validate_envelope(parent_method, status="PASS")
        selected = parent.json("DBLP_H3_SELECTED_SET.json")
        selected_basis = _validate_envelope(selected, status="DBLP_BUDGETED_NEW_SET_SELECTED")
        selected_set = selected_basis.get("selected_set", {})
        if (
            git_basis.get("source_commit") != PARENT_SOURCE_COMMIT
            or git_basis.get("source_tree") != PARENT_SOURCE_TREE
            or method_basis.get("method_name") != METHOD_NAME
            or method_basis.get("trainer") != TRAINER_ID
            or selected_set.get("candidate_set_id") != SELECTED_SET_ID
            or selected_set.get("candidate_set_hash") != SELECTED_SET_HASH
            or tuple(selected_set.get("path_ids", [])) != SELECTED_PATH_IDS
        ):
            raise ValueError("M7F-H4 parent scientific identity mismatch")
        cache = load_dblp_cache(cache_root)
        path_rows = resolve_all_sets(cache)
        usage_rows = _seed_usage_audit(parent)
        pre_table = _pre_confirmation_table(
            parent.csv("FIVE_DATASET_STATUS_TABLE_AFTER_DBLP.csv"),
            _imdb_manual_row(imdb_package),
        )
        canary_rows, canary_seal = build_canary_plan(cache)
        confirmation_rows, confirmation_seal = build_confirmation_plan(cache)
        result.mkdir(parents=True)
        public_identities = _copy_public_roles(Path(seed_role_root).resolve(strict=True), result)
        parent_identity = envelope(
            "metachorus.v53.m7f-h4.parent-identity.v1",
            "PASS",
            {
                "parent_package_filename": PARENT_PACKAGE_FILENAME,
                "parent_package_sha256": PARENT_PACKAGE_SHA256,
                "parent_package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
                "parent_complete_identity": parent_complete["identity"],
                "parent_source_commit": PARENT_SOURCE_COMMIT,
                "parent_source_tree": PARENT_SOURCE_TREE,
                "parent_terminal_status": PARENT_TERMINAL_STATUS,
                "parent_dblp_selection_identity": selected["identity"],
                "parent_dblp_preliminary_status": "DBLP_BUDGETED_PRELIMINARY_PROMISING",
                "imdb_source_package_filename": IMDB_SOURCE_PACKAGE_FILENAME,
                "imdb_source_package_sha256": IMDB_SOURCE_PACKAGE_SHA256,
                "imdb_source_member": IMDB_SOURCE_MEMBER,
                "aminer_correction_status": "AMINER_FINAL_FALLBACK_CORRECTED",
            },
        )
        method_freeze = envelope(
            "metachorus.v53.m7f-h4.method-freeze.v1",
            "PASS",
            {
                "method_name": METHOD_NAME,
                "trainer": TRAINER_ID,
                "view_pipeline": VIEW_PIPELINE,
                "endpoint_update": 3000,
                "algorithm_retuning": False,
                "further_search": False,
                "additional_model_seeds": False,
                "method_freeze_identity": METHOD_FREEZE_IDENTITY,
                "protocol_relative_path": FROZEN_PROTOCOL_RELATIVE_PATH,
                "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "statistics_relative_path": H8_STATISTICS_RELATIVE_PATH,
                "statistics_sha256": H8_STATISTICS_SHA256,
                "statistics_function": H8_STATISTICS_FUNCTION,
                "bootstrap_namespace": H8_BOOTSTRAP_NAMESPACE,
                "bootstrap_resamples": H8_BOOTSTRAP_RESAMPLES,
                "labels_visible_during_training": False,
            },
        )
        git_identity_value = envelope(
            "metachorus.v53.m7f-h4.git-config-identity.v1",
            "PASS",
            {
                "source_commit": commit,
                "source_tree": tree,
                "worktree_clean": True,
                "taskbook_filename": TASKBOOK_FILENAME,
                "taskbook_sha256": TASKBOOK_SHA256,
                "parent_identity": parent_identity["identity"],
                "method_freeze_identity": method_freeze["identity"],
                "dataset_identity": DATASET_IDENTITY,
                "cache_identity": CACHE_IDENTITY,
                "path_library_identity": PATH_LIBRARY_IDENTITY,
                "all_sets_identity": ALL_SETS_IDENTITY,
                "selected_set_hash": SELECTED_SET_HASH,
                "manual_set_hash": MANUAL_SET_HASH,
                "confirmation_model_seeds": list(CONFIRMATION_MODEL_SEEDS),
                "public_seed_role_identities": public_identities,
                "current_automatic_status": CURRENT_AUTOMATIC_STATUS,
                "labels_mounted_during_training": False,
                "algorithm_retuning": False,
            },
        )
        write_json_once(result / "M7F_H4_PARENT_IDENTITY.json", parent_identity)
        write_json_once(result / "M7F_H4_METHOD_FREEZE.json", method_freeze)
        write_csv_once(result / "M7F_H4_DBLP_PATH_SET_RESOLUTION.csv", path_rows)
        write_csv_once(result / "FIVE_DATASET_STATUS_TABLE_PRE_DBLP_CONFIRMATION.csv", pre_table)
        write_text_once(
            result / "FIVE_DATASET_STATUS_TABLE_PRE_DBLP_CONFIRMATION.md",
            _markdown_table(pre_table, "Five-Dataset Status Before DBLP H4 Confirmation"),
        )
        write_csv_once(result / "DBLP_H4_CONFIRMATION_SEED_USAGE_AUDIT.csv", usage_rows)
        write_csv_once(result / "DBLP_H4_CANARY_PLAN.csv", canary_rows)
        write_json_once(result / "DBLP_H4_CANARY_PLAN_SEAL.json", canary_seal)
        write_csv_once(result / "DBLP_H4_CONFIRMATION_MATRIX.csv", confirmation_rows)
        write_json_once(result / "DBLP_H4_CONFIRMATION_MATRIX_SEAL.json", confirmation_seal)
        write_json_once(result / "M7F_H4_GIT_AND_CONFIG_IDENTITY.json", git_identity_value)
        complete = envelope(
            "metachorus.v53.m7f-h4.preflight-complete.v1",
            "READY_FOR_M7F_H4_4GPU_CONFIRMATION",
            {
                "parent_identity": parent_identity["identity"],
                "method_freeze_identity": method_freeze["identity"],
                "git_and_config_identity": git_identity_value["identity"],
                "canary_plan_seal_identity": canary_seal["identity"],
                "confirmation_matrix_seal_identity": confirmation_seal["identity"],
                "confirmation_row_count": len(confirmation_rows),
                "confirmation_physical_run_count": len(confirmation_rows),
                "confirmation_seed_count": len(CONFIRMATION_MODEL_SEEDS),
                "seed_usage_audit_identity": canonical_json_hash(usage_rows),
                "pre_confirmation_table_identity": canonical_json_hash(pre_table),
                "scientific_training_started": False,
                "locked_evaluation_started": False,
                "labels_mounted_during_training": False,
                "full_backup_forbidden": True,
                "minimal_download_only": True,
            },
        )
        write_json_once(result / "M7F_H4_PREFLIGHT_COMPLETE.json", complete)
        write_text_once(
            result / "plan.md",
            "# M7F-H4 Execution Plan\n\n"
            "1. Run four engineering canaries.\n"
            "2. Run the sealed 20-row paired confirmation matrix from step 0 to 3000.\n"
            "3. Seal the training registry, open DBLP labels once, and evaluate the locked protocol.\n"
            "4. Reuse the frozen H8 paired interval implementation, freeze DBLP and the five-dataset table.\n"
            "5. Publish only the canonical minimal ZIP and matching SHA-256 sidecar.\n",
        )
        return complete
    finally:
        parent.close()
        imdb_package.close()


__all__ = ["git_identity", "build_preflight"]
