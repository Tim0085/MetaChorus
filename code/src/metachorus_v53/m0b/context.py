"""Resolved, fail-closed local context for one append-only M0-B run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    utc_now,
)


M0B_TASKBOOK_SHA256 = (
    "86727239dbf680fce0db8d25d8a73526934eeed6fd145970b018beafcbf670cd"
)
PARENT_TASKBOOK_SHA256 = (
    "03eae48c52139754b3421f2d92fa13a1e0e732636b3be9ee186658d0c235442b"
)
M0A_TASKBOOK_SHA256 = (
    "3fb191ff83eb831ebbe5c31ef2e5a18df650ce9d09cbd497ce197c40b6338c06"
)
M0A_COMMIT = "3fe7f64a37407d3cc92c5d873e78e2a0f3ced270"
M0A_TREE = "9392d540fa97700d87877078b6f6a09b949cf28a"
SCIENCE_COMMIT = "b36181329995f731c640226801daa284469d858c"
SCIENCE_TREE = "4745b2fd0a88c6f3ac83f4ad34beb15d2b7b8ea4"
RUNTIME_COMPATIBILITY_HASH = (
    "0b77e895550e48c1fcd579bef25ebe846dc692059dc681068cce6c7c8365484b"
)
ACM_DATA_IDENTITY = (
    "340090de86090bfd524dbb5ab6024e25d9e84165098d679961faf56f96fbd9fa"
)
ACM_PROJECTION_IDENTITY = (
    "a74880f6993ed06cef472b3a8561e2f29ff32d77919c4bbbd716040a25a55774"
)
ACM_CANDIDATE_SPACE_IDENTITY = (
    "1fd801d5890401a952ec8b9525eef000d8c4c1db833dc826dd3d484cbc0bf31c"
)
ACM_PROCESSED_SHA256 = (
    "e14bdb9b73f6baaf1e152899c691dbe8df4bdf54b392a108941e62001dffa143"
)
ACM_RAW_SHA256 = (
    "0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d"
)
ACM_PROJECTION_FILE_SHA256 = (
    "6fa2d5fcc5e24a17a45fc109acb13336e8dd54e54e13bbf18e1f22b8637b8611"
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _find_single_m0a_run(project_root: Path) -> Path:
    candidates = sorted(
        project_root.glob(
            "01_source_worktrees/v53_m0a_*/artifacts/v53/m0a/*/M0A_COMPLETE.json"
        )
    )
    valid: list[Path] = []
    for marker in candidates:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("status") == "COMPLETE":
            valid.append(marker.parent)
    if len(valid) != 1:
        raise RuntimeError(
            f"expected exactly one completed M0-A run, found {len(valid)}: {valid}"
        )
    return valid[0].resolve()


@dataclass(frozen=True)
class M0BContext:
    repo_root: Path
    project_root: Path
    research_root: Path
    output_root: Path
    run_id: str
    m0a_root: Path
    data_source_root: Path
    acm_projection_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        repo_root: str | Path,
        output_root: str | Path,
        run_id: str,
        m0a_root: str | Path | None = None,
    ) -> "M0BContext":
        repo = Path(repo_root).resolve()
        project = repo.parents[1]
        research = project.parent
        output = Path(output_root).resolve()
        expected_parent = (repo / "artifacts" / "v53" / "m0b").resolve()
        if expected_parent not in output.parents:
            raise ValueError(f"M0-B output must be below {expected_parent}: {output}")
        resolved_m0a = (
            Path(m0a_root).resolve()
            if m0a_root is not None
            else _find_single_m0a_run(project)
        )
        return cls(
            repo_root=repo,
            project_root=project,
            research_root=research,
            output_root=output,
            run_id=run_id,
            m0a_root=resolved_m0a,
            data_source_root=repo,
            acm_projection_root=(
                project
                / "02_run_results"
                / "gate16_proxy_r1_8x4090_20260721"
                / "immutable_inputs"
                / "acm_projection_e14bdb9b73f6"
            ).resolve(),
        )

    @property
    def taskbook(self) -> Path:
        return (
            self.research_root
            / 'protocols'
            / 'protocol_m0-b.md'
        )

    @property
    def parent_taskbook(self) -> Path:
        return (
            self.research_root
            / "improve"
            / 'protocol_overview.md'
        )

    @property
    def m0a_taskbook(self) -> Path:
        return (
            self.research_root
            / 'protocols'
            / 'protocol_m0-a.md'
        )

    @property
    def acm_processed(self) -> Path:
        return self.data_source_root / "data" / "acm" / "processed" / "data.pt"

    @property
    def acm_raw(self) -> Path:
        return self.data_source_root / "data" / "acm" / "raw" / "ACM.mat"

    @property
    def acm_projection(self) -> Path:
        return self.acm_projection_root / "projection.pt"

    @property
    def acm_projection_manifest(self) -> Path:
        return self.acm_projection_root / "manifest.json"

    def candidate_space_csv(self) -> Path:
        return self.m0a_root / "registries" / "ACM_CANDIDATE_SPACE.csv"

    def preflight(self) -> dict[str, Any]:
        required = {
            "m0b_taskbook": self.taskbook,
            "parent_taskbook": self.parent_taskbook,
            "m0a_taskbook": self.m0a_taskbook,
            "m0a_complete": self.m0a_root / "M0A_COMPLETE.json",
            "m0a_validation": self.m0a_root / "manifests" / "M0A_VALIDATION.json",
            "acm_processed": self.acm_processed,
            "acm_raw": self.acm_raw,
            "acm_projection": self.acm_projection,
            "acm_projection_manifest": self.acm_projection_manifest,
            "acm_candidate_space": self.candidate_space_csv(),
        }
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for logical_name, path in required.items():
            exists = path.is_file()
            if not exists:
                missing.append(logical_name)
            rows.append(
                {
                    "logical_name": logical_name,
                    "absolute_path_local_only": str(path),
                    "exists": exists,
                    "sha256": sha256_file(path) if exists else None,
                    "size_bytes": path.stat().st_size if exists else None,
                }
            )
        expected = {
            "m0b_taskbook": M0B_TASKBOOK_SHA256,
            "parent_taskbook": PARENT_TASKBOOK_SHA256,
            "m0a_taskbook": M0A_TASKBOOK_SHA256,
            "acm_processed": ACM_PROCESSED_SHA256,
            "acm_raw": ACM_RAW_SHA256,
            "acm_projection": ACM_PROJECTION_FILE_SHA256,
        }
        observed = {row["logical_name"]: row["sha256"] for row in rows}
        mismatches = [
            {"logical_name": name, "expected": digest, "observed": observed.get(name)}
            for name, digest in expected.items()
            if observed.get(name) != digest
        ]
        current_head = _git(self.repo_root, "rev-parse", "HEAD")
        merge_base = _git(self.repo_root, "merge-base", "HEAD", M0A_COMMIT)
        base_tree = _git(self.repo_root, "rev-parse", f"{M0A_COMMIT}^{{tree}}")
        parent_match = merge_base == M0A_COMMIT and base_tree == M0A_TREE
        return {
            "schema_version": "metachorus.v53.m0b.preflight.v1",
            "generated_at_utc": utc_now(),
            "run_id": self.run_id,
            "required_inputs": rows,
            "missing": missing,
            "hash_mismatches": mismatches,
            "engineering_parent": {
                "expected_commit": M0A_COMMIT,
                "observed_merge_base": merge_base,
                "expected_tree": M0A_TREE,
                "observed_tree": base_tree,
                "current_head": current_head,
                "match": parent_match,
            },
            "passed": not missing and not mismatches and parent_match,
        }

    def initialize(self) -> dict[str, Any]:
        if self.output_root.exists():
            raise FileExistsError(f"append-only M0-B run already exists: {self.output_root}")
        for name in (
            "parent_closure",
            "schema",
            "trusted_export",
            "sanitized_cache_local",
            "label_cache_local",
            "equivalence",
            "isolation",
            "adapters",
            "tests",
            "reports",
            "failures",
            "logs",
        ):
            (self.output_root / name).mkdir(parents=True, exist_ok=False)
        preflight = self.preflight()
        if not preflight["passed"]:
            raise RuntimeError(f"M0-B preflight failed: {preflight}")
        payload = {
            "schema_version": "metachorus.v53.m0b.resolved-paths.v1",
            "generated_at_utc": utc_now(),
            "run_id": self.run_id,
            "taskbook_sha256": M0B_TASKBOOK_SHA256,
            "parent_taskbook_sha256": PARENT_TASKBOOK_SHA256,
            "m0a_taskbook_sha256": M0A_TASKBOOK_SHA256,
            "roots": {
                "repo_root": {"absolute_path_local_only": str(self.repo_root)},
                "project_root": {"absolute_path_local_only": str(self.project_root)},
                "research_root": {"absolute_path_local_only": str(self.research_root)},
                "output_root": {"absolute_path_local_only": str(self.output_root)},
                "m0a_root": {"absolute_path_local_only": str(self.m0a_root)},
                "data_source_root": {"absolute_path_local_only": str(self.data_source_root)},
                "acm_projection_root": {
                    "absolute_path_local_only": str(self.acm_projection_root)
                },
            },
            "preflight": preflight,
            "identity_basis_hash": canonical_json_hash(
                {
                    "m0b_taskbook_sha256": M0B_TASKBOOK_SHA256,
                    "m0a_commit": M0A_COMMIT,
                    "m0a_tree": M0A_TREE,
                    "acm_data_identity": ACM_DATA_IDENTITY,
                    "acm_projection_identity": ACM_PROJECTION_IDENTITY,
                    "acm_candidate_space_identity": ACM_CANDIDATE_SPACE_IDENTITY,
                }
            ),
        }
        atomic_write_json(self.output_root / "M0B_RESOLVED_PATHS.json", payload)
        return payload
