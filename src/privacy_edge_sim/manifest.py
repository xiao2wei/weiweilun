"""Reproducibility manifests and conservative output-directory handling.

The manifest deliberately separates the deterministic simulation digest from
engineering diagnostics such as controller wall-clock measurements and host
paths.  A run can therefore be repeated in a different output directory while
retaining the same ``core_digest``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import SimulationConfig
from .evidence import EvidenceVerification, evidence_manifest_record
from .enums import EventKind
from .events import priority_for
from .profiles import FrozenProfileBundle, canonical_json_bytes, thaw_json
from .state import SimulationState


MANIFEST_SCHEMA_VERSION = "1.1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATED_SOURCE_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_CONFIG_PATH_IDENTITY_KEYS = frozenset(
    {"profile_path", "trace_path", "scenario_trace_path", "evidence_path"}
)
KNOWN_OUTPUT_FILES = frozenset(
    {
        "tasks.csv",
        "tasks.parquet",
        "events.jsonl",
        "actions.jsonl",
        "resources.csv",
        "virtual_queues.csv",
        "summary.json",
        "manifest.json",
        "failure.json",
    }
)


def _plain(value: Any) -> Any:
    """Convert dataclasses and immutable containers to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_plain(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 checksum for one file."""

    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_text_bytes(path: Path) -> bytes:
    """Normalize platform newlines only for the portable source identity.

    Raw checksums remain the byte-level evidence.  Undecodable or NUL-bearing
    files are treated as binary and are never transformed.
    """

    return _portable_content_bytes(path.read_bytes())


def _portable_content_bytes(raw: bytes) -> bytes:
    """Normalize line endings for UTF-8 content already held in memory."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    if "\x00" in text:
        return raw
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _head_source_tree_digest(
    *, top: Path, root: Path, relative: str
) -> tuple[str, int]:
    """Hash committed source bytes without trusting index stat shortcuts.

    ``git status`` and ``git diff`` may honor assume-unchanged/skip-worktree.
    Reading a HEAD archive bypasses those flags and lets us compare the actual
    portable working-tree digest with committed blobs and paths.
    """

    completed = subprocess.run(
        ["git", "-C", str(top), "archive", "--format=tar", "HEAD", "--", relative],
        check=True,
        capture_output=True,
        timeout=10,
    )
    prefix = "" if relative == "." else relative.rstrip("/") + "/"
    rows: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if root.is_file():
                source_relative = root.name
            elif prefix and name.startswith(prefix):
                source_relative = name[len(prefix) :]
            else:
                source_relative = name
            parts = set(Path(source_relative).parts)
            if (
                parts & _GENERATED_SOURCE_PARTS
                or Path(source_relative).suffix in {".pyc", ".pyo"}
            ):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"unable to read committed source: {name}")
            rows.append((source_relative, _portable_content_bytes(extracted.read())))
    digest = hashlib.sha256()
    for source_relative, content in sorted(rows):
        encoded = source_relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest(), len(rows)


def _eligible_tree_file(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or parts & _GENERATED_SOURCE_PARTS:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def _source_tree_digest(
    source_root: str | Path | None,
    *,
    portable_text: bool,
) -> tuple[str, int]:
    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    if not root.exists():
        raise FileNotFoundError(f"source root does not exist: {root}")
    paths = (
        [root]
        if root.is_file()
        else [p for p in root.rglob("*") if _eligible_tree_file(p)]
    )
    paths = sorted(
        paths,
        key=lambda p: p.relative_to(root.parent if root.is_file() else root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        file_bytes = _portable_text_bytes(path) if portable_text else path.read_bytes()
        digest.update(hashlib.sha256(file_bytes).digest())
    return digest.hexdigest(), len(paths)


def source_tree_sha256(source_root: str | Path | None = None) -> tuple[str, int]:
    """Hash source paths and raw bytes in stable lexical order.

    By default only the installed package source tree is hashed.  Generated
    experiment outputs can therefore never change the code identity.
    """

    return _source_tree_digest(source_root, portable_text=False)


def source_tree_portable_sha256(
    source_root: str | Path | None = None,
) -> tuple[str, int]:
    """Hash source with UTF-8 text newlines normalized to LF."""

    return _source_tree_digest(source_root, portable_text=True)


def _is_generated_status_path(path_text: str) -> bool:
    normalized = path_text.strip().strip('"').replace("\\", "/")
    parts = set(part for part in normalized.split("/") if part)
    return bool(parts & _GENERATED_SOURCE_PARTS) or normalized.endswith(
        (".pyc", ".pyo")
    )


def _is_generated_status_line(line: str) -> bool:
    path_text = line[3:] if len(line) >= 3 else line
    renamed_paths = path_text.split(" -> ")
    return bool(renamed_paths) and all(
        _is_generated_status_path(item) for item in renamed_paths
    )


def _git_output(arguments: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3,
    ).stdout.strip()


def detect_code_version(source_root: str | Path | None = None) -> dict[str, Any]:
    """Report Git identity plus raw and portable package-source checksums.

    ``git_dirty`` is retained as a whole-repository diagnostic.  Publication
    reproducibility uses ``source_git_dirty``, scoped to the executable source
    tree.  Outputs created elsewhere in the repository therefore cannot make
    the second run of a batch spuriously dirty.
    """

    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    tree_hash, file_count = source_tree_sha256(root)
    portable_tree_hash, portable_file_count = source_tree_portable_sha256(root)
    if portable_file_count != file_count:
        raise RuntimeError("raw and portable source hash file counts disagree")
    result: dict[str, Any] = {
        "kind": "source_tree_sha256",
        "value": tree_hash,
        "source_tree_sha256": tree_hash,
        "source_tree_sha256_semantics": "raw_working_tree_bytes",
        "source_tree_portable_sha256": portable_tree_hash,
        "source_tree_portable_sha256_semantics": (
            "utf8_text_newlines_normalized_to_lf"
        ),
        "source_file_count": file_count,
        "source_git_dirty": None,
        "source_commit_reproducible": False,
        "source_git_status": [],
    }
    try:
        probe = root if root.is_dir() else root.parent
        top = Path(_git_output(["rev-parse", "--show-toplevel"], cwd=probe)).resolve()
        commit = _git_output(["rev-parse", "HEAD"], cwd=top)
        repository_status = _git_output(
            ["status", "--porcelain=v1", "--untracked-files=all"], cwd=top
        )
        relative = root.relative_to(top).as_posix()
        source_status_raw = _git_output(
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative,
            ],
            cwd=top,
        )
        source_status_all = [
            line for line in source_status_raw.splitlines() if line.strip()
        ]
        source_status = [
            line for line in source_status_all if not _is_generated_status_line(line)
        ]
        ignored_generated_count = len(source_status_all) - len(source_status)
        source_object = _git_output(
            ["rev-parse", f"HEAD:{relative}" if relative != "." else "HEAD^{tree}"],
            cwd=top,
        )
        head_portable_hash, head_file_count = _head_source_tree_digest(
            top=top, root=root, relative=relative
        )
        index_rows = _git_output(
            ["ls-files", "-v", "--", relative], cwd=top
        ).splitlines()
        hidden_index_flags = sorted(
            row for row in index_rows if row and (row[0].islower() or row[0] == "S")
        )
    except (FileNotFoundError, subprocess.SubprocessError, tarfile.TarError, OSError):
        return result
    if head_portable_hash != portable_tree_hash or head_file_count != file_count:
        source_status.append("!! working source differs from committed HEAD archive")
    if hidden_index_flags:
        source_status.append("!! assume-unchanged or skip-worktree source entries")
    result.update(
        {
            "kind": "git",
            "value": commit if not source_status else tree_hash,
            "value_semantics": (
                "git_commit_when_source_clean_else_raw_source_tree_sha256"
            ),
            "git_commit": commit,
            "git_dirty": bool(repository_status),
            "source_scope": relative,
            "source_git_object": source_object,
            "source_head_tree_portable_sha256": head_portable_hash,
            "source_head_file_count": head_file_count,
            "source_head_matches_working_tree": (
                head_portable_hash == portable_tree_hash
                and head_file_count == file_count
            ),
            "source_hidden_index_flags": hidden_index_flags,
            "source_git_dirty": bool(source_status),
            "source_commit_reproducible": not source_status,
            "source_git_status": source_status,
            "ignored_generated_source_status_count": ignored_generated_count,
        }
    )
    return result


def _preflight_record(
    code_version: Mapping[str, Any], *, require_clean: bool
) -> dict[str, Any]:
    git_available = code_version.get("kind") == "git"
    source_dirty = code_version.get("source_git_dirty")
    reproducible = bool(code_version.get("source_commit_reproducible"))
    if not git_available:
        assessment = "git_unavailable"
        reason_codes = ["GIT_SOURCE_IDENTITY_UNAVAILABLE"]
    elif source_dirty:
        assessment = "source_dirty"
        reason_codes = ["SOURCE_TREE_DIRTY"]
    else:
        assessment = "source_clean"
        reason_codes = []
    requirement_status = (
        "not_required"
        if not require_clean
        else ("passed" if reproducible else "failed")
    )
    return {
        "require_clean_source": require_clean,
        "requirement_status": requirement_status,
        "assessment": assessment,
        "git_available": git_available,
        "git_commit": code_version.get("git_commit"),
        "source_git_dirty": source_dirty,
        "source_commit_reproducible": reproducible,
        "source_scope": code_version.get("source_scope"),
        "source_git_object": code_version.get("source_git_object"),
        "source_head_tree_portable_sha256": code_version.get(
            "source_head_tree_portable_sha256"
        ),
        "source_head_file_count": code_version.get("source_head_file_count"),
        "source_head_matches_working_tree": code_version.get(
            "source_head_matches_working_tree"
        ),
        "source_hidden_index_flags": list(
            code_version.get("source_hidden_index_flags", [])
        ),
        "source_tree_sha256": code_version.get("source_tree_sha256"),
        "source_tree_portable_sha256": code_version.get(
            "source_tree_portable_sha256"
        ),
        "reason_codes": reason_codes,
        "source_git_status": list(code_version.get("source_git_status", [])),
    }


def source_cleanliness_preflight(
    source_root: str | Path | None = None, *, require_clean: bool = False
) -> dict[str, Any]:
    """Assess or require executable source that exactly matches a Git commit.

    A publication command should call this before simulation with
    ``require_clean=True``.  ``build_manifest`` repeats the gate so a mutation
    during the simulation is also rejected.  Development and smoke runs retain
    the audit record without enforcing it.
    """

    if not isinstance(require_clean, bool):
        raise TypeError("require_clean must be a bool")
    record = _preflight_record(
        detect_code_version(source_root), require_clean=require_clean
    )
    if require_clean and record["requirement_status"] != "passed":
        raise RuntimeError(
            "clean committed source is required for this run: "
            f"assessment={record['assessment']}, "
            f"reason_codes={record['reason_codes']}, "
            f"source_git_status={record['source_git_status']}"
        )
    return record


def prepare_output_directory(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an empty output directory and reject accidental result mixing.

    Explicit overwrite is intentionally conservative: only files generated by
    this package may be removed.  Unknown files or subdirectories still cause
    refusal rather than destructive cleanup.
    """

    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if not existing:
        return output
    if not overwrite:
        names = sorted(item.name for item in existing)
        raise FileExistsError(
            f"output directory is not empty: {output}; existing={names}"
        )
    unsafe = [
        item
        for item in existing
        if item.is_dir() or item.name not in KNOWN_OUTPUT_FILES
    ]
    if unsafe:
        names = sorted(item.name for item in unsafe)
        raise FileExistsError(
            f"refusing to overwrite unknown output entries in {output}: {names}"
        )
    for item in existing:
        item.unlink()
    return output


def _path_checksum_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"trace input does not exist: {resolved}")
    if resolved.is_file():
        files = {resolved.name: sha256_file(resolved)}
        kind = "file"
    else:
        children = sorted(
            (p for p in resolved.rglob("*") if _eligible_tree_file(p)),
            key=lambda p: p.relative_to(resolved).as_posix(),
        )
        files = {p.relative_to(resolved).as_posix(): sha256_file(p) for p in children}
        kind = "directory"
    aggregate = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    return {
        "path": str(resolved),
        "kind": kind,
        "aggregate_sha256": aggregate,
        "files": files,
    }


def _semantic_configuration_value(
    value: Any,
    *,
    path_identities: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Replace runtime locations with role-bound content identities."""

    if isinstance(value, dict):
        return {
            key: (
                _plain(path_identities[key])
                if key in _CONFIG_PATH_IDENTITY_KEYS and key in path_identities
                else _semantic_configuration_value(
                    item, path_identities=path_identities
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _semantic_configuration_value(item, path_identities=path_identities)
            for item in value
        ]
    return value


def _configuration_semantic_identity(
    config_plain: Any,
    *,
    profile_hash: str,
    trace_checksums: Sequence[Mapping[str, Any]],
    evidence_record: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    def content_sha(record: Mapping[str, Any] | None) -> Any:
        if record is None:
            return None
        files = record.get("files")
        if record.get("kind") == "file" and isinstance(files, Mapping):
            values = list(files.values())
            if len(values) == 1:
                return values[0]
        return record.get("aggregate_sha256")

    evidence_sha = evidence_record.get("file_sha256")
    identities: dict[str, dict[str, Any]] = {
        "profile_path": {
            "role": "frozen_profile",
            "content_sha256": profile_hash,
        },
        "trace_path": {
            "role": "evaluation_trace",
            "content_sha256": content_sha(
                trace_checksums[0] if trace_checksums else None
            ),
        },
        "scenario_trace_path": {
            "role": "scenario_trace",
            "content_sha256": content_sha(
                trace_checksums[1] if len(trace_checksums) > 1 else None
            ),
        },
        "evidence_path": {
            "role": "frozen_evidence",
            "content_sha256": evidence_sha,
        },
    }
    semantic = _semantic_configuration_value(
        config_plain, path_identities=identities
    )
    return hashlib.sha256(canonical_json_bytes(semantic)).hexdigest(), identities


_NON_CORE_EXACT = frozenset(
    {
        "generated_at_utc",
        "output_dir",
        "output_path",
        "artifact_paths",
        "controller_diagnostics",
        "diagnostics",
        "wall_clock_s",
        "wall_time_s",
        "controller_wall_time_s",
        "controller_runtime_s",
        "controller_elapsed_s",
    }
)


def _is_non_core_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _NON_CORE_EXACT:
        return True
    if lowered.startswith("output_") or lowered.endswith("_output_path"):
        return True
    return any(
        marker in lowered
        for marker in (
            "wall_clock",
            "wallclock",
            "wall_time",
            "elapsed_wall",
            "controller_runtime",
            "controller_diagnostic",
        )
    )


def deterministic_core_value(value: Any) -> Any:
    """Strip engineering-only fields without removing simulated time fields."""

    plain = _plain(value)
    if isinstance(plain, dict):
        return {
            key: deterministic_core_value(item)
            for key, item in plain.items()
            if not _is_non_core_key(key)
        }
    if isinstance(plain, list):
        return [deterministic_core_value(item) for item in plain]
    return plain


def canonical_core_digest(value: Any) -> str:
    """Hash canonical deterministic content only."""

    return hashlib.sha256(
        canonical_json_bytes(deterministic_core_value(value))
    ).hexdigest()


def _dependency_versions(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    names = ["privacy-edge-sim", "pyarrow"]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    if extra:
        result.update({str(key): str(value) for key, value in extra.items()})
    return dict(sorted(result.items()))


def _task_workload(state: SimulationState) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = sorted(
        state.tasks.values(), key=lambda item: (item.arrival_time_s, item.task_id)
    )
    arrivals = [task.arrival_time_s for task in tasks]
    deadlines = [task.relative_deadline_s for task in tasks]
    by_vehicle: dict[str, int] = {}
    for task in tasks:
        by_vehicle[task.vehicle_id] = by_vehicle.get(task.vehicle_id, 0) + 1
    span = (max(arrivals) - min(arrivals)) if len(arrivals) > 1 else 0.0
    workload = {
        "task_count": len(tasks),
        "tasks_by_vehicle": dict(sorted(by_vehicle.items())),
        "arrival_start_s": min(arrivals) if arrivals else None,
        "arrival_end_s": max(arrivals) if arrivals else None,
        "arrival_span_s": span,
        "empirical_arrival_rate_tasks_per_s": (len(tasks) - 1) / span
        if span > 0
        else None,
    }
    deadline = {
        "count": len(deadlines),
        "relative_deadline_min_s": min(deadlines) if deadlines else None,
        "relative_deadline_mean_s": sum(deadlines) / len(deadlines)
        if deadlines
        else None,
        "relative_deadline_max_s": max(deadlines) if deadlines else None,
    }
    return workload, deadline


def _profile_versions(profile: FrozenProfileBundle) -> dict[str, Any]:
    pipelines = {
        key: {
            "pipeline_hash": item.pipeline_hash,
            "guard_id": item.guard_id,
            "guard_hash": item.guard_hash,
            "encoder_id": item.encoder_id,
            "encoder_hash": item.encoder_hash,
            "deployment_resource_bounds": thaw_json(item.deployment_resource_bounds),
        }
        for key, item in sorted(profile.pipelines.items())
    }
    local_models = {
        key: {
            "model_hash": item.model_hash,
            "protocol_version": item.protocol_version,
            "deployment_resource_bounds": thaw_json(item.deployment_resource_bounds),
        }
        for key, item in sorted(profile.local_models.items())
    }
    edge_models = {
        key: {
            "model_hash": item.model_hash,
            "protocol_version": item.protocol_version,
            "deployment_resource_bounds": thaw_json(item.deployment_resource_bounds),
        }
        for key, item in sorted(profile.edge_models.items())
    }
    return {
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "protocol_version": profile.protocol_version,
        "schema_version": profile.schema_version,
        "preprocessing_resource_bounds": thaw_json(
            profile.preprocessing_resource_bounds
        ),
        "pipelines": pipelines,
        "local_models": local_models,
        "edge_models": edge_models,
        "privacy_policy": {
            "risk_threshold": profile.risk_threshold,
            "confidence_error": profile.confidence_error,
            "min_subjects": profile.min_subjects,
            "min_emission_lcb": profile.min_emission_lcb,
            "quality_bins": list(profile.quality_bins),
        },
    }


def build_manifest(
    *,
    config: SimulationConfig,
    profile: FrozenProfileBundle,
    state: SimulationState,
    metrics_artifacts: Any,
    trace_bundle: Any | None = None,
    scenario_trace_bundle: Any | None = None,
    data_split: Any | None = None,
    trace_paths: Sequence[str | Path] | None = None,
    trace_data_kind: str | None = None,
    invariant_failures: Iterable[Mapping[str, Any]] = (),
    invariants_passed: bool | None = None,
    simulation_start_s: float = 0.0,
    source_root: str | Path | None = None,
    config_content: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, str] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    evidence_verification: EvidenceVerification | None = None,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    """Build a complete, strict-JSON reproducibility manifest."""

    if not isinstance(require_clean_source, bool):
        raise TypeError("require_clean_source must be a bool")
    code_version = detect_code_version(source_root)
    source_preflight = _preflight_record(
        code_version, require_clean=require_clean_source
    )
    if require_clean_source and source_preflight["requirement_status"] != "passed":
        raise RuntimeError(
            "clean committed source is required for this run: "
            f"assessment={source_preflight['assessment']}, "
            f"reason_codes={source_preflight['reason_codes']}, "
            f"source_git_status={source_preflight['source_git_status']}"
        )
    if not math.isfinite(simulation_start_s) or simulation_start_s < 0:
        raise ValueError("simulation_start_s must be finite and nonnegative")
    if state.clock_s < simulation_start_s:
        raise ValueError("simulation end precedes simulation start")

    artifacts = _plain(metrics_artifacts)
    core_digest = artifacts.get("core_digest") if isinstance(artifacts, dict) else None
    if not isinstance(core_digest, str) or _SHA256_RE.fullmatch(core_digest) is None:
        raise ValueError(
            "metrics_artifacts must provide a canonical lowercase SHA-256 core_digest"
        )

    config_plain = (
        _plain(config_content) if config_content is not None else _plain(config)
    )
    config_hash = hashlib.sha256(canonical_json_bytes(config_plain)).hexdigest()
    if trace_paths is None:
        trace_source = getattr(trace_bundle, "source_path", config.trace_path)
        selected_traces = [Path(trace_source)]
        if scenario_trace_bundle is not None:
            scenario_source = getattr(
                scenario_trace_bundle,
                "source_path",
                config.scenario_trace_path,
            )
            selected_traces.append(Path(scenario_source))
    else:
        selected_traces = [Path(path) for path in trace_paths]
    trace_checksums = [_path_checksum_record(path) for path in selected_traces]
    failures = [_plain(item) for item in invariant_failures]
    if (
        isinstance(state.invariant_checks, bool)
        or not isinstance(state.invariant_checks, int)
        or state.invariant_checks < 0
    ):
        raise ValueError("state.invariant_checks must be a nonnegative integer")
    expected_invariants_passed = state.invariant_checks > 0 and not failures
    if invariants_passed is None:
        invariants_passed = expected_invariants_passed
    elif not isinstance(invariants_passed, bool):
        raise TypeError("invariants_passed must be a bool or None")
    elif invariants_passed != expected_invariants_passed:
        raise ValueError(
            "invariants_passed contradicts invariant failure/check counts: "
            f"passed={invariants_passed}, checks={state.invariant_checks}, failures={len(failures)}"
        )
    status = (
        "passed"
        if invariants_passed
        else ("not_run" if state.invariant_checks == 0 and not failures else "failed")
    )

    metadata = thaw_json(profile.metadata)
    trace_metadata = thaw_json(getattr(trace_bundle, "metadata", {}))
    scenario_trace_metadata = thaw_json(getattr(scenario_trace_bundle, "metadata", {}))
    split = data_split
    if split is None:
        split = {
            "evaluation": trace_metadata.get(
                "data_split",
                {"status": "not_declared"},
            ),
            "scenario_training_validation": scenario_trace_metadata.get(
                "data_split",
                {"status": "not_declared"},
            ),
        }
    profile_kind = profile.data_kind
    trace_kind = (
        trace_data_kind
        or getattr(trace_bundle, "data_kind", None)
        or metadata.get("trace_data_kind", "not_declared")
    )
    scenario_trace_kind = (
        getattr(scenario_trace_bundle, "data_kind", None) or "not_declared"
    )
    formal_declared = (
        metadata.get("formal_experiment_eligible") is True
        and trace_metadata.get("formal_experiment_eligible") is True
        and scenario_trace_metadata.get("formal_experiment_eligible") is True
    )
    numerical_declared = (
        metadata.get("numerical_experiment_eligible") is True
        and trace_metadata.get("numerical_experiment_eligible") is True
        and scenario_trace_metadata.get("numerical_experiment_eligible") is True
    )
    evidence_record = evidence_manifest_record(
        evidence_verification or EvidenceVerification.not_required()
    )
    semantic_config_hash, config_path_identities = _configuration_semantic_identity(
        config_plain,
        profile_hash=profile.profile_hash,
        trace_checksums=trace_checksums,
        evidence_record=evidence_record,
    )
    numerical_evidence_verified = bool(
        evidence_record.get("required") and evidence_record.get("verified")
    )
    if (
        profile_kind == "synthetic"
        or trace_kind == "synthetic"
        or scenario_trace_kind == "synthetic"
    ):
        result_label = "synthetic_engineering_only"
    elif (
        profile_kind == "numerical_simulation"
        and trace_kind == "numerical_simulation"
        and scenario_trace_kind == "numerical_simulation"
        and numerical_declared
        and numerical_evidence_verified
    ):
        result_label = "numerical_model_conditional"
    elif (
        profile_kind == "measured"
        and trace_kind == "measured"
        and scenario_trace_kind == "measured"
        and formal_declared
    ):
        result_label = "measured_trace_conditional"
    elif profile_kind == "measured" and trace_kind == "measured":
        result_label = "measured_but_formal_eligibility_unverified"
    else:
        result_label = "mixed_or_unverified"

    workload, deadline = _task_workload(state)
    resource_config = {
        "vehicles": _plain(config.vehicles),
        "rsus": _plain(config.rsus),
    }
    network_config = {
        "max_snapshot_age_s": config.max_snapshot_age_s,
        "rsu_snapshot_period_s": config.rsu_snapshot_period_s,
        "rsu_telemetry_delay_s": config.rsu_telemetry_delay_s,
        "rsu_telemetry_quantum_work_s": config.rsu_telemetry_quantum_work_s,
        "rsu_telemetry_drop_every": config.rsu_telemetry_drop_every,
        "uplink_pause_limit_s": config.uplink_pause_limit_s,
        "downlink_pause_limit_s": config.downlink_pause_limit_s,
        "metadata_bits": config.metadata_bits,
    }
    artifact_files = artifacts.get("files", {}) if isinstance(artifacts, dict) else {}
    parquet_status = (
        artifacts.get("parquet_status", {}) if isinstance(artifacts, dict) else {}
    )
    controller_diagnostics = (
        artifacts.get("controller_diagnostics", {})
        if isinstance(artifacts, dict)
        else {}
    )

    document: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "core_digest": core_digest,
        "code_version": code_version,
        "source_cleanliness_preflight": source_preflight,
        "configuration": {
            "canonical_sha256": config_hash,
            "canonical_sha256_semantics": "runtime_content_including_paths",
            "semantic_sha256": semantic_config_hash,
            "semantic_sha256_semantics": "path_roles_bound_to_content_sha256",
            "path_identities": config_path_identities,
            "content": config_plain,
        },
        "versions": _profile_versions(profile),
        "protocol_version": config.protocol_version,
        "trace_checksums": trace_checksums,
        "frozen_evidence": evidence_record,
        "event_priority": {kind.value: int(priority_for(kind)) for kind in EventKind},
        "trace_identity": {
            "schema_version": getattr(trace_bundle, "schema_version", None),
            "trace_version": getattr(trace_bundle, "trace_version", None),
            "trace_hash": getattr(trace_bundle, "trace_hash", None),
            "profile_hash": getattr(trace_bundle, "profile_hash", None),
            "protocol_version": getattr(trace_bundle, "protocol_version", None),
            "evidence_status": getattr(trace_bundle, "evidence_status", None),
            "seed": getattr(trace_bundle, "seed", None),
            "horizon_start_s": getattr(trace_bundle, "horizon_start_s", None),
            "horizon_end_s": getattr(trace_bundle, "horizon_end_s", None),
        },
        "scenario_trace_identity": {
            "schema_version": getattr(scenario_trace_bundle, "schema_version", None),
            "trace_version": getattr(scenario_trace_bundle, "trace_version", None),
            "trace_hash": getattr(scenario_trace_bundle, "trace_hash", None),
            "profile_hash": getattr(scenario_trace_bundle, "profile_hash", None),
            "protocol_version": getattr(
                scenario_trace_bundle, "protocol_version", None
            ),
            "evidence_status": getattr(scenario_trace_bundle, "evidence_status", None),
            "seed": getattr(scenario_trace_bundle, "seed", None),
        },
        "data_split": _plain(split),
        "profile_metadata": metadata,
        "trace_metadata": trace_metadata,
        "scenario_trace_metadata": scenario_trace_metadata,
        "data_provenance": {
            "profile_data_kind": profile_kind,
            "trace_data_kind": trace_kind,
            "scenario_trace_data_kind": scenario_trace_kind,
            "evidence_status": profile.evidence_status,
            "result_label": result_label,
            "source_commit_reproducible": source_preflight[
                "source_commit_reproducible"
            ],
            "formal_experiment_eligible": bool(
                profile_kind == "measured"
                and trace_kind == "measured"
                and scenario_trace_kind == "measured"
                and formal_declared
            ),
            "numerical_experiment_eligible": bool(
                profile_kind == "numerical_simulation"
                and trace_kind == "numerical_simulation"
                and scenario_trace_kind == "numerical_simulation"
                and numerical_declared
                and numerical_evidence_verified
            ),
            "real_hardware_measurement": bool(
                profile_kind == "measured"
                and trace_kind == "measured"
                and scenario_trace_kind == "measured"
            ),
        },
        "parameter_sources": {
            "configuration": _plain(config.parameter_sources),
            "profile": thaw_json(profile.parameter_sources),
            "trace": thaw_json(getattr(trace_bundle, "parameter_sources", {})),
            "scenario_trace": thaw_json(
                getattr(scenario_trace_bundle, "parameter_sources", {})
            ),
        },
        "resources": resource_config,
        "network": network_config,
        "workload": workload,
        "deadline": deadline,
        "controller": _plain(config.controller),
        "seeds": _plain(config.seeds),
        "software_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os_name": os.name,
            "dependencies": _dependency_versions(dependencies),
        },
        "simulation": {
            "start_time_s": simulation_start_s,
            "end_time_s": state.clock_s,
            "duration_s": state.clock_s - simulation_start_s,
            "terminal_task_count": sum(
                1 for task in state.tasks.values() if task.terminal
            ),
            "task_count": len(state.tasks),
        },
        "invariants": {
            "status": status,
            "passed": bool(invariants_passed),
            "check_count": state.invariant_checks,
            "failure_count": len(failures),
            "failures": failures,
        },
        "outputs": {"files": artifact_files, "parquet": parquet_status},
        "controller_diagnostics": controller_diagnostics,
        "run_metadata": _plain(run_metadata or {}),
    }
    # Validate finite strict JSON now, not after a long experiment has returned.
    canonical_json_bytes(document)
    return document


def write_manifest(
    path: str | Path, manifest: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Write ``manifest.json`` with a self-excluding canonical checksum."""

    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    document = _plain(manifest)
    if not isinstance(document, dict):
        raise TypeError("manifest root must be a mapping")
    material = dict(document)
    material.pop("manifest_sha256", None)
    document["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    encoded = (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    )
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


__all__ = [
    "KNOWN_OUTPUT_FILES",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "canonical_core_digest",
    "detect_code_version",
    "deterministic_core_value",
    "prepare_output_directory",
    "sha256_file",
    "source_cleanliness_preflight",
    "source_tree_portable_sha256",
    "source_tree_sha256",
    "write_manifest",
]
