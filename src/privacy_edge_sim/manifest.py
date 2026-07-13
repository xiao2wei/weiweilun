"""Reproducibility manifests and conservative output-directory handling.

The manifest deliberately separates the deterministic simulation digest from
engineering diagnostics such as controller wall-clock measurements and host
paths.  A run can therefore be repeated in a different output directory while
retaining the same ``core_digest``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
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


MANIFEST_SCHEMA_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


def _eligible_tree_file(path: Path) -> bool:
    parts = set(path.parts)
    if parts & {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def source_tree_sha256(source_root: str | Path | None = None) -> tuple[str, int]:
    """Hash source paths and bytes in stable lexical order.

    By default only the installed package source tree is hashed.  Generated
    experiment outputs can therefore never change the code identity.
    """

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
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(paths)


def detect_code_version(source_root: str | Path | None = None) -> dict[str, Any]:
    """Report Git identity when available and always include a tree checksum."""

    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    tree_hash, file_count = source_tree_sha256(root)
    result: dict[str, Any] = {
        "kind": "source_tree_sha256",
        "value": tree_hash,
        "source_tree_sha256": tree_hash,
        "source_file_count": file_count,
    }
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", top, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return result
    result.update(
        {"kind": "git", "value": commit, "git_commit": commit, "git_dirty": dirty}
    )
    return result


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
) -> dict[str, Any]:
    """Build a complete, strict-JSON reproducibility manifest."""

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
        "code_version": detect_code_version(source_root),
        "configuration": {"canonical_sha256": config_hash, "content": config_plain},
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
    "source_tree_sha256",
    "write_manifest",
]
