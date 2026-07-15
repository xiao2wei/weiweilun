"""Command-line entrypoints for validation, simulation and experiment orchestration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import SimulationConfig, load_config
from .checkpoint import (
    checkpoint_identity,
    load_replay_checkpoint,
    write_replay_checkpoint,
)
from .errors import InvariantViolation
from .experiments import build_numerical_experiment_report
from .evidence import verify_run_evidence
from .evidence_reports import build_subject_cluster_evidence_report
from .manifest import (
    KNOWN_OUTPUT_FILES,
    build_manifest,
    detect_code_version,
    prepare_output_directory,
    sha256_file,
    source_cleanliness_preflight,
    write_manifest,
)
from .numerical import (
    NumericalStudySpec,
    build_numerical_config_from_evidence,
    generate_numerical_replication,
    generate_numerical_study,
)
from .paper_experiment_audits import (
    AuditValidationError,
    audit_failure_cost_coverage,
    audit_failure_cost_completeness,
    audit_hard_mask_counterfactual,
    build_preregistered_one_shot_commitments,
    evaluate_two_stage_information_ablation,
    exact_adaptive_scenario_tree_oracle,
    exact_finite_scenario_oracle,
)
from .profiles import canonical_document_sha256, canonical_json_bytes, load_profile
from .selection import (
    ValidationCandidate,
    ValidationLimits,
    select_feasible_validation_candidate,
)
from .sensitivity_analysis import analyze_registered_sensitivity_sweeps
from .simulator import DiscreteEventSimulator, RunResult
from .statistics import (
    aggregate_preregistered_study_families,
    analyze_paired_strategies,
    apply_holm_family_adjustment,
)
from .synthetic import generate_synthetic_bundle
from .traces import load_trace


POLICIES = (
    "all_local",
    "fixed_safe_lowest_link_cost",
    "fixed_safe_shortest_visible_queue",
    "safe_greedy",
    "safe_lyapunov_h1",
    "esl_smpc",
)
RUNNABLE_POLICIES = (*POLICIES, "safe_one_shot")
_SOURCE_IDENTITY_KEYS = (
    "git_commit",
    "source_git_object",
    "source_tree_sha256",
    "source_tree_portable_sha256",
)
_FROZEN_INPUT_PATHS = (
    ("profile", "profile_path"),
    ("evaluation_trace", "trace_path"),
    ("scenario_trace", "scenario_trace_path"),
    ("evidence", "evidence_path"),
)
_FROZEN_INPUT_DECLARED_HASH_FIELDS = {
    "profile": "profile_hash",
    "evaluation_trace": "trace_hash",
    "scenario_trace": "trace_hash",
    "evidence": "evidence_hash",
}


def _source_identity(preflight: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(preflight.get(key) for key in _SOURCE_IDENTITY_KEYS)


def _require_same_source_identity(
    actual: dict[str, Any], expected: tuple[Any, ...], *, context: str
) -> None:
    actual_identity = _source_identity(actual)
    if actual_identity != expected:
        raise RuntimeError(
            f"executable source identity changed {context}: "
            f"expected={dict(zip(_SOURCE_IDENTITY_KEYS, expected, strict=True))}, "
            f"actual={dict(zip(_SOURCE_IDENTITY_KEYS, actual_identity, strict=True))}"
        )


def _capture_frozen_input_assets(
    config: SimulationConfig, *, include_declared_content_hashes: bool = True
) -> dict[str, Any]:
    """Capture byte-exact identities for every immutable run input.

    The loaders validate each document's semantic/self hash.  This additional
    identity binds the exact raw bytes read at the load boundary so harmless
    JSON reformatting, newline changes, and concurrent replacement are also
    detectable.
    """

    assets: dict[str, dict[str, Any]] = {}
    for role, attribute in _FROZEN_INPUT_PATHS:
        configured = getattr(config, attribute)
        if configured is None:
            assets[role] = {
                "status": "not_configured",
                "path": None,
                "raw_sha256": None,
                "size_bytes": None,
            }
            continue
        path = Path(configured).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"frozen {role} input is not a regular file: {path}"
            )
        digest = hashlib.sha256()
        size_bytes = 0
        raw_bytes: bytes | None = None
        if include_declared_content_hashes:
            raw_bytes = path.read_bytes()
            digest.update(raw_bytes)
            size_bytes = len(raw_bytes)
        else:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size_bytes += len(block)
        record: dict[str, Any] = {
            "status": "captured",
            "path": str(path),
            "raw_sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        }
        if raw_bytes is not None:
            try:
                document = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"frozen {role} input is not strict UTF-8 JSON: {path}"
                ) from exc
            if not isinstance(document, dict):
                raise ValueError(
                    f"frozen {role} input root must be an object: {path}"
                )
            declared_hash_field = _FROZEN_INPUT_DECLARED_HASH_FIELDS[role]
            record.update(
                {
                    "declared_hash_field": declared_hash_field,
                    "declared_content_hash": document.get(declared_hash_field),
                }
            )
            del raw_bytes, document
        assets[role] = record
    return {
        "schema_version": "1.0",
        "hash_algorithm": "sha256",
        "hash_semantics": "exact_raw_file_bytes_at_frozen_load_boundary",
        "assets": assets,
    }


def _require_same_frozen_input_assets(
    config: SimulationConfig,
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    actual = _capture_frozen_input_assets(
        config, include_declared_content_hashes=False
    )
    expected_assets = expected.get("assets", {})
    actual_assets = actual.get("assets", {})
    identity_fields = ("status", "path", "raw_sha256", "size_bytes")
    changed = sorted(
        role
        for role, _ in _FROZEN_INPUT_PATHS
        if tuple(expected_assets.get(role, {}).get(field) for field in identity_fields)
        != tuple(actual_assets.get(role, {}).get(field) for field in identity_fields)
    )
    if not changed:
        return
    raise RuntimeError(
        "frozen input bytes changed "
        f"{context}: roles={changed}, expected={expected_assets}, actual={actual_assets}"
    )


def _require_frozen_input_content_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    roles: tuple[str, ...],
    context: str,
) -> None:
    """Compare immutable input contents while allowing copied paths to differ."""

    fields = ("status", "raw_sha256", "size_bytes", "declared_content_hash")
    actual_assets = actual.get("assets", {})
    expected_assets = expected.get("assets", {})
    changed = [
        role
        for role in roles
        if tuple(actual_assets.get(role, {}).get(field) for field in fields)
        != tuple(expected_assets.get(role, {}).get(field) for field in fields)
    ]
    if changed:
        raise RuntimeError(
            "frozen input content differs from the batch base bundle "
            f"{context}: roles={changed}"
        )


def _require_same_file_bytes(path: Path, expected_sha256: str, *, context: str) -> None:
    """Fail closed when a batch control file changes after registration."""

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"batch control file changed {context}: path={path}, "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )


_NUMERICAL_REPLICATION_SEED_STREAMS = (
    "environment",
    "arrivals",
    "mobility",
    "wireless",
    "vehicle",
    "rsu",
    "fault",
)


def _replication_config_signature(document: dict[str, Any]) -> bytes:
    """Normalize only the preregistered per-environment RNG streams."""

    normalized = json.loads(json.dumps(document))
    seeds = normalized.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("numerical replication config has no seed mapping")
    for stream in _NUMERICAL_REPLICATION_SEED_STREAMS:
        if stream not in seeds:
            raise ValueError(f"numerical replication config is missing seed: {stream}")
        seeds[stream] = "<REGISTERED_ENVIRONMENT_REPLICATION_STREAM>"
    return canonical_json_bytes(normalized)


def _require_loaded_input_identity(
    *,
    frozen_inputs: dict[str, Any],
    profile: Any,
    evaluation_trace: Any,
    scenario_trace: Any,
    evidence_verification: Any,
) -> None:
    """Bind every loader-produced semantic identity to the raw snapshot."""

    assets = frozen_inputs["assets"]
    loaded = {
        "profile": (profile, "profile_hash"),
        "evaluation_trace": (evaluation_trace, "trace_hash"),
        "scenario_trace": (scenario_trace, "trace_hash"),
    }
    for role, (bundle, attribute) in loaded.items():
        expected = assets[role]
        if getattr(bundle, attribute, None) != expected["declared_content_hash"]:
            raise RuntimeError(
                f"loaded {role} identity does not match the pre-load frozen bytes"
            )
        source_path = getattr(bundle, "source_path", None)
        if source_path is None or str(Path(source_path).resolve()) != expected["path"]:
            raise RuntimeError(f"loaded {role} source path changed at load boundary")

    evidence = assets["evidence"]
    if evidence["status"] == "not_configured":
        if evidence_verification.required:
            raise RuntimeError(
                "evidence loader required an input absent from the frozen snapshot"
            )
        return
    if (
        not evidence_verification.required
        or evidence_verification.file_sha256 != evidence["raw_sha256"]
        or evidence_verification.size_bytes != evidence["size_bytes"]
        or evidence_verification.evidence_hash != evidence["declared_content_hash"]
        or evidence_verification.path is None
        or str(evidence_verification.path.resolve()) != evidence["path"]
    ):
        raise RuntimeError(
            "loaded evidence identity does not match the pre-load frozen bytes"
        )


def _require_manifest_frozen_input_identity(
    manifest: dict[str, Any], frozen_inputs: dict[str, Any]
) -> None:
    """Reject a manifest whose existing checksum fields contradict the snapshot."""

    assets = frozen_inputs["assets"]
    trace_checksums = manifest.get("trace_checksums")
    if not isinstance(trace_checksums, list) or len(trace_checksums) != 2:
        raise RuntimeError("manifest must contain both frozen trace checksums")
    for index, role in enumerate(("evaluation_trace", "scenario_trace")):
        record = trace_checksums[index]
        files = record.get("files") if isinstance(record, dict) else None
        values = list(files.values()) if isinstance(files, dict) else []
        if (
            not isinstance(record, dict)
            or record.get("kind") != "file"
            or values != [assets[role]["raw_sha256"]]
        ):
            raise RuntimeError(
                f"manifest {role} checksum does not match the loaded frozen bytes"
            )
    evidence_record = manifest.get("frozen_evidence")
    expected_evidence = assets["evidence"]
    if expected_evidence["status"] == "captured":
        if (
            not isinstance(evidence_record, dict)
            or evidence_record.get("file_sha256")
            != expected_evidence["raw_sha256"]
            or evidence_record.get("size_bytes") != expected_evidence["size_bytes"]
        ):
            raise RuntimeError(
                "manifest evidence checksum does not match the loaded frozen bytes"
            )
    manifest["frozen_input_assets"] = frozen_inputs
    canonical_json_bytes(manifest)


def _config_for_policy(config: SimulationConfig, policy: str) -> SimulationConfig:
    return replace(config, controller=replace(config.controller, policy=policy))


def _failure_snapshot(
    simulator: DiscreteEventSimulator, exc: InvariantViolation
) -> dict[str, Any]:
    state = simulator.state
    return {
        "status": "invariant_failure",
        "time_s": state.clock_s,
        "error": {
            "code": exc.detail.code,
            "message": exc.detail.message,
            "context": exc.detail.context,
        },
        "tasks": {
            task_id: {
                "state": task.state.value,
                "failure_reason": task.failure_reason.value,
                "current_job_id": task.current_job_id,
                "current_transfer_id": task.current_transfer_id,
                "attempt_started_count": task.attempt_started_count,
                "selected_pipeline": task.selected_pipeline,
                "selected_rsu": task.selected_rsu,
            }
            for task_id, task in sorted(state.tasks.items())
        },
        "vehicles": {
            vehicle_id: {
                "battery_j": runtime.battery_j,
                "memory_reserved_bytes": runtime.memory_reserved_bytes,
                "descriptors_reserved": dict(
                    sorted(runtime.descriptors_reserved.items())
                ),
                "failed": runtime.failed,
            }
            for vehicle_id, runtime in sorted(state.vehicles.items())
        },
        "rsus": {
            rsu_id: {
                "descriptors": runtime.admission.descriptors,
                "vram_bytes": runtime.admission.vram_bytes,
                "reserved_work_gpu_s": runtime.admission.reserved_work_gpu_s,
                "failed": runtime.failed,
            }
            for rsu_id, runtime in sorted(state.rsus.items())
        },
        "transfers": {
            transfer_id: {
                "task_id": transfer.task_id,
                "rsu_id": transfer.rsu_id,
                "direction": transfer.direction.value,
                "status": transfer.status.value,
                "total_bits": transfer.total_bits,
                "remaining_bits": transfer.remaining_bits,
                "delivered_bits": transfer.delivered_bits,
            }
            for transfer_id, transfer in sorted(state.transfers.items())
        },
        "recent_events": state.event_log[-20:],
        "note": "fatal invariant failure; simulation stopped without normal result artifacts",
    }


def _execute(
    config: SimulationConfig,
    policy: str,
    output: Path,
    *,
    overwrite: bool,
    run_metadata: dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    resume_checkpoint_path: Path | None = None,
    require_clean_source: bool = False,
    expected_source_identity: tuple[Any, ...] | None = None,
) -> tuple[RunResult, Any, dict[str, Any]]:
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if checkpoint_every and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint_every is set")
    source_at_start = (
        source_cleanliness_preflight(require_clean=True)
        if require_clean_source
        else None
    )
    if expected_source_identity is not None:
        if source_at_start is None:
            raise ValueError(
                "expected_source_identity requires require_clean_source=True"
            )
        _require_same_source_identity(
            source_at_start,
            expected_source_identity,
            context="before formal simulation",
        )
    frozen_inputs = _capture_frozen_input_assets(config)
    output = prepare_output_directory(output, overwrite=overwrite)
    profile = load_profile(config.profile_path)
    trace = load_trace(config.trace_path, profile)
    scenario_trace = load_trace(config.scenario_trace_path, profile)
    evidence_verification = verify_run_evidence(config, profile, trace, scenario_trace)
    _require_same_frozen_input_assets(
        config,
        frozen_inputs,
        context="while loading frozen inputs",
    )
    _require_loaded_input_identity(
        frozen_inputs=frozen_inputs,
        profile=profile,
        evaluation_trace=trace,
        scenario_trace=scenario_trace,
        evidence_verification=evidence_verification,
    )
    run_config = _config_for_policy(config, policy)
    simulator = DiscreteEventSimulator(
        run_config,
        profile,
        trace,
        policy,
        policy_name=policy,
        scenario_trace=scenario_trace,
    )
    code_version = str(detect_code_version()["value"])
    identity = checkpoint_identity(
        config=run_config,
        profile_hash=profile.profile_hash,
        evaluation_trace_hash=trace.trace_hash,
        scenario_trace_hash=scenario_trace.trace_hash,
        protocol_version=run_config.protocol_version,
        policy=policy,
        code_version=code_version,
    )
    replay_document = (
        None
        if resume_checkpoint_path is None
        else load_replay_checkpoint(resume_checkpoint_path, expected_identity=identity)
    )

    def checkpoint_callback(
        compound_events: int, clock_s: float, prefix_sha256: str, complete: bool
    ) -> None:
        if checkpoint_path is None:
            return
        if not complete and (
            checkpoint_every <= 0 or compound_events % checkpoint_every != 0
        ):
            return
        write_replay_checkpoint(
            checkpoint_path,
            identity=identity,
            compound_events=compound_events,
            clock_s=clock_s,
            prefix_sha256=prefix_sha256,
            complete=complete,
        )

    try:
        result = simulator.run(
            replay_checkpoint=replay_document,
            checkpoint_callback=(
                checkpoint_callback if checkpoint_path is not None else None
            ),
        )
    except InvariantViolation as exc:
        _require_same_frozen_input_assets(
            config,
            frozen_inputs,
            context="during failed simulation",
        )
        failure = _failure_snapshot(simulator, exc)
        (output / "failure.json").write_text(
            json.dumps(
                failure, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    _require_same_frozen_input_assets(
        config,
        frozen_inputs,
        context="during simulation",
    )
    if source_at_start is not None:
        source_at_end = source_cleanliness_preflight(require_clean=True)
        _require_same_source_identity(
            source_at_end,
            _source_identity(source_at_start),
            context="during formal simulation",
        )
        if expected_source_identity is not None:
            _require_same_source_identity(
                source_at_end,
                expected_source_identity,
                context="against the formal batch identity",
            )
    _require_same_frozen_input_assets(
        config,
        frozen_inputs,
        context="immediately before output publication",
    )
    artifacts = result.ledger.write_outputs(
        output, result.state, result.config, overwrite=overwrite
    )
    manifest = build_manifest(
        config=result.config,
        profile=result.profile,
        state=result.state,
        metrics_artifacts=artifacts,
        trace_bundle=result.trace,
        scenario_trace_bundle=result.scenario_trace,
        invariant_failures=result.invariant_failures,
        simulation_start_s=result.trace.horizon_start_s,
        run_metadata={
            **(run_metadata or {}),
            "policy": policy,
            "replay_checkpoint_resumed": resume_checkpoint_path is not None,
            "replay_checkpoint_mode": "deterministic_replay",
            **(
                {
                    "source_identity_stable_during_simulation": True,
                    "source_identity_at_start": {
                        key: source_at_start.get(key)
                        for key in _SOURCE_IDENTITY_KEYS
                    },
                }
                if source_at_start is not None
                else {}
            ),
        },
        evidence_verification=evidence_verification,
        require_clean_source=require_clean_source,
    )
    _require_manifest_frozen_input_identity(manifest, frozen_inputs)
    if source_at_start is not None:
        manifest_preflight = manifest.get("source_cleanliness_preflight")
        if not isinstance(manifest_preflight, dict):
            raise RuntimeError("formal manifest lacks source cleanliness identity")
        _require_same_source_identity(
            manifest_preflight,
            _source_identity(source_at_start),
            context="between simulation and manifest construction",
        )
        if expected_source_identity is not None:
            _require_same_source_identity(
                manifest_preflight,
                expected_source_identity,
                context="between formal batch start and manifest construction",
            )
    _require_same_frozen_input_assets(
        config,
        frozen_inputs,
        context="before manifest publication",
    )
    write_manifest(output / "manifest.json", manifest)
    return result, artifacts, manifest


def _prepare_batch_root(
    root: Path,
    *,
    leaf_directories: list[Path],
    index_files: set[str],
    overwrite: bool,
) -> Path:
    """Preflight an entire batch before the first simulator mutation."""

    resolved = root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    existing = list(resolved.rglob("*"))
    if existing and not overwrite:
        raise FileExistsError(f"batch output root is not empty: {resolved}")
    if not existing:
        return resolved

    allowed_dirs: set[Path] = {resolved}
    allowed_files = {resolved / name for name in index_files}
    for relative_leaf in leaf_directories:
        leaf = (resolved / relative_leaf).resolve()
        if resolved not in leaf.parents:
            raise ValueError(f"batch leaf escapes output root: {relative_leaf}")
        cursor = leaf
        while cursor != resolved:
            allowed_dirs.add(cursor)
            cursor = cursor.parent
        allowed_files.update(leaf / name for name in KNOWN_OUTPUT_FILES)
    unsafe = [
        item
        for item in existing
        if (item.is_dir() and item.resolve() not in allowed_dirs)
        or (item.is_file() and item.resolve() not in allowed_files)
    ]
    if unsafe:
        raise FileExistsError(
            f"refusing to mix unknown entries into batch root {resolved}: "
            f"{[str(item) for item in unsafe]}"
        )
    return resolved


def _summary_line(result: RunResult, artifacts: Any, output: Path) -> dict[str, Any]:
    done = sum(task.state.value == "DONE" for task in result.state.tasks.values())
    failed = sum(task.state.value == "FAIL" for task in result.state.tasks.values())
    profile_declared = result.profile.metadata.get("formal_experiment_eligible") is True
    trace_declared = result.trace.metadata.get("formal_experiment_eligible") is True
    scenario_declared = (
        result.scenario_trace.metadata.get("formal_experiment_eligible") is True
    )
    formal_eligible = (
        result.profile.data_kind == "measured"
        and result.trace.data_kind == "measured"
        and result.scenario_trace.data_kind == "measured"
        and profile_declared
        and trace_declared
        and scenario_declared
    )
    numerical_eligible = (
        result.profile.data_kind == "numerical_simulation"
        and result.trace.data_kind == "numerical_simulation"
        and result.scenario_trace.data_kind == "numerical_simulation"
        and result.profile.metadata.get("numerical_experiment_eligible") is True
        and result.trace.metadata.get("numerical_experiment_eligible") is True
        and result.scenario_trace.metadata.get("numerical_experiment_eligible") is True
        and result.config.evidence_path is not None
    )
    return {
        "policy": result.policy_name,
        "tasks": len(result.state.tasks),
        "done": done,
        "failed": failed,
        "sim_end_s": result.state.clock_s,
        "invariant_checks": result.state.invariant_checks,
        "core_digest": artifacts.core_digest,
        "output": str(output.resolve()),
        "data_kind": result.trace.data_kind,
        "formal_experiment_eligible": formal_eligible,
        "numerical_experiment_eligible": numerical_eligible,
        "real_hardware_measurement": result.trace.data_kind == "measured",
    }


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    profile = load_profile(config.profile_path)
    trace = load_trace(config.trace_path, profile)
    scenario_trace = load_trace(config.scenario_trace_path, profile)
    evidence_verification = verify_run_evidence(config, profile, trace, scenario_trace)
    # Constructing the simulator performs the measured split/non-overlap gate
    # in addition to the two individual trace loader checks.
    DiscreteEventSimulator(
        config,
        profile,
        trace,
        "all_local",
        scenario_trace=scenario_trace,
    )
    print(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "profile_version": profile.profile_version,
                "profile_hash": profile.profile_hash,
                "trace_version": trace.trace_version,
                "trace_hash": trace.trace_hash,
                "scenario_trace_version": scenario_trace.trace_version,
                "scenario_trace_hash": scenario_trace.trace_hash,
                "protocol_version": profile.protocol_version,
                "data_kind": trace.data_kind,
                "arrivals": len(trace.arrivals),
                "evidence_required": evidence_verification.required,
                "evidence_verified": evidence_verification.verified,
                "evidence_hash": evidence_verification.evidence_hash,
                "validated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_validate_profile(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    print(
        json.dumps(
            {
                "path": str(profile.source_path),
                "profile_version": profile.profile_version,
                "profile_hash": profile.profile_hash,
                "data_kind": profile.data_kind,
                "pipelines": sorted(profile.pipelines),
                "validated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_validate_trace(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    trace = load_trace(args.trace, profile)
    print(
        json.dumps(
            {
                "path": str(trace.source_path),
                "trace_version": trace.trace_version,
                "trace_hash": trace.trace_hash,
                "data_kind": trace.data_kind,
                "arrivals": len(trace.arrivals),
                "joint_anon_rows": len(trace.anon_rows),
                "validated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    paths = generate_synthetic_bundle(
        args.output_root, seed=args.seed, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "profile_path": str(paths.profile_path),
                "trace_path": str(paths.trace_path),
                "scenario_trace_path": str(paths.scenario_trace_path),
                "profile_hash": paths.profile_hash,
                "trace_hash": paths.trace_hash,
                "scenario_trace_hash": paths.scenario_trace_hash,
                "data_kind": "synthetic",
                "formal_experiment_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


def command_generate_numerical(args: argparse.Namespace) -> int:
    spec = NumericalStudySpec(
        seed=args.seed,
        profile_evaluation_subjects=args.profile_subjects,
        test_subjects=args.test_subjects,
        scenario_subjects=args.scenario_subjects,
        task_count=args.tasks,
        horizon_s=args.horizon,
        arrival_center_s=args.arrival_center_s,
        arrival_window_s=args.arrival_window_s,
        arrival_jitter_fraction=args.arrival_jitter_fraction,
        privacy_threshold=args.privacy_threshold,
        preprocessing_failure_mode=args.preprocessing_failure_mode,
        preprocessing_failure_count=args.preprocessing_failure_count,
        preprocessing_failure_probability=args.preprocessing_failure_probability,
        local_service_scale=args.local_service_scale,
        anon_time_variability_scale=args.anon_time_variability_scale,
        output_size_variability_scale=args.output_size_variability_scale,
    )
    paths = generate_numerical_study(
        args.output_root, spec=spec, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "config_path": str(paths.config_path),
                "profile_path": str(paths.profile_path),
                "evaluation_trace_path": str(paths.evaluation_trace_path),
                "scenario_trace_path": str(paths.scenario_trace_path),
                "evidence_path": str(paths.evidence_path),
                "profile_hash": paths.profile_hash,
                "evaluation_trace_hash": paths.evaluation_trace_hash,
                "scenario_trace_hash": paths.scenario_trace_hash,
                "evidence_hash": paths.evidence_hash,
                "data_kind": "numerical_simulation",
                "real_hardware_measurement": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_generate_numerical_replication(args: argparse.Namespace) -> int:
    paths = generate_numerical_replication(
        args.base_study_root,
        args.output_root,
        args.environment_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "config_path": str(paths.config_path),
                "evaluation_trace_path": str(paths.evaluation_trace_path),
                "scenario_trace_path": str(paths.scenario_trace_path),
                "profile_hash": paths.profile_hash,
                "evaluation_trace_hash": paths.evaluation_trace_hash,
                "scenario_trace_hash": paths.scenario_trace_hash,
                "environment_seed": args.environment_seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    policy = args.policy or config.controller.policy
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint)
    resume_path = (
        None if args.resume_checkpoint is None else Path(args.resume_checkpoint)
    )
    if resume_path is not None and checkpoint_path is None:
        checkpoint_path = resume_path
    result, artifacts, _ = _execute(
        config,
        policy,
        Path(args.output),
        overwrite=args.overwrite,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        resume_checkpoint_path=resume_path,
    )
    print(
        json.dumps(
            _summary_line(result, artifacts, Path(args.output)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_run_all(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = Path(args.output_root)
    policies = tuple(args.policies.split(",")) if args.policies else POLICIES
    unknown = sorted(set(policies) - set(RUNNABLE_POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if len(policies) != len(set(policies)):
        raise ValueError("policies must be unique")
    root = _prepare_batch_root(
        root,
        leaf_directories=[Path(policy) for policy in policies],
        index_files={"runs.json"},
        overwrite=args.overwrite,
    )
    rows: list[dict[str, Any]] = []
    for policy in policies:
        output = root / policy
        result, artifacts, _ = _execute(
            config, policy, output, overwrite=args.overwrite
        )
        rows.append(_summary_line(result, artifacts, output))
    index = root / "runs.json"
    index.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0


def command_multi_seed(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    policy = args.policy or config.controller.policy
    seed_values = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("base seeds must be a non-empty list of nonnegative integers")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("base seeds must be unique")
    root = _prepare_batch_root(
        Path(args.output_root),
        leaf_directories=[Path(policy) / f"seed-{seed}" for seed in seed_values],
        index_files={"runs.json"},
        overwrite=args.overwrite,
    )
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        seeds = {
            key: int.from_bytes(
                hashlib.sha256(
                    f"privacy-edge-sim|base-seed={seed}|stream={key}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            for key in sorted(config.seeds)
        }
        seeded = replace(config, seeds=MappingProxyType(seeds))
        seeded.validate()
        output = root / policy / f"seed-{seed}"
        result, artifacts, _ = _execute(
            seeded,
            policy,
            output,
            overwrite=args.overwrite,
            run_metadata={"base_seed": seed},
        )
        row = _summary_line(result, artifacts, output)
        row["seed"] = seed
        rows.append(row)
    (root / "runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0


def _set_dotted(document: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current: Any = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    if isinstance(current, list):
        current[int(parts[-1])] = value
    else:
        current[parts[-1]] = value


def _get_dotted(document: dict[str, Any], dotted: str) -> Any:
    """Return an existing dotted value, including numeric list components."""

    if not dotted or any(not part for part in dotted.split(".")):
        raise ValueError(f"invalid dotted path: {dotted!r}")
    current: Any = document
    for part in dotted.split("."):
        if isinstance(current, list):
            try:
                index = int(part)
            except (ValueError, IndexError) as exc:
                raise KeyError(f"unknown dotted path: {dotted}") from exc
            if index < 0 or str(index) != part:
                raise KeyError(f"non-canonical list index in dotted path: {dotted}")
            try:
                current = current[index]
            except IndexError as exc:
                raise KeyError(f"unknown dotted path: {dotted}") from exc
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"unknown dotted path: {dotted}")
    return current


_CONFIG_ASSET_PATHS = (
    "profile_path",
    "trace_path",
    "scenario_trace_path",
    "evidence_path",
)


def _portable_asset_path(asset: Path, output_parent: Path) -> str:
    """Represent a validated asset relative to the derived config when possible."""

    try:
        relative = os.path.relpath(asset, output_parent)
    except ValueError:  # Different Windows drives cannot have a relative path.
        return asset.as_posix()
    return Path(relative).as_posix()


def command_derive_config(args: argparse.Namespace) -> int:
    """Apply a named, auditable override block and validate the derived config."""

    source, document = _read_strict_json_object(args.config)
    base_config = load_config(source)
    override_source, override_document = _read_strict_json_object(args.overrides)
    selected = _get_dotted(override_document, args.section)
    if not isinstance(selected, dict) or not selected:
        raise ValueError("selected override section must be a non-empty JSON object")
    if any(not isinstance(key, str) or not key for key in selected):
        raise ValueError("config override keys must be non-empty dotted paths")

    forbidden = sorted(set(selected).intersection(_CONFIG_ASSET_PATHS))
    if forbidden:
        raise ValueError(
            "derived configs cannot override frozen asset paths: " f"{forbidden}"
        )
    for dotted in sorted(selected):
        _get_dotted(document, dotted)  # Reject misspellings instead of creating keys.
        _set_dotted(document, dotted, selected[dotted])

    output = Path(args.output).resolve()
    protected_outputs = {override_source.resolve()}
    protected_outputs.update(
        asset.resolve()
        for asset in (
            base_config.profile_path,
            base_config.trace_path,
            base_config.scenario_trace_path,
            base_config.evidence_path,
        )
        if asset is not None
    )
    if output in protected_outputs:
        raise ValueError(
            "derived config output cannot replace an override document or frozen asset: "
            f"{output}"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"derived config already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_assets: dict[str, Path | None] = {
        "profile_path": base_config.profile_path,
        "trace_path": base_config.trace_path,
        "scenario_trace_path": base_config.scenario_trace_path,
        "evidence_path": base_config.evidence_path,
    }
    for name, asset in resolved_assets.items():
        if asset is not None:
            document[name] = _portable_asset_path(asset, output.parent)

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        load_config(temporary)  # Full schema, unit, capacity and compatibility checks.
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(output),
                "config_sha256": sha256_file(output),
                "source_config": str(source),
                "override_document": str(override_source),
                "override_section": args.section,
                "applied_paths": sorted(selected),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _study_experiment_registration(
    *,
    registration_path: str | None,
    registration_scale: str | None,
    registration_regime: str | None,
    registration_family: str | None,
    base_study_root: str | Path,
    environment_seeds: tuple[int, ...],
    load_level: str,
    family_id: str,
    policies: tuple[str, ...],
    baseline: str,
    metrics: tuple[str, ...],
    require_clean: bool,
) -> dict[str, Any]:
    """Bind a numerical study to a committed generator-matrix record."""

    supplied = (
        registration_path is not None,
        registration_scale is not None,
        registration_regime is not None,
        registration_family is not None,
    )
    if require_clean and not all(supplied):
        raise ValueError(
            "formal numerical studies require --experiment-registration, "
            "--registration-scale, --registration-regime and "
            "--registration-family"
        )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "experiment registration path, scale and regime must be supplied together"
        )
    if not any(supplied):
        return {
            "status": "UNREGISTERED_DEVELOPMENT_OVERRIDE",
            "require_clean_registration": False,
        }

    assert registration_path is not None
    assert registration_scale is not None
    assert registration_regime is not None
    assert registration_family is not None
    source, document = _read_strict_json_object(registration_path)
    registration_preflight = source_cleanliness_preflight(
        source, require_clean=require_clean
    )
    scales = document.get("scales")
    if not isinstance(scales, dict) or registration_scale not in scales:
        raise ValueError(f"registration scale is missing: {registration_scale}")
    scale = scales[registration_scale]
    if not isinstance(scale, dict):
        raise ValueError("registration scale must be an object")
    registered_seeds = scale.get("environment_seeds")
    if registered_seeds != list(environment_seeds):
        raise ValueError(
            "environment seeds differ from experiment registration: "
            f"registered={registered_seeds}, requested={list(environment_seeds)}"
        )
    regimes = scale.get("regimes")
    if not isinstance(regimes, dict) or registration_regime not in regimes:
        raise ValueError(f"registration regime is missing: {registration_regime}")
    if load_level != registration_regime:
        raise ValueError(
            "study load_level must equal the registered regime: "
            f"load_level={load_level!r}, regime={registration_regime!r}"
        )
    regime = regimes[registration_regime]
    shared = scale.get("shared_generator_options")
    controls = document.get("scientific_controls")
    if not all(isinstance(item, dict) for item in (regime, shared, controls)):
        raise ValueError("registration controls, shared options and regime must be objects")

    analysis_plan = document.get("analysis_plan")
    if not isinstance(analysis_plan, dict):
        raise ValueError("experiment registration lacks analysis_plan")
    family_fields = {
        "pilot": ("pilot_family_id", "pilot_policies", "pilot_metrics"),
        "primary": ("primary_family_id", "primary_policies", "primary_metrics"),
        "secondary": (
            "secondary_family_id",
            "secondary_registered_policies",
            "secondary_registered_metrics",
        ),
    }
    try:
        family_id_key, policies_key, metrics_key = family_fields[registration_family]
    except KeyError as exc:
        raise ValueError(
            f"unknown registration family: {registration_family!r}"
        ) from exc
    registered_family_id = analysis_plan.get(family_id_key)
    registered_policies = analysis_plan.get(policies_key)
    registered_metrics = analysis_plan.get(metrics_key)
    registered_baseline = analysis_plan.get("baseline")
    analysis_mismatches: dict[str, Any] = {}
    requested_analysis = {
        "family_id": family_id,
        "policies": list(policies),
        "metrics": list(metrics),
        "baseline": baseline,
    }
    registered_analysis = {
        "family_id": registered_family_id,
        "policies": registered_policies,
        "metrics": registered_metrics,
        "baseline": registered_baseline,
    }
    for key, requested in requested_analysis.items():
        registered = registered_analysis[key]
        if canonical_json_bytes(requested) != canonical_json_bytes(registered):
            analysis_mismatches[key] = {
                "registered": registered,
                "requested": requested,
            }
    if analysis_mismatches:
        raise ValueError(
            "study analysis domain differs from experiment registration: "
            f"{analysis_mismatches}"
        )

    base_config_path = (
        Path(base_study_root).resolve() / "configs" / "numerical_default.json"
    )
    _, config_document = _read_strict_json_object(base_config_path)
    base_config = load_config(base_config_path)
    if base_config.evidence_path is None:
        raise ValueError("registered numerical study requires frozen evidence")
    _, evidence = _read_strict_json_object(base_config.evidence_path)
    spec = evidence.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("numerical evidence is missing its generator spec")
    expected_spec = {
        "seed": controls.get("profile_seed"),
        "profile_evaluation_subjects": controls.get("profile_subjects"),
        "test_subjects": controls.get("test_subjects"),
        "scenario_subjects": controls.get("scenario_subjects"),
        "task_count": shared.get("tasks"),
        "horizon_s": shared.get("horizon_s"),
        "arrival_center_s": shared.get("arrival_center_s"),
        "arrival_window_s": regime.get("arrival_window_s"),
        "arrival_jitter_fraction": controls.get("arrival_jitter_fraction"),
        "privacy_threshold": controls.get("privacy_risk_threshold"),
        "preprocessing_failure_mode": controls.get("preprocessing_failure_mode"),
        "local_service_scale": regime.get("local_service_scale"),
        "anon_time_variability_scale": controls.get(
            "anon_time_variability_scale"
        ),
        "output_size_variability_scale": controls.get(
            "output_size_variability_scale"
        ),
    }
    mismatches = {
        key: {"registered": expected, "evidence_spec": spec.get(key)}
        for key, expected in expected_spec.items()
        if canonical_json_bytes(spec.get(key)) != canonical_json_bytes(expected)
    }
    if mismatches:
        raise ValueError(f"base study differs from registered generator spec: {mismatches}")
    override_ref = regime.get("config_overrides_ref")
    if override_ref is None:
        overrides = regime.get("config_overrides")
        if overrides != {}:
            raise ValueError("unreferenced registration overrides must be an empty object")
        overrides = {}
    else:
        if not isinstance(override_ref, str):
            raise ValueError("config_overrides_ref must be a dotted path")
        overrides = _get_dotted(document, override_ref)
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError("registered config override block must be non-empty")
        override_mismatches = {
            key: {
                "registered": value,
                "config": _get_dotted(config_document, key),
            }
            for key, value in sorted(overrides.items())
            if canonical_json_bytes(_get_dotted(config_document, key))
            != canonical_json_bytes(value)
        }
        if override_mismatches:
            raise ValueError(
                "base config differs from registered overrides: "
                f"{override_mismatches}"
            )

    expected_config = build_numerical_config_from_evidence(evidence)
    for key, value in sorted(overrides.items()):
        _get_dotted(expected_config, key)
        _set_dotted(expected_config, key, value)
    expected_config_sha = hashlib.sha256(
        canonical_json_bytes(expected_config)
    ).hexdigest()
    actual_config_sha = hashlib.sha256(
        canonical_json_bytes(config_document)
    ).hexdigest()
    if actual_config_sha != expected_config_sha:
        raise ValueError(
            "base config contains edits outside the registered generator and "
            f"override set: expected={expected_config_sha}, actual={actual_config_sha}"
        )

    validated = {
        "generator_spec": expected_spec,
        "config_overrides": overrides,
        "analysis_domain": requested_analysis,
    }
    record: dict[str, Any] = {
        "status": "VERIFIED",
        "registration_filename": source.name,
        "registration_file_sha256": sha256_file(source),
        "registration_content_sha256": hashlib.sha256(
            canonical_json_bytes(document)
        ).hexdigest(),
        "registration_scale": registration_scale,
        "registration_regime": registration_regime,
        "registered_scale_regimes": sorted(regimes),
        "registration_family": registration_family,
        "registered_family_id": family_id,
        "registered_load_level": load_level,
        "registered_policies": list(policies),
        "registered_metrics": list(metrics),
        "registered_baseline": baseline,
        "registered_environment_seeds": list(environment_seeds),
        "validated_parameters_sha256": hashlib.sha256(
            canonical_json_bytes(validated)
        ).hexdigest(),
        "expected_config_canonical_sha256": expected_config_sha,
        "source_cleanliness_preflight": registration_preflight,
        "record_sha256": "",
    }
    record["record_sha256"] = canonical_document_sha256(
        record, "record_sha256"
    )
    return record


def command_source_preflight(args: argparse.Namespace) -> int:
    """Report whether package source is bound to the current Git commit."""

    report = source_cleanliness_preflight(
        require_clean=not args.allow_dirty_source
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _sweep_experiment_registration(
    *,
    sensitivity_path: str | None,
    registration_factor: str | None,
    experiment_path: str | None,
    config_document: dict[str, Any],
    config: SimulationConfig,
    grid_document: dict[str, Any],
    policy: str,
    require_clean: bool,
) -> dict[str, Any]:
    """Bind a formal one-factor sweep to committed sensitivity records."""

    supplied = (
        sensitivity_path is not None,
        registration_factor is not None,
        experiment_path is not None,
    )
    if require_clean and not all(supplied):
        raise ValueError(
            "formal sweeps require --sensitivity-registration, "
            "--registration-factor and --experiment-registration"
        )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "sensitivity registration, factor and experiment registration "
            "must be supplied together"
        )
    if not any(supplied):
        return {
            "status": "UNREGISTERED_DEVELOPMENT_OVERRIDE",
            "require_clean_registration": False,
        }

    assert sensitivity_path is not None
    assert registration_factor is not None
    assert experiment_path is not None
    sensitivity_source, sensitivity = _read_strict_json_object(sensitivity_path)
    experiment_source, experiment = _read_strict_json_object(experiment_path)
    sensitivity_preflight = source_cleanliness_preflight(
        sensitivity_source, require_clean=require_clean
    )
    experiment_preflight = source_cleanliness_preflight(
        experiment_source, require_clean=require_clean
    )
    factors = sensitivity.get("factors")
    if not isinstance(factors, dict) or registration_factor not in factors:
        raise ValueError(f"sensitivity factor is not registered: {registration_factor}")
    factor = factors[registration_factor]
    if not isinstance(factor, dict):
        raise ValueError("registered sensitivity factor must be an object")
    registered_policy = factor.get("policy")
    if not isinstance(registered_policy, str) or not registered_policy:
        raise ValueError("registered sensitivity factor must declare its policy")
    if policy != registered_policy:
        raise ValueError(
            "sweep policy differs from the registered sensitivity factor: "
            f"registered={registered_policy}, requested={policy}"
        )
    application = factor.get("application")
    if application == "config_sweep":
        factor_paths = [factor.get("path")]
        factor_values: Any = factor.get("values")
        expected_grid = {factor_paths[0]: factor_values}
        registration_shape_valid = (
            isinstance(factor_paths[0], str) and isinstance(factor_values, list)
        )
    elif application == "paired_config_override":
        factor_paths = factor.get("paths")
        paired_values = factor.get("paired_values_gpu_s")
        registration_shape_valid = (
            isinstance(factor_paths, list)
            and len(factor_paths) >= 2
            and all(isinstance(path, str) and path for path in factor_paths)
            and isinstance(paired_values, list)
            and len(paired_values) >= 2
            and all(
                isinstance(values, list) and len(values) == len(factor_paths)
                for values in paired_values
            )
        )
        factor_values = paired_values
        expected_grid = (
            {
                "$paired": [
                    dict(zip(factor_paths, values, strict=True))
                    for values in paired_values
                ]
            }
            if registration_shape_valid
            else {}
        )
    else:
        raise ValueError(
            "generic sweep supports only config_sweep or paired_config_override factors"
        )
    if (
        not registration_shape_valid
        or canonical_json_bytes(grid_document) != canonical_json_bytes(expected_grid)
    ):
        raise ValueError(
            "sweep grid differs from the registered one-factor levels: "
            f"factor={registration_factor}"
        )

    reference = sensitivity.get("reference")
    scales = experiment.get("scales")
    controls = experiment.get("scientific_controls")
    if not all(isinstance(item, dict) for item in (reference, scales, controls)):
        raise ValueError("sensitivity or experiment reference records are malformed")
    experiment_content_sha = hashlib.sha256(
        canonical_json_bytes(experiment)
    ).hexdigest()
    if reference.get("experiment_registration_content_sha256") != experiment_content_sha:
        raise ValueError(
            "sensitivity plan is not bound to this experiment registration content"
        )
    scale_name = reference.get("scale")
    regime_name = reference.get("regime")
    scale = scales.get(scale_name)
    if not isinstance(scale, dict):
        raise ValueError(f"registered sensitivity scale is missing: {scale_name}")
    regimes = scale.get("regimes")
    shared = scale.get("shared_generator_options")
    regime = regimes.get(regime_name) if isinstance(regimes, dict) else None
    if not isinstance(shared, dict) or not isinstance(regime, dict):
        raise ValueError(f"registered sensitivity regime is missing: {regime_name}")

    if config.evidence_path is None:
        raise ValueError("registered sensitivity sweep requires frozen evidence")
    _, evidence = _read_strict_json_object(config.evidence_path)
    spec = evidence.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("numerical evidence is missing its generator spec")
    expected_spec = {
        "seed": controls.get("profile_seed"),
        "profile_evaluation_subjects": controls.get("profile_subjects"),
        "test_subjects": controls.get("test_subjects"),
        "scenario_subjects": controls.get("scenario_subjects"),
        "task_count": shared.get("tasks"),
        "horizon_s": shared.get("horizon_s"),
        "arrival_center_s": shared.get("arrival_center_s"),
        "arrival_window_s": regime.get("arrival_window_s"),
        "arrival_jitter_fraction": controls.get("arrival_jitter_fraction"),
        "privacy_threshold": controls.get("privacy_risk_threshold"),
        "preprocessing_failure_mode": controls.get("preprocessing_failure_mode"),
        "local_service_scale": regime.get("local_service_scale"),
        "anon_time_variability_scale": controls.get("anon_time_variability_scale"),
        "output_size_variability_scale": controls.get(
            "output_size_variability_scale"
        ),
    }
    mismatches = {
        key: {"registered": value, "evidence_spec": spec.get(key)}
        for key, value in expected_spec.items()
        if canonical_json_bytes(value) != canonical_json_bytes(spec.get(key))
    }
    if mismatches:
        raise ValueError(
            f"sensitivity base differs from registered generator spec: {mismatches}"
        )

    expected_config = build_numerical_config_from_evidence(evidence)
    replicated_streams = (
        "environment",
        "arrivals",
        "mobility",
        "wireless",
        "vehicle",
        "rsu",
        "fault",
    )
    for stream in replicated_streams:
        expected_config["seeds"][stream] = config_document["seeds"][stream]
    expected_config_sha = hashlib.sha256(
        canonical_json_bytes(expected_config)
    ).hexdigest()
    actual_config_sha = hashlib.sha256(
        canonical_json_bytes(config_document)
    ).hexdigest()
    if expected_config_sha != actual_config_sha:
        raise ValueError(
            "sensitivity base config contains edits outside registered environment "
            f"replication streams: expected={expected_config_sha}, actual={actual_config_sha}"
        )

    reference_values = {
        "privacy.risk_threshold": reference.get("privacy_risk_threshold"),
        "controller.lyapunov_v": reference.get("controller.lyapunov_v"),
        "controller.horizon_events": reference.get("controller.horizon_events"),
        "controller.scenarios": reference.get("controller.scenarios"),
    }
    for dotted, expected in reference_values.items():
        if canonical_json_bytes(_get_dotted(config_document, dotted)) != canonical_json_bytes(
            expected
        ):
            raise ValueError(f"sensitivity reference config mismatch at {dotted}")
    rsu_reference = reference.get("rsu_workload_capacity_gpu_s")
    actual_rsu_reference = [
        _get_dotted(config_document, f"rsus.{index}.workload_capacity_gpu_s")
        for index in range(len(config_document.get("rsus", [])))
    ]
    if canonical_json_bytes(actual_rsu_reference) != canonical_json_bytes(rsu_reference):
        raise ValueError("sensitivity reference RSU workload capacities differ")

    _, trace_document = _read_strict_json_object(config.trace_path)
    environment_seed = trace_document.get("seed")
    rules = sensitivity.get("rules")
    registered_seeds = rules.get("environment_seeds") if isinstance(rules, dict) else None
    if not isinstance(environment_seed, int) or environment_seed not in (
        registered_seeds if isinstance(registered_seeds, list) else []
    ):
        raise ValueError(
            "sensitivity evaluation trace seed is outside the registered environment family"
        )

    record: dict[str, Any] = {
        "status": "VERIFIED",
        "registration_factor": registration_factor,
        "registered_policy": registered_policy,
        "factor_paths": factor_paths,
        "factor_values": factor_values,
        "reference_scale": scale_name,
        "reference_regime": regime_name,
        "environment_seed": environment_seed,
        "sensitivity_file_sha256": sha256_file(sensitivity_source),
        "sensitivity_content_sha256": hashlib.sha256(
            canonical_json_bytes(sensitivity)
        ).hexdigest(),
        "experiment_file_sha256": sha256_file(experiment_source),
        "experiment_content_sha256": experiment_content_sha,
        "expected_config_canonical_sha256": expected_config_sha,
        "sensitivity_source_preflight": sensitivity_preflight,
        "experiment_source_preflight": experiment_preflight,
        "record_sha256": "",
    }
    record["record_sha256"] = canonical_document_sha256(record, "record_sha256")
    return record


def _expand_sweep_coordinates(
    base: dict[str, Any], grid: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Expand Cartesian or explicitly paired sweep coordinates deterministically."""

    if set(grid) == {"$paired"}:
        raw_cases = grid["$paired"]
        if not isinstance(raw_cases, list) or len(raw_cases) < 2:
            raise ValueError("paired sweep requires at least two coordinate objects")
        if any(not isinstance(case, dict) or not case for case in raw_cases):
            raise ValueError("paired sweep cases must be non-empty objects")
        keys = sorted(raw_cases[0])
        if "$paired" in keys or any(sorted(case) != keys for case in raw_cases):
            raise ValueError("paired sweep cases must contain the same config paths")
        for key in keys:
            _get_dotted(base, key)
        coordinates = [dict(sorted(case.items())) for case in raw_cases]
        if len({canonical_json_bytes(case) for case in coordinates}) < 2:
            raise ValueError("paired sweep requires at least two distinct cases")
        return keys, coordinates
    if "$paired" in grid:
        raise ValueError("$paired cannot be combined with Cartesian sweep coordinates")
    if (
        not grid
        or any(not isinstance(values, list) or not values for values in grid.values())
    ):
        raise ValueError(
            "sweep grid must be a non-empty object of non-empty value arrays"
        )
    keys = sorted(grid)
    for key in keys:
        _get_dotted(base, key)
        distinct_levels = {canonical_json_bytes(value) for value in grid[key]}
        if len(distinct_levels) < 2:
            raise ValueError(
                "every sweep coordinate requires at least two distinct levels: "
                f"{key}"
            )
    return keys, [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def command_sweep(args: argparse.Namespace) -> int:
    require_clean_source = not args.allow_dirty_source
    sweep_source_preflight = source_cleanliness_preflight(
        require_clean=require_clean_source
    )
    sweep_source_identity = (
        _source_identity(sweep_source_preflight) if require_clean_source else None
    )
    config_path, base = _read_strict_json_object(args.config)
    grid_path, grid = _read_strict_json_object(args.grid)
    # Resolve input assets before moving patched configs into temporary dirs.
    base_config = load_config(config_path)
    sweep_base_frozen_inputs = _capture_frozen_input_assets(base_config)
    sweep_control_files: list[tuple[Path, str]] = [
        (config_path, sha256_file(config_path)),
        (grid_path, sha256_file(grid_path)),
    ]
    for optional_path in (
        args.sensitivity_registration,
        args.experiment_registration,
    ):
        if optional_path is not None:
            resolved = Path(optional_path).resolve()
            sweep_control_files.append((resolved, sha256_file(resolved)))
    sweep_batch_identity: dict[str, Any] = {
        "schema_version": "1.0",
        "control_files": [
            {"path": str(path), "raw_sha256": digest}
            for path, digest in sweep_control_files
        ],
        "frozen_input_assets": sweep_base_frozen_inputs,
    }
    sweep_batch_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(sweep_batch_identity)
    ).hexdigest()

    def require_unchanged_sweep_base(context: str) -> None:
        for path, digest in sweep_control_files:
            _require_same_file_bytes(path, digest, context=context)
        _require_same_frozen_input_assets(
            base_config, sweep_base_frozen_inputs, context=context
        )

    sweep_policy = args.policy or base_config.controller.policy
    sweep_registration = _sweep_experiment_registration(
        sensitivity_path=args.sensitivity_registration,
        registration_factor=args.registration_factor,
        experiment_path=args.experiment_registration,
        config_document=base,
        config=base_config,
        grid_document=grid,
        policy=sweep_policy,
        require_clean=require_clean_source,
    )
    require_unchanged_sweep_base("during sweep registration")
    base["profile_path"] = str(base_config.profile_path)
    base["trace_path"] = str(base_config.trace_path)
    base["scenario_trace_path"] = str(base_config.scenario_trace_path)
    if base_config.evidence_path is not None:
        base["evidence_path"] = str(base_config.evidence_path)
    keys, sweep_coordinates = _expand_sweep_coordinates(base, grid)
    case_count = len(sweep_coordinates)
    root = _prepare_batch_root(
        Path(args.output_root),
        leaf_directories=[Path(f"case-{index:04d}") for index in range(case_count)],
        index_files={
            "sweep.json",
            "sweep_diagnostics.json",
            "sweep.in_progress.json",
        },
        overwrite=args.overwrite,
    )
    marker = root / "sweep.in_progress.json"
    _write_new_json(
        marker,
        {
            "schema_version": "1.0",
            "status": "IN_PROGRESS",
            "config_sha256": sha256_file(config_path),
            "grid_sha256": sha256_file(grid_path),
            "case_count": case_count,
            "policy": sweep_policy,
            "batch_base_identity_sha256": sweep_batch_identity_sha256,
            "source_cleanliness_preflight": sweep_source_preflight,
            "sensitivity_registration": sweep_registration,
        },
        overwrite=True,
    )
    index_paths = (root / "sweep.json", root / "sweep_diagnostics.json")
    for stale_index in index_paths:
        stale_index.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix="privacy-edge-sweep-") as temp_dir:
            for index, coordinates in enumerate(sweep_coordinates):
                require_unchanged_sweep_base(f"before sweep case {index}")
                document = json.loads(json.dumps(base))
                for key, value in coordinates.items():
                    _set_dotted(document, key, value)
                patched_path = Path(temp_dir) / f"config-{index:04d}.json"
                patched_path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                config = load_config(patched_path)
                policy = sweep_policy
                output = root / f"case-{index:04d}"
                result, artifacts, case_manifest = _execute(
                    config,
                    policy,
                    output,
                    overwrite=args.overwrite,
                    run_metadata={
                        "case": index,
                        "parameters": coordinates,
                        "sensitivity_registration_record_sha256": (
                            sweep_registration.get("record_sha256")
                        ),
                        "batch_base_identity_sha256": sweep_batch_identity_sha256,
                    },
                    require_clean_source=require_clean_source,
                    expected_source_identity=sweep_source_identity,
                )
                require_unchanged_sweep_base(f"after sweep case {index}")
                manifest_frozen_inputs = case_manifest.get("frozen_input_assets")
                if not isinstance(manifest_frozen_inputs, dict):
                    raise RuntimeError("sweep case manifest lacks frozen input assets")
                _require_frozen_input_content_match(
                    manifest_frozen_inputs,
                    sweep_base_frozen_inputs,
                    roles=(
                        "profile",
                        "evaluation_trace",
                        "scenario_trace",
                        "evidence",
                    ),
                    context=f"in sweep case {index}",
                )
                row = _summary_line(result, artifacts, output)
                row["case"] = index
                row["parameters"] = coordinates
                summary = json.loads(
                    (output / "summary.json").read_text(encoding="utf-8")
                )
                row["mechanism_metrics"] = _mechanism_snapshot(summary)
                row["output_relative"] = output.relative_to(root).as_posix()
                manifest_path, manifest_document = _read_strict_json_object(
                    output / "manifest.json"
                )
                row["manifest_sha256"] = _require_document_self_hash(
                    manifest_document,
                    field="manifest_sha256",
                    path=manifest_path,
                    context="sweep case manifest",
                )
                rows.append(row)
        require_unchanged_sweep_base("before sweep index publication")
        if sweep_source_identity is not None:
            _require_same_source_identity(
                source_cleanliness_preflight(require_clean=True),
                sweep_source_identity,
                context="before formal sweep index publication",
            )
        diagnostics = _sweep_mechanism_diagnostics(
            rows,
            coordinate_keys=keys,
            paired_coordinates=set(grid) == {"$paired"},
        )
        diagnostics["sweep_rows_sha256"] = hashlib.sha256(
            canonical_json_bytes(rows)
        ).hexdigest()
        diagnostics["sensitivity_registration"] = sweep_registration
        diagnostics["batch_base_identity"] = sweep_batch_identity
        diagnostics["batch_base_identity_sha256"] = sweep_batch_identity_sha256
        diagnostics["report_sha256"] = canonical_document_sha256(
            diagnostics, "report_sha256"
        )
        _write_new_json(root / "sweep.json", rows, overwrite=True)
        _write_new_json(
            root / "sweep_diagnostics.json", diagnostics, overwrite=True
        )
        completed = True
    finally:
        if completed:
            marker.unlink()
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0


def _incomplete_sweep_markers(paths: list[Path]) -> list[Path]:
    """Find an interrupted sweep below or above any supplied result path."""

    markers: set[Path] = set()
    for raw in paths:
        resolved = raw.resolve()
        probe = resolved if resolved.is_dir() else resolved.parent
        for ancestor in (probe, *probe.parents):
            marker = ancestor / "sweep.in_progress.json"
            if marker.is_file():
                markers.add(marker.resolve())
        if probe.is_dir():
            markers.update(
                marker.resolve()
                for marker in probe.rglob("sweep.in_progress.json")
            )
    return sorted(markers, key=lambda path: path.as_posix())


def _canonical_sweep_case_index(name: str) -> int | None:
    """Return the index encoded by a canonical ``case-NNNN`` directory name."""

    if not name.startswith("case-"):
        return None
    suffix = name.removeprefix("case-")
    if not suffix.isdigit():
        return None
    index = int(suffix)
    return index if name == f"case-{index:04d}" else None


def _completed_sweep_roots(paths: list[Path]) -> list[Path]:
    """Discover completed-sweep roots below or above supplied consumer paths.

    A canonical case directory is deliberately treated as evidence of a sweep even
    when an index has been deleted.  This makes passing a surviving case directory
    unable to bypass whole-sweep validation.
    """

    roots: set[Path] = set()
    index_names = ("sweep.json", "sweep_diagnostics.json")
    for raw in paths:
        resolved = raw.resolve()
        probe = resolved if resolved.is_dir() else resolved.parent
        for ancestor in (probe, *probe.parents):
            if any((ancestor / name).exists() for name in index_names):
                roots.add(ancestor.resolve())
            if _canonical_sweep_case_index(ancestor.name) is not None:
                roots.add(ancestor.parent.resolve())
        if not probe.is_dir():
            continue
        for index_name in index_names:
            roots.update(
                path.parent.resolve() for path in probe.rglob(index_name) if path.is_file()
            )
        roots.update(
            path.parent.resolve()
            for path in probe.rglob("case-*")
            if path.is_dir() and _canonical_sweep_case_index(path.name) is not None
        )
    return sorted(roots, key=lambda path: path.as_posix())


def _require_document_self_hash(
    document: dict[str, Any], *, field: str, path: Path, context: str
) -> str:
    expected = document.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{context} self-hash is missing or malformed: {path}")
    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError(f"{context} self-hash is malformed: {path}") from exc
    actual = canonical_document_sha256(document, field)
    if expected != actual:
        raise ValueError(f"{context} self-hash mismatch: {path}")
    return expected


def _validate_completed_sweep_root(root: Path) -> None:
    """Validate a published sweep as one complete, content-bound result set."""

    prefix = f"completed sweep integrity failure at {root}"
    marker = root / "sweep.in_progress.json"
    if marker.is_file():
        raise RuntimeError(f"{prefix}: in-progress marker is still present")

    rows_path = root / "sweep.json"
    diagnostics_path = root / "sweep_diagnostics.json"
    missing_indexes = [
        str(path) for path in (rows_path, diagnostics_path) if not path.is_file()
    ]
    if missing_indexes:
        raise FileNotFoundError(f"{prefix}: missing sweep index files {missing_indexes}")

    _, rows_document = _read_strict_json_value(rows_path)
    if not isinstance(rows_document, list) or not rows_document:
        raise ValueError(f"{prefix}: sweep.json must be a non-empty array")
    rows = rows_document
    _, diagnostics = _read_strict_json_object(diagnostics_path)
    _require_document_self_hash(
        diagnostics,
        field="report_sha256",
        path=diagnostics_path,
        context="completed sweep diagnostics",
    )

    expected_rows_hash = diagnostics.get("sweep_rows_sha256")
    actual_rows_hash = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    if expected_rows_hash != actual_rows_hash:
        raise ValueError(f"{prefix}: sweep rows checksum mismatch")
    declared_case_count = diagnostics.get("case_count")
    if type(declared_case_count) is not int or declared_case_count != len(rows):
        raise ValueError(
            f"{prefix}: diagnostics case_count does not match sweep row count"
        )
    sensitivity_registration = diagnostics.get("sensitivity_registration")
    if not isinstance(sensitivity_registration, dict):
        raise ValueError(f"{prefix}: sensitivity registration record is missing")
    expected_registration_hash = sensitivity_registration.get("record_sha256")
    if sensitivity_registration.get("status") == "VERIFIED":
        _require_document_self_hash(
            sensitivity_registration,
            field="record_sha256",
            path=diagnostics_path,
            context="completed sweep sensitivity registration",
        )

    expected_case_names = {f"case-{index:04d}" for index in range(len(rows))}
    actual_case_names = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("case-")
    }
    if actual_case_names != expected_case_names:
        raise ValueError(
            f"{prefix}: case directories differ from the indexed cases; "
            f"missing={sorted(expected_case_names - actual_case_names)}, "
            f"unexpected={sorted(actual_case_names - expected_case_names)}"
        )

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{prefix}: sweep row {index} must be an object")
        if type(row.get("case")) is not int or row["case"] != index:
            raise ValueError(
                f"{prefix}: sweep cases must be ordered and contiguous from zero"
            )
        expected_relative = f"case-{index:04d}"
        if row.get("output_relative") != expected_relative:
            raise ValueError(
                f"{prefix}: row {index} output_relative must be {expected_relative!r}"
            )
        parameters = row.get("parameters")
        policy = row.get("policy")
        core_digest = row.get("core_digest")
        if not isinstance(parameters, dict):
            raise ValueError(f"{prefix}: row {index} parameters must be an object")
        if not isinstance(policy, str) or not policy:
            raise ValueError(f"{prefix}: row {index} policy is missing")
        if not isinstance(core_digest, str) or not core_digest:
            raise ValueError(f"{prefix}: row {index} core_digest is missing")

        case_directory = root / expected_relative
        if case_directory.is_symlink():
            raise ValueError(f"{prefix}: case directory may not be a symlink")
        manifest_path = case_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{prefix}: case {index} is missing manifest.json"
            )
        _, manifest = _read_strict_json_object(manifest_path)
        manifest_hash = _require_document_self_hash(
            manifest,
            field="manifest_sha256",
            path=manifest_path,
            context="completed sweep case manifest",
        )
        indexed_manifest_hash = row.get("manifest_sha256")
        if "manifest_sha256" in row and indexed_manifest_hash != manifest_hash:
            raise ValueError(f"{prefix}: row {index} manifest checksum mismatch")
        if manifest.get("core_digest") != core_digest:
            raise ValueError(f"{prefix}: row {index} core_digest mismatch")
        run_metadata = manifest.get("run_metadata")
        if not isinstance(run_metadata, dict):
            raise ValueError(f"{prefix}: case {index} run_metadata is missing")
        if run_metadata.get("case") != index:
            raise ValueError(f"{prefix}: case {index} manifest case binding mismatch")
        if run_metadata.get("parameters") != parameters:
            raise ValueError(
                f"{prefix}: case {index} manifest parameter binding mismatch"
            )
        if run_metadata.get("policy") != policy:
            raise ValueError(f"{prefix}: case {index} manifest policy binding mismatch")
        if (
            run_metadata.get("sensitivity_registration_record_sha256")
            != expected_registration_hash
        ):
            raise ValueError(
                f"{prefix}: case {index} sensitivity registration binding mismatch"
            )


def _validate_completed_sweep_inputs(paths: list[Path]) -> None:
    for root in _completed_sweep_roots(paths):
        _validate_completed_sweep_root(root)


def command_aggregate(args: argparse.Namespace) -> int:
    roots = [Path(item) for item in args.inputs]
    incomplete = _incomplete_sweep_markers(roots)
    if incomplete:
        raise RuntimeError(
            "aggregation refuses incomplete sweep roots: "
            f"{[str(path) for path in incomplete]}"
        )
    _validate_completed_sweep_inputs(roots)
    summaries: list[dict[str, Any]] = []
    seen_summaries: set[Path] = set()
    for root in roots:
        paths = (
            [root]
            if root.name == "summary.json"
            else sorted(root.rglob("summary.json"))
        )
        for path in paths:
            resolved_summary = path.resolve()
            if resolved_summary in seen_summaries:
                continue
            seen_summaries.add(resolved_summary)
            manifest_path = path.with_name("manifest.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"verified aggregation requires manifest.json next to every summary: {path}"
                )
            manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_document, dict):
                raise ValueError(f"manifest root must be an object: {manifest_path}")
            expected_manifest_hash = manifest_document.get("manifest_sha256")
            if not isinstance(expected_manifest_hash, str):
                raise ValueError(f"manifest checksum is missing: {manifest_path}")
            manifest = dict(manifest_document)
            manifest.pop("manifest_sha256", None)
            actual_manifest_hash = hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest()
            if expected_manifest_hash != actual_manifest_hash:
                raise ValueError(f"manifest checksum mismatch: {manifest_path}")

            outputs = manifest.get("outputs")
            output_files = outputs.get("files") if isinstance(outputs, dict) else None
            summary_record = (
                output_files.get("summary.json")
                if isinstance(output_files, dict)
                else None
            )
            if not isinstance(summary_record, dict):
                raise ValueError(
                    f"manifest summary.json output record is missing: {manifest_path}"
                )
            expected_summary_hash = summary_record.get("sha256")
            if not isinstance(expected_summary_hash, str) or not expected_summary_hash:
                raise ValueError(
                    f"manifest summary.json checksum is missing: {manifest_path}"
                )
            actual_summary_hash = sha256_file(path)
            if actual_summary_hash != expected_summary_hash:
                raise ValueError(f"summary checksum mismatch: {path}")

            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError(f"summary root must be an object: {path}")
            row["summary_path"] = str(resolved_summary)
            row["manifest_sha256"] = expected_manifest_hash
            row["summary_sha256"] = expected_summary_hash
            run_metadata = manifest.get("run_metadata")
            run_metadata = run_metadata if isinstance(run_metadata, dict) else {}
            controller = manifest.get("controller")
            controller = controller if isinstance(controller, dict) else {}
            row["policy"] = run_metadata.get("policy") or controller.get("policy")
            row["core_digest"] = manifest.get("core_digest")
            provenance = manifest.get("data_provenance", {})
            provenance = provenance if isinstance(provenance, dict) else {}
            row["result_label"] = provenance.get("result_label")
            row["formal_experiment_eligible"] = provenance.get(
                "formal_experiment_eligible"
            )
            code_version = manifest.get("code_version")
            configuration = manifest.get("configuration")
            versions = manifest.get("versions")
            trace_identity = manifest.get("trace_identity")
            scenario_identity = manifest.get("scenario_trace_identity")
            row["code_version"] = (
                code_version.get("value") if isinstance(code_version, dict) else None
            )
            row["configuration_sha256"] = (
                configuration.get("canonical_sha256")
                if isinstance(configuration, dict)
                else None
            )
            row["profile_hash"] = (
                versions.get("profile_hash") if isinstance(versions, dict) else None
            )
            row["evaluation_trace_hash"] = (
                trace_identity.get("trace_hash")
                if isinstance(trace_identity, dict)
                else None
            )
            row["scenario_trace_hash"] = (
                scenario_identity.get("trace_hash")
                if isinstance(scenario_identity, dict)
                else None
            )
            if "base_seed" in run_metadata:
                row["base_seed"] = run_metadata["base_seed"]
            if "case" in run_metadata:
                row["case"] = run_metadata["case"]
            parameters = run_metadata.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise ValueError(
                    f"manifest run_metadata.parameters must be an object: {manifest_path}"
                )
            for key, value in sorted((parameters or {}).items()):
                row[f"parameter.{key}"] = value
            seeds = manifest.get("seeds")
            for key, value in sorted(seeds.items() if isinstance(seeds, dict) else ()):
                row[f"seed_stream.{key}"] = value
            # Reject non-finite or otherwise non-strict summary content before
            # any aggregate output file is created.
            canonical_json_bytes(row)
            summaries.append(row)
    if not summaries:
        raise FileNotFoundError("no summary.json files found")
    requested = Path(args.output)
    csv_path = (
        requested
        if requested.suffix.lower() == ".csv"
        else requested.with_suffix(".csv")
    )
    json_path = (
        requested
        if requested.suffix.lower() == ".json"
        else requested.with_suffix(".json")
    )
    write_parquet = args.parquet or requested.suffix.lower() == ".parquet"
    parquet_path = (
        requested
        if requested.suffix.lower() == ".parquet"
        else requested.with_suffix(".parquet")
    )
    if csv_path.resolve() == json_path.resolve():
        raise ValueError("aggregate CSV and JSON paths must be distinct")
    output_paths = [csv_path, json_path, *([parquet_path] if write_parquet else [])]
    collisions = [path for path in output_paths if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            f"aggregate outputs already exist: {[str(path) for path in collisions]}"
        )
    fields = sorted(
        {
            key
            for row in summaries
            for key in row
            if not isinstance(row[key], (dict, list))
        }
    )
    flat_rows = [{key: row.get(key) for key in fields} for row in summaries]
    parquet_table: Any | None = None
    parquet_module: Any | None = None
    if write_parquet:
        # Import and construct the table before writing CSV/JSON so a missing
        # optional dependency or an Arrow schema error cannot leave a partial
        # aggregate beside an exception.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet aggregation requires the optional 'parquet' dependency"
            ) from exc
        parquet_table = pa.Table.from_pylist(_parquet_safe_rows(flat_rows))
        parquet_module = pq

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    json_path.write_text(
        json.dumps(
            summaries, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    report = {"runs": len(summaries), "csv": str(csv_path), "json": str(json_path)}
    if write_parquet:
        assert parquet_module is not None and parquet_table is not None
        parquet_module.write_table(parquet_table, parquet_path)
        report["parquet"] = str(parquet_path)
    print(json.dumps(report, sort_keys=True))
    return 0


def _parquet_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve unsigned RNG identities without overflowing Arrow int64.

    Seed streams are derived from the full unsigned 64-bit SHA-256 prefix.
    Arrow's inferred integer type is signed int64, so if any value in a column
    falls outside that range the entire column is emitted as decimal text.
    CSV and JSON retain their original numeric representation.
    """

    if not rows:
        return []
    fields = {key for row in rows for key in row}
    text_fields = {
        field
        for field in fields
        if any(
            isinstance(row.get(field), int)
            and not isinstance(row.get(field), bool)
            and not (-(2**63) <= row[field] < 2**63)
            for row in rows
        )
    }
    return [
        {
            key: str(value) if key in text_fields and value is not None else value
            for key, value in row.items()
        }
        for row in rows
    ]


def command_smoke(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"smoke output root must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    main_policy = args.policy or "safe_lyapunov_h1"
    first, first_artifacts, _ = _execute(
        config, main_policy, root / "main-a", overwrite=False
    )
    second, second_artifacts, _ = _execute(
        config, main_policy, root / "main-b", overwrite=False
    )
    baseline, baseline_artifacts, _ = _execute(
        config, "all_local", root / "baseline-all-local", overwrite=False
    )
    deterministic = first_artifacts.core_digest == second_artifacts.core_digest
    report = {
        "main_policy": _summary_line(first, first_artifacts, root / "main-a"),
        "repeat_policy": _summary_line(second, second_artifacts, root / "main-b"),
        "baseline": _summary_line(
            baseline, baseline_artifacts, root / "baseline-all-local"
        ),
        "same_seed_core_digest_equal": deterministic,
        "engineering_smoke_only": True,
    }
    (root / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if deterministic else 2


def command_compare(args: argparse.Namespace) -> int:
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("statistical records file must contain a JSON array")
    result = analyze_paired_strategies(
        records,
        baseline_strategy=args.baseline,
        metric_name=args.metric,
        statistical_seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        sign_flip_permutations=args.permutations,
        confidence_level=args.confidence,
    )
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"statistical output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["statistics_manifest"], ensure_ascii=False, sort_keys=True))
    return 0


def _write_new_json(path: str | Path, document: Any, *, overwrite: bool) -> Path:
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return target


def _verify_run_artifacts(
    root: Path,
    *,
    required_filenames: tuple[str, ...],
    require_formal_provenance: bool = True,
) -> dict[str, Any]:
    """Verify a run manifest and the raw files consumed by an offline audit."""

    manifest_path = root / "manifest.json"
    _, manifest_document = _read_strict_json_object(manifest_path)
    expected_manifest_hash = manifest_document.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str) or not expected_manifest_hash:
        raise ValueError(f"manifest checksum is missing: {manifest_path}")
    material = dict(manifest_document)
    material.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(f"manifest checksum mismatch: {manifest_path}")

    invariants = manifest_document.get("invariants")
    source_preflight = manifest_document.get("source_cleanliness_preflight")
    if require_formal_provenance:
        if (
            not isinstance(invariants, dict)
            or invariants.get("passed") is not True
            or invariants.get("failure_count") != 0
        ):
            raise ValueError(
                f"formal audit requires passed zero-failure invariants: {manifest_path}"
            )
        if (
            not isinstance(source_preflight, dict)
            or source_preflight.get("require_clean_source") is not True
            or source_preflight.get("requirement_status") != "passed"
            or source_preflight.get("source_commit_reproducible") is not True
            or source_preflight.get("source_git_dirty") is not False
            or not isinstance(source_preflight.get("git_commit"), str)
            or not source_preflight["git_commit"]
        ):
            raise ValueError(
                f"formal audit requires clean committed run source: {manifest_path}"
            )

    outputs = manifest_document.get("outputs")
    output_files = outputs.get("files") if isinstance(outputs, dict) else None
    if not isinstance(output_files, dict):
        raise ValueError(f"manifest output records are missing: {manifest_path}")
    checksums: dict[str, str] = {}
    for filename in required_filenames:
        record = output_files.get(filename)
        if not isinstance(record, dict):
            raise ValueError(
                f"manifest output record is missing for {filename}: {manifest_path}"
            )
        expected = record.get("sha256")
        if not isinstance(expected, str) or not expected:
            raise ValueError(
                f"manifest output checksum is missing for {filename}: {manifest_path}"
            )
        artifact = root / filename
        actual = sha256_file(artifact)
        if actual != expected:
            raise ValueError(f"manifest output checksum mismatch: {artifact}")
        expected_size = record.get("size_bytes")
        if isinstance(expected_size, int) and artifact.stat().st_size != expected_size:
            raise ValueError(f"manifest output size mismatch: {artifact}")
        checksums[filename] = actual
    return {
        "manifest_sha256": expected_manifest_hash,
        "input_files_sha256": checksums,
        "invariants": {
            "passed": invariants.get("passed") if isinstance(invariants, dict) else None,
            "failure_count": (
                invariants.get("failure_count") if isinstance(invariants, dict) else None
            ),
        },
        "source_cleanliness": {
            "require_clean_source": (
                source_preflight.get("require_clean_source")
                if isinstance(source_preflight, dict)
                else None
            ),
            "requirement_status": (
                source_preflight.get("requirement_status")
                if isinstance(source_preflight, dict)
                else None
            ),
            "source_commit_reproducible": (
                source_preflight.get("source_commit_reproducible")
                if isinstance(source_preflight, dict)
                else None
            ),
            "git_commit": (
                source_preflight.get("git_commit")
                if isinstance(source_preflight, dict)
                else None
            ),
        },
        "verification_status": "VERIFIED",
    }


def _verify_hard_mask_provenance(actions_path: Path) -> dict[str, Any]:
    """Verify a formal hard-mask audit input against its run manifest."""

    root = actions_path.resolve().parent
    provenance = _verify_run_artifacts(
        root, required_filenames=(actions_path.name,)
    )
    manifest_path = root / "manifest.json"
    _, manifest = _read_strict_json_object(manifest_path)
    code_version = manifest.get("code_version")
    if not isinstance(code_version, dict):
        raise ValueError(f"manifest code_version is missing: {manifest_path}")
    source_status = code_version.get("source_git_status")
    source_clean = (
        code_version.get("source_commit_reproducible") is True
        and code_version.get("source_git_dirty") is False
        and isinstance(source_status, list)
        and not source_status
    )
    if not source_clean:
        raise ValueError(
            f"manifest does not identify a clean committed source tree: {manifest_path}"
        )
    data_provenance = manifest.get("data_provenance")
    if (
        not isinstance(data_provenance, dict)
        or data_provenance.get("source_commit_reproducible") is not True
    ):
        raise ValueError(
            f"manifest data provenance does not confirm clean source: {manifest_path}"
        )
    invariants = manifest.get("invariants")
    invariants_valid = (
        isinstance(invariants, dict)
        and invariants.get("passed") is True
        and invariants.get("status") == "passed"
        and isinstance(invariants.get("check_count"), int)
        and not isinstance(invariants.get("check_count"), bool)
        and invariants["check_count"] > 0
        and isinstance(invariants.get("failure_count"), int)
        and not isinstance(invariants.get("failure_count"), bool)
        and invariants.get("failure_count") == 0
        and invariants.get("failures") == []
    )
    if not invariants_valid:
        raise ValueError(
            f"manifest does not confirm zero-failure invariant checks: {manifest_path}"
        )
    return {
        **provenance,
        "status": "VERIFIED_FORMAL_PROVENANCE",
        "strict_provenance_required": True,
        "manifest_verified": True,
        "manifest_filename": manifest_path.name,
        "actions_sha256": provenance["input_files_sha256"][actions_path.name],
        "source_commit_reproducible": True,
        "source_git_dirty": False,
        "invariants_passed": True,
        "invariant_check_count": invariants["check_count"],
        "invariant_failure_count": 0,
    }


def _read_strict_json_value(path: str | Path) -> tuple[Path, Any]:
    source = Path(path).resolve()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key}")
            result[key] = value
        return result

    document = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    return source, document


def _read_strict_json_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, document = _read_strict_json_value(path)
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {source}")
    return source, document


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    for line_number, text in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {source}:{line_number}")
        rows.append(row)
    return rows


def _read_failure_task_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item: dict[str, Any] = dict(row)
        for name in (
            "attempt_started_count",
            "max_attempts",
        ):
            item[name] = int(item[name])
        for name in (
            "rsu_attributed_energy_j",
            "failure_penalty_cost",
        ):
            item[name] = float(item[name])
        for source_name, target_name in (
            ("anon_attempts_json", "anon_attempts"),
            ("network_audit_json", "network_audit"),
        ):
            value = json.loads(item[source_name])
            if not isinstance(value, list):
                raise ValueError(
                    f"{source_name} must decode to a list in task row {index}"
                )
            item[target_name] = value
        normalized.append(item)
    return normalized


def command_subject_evidence_report(args: argparse.Namespace) -> int:
    source = Path(args.evidence).resolve()
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("evidence root must be an object")
    expected_hash = evidence.get("evidence_hash")
    actual_hash = canonical_document_sha256(evidence, "evidence_hash")
    if expected_hash != actual_hash:
        raise ValueError("evidence self-hash mismatch")
    report = build_subject_cluster_evidence_report(
        evidence,
        statistical_seed=args.seed,
        resamples=args.resamples,
        confidence_level=args.confidence,
    )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": report["report_sha256"],
                "independent_unit": report["independent_unit"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_numerical_evidence_report(args: argparse.Namespace) -> int:
    source = Path(args.evidence).resolve()
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("evidence root must be an object")
    expected_hash = evidence.get("evidence_hash")
    actual_hash = canonical_document_sha256(evidence, "evidence_hash")
    if expected_hash != actual_hash:
        raise ValueError("evidence self-hash mismatch")
    subject_counts = tuple(
        int(item.strip()) for item in args.subject_counts.split(",") if item.strip()
    )
    if not subject_counts:
        raise ValueError("subject-counts must contain at least one integer")
    report = build_numerical_experiment_report(
        evidence,
        subject_counts=subject_counts,
        statistical_seed=args.seed,
        resamples=args.resamples,
        confidence_level=args.confidence,
    )
    report["input_file_sha256"] = sha256_file(source)
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "input_file_sha256": report["input_file_sha256"],
                "report_sha256": report["report_sha256"],
                "subject_counts": list(subject_counts),
            },
            sort_keys=True,
        )
    )
    return 0


def command_audit_hard_mask(args: argparse.Namespace) -> int:
    source = Path(args.actions)
    allow_unverified = bool(getattr(args, "allow_unverified_inputs", False))
    provenance: dict[str, Any]
    action_records: list[dict[str, Any]] = []
    try:
        if allow_unverified:
            provenance = {
                "status": "UNVERIFIED_DEVELOPMENT_OVERRIDE",
                "strict_provenance_required": False,
                "manifest_verified": False,
                "actions_sha256": sha256_file(source),
            }
        else:
            provenance = _verify_hard_mask_provenance(source)
        action_records = _read_jsonl(source)
        if sha256_file(source) != provenance["actions_sha256"]:
            raise ValueError("actions input changed during hard-mask audit")
        report = audit_hard_mask_counterfactual(action_records)
    except (AuditValidationError, OSError, ValueError) as error:
        if isinstance(error, AuditValidationError):
            error_document = error.as_dict()
        else:
            error_document = {
                "status": "REFUSED",
                "error_code": "AUDIT_INPUT_PROVENANCE_INVALID",
                "message": str(error),
                "details": {},
            }
        provenance = {
            "status": "REFUSED",
            "strict_provenance_required": not allow_unverified,
            "manifest_verified": False,
            "actions_sha256": sha256_file(source),
            "error_code": error_document["error_code"],
        }
        report = {
            "schema_version": "1.1",
            "analysis": "hard_mask_safety_counterfactual",
            **error_document,
            "execution_validation_status": "REFUSED",
            "executed_action_count": None,
            "validated_count": None,
            "violation_count": None,
            "hard_mask_bypassed": None,
            "input_sha256": hashlib.sha256(
                canonical_json_bytes(action_records)
            ).hexdigest(),
        }
    report["input_artifact_verification"] = provenance
    report["input_file_sha256"] = sha256_file(source)
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    validation_status = report["execution_validation_status"]
    raw_violation_count = report["violation_count"]
    violation_count = (
        None if raw_violation_count is None else int(raw_violation_count)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "input_sha256": report["input_sha256"],
                "input_file_sha256": report["input_file_sha256"],
                "report_sha256": report["report_sha256"],
                "execution_validation_status": validation_status,
                "executed_action_count": report["executed_action_count"],
                "validated_count": report["validated_count"],
                "violation_count": violation_count,
                "hard_mask_bypassed": report["hard_mask_bypassed"],
                "provenance_status": provenance["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if validation_status == "VALIDATED" and violation_count == 0 else 2


def command_audit_two_stage(args: argparse.Namespace) -> int:
    commitments = json.loads(Path(args.commitments).read_text(encoding="utf-8"))
    if not isinstance(commitments, dict):
        raise ValueError("one-shot commitments must be a JSON object")
    if commitments.get("analysis") != "preregistered_one_shot_commitments":
        raise ValueError(
            "audit-two-stage requires a preregistered commitment document; "
            "run build-one-shot-commitments first"
        )
    report = evaluate_two_stage_information_ablation(
        _read_jsonl(args.actions), one_shot_commitments=commitments
    )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "input_sha256": report["input_sha256"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_build_one_shot_commitments(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("one-shot plan must be a JSON object")
    report = build_preregistered_one_shot_commitments(
        _read_jsonl(args.actions), registered_plan=plan
    )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_visible_input_sha256": report["raw_visible_input_sha256"],
                "registered_plan_sha256": report["registered_plan_sha256"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_audit_failure_integrity(args: argparse.Namespace) -> int:
    report = audit_failure_cost_completeness(
        _read_failure_task_rows(args.tasks),
        _read_jsonl(args.actions),
        _read_jsonl(args.events),
    )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {"output": str(output), "report_sha256": report["report_sha256"]},
            sort_keys=True,
        )
    )
    return 0


def command_audit_failure_coverage(args: argparse.Namespace) -> int:
    """Aggregate observed failure-cost coverage across complete run folders."""

    requested_roots = [Path(raw).resolve() for raw in (args.runs or [])]
    study_roots = [Path(raw).resolve() for raw in (args.study_roots or [])]
    incomplete_studies = _incomplete_sweep_markers(study_roots)
    if incomplete_studies:
        raise RuntimeError(
            "failure coverage refuses incomplete sweep roots: "
            f"{[str(path) for path in incomplete_studies]}"
        )
    _validate_completed_sweep_inputs(study_roots)
    for study_root in study_roots:
        if not study_root.is_dir():
            raise NotADirectoryError(f"failure-coverage study root missing: {study_root}")
        requested_roots.extend(
            sorted(
                (manifest.parent.resolve() for manifest in study_root.rglob("manifest.json")),
                key=lambda path: path.as_posix(),
            )
        )
    if not requested_roots:
        raise ValueError("no run directories with manifest.json were found")
    incomplete = _incomplete_sweep_markers(requested_roots)
    if incomplete:
        raise RuntimeError(
            "failure coverage refuses incomplete sweep roots: "
            f"{[str(path) for path in incomplete]}"
        )
    _validate_completed_sweep_inputs(requested_roots)

    runs: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in requested_roots:
        if root in seen:
            continue
        seen.add(root)
        required_files = {
            "tasks": root / "tasks.csv",
            "actions": root / "actions.jsonl",
            "events": root / "events.jsonl",
        }
        missing = [str(path) for path in required_files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"failure-coverage run is missing required files: {missing}"
            )
        if args.allow_unverified_inputs:
            provenance = {
                "verification_status": "UNVERIFIED_DEVELOPMENT_OVERRIDE",
                "input_files_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(required_files.values())
                },
            }
            run_id = root.as_posix()
        else:
            provenance = _verify_run_artifacts(
                root,
                required_filenames=tuple(
                    path.name for path in sorted(required_files.values())
                ),
            )
            run_id = f"manifest:{provenance['manifest_sha256']}"
        task_rows = _read_failure_task_rows(required_files["tasks"])
        action_records = _read_jsonl(required_files["actions"])
        event_records = _read_jsonl(required_files["events"])
        current_hashes = {
            path.name: sha256_file(path)
            for path in sorted(required_files.values())
        }
        if current_hashes != provenance["input_files_sha256"]:
            raise ValueError(
                f"failure-coverage inputs changed while being read: {root}"
            )
        provenance_records.append({"run_id": run_id, **provenance})
        runs.append(
            {
                "run_id": run_id,
                "input_artifact_provenance": provenance,
                "task_rows": task_rows,
                "action_records": action_records,
                "event_records": event_records,
            }
        )
    report = audit_failure_cost_coverage(runs)
    report["input_artifact_verification"] = {
        "status": (
            "UNVERIFIED_DEVELOPMENT_OVERRIDE"
            if args.allow_unverified_inputs
            else "VERIFIED"
        ),
        "runs": provenance_records,
    }
    required_categories = tuple(
        item.strip()
        for item in (args.require_categories or "").split(",")
        if item.strip()
    )
    unknown = sorted(
        set(required_categories) - set(report.get("observed_coverage", {}))
    )
    if unknown:
        raise ValueError(f"unknown failure coverage categories: {unknown}")
    missing_categories = [
        category
        for category in required_categories
        if category in report["coverage_scope"]["not_observed_categories"]
    ]
    if missing_categories:
        raise ValueError(
            "required failure categories were not observed: "
            f"{missing_categories}; rerun without --require-categories to write a "
            "descriptive partial-coverage report"
        )
    report["required_observed_categories"] = list(required_categories)
    report["required_observed_categories_satisfied"] = True
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "run_count": report["run_count"],
                "status": report["status"],
                "required_observed_categories_satisfied": True,
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_exact_scenario_oracle(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("oracle input must be a JSON object")
    report = exact_finite_scenario_oracle(
        document["scenarios"],
        esl_action_sequence=document["esl_action_sequence"],
        max_sequences=int(document.get("max_sequences", 100_000)),
    )
    report["input_file_sha256"] = sha256_file(source)
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "input_file_sha256": report["input_file_sha256"],
                "input_sha256": report["input_sha256"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_exact_adaptive_scenario_oracle(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("adaptive oracle input must be a JSON object")
    report = exact_adaptive_scenario_tree_oracle(
        document["scenario_tree"],
        esl_contingent_policy=document["esl_contingent_policy"],
        max_policies=int(document.get("max_policies", 100_000)),
    )
    report["input_file_sha256"] = sha256_file(source)
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "input_file_sha256": report["input_file_sha256"],
                "input_sha256": report["input_sha256"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_select_validation(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("validation selection input must be an object")
    limits = ValidationLimits(**document["limits"])
    candidates = [ValidationCandidate(**row) for row in document["candidates"]]
    report = select_feasible_validation_candidate(candidates, limits).to_dict()
    report["input_sha256"] = sha256_file(source)
    report["selection_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_config_id": report["selected_config_id"],
                "selection_sha256": report["selection_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _nested_metric(summary: dict[str, Any], dotted: str) -> float:
    value: Any = summary
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"summary metric is missing: {dotted}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary metric is not numeric: {dotted}")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"summary metric is not finite: {dotted}")
    return result


_MECHANISM_METRICS = (
    "edge_done_rate",
    "pipeline_attempt_rate",
    "pipeline_to_edge_rate",
    "pipeline_to_local_rate",
    "all_task_loss",
    "coverage",
    "failure_rate",
    "timeout_rate",
    "latency_p95_s",
    "energy_j.task_attributed.total",
    "resources.max_utilization",
    "resources.max_waiting_jobs",
)


def _mechanism_snapshot(summary: dict[str, Any]) -> dict[str, float]:
    """Extract path, performance and contention signals from one run.

    These values are deliberately independent of controller wall-clock time and
    configuration identity.  They therefore expose a sweep whose coordinates
    change while its simulated behaviour does not.
    """

    return {name: _nested_metric(summary, name) for name in _MECHANISM_METRICS}


def _distribution_record(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("mechanism diagnostic distribution cannot be empty")
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _study_mechanism_diagnostics(
    rows: list[dict[str, Any]], *, baseline: str
) -> dict[str, Any]:
    """Describe whether a paired study exercised two-stage mechanisms."""

    if not rows:
        raise ValueError("study mechanism diagnostics require at least one run")
    policies = sorted({str(row["policy"]) for row in rows})
    by_policy: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        metrics = {
            metric: _distribution_record(
                [float(row["mechanism_metrics"][metric]) for row in selected]
            )
            for metric in _MECHANISM_METRICS
        }
        by_policy[policy] = {
            "run_count": len(selected),
            "metrics": metrics,
        }
        if policy not in {baseline, "all_local"} and metrics["edge_done_rate"]["max"] == 0.0:
            warnings.append(
                {
                    "code": "POLICY_NO_EDGE_COMPLETION",
                    "policy": policy,
                    "message": "policy never completed an edge path in this study",
                }
            )
        if (
            policy not in {baseline, "all_local"}
            and metrics["pipeline_attempt_rate"]["max"] > 0.0
            and metrics["edge_done_rate"]["max"] == 0.0
        ):
            warnings.append(
                {
                    "code": "PIPELINE_WITHOUT_EDGE_COMPLETION",
                    "policy": policy,
                    "message": (
                        "anonymous pipeline attempts occurred without any completed "
                        "edge path; inspect local fallbacks, failures and timeouts"
                    ),
                }
            )
    if max(
        float(row["mechanism_metrics"]["resources.max_waiting_jobs"])
        for row in rows
    ) == 0.0:
        warnings.append(
            {
                "code": "NO_OBSERVED_RESOURCE_QUEUEING",
                "message": "no resource pool recorded a waiting job in any run",
            }
        )
    return {
        "schema_version": "1.0",
        "baseline": baseline,
        "metrics": list(_MECHANISM_METRICS),
        "by_policy": by_policy,
        "warnings": warnings,
        "interpretation": (
            "diagnostic only; warnings identify unexercised mechanisms and do not "
            "alter registered hypotheses"
        ),
    }


def _sweep_mechanism_diagnostics(
    rows: list[dict[str, Any]],
    *,
    coordinate_keys: list[str],
    paired_coordinates: bool = False,
) -> dict[str, Any]:
    """Detect coordinate changes that produced no observed physical effect."""

    if not rows:
        raise ValueError("sweep diagnostics require at least one case")
    for row in rows:
        snapshot = row["mechanism_metrics"]
        row["behavior_signature_sha256"] = hashlib.sha256(
            canonical_json_bytes(snapshot)
        ).hexdigest()
    unique_signatures = sorted(
        {str(row["behavior_signature_sha256"]) for row in rows}
    )
    parameter_effects: dict[str, bool] = {}
    if paired_coordinates:
        parameter_effects["$paired"] = len(unique_signatures) > 1
    else:
        for key in coordinate_keys:
            conditional_groups: dict[tuple[tuple[str, str], ...], set[str]] = {}
            for row in rows:
                coordinates = row["parameters"]
                other_coordinates = tuple(
                    (name, json.dumps(value, ensure_ascii=False, sort_keys=True))
                    for name, value in sorted(coordinates.items())
                    if name != key
                )
                conditional_groups.setdefault(other_coordinates, set()).add(
                    str(row["behavior_signature_sha256"])
                )
            parameter_effects[key] = any(
                len(signatures) > 1 for signatures in conditional_groups.values()
            )
    warnings: list[dict[str, Any]] = []
    if len(unique_signatures) == 1:
        warnings.append(
            {
                "code": "NO_OBSERVED_BEHAVIOR_VARIATION",
                "message": "all sweep cases produced the same mechanism signature",
            }
        )
    for key, observed in parameter_effects.items():
        if not observed:
            warnings.append(
                {
                    "code": "PARAMETER_NO_OBSERVED_EFFECT",
                    "parameter": key,
                    "message": (
                        "changing this coordinate produced no mechanism-signature "
                        "change while the other coordinates were held fixed"
                    ),
                }
            )
    if max(row["mechanism_metrics"]["edge_done_rate"] for row in rows) == 0.0:
        warnings.append(
            {
                "code": "SWEEP_NO_EDGE_COMPLETION",
                "message": "no sweep case completed an edge path",
            }
        )
    if max(
        row["mechanism_metrics"]["resources.max_waiting_jobs"] for row in rows
    ) == 0.0:
        warnings.append(
            {
                "code": "SWEEP_NO_RESOURCE_QUEUEING",
                "message": "no sweep case exercised resource waiting",
            }
        )
    return {
        "schema_version": "1.0",
        "case_count": len(rows),
        "coordinate_mode": "paired" if paired_coordinates else "cartesian",
        "coordinate_keys": coordinate_keys,
        "behavior_metrics": list(_MECHANISM_METRICS),
        "unique_behavior_signature_count": len(unique_signatures),
        "behavior_signatures_sha256": unique_signatures,
        "parameter_observed_effect": parameter_effects,
        "warnings": warnings,
        "interpretation": (
            "an observed-effect false value is a descriptive result, not proof "
            "that the parameter is globally irrelevant"
        ),
    }


def command_numerical_study(args: argparse.Namespace) -> int:
    policies = tuple(args.policies.split(",")) if args.policies else POLICIES
    if len(policies) < 2 or len(policies) != len(set(policies)):
        raise ValueError("study requires at least two unique policies")
    unknown = sorted(set(policies) - set(RUNNABLE_POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if args.baseline not in policies:
        raise ValueError("study baseline must be included in policies")
    seeds = tuple(
        int(item.strip()) for item in args.environment_seeds.split(",") if item.strip()
    )
    if (
        len(seeds) < 2
        or len(seeds) != len(set(seeds))
        or any(seed < 0 for seed in seeds)
    ):
        raise ValueError(
            "study requires at least two unique nonnegative environment seeds"
        )
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    if not metrics:
        raise ValueError("study metrics cannot be empty")
    if len(metrics) != len(set(metrics)):
        raise ValueError("study metrics must be preregistered as unique names")
    load_level = str(args.load_level).strip()
    if not load_level:
        raise ValueError("study load_level must be non-empty")
    root = Path(args.output_root).resolve()
    family_id = (
        str(args.family_id).strip()
        if args.family_id is not None
        else f"standalone:{root.name}"
    )
    if not family_id:
        raise ValueError("study family_id must be non-empty")
    require_clean_source = not args.allow_dirty_source
    base_config_path, base_config_document = _read_strict_json_object(
        Path(args.base_study_root).resolve() / "configs" / "numerical_default.json"
    )
    batch_base_config = load_config(base_config_path)
    batch_base_frozen_inputs = _capture_frozen_input_assets(batch_base_config)
    batch_base_config_sha256 = sha256_file(base_config_path)
    batch_base_identity: dict[str, Any] = {
        "schema_version": "1.0",
        "config_path": str(base_config_path),
        "config_raw_sha256": batch_base_config_sha256,
        "frozen_input_assets": batch_base_frozen_inputs,
    }
    batch_base_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(batch_base_identity)
    ).hexdigest()

    def require_unchanged_batch_base(context: str) -> None:
        _require_same_file_bytes(
            base_config_path, batch_base_config_sha256, context=context
        )
        _require_same_frozen_input_assets(
            batch_base_config, batch_base_frozen_inputs, context=context
        )

    experiment_registration = _study_experiment_registration(
        registration_path=args.experiment_registration,
        registration_scale=args.registration_scale,
        registration_regime=args.registration_regime,
        registration_family=args.registration_family,
        base_study_root=args.base_study_root,
        environment_seeds=seeds,
        load_level=load_level,
        family_id=family_id,
        policies=policies,
        baseline=args.baseline,
        metrics=metrics,
        require_clean=require_clean_source,
    )
    require_unchanged_batch_base("during formal study registration")
    study_source_preflight = source_cleanliness_preflight(
        require_clean=require_clean_source
    )
    study_source_identity = (
        _source_identity(study_source_preflight) if require_clean_source else None
    )
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"numerical study output root is not empty: {root}")
    if root.exists() and any(root.iterdir()) and args.overwrite:
        raise FileExistsError(
            "numerical study overwrite requires a new root; individual bundles support explicit overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)

    family_registration: dict[str, Any] = {
        "schema_version": "1.0",
        "family_id": family_id,
        "load_level": load_level,
        "registered_metrics": list(metrics),
        "registered_policies": list(policies),
        "baseline": args.baseline,
        "family_dimensions": ["load", "metric", "policy_vs_baseline"],
        "experiment_registration": experiment_registration,
        "registration_sha256": "",
    }
    family_registration["registration_sha256"] = canonical_document_sha256(
        family_registration, "registration_sha256"
    )

    records: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    run_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for environment_seed in seeds:
        require_unchanged_batch_base(
            f"before environment replication {environment_seed}"
        )
        replication_root = root / "inputs" / f"environment-{environment_seed}"
        replication = generate_numerical_replication(
            args.base_study_root,
            replication_root,
            environment_seed,
            overwrite=False,
        )
        require_unchanged_batch_base(
            f"while generating environment replication {environment_seed}"
        )
        _, replication_config_document = _read_strict_json_object(
            replication.config_path
        )
        if _replication_config_signature(
            replication_config_document
        ) != _replication_config_signature(base_config_document):
            raise RuntimeError(
                "environment replication changed config outside the registered "
                f"RNG streams: seed={environment_seed}"
            )
        config = load_config(replication.config_path)
        replication_frozen_inputs = _capture_frozen_input_assets(config)
        _require_frozen_input_content_match(
            replication_frozen_inputs,
            batch_base_frozen_inputs,
            roles=("profile", "scenario_trace", "evidence"),
            context=f"in environment replication {environment_seed}",
        )
        expected_semantic_hashes = {
            role: batch_base_frozen_inputs["assets"][role][
                "declared_content_hash"
            ]
            for role in ("profile", "scenario_trace", "evidence")
        }
        actual_semantic_hashes = {
            "profile": replication.profile_hash,
            "scenario_trace": replication.scenario_trace_hash,
            "evidence": replication.evidence_hash,
        }
        if actual_semantic_hashes != expected_semantic_hashes:
            raise RuntimeError(
                "environment replication semantic identity differs from the "
                f"batch base bundle: seed={environment_seed}"
            )
        environment_id = f"numerical-environment-{environment_seed}"
        for policy in policies:
            output = root / "runs" / environment_id / policy
            result, artifacts, manifest = _execute(
                config,
                policy,
                output,
                overwrite=False,
                run_metadata={
                    "study_id": root.name,
                    "environment_replication_id": environment_id,
                    "pairing_id": "all_tasks",
                    "environment_seed": environment_seed,
                    "experiment_registration_record_sha256": (
                        experiment_registration.get("record_sha256")
                    ),
                    "batch_base_identity_sha256": batch_base_identity_sha256,
                },
                require_clean_source=require_clean_source,
                expected_source_identity=study_source_identity,
            )
            require_unchanged_batch_base(
                f"after environment {environment_seed} policy {policy}"
            )
            manifest_frozen_inputs = manifest.get("frozen_input_assets")
            if not isinstance(manifest_frozen_inputs, dict):
                raise RuntimeError("study run manifest lacks frozen input assets")
            _require_frozen_input_content_match(
                manifest_frozen_inputs,
                batch_base_frozen_inputs,
                roles=("profile", "scenario_trace", "evidence"),
                context=f"in environment {environment_seed} policy {policy}",
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            row = _summary_line(result, artifacts, output)
            row.update(
                environment_id=environment_id,
                environment_seed=environment_seed,
                evaluation_trace_hash=result.trace.trace_hash,
                scenario_trace_hash=result.scenario_trace.trace_hash,
                output_relative=output.relative_to(root).as_posix(),
            )
            mechanism_metrics = _mechanism_snapshot(summary)
            row["mechanism_metrics"] = mechanism_metrics
            run_rows.append(row)
            mechanism_rows.append(
                {
                    "policy": policy,
                    "environment_id": environment_id,
                    "environment_seed": environment_seed,
                    "mechanism_metrics": mechanism_metrics,
                }
            )
            for metric in metrics:
                records[metric].append(
                    {
                        "environment_id": environment_id,
                        "pairing_id": "all_tasks",
                        "strategy": policy,
                        "metric_value": _nested_metric(summary, metric),
                        "evaluation_trace_hash": result.trace.trace_hash,
                        "task_identity_hash": result.trace.trace_hash,
                    }
                )

    require_unchanged_batch_base("before formal study analysis publication")
    if study_source_identity is not None:
        _require_same_source_identity(
            source_cleanliness_preflight(require_clean=True),
            study_source_identity,
            context="before formal study analysis publication",
        )

    analyses: dict[str, Any] = {}
    for metric, metric_records in records.items():
        record_path = root / f"statistical-records-{metric.replace('.', '_')}.json"
        record_path.write_text(
            json.dumps(metric_records, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        analyses[metric] = analyze_paired_strategies(
            metric_records,
            baseline_strategy=args.baseline,
            metric_name=metric,
            statistical_seed=args.statistics_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            sign_flip_permutations=args.permutations,
        )
    analyses, holm_family = apply_holm_family_adjustment(
        analyses,
        family_dimensions=("metric", "policy_vs_baseline"),
    )
    if study_source_identity is not None:
        _require_same_source_identity(
            source_cleanliness_preflight(require_clean=True),
            study_source_identity,
            context="before formal study report publication",
        )
    report = {
        "study_schema_version": "1.1",
        "study_kind": "frozen_numerical_simulation",
        "real_hardware_measurement": False,
        "environment_seeds": list(seeds),
        "policies": list(policies),
        "baseline": args.baseline,
        "load_level": load_level,
        "source_cleanliness_preflight": study_source_preflight,
        "batch_base_identity": batch_base_identity,
        "batch_base_identity_sha256": batch_base_identity_sha256,
        "experiment_registration": experiment_registration,
        "statistical_family_registration": family_registration,
        "runs": run_rows,
        "analyses": analyses,
        "multiple_testing": holm_family,
        "mechanism_diagnostics": _study_mechanism_diagnostics(
            mechanism_rows, baseline=args.baseline
        ),
        "study_report_sha256": "",
    }
    require_unchanged_batch_base("before formal study report publication")
    report["study_report_sha256"] = canonical_document_sha256(
        report, "study_report_sha256"
    )
    _write_new_json(
        root / "study_report.json",
        report,
        overwrite=False,
    )
    print(
        json.dumps(
            {
                "study_report": str((root / "study_report.json").resolve()),
                "run_count": len(run_rows),
                "environment_count": len(seeds),
                "policy_count": len(policies),
                "metrics": list(metrics),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_aggregate_statistical_families(args: argparse.Namespace) -> int:
    if len(args.study_reports) < 2:
        raise ValueError(
            "at least two independently registered load studies are required"
        )
    documents: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for raw_path in args.study_reports:
        source, document = _read_strict_json_object(raw_path)
        if source in paths:
            raise ValueError(f"duplicate study report path: {source}")
        paths.add(source)
        documents.append(document)
        identities.append(
            {
                "path": str(source),
                "file_sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    report = aggregate_preregistered_study_families(
        documents, input_identities=identities
    )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "family_id": report["family_id"],
                "load_levels": report["load_levels"],
                "hypothesis_count": report["hypothesis_count"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_analyze_sensitivity(args: argparse.Namespace) -> int:
    """Analyze registered sweep levels with environments as independent units."""

    requested = [Path(item).resolve() for item in args.sweep_roots]
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate --sweep-roots input")
    incomplete = _incomplete_sweep_markers(requested)
    if incomplete:
        raise RuntimeError(
            "sensitivity analysis refuses incomplete sweep roots: "
            f"{[str(path) for path in incomplete]}"
        )
    roots = _completed_sweep_roots(requested)
    if not roots:
        raise FileNotFoundError("no completed registered sweep roots were found")
    _validate_completed_sweep_inputs(roots)
    analysis_preflight = source_cleanliness_preflight(require_clean=True)
    registration_preflight = source_cleanliness_preflight(
        args.sensitivity_registration, require_clean=True
    )
    report = analyze_registered_sensitivity_sweeps(
        roots,
        sensitivity_path=args.sensitivity_registration,
        metric_name=args.metric,
        statistical_seed=args.statistics_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        sign_flip_permutations=args.permutations,
        confidence_level=args.confidence,
        analysis_source_preflight=analysis_preflight,
        sensitivity_source_preflight=registration_preflight,
    )
    _require_same_source_identity(
        source_cleanliness_preflight(require_clean=True),
        _source_identity(analysis_preflight),
        context="before sensitivity analysis publication",
    )
    if sha256_file(args.sensitivity_registration) != report[
        "sensitivity_registration"
    ]["file_sha256"]:
        raise RuntimeError(
            "sensitivity registration bytes changed before analysis publication"
        )
    output = _write_new_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "metric": report["metric_name"],
                "factor_count": report["factor_count"],
                "environment_count": len(report["environment_seeds"]),
                "report_sha256": report["report_sha256"],
                "study_role": report["study_role"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="privacy-edge-sim")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate", help="validate config, frozen profile and trace together"
    )
    validate.add_argument("--config", default="configs/default.json")
    validate.set_defaults(func=command_validate)

    validate_profile = sub.add_parser("validate-profile")
    validate_profile.add_argument("--profile", required=True)
    validate_profile.set_defaults(func=command_validate_profile)

    validate_trace = sub.add_parser("validate-trace")
    validate_trace.add_argument("--profile", required=True)
    validate_trace.add_argument("--trace", required=True)
    validate_trace.set_defaults(func=command_validate_trace)

    source_preflight = sub.add_parser(
        "source-preflight",
        help="verify that executable package source matches the current Git commit",
    )
    source_preflight.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="development-only diagnostic override",
    )
    source_preflight.set_defaults(func=command_source_preflight)

    generate = sub.add_parser("generate-synthetic")
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--seed", type=int, default=20260712)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(func=command_generate)

    generate_numerical = sub.add_parser(
        "generate-numerical-study",
        help="build a frozen subject-disjoint numerical research bundle",
    )
    generate_numerical.add_argument("--output-root", required=True)
    generate_numerical.add_argument("--seed", type=int, default=20260713)
    generate_numerical.add_argument("--profile-subjects", type=int, default=256)
    generate_numerical.add_argument("--test-subjects", type=int, default=64)
    generate_numerical.add_argument("--scenario-subjects", type=int, default=48)
    generate_numerical.add_argument("--tasks", type=int, default=24)
    generate_numerical.add_argument("--horizon", type=float, default=20.0)
    generate_numerical.add_argument(
        "--arrival-center-s",
        type=float,
        default=None,
        help="center of the explicit task-arrival window in simulated seconds",
    )
    generate_numerical.add_argument(
        "--arrival-window-s",
        type=float,
        default=None,
        help="width of the explicit task-arrival window in simulated seconds",
    )
    generate_numerical.add_argument(
        "--arrival-jitter-fraction",
        type=float,
        default=None,
        help="relative per-task jitter within an explicit arrival window",
    )
    generate_numerical.add_argument("--privacy-threshold", type=float, default=0.35)
    generate_numerical.add_argument(
        "--preprocessing-failure-mode",
        choices=("legacy_last", "none", "fixed_count", "bernoulli"),
        default="legacy_last",
        help="preprocessing-failure assignment; legacy_last preserves older bundles",
    )
    generate_numerical.add_argument(
        "--preprocessing-failure-count",
        type=int,
        default=0,
        help="number of seeded preprocessing failures when mode is fixed_count",
    )
    generate_numerical.add_argument(
        "--preprocessing-failure-probability",
        type=float,
        default=0.0,
        help="per-task seeded preprocessing-failure probability when mode is bernoulli",
    )
    generate_numerical.add_argument(
        "--local-service-scale",
        type=float,
        default=1.0,
        help="multiplicative local FER service-time scale; 1.0 preserves the baseline",
    )
    generate_numerical.add_argument(
        "--anon-time-variability-scale",
        type=float,
        default=1.0,
        help="preregistered within-pipeline anonymization-time variability in [0,3]",
    )
    generate_numerical.add_argument(
        "--output-size-variability-scale",
        type=float,
        default=1.0,
        help="preregistered within-pipeline encoded-size variability in [0,3]",
    )
    generate_numerical.add_argument("--overwrite", action="store_true")
    generate_numerical.set_defaults(func=command_generate_numerical)

    replication = sub.add_parser(
        "generate-numerical-replication",
        help="generate a new environment while freezing profile/evidence/scenarios",
    )
    replication.add_argument("--base-study-root", required=True)
    replication.add_argument("--output-root", required=True)
    replication.add_argument("--environment-seed", type=int, required=True)
    replication.add_argument("--overwrite", action="store_true")
    replication.set_defaults(func=command_generate_numerical_replication)

    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/default.json")
    run.add_argument("--policy", choices=RUNNABLE_POLICIES)
    run.add_argument("--output", required=True)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--checkpoint",
        help="atomically updated deterministic replay checkpoint path",
    )
    run.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write a replay checkpoint every N compound events (0=final only)",
    )
    run.add_argument(
        "--resume-checkpoint",
        help="verify/replay this checkpoint prefix before continuing",
    )
    run.set_defaults(func=command_run)

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--config", default="configs/default.json")
    run_all.add_argument("--output-root", required=True)
    run_all.add_argument("--policies", help="comma-separated subset")
    run_all.add_argument("--overwrite", action="store_true")
    run_all.set_defaults(func=command_run_all)

    multi = sub.add_parser("multi-seed")
    multi.add_argument("--config", default="configs/default.json")
    multi.add_argument("--policy", choices=RUNNABLE_POLICIES)
    multi.add_argument("--seeds", required=True, help="comma-separated base seeds")
    multi.add_argument("--output-root", required=True)
    multi.add_argument("--overwrite", action="store_true")
    multi.set_defaults(func=command_multi_seed)

    derive_config = sub.add_parser(
        "derive-config",
        help="apply a named JSON override block and validate a frozen config copy",
    )
    derive_config.add_argument("--config", required=True)
    derive_config.add_argument("--overrides", required=True)
    derive_config.add_argument(
        "--section",
        required=True,
        help="dotted object path containing config-path to value mappings",
    )
    derive_config.add_argument("--output", required=True)
    derive_config.add_argument("--overwrite", action="store_true")
    derive_config.set_defaults(func=command_derive_config)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--config", default="configs/default.json")
    sweep.add_argument("--policy", choices=RUNNABLE_POLICIES)
    sweep.add_argument(
        "--grid",
        required=True,
        help="JSON object mapping dotted config paths to arrays",
    )
    sweep.add_argument("--output-root", required=True)
    sweep.add_argument(
        "--sensitivity-registration",
        help="committed paper-v1 sensitivity plan; required for formal sweeps",
    )
    sweep.add_argument(
        "--registration-factor",
        help="factor key under sensitivity-registration.factors",
    )
    sweep.add_argument(
        "--experiment-registration",
        help="committed generator/load matrix used by the sensitivity reference",
    )
    sweep.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "development-only override; formal sweeps require source bound to a "
            "Git commit"
        ),
    )
    sweep.add_argument("--overwrite", action="store_true")
    sweep.set_defaults(func=command_sweep)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--inputs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument(
        "--parquet",
        action="store_true",
        help="also write a flattened Parquet aggregate (requires pyarrow)",
    )
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.set_defaults(func=command_aggregate)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", default="configs/default.json")
    smoke.add_argument(
        "--policy", choices=RUNNABLE_POLICIES, default="safe_lyapunov_h1"
    )
    smoke.add_argument("--output-root", required=True)
    smoke.set_defaults(func=command_smoke)

    compare = sub.add_parser(
        "compare", help="strict paired environment-cluster statistical comparison"
    )
    compare.add_argument("--records", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--metric", required=True)
    compare.add_argument("--seed", type=int, required=True)
    compare.add_argument("--bootstrap-resamples", type=int, default=2000)
    compare.add_argument("--permutations", type=int, default=20000)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--output", required=True)
    compare.add_argument("--overwrite", action="store_true")
    compare.set_defaults(func=command_compare)

    evidence_report = sub.add_parser(
        "subject-evidence-report",
        help="subject-cluster bootstrap report for frozen privacy and FER evidence",
    )
    evidence_report.add_argument("--evidence", required=True)
    evidence_report.add_argument("--seed", type=int, required=True)
    evidence_report.add_argument("--resamples", type=int, default=2000)
    evidence_report.add_argument("--confidence", type=float, default=0.95)
    evidence_report.add_argument("--output", required=True)
    evidence_report.add_argument("--overwrite", action="store_true")
    evidence_report.set_defaults(func=command_subject_evidence_report)

    numerical_evidence_report = sub.add_parser(
        "numerical-evidence-report",
        help="post-selection, finite-sample and subject-cluster evidence report",
    )
    numerical_evidence_report.add_argument("--evidence", required=True)
    numerical_evidence_report.add_argument("--subject-counts", required=True)
    numerical_evidence_report.add_argument("--seed", type=int, required=True)
    numerical_evidence_report.add_argument("--resamples", type=int, default=2000)
    numerical_evidence_report.add_argument("--confidence", type=float, default=0.95)
    numerical_evidence_report.add_argument("--output", required=True)
    numerical_evidence_report.add_argument("--overwrite", action="store_true")
    numerical_evidence_report.set_defaults(func=command_numerical_evidence_report)

    hard_mask_audit = sub.add_parser(
        "audit-hard-mask",
        help=(
            "manifest-bound offline audit of counterfactual masks and actual "
            "execution-time rechecks"
        ),
    )
    hard_mask_audit.add_argument("--actions", required=True)
    hard_mask_audit.add_argument("--output", required=True)
    hard_mask_audit.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help="development-only override for inputs without verified formal run provenance",
    )
    hard_mask_audit.add_argument("--overwrite", action="store_true")
    hard_mask_audit.set_defaults(func=command_audit_hard_mask)

    two_stage_audit = sub.add_parser(
        "audit-two-stage",
        help="paired READY recourse and conservative information ablations",
    )
    two_stage_audit.add_argument("--actions", required=True)
    two_stage_audit.add_argument("--commitments", required=True)
    two_stage_audit.add_argument("--output", required=True)
    two_stage_audit.add_argument("--overwrite", action="store_true")
    two_stage_audit.set_defaults(func=command_audit_two_stage)

    one_shot_commitments = sub.add_parser(
        "build-one-shot-commitments",
        help="expand a preregistered READY priority using RAW-visible task identity",
    )
    one_shot_commitments.add_argument("--actions", required=True)
    one_shot_commitments.add_argument("--plan", required=True)
    one_shot_commitments.add_argument("--output", required=True)
    one_shot_commitments.add_argument("--overwrite", action="store_true")
    one_shot_commitments.set_defaults(func=command_build_one_shot_commitments)

    failure_audit = sub.add_parser(
        "audit-failure-integrity",
        help="quantify retry, downlink, RSU-energy and failure-term omissions",
    )
    failure_audit.add_argument("--tasks", required=True)
    failure_audit.add_argument("--actions", required=True)
    failure_audit.add_argument("--events", required=True)
    failure_audit.add_argument("--output", required=True)
    failure_audit.add_argument("--overwrite", action="store_true")
    failure_audit.set_defaults(func=command_audit_failure_integrity)

    failure_coverage = sub.add_parser(
        "audit-failure-coverage",
        help="aggregate observed failure-cost coverage across run directories",
    )
    failure_sources = failure_coverage.add_mutually_exclusive_group(required=True)
    failure_sources.add_argument("--runs", nargs="+")
    failure_sources.add_argument(
        "--study-roots",
        nargs="+",
        help="recursively discover complete run directories below study roots",
    )
    failure_coverage.add_argument(
        "--require-categories",
        help=(
            "optional comma-separated observed categories required before the "
            "report is written"
        ),
    )
    failure_coverage.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help=(
            "development-only override; formal audits require manifest-bound "
            "checksums for every consumed artifact"
        ),
    )
    failure_coverage.add_argument("--output", required=True)
    failure_coverage.add_argument("--overwrite", action="store_true")
    failure_coverage.set_defaults(func=command_audit_failure_coverage)

    exact_oracle = sub.add_parser(
        "exact-scenario-oracle",
        help="bounded hard-safe exact finite scenario-tree ratio oracle",
    )
    exact_oracle.add_argument("--input", required=True)
    exact_oracle.add_argument("--output", required=True)
    exact_oracle.add_argument("--overwrite", action="store_true")
    exact_oracle.set_defaults(func=command_exact_scenario_oracle)

    exact_adaptive_oracle = sub.add_parser(
        "exact-adaptive-scenario-oracle",
        help="exact hard-safe contingent-policy oracle on an observation scenario tree",
    )
    exact_adaptive_oracle.add_argument("--input", required=True)
    exact_adaptive_oracle.add_argument("--output", required=True)
    exact_adaptive_oracle.add_argument("--overwrite", action="store_true")
    exact_adaptive_oracle.set_defaults(func=command_exact_adaptive_scenario_oracle)

    select_validation = sub.add_parser(
        "select-validation",
        help="freeze a feasible-first controller candidate from validation summaries",
    )
    select_validation.add_argument("--input", required=True)
    select_validation.add_argument("--output", required=True)
    select_validation.add_argument("--overwrite", action="store_true")
    select_validation.set_defaults(func=command_select_validation)

    study = sub.add_parser(
        "run-numerical-study",
        help="run paired policies on independent numerical environments and analyze them",
    )
    study.add_argument("--base-study-root", required=True)
    study.add_argument("--environment-seeds", required=True)
    study.add_argument("--policies", help="comma-separated policy subset")
    study.add_argument("--baseline", default="all_local", choices=RUNNABLE_POLICIES)
    study.add_argument(
        "--metrics",
        default="all_task_loss,failure_rate,coverage,latency_p95_s,energy_j.task_attributed.total",
    )
    study.add_argument("--statistics-seed", type=int, default=91001)
    study.add_argument("--bootstrap-resamples", type=int, default=2000)
    study.add_argument("--permutations", type=int, default=20000)
    study.add_argument(
        "--load-level",
        default="default",
        help="preregistered load condition represented by this study",
    )
    study.add_argument(
        "--family-id",
        help="shared preregistered family id required to aggregate multiple load levels",
    )
    study.add_argument(
        "--experiment-registration",
        help=(
            "committed generator-matrix JSON; required unless "
            "--allow-dirty-source is used"
        ),
    )
    study.add_argument(
        "--registration-scale",
        help="scale key under experiment-registration.scales (for example pilot/formal)",
    )
    study.add_argument(
        "--registration-regime",
        help="regime key registered under the selected scale",
    )
    study.add_argument(
        "--registration-family",
        choices=("pilot", "primary", "secondary"),
        help="analysis family registered under experiment-registration.analysis_plan",
    )
    study.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "development-only override; formal numerical studies otherwise require "
            "the executable package source to match a Git commit"
        ),
    )
    study.add_argument("--output-root", required=True)
    study.add_argument("--overwrite", action="store_true")
    study.set_defaults(func=command_numerical_study)

    aggregate_families = sub.add_parser(
        "aggregate-statistical-families",
        help="strict Holm correction across preregistered load, metric and policy hypotheses",
    )
    aggregate_families.add_argument("--study-reports", nargs="+", required=True)
    aggregate_families.add_argument("--output", required=True)
    aggregate_families.add_argument("--overwrite", action="store_true")
    aggregate_families.set_defaults(func=command_aggregate_statistical_families)

    sensitivity_analysis = sub.add_parser(
        "analyze-sensitivity",
        help=(
            "strict exploratory paired analysis of completed registered sweeps; "
            "the environment is the independent unit"
        ),
    )
    sensitivity_analysis.add_argument(
        "--sweep-roots",
        nargs="+",
        required=True,
        help="completed sweep roots or parent directories containing them",
    )
    sensitivity_analysis.add_argument(
        "--sensitivity-registration",
        required=True,
        help="committed exploratory sensitivity plan used by every sweep",
    )
    sensitivity_analysis.add_argument("--metric", default="all_task_loss")
    sensitivity_analysis.add_argument("--statistics-seed", type=int, default=92001)
    sensitivity_analysis.add_argument("--bootstrap-resamples", type=int, default=2000)
    sensitivity_analysis.add_argument("--permutations", type=int, default=20000)
    sensitivity_analysis.add_argument("--confidence", type=float, default=0.95)
    sensitivity_analysis.add_argument("--output", required=True)
    sensitivity_analysis.add_argument("--overwrite", action="store_true")
    sensitivity_analysis.set_defaults(func=command_analyze_sensitivity)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
