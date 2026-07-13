"""Command-line entrypoints for validation, simulation and experiment orchestration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
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
    write_manifest,
)
from .numerical import (
    NumericalStudySpec,
    generate_numerical_replication,
    generate_numerical_study,
)
from .paper_experiment_audits import (
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
) -> tuple[RunResult, Any, dict[str, Any]]:
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if checkpoint_every and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint_every is set")
    output = prepare_output_directory(output, overwrite=overwrite)
    profile = load_profile(config.profile_path)
    trace = load_trace(config.trace_path, profile)
    scenario_trace = load_trace(config.scenario_trace_path, profile)
    evidence_verification = verify_run_evidence(config, profile, trace, scenario_trace)
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
            "policy": policy,
            "replay_checkpoint_resumed": resume_checkpoint_path is not None,
            "replay_checkpoint_mode": "deterministic_replay",
            **(run_metadata or {}),
        },
        evidence_verification=evidence_verification,
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
        privacy_threshold=args.privacy_threshold,
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


def command_sweep(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    base = json.loads(config_path.read_text(encoding="utf-8"))
    # Resolve input assets before moving patched configs into temporary dirs.
    base_config = load_config(config_path)
    base["profile_path"] = str(base_config.profile_path)
    base["trace_path"] = str(base_config.trace_path)
    base["scenario_trace_path"] = str(base_config.scenario_trace_path)
    if base_config.evidence_path is not None:
        base["evidence_path"] = str(base_config.evidence_path)
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    if (
        not isinstance(grid, dict)
        or not grid
        or any(not isinstance(values, list) or not values for values in grid.values())
    ):
        raise ValueError(
            "sweep grid must be a non-empty object of non-empty value arrays"
        )
    keys = sorted(grid)
    case_count = 1
    for key in keys:
        case_count *= len(grid[key])
    root = _prepare_batch_root(
        Path(args.output_root),
        leaf_directories=[Path(f"case-{index:04d}") for index in range(case_count)],
        index_files={"sweep.json"},
        overwrite=args.overwrite,
    )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="privacy-edge-sweep-") as temp_dir:
        for index, values in enumerate(itertools.product(*(grid[key] for key in keys))):
            document = json.loads(json.dumps(base))
            coordinates = dict(zip(keys, values))
            for key, value in coordinates.items():
                _set_dotted(document, key, value)
            patched_path = Path(temp_dir) / f"config-{index:04d}.json"
            patched_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            config = load_config(patched_path)
            policy = args.policy or config.controller.policy
            output = root / f"case-{index:04d}"
            result, artifacts, _ = _execute(
                config,
                policy,
                output,
                overwrite=args.overwrite,
                run_metadata={"case": index, "parameters": coordinates},
            )
            row = _summary_line(result, artifacts, output)
            row["case"] = index
            row["parameters"] = coordinates
            rows.append(row)
    (root / "sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    roots = [Path(item) for item in args.inputs]
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


def _write_new_json(
    path: str | Path, document: dict[str, Any], *, overwrite: bool
) -> Path:
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


def _read_strict_json_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
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
    report = audit_hard_mask_counterfactual(_read_jsonl(args.actions))
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
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"numerical study output root is not empty: {root}")
    if root.exists() and any(root.iterdir()) and args.overwrite:
        raise FileExistsError(
            "numerical study overwrite requires a new root; individual bundles support explicit overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)

    family_id = (
        str(args.family_id).strip()
        if args.family_id is not None
        else f"standalone:{root.name}"
    )
    if not family_id:
        raise ValueError("study family_id must be non-empty")
    family_registration: dict[str, Any] = {
        "schema_version": "1.0",
        "family_id": family_id,
        "load_level": load_level,
        "registered_metrics": list(metrics),
        "registered_policies": list(policies),
        "baseline": args.baseline,
        "family_dimensions": ["load", "metric", "policy_vs_baseline"],
        "registration_sha256": "",
    }
    family_registration["registration_sha256"] = canonical_document_sha256(
        family_registration, "registration_sha256"
    )

    records: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    run_rows: list[dict[str, Any]] = []
    for environment_seed in seeds:
        replication_root = root / "inputs" / f"environment-{environment_seed}"
        replication = generate_numerical_replication(
            args.base_study_root,
            replication_root,
            environment_seed,
            overwrite=False,
        )
        config = load_config(replication.config_path)
        environment_id = f"numerical-environment-{environment_seed}"
        for policy in policies:
            output = root / "runs" / environment_id / policy
            result, artifacts, _ = _execute(
                config,
                policy,
                output,
                overwrite=False,
                run_metadata={
                    "study_id": root.name,
                    "environment_replication_id": environment_id,
                    "pairing_id": "all_tasks",
                    "environment_seed": environment_seed,
                },
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            row = _summary_line(result, artifacts, output)
            row.update(
                environment_id=environment_id,
                environment_seed=environment_seed,
                evaluation_trace_hash=result.trace.trace_hash,
                scenario_trace_hash=result.scenario_trace.trace_hash,
            )
            run_rows.append(row)
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
    report = {
        "study_kind": "frozen_numerical_simulation",
        "real_hardware_measurement": False,
        "environment_seeds": list(seeds),
        "policies": list(policies),
        "baseline": args.baseline,
        "load_level": load_level,
        "statistical_family_registration": family_registration,
        "runs": run_rows,
        "analyses": analyses,
        "multiple_testing": holm_family,
        "study_report_sha256": "",
    }
    report["study_report_sha256"] = canonical_document_sha256(
        report, "study_report_sha256"
    )
    (root / "study_report.json").write_text(
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
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
    generate_numerical.add_argument("--privacy-threshold", type=float, default=0.35)
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

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--config", default="configs/default.json")
    sweep.add_argument("--policy", choices=RUNNABLE_POLICIES)
    sweep.add_argument(
        "--grid",
        required=True,
        help="JSON object mapping dotted config paths to arrays",
    )
    sweep.add_argument("--output-root", required=True)
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
        help="offline counterfactual audit; rejected actions are never executed",
    )
    hard_mask_audit.add_argument("--actions", required=True)
    hard_mask_audit.add_argument("--output", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
