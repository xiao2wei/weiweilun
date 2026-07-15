"""Strict environment-paired analysis for registered sensitivity sweeps.

The generic aggregate command is intentionally flat and is useful for browsing
results, but it is not an inferential unit.  This module treats one frozen
environment as the independent unit and the parameter levels within that
environment as paired observations.  It accepts only completed, manifest-bound
formal sweeps created from one registered exploratory sensitivity plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .manifest import sha256_file
from .statistics import analyze_paired_strategies


SENSITIVITY_ANALYSIS_SCHEMA_VERSION = "1.0.0"


class SensitivityAnalysisError(ValueError):
    """Raised when sensitivity inputs are incomplete, unpaired or unverified."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SensitivityAnalysisError(
            "sensitivity evidence must be finite strict JSON"
        ) from exc


def _document_sha256(document: Mapping[str, Any], field: str) -> str:
    material = dict(document)
    material.pop(field, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SensitivityAnalysisError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensitivityAnalysisError(f"cannot read strict JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise SensitivityAnalysisError(f"JSON root must be an object: {path}")
    _canonical_json_bytes(value)
    return value


def _strict_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensitivityAnalysisError(f"cannot read strict JSON array: {path}") from exc
    if not isinstance(value, list):
        raise SensitivityAnalysisError(f"JSON root must be an array: {path}")
    _canonical_json_bytes(value)
    return value


def _require_self_hash(
    document: Mapping[str, Any], field: str, *, context: str
) -> str:
    expected = document.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise SensitivityAnalysisError(f"{context} is missing {field}")
    actual = _document_sha256(document, field)
    if expected != actual:
        raise SensitivityAnalysisError(f"{context} {field} mismatch")
    return expected


def _require_clean_source(record: Any, *, context: str) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise SensitivityAnalysisError(f"{context} source preflight is missing")
    if (
        record.get("require_clean_source") is not True
        or record.get("requirement_status") != "passed"
        or record.get("source_commit_reproducible") is not True
        or record.get("source_git_dirty") is not False
    ):
        raise SensitivityAnalysisError(
            f"{context} was not produced from a verified clean source tree"
        )
    commit = record.get("git_commit")
    source_object = record.get("source_git_object")
    if not isinstance(commit, str) or len(commit) != 40:
        raise SensitivityAnalysisError(f"{context} git commit is malformed")
    if not isinstance(source_object, str) or len(source_object) != 40:
        raise SensitivityAnalysisError(f"{context} source Git object is malformed")
    try:
        int(commit, 16)
        int(source_object, 16)
    except ValueError as exc:
        raise SensitivityAnalysisError(
            f"{context} source identity is not hexadecimal"
        ) from exc
    return commit, source_object


def _nested_metric(summary: Mapping[str, Any], path: str) -> float:
    current: Any = summary
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise SensitivityAnalysisError(f"summary metric is missing: {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise SensitivityAnalysisError(f"summary metric must be numeric: {path}")
    value = float(current)
    if not math.isfinite(value):
        raise SensitivityAnalysisError(f"summary metric must be finite: {path}")
    return value


def _registered_coordinates(factor: Mapping[str, Any]) -> list[dict[str, Any]]:
    application = factor.get("application")
    if application == "config_sweep":
        path = factor.get("path")
        values = factor.get("values")
        if not isinstance(path, str) or not path or not isinstance(values, list):
            raise SensitivityAnalysisError("registered config_sweep is malformed")
        coordinates = [{path: value} for value in values]
    elif application == "paired_config_override":
        paths = factor.get("paths")
        values = factor.get("paired_values_gpu_s")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or not isinstance(values, list)
            or not all(
                isinstance(level, list) and len(level) == len(paths)
                for level in values
            )
        ):
            raise SensitivityAnalysisError(
                "registered paired_config_override is malformed"
            )
        coordinates = [dict(zip(paths, level, strict=True)) for level in values]
    else:
        raise SensitivityAnalysisError(
            "analysis supports completed config_sweep or paired_config_override roots"
        )
    signatures = [_canonical_json_bytes(coordinate) for coordinate in coordinates]
    if len(coordinates) < 2 or len(set(signatures)) != len(coordinates):
        raise SensitivityAnalysisError(
            "registered sensitivity factor requires at least two unique levels"
        )
    return coordinates


def _reference_coordinate(
    factor_name: str,
    factor: Mapping[str, Any],
    coordinates: list[dict[str, Any]],
    plan_reference: Mapping[str, Any],
) -> dict[str, Any]:
    reference_value = factor.get("reference_value")
    if reference_value is None and factor.get("application") == "config_sweep":
        reference_value = plan_reference.get(str(factor.get("path")))
    if factor.get("application") == "config_sweep":
        candidate = {str(factor["path"]): reference_value}
    else:
        paths = factor["paths"]
        if not isinstance(reference_value, list) or len(reference_value) != len(paths):
            raise SensitivityAnalysisError(
                f"factor {factor_name} requires a registered reference_value array"
            )
        candidate = dict(zip(paths, reference_value, strict=True))
    signature = _canonical_json_bytes(candidate)
    if signature not in {_canonical_json_bytes(item) for item in coordinates}:
        raise SensitivityAnalysisError(
            f"factor {factor_name} reference level is absent from its registered grid"
        )
    return candidate


def _load_case(
    *,
    root: Path,
    row: Mapping[str, Any],
    index: int,
    registration_hash: str,
    environment_seed: int,
    metric_name: str,
) -> dict[str, Any]:
    relative = f"case-{index:04d}"
    case_dir = root / relative
    manifest_path = case_dir / "manifest.json"
    summary_path = case_dir / "summary.json"
    manifest = _strict_object(manifest_path)
    manifest_hash = _require_self_hash(
        manifest, "manifest_sha256", context=f"case manifest {manifest_path}"
    )
    if row.get("manifest_sha256") != manifest_hash:
        raise SensitivityAnalysisError(f"indexed manifest hash mismatch: {manifest_path}")
    run_metadata = manifest.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        raise SensitivityAnalysisError(f"run metadata is missing: {manifest_path}")
    if run_metadata.get("sensitivity_registration_record_sha256") != registration_hash:
        raise SensitivityAnalysisError(
            f"case is not bound to its sensitivity registration: {manifest_path}"
        )
    if run_metadata.get("environment_seed") not in (None, environment_seed):
        raise SensitivityAnalysisError(
            f"case environment seed conflicts with sweep registration: {manifest_path}"
        )
    source_identity = _require_clean_source(
        manifest.get("source_cleanliness_preflight"), context=str(manifest_path)
    )
    outputs = manifest.get("outputs")
    files = outputs.get("files") if isinstance(outputs, Mapping) else None
    summary_record = files.get("summary.json") if isinstance(files, Mapping) else None
    expected_summary_hash = (
        summary_record.get("sha256") if isinstance(summary_record, Mapping) else None
    )
    if not isinstance(expected_summary_hash, str) or len(expected_summary_hash) != 64:
        raise SensitivityAnalysisError(
            f"manifest-bound summary hash is missing: {manifest_path}"
        )
    if sha256_file(summary_path) != expected_summary_hash:
        raise SensitivityAnalysisError(f"summary hash mismatch: {summary_path}")
    summary = _strict_object(summary_path)
    trace_identity = manifest.get("trace_identity")
    if not isinstance(trace_identity, Mapping):
        raise SensitivityAnalysisError(f"trace identity is missing: {manifest_path}")
    trace_hash = trace_identity.get("trace_hash")
    if not isinstance(trace_hash, str) or len(trace_hash) != 64:
        raise SensitivityAnalysisError(f"trace hash is malformed: {manifest_path}")
    if trace_identity.get("seed") != environment_seed:
        raise SensitivityAnalysisError(
            f"trace seed does not match registered environment: {manifest_path}"
        )
    scenario = manifest.get("scenario_trace_identity")
    scenario_hash = scenario.get("trace_hash") if isinstance(scenario, Mapping) else None
    if not isinstance(scenario_hash, str) or len(scenario_hash) != 64:
        raise SensitivityAnalysisError(
            f"scenario trace hash is malformed: {manifest_path}"
        )
    frozen_inputs = manifest.get("frozen_input_assets")
    frozen_assets = (
        frozen_inputs.get("assets") if isinstance(frozen_inputs, Mapping) else None
    )
    if not isinstance(frozen_assets, Mapping):
        raise SensitivityAnalysisError(
            f"manifest frozen input identities are missing: {manifest_path}"
        )
    frozen_content: dict[str, tuple[str, str]] = {}
    for role in ("profile", "scenario_trace", "evidence"):
        asset = frozen_assets.get(role)
        raw_hash = asset.get("raw_sha256") if isinstance(asset, Mapping) else None
        declared_hash = (
            asset.get("declared_content_hash") if isinstance(asset, Mapping) else None
        )
        if (
            not isinstance(asset, Mapping)
            or asset.get("status") != "captured"
            or not isinstance(raw_hash, str)
            or len(raw_hash) != 64
            or not isinstance(declared_hash, str)
            or len(declared_hash) != 64
        ):
            raise SensitivityAnalysisError(
                f"manifest frozen {role} identity is malformed: {manifest_path}"
            )
        frozen_content[role] = (raw_hash, declared_hash)
    if frozen_content["scenario_trace"][1] != scenario_hash:
        raise SensitivityAnalysisError(
            f"scenario trace identity conflicts with frozen inputs: {manifest_path}"
        )
    return {
        "metric_value": _nested_metric(summary, metric_name),
        "trace_hash": trace_hash,
        "scenario_trace_hash": scenario_hash,
        "profile_content_identity": frozen_content["profile"],
        "scenario_content_identity": frozen_content["scenario_trace"],
        "evidence_content_identity": frozen_content["evidence"],
        "source_identity": source_identity,
        "summary_sha256": expected_summary_hash,
        "manifest_sha256": manifest_hash,
    }


def analyze_registered_sensitivity_sweeps(
    sweep_roots: Iterable[str | Path],
    *,
    sensitivity_path: str | Path,
    metric_name: str,
    statistical_seed: int,
    bootstrap_resamples: int = 2_000,
    sign_flip_permutations: int = 20_000,
    confidence_level: float = 0.95,
    analysis_source_preflight: Mapping[str, Any],
    sensitivity_source_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and analyze complete registered sweeps by environment.

    The caller is responsible for whole-sweep structural validation before this
    function is invoked.  This function adds registration, source, artifact and
    statistical-pairing validation.
    """

    metric = metric_name.strip() if isinstance(metric_name, str) else ""
    if not metric:
        raise SensitivityAnalysisError("metric_name must be non-empty")
    roots = [Path(item).resolve() for item in sweep_roots]
    if not roots:
        raise SensitivityAnalysisError("at least one completed sweep is required")
    if len(set(roots)) != len(roots):
        raise SensitivityAnalysisError("duplicate sweep root")
    analysis_identity = _require_clean_source(
        analysis_source_preflight, context="sensitivity analysis"
    )
    registration_identity = _require_clean_source(
        sensitivity_source_preflight, context="sensitivity registration"
    )
    if analysis_identity[0] != registration_identity[0]:
        raise SensitivityAnalysisError(
            "analysis code and sensitivity registration must belong to the same commit"
        )

    plan_path = Path(sensitivity_path).resolve()
    plan = _strict_object(plan_path)
    plan_file_hash = sha256_file(plan_path)
    plan_content_hash = hashlib.sha256(_canonical_json_bytes(plan)).hexdigest()
    factors = plan.get("factors")
    reference = plan.get("reference")
    rules = plan.get("rules")
    if not all(isinstance(item, Mapping) for item in (factors, reference, rules)):
        raise SensitivityAnalysisError("sensitivity plan is malformed")
    if rules.get("confirmatory") is not False:
        raise SensitivityAnalysisError(
            "sensitivity family must be registered as exploratory/non-confirmatory"
        )
    registered_seeds = rules.get("environment_seeds")
    if (
        not isinstance(registered_seeds, list)
        or len(registered_seeds) < 2
        or any(type(seed) is not int or seed < 0 for seed in registered_seeds)
        or len(set(registered_seeds)) != len(registered_seeds)
    ):
        raise SensitivityAnalysisError(
            "registered sensitivity environment seeds are invalid"
        )

    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    root_registration: dict[tuple[str, int], dict[str, Any]] = {}
    for root in sorted(roots, key=lambda item: item.as_posix()):
        diagnostics = _strict_object(root / "sweep_diagnostics.json")
        diagnostics_hash = _require_self_hash(
            diagnostics,
            "report_sha256",
            context=f"sweep diagnostics {root}",
        )
        registration = diagnostics.get("sensitivity_registration")
        if not isinstance(registration, dict) or registration.get("status") != "VERIFIED":
            raise SensitivityAnalysisError(
                f"sweep lacks a verified sensitivity registration: {root}"
            )
        registration_hash = _require_self_hash(
            registration,
            "record_sha256",
            context=f"sweep sensitivity registration {root}",
        )
        if (
            registration.get("sensitivity_file_sha256") != plan_file_hash
            or registration.get("sensitivity_content_sha256") != plan_content_hash
        ):
            raise SensitivityAnalysisError(
                f"sweep was created from a different sensitivity plan: {root}"
            )
        if registration.get("experiment_content_sha256") != reference.get(
            "experiment_registration_content_sha256"
        ):
            raise SensitivityAnalysisError(
                f"sweep experiment registration conflicts with sensitivity plan: {root}"
            )
        registration_source = registration.get("sensitivity_source_preflight")
        experiment_source = registration.get("experiment_source_preflight")
        if _require_clean_source(
            registration_source, context=f"sweep sensitivity file {root}"
        )[0] != analysis_identity[0]:
            raise SensitivityAnalysisError(
                f"sweep sensitivity file source identity differs: {root}"
            )
        if _require_clean_source(
            experiment_source, context=f"sweep experiment file {root}"
        )[0] != analysis_identity[0]:
            raise SensitivityAnalysisError(
                f"sweep experiment file source identity differs: {root}"
            )
        factor_name = registration.get("registration_factor")
        environment_seed = registration.get("environment_seed")
        if not isinstance(factor_name, str) or factor_name not in factors:
            raise SensitivityAnalysisError(f"unregistered sensitivity factor: {root}")
        if type(environment_seed) is not int:
            raise SensitivityAnalysisError(f"sweep environment seed is invalid: {root}")
        factor_record = factors[factor_name]
        if not isinstance(factor_record, Mapping):
            raise SensitivityAnalysisError(f"factor {factor_name} is malformed")
        expected_paths = (
            [factor_record.get("path")]
            if factor_record.get("application") == "config_sweep"
            else factor_record.get("paths")
        )
        if registration.get("factor_paths") != expected_paths:
            raise SensitivityAnalysisError(
                f"sweep factor paths conflict with sensitivity plan: {root}"
            )
        if (
            registration.get("reference_scale") != reference.get("scale")
            or registration.get("reference_regime") != reference.get("regime")
        ):
            raise SensitivityAnalysisError(
                f"sweep reference scale/regime conflicts with sensitivity plan: {root}"
            )
        if environment_seed in grouped[factor_name]:
            raise SensitivityAnalysisError(
                f"duplicate sweep for factor={factor_name}, seed={environment_seed}"
            )
        grouped[factor_name][environment_seed] = root
        root_registration[(factor_name, environment_seed)] = {
            "record": registration,
            "record_sha256": registration_hash,
            "diagnostics_sha256": diagnostics_hash,
            "sweep_rows_sha256": diagnostics.get("sweep_rows_sha256"),
        }

    expected_seed_set = set(registered_seeds)
    expected_factor_set = {
        name
        for name, factor in factors.items()
        if isinstance(factor, Mapping)
        and factor.get("application")
        in {"config_sweep", "paired_config_override"}
    }
    if set(grouped) != expected_factor_set:
        raise SensitivityAnalysisError(
            "registered sweep factor family is incomplete; "
            f"missing={sorted(expected_factor_set - set(grouped))}, "
            f"unexpected={sorted(set(grouped) - expected_factor_set)}"
        )
    source_identities: set[tuple[str, str]] = set()
    profile_content_identities: set[tuple[str, str]] = set()
    scenario_content_identities: set[tuple[str, str]] = set()
    evidence_content_identities: set[tuple[str, str]] = set()
    factor_reports: dict[str, Any] = {}
    for factor_name in sorted(grouped):
        by_seed = grouped[factor_name]
        actual_seed_set = set(by_seed)
        if actual_seed_set != expected_seed_set:
            raise SensitivityAnalysisError(
                f"factor {factor_name} environment set is incomplete; "
                f"missing={sorted(expected_seed_set - actual_seed_set)}, "
                f"unexpected={sorted(actual_seed_set - expected_seed_set)}"
            )
        factor = factors[factor_name]
        if not isinstance(factor, Mapping):
            raise SensitivityAnalysisError(f"factor {factor_name} is malformed")
        coordinates = _registered_coordinates(factor)
        coordinate_signatures = [_canonical_json_bytes(item) for item in coordinates]
        reference_coordinate = _reference_coordinate(
            factor_name, factor, coordinates, reference
        )
        reference_signature = _canonical_json_bytes(reference_coordinate)
        level_ids = {
            signature: f"level:{signature.decode('utf-8')}"
            for signature in coordinate_signatures
        }
        records: list[dict[str, Any]] = []
        inputs: list[dict[str, Any]] = []
        factor_policy: str | None = None
        trace_hashes_by_seed: dict[int, str] = {}
        scenario_hashes_by_seed: dict[int, str] = {}
        for environment_seed in registered_seeds:
            root = by_seed[environment_seed]
            rows = _strict_array(root / "sweep.json")
            if len(rows) != len(coordinates):
                raise SensitivityAnalysisError(
                    f"factor {factor_name}, seed {environment_seed} has incomplete levels"
                )
            actual_signatures: list[bytes] = []
            manifest_hashes: list[str] = []
            summary_hashes: list[str] = []
            environment_trace_hash: str | None = None
            environment_scenario_hash: str | None = None
            registration = root_registration[(factor_name, environment_seed)]
            record = registration["record"]
            if record.get("factor_values") != (
                factor.get("values")
                if factor.get("application") == "config_sweep"
                else factor.get("paired_values_gpu_s")
            ):
                raise SensitivityAnalysisError(
                    f"factor values conflict with sensitivity plan: {root}"
                )
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise SensitivityAnalysisError(f"sweep row is malformed: {root}")
                parameters = row.get("parameters")
                if not isinstance(parameters, dict):
                    raise SensitivityAnalysisError(
                        f"sweep parameters are malformed: {root}"
                    )
                signature = _canonical_json_bytes(parameters)
                actual_signatures.append(signature)
                policy = row.get("policy")
                if not isinstance(policy, str) or not policy:
                    raise SensitivityAnalysisError(f"sweep policy is missing: {root}")
                if factor_policy is None:
                    factor_policy = policy
                elif factor_policy != policy:
                    raise SensitivityAnalysisError(
                        f"factor {factor_name} mixes policies across levels/environments"
                    )
                registered_policy = factor.get("policy")
                if registered_policy is not None and policy != registered_policy:
                    raise SensitivityAnalysisError(
                        f"factor {factor_name} policy differs from its registration"
                    )
                case = _load_case(
                    root=root,
                    row=row,
                    index=index,
                    registration_hash=registration["record_sha256"],
                    environment_seed=environment_seed,
                    metric_name=metric,
                )
                source_identities.add(case["source_identity"])
                profile_content_identities.add(case["profile_content_identity"])
                scenario_content_identities.add(case["scenario_content_identity"])
                evidence_content_identities.add(case["evidence_content_identity"])
                if case["source_identity"] != analysis_identity:
                    raise SensitivityAnalysisError(
                        f"case source identity differs from analysis source: {root}"
                    )
                if environment_trace_hash is None:
                    environment_trace_hash = case["trace_hash"]
                    environment_scenario_hash = case["scenario_trace_hash"]
                elif (
                    environment_trace_hash != case["trace_hash"]
                    or environment_scenario_hash != case["scenario_trace_hash"]
                ):
                    raise SensitivityAnalysisError(
                        f"factor {factor_name}, seed {environment_seed} levels are not trace-paired"
                    )
                records.append(
                    {
                        "environment_id": f"numerical-environment-{environment_seed}",
                        "pairing_id": factor_name,
                        "strategy": level_ids[signature],
                        "metric_value": case["metric_value"],
                        "evaluation_trace_hash": case["trace_hash"],
                        "task_identity_hash": case["trace_hash"],
                    }
                )
                manifest_hashes.append(case["manifest_sha256"])
                summary_hashes.append(case["summary_sha256"])
            if actual_signatures != coordinate_signatures:
                raise SensitivityAnalysisError(
                    f"factor {factor_name}, seed {environment_seed} coordinates differ "
                    "from the registered ordered grid"
                )
            assert environment_trace_hash is not None
            assert environment_scenario_hash is not None
            trace_hashes_by_seed[environment_seed] = environment_trace_hash
            scenario_hashes_by_seed[environment_seed] = environment_scenario_hash
            inputs.append(
                {
                    "environment_seed": environment_seed,
                    "sweep_root": str(root),
                    "sweep_diagnostics_sha256": registration[
                        "diagnostics_sha256"
                    ],
                    "sweep_rows_sha256": registration["sweep_rows_sha256"],
                    "registration_record_sha256": registration["record_sha256"],
                    "evaluation_trace_sha256": environment_trace_hash,
                    "scenario_trace_sha256": environment_scenario_hash,
                    "manifest_sha256": manifest_hashes,
                    "summary_sha256": summary_hashes,
                }
            )
        baseline = level_ids[reference_signature]
        analysis = analyze_paired_strategies(
            records,
            baseline_strategy=baseline,
            metric_name=metric,
            statistical_seed=statistical_seed,
            bootstrap_resamples=bootstrap_resamples,
            sign_flip_permutations=sign_flip_permutations,
            confidence_level=confidence_level,
        )
        factor_reports[factor_name] = {
            "application": factor.get("application"),
            "policy": factor_policy,
            "coordinate_levels": [
                {
                    "level_id": level_ids[signature],
                    "parameters": coordinate,
                    "is_reference": signature == reference_signature,
                }
                for coordinate, signature in zip(
                    coordinates, coordinate_signatures, strict=True
                )
            ],
            "reference_level_id": baseline,
            "environment_count": len(registered_seeds),
            "environment_seeds": list(registered_seeds),
            "evaluation_trace_hashes_by_seed": {
                str(seed): trace_hashes_by_seed[seed] for seed in registered_seeds
            },
            "scenario_trace_hashes_by_seed": {
                str(seed): scenario_hashes_by_seed[seed] for seed in registered_seeds
            },
            "input_sweeps": inputs,
            "paired_records_sha256": hashlib.sha256(
                _canonical_json_bytes(records)
            ).hexdigest(),
            "paired_analysis": analysis,
            "multiplicity": {
                "scope": "levels_vs_registered_reference_within_this_factor",
                "method": "Holm",
                "exploratory": True,
            },
        }

    shared_evaluation_hashes: dict[str, str] = {}
    shared_scenario_hashes: dict[str, str] = {}
    for environment_seed in registered_seeds:
        evaluation_hashes = {
            report["evaluation_trace_hashes_by_seed"][str(environment_seed)]
            for report in factor_reports.values()
        }
        scenario_hashes = {
            report["scenario_trace_hashes_by_seed"][str(environment_seed)]
            for report in factor_reports.values()
        }
        if len(evaluation_hashes) != 1 or len(scenario_hashes) != 1:
            raise SensitivityAnalysisError(
                f"environment {environment_seed} is not trace-paired across factors"
            )
        shared_evaluation_hashes[str(environment_seed)] = next(
            iter(evaluation_hashes)
        )
        shared_scenario_hashes[str(environment_seed)] = next(iter(scenario_hashes))

    if len(source_identities) != 1 or next(iter(source_identities)) != analysis_identity:
        raise SensitivityAnalysisError(
            "all sensitivity cases must share one verified source identity"
        )
    if len(profile_content_identities) != 1:
        raise SensitivityAnalysisError(
            "all sensitivity cases must share one frozen profile identity"
        )
    if len(evidence_content_identities) != 1:
        raise SensitivityAnalysisError(
            "all sensitivity cases must share one frozen evidence identity"
        )
    if len(scenario_content_identities) != 1 or len(
        set(shared_scenario_hashes.values())
    ) != 1:
        raise SensitivityAnalysisError(
            "all sensitivity environments must share the registered frozen scenario trace"
        )
    profile_raw, profile_declared = next(iter(profile_content_identities))
    scenario_raw, scenario_declared = next(iter(scenario_content_identities))
    evidence_raw, evidence_declared = next(iter(evidence_content_identities))
    result: dict[str, Any] = {
        "schema_version": SENSITIVITY_ANALYSIS_SCHEMA_VERSION,
        "analysis": "registered_environment_paired_sensitivity",
        "study_role": "exploratory_non_confirmatory",
        "synthetic_numerical_only": True,
        "independent_unit": "environment",
        "metric_name": metric,
        "environment_seeds": list(registered_seeds),
        "factor_count": len(factor_reports),
        "factors": factor_reports,
        "shared_evaluation_trace_hashes_by_seed": shared_evaluation_hashes,
        "shared_scenario_trace_hashes_by_seed": shared_scenario_hashes,
        "shared_frozen_inputs": {
            "profile": {
                "raw_sha256": profile_raw,
                "declared_content_hash": profile_declared,
            },
            "scenario_trace": {
                "raw_sha256": scenario_raw,
                "declared_content_hash": scenario_declared,
            },
            "evidence": {
                "raw_sha256": evidence_raw,
                "declared_content_hash": evidence_declared,
            },
        },
        "multiplicity": {
            "within_factor": "Holm across level-vs-reference contrasts",
            "across_factors": "none; factors are separate exploratory analyses",
            "primary_or_secondary_family_membership": False,
        },
        "interpretation": (
            "Differences are level minus the registered reference. Each environment "
            "receives equal inferential weight. Results are exploratory synthetic "
            "simulation evidence and are not confirmatory hardware claims."
        ),
        "sensitivity_registration": {
            "path": str(plan_path),
            "file_sha256": plan_file_hash,
            "content_sha256": plan_content_hash,
            "source_cleanliness_preflight": dict(sensitivity_source_preflight),
        },
        "analysis_source_cleanliness_preflight": dict(analysis_source_preflight),
        "report_sha256": "",
    }
    result["report_sha256"] = _document_sha256(result, "report_sha256")
    _canonical_json_bytes(result)
    return result


__all__ = [
    "SENSITIVITY_ANALYSIS_SCHEMA_VERSION",
    "SensitivityAnalysisError",
    "analyze_registered_sensitivity_sweeps",
]
