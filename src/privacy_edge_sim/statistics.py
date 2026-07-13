"""Reproducible paired and subject-cluster statistical analysis.

The simulator deliberately keeps environment randomness separate from policy
randomness.  This module applies the same rule to analysis: every stochastic
operation receives an explicit statistical seed and creates a private
``random.Random`` stream derived from that seed.  It never reads or mutates the
module-level random generator.

Policy comparisons use the environment as the independent unit.  Runs must be
paired on ``(environment_id, pairing_id)`` and must agree on both the frozen
evaluation trace identity and the task-set identity before any statistic is
computed.  Subject-cluster bootstrap is exposed separately for privacy and FER
evidence whose independent unit is a subject rather than a frame.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


STATISTICS_SCHEMA_VERSION = "1.0.0"
_DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
_DEFAULT_SIGN_FLIP_PERMUTATIONS = 20_000
_EXACT_SIGN_FLIP_MAX_CLUSTERS = 18


class StatisticalValidationError(ValueError):
    """Raised when statistical evidence is unpaired or internally inconsistent."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.context = context


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise StatisticalValidationError(
            "STAT_NOT_JSON",
            "statistical evidence must be finite and JSON serializable",
        ) from exc
    return text.encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StatisticalValidationError(
            "STAT_FIELD",
            f"{field} must be a non-empty string",
            field=field,
        )
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticalValidationError(
            "STAT_VALUE",
            f"{field} must be a finite number",
            field=field,
        )
    result = float(value)
    if not math.isfinite(result):
        raise StatisticalValidationError(
            "STAT_VALUE",
            f"{field} must be a finite number",
            field=field,
        )
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StatisticalValidationError(
            "STAT_PARAMETER",
            f"{field} must be an integer >= 1",
            field=field,
        )
    return value


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StatisticalValidationError(
            "STAT_SEED",
            "statistical_seed must be a non-negative integer",
        )
    return value


def _derived_rng(seed: int, purpose: str) -> random.Random:
    digest = hashlib.sha256(
        f"privacy-edge-sim|statistics|seed={seed}|purpose={purpose}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise StatisticalValidationError(
            "STAT_EMPTY", "a percentile requires at least one value"
        )
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_standard_deviation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


@dataclass(frozen=True, slots=True)
class PairedRunRecord:
    """One policy result on one frozen environment/workload pairing."""

    environment_id: str
    pairing_id: str
    strategy: str
    metric_value: float
    evaluation_trace_hash: str
    task_identity_hash: str

    @classmethod
    def from_value(
        cls, value: "PairedRunRecord | Mapping[str, Any]"
    ) -> "PairedRunRecord":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise StatisticalValidationError(
                "STAT_RECORD", "each paired record must be an object"
            )
        required = (
            "environment_id",
            "pairing_id",
            "strategy",
            "metric_value",
            "evaluation_trace_hash",
            "task_identity_hash",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise StatisticalValidationError(
                "STAT_RECORD_FIELD",
                "paired record is missing required fields",
                missing=missing,
            )
        return cls(
            environment_id=_nonempty_text(value["environment_id"], "environment_id"),
            pairing_id=_nonempty_text(value["pairing_id"], "pairing_id"),
            strategy=_nonempty_text(value["strategy"], "strategy"),
            metric_value=_finite(value["metric_value"], "metric_value"),
            evaluation_trace_hash=_nonempty_text(
                value["evaluation_trace_hash"], "evaluation_trace_hash"
            ),
            task_identity_hash=_nonempty_text(
                value["task_identity_hash"], "task_identity_hash"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "pairing_id": self.pairing_id,
            "strategy": self.strategy,
            "metric_value": self.metric_value,
            "evaluation_trace_hash": self.evaluation_trace_hash,
            "task_identity_hash": self.task_identity_hash,
        }


def _validate_and_index_paired_records(
    values: Iterable[PairedRunRecord | Mapping[str, Any]],
    baseline_strategy: str,
) -> tuple[
    tuple[PairedRunRecord, ...],
    tuple[str, ...],
    dict[tuple[str, str], dict[str, PairedRunRecord]],
]:
    records = tuple(PairedRunRecord.from_value(value) for value in values)
    if not records:
        raise StatisticalValidationError(
            "STAT_EMPTY", "paired policy analysis requires at least one record"
        )
    baseline = _nonempty_text(baseline_strategy, "baseline_strategy")
    strategies = tuple(sorted({record.strategy for record in records}))
    if baseline not in strategies:
        raise StatisticalValidationError(
            "STAT_BASELINE",
            "baseline strategy is absent from the evidence",
            baseline_strategy=baseline,
        )
    if len(strategies) < 2:
        raise StatisticalValidationError(
            "STAT_STRATEGIES", "at least two strategies are required"
        )

    indexed: dict[tuple[str, str], dict[str, PairedRunRecord]] = {}
    trace_by_environment: dict[str, str] = {}
    for record in records:
        previous_trace = trace_by_environment.setdefault(
            record.environment_id, record.evaluation_trace_hash
        )
        if previous_trace != record.evaluation_trace_hash:
            raise StatisticalValidationError(
                "STAT_ENVIRONMENT_TRACE_MISMATCH",
                "one environment_id refers to multiple evaluation traces",
                environment_id=record.environment_id,
            )
        key = (record.environment_id, record.pairing_id)
        group = indexed.setdefault(key, {})
        if record.strategy in group:
            raise StatisticalValidationError(
                "STAT_DUPLICATE_PAIR",
                "a strategy appears more than once in one environment pairing",
                environment_id=record.environment_id,
                pairing_id=record.pairing_id,
                strategy=record.strategy,
            )
        group[record.strategy] = record

    expected = set(strategies)
    for (environment_id, pairing_id), group in sorted(indexed.items()):
        present = set(group)
        if present != expected:
            raise StatisticalValidationError(
                "STAT_MISSING_PAIR",
                "every environment pairing must contain every strategy",
                environment_id=environment_id,
                pairing_id=pairing_id,
                missing=sorted(expected - present),
                unexpected=sorted(present - expected),
            )
        traces = {record.evaluation_trace_hash for record in group.values()}
        if len(traces) != 1:
            raise StatisticalValidationError(
                "STAT_TRACE_MISMATCH",
                "paired strategies must use the same evaluation trace",
                environment_id=environment_id,
                pairing_id=pairing_id,
            )
        task_identities = {record.task_identity_hash for record in group.values()}
        if len(task_identities) != 1:
            raise StatisticalValidationError(
                "STAT_TASK_IDENTITY_MISMATCH",
                "paired strategies must use exactly the same task set",
                environment_id=environment_id,
                pairing_id=pairing_id,
            )

    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.environment_id,
                item.pairing_id,
                item.strategy,
            ),
        )
    )
    return ordered, strategies, indexed


def environment_cluster_bootstrap_ci(
    environment_values: Mapping[str, float],
    *,
    statistical_seed: int,
    resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    purpose: str = "environment-cluster-bootstrap",
) -> dict[str, Any]:
    """Percentile bootstrap CI with environments as equally weighted clusters."""

    seed = _seed(statistical_seed)
    draws = _positive_integer(resamples, "resamples")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise StatisticalValidationError(
            "STAT_CONFIDENCE", "confidence_level must lie strictly between 0 and 1"
        )
    normalized = {
        _nonempty_text(environment_id, "environment_id"): _finite(
            value, "environment_value"
        )
        for environment_id, value in environment_values.items()
    }
    if len(normalized) < 2:
        raise StatisticalValidationError(
            "STAT_ENVIRONMENT_COUNT",
            "cluster inference requires at least two independent environments",
            environment_count=len(normalized),
        )
    ids = tuple(sorted(normalized))
    rng = _derived_rng(seed, purpose)
    bootstrap: list[float] = []
    for _ in range(draws):
        sampled = [normalized[ids[rng.randrange(len(ids))]] for _ in ids]
        bootstrap.append(math.fsum(sampled) / len(sampled))
    alpha = 1.0 - confidence
    return {
        "method": "environment_cluster_percentile_bootstrap",
        "independent_unit": "environment",
        "confidence_level": confidence,
        "resamples": draws,
        "lower": _percentile(bootstrap, alpha / 2.0),
        "upper": _percentile(bootstrap, 1.0 - alpha / 2.0),
        "statistical_seed": seed,
    }


def two_sided_sign_flip_test(
    environment_differences: Mapping[str, float],
    *,
    statistical_seed: int,
    permutations: int = _DEFAULT_SIGN_FLIP_PERMUTATIONS,
    purpose: str = "environment-sign-flip",
) -> dict[str, Any]:
    """Paired two-sided sign-flip test over environment-level differences."""

    seed = _seed(statistical_seed)
    requested = _positive_integer(permutations, "permutations")
    differences = tuple(
        _finite(value, "environment_difference")
        for _, value in sorted(environment_differences.items())
    )
    if len(differences) < 2:
        raise StatisticalValidationError(
            "STAT_ENVIRONMENT_COUNT",
            "sign-flip inference requires at least two independent environments",
            environment_count=len(differences),
        )
    observed = abs(math.fsum(differences) / len(differences))
    tolerance = max(1e-15, observed * 1e-12)
    exceedances = 0
    if len(differences) <= _EXACT_SIGN_FLIP_MAX_CLUSTERS:
        total = 2 ** len(differences)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            statistic = abs(
                math.fsum(sign * value for sign, value in zip(signs, differences))
                / len(differences)
            )
            exceedances += statistic + tolerance >= observed
        p_value = exceedances / total
        method = "exact_environment_sign_flip"
    else:
        total = requested
        rng = _derived_rng(seed, purpose)
        for _ in range(total):
            statistic = abs(
                math.fsum(
                    (1.0 if rng.getrandbits(1) else -1.0) * value
                    for value in differences
                )
                / len(differences)
            )
            exceedances += statistic + tolerance >= observed
        # The add-one correction prevents a Monte Carlo p-value of zero.
        p_value = (exceedances + 1.0) / (total + 1.0)
        method = "monte_carlo_environment_sign_flip"
    return {
        "method": method,
        "independent_unit": "environment",
        "environment_count": len(differences),
        "observed_absolute_mean_difference": observed,
        "permutations": total,
        "extreme_count": exceedances,
        "p_value": p_value,
        "statistical_seed": seed,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values by hypothesis ID."""

    if not p_values:
        return {}
    normalized = {
        _nonempty_text(hypothesis, "hypothesis_id"): _finite(value, "p_value")
        for hypothesis, value in p_values.items()
    }
    for hypothesis, value in normalized.items():
        if not 0.0 <= value <= 1.0:
            raise StatisticalValidationError(
                "STAT_P_VALUE",
                "p-values must lie in [0, 1]",
                hypothesis=hypothesis,
                p_value=value,
            )
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (hypothesis, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[hypothesis] = running
    return dict(sorted(adjusted.items()))


def apply_holm_family_adjustment(
    analyses: Mapping[str, Mapping[str, Any]],
    *,
    family_name: str = "preregistered_metric_policy_family",
    family_dimensions: Sequence[str] = ("metric", "policy_vs_baseline"),
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply one Holm correction across every registered analysis comparison.

    ``analyze_paired_strategies`` retains its useful within-metric diagnostic,
    while this helper replaces the public ``holm_adjusted_p_value`` with the
    study-wide family result.  Callers may encode an additional preregistered
    load level in the analysis key and declare ``load`` in
    ``family_dimensions``; it then participates in the same family rather than
    being corrected separately.
    """

    name = _nonempty_text(family_name, "family_name")
    dimensions = tuple(
        _nonempty_text(dimension, "family_dimension") for dimension in family_dimensions
    )
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise StatisticalValidationError(
            "STAT_HOLM_DIMENSIONS",
            "Holm family dimensions must be non-empty and unique",
        )
    if not analyses:
        raise StatisticalValidationError(
            "STAT_HOLM_FAMILY", "Holm family must contain at least one analysis"
        )
    normalized: dict[str, dict[str, Any]] = json.loads(
        json.dumps(analyses, ensure_ascii=False, allow_nan=False)
    )
    p_values: dict[str, float] = {}
    locations: dict[str, tuple[str, str]] = {}
    for analysis_id, analysis in sorted(normalized.items()):
        _nonempty_text(analysis_id, "analysis_id")
        comparisons = analysis.get("comparisons")
        if not isinstance(comparisons, dict) or not comparisons:
            raise StatisticalValidationError(
                "STAT_HOLM_COMPARISONS",
                "each Holm-family analysis must contain comparisons",
                analysis_id=analysis_id,
            )
        for comparison_id, comparison in sorted(comparisons.items()):
            if not isinstance(comparison, dict):
                raise StatisticalValidationError(
                    "STAT_HOLM_COMPARISON",
                    "Holm-family comparison must be an object",
                    analysis_id=analysis_id,
                    comparison_id=comparison_id,
                )
            sign_flip = comparison.get("sign_flip_test")
            if not isinstance(sign_flip, dict):
                raise StatisticalValidationError(
                    "STAT_HOLM_P_VALUE",
                    "Holm-family comparison lacks its preregistered sign-flip test",
                    analysis_id=analysis_id,
                    comparison_id=comparison_id,
                )
            hypothesis_id = f"{analysis_id}::{comparison_id}"
            p_values[hypothesis_id] = _finite(
                sign_flip.get("p_value"), "sign_flip_test.p_value"
            )
            locations[hypothesis_id] = (analysis_id, comparison_id)
    adjusted = holm_adjust(p_values)
    for hypothesis_id, adjusted_value in adjusted.items():
        analysis_id, comparison_id = locations[hypothesis_id]
        comparison = normalized[analysis_id]["comparisons"][comparison_id]
        comparison["within_analysis_holm_adjusted_p_value"] = comparison.get(
            "holm_adjusted_p_value"
        )
        comparison["holm_adjusted_p_value"] = adjusted_value
        comparison["holm_family_hypothesis_id"] = hypothesis_id
        comparison["holm_family_name"] = name
    family = {
        "method": "Holm step-down family-wise error correction",
        "family_name": name,
        "family_dimensions": list(dimensions),
        "hypothesis_count": len(p_values),
        "hypothesis_ids": sorted(p_values),
        "raw_p_values": dict(sorted(p_values.items())),
        "adjusted_p_values": adjusted,
    }
    family["family_sha256"] = _sha256(family)
    for analysis in normalized.values():
        manifest = analysis.get("statistics_manifest")
        if not isinstance(manifest, dict):
            raise StatisticalValidationError(
                "STAT_HOLM_MANIFEST",
                "Holm-family analysis lacks its statistics manifest",
            )
        manifest["pre_family_result_core_sha256"] = manifest["result_core_sha256"]
        manifest["multiple_testing_family_sha256"] = family["family_sha256"]
        analysis_core = {
            key: value
            for key, value in analysis.items()
            if key != "statistics_manifest"
        }
        manifest["result_core_sha256"] = _sha256(analysis_core)
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = _sha256(manifest)
    _canonical_json_bytes(normalized)
    return normalized, family


def _document_sha256(document: Mapping[str, Any], hash_field: str) -> str:
    material = dict(document)
    material.pop(hash_field, None)
    return _sha256(material)


def aggregate_preregistered_study_families(
    studies: Sequence[Mapping[str, Any]],
    *,
    input_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Unify load × metric × policy hypotheses into one strict Holm family.

    Every source study must be independently self-hashed and contain a hashed
    registration made before its runs.  Per-load corrections are verified but
    never reused as raw evidence: this function reconstructs the complete
    family from each comparison's sign-flip p-value.
    """

    if not isinstance(studies, Sequence) or len(studies) < 2:
        raise StatisticalValidationError(
            "STAT_LOAD_STUDY_COUNT",
            "load-family aggregation requires at least two study reports",
        )
    if input_identities is not None and len(input_identities) != len(studies):
        raise StatisticalValidationError(
            "STAT_LOAD_INPUT_IDENTITIES",
            "input identities must correspond one-to-one with study reports",
        )
    normalized: list[dict[str, Any]] = []
    for index, study in enumerate(studies):
        if not isinstance(study, Mapping):
            raise StatisticalValidationError(
                "STAT_LOAD_STUDY",
                "each load-family input must be a study-report object",
                index=index,
            )
        normalized.append(
            json.loads(json.dumps(study, ensure_ascii=False, allow_nan=False))
        )

    reference_signature: tuple[Any, ...] | None = None
    family_id: str | None = None
    loads: set[str] = set()
    hypothesis_locations: dict[str, tuple[str, str, str]] = {}
    raw_p_values: dict[str, float] = {}
    input_rows: list[dict[str, Any]] = []
    expected_dimensions = ["load", "metric", "policy_vs_baseline"]
    for index, study in enumerate(normalized):
        expected_study_hash = _nonempty_text(
            study.get("study_report_sha256"), "study_report_sha256"
        )
        actual_study_hash = _document_sha256(study, "study_report_sha256")
        if expected_study_hash != actual_study_hash:
            raise StatisticalValidationError(
                "STAT_LOAD_STUDY_HASH",
                "study report self-hash mismatch",
                index=index,
                expected=expected_study_hash,
                actual=actual_study_hash,
            )
        registration = study.get("statistical_family_registration")
        if not isinstance(registration, dict):
            raise StatisticalValidationError(
                "STAT_LOAD_REGISTRATION",
                "study report lacks preregistered load-family metadata",
                index=index,
            )
        expected_registration_hash = _nonempty_text(
            registration.get("registration_sha256"), "registration_sha256"
        )
        actual_registration_hash = _document_sha256(registration, "registration_sha256")
        if expected_registration_hash != actual_registration_hash:
            raise StatisticalValidationError(
                "STAT_LOAD_REGISTRATION_HASH",
                "load-family registration self-hash mismatch",
                index=index,
            )
        current_family_id = _nonempty_text(registration.get("family_id"), "family_id")
        load_level = _nonempty_text(registration.get("load_level"), "load_level")
        if load_level in loads:
            raise StatisticalValidationError(
                "STAT_LOAD_DUPLICATE",
                "load-family inputs contain a duplicate load level",
                load_level=load_level,
            )
        loads.add(load_level)
        dimensions = registration.get("family_dimensions")
        if dimensions != expected_dimensions:
            raise StatisticalValidationError(
                "STAT_LOAD_DIMENSIONS",
                "load-family registration must preregister load, metric and policy dimensions",
                index=index,
                actual=dimensions,
            )
        metrics = registration.get("registered_metrics")
        policies = registration.get("registered_policies")
        baseline = registration.get("baseline")
        if (
            not isinstance(metrics, list)
            or not metrics
            or len(metrics) != len(set(metrics))
            or any(not isinstance(value, str) or not value for value in metrics)
            or not isinstance(policies, list)
            or len(policies) < 2
            or len(policies) != len(set(policies))
            or any(not isinstance(value, str) or not value for value in policies)
            or not isinstance(baseline, str)
            or baseline not in policies
        ):
            raise StatisticalValidationError(
                "STAT_LOAD_REGISTERED_FAMILY",
                "registered metrics, policies or baseline are invalid",
                index=index,
            )
        signature = (
            current_family_id,
            tuple(metrics),
            tuple(policies),
            baseline,
            tuple(dimensions),
        )
        if reference_signature is None:
            reference_signature = signature
            family_id = current_family_id
        elif signature != reference_signature:
            raise StatisticalValidationError(
                "STAT_LOAD_FAMILY_MISMATCH",
                "load-family registrations disagree on family id or hypothesis domain",
                index=index,
            )

        local_family = study.get("multiple_testing")
        if not isinstance(local_family, dict):
            raise StatisticalValidationError(
                "STAT_LOAD_LOCAL_FAMILY",
                "study report lacks its verified per-load multiple-testing family",
                index=index,
            )
        expected_family_hash = _nonempty_text(
            local_family.get("family_sha256"), "family_sha256"
        )
        if expected_family_hash != _document_sha256(local_family, "family_sha256"):
            raise StatisticalValidationError(
                "STAT_LOAD_LOCAL_FAMILY_HASH",
                "per-load multiple-testing family hash mismatch",
                index=index,
            )
        if local_family.get("family_dimensions") != [
            "metric",
            "policy_vs_baseline",
        ]:
            raise StatisticalValidationError(
                "STAT_LOAD_LOCAL_DIMENSIONS",
                "each source study must first register its metric-policy family",
                index=index,
            )
        analyses = study.get("analyses")
        if not isinstance(analyses, dict) or set(analyses) != set(metrics):
            raise StatisticalValidationError(
                "STAT_LOAD_ANALYSES",
                "study analyses do not match their registered metrics",
                index=index,
            )
        expected_comparisons = {
            f"{policy}__vs__{baseline}" for policy in policies if policy != baseline
        }
        local_raw: dict[str, float] = {}
        for metric in metrics:
            analysis = analyses[metric]
            if not isinstance(analysis, dict) or analysis.get("metric_name") != metric:
                raise StatisticalValidationError(
                    "STAT_LOAD_METRIC",
                    "analysis metric identity differs from its registration",
                    index=index,
                    metric=metric,
                )
            comparisons = analysis.get("comparisons")
            if (
                not isinstance(comparisons, dict)
                or set(comparisons) != expected_comparisons
            ):
                raise StatisticalValidationError(
                    "STAT_LOAD_COMPARISONS",
                    "analysis comparisons do not cover the registered policy family",
                    index=index,
                    metric=metric,
                )
            for comparison_id, comparison in sorted(comparisons.items()):
                if not isinstance(comparison, dict) or not isinstance(
                    comparison.get("sign_flip_test"), dict
                ):
                    raise StatisticalValidationError(
                        "STAT_LOAD_P_VALUE",
                        "registered comparison lacks its sign-flip p-value",
                        index=index,
                        metric=metric,
                        comparison_id=comparison_id,
                    )
                p_value = _finite(
                    comparison["sign_flip_test"].get("p_value"), "p_value"
                )
                if not 0.0 <= p_value <= 1.0:
                    raise StatisticalValidationError(
                        "STAT_P_VALUE", "p-values must lie in [0, 1]"
                    )
                local_id = f"{metric}::{comparison_id}"
                local_raw[local_id] = p_value
                hypothesis_id = json.dumps(
                    [load_level, metric, comparison_id],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if hypothesis_id in hypothesis_locations:
                    raise StatisticalValidationError(
                        "STAT_LOAD_HYPOTHESIS_DUPLICATE",
                        "load-family contains a duplicate hypothesis",
                        hypothesis_id=hypothesis_id,
                    )
                hypothesis_locations[hypothesis_id] = (
                    load_level,
                    metric,
                    comparison_id,
                )
                raw_p_values[hypothesis_id] = p_value
        if (
            local_family.get("hypothesis_count") != len(local_raw)
            or local_family.get("raw_p_values") != dict(sorted(local_raw.items()))
            or local_family.get("hypothesis_ids") != sorted(local_raw)
        ):
            raise StatisticalValidationError(
                "STAT_LOAD_LOCAL_FAMILY_CONTENT",
                "per-load Holm family does not match the registered analyses",
                index=index,
            )
        identity = dict(input_identities[index]) if input_identities is not None else {}
        input_rows.append(
            {
                **identity,
                "load_level": load_level,
                "study_report_sha256": expected_study_hash,
                "registration_sha256": expected_registration_hash,
                "local_family_sha256": expected_family_hash,
            }
        )

    adjusted = holm_adjust(raw_p_values)
    hypotheses = [
        {
            "hypothesis_id": hypothesis_id,
            "load_level": hypothesis_locations[hypothesis_id][0],
            "metric": hypothesis_locations[hypothesis_id][1],
            "comparison_id": hypothesis_locations[hypothesis_id][2],
            "raw_p_value": raw_p_values[hypothesis_id],
            "holm_adjusted_p_value": adjusted[hypothesis_id],
        }
        for hypothesis_id in sorted(raw_p_values)
    ]
    result: dict[str, Any] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "analysis": "aggregate_preregistered_load_metric_policy_holm_family",
        "family_id": family_id,
        "family_dimensions": expected_dimensions,
        "load_levels": sorted(loads),
        "registered_metrics": list(reference_signature[1]),
        "registered_policies": list(reference_signature[2]),
        "baseline": reference_signature[3],
        "study_count": len(normalized),
        "hypothesis_count": len(hypotheses),
        "inputs": sorted(input_rows, key=lambda row: str(row["load_level"])),
        "hypotheses": hypotheses,
        "raw_p_values": dict(sorted(raw_p_values.items())),
        "holm_adjusted_p_values": adjusted,
        "report_sha256": "",
    }
    result["report_sha256"] = _document_sha256(result, "report_sha256")
    _canonical_json_bytes(result)
    return result


def analyze_paired_strategies(
    values: Iterable[PairedRunRecord | Mapping[str, Any]],
    *,
    baseline_strategy: str,
    metric_name: str,
    statistical_seed: int,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    sign_flip_permutations: int = _DEFAULT_SIGN_FLIP_PERMUTATIONS,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare every strategy to a baseline on strictly paired environments.

    Differences are always ``strategy - baseline``.  Pair-level differences
    within an environment are averaged first, so an environment containing
    more task pairings cannot receive greater inferential weight.
    """

    seed = _seed(statistical_seed)
    baseline = _nonempty_text(baseline_strategy, "baseline_strategy")
    metric = _nonempty_text(metric_name, "metric_name")
    records, strategies, indexed = _validate_and_index_paired_records(values, baseline)
    environment_ids = tuple(sorted({key[0] for key in indexed}))
    if len(environment_ids) < 2:
        raise StatisticalValidationError(
            "STAT_ENVIRONMENT_COUNT",
            "paired strategy inference requires at least two environments",
            environment_count=len(environment_ids),
        )

    comparisons: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    for strategy in strategies:
        if strategy == baseline:
            continue
        pair_differences: list[float] = []
        by_environment: dict[str, list[float]] = defaultdict(list)
        for (environment_id, _pairing_id), group in sorted(indexed.items()):
            difference = group[strategy].metric_value - group[baseline].metric_value
            pair_differences.append(difference)
            by_environment[environment_id].append(difference)
        environment_differences = {
            environment_id: math.fsum(differences) / len(differences)
            for environment_id, differences in sorted(by_environment.items())
        }
        cluster_values = tuple(environment_differences.values())
        mean_difference = math.fsum(cluster_values) / len(cluster_values)
        standard_deviation = _sample_standard_deviation(cluster_values)
        if standard_deviation is None or standard_deviation == 0.0:
            effect_size: float | None = None
        else:
            effect_size = mean_difference / standard_deviation
        comparison_id = f"{strategy}__vs__{baseline}"
        bootstrap = environment_cluster_bootstrap_ci(
            environment_differences,
            statistical_seed=seed,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            purpose=f"bootstrap|{metric}|{comparison_id}",
        )
        sign_flip = two_sided_sign_flip_test(
            environment_differences,
            statistical_seed=seed,
            permutations=sign_flip_permutations,
            purpose=f"sign-flip|{metric}|{comparison_id}",
        )
        raw_p_values[comparison_id] = float(sign_flip["p_value"])
        comparisons[comparison_id] = {
            "strategy": strategy,
            "baseline_strategy": baseline,
            "difference_definition": "strategy_minus_baseline",
            "pair_count": len(pair_differences),
            "environment_count": len(environment_differences),
            "mean_pair_difference": math.fsum(pair_differences) / len(pair_differences),
            "mean_environment_difference": mean_difference,
            "environment_difference_sample_sd": standard_deviation,
            "standardized_paired_effect_dz": effect_size,
            "bootstrap_ci": bootstrap,
            "sign_flip_test": sign_flip,
        }

    adjusted = holm_adjust(raw_p_values)
    for comparison_id, comparison in comparisons.items():
        comparison["holm_adjusted_p_value"] = adjusted[comparison_id]

    normalized_input = [record.to_json() for record in records]
    result_core: dict[str, Any] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "analysis": "paired_strategy_comparison",
        "metric_name": metric,
        "baseline_strategy": baseline,
        "difference_definition": "strategy_minus_baseline",
        "independent_unit": "environment",
        "strategies": list(strategies),
        "environment_count": len(environment_ids),
        "pairing_count": len(indexed),
        "statistical_seed": seed,
        "bootstrap_resamples": _positive_integer(
            bootstrap_resamples, "bootstrap_resamples"
        ),
        "sign_flip_permutations_requested": _positive_integer(
            sign_flip_permutations, "sign_flip_permutations"
        ),
        "confidence_level": _finite(confidence_level, "confidence_level"),
        "comparisons": dict(sorted(comparisons.items())),
    }
    manifest: dict[str, Any] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "analysis": "paired_strategy_comparison",
        "input_record_count": len(records),
        "input_sha256": _sha256(normalized_input),
        "result_core_sha256": _sha256(result_core),
        "environment_ids": list(environment_ids),
        "evaluation_trace_hashes": sorted(
            {record.evaluation_trace_hash for record in records}
        ),
        "task_identity_hashes": sorted(
            {record.task_identity_hash for record in records}
        ),
        "statistical_seed": seed,
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    result_core["statistics_manifest"] = manifest
    # This final validation guarantees callers can pass the result directly to
    # a strict JSON writer without NaN, tuple, dataclass, or custom objects.
    _canonical_json_bytes(result_core)
    return result_core


SubjectCluster: TypeAlias = tuple[Mapping[str, Any], ...]
SubjectStatistic: TypeAlias = Callable[[Sequence[SubjectCluster]], float]


def subject_cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    subject_key: str,
    statistic: SubjectStatistic,
    statistic_name: str,
    statistical_seed: int,
    resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap arbitrary evidence with subjects as the resampling unit.

    ``statistic`` receives a sequence of subject clusters.  A repeated cluster
    in a bootstrap sample appears repeatedly in that sequence, and every row
    belonging to the subject stays together.  This supports privacy ratios,
    FER metrics, and other statistics without pretending frames are
    independent.
    """

    key = _nonempty_text(subject_key, "subject_key")
    name = _nonempty_text(statistic_name, "statistic_name")
    seed = _seed(statistical_seed)
    draws = _positive_integer(resamples, "resamples")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise StatisticalValidationError(
            "STAT_CONFIDENCE", "confidence_level must lie strictly between 0 and 1"
        )
    if not callable(statistic):
        raise StatisticalValidationError("STAT_CALLABLE", "statistic must be callable")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StatisticalValidationError(
                "STAT_SUBJECT_ROW",
                "subject evidence rows must be objects",
                index=index,
            )
        if key not in row:
            raise StatisticalValidationError(
                "STAT_SUBJECT_KEY",
                "subject evidence row is missing its subject identifier",
                index=index,
                subject_key=key,
            )
        subject_id = _nonempty_text(row[key], key)
        normalized = dict(row)
        _canonical_json_bytes(normalized)
        grouped[subject_id].append(normalized)
        normalized_rows.append(normalized)
    if len(grouped) < 2:
        raise StatisticalValidationError(
            "STAT_SUBJECT_COUNT",
            "subject-cluster bootstrap requires at least two subjects",
            subject_count=len(grouped),
        )

    subject_ids = tuple(sorted(grouped))
    clusters: tuple[SubjectCluster, ...] = tuple(
        tuple(sorted(grouped[subject_id], key=_sha256)) for subject_id in subject_ids
    )
    estimate = _finite(statistic(clusters), "statistic estimate")
    rng = _derived_rng(seed, f"subject-bootstrap|{name}")
    bootstrap: list[float] = []
    for _ in range(draws):
        sampled = tuple(clusters[rng.randrange(len(clusters))] for _ in clusters)
        bootstrap.append(_finite(statistic(sampled), "bootstrap statistic"))
    alpha = 1.0 - confidence
    normalized_rows.sort(key=lambda row: (_nonempty_text(row[key], key), _sha256(row)))
    result = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "analysis": "subject_cluster_bootstrap",
        "statistic_name": name,
        "independent_unit": "subject",
        "subject_key": key,
        "subject_count": len(clusters),
        "row_count": len(normalized_rows),
        "estimate": estimate,
        "confidence_level": confidence,
        "resamples": draws,
        "ci_lower": _percentile(bootstrap, alpha / 2.0),
        "ci_upper": _percentile(bootstrap, 1.0 - alpha / 2.0),
        "statistical_seed": seed,
        "input_sha256": _sha256(normalized_rows),
    }
    result["manifest_sha256"] = _sha256(result)
    _canonical_json_bytes(result)
    return result


__all__ = [
    "PairedRunRecord",
    "STATISTICS_SCHEMA_VERSION",
    "StatisticalValidationError",
    "analyze_paired_strategies",
    "aggregate_preregistered_study_families",
    "apply_holm_family_adjustment",
    "environment_cluster_bootstrap_ci",
    "holm_adjust",
    "subject_cluster_bootstrap",
    "two_sided_sign_flip_test",
]
