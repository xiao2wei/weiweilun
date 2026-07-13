"""Frozen privacy/model profiles and conservative hard-safety queries.

The online simulator loads one :class:`FrozenProfileBundle`.  The loader
verifies the canonical content hash before constructing deeply immutable
objects.  There is deliberately no update API: changing any nested profile
value requires writing a new file with a new hash and starting a new run.

Privacy values in a profile are empirical, subject-cluster bounds for a
pre-registered population and attacker protocol.  They are not an absolute
anonymity claim.  Synthetic fixtures are explicitly rejected as formal
evidence by the metadata checks below.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .enums import ReasonCode
from .errors import ProfileValidationError


PRIVACY_RISK_TYPES: tuple[str, ...] = ("identity", "verification", "link")
PARAMETER_SOURCE_CATEGORIES: frozenset[str] = frozenset(
    {
        "measured",
        "public_specification",
        "literature_range",
        "engineering_assumption",
        "stress_test_boundary",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/\-]{0,127}$")


def deep_freeze(value: Any) -> Any:
    """Recursively convert JSON-like values into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return a JSON-serializable copy of a deeply frozen value."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(thaw_json(item) for item in value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by profile, trace and manifest hashes."""

    try:
        encoded = json.dumps(
            thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(
            "CANONICAL_JSON_INVALID",
            "value cannot be represented as finite canonical JSON",
            error=str(exc),
        ) from exc
    return encoded.encode("utf-8")


def canonical_document_sha256(document: Mapping[str, Any], hash_field: str) -> str:
    """Hash a document while excluding its self-referential top-level field."""

    material = dict(document)
    material.pop(hash_field, None)
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileValidationError(
                "JSON_DUPLICATE_KEY",
                "duplicate JSON object key is not auditable",
                key=key,
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise ProfileValidationError(
        "JSON_NONFINITE",
        "NaN and infinity are not valid physical/profile values",
        token=token,
    )


def _validate_text_health(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        bad = [ord(ch) for ch in value if ord(ch) < 32 or 0x7F <= ord(ch) <= 0x9F]
        if bad:
            raise ProfileValidationError(
                "JSON_CONTROL_CHARACTER",
                "decoded JSON contains a control character",
                path=path,
                codepoints=[f"U+{code:04X}" for code in bad],
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_text_health(str(key), f"{path}.<key>")
            _validate_text_health(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_text_health(item, f"{path}[{index}]")


def load_strict_json(path: str | Path, *, purpose: str = "profile") -> dict[str, Any]:
    """Read strict UTF-8 JSON, rejecting duplicate keys and non-finite values."""

    resolved = Path(path).resolve()
    try:
        text = resolved.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ProfileValidationError(
            "JSON_READ",
            f"cannot read strict UTF-8 {purpose} JSON",
            path=str(resolved),
            error=str(exc),
        ) from exc
    if not text.strip():
        raise ProfileValidationError(
            "JSON_EMPTY", f"{purpose} file is empty", path=str(resolved)
        )
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except ProfileValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(
            "JSON_SYNTAX",
            f"invalid {purpose} JSON",
            path=str(resolved),
            line=exc.lineno,
            column=exc.colno,
            error=exc.msg,
        ) from exc
    if not isinstance(raw, dict):
        raise ProfileValidationError("JSON_ROOT", f"{purpose} root must be an object")
    _validate_text_health(raw)
    return raw


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ProfileValidationError(
            "PROFILE_FIELD_MISSING",
            "required profile field is missing",
            path=f"{path}.{key}",
        )
    return mapping[key]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileValidationError("PROFILE_FIELD_TYPE", "expected object", path=path)
    return value


def _array(value: Any, path: str, *, nonempty: bool = False) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProfileValidationError("PROFILE_FIELD_TYPE", "expected array", path=path)
    if nonempty and not value:
        raise ProfileValidationError(
            "PROFILE_EMPTY_ARRAY", "array must be non-empty", path=path
        )
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileValidationError(
            "PROFILE_FIELD_TYPE", "expected non-empty string", path=path
        )
    return value


def _version(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _VERSION_RE.fullmatch(result):
        raise ProfileValidationError(
            "PROFILE_VERSION_FORMAT", "invalid version token", path=path, value=result
        )
    return result


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path).lower()
    if not _SHA256_RE.fullmatch(result):
        raise ProfileValidationError(
            "PROFILE_HASH_FORMAT", "expected lowercase SHA-256 hex", path=path
        )
    return result


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ProfileValidationError(
            "PROFILE_NUMBER", "expected finite number", path=path, value=value
        )
    result = float(value)
    if strict_positive and result <= 0:
        raise ProfileValidationError(
            "PROFILE_NUMBER_RANGE", "number must be > 0", path=path, value=result
        )
    if minimum is not None and result < minimum:
        raise ProfileValidationError(
            "PROFILE_NUMBER_RANGE", "number is below minimum", path=path, value=result
        )
    if maximum is not None and result > maximum:
        raise ProfileValidationError(
            "PROFILE_NUMBER_RANGE", "number is above maximum", path=path, value=result
        )
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileValidationError(
            "PROFILE_INTEGER_RANGE", "expected integer >= 1", path=path, value=value
        )
    return value


def _unique_strings(value: Any, path: str, *, nonempty: bool = True) -> tuple[str, ...]:
    values = tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )
    if nonempty and not values:
        raise ProfileValidationError(
            "PROFILE_EMPTY_ARRAY", "array must be non-empty", path=path
        )
    if len(set(values)) != len(values):
        raise ProfileValidationError(
            "PROFILE_DUPLICATE", "array values must be unique", path=path
        )
    return values


def _deduplicate_reasons(reasons: Iterable[ReasonCode]) -> tuple[ReasonCode, ...]:
    return tuple(dict.fromkeys(reasons))


def validate_parameter_sources(
    sources: Mapping[str, Any], *, data_kind: str, error_prefix: str = "PROFILE"
) -> None:
    for name, raw in sources.items():
        if not isinstance(raw, Mapping):
            raise ProfileValidationError(
                f"{error_prefix}_SOURCE_TYPE",
                "parameter source entry must be an object",
                source=name,
            )
        category = str(raw.get("category", ""))
        description = raw.get("description")
        unit = raw.get("unit")
        if category not in PARAMETER_SOURCE_CATEGORIES:
            raise ProfileValidationError(
                f"{error_prefix}_SOURCE_CATEGORY",
                "unknown parameter source category",
                source=name,
                category=category,
            )
        if (
            not isinstance(description, str)
            or not description
            or not isinstance(unit, str)
            or not unit
        ):
            raise ProfileValidationError(
                f"{error_prefix}_SOURCE_METADATA",
                "parameter source requires non-empty description and unit",
                source=name,
            )
        if data_kind in {"synthetic", "numerical_simulation"} and category not in {
            "engineering_assumption",
            "stress_test_boundary",
        }:
            raise ProfileValidationError(
                f"{error_prefix}_NONMEASURED_SOURCE",
                "non-measured fixtures may not claim measured/specification/literature provenance",
                source=name,
                category=category,
            )


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    pipeline_id: str
    pipeline_hash: str
    guard_id: str
    guard_hash: str
    encoder_id: str
    encoder_hash: str
    protocol_version: str
    max_attempts: int
    fallback_local_model: str | None
    supported_devices: tuple[str, ...]
    retryable_reasons: tuple[str, ...]
    deployment_resource_bounds: Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    model_hash: str
    model_kind: str
    protocol_version: str
    supported_devices: tuple[str, ...]
    supported_rsus: tuple[str, ...]
    supported_pipelines: tuple[str, ...]
    deployment_resource_bounds: Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class RiskBound:
    risk_type: str
    attacker_id: str
    threshold_id: str
    ucb: float
    subject_count: int
    emission_lcb: float
    confidence_error: float


@dataclass(frozen=True, slots=True)
class SubjectRiskStatistics:
    """Offline subject-cluster ratio bound used to construct frozen cells."""

    subject_count: int
    mean_emit_and_attack: float
    mean_emission: float
    hoeffding_radius: float
    joint_risk_ucb: float
    emission_lcb: float
    conditional_risk_ucb: float


def compute_subject_risk_ucb(
    subject_rows: Iterable[tuple[float, float]],
    *,
    registered_hypotheses: int,
    confidence_error: float,
) -> SubjectRiskStatistics:
    """Compute the scheme's simultaneous subject-level emission-risk bound.

    Each row is ``(X_i, Y_i)`` where ``X_i`` is the subject's fraction of
    frames that were both emitted and successfully attacked, and ``Y_i`` is
    its emitted fraction.  Frames within one subject may be arbitrarily
    dependent; the caller is responsible for supplying independent subjects.
    This helper is for offline profile construction and never updates a loaded
    profile.
    """

    if isinstance(registered_hypotheses, bool) or not isinstance(
        registered_hypotheses, int
    ):
        raise ProfileValidationError(
            "PROFILE_HYPOTHESIS_COUNT", "registered_hypotheses must be an integer >= 1"
        )
    hypothesis_count = _positive_int(registered_hypotheses, "registered_hypotheses")
    delta = _number(
        confidence_error,
        "confidence_error",
        strict_positive=True,
        maximum=1.0,
    )
    rows = tuple(subject_rows)
    if not rows:
        raise ProfileValidationError(
            "PROFILE_SUBJECT_ROWS", "at least one subject row is required"
        )
    normalized: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ProfileValidationError(
                "PROFILE_SUBJECT_ROW",
                "subject row must be a pair (emit_and_attack, emission)",
                index=index,
            )
        joint = _number(
            row[0], f"subject_rows[{index}].emit_and_attack", minimum=0.0, maximum=1.0
        )
        emission = _number(
            row[1], f"subject_rows[{index}].emission", minimum=0.0, maximum=1.0
        )
        if joint > emission:
            raise ProfileValidationError(
                "PROFILE_SUBJECT_LOGIC",
                "emit-and-attack fraction cannot exceed emission fraction",
                index=index,
                emit_and_attack=joint,
                emission=emission,
            )
        normalized.append((joint, emission))
    count = len(normalized)
    mean_joint = sum(row[0] for row in normalized) / count
    mean_emission = sum(row[1] for row in normalized) / count
    radius = math.sqrt(math.log(2.0 * hypothesis_count / delta) / (2.0 * count))
    joint_ucb = min(1.0, mean_joint + radius)
    emission_lcb = max(0.0, mean_emission - radius)
    conditional_ucb = 1.0 if emission_lcb <= 0.0 else min(1.0, joint_ucb / emission_lcb)
    return SubjectRiskStatistics(
        subject_count=count,
        mean_emit_and_attack=mean_joint,
        mean_emission=mean_emission,
        hoeffding_radius=radius,
        joint_risk_ucb=joint_ucb,
        emission_lcb=emission_lcb,
        conditional_risk_ucb=conditional_ucb,
    )


@dataclass(frozen=True, slots=True)
class PrivacyCell:
    pipeline_id: str
    quality_bin: str
    device_type: str
    joint_trace_supported: bool
    bounds: tuple[RiskBound, ...]


@dataclass(frozen=True, slots=True)
class PrivacyDecision:
    safe: bool
    pipeline_id: str
    quality_bins: tuple[str, ...]
    device_type: str
    worst_ucb: float
    per_risk_ucb: Mapping[str, float]
    min_subject_count: int
    min_emission_lcb: float
    reasons: tuple[ReasonCode, ...]
    evaluated_cells: tuple[PrivacyCell, ...]


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[ReasonCode, ...]
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenProfileBundle:
    schema_version: str
    protocol_version: str
    profile_version: str
    profile_hash: str
    data_kind: str
    evidence_status: str
    online_mutable: bool
    risk_threshold: float
    confidence_error: float
    min_subjects: int
    min_emission_lcb: float
    quality_bins: tuple[str, ...]
    preprocessing_resource_bounds: Mapping[str, int | float]
    pipelines: Mapping[str, PipelineProfile]
    local_models: Mapping[str, ModelProfile]
    edge_models: Mapping[str, ModelProfile]
    privacy_cells: Mapping[tuple[str, str, str], PrivacyCell]
    parameter_sources: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_path: Path

    def query_privacy(
        self,
        pipeline_id: str,
        quality_bins: Iterable[str],
        device_type: str,
        *,
        risk_threshold: float | None = None,
        min_subjects: int | None = None,
        min_emission_lcb: float | None = None,
    ) -> PrivacyDecision:
        """Return the conservative intersection decision for every candidate bin."""

        bins = tuple(dict.fromkeys(str(item) for item in quality_bins))
        threshold = (
            self.risk_threshold
            if risk_threshold is None
            else _number(risk_threshold, "risk_threshold", minimum=0.0, maximum=1.0)
        )
        required_subjects = (
            self.min_subjects
            if min_subjects is None
            else _positive_int(min_subjects, "min_subjects")
        )
        required_emission = (
            self.min_emission_lcb
            if min_emission_lcb is None
            else _number(min_emission_lcb, "min_emission_lcb", minimum=0.0, maximum=1.0)
        )

        reasons: list[ReasonCode] = []
        cells: list[PrivacyCell] = []
        per_risk: dict[str, float] = {risk: 1.0 for risk in PRIVACY_RISK_TYPES}
        observed_by_risk: dict[str, list[float]] = {
            risk: [] for risk in PRIVACY_RISK_TYPES
        }
        subjects: list[int] = []
        emissions: list[float] = []

        pipeline = self.pipelines.get(pipeline_id)
        if pipeline is None:
            reasons.append(ReasonCode.DEVICE_UNSUPPORTED)
        elif device_type not in pipeline.supported_devices:
            reasons.append(ReasonCode.DEVICE_UNSUPPORTED)
        if not bins or any(
            quality_bin not in self.quality_bins for quality_bin in bins
        ):
            reasons.append(ReasonCode.OOD)

        if pipeline is not None and not reasons:
            for quality_bin in bins:
                cell = self.privacy_cells.get((pipeline_id, quality_bin, device_type))
                if cell is None or not cell.joint_trace_supported:
                    reasons.append(ReasonCode.JOINT_TRACE_MISSING)
                    continue
                cells.append(cell)
                present_types = {bound.risk_type for bound in cell.bounds}
                if present_types != set(PRIVACY_RISK_TYPES):
                    reasons.append(ReasonCode.JOINT_TRACE_MISSING)
                for bound in cell.bounds:
                    observed_by_risk[bound.risk_type].append(bound.ucb)
                    subjects.append(bound.subject_count)
                    emissions.append(bound.emission_lcb)
                    if bound.subject_count < required_subjects:
                        reasons.append(ReasonCode.IDENTITY_SUPPORT)
                    if bound.emission_lcb < required_emission:
                        reasons.append(ReasonCode.EMISSION_SUPPORT)
                    if bound.ucb > threshold:
                        reasons.append(ReasonCode.PRIVACY_RISK)

        for risk_type, values in observed_by_risk.items():
            if values:
                per_risk[risk_type] = max(values)
        worst_ucb = max(per_risk.values()) if per_risk else 1.0
        min_subject_count = min(subjects) if subjects else 0
        min_emit = min(emissions) if emissions else 0.0
        frozen_risk = MappingProxyType(dict(sorted(per_risk.items())))
        final_reasons = _deduplicate_reasons(reasons)
        return PrivacyDecision(
            safe=not final_reasons and len(cells) == len(bins),
            pipeline_id=pipeline_id,
            quality_bins=bins,
            device_type=device_type,
            worst_ucb=worst_ucb,
            per_risk_ucb=frozen_risk,
            min_subject_count=min_subject_count,
            min_emission_lcb=min_emit,
            reasons=final_reasons,
            evaluated_cells=tuple(cells),
        )

    def safe_pipelines(
        self,
        quality_bins: Iterable[str],
        device_type: str,
        **threshold_overrides: Any,
    ) -> tuple[str, ...]:
        """Stable tuple of pipelines safe for the full candidate-bin intersection."""

        return tuple(
            pipeline_id
            for pipeline_id in sorted(self.pipelines)
            if self.query_privacy(
                pipeline_id,
                quality_bins,
                device_type,
                **threshold_overrides,
            ).safe
        )

    def validate_compatibility(
        self,
        *,
        protocol_version: str,
        profile_hash: str,
        pipeline_id: str | None = None,
        pipeline_hash: str | None = None,
        guard_hash: str | None = None,
        encoder_hash: str | None = None,
        edge_model_id: str | None = None,
        edge_model_hash: str | None = None,
        device_type: str | None = None,
        rsu_id: str | None = None,
    ) -> CompatibilityResult:
        """Validate all supplied protocol/model evidence without mutating state."""

        reasons: list[ReasonCode] = []
        details: dict[str, Any] = {}
        if protocol_version != self.protocol_version:
            reasons.append(ReasonCode.PROTOCOL_MISMATCH)
            details["protocol"] = {
                "expected": self.protocol_version,
                "actual": protocol_version,
            }
        if profile_hash != self.profile_hash:
            reasons.append(ReasonCode.PROFILE_MISMATCH)
            details["profile_hash"] = {
                "expected": self.profile_hash,
                "actual": profile_hash,
            }

        pipeline: PipelineProfile | None = None
        if pipeline_id is not None:
            pipeline = self.pipelines.get(pipeline_id)
            if pipeline is None:
                reasons.append(ReasonCode.VERSION_MISMATCH)
                details["pipeline_id"] = pipeline_id
            else:
                supplied = {
                    "pipeline_hash": (pipeline_hash, pipeline.pipeline_hash),
                    "guard_hash": (guard_hash, pipeline.guard_hash),
                    "encoder_hash": (encoder_hash, pipeline.encoder_hash),
                }
                for name, (actual, expected) in supplied.items():
                    if actual is None or actual != expected:
                        reasons.append(ReasonCode.VERSION_MISMATCH)
                        details[name] = {"expected": expected, "actual": actual}
                if (
                    device_type is not None
                    and device_type not in pipeline.supported_devices
                ):
                    reasons.append(ReasonCode.DEVICE_UNSUPPORTED)
                    details["device_type"] = device_type

        if edge_model_id is not None:
            model = self.edge_models.get(edge_model_id)
            if model is None or edge_model_hash != model.model_hash:
                reasons.append(ReasonCode.VERSION_MISMATCH)
                details["edge_model"] = {
                    "model_id": edge_model_id,
                    "model_hash": edge_model_hash,
                }
            elif (
                pipeline_id is not None
                and model.supported_pipelines
                and pipeline_id not in model.supported_pipelines
            ):
                reasons.append(ReasonCode.PAIRED_MEASUREMENT_MISSING)
                details["edge_model_pipeline"] = {
                    "model_id": edge_model_id,
                    "pipeline_id": pipeline_id,
                }
            if (
                model is not None
                and rsu_id is not None
                and rsu_id not in model.supported_rsus
            ):
                reasons.append(ReasonCode.MODEL_CACHE_MISSING)
                details["rsu_id"] = rsu_id

        final_reasons = _deduplicate_reasons(reasons)
        return CompatibilityResult(
            compatible=not final_reasons,
            reasons=final_reasons,
            details=deep_freeze(details),
        )


_PIPELINE_DEPLOYMENT_BOUND_TYPES = {
    "max_peak_memory_bytes": "integer",
    "max_anon_work_s": "number",
    "max_anon_energy_j": "number",
    "max_guard_work_s": "number",
    "max_guard_energy_j": "number",
    "max_encode_work_s": "number",
    "max_encode_energy_j": "number",
    "max_output_bytes": "integer",
}
_LOCAL_DEPLOYMENT_BOUND_TYPES = {
    "max_memory_bytes": "integer",
    "max_service_work_s": "number",
    "max_dynamic_energy_j": "number",
}
_EDGE_DEPLOYMENT_BOUND_TYPES = {
    "max_vram_bytes": "integer",
    "max_ingress_work_s": "number",
    "max_ingress_energy_j": "number",
    "max_gpu_work_s": "number",
    "max_gpu_energy_j": "number",
    "max_result_size_bits": "integer",
}


def _parse_deployment_resource_bounds(
    item: Mapping[str, Any],
    path: str,
    expected: Mapping[str, str],
    *,
    field_name: str = "deployment_resource_bounds",
) -> Mapping[str, int | float]:
    """Parse preregistered physical support bounds, never trace-derived online."""

    field_path = f"{path}.{field_name}"
    raw = _object(
        _required(item, field_name, path),
        field_path,
    )
    missing = sorted(set(expected) - set(raw))
    extra = sorted(set(raw) - set(expected))
    if missing or extra:
        raise ProfileValidationError(
            "PROFILE_DEPLOYMENT_BOUNDS_SCHEMA",
            "deployment resource bounds must contain exactly the registered fields",
            path=field_path,
            missing=missing,
            extra=extra,
        )
    parsed: dict[str, int | float] = {}
    for key, kind in expected.items():
        value_path = f"{field_path}.{key}"
        if kind == "integer":
            parsed[key] = _positive_int(raw[key], value_path)
        else:
            parsed[key] = _number(raw[key], value_path, strict_positive=True)
    return deep_freeze(parsed)


def _parse_pipeline(raw: Any, index: int, protocol_version: str) -> PipelineProfile:
    path = f"$.pipelines[{index}]"
    item = _object(raw, path)
    fallback_raw = item.get("fallback_local_model")
    fallback = (
        None
        if fallback_raw is None
        else _string(fallback_raw, f"{path}.fallback_local_model")
    )
    pipeline = PipelineProfile(
        pipeline_id=_string(
            _required(item, "pipeline_id", path), f"{path}.pipeline_id"
        ),
        pipeline_hash=_sha256(
            _required(item, "pipeline_hash", path), f"{path}.pipeline_hash"
        ),
        guard_id=_string(_required(item, "guard_id", path), f"{path}.guard_id"),
        guard_hash=_sha256(_required(item, "guard_hash", path), f"{path}.guard_hash"),
        encoder_id=_string(_required(item, "encoder_id", path), f"{path}.encoder_id"),
        encoder_hash=_sha256(
            _required(item, "encoder_hash", path), f"{path}.encoder_hash"
        ),
        protocol_version=_version(
            _required(item, "protocol_version", path), f"{path}.protocol_version"
        ),
        max_attempts=_positive_int(
            _required(item, "max_attempts", path), f"{path}.max_attempts"
        ),
        fallback_local_model=fallback,
        supported_devices=_unique_strings(
            _required(item, "supported_devices", path), f"{path}.supported_devices"
        ),
        retryable_reasons=_unique_strings(
            _required(item, "retryable_reasons", path),
            f"{path}.retryable_reasons",
            nonempty=False,
        ),
        deployment_resource_bounds=_parse_deployment_resource_bounds(
            item, path, _PIPELINE_DEPLOYMENT_BOUND_TYPES
        ),
    )
    if pipeline.protocol_version != protocol_version:
        raise ProfileValidationError(
            "PROFILE_PROTOCOL_MISMATCH",
            "pipeline protocol differs from bundle protocol",
            pipeline_id=pipeline.pipeline_id,
        )
    return pipeline


def _parse_model(
    raw: Any, index: int, expected_kind: str, protocol_version: str
) -> ModelProfile:
    section = "local_models" if expected_kind == "local" else "edge_models"
    path = f"$.{section}[{index}]"
    item = _object(raw, path)
    kind = _string(_required(item, "model_kind", path), f"{path}.model_kind")
    if kind != expected_kind:
        raise ProfileValidationError(
            "PROFILE_MODEL_KIND",
            "model appears in the wrong profile section",
            path=path,
            expected=expected_kind,
            actual=kind,
        )
    deployment_types = (
        _LOCAL_DEPLOYMENT_BOUND_TYPES
        if expected_kind == "local"
        else _EDGE_DEPLOYMENT_BOUND_TYPES
    )
    model = ModelProfile(
        model_id=_string(_required(item, "model_id", path), f"{path}.model_id"),
        model_hash=_sha256(_required(item, "model_hash", path), f"{path}.model_hash"),
        model_kind=kind,
        protocol_version=_version(
            _required(item, "protocol_version", path), f"{path}.protocol_version"
        ),
        supported_devices=_unique_strings(
            item.get("supported_devices", []),
            f"{path}.supported_devices",
            nonempty=False,
        ),
        supported_rsus=_unique_strings(
            item.get("supported_rsus", []), f"{path}.supported_rsus", nonempty=False
        ),
        supported_pipelines=_unique_strings(
            item.get("supported_pipelines", []),
            f"{path}.supported_pipelines",
            nonempty=False,
        ),
        deployment_resource_bounds=_parse_deployment_resource_bounds(
            item, path, deployment_types
        ),
    )
    if model.protocol_version != protocol_version:
        raise ProfileValidationError(
            "PROFILE_PROTOCOL_MISMATCH",
            "model protocol differs from bundle protocol",
            model_id=model.model_id,
        )
    if kind == "local" and not model.supported_devices:
        raise ProfileValidationError(
            "PROFILE_MODEL_SUPPORT",
            "local model requires supported_devices",
            model_id=model.model_id,
        )
    if kind == "edge" and not model.supported_rsus:
        raise ProfileValidationError(
            "PROFILE_MODEL_SUPPORT",
            "edge model requires supported_rsus",
            model_id=model.model_id,
        )
    return model


def _parse_privacy_cell(raw: Any, index: int, confidence_error: float) -> PrivacyCell:
    path = f"$.privacy_cells[{index}]"
    item = _object(raw, path)
    bounds: list[RiskBound] = []
    seen_hypotheses: set[tuple[str, str, str]] = set()
    for bound_index, bound_raw in enumerate(
        _array(_required(item, "bounds", path), f"{path}.bounds", nonempty=True)
    ):
        bound_path = f"{path}.bounds[{bound_index}]"
        value = _object(bound_raw, bound_path)
        risk_type = _string(
            _required(value, "risk_type", bound_path), f"{bound_path}.risk_type"
        )
        if risk_type not in PRIVACY_RISK_TYPES:
            raise ProfileValidationError(
                "PROFILE_RISK_TYPE",
                "risk type is not one of the three registered privacy risks",
                path=f"{bound_path}.risk_type",
                value=risk_type,
            )
        bound = RiskBound(
            risk_type=risk_type,
            attacker_id=_string(
                _required(value, "attacker_id", bound_path), f"{bound_path}.attacker_id"
            ),
            threshold_id=_string(
                _required(value, "threshold_id", bound_path),
                f"{bound_path}.threshold_id",
            ),
            ucb=_number(
                _required(value, "ucb", bound_path),
                f"{bound_path}.ucb",
                minimum=0.0,
                maximum=1.0,
            ),
            subject_count=_positive_int(
                _required(value, "subject_count", bound_path),
                f"{bound_path}.subject_count",
            ),
            emission_lcb=_number(
                _required(value, "emission_lcb", bound_path),
                f"{bound_path}.emission_lcb",
                minimum=0.0,
                maximum=1.0,
            ),
            confidence_error=_number(
                value.get("confidence_error", confidence_error),
                f"{bound_path}.confidence_error",
                strict_positive=True,
                maximum=1.0,
            ),
        )
        hypothesis = (bound.risk_type, bound.attacker_id, bound.threshold_id)
        if hypothesis in seen_hypotheses:
            raise ProfileValidationError(
                "PROFILE_DUPLICATE_HYPOTHESIS",
                "privacy hypothesis appears more than once in one cell",
                path=bound_path,
                hypothesis=hypothesis,
            )
        seen_hypotheses.add(hypothesis)
        bounds.append(bound)
    present = {bound.risk_type for bound in bounds}
    if present != set(PRIVACY_RISK_TYPES):
        raise ProfileValidationError(
            "PROFILE_RISK_COVERAGE",
            "each privacy cell must cover identity, verification and link risk",
            path=path,
            present=sorted(present),
        )
    joint_supported = _required(item, "joint_trace_supported", path)
    if not isinstance(joint_supported, bool):
        raise ProfileValidationError(
            "PROFILE_FIELD_TYPE",
            "joint_trace_supported must be boolean",
            path=f"{path}.joint_trace_supported",
        )
    return PrivacyCell(
        pipeline_id=_string(
            _required(item, "pipeline_id", path), f"{path}.pipeline_id"
        ),
        quality_bin=_string(
            _required(item, "quality_bin", path), f"{path}.quality_bin"
        ),
        device_type=_string(
            _required(item, "device_type", path), f"{path}.device_type"
        ),
        joint_trace_supported=joint_supported,
        bounds=tuple(bounds),
    )


def _unique_index(items: Iterable[Any], attr: str, section: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key = getattr(item, attr)
        if key in result:
            raise ProfileValidationError(
                "PROFILE_DUPLICATE_ID",
                "profile IDs must be unique within a section",
                section=section,
                value=key,
            )
        result[key] = item
    return MappingProxyType(dict(sorted(result.items())))


def load_profile(path: str | Path) -> FrozenProfileBundle:
    """Load, hash-check, validate and deeply freeze a profile document."""

    resolved = Path(path).resolve()
    raw = load_strict_json(resolved, purpose="profile")
    profile_hash = _sha256(_required(raw, "profile_hash", "$"), "$.profile_hash")
    calculated_hash = canonical_document_sha256(raw, "profile_hash")
    if calculated_hash != profile_hash:
        raise ProfileValidationError(
            "PROFILE_HASH_MISMATCH",
            "profile canonical content does not match profile_hash",
            expected=profile_hash,
            calculated=calculated_hash,
            path=str(resolved),
        )

    schema_version = _version(_required(raw, "schema_version", "$"), "$.schema_version")
    protocol_version = _version(
        _required(raw, "protocol_version", "$"), "$.protocol_version"
    )
    profile_version = _version(
        _required(raw, "profile_version", "$"), "$.profile_version"
    )
    data_kind = _string(_required(raw, "data_kind", "$"), "$.data_kind")
    if data_kind not in {"synthetic", "measured", "numerical_simulation"}:
        raise ProfileValidationError(
            "PROFILE_DATA_KIND",
            "data_kind must be synthetic, measured or numerical_simulation",
            value=data_kind,
        )
    evidence_status = _string(
        _required(raw, "evidence_status", "$"), "$.evidence_status"
    )
    online_mutable = _required(raw, "online_mutable", "$")
    if online_mutable is not False:
        raise ProfileValidationError(
            "PROFILE_NOT_FROZEN",
            "online_mutable must be false; online profile updates are prohibited",
            value=online_mutable,
        )
    if data_kind == "synthetic" and evidence_status != "synthetic_fixture_only":
        raise ProfileValidationError(
            "PROFILE_SYNTHETIC_CLAIM",
            "synthetic profile must be marked synthetic_fixture_only",
            evidence_status=evidence_status,
        )
    if (
        data_kind == "numerical_simulation"
        and evidence_status != "frozen_numerical_model"
    ):
        raise ProfileValidationError(
            "PROFILE_NUMERICAL_CLAIM",
            "numerical simulation profile must be marked frozen_numerical_model",
            evidence_status=evidence_status,
        )

    policy = _object(_required(raw, "privacy_policy", "$"), "$.privacy_policy")
    registered = _unique_strings(
        _required(policy, "registered_risk_types", "$.privacy_policy"),
        "$.privacy_policy.registered_risk_types",
    )
    if set(registered) != set(PRIVACY_RISK_TYPES):
        raise ProfileValidationError(
            "PROFILE_RISK_REGISTRY",
            "profile must register exactly identity, verification and link risks",
            actual=registered,
        )
    risk_threshold = _number(
        _required(policy, "risk_threshold", "$.privacy_policy"),
        "$.privacy_policy.risk_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    confidence_error = _number(
        _required(policy, "confidence_error", "$.privacy_policy"),
        "$.privacy_policy.confidence_error",
        strict_positive=True,
        maximum=1.0,
    )
    min_subjects = _positive_int(
        _required(policy, "min_subjects", "$.privacy_policy"),
        "$.privacy_policy.min_subjects",
    )
    min_emission_lcb = _number(
        _required(policy, "min_emission_lcb", "$.privacy_policy"),
        "$.privacy_policy.min_emission_lcb",
        minimum=0.0,
        maximum=1.0,
    )

    quality_bins = _unique_strings(
        _required(raw, "quality_bins", "$"), "$.quality_bins"
    )
    preprocessing_resource_bounds = _parse_deployment_resource_bounds(
        raw,
        "$",
        _LOCAL_DEPLOYMENT_BOUND_TYPES,
        field_name="preprocessing_resource_bounds",
    )
    pipelines = tuple(
        _parse_pipeline(item, index, protocol_version)
        for index, item in enumerate(
            _array(_required(raw, "pipelines", "$"), "$.pipelines", nonempty=True)
        )
    )
    local_models = tuple(
        _parse_model(item, index, "local", protocol_version)
        for index, item in enumerate(
            _array(_required(raw, "local_models", "$"), "$.local_models", nonempty=True)
        )
    )
    edge_models = tuple(
        _parse_model(item, index, "edge", protocol_version)
        for index, item in enumerate(
            _array(_required(raw, "edge_models", "$"), "$.edge_models", nonempty=True)
        )
    )
    pipeline_index = _unique_index(pipelines, "pipeline_id", "pipelines")
    local_index = _unique_index(local_models, "model_id", "local_models")
    edge_index = _unique_index(edge_models, "model_id", "edge_models")

    for pipeline in pipelines:
        if (
            pipeline.fallback_local_model is not None
            and pipeline.fallback_local_model not in local_index
        ):
            raise ProfileValidationError(
                "PROFILE_FALLBACK_MODEL",
                "pipeline fallback references an unknown local model",
                pipeline_id=pipeline.pipeline_id,
                fallback=pipeline.fallback_local_model,
            )
    for model in edge_models:
        unknown = set(model.supported_pipelines) - set(pipeline_index)
        if unknown:
            raise ProfileValidationError(
                "PROFILE_MODEL_PIPELINE",
                "edge model references unknown pipelines",
                model_id=model.model_id,
                unknown=sorted(unknown),
            )

    cells = tuple(
        _parse_privacy_cell(item, index, confidence_error)
        for index, item in enumerate(
            _array(
                _required(raw, "privacy_cells", "$"), "$.privacy_cells", nonempty=True
            )
        )
    )
    cell_index: dict[tuple[str, str, str], PrivacyCell] = {}
    for cell in cells:
        key = (cell.pipeline_id, cell.quality_bin, cell.device_type)
        if key in cell_index:
            raise ProfileValidationError(
                "PROFILE_DUPLICATE_CELL",
                "privacy cell key appears more than once",
                key=key,
            )
        pipeline = pipeline_index.get(cell.pipeline_id)
        if pipeline is None:
            raise ProfileValidationError(
                "PROFILE_CELL_PIPELINE",
                "privacy cell references unknown pipeline",
                key=key,
            )
        if cell.quality_bin not in quality_bins:
            raise ProfileValidationError(
                "PROFILE_CELL_QUALITY",
                "privacy cell references unknown quality bin",
                key=key,
            )
        if cell.device_type not in pipeline.supported_devices:
            raise ProfileValidationError(
                "PROFILE_CELL_DEVICE",
                "privacy cell device is not supported by pipeline",
                key=key,
            )
        cell_index[key] = cell

    sources = _object(_required(raw, "parameter_sources", "$"), "$.parameter_sources")
    metadata = _object(_required(raw, "metadata", "$"), "$.metadata")
    validate_parameter_sources(sources, data_kind=data_kind)

    return FrozenProfileBundle(
        schema_version=schema_version,
        protocol_version=protocol_version,
        profile_version=profile_version,
        profile_hash=profile_hash,
        data_kind=data_kind,
        evidence_status=evidence_status,
        online_mutable=False,
        risk_threshold=risk_threshold,
        confidence_error=confidence_error,
        min_subjects=min_subjects,
        min_emission_lcb=min_emission_lcb,
        quality_bins=quality_bins,
        preprocessing_resource_bounds=preprocessing_resource_bounds,
        pipelines=pipeline_index,
        local_models=local_index,
        edge_models=edge_index,
        privacy_cells=MappingProxyType(dict(sorted(cell_index.items()))),
        parameter_sources=deep_freeze(sources),
        metadata=deep_freeze(metadata),
        source_path=resolved,
    )


__all__ = [
    "CompatibilityResult",
    "FrozenProfileBundle",
    "ModelProfile",
    "PARAMETER_SOURCE_CATEGORIES",
    "PRIVACY_RISK_TYPES",
    "PipelineProfile",
    "PrivacyCell",
    "PrivacyDecision",
    "RiskBound",
    "SubjectRiskStatistics",
    "canonical_document_sha256",
    "canonical_json_bytes",
    "compute_subject_risk_ucb",
    "deep_freeze",
    "load_profile",
    "load_strict_json",
    "thaw_json",
    "validate_parameter_sources",
]
