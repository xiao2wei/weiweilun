"""Validated joint traces for compute, radio and exogenous events.

All transaction variables that arise from one anonymization output live in one
``AnonTraceRow`` and are sampled as that immutable row.  This module never
offers APIs for independently drawing its time, energy, guard, encoding, size
or FER components.  Local and edge FER samples are exact-key lookups; absent
artifact/model pairings return a structured ``unsupported`` result.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

from .enums import ReasonCode, TransferDirection
from .errors import ProfileValidationError, TraceValidationError, UnsupportedCondition
from .events import _same_representable_instant, _strict_future_instant
from .profiles import (
    FrozenProfileBundle,
    canonical_document_sha256,
    deep_freeze,
    load_strict_json,
    validate_parameter_sources,
)


T = TypeVar("T")
_LINK_STATES = {"connected", "temporary_outage", "permanent_loss", "handover"}
_OWNER_TYPES = {"vehicle", "rsu"}
_EVENT_TYPES = {
    "DEVICE_FAULT_START",
    "DEVICE_FAULT_END",
    "DEVICE_FAULT_PERMANENT",
    "LINK_CHANGE",
    "MODEL_VERSION",
    "PROFILE_VERSION",
    "PROTOCOL_VERSION",
    "MODEL_CACHE",
}

# These tolerances protect physical resource comparisons only.  Event-time
# ordering and completion detection use finite-ULP semantics instead.
_BATTERY_ENERGY_TOLERANCE_J = 1e-12
_RSU_WORKLOAD_CAPACITY_TOLERANCE_GPU_S = 1e-12


def _trace_error(code: str, message: str, **context: Any) -> TraceValidationError:
    return TraceValidationError(code, message, **context)


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise _trace_error(
            "TRACE_FIELD_MISSING",
            "required trace field is missing",
            path=f"{path}.{key}",
        )
    return mapping[key]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _trace_error("TRACE_FIELD_TYPE", "expected object", path=path)
    return value


def _array(value: Any, path: str, *, nonempty: bool = False) -> Sequence[Any]:
    if not isinstance(value, list):
        raise _trace_error("TRACE_FIELD_TYPE", "expected array", path=path)
    if nonempty and not value:
        raise _trace_error("TRACE_EMPTY_ARRAY", "array must be non-empty", path=path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _trace_error("TRACE_FIELD_TYPE", "expected non-empty string", path=path)
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _trace_error("TRACE_FIELD_TYPE", "expected boolean", path=path)
    return value


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
        raise _trace_error(
            "TRACE_NUMBER", "expected finite number", path=path, value=value
        )
    result = float(value)
    if strict_positive and result <= 0:
        raise _trace_error(
            "TRACE_NUMBER_RANGE", "number must be > 0", path=path, value=result
        )
    if minimum is not None and result < minimum:
        raise _trace_error(
            "TRACE_NUMBER_RANGE", "number is below minimum", path=path, value=result
        )
    if maximum is not None and result > maximum:
        raise _trace_error(
            "TRACE_NUMBER_RANGE", "number is above maximum", path=path, value=result
        )
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _trace_error(
            "TRACE_INTEGER_RANGE",
            f"expected integer >= {minimum}",
            path=path,
            value=value,
        )
    return value


def _optional_number(
    value: Any,
    path: str,
    *,
    strict_positive: bool = False,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, strict_positive=strict_positive, minimum=minimum)


def _optional_bool(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _stable_unique_strings(
    value: Any, path: str, *, nonempty: bool = True
) -> tuple[str, ...]:
    items = tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )
    if nonempty and not items:
        raise _trace_error("TRACE_EMPTY_ARRAY", "array must be non-empty", path=path)
    if len(set(items)) != len(items):
        raise _trace_error(
            "TRACE_DUPLICATE", "array contains duplicate values", path=path
        )
    return items


@dataclass(frozen=True, slots=True)
class DeviceContext:
    thermal_state: str
    power_mode: str
    memory_pressure: str

    @classmethod
    def from_value(cls, value: "DeviceContext | Mapping[str, Any]") -> "DeviceContext":
        if isinstance(value, DeviceContext):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("device context must be DeviceContext or a mapping")
        try:
            return cls(
                thermal_state=str(value["thermal_state"]),
                power_mode=str(value["power_mode"]),
                memory_pressure=str(value["memory_pressure"]),
            )
        except KeyError as exc:
            raise ValueError(f"device context is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class FERMeasurement:
    model_id: str
    model_hash: str
    valid: bool
    fer_loss: float | None
    true_label: str | None = None
    class_probabilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class AnonAttempt:
    attempt_index: int
    anon_work_s: float
    anon_energy_j: float
    peak_memory_bytes: int
    anon_oom: bool
    guard_work_s: float | None
    guard_energy_j: float | None
    guard_passed: bool | None
    encode_work_s: float | None
    encode_energy_j: float | None
    encode_success: bool | None
    encoded_size_bytes: int | None
    artifact_key: str | None

    @property
    def executed_work_s(self) -> float:
        return (
            self.anon_work_s + (self.guard_work_s or 0.0) + (self.encode_work_s or 0.0)
        )

    @property
    def executed_energy_j(self) -> float:
        return (
            self.anon_energy_j
            + (self.guard_energy_j or 0.0)
            + (self.encode_energy_j or 0.0)
        )


@dataclass(frozen=True, slots=True)
class AnonTraceRow:
    row_id: str
    subject_cluster_id: str
    pipeline_id: str
    pipeline_hash: str
    guard_hash: str
    encoder_hash: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    attempts: tuple[AnonAttempt, ...]
    formed_packet: bool
    final_encoded_size_bytes: int
    artifact_key: str | None
    fer_measurements: Mapping[str, FERMeasurement]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def total_work_s(self) -> float:
        return sum(attempt.executed_work_s for attempt in self.attempts)

    @property
    def total_energy_j(self) -> float:
        return sum(attempt.executed_energy_j for attempt in self.attempts)


@dataclass(frozen=True, slots=True)
class PrepTraceRow:
    row_id: str
    fixture_key: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    service_work_s: float
    dynamic_energy_j: float
    memory_bytes: int
    failed: bool


@dataclass(frozen=True, slots=True)
class LocalFERTraceRow:
    row_id: str
    model_id: str
    model_hash: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    service_work_s: float
    dynamic_energy_j: float
    memory_bytes: int
    failed: bool
    fer_loss: float | None
    subject_cluster_id: str | None = None
    true_label: str | None = None
    class_probabilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class EdgeFERTraceRow:
    row_id: str
    artifact_key: str
    pipeline_id: str
    quality_bin: str
    rsu_id: str
    model_id: str
    model_hash: str
    context: DeviceContext
    ingress_work_s: float
    ingress_energy_j: float
    gpu_work_s: float
    gpu_energy_j: float
    vram_bytes: int
    result_size_bits: int
    ingress_failed: bool
    failed: bool
    fer_loss: float | None
    true_label: str | None = None
    class_probabilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ScenarioAttempt:
    """Identity-free attempt record exposed to a controller rollout.

    ``artifact_token`` is scoped to one :class:`ScenarioLibrary`; it is not
    the source trace artifact key and cannot be joined to evaluation data.
    All physical variables from one attempt remain together in this immutable
    object, so a controller cannot independently resample marginal fields.
    """

    attempt_index: int
    anon_work_s: float
    anon_energy_j: float
    peak_memory_bytes: int
    anon_oom: bool
    guard_work_s: float | None
    guard_energy_j: float | None
    guard_passed: bool | None
    encode_work_s: float | None
    encode_energy_j: float | None
    encode_success: bool | None
    encoded_size_bytes: int | None
    artifact_token: str | None

    @property
    def executed_work_s(self) -> float:
        return (
            self.anon_work_s + (self.guard_work_s or 0.0) + (self.encode_work_s or 0.0)
        )

    @property
    def executed_energy_j(self) -> float:
        return (
            self.anon_energy_j
            + (self.guard_energy_j or 0.0)
            + (self.encode_energy_j or 0.0)
        )


@dataclass(frozen=True, slots=True)
class ScenarioPrepRow:
    """Sanitized preprocessing row with no fixture or source-row identifier."""

    scenario_id: str
    cluster_token: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    service_work_s: float
    dynamic_energy_j: float
    memory_bytes: int
    failed: bool


@dataclass(frozen=True, slots=True)
class ScenarioAnonRow:
    """Sanitized joint anonymization row for controller-only scenarios."""

    scenario_id: str
    cluster_token: str
    pipeline_id: str
    pipeline_hash: str
    guard_hash: str
    encoder_hash: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    attempts: tuple[ScenarioAttempt, ...]
    formed_packet: bool
    final_encoded_size_bytes: int
    artifact_token: str | None
    fer_measurements: Mapping[str, FERMeasurement]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def total_work_s(self) -> float:
        return sum(attempt.executed_work_s for attempt in self.attempts)

    @property
    def total_energy_j(self) -> float:
        return sum(attempt.executed_energy_j for attempt in self.attempts)


@dataclass(frozen=True, slots=True)
class ScenarioLocalFERRow:
    """Sanitized local-FER row with no source row or fixture identifier."""

    scenario_id: str
    cluster_token: str
    model_id: str
    model_hash: str
    quality_bin: str
    device_type: str
    context: DeviceContext
    service_work_s: float
    dynamic_energy_j: float
    memory_bytes: int
    failed: bool
    fer_loss: float | None


@dataclass(frozen=True, slots=True)
class ScenarioEdgeFERRow:
    """Sanitized paired edge-FER row for one scenario-local artifact."""

    scenario_id: str
    cluster_token: str
    artifact_token: str
    pipeline_id: str
    quality_bin: str
    rsu_id: str
    model_id: str
    model_hash: str
    context: DeviceContext
    ingress_work_s: float
    ingress_energy_j: float
    gpu_work_s: float
    gpu_energy_j: float
    vram_bytes: int
    result_size_bits: int
    ingress_failed: bool
    failed: bool
    fer_loss: float | None


@dataclass(frozen=True, slots=True)
class ScenarioWirelessSegment:
    """Relative-time radio service available only to prediction branches."""

    vehicle_id: str
    rsu_id: str
    direction: TransferDirection
    start_offset_s: float
    end_offset_s: float
    goodput_bps: float
    transmitter_power_w: float
    receiver_power_w: float
    link_state: str


@dataclass(frozen=True, slots=True)
class ScenarioThermalSegment:
    owner_type: str
    owner_id: str
    resource: str
    start_offset_s: float
    end_offset_s: float
    state: str
    service_rate_multiplier: float
    dynamic_power_multiplier: float


@dataclass(frozen=True, slots=True)
class ScenarioFaultEvent:
    offset_s: float
    event_type: str
    target_type: str
    target_id: str
    resource: str | None
    permanent: bool


@dataclass(frozen=True, slots=True)
class ScenarioComputeStage:
    """One frozen finite-resource stage in a scenario task continuation."""

    stage: str
    resource: str
    work_s: float
    energy_j: float
    memory_bytes: int
    failed: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioBackgroundLoad:
    """Sanitized workload impulse derived from one scenario-trace arrival."""

    offset_s: float
    vehicle_id: str
    vehicle_resource: str
    vehicle_work_s: float
    vehicle_energy_j: float
    rsu_id: str | None
    ingress_work_s: float
    ingress_energy_j: float
    gpu_work_s: float
    gpu_energy_j: float
    rsu_energy_j: float
    vram_bytes: int
    admission_vram_upper_bytes: int
    admission_gpu_work_upper_s: float
    relative_deadline_s: float
    vehicle_memory_bytes: int
    vehicle_descriptor_count: int
    uplink_bits: float
    downlink_bits: float
    path_kind: str
    pipeline_id: str | None
    pipeline_hash: str | None
    artifact_token: str | None
    model_id: str | None
    model_hash: str | None
    realized_quality_bin: str | None
    ingress_failed: bool
    inference_failed: bool
    fer_loss: float | None
    fallback_local_rows: tuple[ScenarioLocalFERRow, ...]
    edge_rows: tuple[ScenarioEdgeFERRow, ...]
    prep_row: ScenarioPrepRow | None
    vehicle_stages: tuple[ScenarioComputeStage, ...]
    complete_support: bool = True
    support_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioTelemetryEvent:
    """Causal sample/delivery schedule for one public RSU snapshot.

    ``offset_s`` is the sampling time.  A non-dropped sample is frozen from
    the branch's live RSU state at that instant and becomes policy-visible
    only at ``delivery_offset_s``.  ``None`` preserves the legacy immediate
    sample/delivery fixture semantics.  The schedule, delay, drop decision and
    quantization are frozen exogenous inputs shared by every action branch;
    the sampled payload itself remains branch-local.
    """

    offset_s: float
    rsu_id: str
    delivery_offset_s: float | None = None
    sample_sequence: int = 0
    dropped: bool = False
    work_quantum_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ScenarioVersionEvent:
    """Relative, identity-free deployment/cache version transition."""

    offset_s: float
    event_type: str
    target_type: str
    target_id: str
    resource: str | None
    old_version: str | None
    new_version: str | None
    model_id: str | None
    remove: bool
    maintenance_work_s: float | None = None
    maintenance_energy_j: float | None = None


@dataclass(frozen=True, slots=True)
class ScenarioFutureTask:
    """Identity-free future arrival in one relative-time scenario window.

    ``task_token`` is scoped to the containing environment window and cannot
    be joined to a source task, fixture, subject, or evaluation artifact.  The
    referenced compute rows are the already-sanitized, complete scenario rows;
    no source row identifier is retained and no marginal row fields are
    resampled independently.

    Preprocessing values are probability-weighted over the candidate quality
    bins for service work and energy, conservative for peak memory and failure.
    ``complete_support`` is false unless every quality candidate has a
    preprocessing row and matching local, anonymization, and paired edge rows.
    """

    task_token: str
    arrival_offset_s: float
    relative_deadline_s: float
    vehicle_id: str
    device_type: str
    context: DeviceContext
    quality_candidates: tuple[str, ...]
    quality_probabilities: tuple[tuple[str, float], ...]
    ood: bool
    quality_features: Mapping[str, float]
    prep_work_s: float
    prep_energy_j: float
    prep_memory_bytes: int
    prep_failed: bool
    local_rows: tuple[ScenarioLocalFERRow, ...]
    anon_rows: tuple[ScenarioAnonRow, ...]
    edge_rows: tuple[ScenarioEdgeFERRow, ...]
    complete_support: bool
    support_reason: str | None


@dataclass(frozen=True, slots=True)
class ScenarioTaskAnchor:
    """Opaque active-task continuation frozen at a scenario anchor."""

    task_token: str
    vehicle_id: str
    state: str
    deadline_offset_s: float
    path_kind: str
    resource: str | None
    memory_reserved_bytes: int
    descriptor_tokens: Mapping[str, int]
    remaining_work_s: float
    total_work_s: float
    total_energy_j: float
    uplink_bits: float
    downlink_bits: float
    rsu_id: str | None
    rsu_remaining_s: float
    rsu_total_work_s: float
    rsu_total_energy_j: float
    vram_bytes: int
    admission_vram_upper_bytes: int
    admission_gpu_work_upper_s: float
    pipeline_id: str | None
    pipeline_hash: str | None
    artifact_token: str | None
    model_id: str | None
    model_hash: str | None
    realized_quality_bin: str | None
    ingress_remaining_work_s: float
    ingress_total_work_s: float
    ingress_total_energy_j: float
    gpu_remaining_work_s: float
    gpu_total_work_s: float
    gpu_total_energy_j: float
    result_size_bits: float
    ingress_failed: bool
    inference_failed: bool
    fer_loss: float | None
    fallback_local_rows: tuple[ScenarioLocalFERRow, ...]
    edge_rows: tuple[ScenarioEdgeFERRow, ...]
    remaining_vehicle_stages: tuple[ScenarioComputeStage, ...]
    prep_failed: bool
    action_memory_bytes: int
    action_descriptor_tokens: Mapping[str, int]
    controller_remaining_s: float
    controller_next: str | None
    complete_support: bool = True
    support_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioTransferAnchor:
    """Identity-free in-flight packet state at a scenario-window anchor."""

    transfer_token: str
    task_token: str
    vehicle_id: str
    rsu_id: str
    direction: TransferDirection
    total_bits: float
    remaining_bits: float
    status: str
    pause_age_s: float


@dataclass(frozen=True, slots=True)
class ScenarioVehicleAnchor:
    """Frozen causal vehicle state at one scenario-window anchor."""

    vehicle_id: str
    battery_j: float
    memory_capacity_bytes: int
    memory_reserved_bytes: int
    descriptor_capacity: Mapping[str, int]
    descriptors_reserved: Mapping[str, int]
    resources: Mapping[str, Mapping[str, Any]]
    active_task_count: int = 0
    tasks: tuple[ScenarioTaskAnchor, ...] = ()
    transfers: tuple[ScenarioTransferAnchor, ...] = ()
    complete_support: bool = True
    support_reason: str | None = None
    failed: bool = False
    permanent_failure: bool = False
    battery_depleted: bool = False
    physical_energy_j: float = 0.0
    source_kind: str = "frozen_scenario_replay"


@dataclass(frozen=True, slots=True)
class ScenarioRSUAnchor:
    """Frozen private RSU state at one numerical-scenario anchor."""

    rsu_id: str
    descriptor_capacity: int
    descriptors_reserved: int
    vram_capacity_bytes: int
    vram_reserved_bytes: int
    workload_capacity_gpu_s: float
    workload_reserved_gpu_s: float
    cached_models: Mapping[str, str]
    resources: Mapping[str, Mapping[str, Any]]
    active_task_count: int
    physical_energy_j: float
    failed: bool
    permanent_failure: bool
    complete_support: bool = True
    support_reason: str | None = None
    source_kind: str = "frozen_scenario_replay"


@dataclass(frozen=True, slots=True)
class ScenarioEnvironment:
    """One identity-free, relative-time joint exogenous realization."""

    scenario_id: str
    cluster_token: str
    duration_s: float
    wireless: tuple[ScenarioWirelessSegment, ...]
    thermal: tuple[ScenarioThermalSegment, ...]
    faults: tuple[ScenarioFaultEvent, ...]
    background_loads: tuple[ScenarioBackgroundLoad, ...]
    telemetry: tuple[ScenarioTelemetryEvent, ...]
    versions: tuple[ScenarioVersionEvent, ...] = ()
    future_tasks: tuple[ScenarioFutureTask, ...] = ()
    vehicle_anchors: tuple[ScenarioVehicleAnchor, ...] = ()
    rsu_anchors: tuple[ScenarioRSUAnchor, ...] = ()

    @property
    def macro_event_offsets(self) -> tuple[float, ...]:
        values = (
            {item.offset_s for item in self.faults}
            | {item.offset_s for item in self.background_loads}
            | {
                value
                for item in self.telemetry
                for value in (item.offset_s, item.delivery_offset_s)
                if value is not None
            }
            | {item.offset_s for item in self.versions}
            | {item.arrival_offset_s for item in self.future_tasks}
        )
        values.update(
            value
            for item in (*self.wireless, *self.thermal)
            for value in (item.start_offset_s, item.end_offset_s)
        )
        return tuple(sorted(value for value in values if value > 1e-12))


@dataclass(frozen=True, slots=True)
class WirelessSegment:
    segment_id: str
    vehicle_id: str
    rsu_id: str
    direction: TransferDirection
    start_time_s: float
    end_time_s: float
    goodput_bps: float
    transmitter_power_w: float
    receiver_power_w: float
    link_state: str

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def service_bits(self) -> float:
        return self.goodput_bps * self.duration_s


@dataclass(frozen=True, slots=True)
class ThermalSegment:
    segment_id: str
    owner_type: str
    owner_id: str
    resource: str
    start_time_s: float
    end_time_s: float
    state: str
    service_rate_multiplier: float
    dynamic_power_multiplier: float


@dataclass(frozen=True, slots=True)
class ExogenousEvent:
    event_id: str
    time_s: float
    event_type: str
    target_type: str
    target_id: str
    resource: str | None
    old_version: str | None
    new_version: str | None
    permanent: bool
    details: Mapping[str, Any]
    maintenance_work_s: float | None = None
    maintenance_energy_j: float | None = None


@dataclass(frozen=True, slots=True)
class TaskArrival:
    task_id: str
    fixture_key: str
    vehicle_id: str
    arrival_time_s: float
    relative_deadline_s: float
    quality_candidates: tuple[str, ...]
    quality_probabilities: tuple[tuple[str, float], ...]
    true_quality_region: str
    ood: bool
    quality_features: Mapping[str, float]

    @property
    def absolute_deadline_s(self) -> float:
        return self.arrival_time_s + self.relative_deadline_s


@dataclass(frozen=True, slots=True)
class AnonCertifiedBounds:
    max_attempts: int
    max_peak_memory_bytes: int
    max_anon_work_s: float
    max_guard_work_s: float
    max_encode_work_s: float
    max_total_energy_j: float
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class EdgeCertifiedBounds:
    max_vram_bytes: int
    max_gpu_work_s: float
    max_ingress_work_s: float
    max_total_dynamic_energy_j: float
    max_result_size_bits: int


@dataclass(frozen=True, slots=True)
class SupportResult(Generic[T]):
    supported: bool
    value: T | None
    reason: ReasonCode | None
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def require(self) -> T:
        if not self.supported or self.value is None:
            reason = self.reason or ReasonCode.JOINT_TRACE_MISSING
            raise UnsupportedCondition(
                reason.value,
                "requested frozen trace condition is unsupported",
                details=dict(self.details),
            )
        return self.value


def _supported(value: T) -> SupportResult[T]:
    return SupportResult(True, value, None, MappingProxyType({}))


def _unsupported(reason: ReasonCode, **details: Any) -> SupportResult[Any]:
    return SupportResult(False, None, reason, deep_freeze(details))


@dataclass(frozen=True, slots=True)
class TraceBundle:
    schema_version: str
    protocol_version: str
    trace_version: str
    trace_hash: str
    profile_hash: str
    data_kind: str
    evidence_status: str
    seed: int
    horizon_start_s: float
    horizon_end_s: float
    anon_rows: tuple[AnonTraceRow, ...]
    prep_rows: tuple[PrepTraceRow, ...]
    local_rows: tuple[LocalFERTraceRow, ...]
    edge_rows: tuple[EdgeFERTraceRow, ...]
    wireless: tuple[WirelessSegment, ...]
    thermal: tuple[ThermalSegment, ...]
    exogenous_events: tuple[ExogenousEvent, ...]
    arrivals: tuple[TaskArrival, ...]
    parameter_sources: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_path: Path
    _prep_index: Mapping[
        tuple[str, str, str, str, str, str], tuple[PrepTraceRow, ...]
    ] = field(repr=False, compare=False)
    _anon_index: Mapping[
        tuple[str, str, str, str, str, str], tuple[AnonTraceRow, ...]
    ] = field(repr=False, compare=False)
    _local_index: Mapping[
        tuple[str, str, str, str, str, str], tuple[LocalFERTraceRow, ...]
    ] = field(repr=False, compare=False)
    _edge_index: Mapping[
        tuple[str, str, str, str, str, str, str, str], tuple[EdgeFERTraceRow, ...]
    ] = field(repr=False, compare=False)
    _wireless_index: Mapping[
        tuple[str, str, TransferDirection], tuple[WirelessSegment, ...]
    ] = field(repr=False, compare=False)
    _thermal_index: Mapping[tuple[str, str, str], tuple[ThermalSegment, ...]] = field(
        repr=False, compare=False
    )

    def sample_anon_transaction(
        self,
        pipeline_id: str,
        candidate_quality_bins: Iterable[str],
        device_type: str,
        device_context: DeviceContext | Mapping[str, Any],
        rng: random.Random,
    ) -> SupportResult[AnonTraceRow]:
        """Sample one complete transaction row; never sample row fields separately."""

        context = DeviceContext.from_value(device_context)
        bins = tuple(dict.fromkeys(str(value) for value in candidate_quality_bins))
        if not bins:
            return _unsupported(ReasonCode.OOD, candidate_quality_bins=bins)
        grouped: list[tuple[AnonTraceRow, ...]] = []
        for quality_bin in bins:
            key = (
                pipeline_id,
                quality_bin,
                device_type,
                context.thermal_state,
                context.power_mode,
                context.memory_pressure,
            )
            rows = self._anon_index.get(key)
            if not rows:
                return _unsupported(
                    ReasonCode.JOINT_TRACE_MISSING,
                    pipeline_id=pipeline_id,
                    quality_bin=quality_bin,
                    device_type=device_type,
                    context=context,
                )
            grouped.append(rows)
        chosen_group = grouped[rng.randrange(len(grouped))]
        # Registered cluster bootstrap: a prolific synthetic/measured subject
        # must not receive more sampling mass merely because it has more rows.
        by_subject: dict[str, list[AnonTraceRow]] = {}
        for row in chosen_group:
            by_subject.setdefault(row.subject_cluster_id, []).append(row)
        subject_ids = sorted(by_subject)
        chosen_subject = subject_ids[rng.randrange(len(subject_ids))]
        subject_rows = sorted(by_subject[chosen_subject], key=lambda row: row.row_id)
        return _supported(subject_rows[rng.randrange(len(subject_rows))])

    def sample_prep(
        self,
        fixture_key: str,
        quality_bin: str,
        device_type: str,
        context: DeviceContext | Mapping[str, Any],
        rng: random.Random,
    ) -> SupportResult[PrepTraceRow]:
        """Return an exact public-preprocessing service/energy/memory row."""

        device_context = DeviceContext.from_value(context)
        key = (
            fixture_key,
            quality_bin,
            device_type,
            device_context.thermal_state,
            device_context.power_mode,
            device_context.memory_pressure,
        )
        rows = self._prep_index.get(key)
        if not rows:
            return _unsupported(
                ReasonCode.PAIRED_MEASUREMENT_MISSING,
                fixture_key=fixture_key,
                quality_bin=quality_bin,
                device_type=device_type,
                context=device_context,
            )
        return _supported(rows[rng.randrange(len(rows))])

    def anon_certified_bounds(
        self,
        pipeline_id: str,
        candidate_quality_bins: Iterable[str],
        device_type: str,
        context: DeviceContext | Mapping[str, Any],
    ) -> SupportResult[AnonCertifiedBounds]:
        """Conservative maxima over exact supported anonymous transaction cells."""

        device_context = DeviceContext.from_value(context)
        bins = tuple(dict.fromkeys(str(item) for item in candidate_quality_bins))
        rows: list[AnonTraceRow] = []
        if not bins:
            return _unsupported(ReasonCode.OOD, candidate_quality_bins=bins)
        for quality_bin in bins:
            key = (
                pipeline_id,
                quality_bin,
                device_type,
                device_context.thermal_state,
                device_context.power_mode,
                device_context.memory_pressure,
            )
            exact = self._anon_index.get(key)
            if not exact:
                return _unsupported(
                    ReasonCode.JOINT_TRACE_MISSING,
                    pipeline_id=pipeline_id,
                    quality_bin=quality_bin,
                    device_type=device_type,
                    context=device_context,
                )
            rows.extend(exact)
        attempts = [attempt for row in rows for attempt in row.attempts]
        return _supported(
            AnonCertifiedBounds(
                max_attempts=max(len(row.attempts) for row in rows),
                max_peak_memory_bytes=max(
                    attempt.peak_memory_bytes for attempt in attempts
                ),
                max_anon_work_s=max(attempt.anon_work_s for attempt in attempts),
                max_guard_work_s=max(
                    (attempt.guard_work_s or 0.0) for attempt in attempts
                ),
                max_encode_work_s=max(
                    (attempt.encode_work_s or 0.0) for attempt in attempts
                ),
                max_total_energy_j=max(row.total_energy_j for row in rows),
                max_output_bytes=max(row.final_encoded_size_bytes for row in rows),
            )
        )

    def edge_certified_bounds(
        self,
        rsu_id: str,
        model_id: str,
        pipeline_id: str,
        artifact_key: str,
        quality_bin: str,
        context: DeviceContext | Mapping[str, Any],
    ) -> SupportResult[EdgeCertifiedBounds]:
        """Certified admission maxima for one exact artifact/model/context key."""

        rsu_context = DeviceContext.from_value(context)
        key = (
            rsu_id,
            model_id,
            pipeline_id,
            artifact_key,
            quality_bin,
            rsu_context.thermal_state,
            rsu_context.power_mode,
            rsu_context.memory_pressure,
        )
        rows = self._edge_index.get(key)
        if not rows:
            return _unsupported(
                ReasonCode.PAIRED_MEASUREMENT_MISSING,
                rsu_id=rsu_id,
                model_id=model_id,
                pipeline_id=pipeline_id,
                artifact_key=artifact_key,
                quality_bin=quality_bin,
                context=rsu_context,
            )
        return _supported(
            EdgeCertifiedBounds(
                max_vram_bytes=max(row.vram_bytes for row in rows),
                max_gpu_work_s=max(row.gpu_work_s for row in rows),
                max_ingress_work_s=max(row.ingress_work_s for row in rows),
                max_total_dynamic_energy_j=max(
                    row.ingress_energy_j + row.gpu_energy_j for row in rows
                ),
                max_result_size_bits=max(row.result_size_bits for row in rows),
            )
        )

    def sample_local_fer(
        self,
        model_id: str,
        quality_bin: str,
        device_type: str,
        context: DeviceContext | Mapping[str, Any],
        rng: random.Random,
    ) -> SupportResult[LocalFERTraceRow]:
        device_context = DeviceContext.from_value(context)
        key = (
            model_id,
            quality_bin,
            device_type,
            device_context.thermal_state,
            device_context.power_mode,
            device_context.memory_pressure,
        )
        rows = self._local_index.get(key)
        if not rows:
            return _unsupported(
                ReasonCode.PAIRED_MEASUREMENT_MISSING,
                model_id=model_id,
                quality_bin=quality_bin,
                device_type=device_type,
                context=device_context,
            )
        return _supported(rows[rng.randrange(len(rows))])

    def sample_edge_fer(
        self,
        rsu_id: str,
        model_id: str,
        pipeline_id: str,
        artifact_key: str,
        quality_bin: str,
        context: DeviceContext | Mapping[str, Any],
        rng: random.Random,
    ) -> SupportResult[EdgeFERTraceRow]:
        rsu_context = DeviceContext.from_value(context)
        key = (
            rsu_id,
            model_id,
            pipeline_id,
            artifact_key,
            quality_bin,
            rsu_context.thermal_state,
            rsu_context.power_mode,
            rsu_context.memory_pressure,
        )
        rows = self._edge_index.get(key)
        if not rows:
            return _unsupported(
                ReasonCode.PAIRED_MEASUREMENT_MISSING,
                rsu_id=rsu_id,
                model_id=model_id,
                pipeline_id=pipeline_id,
                artifact_key=artifact_key,
                quality_bin=quality_bin,
                context=rsu_context,
            )
        return _supported(rows[rng.randrange(len(rows))])

    def wireless_segments(
        self, vehicle_id: str, rsu_id: str, direction: TransferDirection | str
    ) -> SupportResult[tuple[WirelessSegment, ...]]:
        try:
            normalized = (
                direction
                if isinstance(direction, TransferDirection)
                else TransferDirection(direction)
            )
        except ValueError:
            return _unsupported(ReasonCode.STAGE_ILLEGAL, direction=str(direction))
        rows = self._wireless_index.get((vehicle_id, rsu_id, normalized))
        if not rows:
            return _unsupported(
                ReasonCode.JOINT_TRACE_MISSING,
                vehicle_id=vehicle_id,
                rsu_id=rsu_id,
                direction=normalized.value,
            )
        return _supported(rows)

    def thermal_segments(
        self, owner_type: str, owner_id: str, resource: str = "all"
    ) -> SupportResult[tuple[ThermalSegment, ...]]:
        rows = self._thermal_index.get((owner_type, owner_id, resource))
        if not rows:
            return _unsupported(
                ReasonCode.JOINT_TRACE_MISSING,
                owner_type=owner_type,
                owner_id=owner_id,
                resource=resource,
            )
        return _supported(rows)


@dataclass(frozen=True, slots=True)
class ScenarioLibrary:
    """Causal controller view of a frozen training/validation trace.

    It intentionally contains no absolute scenario timestamps, source row
    identifiers, subject identifiers, evaluation arrivals, or evaluation trace
    cursor.  Training/validation environment histories are converted into
    relative-time windows; source subject/artifact keys are replaced by opaque
    tokens scoped to this library.  Controllers can therefore sample complete
    paired compute/FER/environment realizations without inspecting future test
    events or joining records back to evaluation identities.
    """

    trace_version: str
    trace_hash: str
    profile_hash: str
    protocol_version: str
    data_kind: str
    evidence_status: str
    seed: int
    split_role: str
    prep_rows: tuple[ScenarioPrepRow, ...]
    anon_rows: tuple[ScenarioAnonRow, ...]
    local_rows: tuple[ScenarioLocalFERRow, ...]
    edge_rows: tuple[ScenarioEdgeFERRow, ...]
    environment_scenarios: tuple[ScenarioEnvironment, ...]

    @classmethod
    def from_trace(
        cls,
        trace: TraceBundle,
        *,
        rsu_snapshot_period_s: float | None = None,
        rsu_telemetry_delay_s: float = 0.0,
        rsu_telemetry_quantum_work_s: float = 0.0,
        rsu_telemetry_drop_every: int = 0,
        metadata_bits: int = 0,
        uplink_pause_limit_s: float | None = None,
        downlink_pause_limit_s: float | None = None,
        vehicle_anchor_parameters: Mapping[str, Mapping[str, Any]] | None = None,
        rsu_anchor_parameters: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "ScenarioLibrary":
        if rsu_snapshot_period_s is not None and (
            not math.isfinite(rsu_snapshot_period_s) or rsu_snapshot_period_s <= 0
        ):
            raise ValueError("rsu_snapshot_period_s must be finite and positive")
        for name, value in (
            ("rsu_telemetry_delay_s", rsu_telemetry_delay_s),
            ("rsu_telemetry_quantum_work_s", rsu_telemetry_quantum_work_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if rsu_telemetry_drop_every < 0:
            raise ValueError("rsu_telemetry_drop_every must be nonnegative")
        if (
            isinstance(metadata_bits, bool)
            or not isinstance(metadata_bits, int)
            or metadata_bits < 0
        ):
            raise ValueError("metadata_bits must be a nonnegative integer")
        for name, value in (
            ("uplink_pause_limit_s", uplink_pause_limit_s),
            ("downlink_pause_limit_s", downlink_pause_limit_s),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when supplied")
        split = trace.metadata.get("data_split", {})
        role = (
            str(split.get("role", "not_declared"))
            if isinstance(split, Mapping)
            else "not_declared"
        )
        namespace = trace.trace_hash[:16]
        source_artifacts = {
            artifact
            for row in trace.anon_rows
            for artifact in (
                row.artifact_key,
                *(attempt.artifact_key for attempt in row.attempts),
            )
            if artifact is not None
        }
        source_artifacts.update(row.artifact_key for row in trace.edge_rows)
        artifact_tokens = {
            artifact: f"scenario:{namespace}:artifact:{index:06d}"
            for index, artifact in enumerate(sorted(source_artifacts))
        }
        source_clusters = sorted(
            {row.subject_cluster_id for row in trace.anon_rows}
            | {
                row.subject_cluster_id
                for row in trace.local_rows
                if row.subject_cluster_id is not None
            }
        )
        cluster_tokens = {
            cluster: f"scenario:{namespace}:cluster:{index:06d}"
            for index, cluster in enumerate(source_clusters)
        }
        artifact_clusters = {
            row.artifact_key: cluster_tokens[row.subject_cluster_id]
            for row in trace.anon_rows
            if row.artifact_key is not None
        }

        fixture_tokens = {
            fixture: f"scenario:{namespace}:prep-cluster:{index:06d}"
            for index, fixture in enumerate(
                sorted({row.fixture_key for row in trace.prep_rows})
            )
        }
        prep_rows = tuple(
            ScenarioPrepRow(
                scenario_id=f"scenario:{namespace}:prep:{index:06d}",
                cluster_token=fixture_tokens[row.fixture_key],
                quality_bin=row.quality_bin,
                device_type=row.device_type,
                context=row.context,
                service_work_s=row.service_work_s,
                dynamic_energy_j=row.dynamic_energy_j,
                memory_bytes=row.memory_bytes,
                failed=row.failed,
            )
            for index, row in enumerate(trace.prep_rows)
        )

        anon_rows = tuple(
            ScenarioAnonRow(
                scenario_id=f"scenario:{namespace}:anon:{index:06d}",
                cluster_token=cluster_tokens[row.subject_cluster_id],
                pipeline_id=row.pipeline_id,
                pipeline_hash=row.pipeline_hash,
                guard_hash=row.guard_hash,
                encoder_hash=row.encoder_hash,
                quality_bin=row.quality_bin,
                device_type=row.device_type,
                context=row.context,
                attempts=tuple(
                    ScenarioAttempt(
                        attempt_index=attempt.attempt_index,
                        anon_work_s=attempt.anon_work_s,
                        anon_energy_j=attempt.anon_energy_j,
                        peak_memory_bytes=attempt.peak_memory_bytes,
                        anon_oom=attempt.anon_oom,
                        guard_work_s=attempt.guard_work_s,
                        guard_energy_j=attempt.guard_energy_j,
                        guard_passed=attempt.guard_passed,
                        encode_work_s=attempt.encode_work_s,
                        encode_energy_j=attempt.encode_energy_j,
                        encode_success=attempt.encode_success,
                        encoded_size_bytes=attempt.encoded_size_bytes,
                        artifact_token=(
                            None
                            if attempt.artifact_key is None
                            else artifact_tokens[attempt.artifact_key]
                        ),
                    )
                    for attempt in row.attempts
                ),
                formed_packet=row.formed_packet,
                final_encoded_size_bytes=row.final_encoded_size_bytes,
                artifact_token=(
                    None
                    if row.artifact_key is None
                    else artifact_tokens[row.artifact_key]
                ),
                fer_measurements=MappingProxyType(dict(row.fer_measurements)),
            )
            for index, row in enumerate(trace.anon_rows)
        )
        local_rows = tuple(
            ScenarioLocalFERRow(
                scenario_id=f"scenario:{namespace}:local:{index:06d}",
                cluster_token=(
                    cluster_tokens[row.subject_cluster_id]
                    if row.subject_cluster_id in cluster_tokens
                    else f"scenario:{namespace}:local-cluster:{index:06d}"
                ),
                model_id=row.model_id,
                model_hash=row.model_hash,
                quality_bin=row.quality_bin,
                device_type=row.device_type,
                context=row.context,
                service_work_s=row.service_work_s,
                dynamic_energy_j=row.dynamic_energy_j,
                memory_bytes=row.memory_bytes,
                failed=row.failed,
                fer_loss=row.fer_loss,
            )
            for index, row in enumerate(trace.local_rows)
        )
        edge_rows = tuple(
            ScenarioEdgeFERRow(
                scenario_id=f"scenario:{namespace}:edge:{index:06d}",
                cluster_token=artifact_clusters[row.artifact_key],
                artifact_token=artifact_tokens[row.artifact_key],
                pipeline_id=row.pipeline_id,
                quality_bin=row.quality_bin,
                rsu_id=row.rsu_id,
                model_id=row.model_id,
                model_hash=row.model_hash,
                context=row.context,
                ingress_work_s=row.ingress_work_s,
                ingress_energy_j=row.ingress_energy_j,
                gpu_work_s=row.gpu_work_s,
                gpu_energy_j=row.gpu_energy_j,
                vram_bytes=row.vram_bytes,
                result_size_bits=row.result_size_bits,
                ingress_failed=row.ingress_failed,
                failed=row.failed,
                fer_loss=row.fer_loss,
            )
            for index, row in enumerate(trace.edge_rows)
        )
        environment_scenarios = _scenario_environments(
            trace,
            namespace,
            prep_rows=prep_rows,
            anon_rows=anon_rows,
            local_rows=local_rows,
            edge_rows=edge_rows,
            rsu_snapshot_period_s=rsu_snapshot_period_s,
            rsu_telemetry_delay_s=rsu_telemetry_delay_s,
            rsu_telemetry_quantum_work_s=rsu_telemetry_quantum_work_s,
            rsu_telemetry_drop_every=rsu_telemetry_drop_every,
            metadata_bits=metadata_bits,
            uplink_pause_limit_s=(
                math.inf if uplink_pause_limit_s is None else uplink_pause_limit_s
            ),
            downlink_pause_limit_s=(
                math.inf if downlink_pause_limit_s is None else downlink_pause_limit_s
            ),
            vehicle_anchor_parameters=(
                {} if vehicle_anchor_parameters is None else vehicle_anchor_parameters
            ),
            rsu_anchor_parameters=(
                {} if rsu_anchor_parameters is None else rsu_anchor_parameters
            ),
        )
        return cls(
            trace_version=trace.trace_version,
            trace_hash=trace.trace_hash,
            profile_hash=trace.profile_hash,
            protocol_version=trace.protocol_version,
            data_kind=trace.data_kind,
            evidence_status=trace.evidence_status,
            seed=trace.seed,
            split_role=role,
            prep_rows=prep_rows,
            anon_rows=anon_rows,
            local_rows=local_rows,
            edge_rows=edge_rows,
            environment_scenarios=environment_scenarios,
        )

    @staticmethod
    def _cluster_sample(rows: Sequence[T], rng: random.Random) -> T:
        """Sample an opaque cluster uniformly, then one complete row."""

        if not rows:
            raise LookupError("cannot sample an empty scenario cell")
        grouped: dict[str, list[T]] = {}
        for row in rows:
            grouped.setdefault(str(getattr(row, "cluster_token")), []).append(row)
        cluster = sorted(grouped)[rng.randrange(len(grouped))]
        members = sorted(
            grouped[cluster], key=lambda item: str(getattr(item, "scenario_id"))
        )
        return members[rng.randrange(len(members))]

    def sample_rows(self, rows: Sequence[T], rng: random.Random) -> T:
        return self._cluster_sample(rows, rng)

    def sample_environment(self, rng: random.Random) -> ScenarioEnvironment:
        return self._cluster_sample(self.environment_scenarios, rng)


def _scenario_environments(
    trace: TraceBundle,
    namespace: str,
    *,
    prep_rows: tuple[ScenarioPrepRow, ...],
    anon_rows: tuple[ScenarioAnonRow, ...],
    local_rows: tuple[ScenarioLocalFERRow, ...],
    edge_rows: tuple[ScenarioEdgeFERRow, ...],
    rsu_snapshot_period_s: float | None,
    rsu_telemetry_delay_s: float,
    rsu_telemetry_quantum_work_s: float,
    rsu_telemetry_drop_every: int,
    metadata_bits: int,
    uplink_pause_limit_s: float,
    downlink_pause_limit_s: float,
    vehicle_anchor_parameters: Mapping[str, Mapping[str, Any]],
    rsu_anchor_parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[ScenarioEnvironment, ...]:
    """Convert a training/validation environment into relative-time windows.

    Absolute timestamps and source task IDs never enter the returned objects.
    Every window retains the joint radio/thermal/fault/load history following
    one training/validation macro-event anchor.
    """

    start = trace.horizon_start_s
    end = trace.horizon_end_s
    anchors = {start}
    anchors.update(row.arrival_time_s for row in trace.arrivals)
    anchors.update(row.time_s for row in trace.exogenous_events)
    anchors.update(
        value
        for row in (*trace.wireless, *trace.thermal)
        for value in (row.start_time_s, row.end_time_s)
    )
    usable = tuple(value for value in sorted(anchors) if value < end - 1e-12)
    if not usable:
        usable = (start,)

    anon = tuple(sorted(trace.anon_rows, key=lambda row: row.row_id))
    prep = tuple(sorted(trace.prep_rows, key=lambda row: row.row_id))
    local = tuple(sorted(trace.local_rows, key=lambda row: row.row_id))
    edge = tuple(sorted(trace.edge_rows, key=lambda row: row.row_id))
    scenario_anon_by_row = {
        source.row_id: scenario
        for source, scenario in zip(anon, anon_rows, strict=True)
    }
    scenario_prep_by_row = {
        source.row_id: scenario
        for source, scenario in zip(prep, prep_rows, strict=True)
    }
    scenario_local_by_row = {
        source.row_id: scenario
        for source, scenario in zip(local, local_rows, strict=True)
    }
    scenario_edge_by_row = {
        source.row_id: scenario
        for source, scenario in zip(edge, edge_rows, strict=True)
    }
    edge_by_artifact: dict[str, tuple[EdgeFERTraceRow, ...]] = {}
    for artifact in sorted({row.artifact_key for row in edge}):
        edge_by_artifact[artifact] = tuple(
            row for row in edge if row.artifact_key == artifact
        )

    def background(
        row: TaskArrival, index: int, anchor: float
    ) -> ScenarioBackgroundLoad:
        digest = hashlib.sha256(row.task_id.encode("utf-8")).digest()
        configured_vehicle = vehicle_anchor_parameters.get(row.vehicle_id, {})
        device_type = str(configured_vehicle.get("device_type", ""))
        prep_candidates = tuple(
            item
            for item in prep
            if item.fixture_key == row.fixture_key
            and item.quality_bin in row.quality_candidates
            and (not device_type or item.device_type == device_type)
        )
        prep_row = (
            None
            if not prep_candidates
            else prep_candidates[
                int.from_bytes(digest[9:13], "big") % len(prep_candidates)
            ]
        )
        scenario_prep = (
            None if prep_row is None else scenario_prep_by_row[prep_row.row_id]
        )
        use_edge = bool(digest[0] & 1) and bool(anon) and bool(edge)
        edge_joint_missing = False
        if use_edge:
            joint_anon_candidates = tuple(
                item
                for item in anon
                if prep_row is not None
                and item.formed_packet
                and item.device_type == prep_row.device_type
                and item.quality_bin == prep_row.quality_bin
                and item.context == prep_row.context
                and edge_by_artifact.get(item.artifact_key or "", ())
            )
            anon_row = (
                None
                if not joint_anon_candidates
                else joint_anon_candidates[
                    int.from_bytes(digest[1:5], "big") % len(joint_anon_candidates)
                ]
            )
            paired = (
                ()
                if anon_row is None
                else edge_by_artifact.get(anon_row.artifact_key or "", ())
            )
            edge_joint_missing = anon_row is None or not paired
            if anon_row is not None and paired:
                edge_row = paired[int.from_bytes(digest[5:9], "big") % len(paired)]
                scenario_anon = scenario_anon_by_row[anon_row.row_id]
                scenario_edge = scenario_edge_by_row[edge_row.row_id]
                conservative_edge_rows: list[ScenarioEdgeFERRow] = []
                conservative_anon_rows: list[AnonTraceRow] = []
                quality_edge_complete = True
                for quality_index, quality_bin in enumerate(row.quality_candidates):
                    quality_anon = tuple(
                        item
                        for item in anon
                        if item.pipeline_id == anon_row.pipeline_id
                        and item.device_type == anon_row.device_type
                        and item.quality_bin == quality_bin
                        and item.context == anon_row.context
                        and item.formed_packet
                        and any(
                            edge_item.rsu_id == edge_row.rsu_id
                            and edge_item.model_id == edge_row.model_id
                            for edge_item in edge_by_artifact.get(
                                item.artifact_key or "", ()
                            )
                        )
                    )
                    if not quality_anon:
                        quality_edge_complete = False
                        continue
                    selected_quality_anon = (
                        anon_row
                        if quality_bin == anon_row.quality_bin
                        else quality_anon[
                            (int.from_bytes(digest[1:5], "big") + quality_index)
                            % len(quality_anon)
                        ]
                    )
                    conservative_anon_rows.append(selected_quality_anon)
                    quality_edges = tuple(
                        item
                        for item in edge_by_artifact.get(
                            selected_quality_anon.artifact_key or "", ()
                        )
                        if item.rsu_id == edge_row.rsu_id
                        and item.model_id == edge_row.model_id
                        and item.quality_bin == quality_bin
                    )
                    if not quality_edges:
                        quality_edge_complete = False
                        continue
                    conservative_edge_rows.extend(
                        scenario_edge_by_row[item.row_id] for item in quality_edges
                    )
                scenario_edges = tuple(
                    sorted(
                        conservative_edge_rows,
                        key=lambda item: (
                            item.quality_bin,
                            item.context.thermal_state,
                            item.scenario_id,
                        ),
                    )
                )
                fallback_local_rows = tuple(
                    sorted(
                        (
                            item
                            for item in local_rows
                            if item.quality_bin in row.quality_candidates
                            and item.device_type == scenario_anon.device_type
                        ),
                        key=lambda item: (
                            item.model_id,
                            item.context.thermal_state,
                            item.scenario_id,
                        ),
                    )
                )
                vehicle_stages: list[ScenarioComputeStage] = []
                for attempt in scenario_anon.attempts:
                    vehicle_stages.append(
                        ScenarioComputeStage(
                            stage=f"ANON#{attempt.attempt_index}",
                            resource="accelerator",
                            work_s=attempt.anon_work_s,
                            energy_j=attempt.anon_energy_j,
                            memory_bytes=attempt.peak_memory_bytes,
                        )
                    )
                    if attempt.guard_work_s is not None:
                        vehicle_stages.append(
                            ScenarioComputeStage(
                                stage=f"GUARD#{attempt.attempt_index}",
                                resource="cpu",
                                work_s=attempt.guard_work_s,
                                energy_j=attempt.guard_energy_j or 0.0,
                                memory_bytes=attempt.peak_memory_bytes,
                            )
                        )
                    if attempt.encode_work_s is not None:
                        vehicle_stages.append(
                            ScenarioComputeStage(
                                stage=f"ENCODE#{attempt.attempt_index}",
                                resource="encoder",
                                work_s=attempt.encode_work_s,
                                energy_j=attempt.encode_energy_j or 0.0,
                                memory_bytes=attempt.peak_memory_bytes,
                            )
                        )
                complete = bool(
                    fallback_local_rows
                    and scenario_edges
                    and scenario_prep is not None
                    and quality_edge_complete
                    and all(
                        any(
                            item.quality_bin == quality_bin
                            and item.context == scenario_anon.context
                            for item in fallback_local_rows
                        )
                        for quality_bin in row.quality_candidates
                    )
                )
                conservative_vram = max(item.vram_bytes for item in scenario_edges)
                conservative_gpu_work = max(item.gpu_work_s for item in scenario_edges)
                conservative_memory = max(
                    max(
                        (
                            attempt.peak_memory_bytes
                            for candidate in conservative_anon_rows
                            for attempt in candidate.attempts
                        ),
                        default=0,
                    ),
                    max((item.memory_bytes for item in fallback_local_rows), default=0),
                )
                return ScenarioBackgroundLoad(
                    offset_s=max(0.0, row.arrival_time_s - anchor),
                    vehicle_id=row.vehicle_id,
                    vehicle_resource="accelerator",
                    vehicle_work_s=anon_row.total_work_s,
                    vehicle_energy_j=anon_row.total_energy_j,
                    rsu_id=edge_row.rsu_id,
                    ingress_work_s=edge_row.ingress_work_s,
                    ingress_energy_j=edge_row.ingress_energy_j,
                    gpu_work_s=edge_row.gpu_work_s,
                    gpu_energy_j=edge_row.gpu_energy_j,
                    rsu_energy_j=(
                        edge_row.ingress_energy_j
                        if edge_row.ingress_failed
                        else edge_row.ingress_energy_j + edge_row.gpu_energy_j
                    ),
                    vram_bytes=edge_row.vram_bytes,
                    admission_vram_upper_bytes=conservative_vram,
                    admission_gpu_work_upper_s=conservative_gpu_work,
                    relative_deadline_s=row.relative_deadline_s,
                    vehicle_memory_bytes=conservative_memory,
                    vehicle_descriptor_count=1,
                    uplink_bits=float(
                        anon_row.final_encoded_size_bytes * 8 + metadata_bits
                    ),
                    downlink_bits=float(edge_row.result_size_bits),
                    path_kind="edge",
                    pipeline_id=scenario_anon.pipeline_id,
                    pipeline_hash=scenario_anon.pipeline_hash,
                    artifact_token=scenario_anon.artifact_token,
                    model_id=scenario_edge.model_id,
                    model_hash=scenario_edge.model_hash,
                    realized_quality_bin=scenario_anon.quality_bin,
                    ingress_failed=scenario_edge.ingress_failed,
                    inference_failed=scenario_edge.failed,
                    fer_loss=scenario_edge.fer_loss,
                    fallback_local_rows=fallback_local_rows,
                    edge_rows=scenario_edges,
                    prep_row=scenario_prep,
                    vehicle_stages=tuple(vehicle_stages),
                    complete_support=complete,
                    support_reason=(
                        None
                        if complete
                        else "prep_pair_missing"
                        if scenario_prep is None
                        else "quality_candidate_edge_pair_missing"
                        if not quality_edge_complete
                        else "fallback_local_pair_missing"
                    ),
                )
        joint_local_candidates = tuple(
            item
            for item in local
            if prep_row is not None
            and item.device_type == prep_row.device_type
            and item.quality_bin == prep_row.quality_bin
            and item.context == prep_row.context
        )
        local_row = (
            None
            if not joint_local_candidates
            else joint_local_candidates[index % len(joint_local_candidates)]
        )
        scenario_local = (
            None if local_row is None else scenario_local_by_row[local_row.row_id]
        )
        local_stages = (
            ()
            if scenario_local is None
            else (
                ScenarioComputeStage(
                    stage="LOCAL_FER",
                    resource="accelerator",
                    work_s=scenario_local.service_work_s,
                    energy_j=scenario_local.dynamic_energy_j,
                    memory_bytes=scenario_local.memory_bytes,
                    failed=scenario_local.failed,
                ),
            )
        )
        local_complete = bool(
            scenario_local is not None
            and scenario_prep is not None
            and not edge_joint_missing
            and all(
                any(
                    item.model_id == scenario_local.model_id
                    and item.device_type == scenario_local.device_type
                    and item.quality_bin == quality_bin
                    and item.context == scenario_local.context
                    for item in local_rows
                )
                for quality_bin in row.quality_candidates
            )
        )
        return ScenarioBackgroundLoad(
            offset_s=max(0.0, row.arrival_time_s - anchor),
            vehicle_id=row.vehicle_id,
            vehicle_resource="accelerator",
            vehicle_work_s=0.0 if local_row is None else local_row.service_work_s,
            vehicle_energy_j=0.0 if local_row is None else local_row.dynamic_energy_j,
            rsu_id=None,
            ingress_work_s=0.0,
            ingress_energy_j=0.0,
            gpu_work_s=0.0,
            gpu_energy_j=0.0,
            rsu_energy_j=0.0,
            vram_bytes=0,
            admission_vram_upper_bytes=0,
            admission_gpu_work_upper_s=0.0,
            relative_deadline_s=row.relative_deadline_s,
            vehicle_memory_bytes=(0 if local_row is None else local_row.memory_bytes),
            vehicle_descriptor_count=1,
            uplink_bits=0.0,
            downlink_bits=0.0,
            path_kind="local",
            pipeline_id=None,
            pipeline_hash=None,
            artifact_token=None,
            model_id=None if scenario_local is None else scenario_local.model_id,
            model_hash=None if scenario_local is None else scenario_local.model_hash,
            realized_quality_bin=(
                None if scenario_local is None else scenario_local.quality_bin
            ),
            ingress_failed=False,
            inference_failed=(
                True if scenario_local is None else scenario_local.failed
            ),
            fer_loss=None if scenario_local is None else scenario_local.fer_loss,
            fallback_local_rows=(),
            edge_rows=(),
            prep_row=scenario_prep,
            vehicle_stages=local_stages,
            complete_support=local_complete,
            support_reason=(
                None
                if local_complete
                else "prep_pair_missing"
                if scenario_prep is None
                else "joint_edge_pair_missing"
                if edge_joint_missing
                else "local_row_missing"
            ),
        )

    # The scenario-window anchor is a left-limit state: source arrivals and
    # exogenous events strictly before the anchor have taken effect, while an
    # arrival/fault exactly at the anchor remains an offset-zero macro event.
    # Replaying from the frozen training/validation horizon prevents a future
    # vehicle from being silently reset to deployment initial conditions.
    source_loads = tuple(
        background(row, arrival_index, start)
        for arrival_index, row in enumerate(trace.arrivals)
    )

    def joint_anchors(
        anchor: float, window_index: int
    ) -> tuple[tuple[ScenarioVehicleAnchor, ...], tuple[ScenarioRSUAnchor, ...]]:
        """Replay the frozen numerical scenario jointly up to ``anchor``.

        This is deliberately a continuous-time event replay rather than an
        independent per-vehicle marginal construction.  Vehicle jobs, radio
        packets and atomically admitted RSU ingress/GPU jobs therefore retain
        their joint contention, reservations, energy and failure history.
        """

        vehicle_reasons: dict[str, set[str]] = {
            vehicle_id: set() for vehicle_id in vehicle_anchor_parameters
        }
        rsu_reasons: dict[str, set[str]] = {
            rsu_id: set() for rsu_id in rsu_anchor_parameters
        }

        def nonnegative_number(
            raw: Mapping[str, Any],
            name: str,
            reasons: set[str],
            *,
            default: float = 0.0,
        ) -> float:
            value = raw.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                reasons.add(f"static_parameter:{name}")
                return default
            return float(value)

        def positive_int(
            raw: Mapping[str, Any],
            name: str,
            reasons: set[str],
            *,
            default: int = 1,
        ) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                reasons.add(f"static_parameter:{name}")
                return default
            return int(value)

        vehicles: dict[str, dict[str, Any]] = {}
        for vehicle_id, raw in sorted(vehicle_anchor_parameters.items()):
            reasons = vehicle_reasons[vehicle_id]
            descriptors_raw = raw.get("descriptor_capacity", {})
            servers_raw = raw.get("server_count", {})
            if not isinstance(descriptors_raw, Mapping):
                reasons.add("static_parameter:descriptor_capacity")
                descriptors_raw = {}
            if not isinstance(servers_raw, Mapping):
                reasons.add("static_parameter:server_count")
                servers_raw = {}
            descriptors: dict[str, int] = {}
            resources: dict[str, dict[str, Any]] = {}
            for resource in ("accelerator", "cpu", "encoder"):
                descriptor_value = descriptors_raw.get(resource, 0)
                if (
                    isinstance(descriptor_value, bool)
                    or not isinstance(descriptor_value, int)
                    or descriptor_value < 1
                ):
                    reasons.add(f"static_parameter:descriptor:{resource}")
                    descriptor_value = 0
                server_value = servers_raw.get(resource, 1)
                if (
                    isinstance(server_value, bool)
                    or not isinstance(server_value, int)
                    or server_value < 1
                ):
                    reasons.add(f"static_parameter:server:{resource}")
                    server_value = 1
                descriptors[resource] = int(descriptor_value)
                resources[resource] = {
                    "server_count": int(server_value),
                    "waiting": [],
                    "running": [],
                }
            initial_battery = nonnegative_number(raw, "initial_battery_j", reasons)
            battery_capacity = nonnegative_number(
                raw,
                "battery_capacity_j",
                reasons,
                default=initial_battery,
            )
            if initial_battery > battery_capacity + _BATTERY_ENERGY_TOLERANCE_J:
                reasons.add("static_parameter:battery_conflict")
                initial_battery = battery_capacity
            memory_value = raw.get("memory_capacity_bytes", 0)
            if (
                isinstance(memory_value, bool)
                or not isinstance(memory_value, int)
                or memory_value < 1
            ):
                reasons.add("static_parameter:memory_capacity_bytes")
                memory_value = 0
            vehicles[vehicle_id] = {
                "battery_j": initial_battery,
                "battery_capacity_j": battery_capacity,
                "memory_capacity_bytes": int(memory_value),
                "descriptor_capacity": descriptors,
                "resources": resources,
                "idle_power_w": nonnegative_number(raw, "idle_power_w", reasons),
                "hold_power_w": nonnegative_number(raw, "hold_power_w", reasons),
                "controller_overhead_s": nonnegative_number(
                    raw, "controller_overhead_s", reasons
                ),
                "controller_energy_j": nonnegative_number(
                    raw, "controller_energy_j", reasons
                ),
                "physical_energy_j": 0.0,
                "failed": False,
                "permanent_failure": False,
                "battery_depleted": initial_battery <= 0.0,
            }

        rsus: dict[str, dict[str, Any]] = {}
        for rsu_id, raw in sorted(rsu_anchor_parameters.items()):
            reasons = rsu_reasons[rsu_id]
            cache_raw = raw.get("cached_models", {})
            if not isinstance(cache_raw, Mapping):
                reasons.add("static_parameter:cached_models")
                cache_raw = {}
            descriptor_capacity = positive_int(raw, "descriptor_capacity", reasons)
            vram_value = raw.get("vram_capacity_bytes", 0)
            if (
                isinstance(vram_value, bool)
                or not isinstance(vram_value, int)
                or vram_value < 1
            ):
                reasons.add("static_parameter:vram_capacity_bytes")
                vram_value = 0
            rsus[rsu_id] = {
                "descriptor_capacity": descriptor_capacity,
                "descriptors_reserved": 0,
                "vram_capacity_bytes": int(vram_value),
                "vram_reserved_bytes": 0,
                "workload_capacity_gpu_s": nonnegative_number(
                    raw, "workload_capacity_gpu_s", reasons
                ),
                "workload_reserved_gpu_s": 0.0,
                "cached_models": {
                    str(model_id): str(model_hash)
                    for model_id, model_hash in cache_raw.items()
                },
                "resources": {
                    "ingress": {
                        "server_count": positive_int(raw, "ingress_servers", reasons),
                        "waiting": [],
                        "running": [],
                    },
                    "gpu": {
                        "server_count": positive_int(raw, "gpu_servers", reasons),
                        "waiting": [],
                        "running": [],
                    },
                },
                "idle_power_w": nonnegative_number(raw, "idle_power_w", reasons),
                "hold_power_w": nonnegative_number(raw, "hold_power_w", reasons),
                "physical_energy_j": 0.0,
                "failed": False,
                "permanent_failure": False,
            }

        tasks: dict[int, dict[str, Any]] = {}

        def active(task: Mapping[str, Any]) -> bool:
            return str(task.get("phase")) in {
                "PREP",
                "RAW_CONTROL",
                "COMPUTE",
                "READY_CONTROL",
                "UL",
                "RSU_INGRESS",
                "RSU_GPU",
                "DL",
                "LOCAL_FALLBACK",
            }

        def task_token(task_index: int) -> str:
            return (
                f"scenario:{namespace}:environment:{window_index:06d}:"
                f"background-task:{task_index:06d}"
            )

        def thermal_at(
            owner_type: str, owner_id: str, resource: str, time_s: float
        ) -> tuple[float, float]:
            candidates = tuple(
                row
                for row in trace.thermal
                if row.owner_type == owner_type
                and row.owner_id == owner_id
                and row.resource in {resource, "all"}
                and (
                    row.start_time_s < time_s
                    or _same_representable_instant(row.start_time_s, time_s)
                )
                and time_s < row.end_time_s
                and not _same_representable_instant(time_s, row.end_time_s)
            )
            exact = tuple(row for row in candidates if row.resource == resource)
            selected = exact or candidates
            if not selected:
                return 1.0, 1.0
            row = sorted(selected, key=lambda item: item.segment_id)[0]
            return row.service_rate_multiplier, row.dynamic_power_multiplier

        def wireless_at(
            task: Mapping[str, Any], time_s: float
        ) -> WirelessSegment | None:
            direction = task.get("direction")
            if not isinstance(direction, TransferDirection):
                return None
            matches = tuple(
                row
                for row in trace.wireless
                if row.vehicle_id == task.get("vehicle_id")
                and row.rsu_id == task.get("rsu_id")
                and row.direction is direction
                and (
                    row.start_time_s < time_s
                    or _same_representable_instant(row.start_time_s, time_s)
                )
                and time_s < row.end_time_s
                and not _same_representable_instant(time_s, row.end_time_s)
            )
            return (
                None
                if not matches
                else sorted(matches, key=lambda item: item.segment_id)[0]
            )

        def vehicle_reservations(
            vehicle_id: str,
        ) -> tuple[int, dict[str, int]]:
            memory = sum(
                int(task["memory_bytes"])
                for task in tasks.values()
                if task["vehicle_id"] == vehicle_id
                and bool(task.get("reservation_active"))
            )
            descriptors = {
                resource: sum(
                    int(task["descriptor_tokens"].get(resource, 0))
                    for task in tasks.values()
                    if task["vehicle_id"] == vehicle_id
                    and bool(task.get("reservation_active"))
                )
                for resource in vehicles[vehicle_id]["descriptor_capacity"]
            }
            return memory, descriptors

        def release_rsu(task: dict[str, Any]) -> None:
            if not task.get("rsu_reserved"):
                return
            rsu = rsus.get(str(task.get("rsu_id")))
            if rsu is not None:
                rsu["descriptors_reserved"] = max(
                    0, int(rsu["descriptors_reserved"]) - 1
                )
                rsu["vram_reserved_bytes"] = max(
                    0,
                    int(rsu["vram_reserved_bytes"])
                    - int(task["admission_vram_upper_bytes"]),
                )
                rsu["workload_reserved_gpu_s"] = max(
                    0.0,
                    float(rsu["workload_reserved_gpu_s"])
                    - float(task["admission_gpu_work_upper_s"]),
                )
            task["rsu_reserved"] = False

        def finish_task(task_index: int, terminal: str) -> None:
            task = tasks[task_index]
            for vehicle in vehicles.values():
                for pool in vehicle["resources"].values():
                    for bucket in ("waiting", "running"):
                        if task_index in pool[bucket]:
                            pool[bucket].remove(task_index)
            for rsu in rsus.values():
                for pool in rsu["resources"].values():
                    for bucket in ("waiting", "running"):
                        if task_index in pool[bucket]:
                            pool[bucket].remove(task_index)
            release_rsu(task)
            task["reservation_active"] = False
            task["phase"] = terminal
            task["direction"] = None
            task["pause_age_s"] = 0.0

        def enqueue_load(
            arrival_s: float, task_index: int, load: ScenarioBackgroundLoad
        ) -> None:
            vehicle = vehicles.get(load.vehicle_id)
            if vehicle is None:
                return
            action_descriptor_tokens = (
                {"accelerator": 1, "cpu": 1, "encoder": 1}
                if load.path_kind == "edge"
                else {load.vehicle_resource: 1}
            )
            prep_row = load.prep_row
            task = {
                "vehicle_id": load.vehicle_id,
                "phase": "FAIL",
                "arrival_s": arrival_s,
                "deadline_s": arrival_s + load.relative_deadline_s,
                "path_kind": load.path_kind,
                "memory_bytes": 0 if prep_row is None else prep_row.memory_bytes,
                "descriptor_tokens": {"accelerator": 1},
                "action_memory_bytes": max(
                    load.vehicle_memory_bytes,
                    max(
                        (stage.memory_bytes for stage in load.vehicle_stages),
                        default=0,
                    ),
                ),
                "action_descriptor_tokens": action_descriptor_tokens,
                "reservation_active": False,
                "prep_row": prep_row,
                "vehicle_stages": load.vehicle_stages,
                "stage_cursor": -1,
                "control_remaining_s": 0.0,
                "control_next": None,
                "job_owner_type": "vehicle",
                "job_owner_id": load.vehicle_id,
                "job_resource": "accelerator",
                "job_total_work_s": 0.0
                if prep_row is None
                else prep_row.service_work_s,
                "job_remaining_work_s": (
                    0.0 if prep_row is None else prep_row.service_work_s
                ),
                "job_total_energy_j": (
                    0.0 if prep_row is None else prep_row.dynamic_energy_j
                ),
                "vehicle_total_work_s": float(load.vehicle_work_s),
                "vehicle_total_energy_j": float(load.vehicle_energy_j),
                "rsu_id": load.rsu_id,
                "ingress_total_work_s": float(load.ingress_work_s),
                "ingress_remaining_work_s": float(load.ingress_work_s),
                "ingress_total_energy_j": float(load.ingress_energy_j),
                "gpu_total_work_s": float(load.gpu_work_s),
                "gpu_remaining_work_s": float(load.gpu_work_s),
                "gpu_total_energy_j": float(load.gpu_energy_j),
                "rsu_reserved": False,
                "vram_bytes": int(load.vram_bytes),
                "admission_vram_upper_bytes": int(load.admission_vram_upper_bytes),
                "admission_gpu_work_upper_s": float(load.admission_gpu_work_upper_s),
                "uplink_bits": float(load.uplink_bits),
                "downlink_bits": float(load.downlink_bits),
                "total_bits": 0.0,
                "remaining_bits": 0.0,
                "direction": None,
                "pause_age_s": 0.0,
                "pipeline_id": load.pipeline_id,
                "pipeline_hash": load.pipeline_hash,
                "artifact_token": load.artifact_token,
                "model_id": load.model_id,
                "model_hash": load.model_hash,
                "realized_quality_bin": load.realized_quality_bin,
                "ingress_failed": load.ingress_failed,
                "inference_failed": load.inference_failed,
                "fer_loss": load.fer_loss,
                "fallback_local_rows": load.fallback_local_rows,
                "edge_rows": load.edge_rows,
                "complete_support": load.complete_support,
                "support_reason": load.support_reason,
            }
            tasks[task_index] = task
            if not load.complete_support:
                vehicle_reasons[load.vehicle_id].add(
                    f"background:{load.support_reason or 'paired_row_missing'}"
                )
                return
            memory_reserved, descriptors_reserved = vehicle_reservations(
                load.vehicle_id
            )
            feasible = bool(
                not vehicle["failed"]
                and not vehicle["battery_depleted"]
                and prep_row is not None
                and prep_row.service_work_s > 0
                and memory_reserved + prep_row.memory_bytes
                <= vehicle["memory_capacity_bytes"]
                and descriptors_reserved.get("accelerator", 0) + 1
                <= vehicle["descriptor_capacity"].get("accelerator", 0)
            )
            if feasible:
                task["phase"] = "PREP"
                task["reservation_active"] = True
                vehicle["resources"]["accelerator"]["waiting"].append(task_index)

        def reserve_action(task_index: int) -> bool:
            task = tasks[task_index]
            vehicle = vehicles[task["vehicle_id"]]
            memory_reserved, descriptors_reserved = vehicle_reservations(
                task["vehicle_id"]
            )
            tokens = task["action_descriptor_tokens"]
            feasible = bool(
                memory_reserved + task["action_memory_bytes"]
                <= vehicle["memory_capacity_bytes"]
                and all(
                    descriptors_reserved.get(resource, 0) + count
                    <= vehicle["descriptor_capacity"].get(resource, 0)
                    for resource, count in tokens.items()
                )
            )
            if not feasible:
                return False
            task["memory_bytes"] = task["action_memory_bytes"]
            task["descriptor_tokens"] = dict(tokens)
            task["reservation_active"] = True
            return True

        def start_vehicle_stage(task_index: int, cursor: int) -> None:
            task = tasks[task_index]
            stages = tuple(task["vehicle_stages"])
            if cursor >= len(stages):
                if task["path_kind"] == "local":
                    finish_task(
                        task_index,
                        "FAIL" if task["inference_failed"] else "DONE",
                    )
                else:
                    begin_control(task_index, "READY_CONTROL", "UPLINK")
                return
            stage = stages[cursor]
            task["stage_cursor"] = cursor
            task["phase"] = "COMPUTE"
            task["job_owner_type"] = "vehicle"
            task["job_owner_id"] = task["vehicle_id"]
            task["job_resource"] = stage.resource
            task["job_total_work_s"] = stage.work_s
            task["job_remaining_work_s"] = stage.work_s
            task["job_total_energy_j"] = stage.energy_j
            vehicles[task["vehicle_id"]]["resources"][stage.resource]["waiting"].append(
                task_index
            )

        def begin_control(task_index: int, phase: str, next_step: str) -> None:
            task = tasks[task_index]
            vehicle = vehicles[task["vehicle_id"]]
            energy = float(vehicle["controller_energy_j"])
            if vehicle["battery_j"] + _BATTERY_ENERGY_TOLERANCE_J < energy:
                finish_task(task_index, "FAIL")
                return
            vehicle["battery_j"] -= energy
            vehicle["physical_energy_j"] += energy
            task["phase"] = phase
            task["control_remaining_s"] = float(vehicle["controller_overhead_s"])
            task["control_next"] = next_step

        def complete_control(task_index: int) -> None:
            task = tasks[task_index]
            next_step = task["control_next"]
            task["control_remaining_s"] = 0.0
            task["control_next"] = None
            if next_step == "ACTION":
                if not reserve_action(task_index):
                    finish_task(task_index, "FAIL")
                    return
                start_vehicle_stage(task_index, 0)
            elif next_step == "UPLINK":
                task["phase"] = "UL"
                task["direction"] = TransferDirection.UL
                task["total_bits"] = task["uplink_bits"]
                task["remaining_bits"] = task["uplink_bits"]
                task["pause_age_s"] = 0.0

        def start_fallback(task_index: int) -> None:
            task = tasks[task_index]
            release_rsu(task)
            rows = tuple(task["fallback_local_rows"])
            if not rows:
                vehicle_reasons[task["vehicle_id"]].add("fallback_local_pair_missing")
                finish_task(task_index, "FAIL")
                return
            row = rows[0]
            vehicle = vehicles[task["vehicle_id"]]
            extra_memory = max(0, row.memory_bytes - int(task["memory_bytes"]))
            memory_reserved, _ = vehicle_reservations(task["vehicle_id"])
            if memory_reserved + extra_memory > vehicle["memory_capacity_bytes"]:
                finish_task(task_index, "FAIL")
                return
            task["memory_bytes"] = max(int(task["memory_bytes"]), row.memory_bytes)
            task["phase"] = "LOCAL_FALLBACK"
            task["direction"] = None
            task["job_owner_type"] = "vehicle"
            task["job_owner_id"] = task["vehicle_id"]
            task["job_resource"] = "accelerator"
            task["job_total_work_s"] = row.service_work_s
            task["job_remaining_work_s"] = row.service_work_s
            task["job_total_energy_j"] = row.dynamic_energy_j
            task["inference_failed"] = row.failed
            task["fer_loss"] = row.fer_loss
            vehicle["resources"]["accelerator"]["waiting"].append(task_index)

        def dispatch() -> None:
            for vehicle in vehicles.values():
                if vehicle["failed"] or vehicle["battery_depleted"]:
                    continue
                for pool in vehicle["resources"].values():
                    pool["waiting"].sort(
                        key=lambda index: (tasks[index]["deadline_s"], index)
                    )
                    while (
                        pool["waiting"] and len(pool["running"]) < pool["server_count"]
                    ):
                        pool["running"].append(pool["waiting"].pop(0))
            for rsu in rsus.values():
                if rsu["failed"]:
                    continue
                for pool in rsu["resources"].values():
                    pool["waiting"].sort(
                        key=lambda index: (tasks[index]["deadline_s"], index)
                    )
                    while (
                        pool["waiting"] and len(pool["running"]) < pool["server_count"]
                    ):
                        pool["running"].append(pool["waiting"].pop(0))

        def admit(task_index: int) -> None:
            task = tasks[task_index]
            rsu = rsus.get(str(task.get("rsu_id")))
            valid = bool(
                rsu is not None
                and not rsu["failed"]
                and task.get("model_id") is not None
                and rsu["cached_models"].get(task["model_id"]) == task.get("model_hash")
                and rsu["descriptors_reserved"] + 1 <= rsu["descriptor_capacity"]
                and rsu["vram_reserved_bytes"] + task["admission_vram_upper_bytes"]
                <= rsu["vram_capacity_bytes"]
                and rsu["workload_reserved_gpu_s"] + task["admission_gpu_work_upper_s"]
                <= rsu["workload_capacity_gpu_s"]
                + _RSU_WORKLOAD_CAPACITY_TOLERANCE_GPU_S
            )
            if not valid or rsu is None:
                start_fallback(task_index)
                return
            # One atomic mutation after all checks; rejection above has no
            # descriptor, VRAM or conservative-workload side effect.
            rsu["descriptors_reserved"] += 1
            rsu["vram_reserved_bytes"] += task["admission_vram_upper_bytes"]
            rsu["workload_reserved_gpu_s"] += task["admission_gpu_work_upper_s"]
            task["rsu_reserved"] = True
            task["phase"] = "RSU_INGRESS"
            task["job_owner_type"] = "rsu"
            task["job_owner_id"] = task["rsu_id"]
            task["job_resource"] = "ingress"
            task["job_total_work_s"] = task["ingress_total_work_s"]
            task["job_remaining_work_s"] = task["ingress_remaining_work_s"]
            task["job_total_energy_j"] = task["ingress_total_energy_j"]
            rsu["resources"]["ingress"]["waiting"].append(task_index)

        def complete_job(task_index: int) -> None:
            task = tasks[task_index]
            owner_type = str(task["job_owner_type"])
            owner_id = str(task["job_owner_id"])
            resource = str(task["job_resource"])
            owner = vehicles[owner_id] if owner_type == "vehicle" else rsus[owner_id]
            pool = owner["resources"][resource]
            if task_index in pool["running"]:
                pool["running"].remove(task_index)
            if task["phase"] == "PREP":
                task["reservation_active"] = False
                prep_row = task["prep_row"]
                if prep_row is None or prep_row.failed:
                    finish_task(task_index, "FAIL")
                else:
                    begin_control(task_index, "RAW_CONTROL", "ACTION")
            elif task["phase"] == "COMPUTE":
                start_vehicle_stage(task_index, int(task["stage_cursor"]) + 1)
            elif task["phase"] == "LOCAL_FALLBACK":
                finish_task(
                    task_index,
                    "FAIL" if task["inference_failed"] else "DONE",
                )
            elif task["phase"] == "RSU_INGRESS":
                if task["ingress_failed"]:
                    start_fallback(task_index)
                else:
                    task["phase"] = "RSU_GPU"
                    task["job_resource"] = "gpu"
                    task["job_total_work_s"] = task["gpu_total_work_s"]
                    task["job_remaining_work_s"] = task["gpu_remaining_work_s"]
                    task["job_total_energy_j"] = task["gpu_total_energy_j"]
                    rsus[owner_id]["resources"]["gpu"]["waiting"].append(task_index)
            elif task["phase"] == "RSU_GPU":
                release_rsu(task)
                if task["inference_failed"]:
                    start_fallback(task_index)
                else:
                    task["phase"] = "DL"
                    task["direction"] = TransferDirection.DL
                    task["total_bits"] = task["downlink_bits"]
                    task["remaining_bits"] = task["downlink_bits"]
                    task["pause_age_s"] = 0.0

        def before_instant(left_s: float, right_s: float) -> bool:
            return left_s < right_s and not _same_representable_instant(left_s, right_s)

        def at_or_before_instant(left_s: float, right_s: float) -> bool:
            return left_s < right_s or _same_representable_instant(left_s, right_s)

        def at_or_after_instant(left_s: float, right_s: float) -> bool:
            return left_s > right_s or _same_representable_instant(left_s, right_s)

        arrival_rows = tuple(
            (start + load.offset_s, index, load)
            for index, load in enumerate(source_loads)
            if before_instant(start + load.offset_s, anchor)
        )
        fault_rows = tuple(
            row
            for row in trace.exogenous_events
            if row.event_type.startswith("DEVICE_FAULT")
            and before_instant(row.time_s, anchor)
        )
        version_rows = tuple(
            row
            for row in trace.exogenous_events
            if row.event_type in {"MODEL_CACHE", "MODEL_VERSION"}
            and row.target_type == "rsu"
            and before_instant(row.time_s, anchor)
        )
        boundary_times = {
            value
            for row in (*trace.thermal, *trace.wireless)
            for value in (row.start_time_s, row.end_time_s)
            if before_instant(start, value) and before_instant(value, anchor)
        }
        arrival_cursor = fault_cursor = version_cursor = 0
        now = start

        def apply_environment_at(time_s: float) -> None:
            nonlocal fault_cursor, version_cursor
            while fault_cursor < len(fault_rows) and at_or_before_instant(
                fault_rows[fault_cursor].time_s, time_s
            ):
                event = fault_rows[fault_cursor]
                collection = vehicles if event.target_type == "vehicle" else rsus
                target = collection.get(event.target_id)
                if target is not None:
                    recovery = event.event_type.endswith(("RECOVER", "END"))
                    if recovery and not target["permanent_failure"]:
                        target["failed"] = False
                    elif not recovery:
                        target["failed"] = True
                        target["permanent_failure"] = bool(
                            target["permanent_failure"] or event.permanent
                        )
                        affected = tuple(
                            task_index
                            for task_index, task in tasks.items()
                            if active(task)
                            and (
                                (
                                    event.target_type == "vehicle"
                                    and task["vehicle_id"] == event.target_id
                                )
                                or (
                                    event.target_type == "rsu"
                                    and task.get("rsu_id") == event.target_id
                                )
                            )
                        )
                        for task_index in affected:
                            if event.target_type == "rsu":
                                start_fallback(task_index)
                            else:
                                finish_task(task_index, "FAIL")
                fault_cursor += 1
            while version_cursor < len(version_rows) and at_or_before_instant(
                version_rows[version_cursor].time_s, time_s
            ):
                event = version_rows[version_cursor]
                rsu = rsus.get(event.target_id)
                if rsu is not None:
                    model_id = event.details.get("model_id")
                    if model_id is not None:
                        if bool(event.details.get("remove", False)):
                            rsu["cached_models"].pop(str(model_id), None)
                        elif event.new_version:
                            rsu["cached_models"][str(model_id)] = event.new_version
                version_cursor += 1

        def apply_arrivals_at(time_s: float) -> None:
            nonlocal arrival_cursor
            while arrival_cursor < len(arrival_rows) and at_or_before_instant(
                arrival_rows[arrival_cursor][0], time_s
            ):
                enqueue_load(*arrival_rows[arrival_cursor])
                arrival_cursor += 1

        apply_environment_at(now)
        apply_arrivals_at(now)
        while before_instant(now, anchor):
            for task_index, task in tuple(sorted(tasks.items())):
                if task["phase"] not in {"UL", "DL"}:
                    continue
                segment = wireless_at(task, now)
                if segment is None:
                    vehicle_reasons[task["vehicle_id"]].add("wireless_segment_missing")
                elif segment.link_state in {"permanent_loss", "handover"}:
                    start_fallback(task_index)
                elif segment.link_state == "temporary_outage":
                    pause_limit = (
                        uplink_pause_limit_s
                        if task["direction"] is TransferDirection.UL
                        else downlink_pause_limit_s
                    )
                    if at_or_after_instant(task["pause_age_s"], pause_limit):
                        start_fallback(task_index)
            dispatch()
            candidates = [anchor]
            if arrival_cursor < len(arrival_rows):
                candidates.append(arrival_rows[arrival_cursor][0])
            if fault_cursor < len(fault_rows):
                candidates.append(fault_rows[fault_cursor].time_s)
            if version_cursor < len(version_rows):
                candidates.append(version_rows[version_cursor].time_s)
            candidates.extend(value for value in boundary_times if value > now)
            candidates.extend(
                float(task["deadline_s"])
                for task in tasks.values()
                if active(task) and float(task["deadline_s"]) > now
            )
            candidates.extend(
                _strict_future_instant(now, float(task["control_remaining_s"]))
                for task in tasks.values()
                if task["phase"] in {"RAW_CONTROL", "READY_CONTROL"}
                and float(task["control_remaining_s"]) > 0.0
            )
            for owner_type, collection in (("vehicle", vehicles), ("rsu", rsus)):
                for owner_id, owner in collection.items():
                    for resource, pool in owner["resources"].items():
                        rate, _ = thermal_at(owner_type, owner_id, resource, now)
                        if rate > 0:
                            candidates.extend(
                                _strict_future_instant(
                                    now,
                                    tasks[index]["job_remaining_work_s"] / rate,
                                )
                                for index in pool["running"]
                                if tasks[index]["job_remaining_work_s"] > 0.0
                            )

            link_counts: dict[tuple[str, str, TransferDirection], int] = {}
            link_rows: dict[int, WirelessSegment] = {}
            for task_index, task in tasks.items():
                if task["phase"] not in {"UL", "DL"}:
                    continue
                segment = wireless_at(task, now)
                if segment is None or segment.link_state not in {
                    "connected",
                    "temporary_outage",
                }:
                    continue
                direction = task["direction"]
                assert isinstance(direction, TransferDirection)
                key = (task["vehicle_id"], str(task["rsu_id"]), direction)
                link_counts[key] = link_counts.get(key, 0) + 1
                link_rows[task_index] = segment
            for task_index, segment in link_rows.items():
                task = tasks[task_index]
                direction = task["direction"]
                assert isinstance(direction, TransferDirection)
                count = link_counts[
                    (task["vehicle_id"], str(task["rsu_id"]), direction)
                ]
                rate = (
                    segment.goodput_bps / count
                    if segment.link_state == "connected"
                    else 0.0
                )
                if rate > 0:
                    if task["remaining_bits"] > 0.0:
                        candidates.append(
                            _strict_future_instant(now, task["remaining_bits"] / rate)
                        )
                elif segment.link_state == "temporary_outage":
                    pause_limit = (
                        uplink_pause_limit_s
                        if direction is TransferDirection.UL
                        else downlink_pause_limit_s
                    )
                    remaining_pause = pause_limit - float(task["pause_age_s"])
                    # An infinite pause limit means that outage expiry is not
                    # an event.  Link/thermal/fault/deadline boundaries still
                    # provide finite candidates; never feed ``inf`` to the
                    # strict-future helper, whose contract is a finite
                    # physical duration.
                    if math.isfinite(remaining_pause) and remaining_pause > 0.0:
                        candidates.append(_strict_future_instant(now, remaining_pause))

            vehicle_power: dict[str, float] = {}
            for vehicle_id, vehicle in vehicles.items():
                power = (
                    0.0
                    if vehicle["failed"] or vehicle["battery_depleted"]
                    else vehicle["idle_power_w"]
                    + vehicle["hold_power_w"]
                    * sum(
                        task["vehicle_id"] == vehicle_id and active(task)
                        for task in tasks.values()
                    )
                )
                for resource, pool in vehicle["resources"].items():
                    service_rate, _ = thermal_at("vehicle", vehicle_id, resource, now)
                    # Frozen transaction energy is paired with total busy work.
                    # This instantaneous form integrates exactly as
                    # total_energy * served_work / total_work; thermal evidence
                    # changes service progress, never the paired energy total.
                    power += sum(
                        tasks[index]["job_total_energy_j"]
                        / tasks[index]["job_total_work_s"]
                        * service_rate
                        for index in pool["running"]
                    )
                vehicle_power[vehicle_id] = power
            rsu_power: dict[str, float] = {}
            for rsu_id, rsu in rsus.items():
                power = (
                    0.0
                    if rsu["failed"]
                    else rsu["idle_power_w"]
                    + rsu["hold_power_w"]
                    * sum(
                        task.get("rsu_id") == rsu_id
                        and task.get("rsu_reserved")
                        and active(task)
                        for task in tasks.values()
                    )
                )
                for resource, pool in rsu["resources"].items():
                    service_rate, _ = thermal_at("rsu", rsu_id, resource, now)
                    power += sum(
                        tasks[index]["job_total_energy_j"]
                        / tasks[index]["job_total_work_s"]
                        * service_rate
                        for index in pool["running"]
                    )
                rsu_power[rsu_id] = power
            for task_index, segment in link_rows.items():
                task = tasks[task_index]
                direction = task["direction"]
                assert isinstance(direction, TransferDirection)
                count = link_counts[
                    (task["vehicle_id"], str(task["rsu_id"]), direction)
                ]
                vehicle_power[task["vehicle_id"]] += (
                    segment.transmitter_power_w
                    if direction is TransferDirection.UL
                    else segment.receiver_power_w
                ) / count
                if str(task["rsu_id"]) in rsu_power:
                    rsu_power[str(task["rsu_id"])] += (
                        segment.receiver_power_w
                        if direction is TransferDirection.UL
                        else segment.transmitter_power_w
                    ) / count
            for vehicle_id, vehicle in vehicles.items():
                power = vehicle_power[vehicle_id]
                if (
                    power > 0
                    and not vehicle["battery_depleted"]
                    and vehicle["battery_j"] > 0.0
                ):
                    candidates.append(
                        _strict_future_instant(now, vehicle["battery_j"] / power)
                    )
            future_candidates = tuple(
                value for value in candidates if math.isfinite(value) and value > now
            )
            if not future_candidates:
                raise TraceValidationError(
                    "SCENARIO_ANCHOR_NO_FUTURE_EVENT",
                    "joint anchor replay has active state but no future event",
                    anchor_s=anchor,
                    current_time_s=now,
                )
            earliest = min(future_candidates)
            # This is the same compound-instant rule as EventQueue: integrate
            # through the latest IEEE-754 representation of the earliest
            # mathematical event time, then execute completion first.
            next_time = max(
                value
                for value in future_candidates
                if _same_representable_instant(value, earliest)
            )
            dt_s = max(0.0, next_time - now)

            for vehicle_id, vehicle in vehicles.items():
                energy = vehicle_power[vehicle_id] * dt_s
                vehicle["physical_energy_j"] += energy
                if not vehicle["battery_depleted"]:
                    vehicle["battery_j"] = max(0.0, vehicle["battery_j"] - energy)
            for rsu_id, rsu in rsus.items():
                rsu["physical_energy_j"] += rsu_power[rsu_id] * dt_s
            for owner_type, collection in (("vehicle", vehicles), ("rsu", rsus)):
                for owner_id, owner in collection.items():
                    for resource, pool in owner["resources"].items():
                        rate, _ = thermal_at(owner_type, owner_id, resource, now)
                        for task_index in pool["running"]:
                            tasks[task_index]["job_remaining_work_s"] = max(
                                0.0,
                                tasks[task_index]["job_remaining_work_s"] - rate * dt_s,
                            )
            for task in tasks.values():
                if task["phase"] in {"RAW_CONTROL", "READY_CONTROL"}:
                    task["control_remaining_s"] = max(
                        0.0, float(task["control_remaining_s"]) - dt_s
                    )
            for task_index, segment in link_rows.items():
                task = tasks[task_index]
                direction = task["direction"]
                assert isinstance(direction, TransferDirection)
                count = link_counts[
                    (task["vehicle_id"], str(task["rsu_id"]), direction)
                ]
                rate = (
                    segment.goodput_bps / count
                    if segment.link_state == "connected"
                    else 0.0
                )
                task["remaining_bits"] = max(0.0, task["remaining_bits"] - rate * dt_s)
                task["pause_age_s"] = (
                    task["pause_age_s"] + dt_s
                    if segment.link_state == "temporary_outage"
                    else 0.0
                )
            now = next_time

            # Compute/GPU completion wins over the same absolute deadline.
            completed_jobs: list[int] = []
            for collection in (vehicles, rsus):
                for owner in collection.values():
                    for pool in owner["resources"].values():
                        completed_jobs.extend(
                            index
                            for index in tuple(pool["running"])
                            if tasks[index]["job_remaining_work_s"] <= 0.0
                        )
            for task_index in sorted(set(completed_jobs)):
                complete_job(task_index)
            for task_index, task in tuple(sorted(tasks.items())):
                if task["phase"] in {"UL", "DL"} and task["remaining_bits"] <= 0.0:
                    if task["phase"] == "UL":
                        task["direction"] = None
                        admit(task_index)
                    else:
                        finish_task(task_index, "DONE")
            # Exogenous faults/cache changes and link boundaries precede
            # battery/deadline processing at the same timestamp.
            apply_environment_at(now)
            for task_index, task in tuple(sorted(tasks.items())):
                if task["phase"] not in {"UL", "DL"}:
                    continue
                segment = wireless_at(task, now)
                if segment is None or segment.link_state in {
                    "permanent_loss",
                    "handover",
                }:
                    start_fallback(task_index)
                    continue
                if segment.link_state != "temporary_outage":
                    continue
                pause_limit = (
                    uplink_pause_limit_s
                    if task["direction"] is TransferDirection.UL
                    else downlink_pause_limit_s
                )
                if at_or_after_instant(task["pause_age_s"], pause_limit):
                    start_fallback(task_index)
            for vehicle_id, vehicle in vehicles.items():
                if vehicle["battery_j"] <= 0.0 and not vehicle["battery_depleted"]:
                    vehicle["battery_depleted"] = True
                    for task_index, task in tuple(tasks.items()):
                        if task["vehicle_id"] == vehicle_id and active(task):
                            finish_task(task_index, "FAIL")
            for task_index, task in tuple(sorted(tasks.items())):
                if active(task) and at_or_before_instant(task["deadline_s"], now):
                    finish_task(task_index, "FAIL")
            apply_arrivals_at(now)
            # Controller overhead completion is a DISPATCH_DECISION event,
            # not compute service.  It therefore commits only after same-time
            # faults, battery guard, deadline and arrivals.
            for task_index, task in tuple(sorted(tasks.items())):
                if (
                    task["phase"] in {"RAW_CONTROL", "READY_CONTROL"}
                    and task["control_remaining_s"] <= 0.0
                ):
                    complete_control(task_index)

        def job_snapshot(task_index: int) -> Mapping[str, Any]:
            task = tasks[task_index]
            remaining = float(task["job_remaining_work_s"])
            total = float(task["job_total_work_s"])
            energy = float(task["job_total_energy_j"])
            return MappingProxyType(
                {
                    "task_token": task_token(task_index),
                    "remaining_work_s": remaining,
                    "total_work_s": total,
                    "total_energy_j": energy,
                    "remaining_dynamic_energy_j": energy * remaining / total,
                    "nominal_dynamic_power_w": energy / total,
                    "deadline_offset_s": float(task["deadline_s"]) - anchor,
                    "enqueue_seq": task_index,
                }
            )

        task_anchors_by_vehicle: dict[str, list[ScenarioTaskAnchor]] = {
            vehicle_id: [] for vehicle_id in vehicles
        }
        transfer_anchors_by_vehicle: dict[str, list[ScenarioTransferAnchor]] = {
            vehicle_id: [] for vehicle_id in vehicles
        }
        for task_index, task in sorted(tasks.items()):
            if not active(task):
                continue
            phase = str(task["phase"])
            task_anchors_by_vehicle[task["vehicle_id"]].append(
                ScenarioTaskAnchor(
                    task_token=task_token(task_index),
                    vehicle_id=task["vehicle_id"],
                    state=phase,
                    deadline_offset_s=float(task["deadline_s"]) - anchor,
                    path_kind=str(task["path_kind"]),
                    resource=(
                        str(task["job_resource"])
                        if task["job_owner_type"] == "vehicle"
                        and phase in {"PREP", "COMPUTE", "LOCAL_FALLBACK"}
                        else None
                    ),
                    memory_reserved_bytes=(
                        int(task["memory_bytes"]) if task["reservation_active"] else 0
                    ),
                    descriptor_tokens=MappingProxyType(
                        dict(task["descriptor_tokens"])
                        if task["reservation_active"]
                        else {}
                    ),
                    remaining_work_s=(
                        float(task["job_remaining_work_s"])
                        if task["job_owner_type"] == "vehicle"
                        and phase in {"PREP", "COMPUTE", "LOCAL_FALLBACK"}
                        else 0.0
                    ),
                    total_work_s=float(task["vehicle_total_work_s"]),
                    total_energy_j=float(task["vehicle_total_energy_j"]),
                    uplink_bits=float(task["uplink_bits"]),
                    downlink_bits=float(task["downlink_bits"]),
                    rsu_id=(None if task["rsu_id"] is None else str(task["rsu_id"])),
                    rsu_remaining_s=(
                        float(task["job_remaining_work_s"])
                        if task["job_owner_type"] == "rsu"
                        else 0.0
                    ),
                    rsu_total_work_s=float(task["ingress_total_work_s"])
                    + float(task["gpu_total_work_s"]),
                    rsu_total_energy_j=float(task["ingress_total_energy_j"])
                    + float(task["gpu_total_energy_j"]),
                    vram_bytes=int(task["vram_bytes"]),
                    admission_vram_upper_bytes=int(task["admission_vram_upper_bytes"]),
                    admission_gpu_work_upper_s=float(
                        task["admission_gpu_work_upper_s"]
                    ),
                    pipeline_id=task["pipeline_id"],
                    pipeline_hash=task["pipeline_hash"],
                    artifact_token=task["artifact_token"],
                    model_id=task["model_id"],
                    model_hash=task["model_hash"],
                    realized_quality_bin=task["realized_quality_bin"],
                    ingress_remaining_work_s=(
                        float(task["job_remaining_work_s"])
                        if phase == "RSU_INGRESS"
                        else 0.0
                    ),
                    ingress_total_work_s=float(task["ingress_total_work_s"]),
                    ingress_total_energy_j=float(task["ingress_total_energy_j"]),
                    gpu_remaining_work_s=(
                        float(task["job_remaining_work_s"])
                        if phase == "RSU_GPU"
                        else float(task["gpu_total_work_s"])
                        if phase == "RSU_INGRESS"
                        else 0.0
                    ),
                    gpu_total_work_s=float(task["gpu_total_work_s"]),
                    gpu_total_energy_j=float(task["gpu_total_energy_j"]),
                    result_size_bits=float(task["downlink_bits"]),
                    ingress_failed=bool(task["ingress_failed"]),
                    inference_failed=bool(task["inference_failed"]),
                    fer_loss=(
                        None if task["fer_loss"] is None else float(task["fer_loss"])
                    ),
                    fallback_local_rows=tuple(task["fallback_local_rows"]),
                    edge_rows=tuple(task["edge_rows"]),
                    remaining_vehicle_stages=(
                        tuple(task["vehicle_stages"])[int(task["stage_cursor"]) + 1 :]
                        if phase == "COMPUTE"
                        else tuple(task["vehicle_stages"])
                        if phase in {"PREP", "RAW_CONTROL"}
                        else ()
                    ),
                    prep_failed=bool(
                        task["prep_row"] is None or task["prep_row"].failed
                    ),
                    action_memory_bytes=int(task["action_memory_bytes"]),
                    action_descriptor_tokens=MappingProxyType(
                        dict(task["action_descriptor_tokens"])
                    ),
                    controller_remaining_s=float(task["control_remaining_s"]),
                    controller_next=task["control_next"],
                    complete_support=bool(task["complete_support"]),
                    support_reason=task["support_reason"],
                )
            )
            if phase in {"UL", "DL"}:
                segment = wireless_at(task, anchor)
                status = "missing" if segment is None else segment.link_state
                if segment is None:
                    vehicle_reasons[task["vehicle_id"]].add(
                        "wireless_segment_missing_at_anchor"
                    )
                transfer_anchors_by_vehicle[task["vehicle_id"]].append(
                    ScenarioTransferAnchor(
                        transfer_token=(
                            f"scenario:{namespace}:environment:{window_index:06d}:"
                            f"background-transfer:{task_index:06d}"
                        ),
                        task_token=task_token(task_index),
                        vehicle_id=task["vehicle_id"],
                        rsu_id=str(task["rsu_id"]),
                        direction=task["direction"],
                        total_bits=float(task["total_bits"]),
                        remaining_bits=float(task["remaining_bits"]),
                        status=status,
                        pause_age_s=max(0.0, float(task["pause_age_s"])),
                    )
                )

        vehicle_anchors: list[ScenarioVehicleAnchor] = []
        for vehicle_id, vehicle in sorted(vehicles.items()):
            memory_reserved, descriptors_reserved = vehicle_reservations(vehicle_id)
            resource_rows: dict[str, Mapping[str, Any]] = {}
            for resource, pool in sorted(vehicle["resources"].items()):
                running_jobs = tuple(job_snapshot(index) for index in pool["running"])
                waiting_jobs = tuple(job_snapshot(index) for index in pool["waiting"])
                all_jobs = (*running_jobs, *waiting_jobs)
                resource_rows[resource] = MappingProxyType(
                    {
                        "server_count": int(pool["server_count"]),
                        "running_count": len(running_jobs),
                        "waiting_count": len(waiting_jobs),
                        "residual_work_s": sum(
                            float(job["remaining_work_s"]) for job in all_jobs
                        ),
                        "remaining_dynamic_energy_j": sum(
                            float(job["remaining_dynamic_energy_j"]) for job in all_jobs
                        ),
                        "running_jobs": running_jobs,
                        "waiting_jobs": waiting_jobs,
                    }
                )
            reasons = vehicle_reasons[vehicle_id]
            vehicle_anchors.append(
                ScenarioVehicleAnchor(
                    vehicle_id=vehicle_id,
                    battery_j=max(0.0, float(vehicle["battery_j"])),
                    memory_capacity_bytes=int(vehicle["memory_capacity_bytes"]),
                    memory_reserved_bytes=memory_reserved,
                    descriptor_capacity=MappingProxyType(
                        dict(vehicle["descriptor_capacity"])
                    ),
                    descriptors_reserved=MappingProxyType(descriptors_reserved),
                    resources=MappingProxyType(resource_rows),
                    active_task_count=len(task_anchors_by_vehicle[vehicle_id]),
                    tasks=tuple(task_anchors_by_vehicle[vehicle_id]),
                    transfers=tuple(transfer_anchors_by_vehicle[vehicle_id]),
                    complete_support=not reasons,
                    support_reason=(None if not reasons else ",".join(sorted(reasons))),
                    failed=bool(vehicle["failed"]),
                    permanent_failure=bool(vehicle["permanent_failure"]),
                    battery_depleted=bool(vehicle["battery_depleted"]),
                    physical_energy_j=max(0.0, float(vehicle["physical_energy_j"])),
                )
            )

        rsu_anchors: list[ScenarioRSUAnchor] = []
        for rsu_id, rsu in sorted(rsus.items()):
            resource_rows: dict[str, Mapping[str, Any]] = {}
            for resource, pool in sorted(rsu["resources"].items()):
                running_jobs = tuple(job_snapshot(index) for index in pool["running"])
                waiting_jobs = tuple(job_snapshot(index) for index in pool["waiting"])
                all_jobs = (*running_jobs, *waiting_jobs)
                resource_rows[resource] = MappingProxyType(
                    {
                        "server_count": int(pool["server_count"]),
                        "running_count": len(running_jobs),
                        "waiting_count": len(waiting_jobs),
                        "residual_work_s": sum(
                            float(job["remaining_work_s"]) for job in all_jobs
                        ),
                        "remaining_dynamic_energy_j": sum(
                            float(job["remaining_dynamic_energy_j"]) for job in all_jobs
                        ),
                        "running_jobs": running_jobs,
                        "waiting_jobs": waiting_jobs,
                    }
                )
            reasons = rsu_reasons[rsu_id]
            rsu_anchors.append(
                ScenarioRSUAnchor(
                    rsu_id=rsu_id,
                    descriptor_capacity=int(rsu["descriptor_capacity"]),
                    descriptors_reserved=int(rsu["descriptors_reserved"]),
                    vram_capacity_bytes=int(rsu["vram_capacity_bytes"]),
                    vram_reserved_bytes=int(rsu["vram_reserved_bytes"]),
                    workload_capacity_gpu_s=float(rsu["workload_capacity_gpu_s"]),
                    workload_reserved_gpu_s=float(rsu["workload_reserved_gpu_s"]),
                    cached_models=MappingProxyType(dict(rsu["cached_models"])),
                    resources=MappingProxyType(resource_rows),
                    active_task_count=sum(
                        task.get("rsu_id") == rsu_id
                        and task.get("rsu_reserved")
                        and active(task)
                        for task in tasks.values()
                    ),
                    physical_energy_j=max(0.0, float(rsu["physical_energy_j"])),
                    failed=bool(rsu["failed"]),
                    permanent_failure=bool(rsu["permanent_failure"]),
                    complete_support=not reasons,
                    support_reason=(None if not reasons else ",".join(sorted(reasons))),
                )
            )
        return tuple(vehicle_anchors), tuple(rsu_anchors)

    def future_task(
        row: TaskArrival,
        *,
        anchor: float,
        window_index: int,
        task_index: int,
    ) -> ScenarioFutureTask:
        source_prep = tuple(
            item for item in trace.prep_rows if item.fixture_key == row.fixture_key
        )
        device_types = sorted({item.device_type for item in source_prep})
        device_type = device_types[0] if device_types else "unsupported"

        thermal_states = sorted(
            {
                item.state
                for item in trace.thermal
                if item.owner_type == "vehicle"
                and item.owner_id == row.vehicle_id
                and item.start_time_s <= row.arrival_time_s + 1e-12
                and row.arrival_time_s < item.end_time_s - 1e-12
            }
        )
        contexts = sorted(
            {
                item.context
                for item in source_prep
                if not thermal_states or item.context.thermal_state == thermal_states[0]
            },
            key=lambda item: (
                item.thermal_state,
                item.power_mode,
                item.memory_pressure,
            ),
        )
        context = (
            contexts[0]
            if contexts
            else DeviceContext(
                thermal_state=thermal_states[0] if thermal_states else "unsupported",
                power_mode="unsupported",
                memory_pressure="unsupported",
            )
        )
        candidates = row.quality_candidates
        probabilities = row.quality_probabilities
        probability_by_region = dict(probabilities)

        matching_prep = tuple(
            item
            for item in source_prep
            if item.device_type == device_type
            and item.context == context
            and item.quality_bin in candidates
        )
        prep_by_quality = {
            quality_bin: tuple(
                item for item in matching_prep if item.quality_bin == quality_bin
            )
            for quality_bin in candidates
        }

        matching_local = tuple(
            item
            for item in local_rows
            if item.device_type == device_type
            and item.context == context
            and item.quality_bin in candidates
        )
        matching_anon = tuple(
            item
            for item in anon_rows
            if item.device_type == device_type
            and item.context == context
            and item.quality_bin in candidates
        )
        artifact_tokens = {
            item.artifact_token
            for item in matching_anon
            if item.formed_packet and item.artifact_token is not None
        }
        matching_edge = tuple(
            item for item in edge_rows if item.artifact_token in artifact_tokens
        )

        gaps: list[str] = []
        if len(device_types) != 1:
            gaps.append("device_type")
        if len(contexts) != 1 or len(thermal_states) > 1:
            gaps.append("context")
        for quality_bin in candidates:
            if not prep_by_quality[quality_bin]:
                gaps.append(f"prep:{quality_bin}")
            if not any(item.quality_bin == quality_bin for item in matching_local):
                gaps.append(f"local:{quality_bin}")
            bin_anon = tuple(
                item for item in matching_anon if item.quality_bin == quality_bin
            )
            if not bin_anon:
                gaps.append(f"anon:{quality_bin}")
            bin_artifacts = {
                item.artifact_token
                for item in bin_anon
                if item.formed_packet and item.artifact_token is not None
            }
            if not any(item.artifact_token in bin_artifacts for item in matching_edge):
                gaps.append(f"edge:{quality_bin}")

        prep_work_s = sum(
            probability_by_region[quality_bin]
            * (
                sum(item.service_work_s for item in prep_by_quality[quality_bin])
                / len(prep_by_quality[quality_bin])
            )
            for quality_bin in candidates
            if prep_by_quality[quality_bin]
        )
        prep_energy_j = sum(
            probability_by_region[quality_bin]
            * (
                sum(item.dynamic_energy_j for item in prep_by_quality[quality_bin])
                / len(prep_by_quality[quality_bin])
            )
            for quality_bin in candidates
            if prep_by_quality[quality_bin]
        )
        return ScenarioFutureTask(
            task_token=(
                f"scenario:{namespace}:environment:{window_index:06d}:"
                f"future-task:{task_index:06d}"
            ),
            arrival_offset_s=row.arrival_time_s - anchor,
            relative_deadline_s=row.relative_deadline_s,
            vehicle_id=row.vehicle_id,
            device_type=device_type,
            context=context,
            quality_candidates=candidates,
            quality_probabilities=probabilities,
            ood=row.ood,
            quality_features=MappingProxyType(dict(row.quality_features)),
            prep_work_s=prep_work_s,
            prep_energy_j=prep_energy_j,
            prep_memory_bytes=max(
                (item.memory_bytes for item in matching_prep), default=0
            ),
            prep_failed=any(item.failed for item in matching_prep),
            local_rows=matching_local,
            anon_rows=matching_anon,
            edge_rows=matching_edge,
            complete_support=not gaps,
            support_reason=None if not gaps else ",".join(gaps),
        )

    windows: list[ScenarioEnvironment] = []
    cluster_count = max(1, min(4, int(math.sqrt(len(usable))) or 1))
    for index, anchor in enumerate(usable):
        wireless = tuple(
            ScenarioWirelessSegment(
                vehicle_id=row.vehicle_id,
                rsu_id=row.rsu_id,
                direction=row.direction,
                start_offset_s=max(0.0, row.start_time_s - anchor),
                end_offset_s=row.end_time_s - anchor,
                goodput_bps=row.goodput_bps,
                transmitter_power_w=row.transmitter_power_w,
                receiver_power_w=row.receiver_power_w,
                link_state=row.link_state,
            )
            for row in trace.wireless
            if row.end_time_s > anchor + 1e-12
        )
        thermal = tuple(
            ScenarioThermalSegment(
                owner_type=row.owner_type,
                owner_id=row.owner_id,
                resource=row.resource,
                start_offset_s=max(0.0, row.start_time_s - anchor),
                end_offset_s=row.end_time_s - anchor,
                state=row.state,
                service_rate_multiplier=row.service_rate_multiplier,
                dynamic_power_multiplier=row.dynamic_power_multiplier,
            )
            for row in trace.thermal
            if row.end_time_s > anchor + 1e-12
        )
        faults = tuple(
            ScenarioFaultEvent(
                offset_s=row.time_s - anchor,
                event_type=row.event_type,
                target_type=row.target_type,
                target_id=row.target_id,
                resource=row.resource,
                permanent=row.permanent,
            )
            for row in trace.exogenous_events
            if row.time_s >= anchor - 1e-12
            and row.event_type.startswith("DEVICE_FAULT")
        )
        versions = tuple(
            ScenarioVersionEvent(
                offset_s=row.time_s - anchor,
                event_type=row.event_type,
                target_type=row.target_type,
                target_id=row.target_id,
                resource=row.resource,
                old_version=row.old_version,
                new_version=row.new_version,
                model_id=(
                    str(row.details.get("model_id"))
                    if row.details.get("model_id") is not None
                    else None
                ),
                remove=bool(row.details.get("remove", False)),
                maintenance_work_s=row.maintenance_work_s,
                maintenance_energy_j=row.maintenance_energy_j,
            )
            for row in trace.exogenous_events
            if row.time_s >= anchor - 1e-12
            and row.event_type
            in {"MODEL_VERSION", "PROFILE_VERSION", "PROTOCOL_VERSION", "MODEL_CACHE"}
        )
        loads = tuple(
            background(row, arrival_index, anchor)
            for arrival_index, row in enumerate(trace.arrivals)
            if row.arrival_time_s >= anchor - 1e-12
        )
        rsu_ids = sorted(
            {row.rsu_id for row in wireless}
            | {row.rsu_id for row in edge}
            | {row.target_id for row in faults if row.target_type == "rsu"}
            | {row.target_id for row in versions if row.target_type == "rsu"}
        )
        if rsu_snapshot_period_s is None:
            # Backward-compatible fixture mode: environmental boundaries are
            # immediate public telemetry deliveries.  Production simulator
            # construction always supplies the configured periodic schedule.
            telemetry_offsets = {0.0}
            telemetry_offsets.update(
                value
                for item in (*wireless, *thermal)
                for value in (item.start_offset_s, item.end_offset_s)
                if value >= 0.0
            )
            telemetry_offsets.update(item.offset_s for item in faults)
            telemetry_offsets.update(item.offset_s for item in loads)
            telemetry = tuple(
                ScenarioTelemetryEvent(
                    offset_s=offset,
                    rsu_id=rsu_id,
                    delivery_offset_s=offset,
                    sample_sequence=sequence,
                    work_quantum_s=rsu_telemetry_quantum_work_s,
                )
                for sequence, offset in enumerate(sorted(telemetry_offsets), start=1)
                for rsu_id in rsu_ids
                if offset <= end - anchor + 1e-12
            )
        else:
            first_sequence = max(
                1,
                int(math.ceil((anchor - start) / rsu_snapshot_period_s - 1e-12)),
            )
            scheduled: list[ScenarioTelemetryEvent] = []
            sequence = first_sequence
            while True:
                sample_time = start + sequence * rsu_snapshot_period_s
                if sample_time > end + 1e-12:
                    break
                if sample_time >= anchor - 1e-12:
                    sample_offset = max(0.0, sample_time - anchor)
                    dropped = bool(
                        rsu_telemetry_drop_every > 0
                        and sequence % rsu_telemetry_drop_every == 0
                    )
                    delivery_offset = sample_offset + rsu_telemetry_delay_s
                    deliverable = delivery_offset <= end - anchor + 1e-12
                    for rsu_id in rsu_ids:
                        scheduled.append(
                            ScenarioTelemetryEvent(
                                offset_s=sample_offset,
                                rsu_id=rsu_id,
                                delivery_offset_s=(
                                    delivery_offset
                                    if not dropped and deliverable
                                    else None
                                ),
                                sample_sequence=sequence,
                                dropped=dropped or not deliverable,
                                work_quantum_s=rsu_telemetry_quantum_work_s,
                            )
                        )
                sequence += 1
            telemetry = tuple(scheduled)
        future_tasks = tuple(
            future_task(
                row,
                anchor=anchor,
                window_index=index,
                task_index=task_index,
            )
            for task_index, row in enumerate(
                item for item in trace.arrivals if item.arrival_time_s >= anchor - 1e-12
            )
        )
        vehicle_anchors, rsu_anchors = joint_anchors(anchor, index)
        windows.append(
            ScenarioEnvironment(
                scenario_id=f"scenario:{namespace}:environment:{index:06d}",
                cluster_token=(
                    f"scenario:{namespace}:environment-cluster:"
                    f"{index % cluster_count:06d}"
                ),
                duration_s=end - anchor,
                wireless=wireless,
                thermal=thermal,
                faults=faults,
                background_loads=loads,
                telemetry=telemetry,
                versions=versions,
                future_tasks=future_tasks,
                vehicle_anchors=vehicle_anchors,
                rsu_anchors=rsu_anchors,
            )
        )
    return tuple(windows)


def _parse_context(raw: Any, path: str) -> DeviceContext:
    value = _object(raw, path)
    return DeviceContext(
        thermal_state=_string(
            _required(value, "thermal_state", path), f"{path}.thermal_state"
        ),
        power_mode=_string(_required(value, "power_mode", path), f"{path}.power_mode"),
        memory_pressure=_string(
            _required(value, "memory_pressure", path), f"{path}.memory_pressure"
        ),
    )


def _parse_attempt(raw: Any, path: str, expected_index: int) -> AnonAttempt:
    value = _object(raw, path)
    index = _integer(
        _required(value, "attempt_index", path), f"{path}.attempt_index", minimum=1
    )
    if index != expected_index:
        raise _trace_error(
            "TRACE_ATTEMPT_SEQUENCE",
            "attempt indices must be consecutive and one-based",
            path=f"{path}.attempt_index",
            expected=expected_index,
            actual=index,
        )
    attempt = AnonAttempt(
        attempt_index=index,
        anon_work_s=_number(
            _required(value, "anon_work_s", path),
            f"{path}.anon_work_s",
            strict_positive=True,
        ),
        anon_energy_j=_number(
            _required(value, "anon_energy_j", path),
            f"{path}.anon_energy_j",
            strict_positive=True,
        ),
        peak_memory_bytes=_integer(
            _required(value, "peak_memory_bytes", path),
            f"{path}.peak_memory_bytes",
            minimum=1,
        ),
        anon_oom=_boolean(_required(value, "anon_oom", path), f"{path}.anon_oom"),
        guard_work_s=_optional_number(
            value.get("guard_work_s"), f"{path}.guard_work_s", strict_positive=True
        ),
        guard_energy_j=_optional_number(
            value.get("guard_energy_j"), f"{path}.guard_energy_j", strict_positive=True
        ),
        guard_passed=_optional_bool(value.get("guard_passed"), f"{path}.guard_passed"),
        encode_work_s=_optional_number(
            value.get("encode_work_s"), f"{path}.encode_work_s", strict_positive=True
        ),
        encode_energy_j=_optional_number(
            value.get("encode_energy_j"),
            f"{path}.encode_energy_j",
            strict_positive=True,
        ),
        encode_success=_optional_bool(
            value.get("encode_success"), f"{path}.encode_success"
        ),
        encoded_size_bytes=(
            None
            if value.get("encoded_size_bytes") is None
            else _integer(
                value.get("encoded_size_bytes"), f"{path}.encoded_size_bytes", minimum=0
            )
        ),
        artifact_key=_optional_string(
            value.get("artifact_key"), f"{path}.artifact_key"
        ),
    )
    guard_fields = (attempt.guard_work_s, attempt.guard_energy_j, attempt.guard_passed)
    encode_fields = (
        attempt.encode_work_s,
        attempt.encode_energy_j,
        attempt.encode_success,
        attempt.encoded_size_bytes,
        attempt.artifact_key,
    )
    if attempt.anon_oom:
        if any(item is not None for item in guard_fields + encode_fields):
            raise _trace_error(
                "TRACE_PREFIX_SEMANTICS",
                "guard/encode stages cannot execute after anonymizer OOM",
                path=path,
            )
        return attempt
    if any(item is None for item in guard_fields):
        raise _trace_error(
            "TRACE_PREFIX_SEMANTICS",
            "successful anonymizer stage requires complete guard record",
            path=path,
        )
    if attempt.guard_passed is False:
        if any(item is not None for item in encode_fields):
            raise _trace_error(
                "TRACE_PREFIX_SEMANTICS",
                "encoder cannot execute after guard rejection",
                path=path,
            )
        return attempt
    if any(item is None for item in encode_fields[:4]):
        raise _trace_error(
            "TRACE_PREFIX_SEMANTICS",
            "guard pass requires complete encoder record",
            path=path,
        )
    if attempt.encode_success:
        if (
            not attempt.artifact_key
            or not attempt.encoded_size_bytes
            or attempt.encoded_size_bytes <= 0
        ):
            raise _trace_error(
                "TRACE_ENCODING_SIZE",
                "successful encoding requires positive bytes and artifact key",
                path=path,
            )
    elif attempt.artifact_key is not None or attempt.encoded_size_bytes != 0:
        raise _trace_error(
            "TRACE_ENCODING_FAILURE",
            "failed encoding must not claim bytes or artifact",
            path=path,
        )
    return attempt


def _parse_fer_prediction(
    value: Mapping[str, Any], path: str, *, valid: bool
) -> tuple[str | None, tuple[tuple[str, float], ...]]:
    """Parse simulator-only evaluation labels and frozen probability vectors."""

    label_raw = value.get("true_label")
    probabilities_raw = value.get("class_probabilities")
    if label_raw is None and probabilities_raw is None:
        return None, ()
    if not valid:
        raise _trace_error(
            "TRACE_FER_PREDICTION_INVALID",
            "failed FER rows cannot claim labels or probability vectors",
            path=path,
        )
    if label_raw is None or probabilities_raw is None:
        raise _trace_error(
            "TRACE_FER_PREDICTION_INCOMPLETE",
            "true_label and class_probabilities must be declared together",
            path=path,
        )
    label = _string(label_raw, f"{path}.true_label")
    probabilities_object = _object(probabilities_raw, f"{path}.class_probabilities")
    if len(probabilities_object) < 2:
        raise _trace_error(
            "TRACE_FER_CLASS_COUNT",
            "FER probability vectors require at least two classes",
            path=path,
        )
    probabilities = tuple(
        sorted(
            (
                _string(class_name, f"{path}.class_probabilities key"),
                _number(
                    probability,
                    f"{path}.class_probabilities.{class_name}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
            for class_name, probability in probabilities_object.items()
        )
    )
    if label not in dict(probabilities):
        raise _trace_error(
            "TRACE_FER_TRUE_CLASS_MISSING",
            "FER probability vector does not contain its true class",
            path=path,
            true_label=label,
        )
    total = sum(probability for _, probability in probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise _trace_error(
            "TRACE_FER_PROBABILITY_SUM",
            "FER class probabilities must sum to one",
            path=path,
            total=total,
        )
    return label, probabilities


def _parse_measurements(raw: Any, path: str) -> Mapping[str, FERMeasurement]:
    result: dict[str, FERMeasurement] = {}
    for index, item_raw in enumerate(_array(raw, path)):
        item_path = f"{path}[{index}]"
        item = _object(item_raw, item_path)
        valid = _boolean(_required(item, "valid", item_path), f"{item_path}.valid")
        loss_raw = item.get("fer_loss")
        loss = (
            None
            if loss_raw is None
            else _number(loss_raw, f"{item_path}.fer_loss", minimum=0.0, maximum=1.0)
        )
        if valid != (loss is not None):
            raise _trace_error(
                "TRACE_FER_VALIDITY",
                "valid FER measurement must have loss and invalid one must not",
                path=item_path,
            )
        true_label, class_probabilities = _parse_fer_prediction(
            item, item_path, valid=valid
        )
        measurement = FERMeasurement(
            model_id=_string(
                _required(item, "model_id", item_path), f"{item_path}.model_id"
            ),
            model_hash=_string(
                _required(item, "model_hash", item_path), f"{item_path}.model_hash"
            ),
            valid=valid,
            fer_loss=loss,
            true_label=true_label,
            class_probabilities=class_probabilities,
        )
        if measurement.model_id in result:
            raise _trace_error(
                "TRACE_DUPLICATE_MEASUREMENT",
                "artifact has duplicate model measurement",
                model_id=measurement.model_id,
            )
        result[measurement.model_id] = measurement
    return MappingProxyType(dict(sorted(result.items())))


def _parse_anon_row(
    raw: Any, index: int, profile: FrozenProfileBundle | None
) -> AnonTraceRow:
    path = f"$.anon_transactions[{index}]"
    value = _object(raw, path)
    pipeline_id = _string(_required(value, "pipeline_id", path), f"{path}.pipeline_id")
    attempts = tuple(
        _parse_attempt(item, f"{path}.attempts[{attempt_index}]", attempt_index + 1)
        for attempt_index, item in enumerate(
            _array(
                _required(value, "attempts", path), f"{path}.attempts", nonempty=True
            )
        )
    )
    successes = [attempt for attempt in attempts if attempt.encode_success is True]
    if successes and successes[-1] is not attempts[-1]:
        raise _trace_error(
            "TRACE_ATTEMPT_AFTER_SUCCESS",
            "transaction cannot contain attempts after successful encoding",
            path=path,
        )
    if len(successes) > 1:
        raise _trace_error(
            "TRACE_MULTIPLE_SUCCESS",
            "transaction can form only one final artifact",
            path=path,
        )
    formed = _boolean(_required(value, "formed_packet", path), f"{path}.formed_packet")
    final_bytes = _integer(
        _required(value, "final_encoded_size_bytes", path),
        f"{path}.final_encoded_size_bytes",
        minimum=0,
    )
    artifact_key = _optional_string(value.get("artifact_key"), f"{path}.artifact_key")
    measurements = _parse_measurements(
        value.get("fer_measurements", []), f"{path}.fer_measurements"
    )
    if formed:
        if (
            len(successes) != 1
            or successes[0].artifact_key != artifact_key
            or successes[0].encoded_size_bytes != final_bytes
        ):
            raise _trace_error(
                "TRACE_TRANSACTION_SUMMARY",
                "formed-packet summary must exactly match the final successful attempt",
                path=path,
            )
        if not measurements:
            raise _trace_error(
                "TRACE_PAIRED_FER_MISSING",
                "formed artifact requires same-output FER measurements",
                path=path,
            )
    elif successes or final_bytes != 0 or artifact_key is not None or measurements:
        raise _trace_error(
            "TRACE_TRANSACTION_SUMMARY",
            "failed transaction cannot claim artifact, bytes or FER",
            path=path,
        )
    row = AnonTraceRow(
        row_id=_string(_required(value, "row_id", path), f"{path}.row_id"),
        subject_cluster_id=_string(
            _required(value, "subject_cluster_id", path), f"{path}.subject_cluster_id"
        ),
        pipeline_id=pipeline_id,
        pipeline_hash=_string(
            _required(value, "pipeline_hash", path), f"{path}.pipeline_hash"
        ),
        guard_hash=_string(_required(value, "guard_hash", path), f"{path}.guard_hash"),
        encoder_hash=_string(
            _required(value, "encoder_hash", path), f"{path}.encoder_hash"
        ),
        quality_bin=_string(
            _required(value, "quality_bin", path), f"{path}.quality_bin"
        ),
        device_type=_string(
            _required(value, "device_type", path), f"{path}.device_type"
        ),
        context=_parse_context(_required(value, "context", path), f"{path}.context"),
        attempts=attempts,
        formed_packet=formed,
        final_encoded_size_bytes=final_bytes,
        artifact_key=artifact_key,
        fer_measurements=measurements,
    )
    if profile is not None:
        pipeline = profile.pipelines.get(row.pipeline_id)
        if pipeline is None:
            raise _trace_error(
                "TRACE_PIPELINE_UNKNOWN",
                "anon row references unknown pipeline",
                row_id=row.row_id,
            )
        if len(row.attempts) > pipeline.max_attempts:
            raise _trace_error(
                "TRACE_RETRY_BOUND",
                "anon transaction exceeds frozen pipeline attempt limit",
                row_id=row.row_id,
                attempts=len(row.attempts),
                maximum=pipeline.max_attempts,
            )
        if (
            row.pipeline_hash != pipeline.pipeline_hash
            or row.guard_hash != pipeline.guard_hash
            or row.encoder_hash != pipeline.encoder_hash
        ):
            raise _trace_error(
                "TRACE_PIPELINE_VERSION",
                "anon row pipeline evidence differs from frozen profile",
                row_id=row.row_id,
            )
        if (
            row.device_type not in pipeline.supported_devices
            or row.quality_bin not in profile.quality_bins
        ):
            raise _trace_error(
                "TRACE_PIPELINE_SUPPORT",
                "anon row lies outside frozen device/quality support",
                row_id=row.row_id,
            )
        for measurement in row.fer_measurements.values():
            model = profile.edge_models.get(measurement.model_id)
            if model is None or measurement.model_hash != model.model_hash:
                raise _trace_error(
                    "TRACE_FER_VERSION",
                    "paired FER measurement has unknown model/hash",
                    row_id=row.row_id,
                )
    return row


def _parse_prep_row(
    raw: Any, index: int, profile: FrozenProfileBundle | None
) -> PrepTraceRow:
    path = f"$.prep[{index}]"
    value = _object(raw, path)
    row = PrepTraceRow(
        row_id=_string(_required(value, "row_id", path), f"{path}.row_id"),
        fixture_key=_string(
            _required(value, "fixture_key", path), f"{path}.fixture_key"
        ),
        quality_bin=_string(
            _required(value, "quality_bin", path), f"{path}.quality_bin"
        ),
        device_type=_string(
            _required(value, "device_type", path), f"{path}.device_type"
        ),
        context=_parse_context(_required(value, "context", path), f"{path}.context"),
        service_work_s=_number(
            _required(value, "service_work_s", path),
            f"{path}.service_work_s",
            strict_positive=True,
        ),
        dynamic_energy_j=_number(
            _required(value, "dynamic_energy_j", path),
            f"{path}.dynamic_energy_j",
            strict_positive=True,
        ),
        memory_bytes=_integer(
            _required(value, "memory_bytes", path), f"{path}.memory_bytes", minimum=1
        ),
        failed=_boolean(_required(value, "failed", path), f"{path}.failed"),
    )
    if profile is not None:
        supported_devices = {
            device
            for model in profile.local_models.values()
            for device in model.supported_devices
        } | {
            device
            for pipeline in profile.pipelines.values()
            for device in pipeline.supported_devices
        }
        if (
            row.device_type not in supported_devices
            or row.quality_bin not in profile.quality_bins
        ):
            raise _trace_error(
                "TRACE_PREP_SUPPORT",
                "prep row lies outside frozen device/quality support",
                row_id=row.row_id,
            )
    return row


def _parse_local_row(
    raw: Any, index: int, profile: FrozenProfileBundle | None
) -> LocalFERTraceRow:
    path = f"$.local_fer[{index}]"
    value = _object(raw, path)
    failed = _boolean(_required(value, "failed", path), f"{path}.failed")
    loss_raw = value.get("fer_loss")
    loss = (
        None
        if loss_raw is None
        else _number(loss_raw, f"{path}.fer_loss", minimum=0.0, maximum=1.0)
    )
    if failed == (loss is not None):
        raise _trace_error(
            "TRACE_FER_VALIDITY",
            "successful local row needs FER loss; failed row must not have one",
            path=path,
        )
    true_label, class_probabilities = _parse_fer_prediction(
        value, path, valid=not failed
    )
    row = LocalFERTraceRow(
        row_id=_string(_required(value, "row_id", path), f"{path}.row_id"),
        model_id=_string(_required(value, "model_id", path), f"{path}.model_id"),
        model_hash=_string(_required(value, "model_hash", path), f"{path}.model_hash"),
        quality_bin=_string(
            _required(value, "quality_bin", path), f"{path}.quality_bin"
        ),
        device_type=_string(
            _required(value, "device_type", path), f"{path}.device_type"
        ),
        context=_parse_context(_required(value, "context", path), f"{path}.context"),
        service_work_s=_number(
            _required(value, "service_work_s", path),
            f"{path}.service_work_s",
            strict_positive=True,
        ),
        dynamic_energy_j=_number(
            _required(value, "dynamic_energy_j", path),
            f"{path}.dynamic_energy_j",
            strict_positive=True,
        ),
        memory_bytes=_integer(
            _required(value, "memory_bytes", path), f"{path}.memory_bytes", minimum=1
        ),
        failed=failed,
        fer_loss=loss,
        subject_cluster_id=_optional_string(
            value.get("subject_cluster_id"), f"{path}.subject_cluster_id"
        ),
        true_label=true_label,
        class_probabilities=class_probabilities,
    )
    if profile is not None:
        model = profile.local_models.get(row.model_id)
        if model is None or model.model_hash != row.model_hash:
            raise _trace_error(
                "TRACE_LOCAL_MODEL",
                "local row has unknown model/hash",
                row_id=row.row_id,
            )
        if (
            row.device_type not in model.supported_devices
            or row.quality_bin not in profile.quality_bins
        ):
            raise _trace_error(
                "TRACE_LOCAL_SUPPORT",
                "local row lies outside model support",
                row_id=row.row_id,
            )
    return row


def _parse_edge_row(
    raw: Any, index: int, profile: FrozenProfileBundle | None
) -> EdgeFERTraceRow:
    path = f"$.edge_fer[{index}]"
    value = _object(raw, path)
    failed = _boolean(_required(value, "failed", path), f"{path}.failed")
    loss_raw = value.get("fer_loss")
    loss = (
        None
        if loss_raw is None
        else _number(loss_raw, f"{path}.fer_loss", minimum=0.0, maximum=1.0)
    )
    if failed == (loss is not None):
        raise _trace_error(
            "TRACE_FER_VALIDITY",
            "successful edge row needs FER loss; failed row must not have one",
            path=path,
        )
    true_label, class_probabilities = _parse_fer_prediction(
        value, path, valid=not failed
    )
    row = EdgeFERTraceRow(
        row_id=_string(_required(value, "row_id", path), f"{path}.row_id"),
        artifact_key=_string(
            _required(value, "artifact_key", path), f"{path}.artifact_key"
        ),
        pipeline_id=_string(
            _required(value, "pipeline_id", path), f"{path}.pipeline_id"
        ),
        quality_bin=_string(
            _required(value, "quality_bin", path), f"{path}.quality_bin"
        ),
        rsu_id=_string(_required(value, "rsu_id", path), f"{path}.rsu_id"),
        model_id=_string(_required(value, "model_id", path), f"{path}.model_id"),
        model_hash=_string(_required(value, "model_hash", path), f"{path}.model_hash"),
        context=_parse_context(_required(value, "context", path), f"{path}.context"),
        ingress_work_s=_number(
            _required(value, "ingress_work_s", path),
            f"{path}.ingress_work_s",
            strict_positive=True,
        ),
        ingress_energy_j=_number(
            _required(value, "ingress_energy_j", path),
            f"{path}.ingress_energy_j",
            strict_positive=True,
        ),
        gpu_work_s=_number(
            _required(value, "gpu_work_s", path),
            f"{path}.gpu_work_s",
            strict_positive=True,
        ),
        gpu_energy_j=_number(
            _required(value, "gpu_energy_j", path),
            f"{path}.gpu_energy_j",
            strict_positive=True,
        ),
        vram_bytes=_integer(
            _required(value, "vram_bytes", path), f"{path}.vram_bytes", minimum=1
        ),
        result_size_bits=_integer(
            _required(value, "result_size_bits", path),
            f"{path}.result_size_bits",
            minimum=1,
        ),
        ingress_failed=_boolean(
            _required(value, "ingress_failed", path), f"{path}.ingress_failed"
        ),
        failed=failed,
        fer_loss=loss,
        true_label=true_label,
        class_probabilities=class_probabilities,
    )
    if profile is not None:
        model = profile.edge_models.get(row.model_id)
        if model is None or model.model_hash != row.model_hash:
            raise _trace_error(
                "TRACE_EDGE_MODEL", "edge row has unknown model/hash", row_id=row.row_id
            )
        if (
            row.rsu_id not in model.supported_rsus
            or row.pipeline_id not in model.supported_pipelines
        ):
            raise _trace_error(
                "TRACE_EDGE_SUPPORT",
                "edge row lies outside model support",
                row_id=row.row_id,
            )
        if row.quality_bin not in profile.quality_bins:
            raise _trace_error(
                "TRACE_EDGE_QUALITY",
                "edge row has unknown quality bin",
                row_id=row.row_id,
            )
    return row


def _parse_wireless(raw: Any, index: int, start: float, end: float) -> WirelessSegment:
    path = f"$.environment.wireless_segments[{index}]"
    value = _object(raw, path)
    try:
        direction = TransferDirection(
            _string(_required(value, "direction", path), f"{path}.direction")
        )
    except ValueError as exc:
        raise _trace_error(
            "TRACE_DIRECTION", "direction must be UL or DL", path=f"{path}.direction"
        ) from exc
    link_state = _string(_required(value, "link_state", path), f"{path}.link_state")
    if link_state not in _LINK_STATES:
        raise _trace_error(
            "TRACE_LINK_STATE", "unknown link state", path=f"{path}.link_state"
        )
    segment = WirelessSegment(
        segment_id=_string(_required(value, "segment_id", path), f"{path}.segment_id"),
        vehicle_id=_string(_required(value, "vehicle_id", path), f"{path}.vehicle_id"),
        rsu_id=_string(_required(value, "rsu_id", path), f"{path}.rsu_id"),
        direction=direction,
        start_time_s=_number(
            _required(value, "start_time_s", path),
            f"{path}.start_time_s",
            minimum=start,
        ),
        end_time_s=_number(
            _required(value, "end_time_s", path), f"{path}.end_time_s", maximum=end
        ),
        goodput_bps=_number(
            _required(value, "goodput_bps", path), f"{path}.goodput_bps", minimum=0.0
        ),
        transmitter_power_w=_number(
            _required(value, "transmitter_power_w", path),
            f"{path}.transmitter_power_w",
            minimum=0.0,
        ),
        receiver_power_w=_number(
            _required(value, "receiver_power_w", path),
            f"{path}.receiver_power_w",
            minimum=0.0,
        ),
        link_state=link_state,
    )
    if segment.end_time_s <= segment.start_time_s:
        raise _trace_error(
            "TRACE_INTERVAL", "wireless interval must have positive duration", path=path
        )
    if link_state != "connected" and segment.goodput_bps != 0:
        raise _trace_error(
            "TRACE_LINK_SERVICE",
            "outage/loss/handover interval must have zero goodput",
            path=path,
        )
    if link_state == "connected" and segment.goodput_bps <= 0:
        raise _trace_error(
            "TRACE_LINK_SERVICE",
            "connected interval must have positive goodput",
            path=path,
        )
    if link_state == "connected" and (
        segment.transmitter_power_w <= 0 or segment.receiver_power_w <= 0
    ):
        raise _trace_error(
            "TRACE_LINK_POWER",
            "connected interval must have positive paired transmitter and receiver power",
            path=path,
        )
    return segment


def _parse_thermal(raw: Any, index: int, start: float, end: float) -> ThermalSegment:
    path = f"$.environment.thermal_segments[{index}]"
    value = _object(raw, path)
    owner_type = _string(_required(value, "owner_type", path), f"{path}.owner_type")
    if owner_type not in _OWNER_TYPES:
        raise _trace_error(
            "TRACE_OWNER_TYPE",
            "owner_type must be vehicle or rsu",
            path=f"{path}.owner_type",
        )
    segment = ThermalSegment(
        segment_id=_string(_required(value, "segment_id", path), f"{path}.segment_id"),
        owner_type=owner_type,
        owner_id=_string(_required(value, "owner_id", path), f"{path}.owner_id"),
        resource=_string(_required(value, "resource", path), f"{path}.resource"),
        start_time_s=_number(
            _required(value, "start_time_s", path),
            f"{path}.start_time_s",
            minimum=start,
        ),
        end_time_s=_number(
            _required(value, "end_time_s", path), f"{path}.end_time_s", maximum=end
        ),
        state=_string(_required(value, "state", path), f"{path}.state"),
        service_rate_multiplier=_number(
            _required(value, "service_rate_multiplier", path),
            f"{path}.service_rate_multiplier",
            strict_positive=True,
            maximum=1.0,
        ),
        dynamic_power_multiplier=_number(
            _required(value, "dynamic_power_multiplier", path),
            f"{path}.dynamic_power_multiplier",
            strict_positive=True,
        ),
    )
    if segment.end_time_s <= segment.start_time_s:
        raise _trace_error(
            "TRACE_INTERVAL", "thermal interval must have positive duration", path=path
        )
    return segment


def _parse_event(raw: Any, index: int, start: float, end: float) -> ExogenousEvent:
    path = f"$.environment.events[{index}]"
    value = _object(raw, path)
    event_type = _string(_required(value, "event_type", path), f"{path}.event_type")
    if event_type not in _EVENT_TYPES:
        raise _trace_error(
            "TRACE_EVENT_TYPE",
            "unknown exogenous event type",
            path=f"{path}.event_type",
        )
    target_type = _string(_required(value, "target_type", path), f"{path}.target_type")
    old = _optional_string(value.get("old_version"), f"{path}.old_version")
    new = _optional_string(value.get("new_version"), f"{path}.new_version")
    if event_type in {"MODEL_VERSION", "PROFILE_VERSION", "PROTOCOL_VERSION"} and (
        old is None or new is None
    ):
        raise _trace_error(
            "TRACE_VERSION_EVENT",
            "version event requires old_version and new_version",
            path=path,
        )
    details = value.get("details", {})
    if not isinstance(details, Mapping):
        raise _trace_error(
            "TRACE_FIELD_TYPE", "event details must be object", path=f"{path}.details"
        )
    maintenance_work_raw = value.get("maintenance_work_s")
    maintenance_energy_raw = value.get("maintenance_energy_j")
    is_rsu_model_maintenance = target_type == "rsu" and event_type in {
        "MODEL_VERSION",
        "MODEL_CACHE",
    }
    if is_rsu_model_maintenance and (
        maintenance_work_raw is None or maintenance_energy_raw is None
    ):
        raise _trace_error(
            "TRACE_MODEL_MAINTENANCE_PHYSICS",
            "RSU model/cache event requires positive maintenance_work_s and maintenance_energy_j",
            path=path,
        )
    if not is_rsu_model_maintenance and (
        maintenance_work_raw is not None or maintenance_energy_raw is not None
    ):
        raise _trace_error(
            "TRACE_MODEL_MAINTENANCE_SCOPE",
            "maintenance work/energy are valid only for RSU MODEL_VERSION or MODEL_CACHE events",
            path=path,
        )
    maintenance_work_s = (
        None
        if maintenance_work_raw is None
        else _number(
            maintenance_work_raw,
            f"{path}.maintenance_work_s",
            strict_positive=True,
        )
    )
    maintenance_energy_j = (
        None
        if maintenance_energy_raw is None
        else _number(
            maintenance_energy_raw,
            f"{path}.maintenance_energy_j",
            strict_positive=True,
        )
    )
    return ExogenousEvent(
        event_id=_string(_required(value, "event_id", path), f"{path}.event_id"),
        time_s=_number(
            _required(value, "time_s", path),
            f"{path}.time_s",
            minimum=start,
            maximum=end,
        ),
        event_type=event_type,
        target_type=target_type,
        target_id=_string(_required(value, "target_id", path), f"{path}.target_id"),
        resource=_optional_string(value.get("resource"), f"{path}.resource"),
        old_version=old,
        new_version=new,
        permanent=_boolean(value.get("permanent", False), f"{path}.permanent"),
        details=deep_freeze(details),
        maintenance_work_s=maintenance_work_s,
        maintenance_energy_j=maintenance_energy_j,
    )


def _parse_arrival(raw: Any, index: int, start: float, end: float) -> TaskArrival:
    path = f"$.task_arrivals[{index}]"
    value = _object(raw, path)
    features_raw = _object(
        _required(value, "quality_features", path), f"{path}.quality_features"
    )
    features = {
        str(key): _number(number, f"{path}.quality_features.{key}")
        for key, number in features_raw.items()
    }
    candidates = _stable_unique_strings(
        _required(value, "quality_candidates", path), f"{path}.quality_candidates"
    )
    probability_object = _object(
        _required(value, "quality_probabilities", path),
        f"{path}.quality_probabilities",
    )
    parsed_probabilities = {
        _string(region_id, f"{path}.quality_probabilities key"): _number(
            probability,
            f"{path}.quality_probabilities.{region_id}",
            minimum=0.0,
            maximum=1.0,
        )
        for region_id, probability in probability_object.items()
    }
    if set(parsed_probabilities) != set(candidates):
        raise _trace_error(
            "TRACE_QUALITY_PROBABILITY_SUPPORT",
            "quality probabilities must cover exactly the conformal candidate regions",
            path=path,
        )
    # Preserve the conformal candidate order in the paired probability tuple.
    # The tuple crosses the online observation/scenario boundary, so an
    # independent alphabetical sort would silently detach positional consumers
    # from the candidate sequence even though the mapping support is identical.
    probabilities = tuple(
        (region_id, parsed_probabilities[region_id]) for region_id in candidates
    )
    probability_total = sum(probability for _, probability in probabilities)
    if not math.isclose(probability_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise _trace_error(
            "TRACE_QUALITY_PROBABILITY_SUM",
            "candidate quality probabilities must sum to one",
            path=path,
            total=probability_total,
        )
    arrival = TaskArrival(
        task_id=_string(_required(value, "task_id", path), f"{path}.task_id"),
        fixture_key=_string(
            _required(value, "fixture_key", path), f"{path}.fixture_key"
        ),
        vehicle_id=_string(_required(value, "vehicle_id", path), f"{path}.vehicle_id"),
        arrival_time_s=_number(
            _required(value, "arrival_time_s", path),
            f"{path}.arrival_time_s",
            minimum=start,
            maximum=end,
        ),
        relative_deadline_s=_number(
            _required(value, "relative_deadline_s", path),
            f"{path}.relative_deadline_s",
            strict_positive=True,
        ),
        quality_candidates=candidates,
        quality_probabilities=probabilities,
        true_quality_region=_string(
            _required(value, "true_quality_region", path),
            f"{path}.true_quality_region",
        ),
        ood=_boolean(_required(value, "ood", path), f"{path}.ood"),
        quality_features=MappingProxyType(dict(sorted(features.items()))),
    )
    if arrival.absolute_deadline_s > end + 1e-12:
        raise _trace_error(
            "TRACE_DEADLINE_HORIZON",
            "trace horizon must include every task absolute deadline",
            task_id=arrival.task_id,
            deadline=arrival.absolute_deadline_s,
            horizon_end=end,
        )
    return arrival


def _ensure_unique_ids(section: str, rows: Iterable[Any], attr: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = getattr(row, attr)
        if value in seen:
            raise _trace_error(
                "TRACE_DUPLICATE_ID",
                "trace object ID must be unique",
                section=section,
                value=value,
            )
        seen.add(value)


def _build_index(rows: Iterable[T], key_function: Any) -> Mapping[Any, tuple[T, ...]]:
    index: dict[Any, list[T]] = {}
    for row in rows:
        index.setdefault(key_function(row), []).append(row)
    return MappingProxyType(
        {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: getattr(
                        item, "row_id", getattr(item, "segment_id", "")
                    ),
                )
            )
            for key, values in sorted(index.items(), key=lambda pair: str(pair[0]))
        }
    )


def _validate_contiguous_segments(
    index: Mapping[Any, tuple[Any, ...]], start: float, end: float, section: str
) -> None:
    tolerance = 1e-12
    for key, segments in index.items():
        ordered = tuple(
            sorted(
                segments,
                key=lambda segment: (
                    segment.start_time_s,
                    segment.end_time_s,
                    segment.segment_id,
                ),
            )
        )
        if (
            abs(ordered[0].start_time_s - start) > tolerance
            or abs(ordered[-1].end_time_s - end) > tolerance
        ):
            raise _trace_error(
                "TRACE_COVERAGE",
                "piecewise environment trace must cover the declared horizon",
                section=section,
                key=str(key),
                expected=[start, end],
                actual=[ordered[0].start_time_s, ordered[-1].end_time_s],
            )
        cursor = start
        for segment in ordered:
            if abs(segment.start_time_s - cursor) > tolerance:
                raise _trace_error(
                    "TRACE_SEGMENT_GAP_OVERLAP",
                    "segments must be contiguous and non-overlapping",
                    section=section,
                    key=str(key),
                    cursor=cursor,
                    next_start=segment.start_time_s,
                )
            cursor = segment.end_time_s


def _validate_artifact_pairing(
    anon_rows: tuple[AnonTraceRow, ...], edge_rows: tuple[EdgeFERTraceRow, ...]
) -> None:
    artifacts = {
        row.artifact_key: row
        for row in anon_rows
        if row.formed_packet and row.artifact_key
    }
    edge_by_artifact: dict[str, list[EdgeFERTraceRow]] = {}
    for edge in edge_rows:
        anon = artifacts.get(edge.artifact_key)
        if anon is None:
            raise _trace_error(
                "TRACE_EDGE_ARTIFACT",
                "edge measurement has no paired anonymization artifact",
                row_id=edge.row_id,
            )
        if edge.pipeline_id != anon.pipeline_id or edge.quality_bin != anon.quality_bin:
            raise _trace_error(
                "TRACE_EDGE_PAIR_CONTEXT",
                "edge row pipeline/quality does not match artifact",
                row_id=edge.row_id,
            )
        measurement = anon.fer_measurements.get(edge.model_id)
        if measurement is None or measurement.model_hash != edge.model_hash:
            raise _trace_error(
                "TRACE_PAIRED_FER_MISSING",
                "edge row model is not measured on the same artifact",
                row_id=edge.row_id,
            )
        if (
            measurement.valid != (not edge.failed)
            or measurement.fer_loss != edge.fer_loss
            or measurement.true_label != edge.true_label
            or measurement.class_probabilities != edge.class_probabilities
        ):
            raise _trace_error(
                "TRACE_PAIRED_FER_MISMATCH",
                "edge row FER outcome differs from artifact measurement",
                row_id=edge.row_id,
            )
        edge_by_artifact.setdefault(edge.artifact_key, []).append(edge)
    for artifact_key, anon in artifacts.items():
        rows = edge_by_artifact.get(artifact_key, [])
        measured_models = set(anon.fer_measurements)
        paired_models = {row.model_id for row in rows}
        if not measured_models.issubset(paired_models):
            raise _trace_error(
                "TRACE_PAIRED_FER_MISSING",
                "formed artifact lacks an edge trace for a measured model",
                artifact_key=artifact_key,
                missing=sorted(measured_models - paired_models),
            )


def _assert_deployment_bound(
    *,
    component_id: str,
    row_id: str,
    field: str,
    actual: int | float,
    bound: int | float,
) -> None:
    if float(actual) > float(bound) + 1e-12:
        raise _trace_error(
            "TRACE_DEPLOYMENT_BOUND_EXCEEDED",
            "trace row exceeds a preregistered physical support bound",
            component_id=component_id,
            row_id=row_id,
            field=field,
            actual=actual,
            bound=bound,
            bound_source="frozen_profile",
        )


def _validate_deployment_resource_bounds(
    profile: FrozenProfileBundle,
    prep_rows: tuple[PrepTraceRow, ...],
    anon_rows: tuple[AnonTraceRow, ...],
    local_rows: tuple[LocalFERTraceRow, ...],
    edge_rows: tuple[EdgeFERTraceRow, ...],
) -> None:
    """Reject evaluation/scenario rows outside offline physical envelopes."""

    prep_bounds = profile.preprocessing_resource_bounds
    for row in prep_rows:
        for metric, actual, key in (
            ("memory_bytes", row.memory_bytes, "max_memory_bytes"),
            ("service_work_s", row.service_work_s, "max_service_work_s"),
            ("dynamic_energy_j", row.dynamic_energy_j, "max_dynamic_energy_j"),
        ):
            _assert_deployment_bound(
                component_id="public_preprocessing",
                row_id=row.row_id,
                field=metric,
                actual=actual,
                bound=prep_bounds[key],
            )

    for row in anon_rows:
        bounds = profile.pipelines[row.pipeline_id].deployment_resource_bounds
        for attempt in row.attempts:
            attempt_id = f"{row.row_id}:attempt-{attempt.attempt_index}"
            for metric, actual, key in (
                (
                    "peak_memory_bytes",
                    attempt.peak_memory_bytes,
                    "max_peak_memory_bytes",
                ),
                ("anon_work_s", attempt.anon_work_s, "max_anon_work_s"),
                ("anon_energy_j", attempt.anon_energy_j, "max_anon_energy_j"),
            ):
                _assert_deployment_bound(
                    component_id=row.pipeline_id,
                    row_id=attempt_id,
                    field=metric,
                    actual=actual,
                    bound=bounds[key],
                )
            for metric, actual, key in (
                ("guard_work_s", attempt.guard_work_s, "max_guard_work_s"),
                ("guard_energy_j", attempt.guard_energy_j, "max_guard_energy_j"),
                ("encode_work_s", attempt.encode_work_s, "max_encode_work_s"),
                ("encode_energy_j", attempt.encode_energy_j, "max_encode_energy_j"),
                ("encoded_size_bytes", attempt.encoded_size_bytes, "max_output_bytes"),
            ):
                if actual is not None:
                    _assert_deployment_bound(
                        component_id=row.pipeline_id,
                        row_id=attempt_id,
                        field=metric,
                        actual=actual,
                        bound=bounds[key],
                    )
        _assert_deployment_bound(
            component_id=row.pipeline_id,
            row_id=row.row_id,
            field="final_encoded_size_bytes",
            actual=row.final_encoded_size_bytes,
            bound=bounds["max_output_bytes"],
        )

    for row in local_rows:
        bounds = profile.local_models[row.model_id].deployment_resource_bounds
        for metric, actual, key in (
            ("memory_bytes", row.memory_bytes, "max_memory_bytes"),
            ("service_work_s", row.service_work_s, "max_service_work_s"),
            ("dynamic_energy_j", row.dynamic_energy_j, "max_dynamic_energy_j"),
        ):
            _assert_deployment_bound(
                component_id=row.model_id,
                row_id=row.row_id,
                field=metric,
                actual=actual,
                bound=bounds[key],
            )

    for row in edge_rows:
        bounds = profile.edge_models[row.model_id].deployment_resource_bounds
        for metric, actual, key in (
            ("vram_bytes", row.vram_bytes, "max_vram_bytes"),
            ("ingress_work_s", row.ingress_work_s, "max_ingress_work_s"),
            ("ingress_energy_j", row.ingress_energy_j, "max_ingress_energy_j"),
            ("gpu_work_s", row.gpu_work_s, "max_gpu_work_s"),
            ("gpu_energy_j", row.gpu_energy_j, "max_gpu_energy_j"),
            ("result_size_bits", row.result_size_bits, "max_result_size_bits"),
        ):
            _assert_deployment_bound(
                component_id=row.model_id,
                row_id=row.row_id,
                field=metric,
                actual=actual,
                bound=bounds[key],
            )


def _validate_exogenous_events(
    events: tuple[ExogenousEvent, ...], profile: FrozenProfileBundle | None
) -> None:
    active_faults: set[tuple[str, str, str | None]] = set()
    profile_version = None if profile is None else profile.profile_hash
    protocol_version = None if profile is None else profile.protocol_version
    model_versions: dict[tuple[str, str], str] = {}
    for event in events:
        if (
            event.target_type == "rsu"
            and event.event_type in {"MODEL_VERSION", "MODEL_CACHE"}
            and (
                event.maintenance_work_s is None
                or event.maintenance_work_s <= 0
                or event.maintenance_energy_j is None
                or event.maintenance_energy_j <= 0
            )
        ):
            raise _trace_error(
                "TRACE_MODEL_MAINTENANCE_PHYSICS",
                "RSU model/cache event requires positive maintenance work and energy",
                event_id=event.event_id,
            )
        fault_key = (event.target_type, event.target_id, event.resource)
        if event.event_type == "DEVICE_FAULT_START":
            if fault_key in active_faults:
                raise _trace_error(
                    "TRACE_FAULT_SEQUENCE",
                    "fault starts twice without recovery",
                    event_id=event.event_id,
                )
            active_faults.add(fault_key)
        elif event.event_type == "DEVICE_FAULT_END":
            if fault_key not in active_faults:
                raise _trace_error(
                    "TRACE_FAULT_SEQUENCE",
                    "fault recovery has no active fault",
                    event_id=event.event_id,
                )
            active_faults.remove(fault_key)
        elif event.event_type == "DEVICE_FAULT_PERMANENT":
            if fault_key in active_faults:
                active_faults.remove(fault_key)

        if profile is None:
            continue
        if event.event_type in {"MODEL_VERSION", "MODEL_CACHE"}:
            model_id = str(event.details.get("model_id", ""))
            model = (
                profile.edge_models.get(model_id)
                if event.target_type == "rsu"
                else profile.local_models.get(model_id)
            )
            if model is None:
                raise _trace_error(
                    "TRACE_VERSION_MODEL",
                    "model/cache event requires a known target-compatible details.model_id",
                    event_id=event.event_id,
                )
            if event.event_type == "MODEL_CACHE":
                continue
            key = (event.target_id, model_id)
            expected_old = model_versions.get(key, model.model_hash)
            if event.old_version != expected_old:
                raise _trace_error(
                    "TRACE_VERSION_SEQUENCE",
                    "model version event old_version does not match the frozen/current version",
                    event_id=event.event_id,
                    expected=expected_old,
                    actual=event.old_version,
                )
            model_versions[key] = event.new_version or ""
        elif event.event_type == "PROFILE_VERSION":
            if event.old_version != profile_version:
                raise _trace_error(
                    "TRACE_VERSION_SEQUENCE",
                    "profile version event old_version does not match current hash",
                    event_id=event.event_id,
                    expected=profile_version,
                    actual=event.old_version,
                )
            profile_version = event.new_version
        elif event.event_type == "PROTOCOL_VERSION":
            if event.old_version != protocol_version:
                raise _trace_error(
                    "TRACE_VERSION_SEQUENCE",
                    "protocol version event old_version does not match current protocol",
                    event_id=event.event_id,
                    expected=protocol_version,
                    actual=event.old_version,
                )
            protocol_version = event.new_version
    if active_faults:
        raise _trace_error(
            "TRACE_FAULT_UNCLOSED",
            "transient fault remains active at trace end; use a permanent-fault event if intended",
            active_faults=sorted(str(item) for item in active_faults),
        )


def load_trace(
    path: str | Path, profile: FrozenProfileBundle | None = None
) -> TraceBundle:
    """Load and validate one immutable joint trace bundle."""

    resolved = Path(path).resolve()
    try:
        raw = load_strict_json(resolved, purpose="trace")
    except ProfileValidationError as exc:
        context = dict(exc.detail.context)
        raise TraceValidationError(
            exc.detail.code, exc.detail.message, **context
        ) from exc
    trace_hash = _string(_required(raw, "trace_hash", "$"), "$.trace_hash").lower()
    calculated = canonical_document_sha256(raw, "trace_hash")
    if trace_hash != calculated:
        raise _trace_error(
            "TRACE_HASH_MISMATCH",
            "trace canonical content does not match trace_hash",
            expected=trace_hash,
            calculated=calculated,
            path=str(resolved),
        )
    schema_version = _string(_required(raw, "schema_version", "$"), "$.schema_version")
    protocol_version = _string(
        _required(raw, "protocol_version", "$"), "$.protocol_version"
    )
    trace_version = _string(_required(raw, "trace_version", "$"), "$.trace_version")
    profile_hash = _string(_required(raw, "profile_hash", "$"), "$.profile_hash")
    data_kind = _string(_required(raw, "data_kind", "$"), "$.data_kind")
    evidence_status = _string(
        _required(raw, "evidence_status", "$"), "$.evidence_status"
    )
    if data_kind not in {"synthetic", "measured", "numerical_simulation"}:
        raise _trace_error(
            "TRACE_DATA_KIND",
            "data_kind must be synthetic, measured or numerical_simulation",
            value=data_kind,
        )
    if data_kind == "synthetic" and evidence_status != "synthetic_fixture_only":
        raise _trace_error(
            "TRACE_SYNTHETIC_CLAIM",
            "synthetic trace must be marked synthetic_fixture_only",
        )
    if (
        data_kind == "numerical_simulation"
        and evidence_status != "frozen_numerical_model"
    ):
        raise _trace_error(
            "TRACE_NUMERICAL_CLAIM",
            "numerical simulation trace must be marked frozen_numerical_model",
        )
    if profile is not None:
        if profile_hash != profile.profile_hash:
            raise _trace_error(
                "TRACE_PROFILE_MISMATCH",
                "trace was not built against the loaded frozen profile",
                trace_profile_hash=profile_hash,
                loaded_profile_hash=profile.profile_hash,
            )
        if protocol_version != profile.protocol_version:
            raise _trace_error(
                "TRACE_PROTOCOL_MISMATCH",
                "trace protocol differs from loaded profile protocol",
                trace_protocol=protocol_version,
                profile_protocol=profile.protocol_version,
            )

    horizon = _object(_required(raw, "horizon", "$"), "$.horizon")
    start = _number(
        _required(horizon, "start_time_s", "$.horizon"),
        "$.horizon.start_time_s",
        minimum=0.0,
    )
    end = _number(_required(horizon, "end_time_s", "$.horizon"), "$.horizon.end_time_s")
    if end <= start:
        raise _trace_error(
            "TRACE_HORIZON",
            "trace horizon must have positive duration",
            start=start,
            end=end,
        )
    seed = _integer(_required(raw, "seed", "$"), "$.seed", minimum=0)

    anon_rows = tuple(
        _parse_anon_row(item, index, profile)
        for index, item in enumerate(
            _array(
                _required(raw, "anon_transactions", "$"),
                "$.anon_transactions",
                nonempty=True,
            )
        )
    )
    prep_rows = tuple(
        _parse_prep_row(item, index, profile)
        for index, item in enumerate(
            _array(_required(raw, "prep", "$"), "$.prep", nonempty=True)
        )
    )
    local_rows = tuple(
        _parse_local_row(item, index, profile)
        for index, item in enumerate(
            _array(_required(raw, "local_fer", "$"), "$.local_fer", nonempty=True)
        )
    )
    edge_rows = tuple(
        _parse_edge_row(item, index, profile)
        for index, item in enumerate(
            _array(_required(raw, "edge_fer", "$"), "$.edge_fer", nonempty=True)
        )
    )
    environment = _object(_required(raw, "environment", "$"), "$.environment")
    wireless = tuple(
        _parse_wireless(item, index, start, end)
        for index, item in enumerate(
            _array(
                _required(environment, "wireless_segments", "$.environment"),
                "$.environment.wireless_segments",
                nonempty=True,
            )
        )
    )
    thermal = tuple(
        _parse_thermal(item, index, start, end)
        for index, item in enumerate(
            _array(
                _required(environment, "thermal_segments", "$.environment"),
                "$.environment.thermal_segments",
                nonempty=True,
            )
        )
    )
    events = tuple(
        sorted(
            (
                _parse_event(item, index, start, end)
                for index, item in enumerate(
                    _array(environment.get("events", []), "$.environment.events")
                )
            ),
            key=lambda item: (item.time_s, item.event_id),
        )
    )
    arrivals = tuple(
        sorted(
            (
                _parse_arrival(item, index, start, end)
                for index, item in enumerate(
                    _array(
                        _required(raw, "task_arrivals", "$"),
                        "$.task_arrivals",
                        nonempty=True,
                    )
                )
            ),
            key=lambda item: (item.arrival_time_s, item.vehicle_id, item.task_id),
        )
    )
    known_quality_regions = set(profile.quality_bins)
    for arrival in arrivals:
        unknown_candidates = set(arrival.quality_candidates) - known_quality_regions
        if unknown_candidates:
            raise _trace_error(
                "TRACE_ARRIVAL_QUALITY_UNKNOWN",
                "arrival conformal set references unknown frozen quality regions",
                task_id=arrival.task_id,
                unknown=sorted(unknown_candidates),
            )
        if arrival.true_quality_region not in known_quality_regions:
            raise _trace_error(
                "TRACE_TRUE_QUALITY_UNKNOWN",
                "simulator-only true quality region is not in the frozen partition",
                task_id=arrival.task_id,
                true_quality_region=arrival.true_quality_region,
            )

    _ensure_unique_ids("anon_transactions", anon_rows, "row_id")
    _ensure_unique_ids("prep", prep_rows, "row_id")
    _ensure_unique_ids("local_fer", local_rows, "row_id")
    _ensure_unique_ids("edge_fer", edge_rows, "row_id")
    _ensure_unique_ids("wireless_segments", wireless, "segment_id")
    _ensure_unique_ids("thermal_segments", thermal, "segment_id")
    _ensure_unique_ids("events", events, "event_id")
    _ensure_unique_ids("task_arrivals", arrivals, "task_id")
    if profile is not None:
        _validate_deployment_resource_bounds(
            profile, prep_rows, anon_rows, local_rows, edge_rows
        )
    _validate_artifact_pairing(anon_rows, edge_rows)
    if data_kind == "numerical_simulation":
        missing_predictions = (
            [
                row.row_id
                for row in local_rows
                if not row.failed
                and (row.true_label is None or not row.class_probabilities)
            ]
            + [
                row.row_id
                for row in edge_rows
                if not row.failed
                and (row.true_label is None or not row.class_probabilities)
            ]
            + [
                f"{row.row_id}:{measurement.model_id}"
                for row in anon_rows
                for measurement in row.fer_measurements.values()
                if measurement.valid
                and (
                    measurement.true_label is None
                    or not measurement.class_probabilities
                )
            ]
        )
        if missing_predictions:
            raise _trace_error(
                "TRACE_NUMERICAL_FER_PREDICTION",
                "successful numerical FER rows require frozen labels and probability vectors",
                rows=missing_predictions[:20],
                missing_count=len(missing_predictions),
            )
        class_spaces = {
            tuple(class_name for class_name, _ in probabilities)
            for probabilities in (
                [row.class_probabilities for row in local_rows if not row.failed]
                + [row.class_probabilities for row in edge_rows if not row.failed]
            )
        }
        if len(class_spaces) != 1:
            raise _trace_error(
                "TRACE_NUMERICAL_FER_CLASS_SPACE",
                "all numerical FER probability vectors must use one frozen class space",
                class_spaces=[list(value) for value in sorted(class_spaces)],
            )
    _validate_exogenous_events(events, profile)

    prep_index = _build_index(
        prep_rows,
        lambda row: (
            row.fixture_key,
            row.quality_bin,
            row.device_type,
            row.context.thermal_state,
            row.context.power_mode,
            row.context.memory_pressure,
        ),
    )
    anon_index = _build_index(
        anon_rows,
        lambda row: (
            row.pipeline_id,
            row.quality_bin,
            row.device_type,
            row.context.thermal_state,
            row.context.power_mode,
            row.context.memory_pressure,
        ),
    )
    local_index = _build_index(
        local_rows,
        lambda row: (
            row.model_id,
            row.quality_bin,
            row.device_type,
            row.context.thermal_state,
            row.context.power_mode,
            row.context.memory_pressure,
        ),
    )
    edge_index = _build_index(
        edge_rows,
        lambda row: (
            row.rsu_id,
            row.model_id,
            row.pipeline_id,
            row.artifact_key,
            row.quality_bin,
            row.context.thermal_state,
            row.context.power_mode,
            row.context.memory_pressure,
        ),
    )
    wireless_index = _build_index(
        wireless, lambda row: (row.vehicle_id, row.rsu_id, row.direction)
    )
    thermal_index = _build_index(
        thermal, lambda row: (row.owner_type, row.owner_id, row.resource)
    )
    _validate_contiguous_segments(wireless_index, start, end, "wireless")
    _validate_contiguous_segments(thermal_index, start, end, "thermal")

    sources = _object(_required(raw, "parameter_sources", "$"), "$.parameter_sources")
    metadata = _object(_required(raw, "metadata", "$"), "$.metadata")
    try:
        validate_parameter_sources(sources, data_kind=data_kind, error_prefix="TRACE")
    except ProfileValidationError as exc:
        raise TraceValidationError(
            exc.detail.code, exc.detail.message, **dict(exc.detail.context)
        ) from exc

    return TraceBundle(
        schema_version=schema_version,
        protocol_version=protocol_version,
        trace_version=trace_version,
        trace_hash=trace_hash,
        profile_hash=profile_hash,
        data_kind=data_kind,
        evidence_status=evidence_status,
        seed=seed,
        horizon_start_s=start,
        horizon_end_s=end,
        anon_rows=tuple(sorted(anon_rows, key=lambda row: row.row_id)),
        prep_rows=tuple(sorted(prep_rows, key=lambda row: row.row_id)),
        local_rows=tuple(sorted(local_rows, key=lambda row: row.row_id)),
        edge_rows=tuple(sorted(edge_rows, key=lambda row: row.row_id)),
        wireless=tuple(
            sorted(wireless, key=lambda row: (row.start_time_s, row.segment_id))
        ),
        thermal=tuple(
            sorted(thermal, key=lambda row: (row.start_time_s, row.segment_id))
        ),
        exogenous_events=events,
        arrivals=arrivals,
        parameter_sources=deep_freeze(sources),
        metadata=deep_freeze(metadata),
        source_path=resolved,
        _prep_index=prep_index,
        _anon_index=anon_index,
        _local_index=local_index,
        _edge_index=edge_index,
        _wireless_index=wireless_index,
        _thermal_index=thermal_index,
    )


__all__ = [
    "AnonCertifiedBounds",
    "AnonAttempt",
    "AnonTraceRow",
    "DeviceContext",
    "EdgeCertifiedBounds",
    "EdgeFERTraceRow",
    "ExogenousEvent",
    "FERMeasurement",
    "LocalFERTraceRow",
    "PrepTraceRow",
    "ScenarioAnonRow",
    "ScenarioAttempt",
    "ScenarioBackgroundLoad",
    "ScenarioComputeStage",
    "ScenarioEdgeFERRow",
    "ScenarioEnvironment",
    "ScenarioFaultEvent",
    "ScenarioFutureTask",
    "ScenarioLibrary",
    "ScenarioLocalFERRow",
    "ScenarioPrepRow",
    "ScenarioRSUAnchor",
    "ScenarioTelemetryEvent",
    "ScenarioTaskAnchor",
    "ScenarioThermalSegment",
    "ScenarioTransferAnchor",
    "ScenarioVehicleAnchor",
    "ScenarioVersionEvent",
    "ScenarioWirelessSegment",
    "SupportResult",
    "TaskArrival",
    "ThermalSegment",
    "TraceBundle",
    "WirelessSegment",
    "load_trace",
]
