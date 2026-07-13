"""Observable-state isolation, hard action masks and deterministic repair.

This module is deliberately independent from every concrete controller.  A
policy may rank actions, but it cannot manufacture an action outside the
result returned by :class:`HardMaskEngine`.  Likewise, execution-time repair
recomputes the mask from current physical state instead of trusting a stale
policy observation.

The optional ``trace_support`` object is intentionally duck typed.  A real
trace store can expose any of the following read-only methods::

    has_anon_support(pipeline_id=..., quality_bins=..., device_type=...,
                     device_context=..., profile_hash=...)
    has_local_support(model_id=..., quality_bins=..., device_type=...,
                      device_context=..., profile_hash=...)
    has_edge_support(rsu_id=..., model_id=..., pipeline_id=...,
                     artifact_token=..., evaluation_pair_supported=...,
                     quality_bins=..., profile_hash=...)
    action_bounds(action=..., observation=...)

The ``has_*`` methods return a bool, ``(bool, details)`` or an object with a
``supported`` attribute.  ``action_bounds`` returns conservative, finite SI
bounds.  Missing paired support is *unsupported*; this module never fills it
with a cross-pipeline average.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import total_ordering
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .config import (
    ControllerConfig,
    CostConfig,
    LongTermConfig,
    PrivacyConfig,
    SimulationConfig,
)
from .enums import (
    ActionKind,
    ActionStage,
    FailureReason,
    JobStatus,
    ReasonCode,
    TaskState,
)
from .packets import AlignedTensorHandle, EncodedAnon, RawImageHandle
from .profiles import PRIVACY_RISK_TYPES, FrozenProfileBundle, deep_freeze, thaw_json
from .state import SimulationState, TaskRecord


_EPS = 1e-12
_REASON_ORDER = {reason: index for index, reason in enumerate(ReasonCode)}


@dataclass(frozen=True, slots=True)
class OnlineDecisionConfigView:
    """Immutable allow-list of configuration fields visible to policies.

    File paths, environment seeds, vehicle/RSU inventories, parameter-source
    metadata, and other simulator-only configuration are deliberately absent.
    Nested records are copied so this view has no reference back to the full
    :class:`SimulationConfig` object graph.
    """

    max_snapshot_age_s: float
    metadata_bits: int
    uplink_pause_limit_s: float
    downlink_pause_limit_s: float
    controller: ControllerConfig
    privacy: PrivacyConfig
    cost: CostConfig
    long_term: LongTermConfig
    vehicle_power_budgets_w: Mapping[str, float]
    rsu_power_budgets_w: Mapping[str, float]
    vehicle_branch_parameters: Mapping[str, Any]
    rsu_branch_parameters: Mapping[str, Any]
    seeds: Mapping[str, int]

    @classmethod
    def from_simulation_config(
        cls, config: SimulationConfig
    ) -> "OnlineDecisionConfigView":
        controller = ControllerConfig(
            policy=config.controller.policy,
            horizon_events=config.controller.horizon_events,
            scenarios=config.controller.scenarios,
            lyapunov_v=config.controller.lyapunov_v,
            controller_overhead_s=config.controller.controller_overhead_s,
            controller_energy_j=config.controller.controller_energy_j,
            rollout_policy=config.controller.rollout_policy,
            physical_queue_weight=config.controller.physical_queue_weight,
            vehicle_resource_theta=MappingProxyType(
                dict(config.controller.vehicle_resource_theta)
            ),
            rsu_resource_theta=MappingProxyType(
                dict(config.controller.rsu_resource_theta)
            ),
        )
        privacy = PrivacyConfig(
            risk_threshold=config.privacy.risk_threshold,
            confidence_error=config.privacy.confidence_error,
            quality_miscoverage=config.privacy.quality_miscoverage,
            min_subjects=config.privacy.min_subjects,
            min_emission_lcb=config.privacy.min_emission_lcb,
        )
        cost = CostConfig(
            latency_scale_s=config.cost.latency_scale_s,
            vehicle_energy_scale_j=config.cost.vehicle_energy_scale_j,
            rsu_energy_scale_j=config.cost.rsu_energy_scale_j,
            utility_scale=config.cost.utility_scale,
            failure_loss=config.cost.failure_loss,
            weights=MappingProxyType(dict(config.cost.weights)),
        )
        long_term = LongTermConfig(
            timeout_rate_limit=float(config.long_term.timeout_rate_limit),
            failure_rate_limit=float(config.long_term.failure_rate_limit),
            coverage_rate_minimum=float(config.long_term.coverage_rate_minimum),
        )
        return cls(
            max_snapshot_age_s=float(config.max_snapshot_age_s),
            metadata_bits=int(config.metadata_bits),
            uplink_pause_limit_s=float(config.uplink_pause_limit_s),
            downlink_pause_limit_s=float(config.downlink_pause_limit_s),
            controller=controller,
            privacy=privacy,
            cost=cost,
            long_term=long_term,
            vehicle_power_budgets_w=MappingProxyType(
                {
                    row.vehicle_id: float(row.average_power_budget_w)
                    for row in config.vehicles
                }
            ),
            rsu_power_budgets_w=MappingProxyType(
                {row.rsu_id: float(row.average_power_budget_w) for row in config.rsus}
            ),
            vehicle_branch_parameters=deep_freeze(
                {
                    row.vehicle_id: {
                        "device_type": row.device_type,
                        "battery_capacity_j": float(row.battery_capacity_j),
                        "initial_battery_j": float(row.initial_battery_j),
                        "memory_capacity_bytes": int(row.memory_capacity_bytes),
                        "idle_power_w": float(row.idle_power_w),
                        "hold_power_w": float(row.hold_power_w),
                        "descriptor_capacity": {
                            "accelerator": int(row.accelerator_descriptors),
                            "cpu": int(row.cpu_descriptors),
                            "encoder": int(row.encoder_descriptors),
                        },
                        "server_count": {
                            "accelerator": 1,
                            "cpu": 1,
                            "encoder": 1,
                        },
                    }
                    for row in config.vehicles
                }
            ),
            rsu_branch_parameters=deep_freeze(
                {
                    row.rsu_id: {
                        "idle_power_w": float(row.idle_power_w),
                        "hold_power_w": float(row.hold_power_w),
                    }
                    for row in config.rsus
                }
            ),
            seeds=MappingProxyType({"scenario": int(config.seeds["scenario"])}),
        )


def _stable_reasons(reasons: Iterable[ReasonCode]) -> tuple[ReasonCode, ...]:
    """Deduplicate reason codes in one version-stable order."""

    return tuple(
        sorted(set(reasons), key=lambda item: (_REASON_ORDER[item], item.value))
    )


def _finite_number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_key_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


@total_ordering
@dataclass(frozen=True, slots=True)
class Action:
    """Closed, hashable encoding of one RAW or READY decision.

    Invalid combinations cannot be instantiated.  ``sort_key`` and
    ``canonical_id`` are independent of set/dict iteration order and are used
    for every policy and repair tie break.
    """

    stage: ActionStage
    kind: ActionKind
    local_model_id: str | None = None
    pipeline_id: str | None = None
    rsu_id: str | None = None
    edge_model_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ActionStage) or not isinstance(
            self.kind, ActionKind
        ):
            raise TypeError("stage and kind must be ActionStage and ActionKind")
        values = {
            "local_model_id": self.local_model_id,
            "pipeline_id": self.pipeline_id,
            "rsu_id": self.rsu_id,
            "edge_model_id": self.edge_model_id,
        }
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in values.values()
        ):
            raise ValueError("action identifiers must be non-empty strings")

        expected: set[str]
        if self.kind is ActionKind.FAIL:
            expected = set()
        elif self.kind is ActionKind.LOCAL:
            expected = {"local_model_id"}
        elif self.kind is ActionKind.PIPE:
            if self.stage is not ActionStage.RAW:
                raise ValueError("PIPE is legal only at RAW")
            expected = {"pipeline_id"}
        elif self.kind is ActionKind.EDGE:
            if self.stage is not ActionStage.READY:
                raise ValueError("EDGE is legal only at READY")
            expected = {"rsu_id", "edge_model_id"}
        else:  # pragma: no cover - closed enum guard
            raise ValueError(f"unknown action kind: {self.kind!r}")
        if self.stage is ActionStage.RAW and self.kind is ActionKind.EDGE:
            raise ValueError("EDGE is not a RAW action")
        if self.stage is ActionStage.READY and self.kind is ActionKind.PIPE:
            raise ValueError("PIPE is not a READY action")
        present = {name for name, value in values.items() if value is not None}
        if present != expected:
            raise ValueError(
                f"{self.stage.value}/{self.kind.value} requires exactly {sorted(expected)}, got {sorted(present)}"
            )

    @classmethod
    def fail(cls, stage: ActionStage) -> "Action":
        return cls(stage, ActionKind.FAIL)

    @classmethod
    def local(cls, stage: ActionStage, model_id: str) -> "Action":
        return cls(stage, ActionKind.LOCAL, local_model_id=model_id)

    @classmethod
    def pipeline(cls, pipeline_id: str) -> "Action":
        return cls(ActionStage.RAW, ActionKind.PIPE, pipeline_id=pipeline_id)

    @classmethod
    def edge(cls, rsu_id: str, model_id: str) -> "Action":
        return cls(
            ActionStage.READY, ActionKind.EDGE, rsu_id=rsu_id, edge_model_id=model_id
        )

    @property
    def sort_key(self) -> tuple[str, ...]:
        return (
            self.stage.value,
            self.kind.value,
            self.local_model_id or "",
            self.pipeline_id or "",
            self.rsu_id or "",
            self.edge_model_id or "",
        )

    @property
    def canonical_id(self) -> str:
        return "|".join(self.sort_key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.sort_key < other.sort_key

    def to_dict(self) -> dict[str, str]:
        result = {"stage": self.stage.value, "kind": self.kind.value}
        for name in ("local_model_id", "pipeline_id", "rsu_id", "edge_model_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


_FORBIDDEN_KEY_FRAGMENTS = (
    "artifact_key",
    "raw_handle",
    "raw_image",
    "raw_crop",
    "aligned_handle",
    "aligned_tensor",
    "true_identity",
    "true_expression",
    "true_label",
    "expression_label",
    "true_quality",
    "attack_outcome",
    "attack_truth",
    "realized_attack",
    "realized_fer",
    "fer_label",
    "future_trace",
    "future_cursor",
    "trace_cursor",
    "hidden_cursor",
    "simulator_only",
)


def _assert_observation_safe(value: Any, path: str = "observation") -> None:
    """Reject accidental policy leakage instead of silently redacting it."""

    if isinstance(
        value,
        (
            RawImageHandle,
            AlignedTensorHandle,
            EncodedAnon,
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise ValueError(f"{path} contains a vehicle-local or payload object")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(
                    f"{path}.{key} is simulator-only or future information"
                )
            _assert_observation_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _assert_observation_safe(item, f"{path}[{index}]")
    elif is_dataclass(value):
        for item_field in fields(value):
            normalized = item_field.name.lower().replace("-", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(
                    f"{path}.{item_field.name} is simulator-only or future information"
                )
            _assert_observation_safe(
                getattr(value, item_field.name),
                f"{path}.{item_field.name}",
            )
    elif not isinstance(value, (str, int, float, bool, type(None), Enum)):
        raise ValueError(f"{path} contains a non-JSON observation object")


def _snapshot_task_tokens(state: SimulationState, vehicle_id: str) -> dict[str, str]:
    """Create identity-free task tokens scoped to one observation snapshot."""

    active = sorted(
        (
            task
            for task in state.tasks.values()
            if task.vehicle_id == vehicle_id and not task.terminal
        ),
        key=lambda task: (
            task.absolute_deadline_s,
            task.arrival_time_s,
            task.task_id,
        ),
    )
    return {
        task.task_id: f"observation-active-task:{index:06d}"
        for index, task in enumerate(active)
    }


class _ArtifactPairingTokenRegistry:
    """Session-local, non-enumerating artifact support capabilities.

    The registry deliberately never receives or stores an evaluation artifact
    key.  The trusted simulator converts that key into the finite set of
    ``(rsu, model, pipeline)`` tuples for which a paired measurement exists and
    passes only that sanitized set here.  Tokens are scoped to one task, so
    copying a token to another task cannot grant support.  A controller that
    can reach this registry may at most inspect sanitized action tuples; it
    cannot recover a subject-bearing key or enumerate future trace artifacts.
    """

    __slots__ = ("_counter", "_issued", "_resolved")

    def __init__(self) -> None:
        self._counter = 0
        self._issued: dict[tuple[str, tuple[tuple[str, str, str], ...]], str] = {}
        self._resolved: dict[tuple[str, str], frozenset[tuple[str, str, str]]] = {}

    def issue(
        self,
        task_token: str,
        capabilities: Iterable[tuple[str, str, str]],
    ) -> str:
        scope = str(task_token)
        sanitized = tuple(
            sorted(
                {
                    (str(rsu_id), str(model_id), str(pipeline_id))
                    for rsu_id, model_id, pipeline_id in capabilities
                }
            )
        )
        cache_key = (scope, sanitized)
        existing = self._issued.get(cache_key)
        if existing is not None:
            return existing
        self._counter += 1
        token = f"artifact-pairing-token:{self._counter:016x}"
        self._issued[cache_key] = token
        self._resolved[(scope, token)] = frozenset(sanitized)
        return token

    def allows(
        self,
        task_token: str,
        token: str | None,
        *,
        rsu_id: str | None,
        model_id: str | None,
        pipeline_id: str | None,
    ) -> bool:
        if token is None or not rsu_id or not model_id or not pipeline_id:
            return False
        capabilities = self._resolved.get((str(task_token), str(token)))
        return (
            capabilities is not None
            and (
                str(rsu_id),
                str(model_id),
                str(pipeline_id),
            )
            in capabilities
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable online information state with an explicit safe field set."""

    time_s: float
    task_id: str
    vehicle_id: str
    stage: ActionStage
    task_state: TaskState
    arrival_time_s: float
    absolute_deadline_s: float
    slack_s: float
    quality_features: tuple[float, ...]
    quality_probabilities: tuple[tuple[str, float], ...]
    conformal_quality_bins: tuple[str, ...]
    ood: bool
    device_type: str
    device_context: str
    selected_pipeline: str | None
    selected_local_model: str | None
    selected_rsu: str | None
    selected_edge_model: str | None
    attempt_started_count: int
    max_attempts: int
    artifact_token: str | None
    encoded_size_bytes: int | None
    encoded_evidence: Mapping[str, Any]
    remaining_bits: Mapping[str, float]
    vehicle: Mapping[str, Any]
    rsus: Mapping[str, Any]
    links: Mapping[str, Any]
    virtual_queues: Mapping[str, Any]
    versions: Mapping[str, Any]
    support: Mapping[str, Any]
    estimates: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or not math.isfinite(self.slack_s):
            raise ValueError("observation times must be finite")
        _assert_observation_safe(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-like view; protected task fields never enter it."""

        return {
            "time_s": self.time_s,
            "task_id": self.task_id,
            "vehicle_id": self.vehicle_id,
            "stage": self.stage.value,
            "task_state": self.task_state.value,
            "arrival_time_s": self.arrival_time_s,
            "absolute_deadline_s": self.absolute_deadline_s,
            "slack_s": self.slack_s,
            "quality_features": list(self.quality_features),
            "quality_probabilities": [
                list(item) for item in self.quality_probabilities
            ],
            "conformal_quality_bins": list(self.conformal_quality_bins),
            "ood": self.ood,
            "device_type": self.device_type,
            "device_context": self.device_context,
            "selected_pipeline": self.selected_pipeline,
            "selected_local_model": self.selected_local_model,
            "selected_rsu": self.selected_rsu,
            "selected_edge_model": self.selected_edge_model,
            "attempt_started_count": self.attempt_started_count,
            "max_attempts": self.max_attempts,
            "artifact_token": self.artifact_token,
            "encoded_size_bytes": self.encoded_size_bytes,
            "encoded_evidence": thaw_json(self.encoded_evidence),
            "remaining_bits": thaw_json(self.remaining_bits),
            "vehicle": thaw_json(self.vehicle),
            "rsus": thaw_json(self.rsus),
            "links": thaw_json(self.links),
            "virtual_queues": thaw_json(self.virtual_queues),
            "versions": thaw_json(self.versions),
            "support": thaw_json(self.support),
            "estimates": thaw_json(self.estimates),
            "metadata": thaw_json(self.metadata),
        }


class ObservationBuilder:
    """Build observations by positive allow-listing; never copy ``TaskRecord`` wholesale."""

    _CONTEXT_KEYS = frozenset(
        {
            "links",
            "rsu_snapshots",
            "versions",
            "support",
            "trace_support",
            "estimates",
            "action_estimates",
            "metadata",
            "max_snapshot_age_s",
        }
    )

    @staticmethod
    def _stage(task: TaskRecord, forced: ActionStage | None) -> ActionStage:
        if forced is not None:
            return forced
        if task.state is TaskState.RAW:
            return ActionStage.RAW
        if task.state is TaskState.READY:
            return ActionStage.READY
        raise ValueError(
            f"task {task.task_id} is not at a decision state: {task.state.value}"
        )

    @classmethod
    def build(
        cls,
        task: TaskRecord,
        state: SimulationState,
        *,
        profile: FrozenProfileBundle | None = None,
        stage: ActionStage | None = None,
        context: Mapping[str, Any] | None = None,
        links: Mapping[str, Any] | None = None,
        rsu_snapshots: Mapping[str, Any] | None = None,
        versions: Mapping[str, Any] | None = None,
        support: Mapping[str, Any] | None = None,
        estimates: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        pending_decisions: Mapping[str, tuple[Action, float]] | None = None,
        pairing_tokens: _ArtifactPairingTokenRegistry | None = None,
        artifact_capabilities: Mapping[str, Iterable[tuple[str, str, str]]]
        | None = None,
    ) -> Observation:
        if task.task_id not in state.tasks or state.tasks[task.task_id] is not task:
            raise ValueError(
                "observation task must be the canonical task in SimulationState"
            )
        if task.vehicle_id not in state.vehicles:
            raise ValueError("task vehicle is missing from SimulationState")
        ctx = dict(context or {})
        unknown = sorted(set(ctx) - cls._CONTEXT_KEYS)
        if unknown:
            raise ValueError(f"unknown observation context keys: {unknown}")
        _assert_observation_safe(ctx, "context")

        link_rows = _string_key_mapping(
            links if links is not None else ctx.get("links", {})
        )
        snapshot_overrides = _string_key_mapping(
            rsu_snapshots if rsu_snapshots is not None else ctx.get("rsu_snapshots", {})
        )
        version_rows = _string_key_mapping(
            versions if versions is not None else ctx.get("versions", {})
        )
        support_rows = _string_key_mapping(
            support
            if support is not None
            else ctx.get("support", ctx.get("trace_support", {}))
        )
        estimate_rows = _string_key_mapping(
            estimates
            if estimates is not None
            else ctx.get("estimates", ctx.get("action_estimates", {}))
        )
        metadata_rows = _string_key_mapping(
            metadata if metadata is not None else ctx.get("metadata", {})
        )
        pending_rows = dict(pending_decisions or {})
        capability_rows = dict(artifact_capabilities or {})
        if "max_snapshot_age_s" in ctx and "max_snapshot_age_s" not in metadata_rows:
            metadata_rows["max_snapshot_age_s"] = ctx["max_snapshot_age_s"]

        vehicle = state.vehicles[task.vehicle_id]
        snapshot_tokens = _snapshot_task_tokens(state, task.vehicle_id)
        active_artifact_tokens = {
            item.task_id: (
                None
                if item.artifact_key is None
                else pairing_tokens.issue(
                    snapshot_tokens[item.task_id],
                    capability_rows.get(item.task_id, ()),
                )
                if pairing_tokens is not None
                else f"{snapshot_tokens[item.task_id]}:artifact"
            )
            for item in state.tasks.values()
            if item.vehicle_id == task.vehicle_id and not item.terminal
        }
        focal_artifact_token = (
            None
            if task.artifact_key is None
            else pairing_tokens.issue(
                task.task_id,
                capability_rows.get(task.task_id, ()),
            )
            if pairing_tokens is not None
            else f"{snapshot_tokens[task.task_id]}:artifact"
        )
        resource_rows: dict[str, Any] = {}
        for name, pool in sorted(vehicle.resources.items()):
            active_jobs = tuple(
                {
                    # Snapshot-local queue identity only: no simulator job or
                    # task identifier crosses the observation boundary.
                    "queue_rank": rank,
                    "task_token": snapshot_tokens.get(job.task_id),
                    "status": job.status.value,
                    "operation": job.operation.value,
                    "server_index": job.server_index,
                    "residual_work_s": job.residual_work_s,
                    "remaining_nominal_dynamic_energy_j": (
                        job.total_dynamic_energy_j
                        * job.residual_work_s
                        / job.total_work_s
                    ),
                    "deadline_offset_s": max(
                        0.0, job.absolute_deadline_s - state.clock_s
                    ),
                    "enqueue_seq": job.enqueue_seq,
                }
                for rank, job in enumerate(
                    sorted(
                        (
                            job
                            for job in pool.jobs.values()
                            if job.status in {JobStatus.WAITING, JobStatus.RUNNING}
                        ),
                        key=lambda job: (
                            0 if job.status is JobStatus.RUNNING else 1,
                            -1 if job.server_index is None else job.server_index,
                            job.absolute_deadline_s,
                            job.enqueue_seq,
                            job.job_id,
                        ),
                    )
                )
            )
            remaining_dynamic_energy_j = sum(
                row["remaining_nominal_dynamic_energy_j"] for row in active_jobs
            )
            resource_rows[name] = {
                "resource_id": pool.resource_id,
                "kind": pool.kind.value,
                "server_count": pool.server_count,
                "waiting_count": pool.waiting_count,
                "running_count": pool.running_count,
                "residual_work_s": pool.residual_work_s,
                "remaining_dynamic_energy_j": remaining_dynamic_energy_j,
                "active_jobs": active_jobs,
            }
        descriptor_remaining = {
            name: max(0, capacity - vehicle.descriptors_reserved.get(name, 0))
            for name, capacity in sorted(vehicle.descriptor_capacity.items())
        }

        def edge_runtime_row(item: TaskRecord) -> Mapping[str, Any]:
            """Expose only task-owned RSU continuation state.

            The opaque snapshot task token remains the sole association key;
            simulator job/admission identifiers and other tasks' reservations
            are never copied into the observation.
            """

            rsu_id = item.selected_rsu
            if not rsu_id or rsu_id not in state.rsus:
                return {}
            runtime = state.rsus[rsu_id]
            reservation = next(
                (
                    row
                    for row in runtime.admission.snapshot().reservations
                    if row[0] == item.task_id
                ),
                None,
            )
            jobs = tuple(
                job
                for pool in (runtime.ingress, runtime.gpu)
                for job in pool.active_jobs_for_task(item.task_id)
            )
            if len(jobs) > 1:
                raise ValueError("one task cannot own multiple active RSU jobs")
            result: dict[str, Any] = {}
            if reservation is not None:
                _, descriptors, vram, work, model_id, model_hash = reservation
                result["reservation"] = {
                    "descriptor_count": descriptors,
                    "vram_bytes": vram,
                    "conservative_work_gpu_s": work,
                    "model_id": model_id,
                    "model_hash": model_hash,
                }
            if jobs:
                job = jobs[0]
                result["job"] = {
                    "status": job.status.value,
                    "operation": job.operation.value,
                    "resource": job.resource_kind.value,
                    "server_index": job.server_index,
                    "residual_work_s": job.residual_work_s,
                    "remaining_nominal_dynamic_energy_j": (
                        job.total_dynamic_energy_j
                        * job.residual_work_s
                        / job.total_work_s
                    ),
                    "deadline_offset_s": max(
                        0.0, job.absolute_deadline_s - state.clock_s
                    ),
                    "enqueue_seq": job.enqueue_seq,
                }
            return result

        def pending_decision_row(item: TaskRecord) -> Mapping[str, Any]:
            pending = pending_rows.get(item.task_id)
            if pending is None:
                return {}
            action, due_s = pending
            if not isinstance(action, Action) or not math.isfinite(due_s):
                raise ValueError("pending decision snapshot is invalid")
            return {
                "proposed": action.to_dict(),
                "remaining_overhead_s": max(0.0, due_s - state.clock_s),
                "controller_energy_already_charged": True,
            }

        active_task_rows = tuple(
            {
                "task_token": snapshot_tokens[item.task_id],
                "is_focal": item.task_id == task.task_id,
                "state": item.state.value,
                "arrival_age_s": max(0.0, state.clock_s - item.arrival_time_s),
                "deadline_offset_s": max(0.0, item.absolute_deadline_s - state.clock_s),
                "quality_features": item.quality_features,
                "quality_probabilities": item.quality_probabilities,
                "conformal_quality_bins": item.conformal_quality_bins,
                "ood": item.ood,
                "device_context": item.device_context,
                "selected_pipeline": item.selected_pipeline,
                "selected_local_model": item.selected_local_model,
                "selected_rsu": item.selected_rsu,
                "selected_edge_model": item.selected_edge_model,
                "attempt_started_count": item.attempt_started_count,
                "max_attempts": item.max_attempts,
                # The artifact itself and its evaluation-trace key remain
                # hidden.  This token only links causal rows in this one
                # snapshot.
                "artifact_token": (active_artifact_tokens[item.task_id]),
                "encoded_size_bytes": item.encoded_size_bytes,
                "reservation_tokens": dict(sorted(item.reservation_tokens.items())),
                "memory_reservation_bytes": item.memory_reservation_bytes,
                "rsu_continuation": edge_runtime_row(item),
                "pending_decision": pending_decision_row(item),
            }
            for item in sorted(
                (
                    item
                    for item in state.tasks.values()
                    if item.vehicle_id == task.vehicle_id and not item.terminal
                ),
                key=lambda item: (
                    item.absolute_deadline_s,
                    item.arrival_time_s,
                    item.task_id,
                ),
            )
        )
        vehicle_row = {
            "battery_j": vehicle.battery_j,
            "battery_capacity_j": vehicle.battery_capacity_j,
            "memory_capacity_bytes": vehicle.memory_capacity_bytes,
            "memory_reserved_bytes": vehicle.memory_reserved_bytes,
            "memory_remaining_bytes": max(
                0, vehicle.memory_capacity_bytes - vehicle.memory_reserved_bytes
            ),
            "descriptor_capacity": dict(sorted(vehicle.descriptor_capacity.items())),
            "descriptors_reserved": dict(sorted(vehicle.descriptors_reserved.items())),
            "descriptor_remaining": descriptor_remaining,
            "resources": resource_rows,
            "task_energy_j": task.vehicle_energy_j,
            "active_task_count": sum(
                item.vehicle_id == task.vehicle_id and not item.terminal
                for item in state.tasks.values()
            ),
            "active_tasks": active_task_rows,
        }

        rsu_rows: dict[str, Any] = {}
        for rsu_id, runtime in sorted(state.rsus.items()):
            if runtime.public_snapshot is None:
                # Missing telemetry is conservative unavailable, never a live
                # read disguised with an old timestamp.  Production simulator
                # states install and periodically refresh explicit snapshots.
                telemetry = {
                    "failed": True,
                    "descriptors": 0,
                    "descriptor_capacity": runtime.admission.descriptor_capacity,
                    "vram_bytes": 0,
                    "vram_capacity_bytes": runtime.admission.vram_capacity_bytes,
                    "reserved_work_gpu_s": 0.0,
                    "workload_capacity_gpu_s": runtime.admission.workload_capacity_gpu_s,
                    "cached_models": {},
                    "ingress_waiting": 0,
                    "ingress_running": 0,
                    "ingress_residual_work_s": 0.0,
                    "gpu_waiting": 0,
                    "gpu_running": 0,
                    "gpu_servers": runtime.gpu.server_count,
                    "gpu_residual_work_s": 0.0,
                }
            else:
                telemetry = dict(runtime.public_snapshot)
            base = {
                "snapshot_time_s": runtime.current_snapshot_time_s,
                "snapshot_age_s": max(
                    0.0, state.clock_s - runtime.current_snapshot_time_s
                ),
                **telemetry,
            }
            override = snapshot_overrides.get(rsu_id)
            if isinstance(override, Mapping):
                base.update({str(key): value for key, value in override.items()})
            rsu_rows[rsu_id] = base

        if profile is not None:
            version_rows.setdefault("protocol_version", profile.protocol_version)
            version_rows.setdefault("profile_hash", profile.profile_hash)
            version_rows.setdefault("profile_version", profile.profile_version)
            version_rows.setdefault(
                "pipelines",
                {
                    pipeline_id: {
                        "pipeline_hash": item.pipeline_hash,
                        "guard_hash": item.guard_hash,
                        "encoder_hash": item.encoder_hash,
                        "protocol_version": item.protocol_version,
                    }
                    for pipeline_id, item in sorted(profile.pipelines.items())
                },
            )
            version_rows.setdefault(
                "local_models",
                {
                    model_id: {
                        "model_hash": item.model_hash,
                        "protocol_version": item.protocol_version,
                    }
                    for model_id, item in sorted(profile.local_models.items())
                },
            )
            version_rows.setdefault(
                "edge_models",
                {
                    model_id: {
                        "model_hash": item.model_hash,
                        "protocol_version": item.protocol_version,
                    }
                    for model_id, item in sorted(profile.edge_models.items())
                },
            )

        encoded: dict[str, Any] = {}
        if task.encoded_anon is not None:
            encoded = {
                "message_source_type": type(task.encoded_anon).__name__,
                "artifact_token": focal_artifact_token,
                "pipeline_id": task.encoded_anon.pipeline_id,
                "pipeline_hash": task.encoded_anon.pipeline_hash,
                "guard_hash": task.encoded_anon.guard_hash,
                "encoder_hash": task.encoded_anon.encoder_hash,
                "profile_hash": task.encoded_anon.profile_hash,
                "quality_bins": tuple(task.encoded_anon.quality_bins),
                "size_bytes": task.encoded_anon.size_bytes,
            }

        queues = state.virtual_queues
        vq_row = {
            "vehicle_power": dict(sorted(queues.vehicle_power.items())),
            "rsu_power": dict(sorted(queues.rsu_power.items())),
            "timeout": queues.timeout,
            "failure": queues.failure,
            "coverage": queues.coverage,
        }
        result = Observation(
            time_s=state.clock_s,
            task_id=task.task_id,
            vehicle_id=task.vehicle_id,
            stage=cls._stage(task, stage),
            task_state=task.state,
            arrival_time_s=task.arrival_time_s,
            absolute_deadline_s=task.absolute_deadline_s,
            slack_s=task.absolute_deadline_s - state.clock_s,
            quality_features=tuple(float(item) for item in task.quality_features),
            quality_probabilities=tuple(
                (str(name), float(prob)) for name, prob in task.quality_probabilities
            ),
            conformal_quality_bins=tuple(
                str(item) for item in task.conformal_quality_bins
            ),
            ood=bool(task.ood),
            device_type=vehicle.device_type,
            device_context=task.device_context,
            selected_pipeline=task.selected_pipeline,
            selected_local_model=task.selected_local_model,
            selected_rsu=task.selected_rsu,
            selected_edge_model=task.selected_edge_model,
            attempt_started_count=task.attempt_started_count,
            max_attempts=task.max_attempts,
            artifact_token=focal_artifact_token,
            encoded_size_bytes=None
            if task.encoded_anon is None
            else task.encoded_anon.size_bytes,
            encoded_evidence=deep_freeze(encoded),
            remaining_bits=deep_freeze(
                {"uplink": task.ul_remaining_bits, "downlink": task.dl_remaining_bits}
            ),
            vehicle=deep_freeze(vehicle_row),
            rsus=deep_freeze(rsu_rows),
            links=deep_freeze(link_rows),
            virtual_queues=deep_freeze(vq_row),
            versions=deep_freeze(version_rows),
            support=deep_freeze(support_rows),
            estimates=deep_freeze(estimate_rows),
            metadata=deep_freeze(metadata_rows),
        )
        return result


@dataclass(frozen=True, slots=True)
class RemovalRecord:
    action: Action
    reasons: tuple[ReasonCode, ...]
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class MaskResult:
    stage: ActionStage
    candidates: tuple[Action, ...]
    allowed: tuple[Action, ...]
    removed: Mapping[Action, tuple[ReasonCode, ...]]
    records: Mapping[Action, RemovalRecord]

    def __post_init__(self) -> None:
        if (
            tuple(sorted(self.candidates)) != self.candidates
            or tuple(sorted(self.allowed)) != self.allowed
        ):
            raise ValueError("mask actions must use stable lexical order")
        if set(self.allowed) | set(self.removed) != set(self.candidates):
            raise ValueError(
                "every mask candidate must be allowed or have removal reasons"
            )
        if set(self.allowed) & set(self.removed):
            raise ValueError("an action cannot be both allowed and removed")
        if any(not reasons for reasons in self.removed.values()):
            raise ValueError("removed actions require at least one reason code")

    def reasons_for(self, action: Action) -> tuple[ReasonCode, ...]:
        return self.removed.get(action, ())

    def is_allowed(self, action: Action) -> bool:
        return action in self.allowed

    def audit_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for action in self.candidates:
            record = self.records.get(action)
            rows.append(
                {
                    "action": action.to_dict(),
                    "action_id": action.canonical_id,
                    "allowed": action in self.allowed,
                    "reason_codes": [
                        item.value for item in self.removed.get(action, ())
                    ],
                    "details": {} if record is None else thaw_json(record.details),
                }
            )
        return tuple(rows)


def action_estimate(
    action: Action,
    observation: Observation,
    provider: Any = None,
) -> Mapping[str, Any]:
    """Read one immutable estimate/bound row without drawing random samples."""

    merged: dict[str, Any] = {}
    if provider is not None:
        intrinsic = _trace_bundle_action_bounds(provider, action, observation)
        merged.update(intrinsic)
        method = getattr(provider, "action_bounds", None)
        if callable(method):
            row = method(action=action, observation=observation)
            if isinstance(row, Mapping):
                merged.update({str(key): value for key, value in row.items()})
    estimates = observation.estimates
    direct = estimates.get(action.canonical_id)
    if isinstance(direct, Mapping):
        merged.update({str(key): value for key, value in direct.items()})
    kind_rows = estimates.get(action.kind.value) or estimates.get(
        action.kind.value.lower()
    )
    if isinstance(kind_rows, Mapping):
        key_candidates = (
            action.local_model_id,
            action.pipeline_id,
            "|".join(filter(None, (action.rsu_id, action.edge_model_id))),
            action.edge_model_id,
            action.rsu_id,
            "default",
        )
        for key in key_candidates:
            if key is not None and isinstance(kind_rows.get(key), Mapping):
                merged.update(
                    {str(name): value for name, value in kind_rows[key].items()}
                )
                break
    return MappingProxyType(merged)


def _trace_context_matches(row: Any, observation: Observation) -> bool:
    return _trace_context_value_matches(row, observation.device_context)


def _trace_context_value_matches(row: Any, requested: Any) -> bool:
    context = getattr(row, "context", None)
    if context is None or requested is None:
        return True
    requested_value = str(requested)
    values = tuple(
        str(getattr(context, name, ""))
        for name in ("thermal_state", "power_mode", "memory_pressure")
    )
    return requested_value in set(values) | {"|".join(values)} or (
        requested_value == "nominal"
        and all(value in {"nominal", "normal"} for value in values)
    )


def _trace_bundle_rows(
    provider: Any,
    action: Action,
    observation: Observation,
    pairing_key: str | None = None,
) -> list[Any]:
    """Exact immutable-row support adapter for the bundled ``TraceBundle``.

    This is an index/read operation only.  It never advances a trace cursor or
    samples a component, and it keeps anonymization attempts joined in their
    parent row.
    """

    bins = set(observation.conformal_quality_bins)
    if action.kind is ActionKind.PIPE and hasattr(provider, "anon_rows"):
        return [
            row
            for row in provider.anon_rows
            if row.pipeline_id == action.pipeline_id
            and row.quality_bin in bins
            and row.device_type == observation.device_type
            and _trace_context_matches(row, observation)
        ]
    if action.kind is ActionKind.LOCAL and hasattr(provider, "local_rows"):
        return [
            row
            for row in provider.local_rows
            if row.model_id == action.local_model_id
            and row.quality_bin in bins
            and row.device_type == observation.device_type
            and _trace_context_matches(row, observation)
        ]
    if action.kind is ActionKind.EDGE and hasattr(provider, "edge_rows"):
        rsu = _as_mapping(observation.rsus.get(action.rsu_id or ""))
        rsu_context = rsu.get("device_context")
        return [
            row
            for row in provider.edge_rows
            if row.rsu_id == action.rsu_id
            and row.model_id == action.edge_model_id
            and row.pipeline_id == observation.selected_pipeline
            # Identity-free scenario rows carry a library-local artifact
            # token.  Evaluation artifact pairing is checked separately by
            # ``has_edge_support``; it must not be used to select a training
            # scenario outcome.  A raw TraceBundle retains its legacy exact
            # artifact behavior for non-controller fixture adapters.
            and (
                hasattr(row, "artifact_token")
                or getattr(row, "artifact_key", None) == pairing_key
            )
            and _trace_context_value_matches(row, rsu_context)
        ]
    return []


def _trace_bundle_action_bounds(
    provider: Any, action: Action, observation: Observation
) -> dict[str, Any]:
    rows = _trace_bundle_rows(provider, action, observation)
    if not rows:
        return {}
    result: dict[str, Any] = {}
    if action.kind is ActionKind.LOCAL:
        work = [float(row.service_work_s) for row in rows]
        energy = [float(row.dynamic_energy_j) for row in rows]
        result.update(
            {
                "duration_lower_bound_s": min(work),
                "expected_duration_s": sum(work) / len(work),
                "vehicle_work_s": sum(work) / len(work),
                "vehicle_energy_upper_j": max(energy),
                "expected_vehicle_energy_j": sum(energy) / len(energy),
                "vehicle_memory_upper_bytes": max(
                    int(row.memory_bytes) for row in rows
                ),
                "descriptor_tokens": {"accelerator": 1},
                "failure_probability": sum(bool(row.failed) for row in rows)
                / len(rows),
                "completion_probability": sum(not bool(row.failed) for row in rows)
                / len(rows),
            }
        )
        valid_losses = [
            float(row.fer_loss)
            for row in rows
            if not row.failed and row.fer_loss is not None
        ]
        if valid_losses:
            result["expected_fer_loss"] = sum(valid_losses) / len(valid_losses)
    elif action.kind is ActionKind.PIPE:
        work = [float(row.total_work_s) for row in rows]
        energy = [float(row.total_energy_j) for row in rows]
        attempts = [attempt for row in rows for attempt in row.attempts]
        result.update(
            {
                "duration_lower_bound_s": min(work),
                "expected_duration_s": sum(work) / len(work),
                "vehicle_work_s": {
                    "accelerator": sum(
                        sum(float(attempt.anon_work_s) for attempt in row.attempts)
                        for row in rows
                    )
                    / len(rows),
                    "cpu": sum(
                        sum(
                            float(attempt.guard_work_s or 0.0)
                            for attempt in row.attempts
                        )
                        for row in rows
                    )
                    / len(rows),
                    "encoder": sum(
                        sum(
                            float(attempt.encode_work_s or 0.0)
                            for attempt in row.attempts
                        )
                        for row in rows
                    )
                    / len(rows),
                },
                "vehicle_energy_upper_j": max(energy),
                "expected_vehicle_energy_j": sum(energy) / len(energy),
                "vehicle_memory_upper_bytes": max(
                    int(attempt.peak_memory_bytes) for attempt in attempts
                ),
                "descriptor_tokens": {"accelerator": 1, "cpu": 1, "encoder": 1},
                "failure_probability": sum(not bool(row.formed_packet) for row in rows)
                / len(rows),
                "completion_probability": 0.0,
                "max_attempts": max(len(row.attempts) for row in rows),
                "max_output_bytes": max(
                    int(row.final_encoded_size_bytes) for row in rows
                ),
            }
        )
    elif action.kind is ActionKind.EDGE:
        ingress_failed = [bool(getattr(row, "ingress_failed", False)) for row in rows]
        failed = [
            ingress or bool(row.failed)
            for ingress, row in zip(ingress_failed, rows, strict=True)
        ]
        compute = [
            float(row.ingress_work_s + (0.0 if ingress else row.gpu_work_s))
            for ingress, row in zip(ingress_failed, rows, strict=True)
        ]
        energy = [
            float(row.ingress_energy_j + (0.0 if ingress else row.gpu_energy_j))
            for ingress, row in zip(ingress_failed, rows, strict=True)
        ]
        result.update(
            {
                "duration_lower_bound_s": min(compute),
                "expected_duration_s": sum(compute) / len(compute),
                "expected_rsu_energy_j": sum(energy) / len(energy),
                "rsu_ingress_work_s": sum(float(row.ingress_work_s) for row in rows)
                / len(rows),
                "rsu_gpu_work_s": sum(
                    0.0 if ingress else float(row.gpu_work_s)
                    for ingress, row in zip(ingress_failed, rows, strict=True)
                )
                / len(rows),
                "rsu_descriptor_count": 1,
                "rsu_vram_upper_bytes": max(int(row.vram_bytes) for row in rows),
                "rsu_work_upper_gpu_s": max(float(row.gpu_work_s) for row in rows),
                "failure_probability": sum(failed) / len(rows),
                "completion_probability": sum(not value for value in failed)
                / len(rows),
            }
        )
        valid_losses = [
            float(row.fer_loss)
            for row in rows
            if not getattr(row, "ingress_failed", False)
            and not row.failed
            and row.fer_loss is not None
        ]
        if valid_losses:
            result["expected_fer_loss"] = sum(valid_losses) / len(valid_losses)
    return result


def _support_result(result: Any) -> bool | None:
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        return result[0]
    supported = getattr(result, "supported", None)
    if isinstance(supported, bool):
        return supported
    if isinstance(result, Mapping) and isinstance(result.get("supported"), bool):
        return bool(result["supported"])
    return None


class HardMaskEngine:
    """Pure conservative action enumeration shared by every controller."""

    def __init__(
        self,
        profile: FrozenProfileBundle,
        config: SimulationConfig | None = None,
        *,
        trace_support: Any = None,
    ) -> None:
        self.profile = profile
        self.config = (
            None
            if config is None
            else OnlineDecisionConfigView.from_simulation_config(config)
        )
        self.trace_support = trace_support
        self._pairing_tokens = _ArtifactPairingTokenRegistry()

    def _evaluation_pair_supported(
        self, action: Action, observation: Observation
    ) -> bool:
        return self._pairing_tokens.allows(
            observation.task_id,
            observation.artifact_token,
            rsu_id=action.rsu_id,
            model_id=action.edge_model_id,
            pipeline_id=observation.selected_pipeline,
        )

    def _candidates(self, observation: Observation) -> tuple[Action, ...]:
        actions: list[Action] = [Action.fail(observation.stage)]
        actions.extend(
            Action.local(observation.stage, model_id)
            for model_id in self.profile.local_models
        )
        if observation.stage is ActionStage.RAW:
            actions.extend(
                Action.pipeline(pipeline_id) for pipeline_id in self.profile.pipelines
            )
        else:
            rsu_ids = set(observation.rsus)
            for model in self.profile.edge_models.values():
                rsu_ids.update(model.supported_rsus)
            actions.extend(
                Action.edge(rsu_id, model_id)
                for rsu_id in rsu_ids
                for model_id in self.profile.edge_models
            )
        return tuple(sorted(set(actions)))

    def _call_support(
        self,
        kind: str,
        action: Action,
        observation: Observation,
        *,
        evaluation_pair_supported: bool = False,
    ) -> bool | None:
        provider = self.trace_support
        method_names = {
            "anon": ("has_anon_support", "supports_anon", "has_anon_transaction"),
            "local": ("has_local_support", "supports_local", "has_local_measurement"),
            "edge": ("has_edge_support", "supports_edge", "has_edge_measurement"),
        }[kind]
        kwargs: dict[str, Any] = {
            "quality_bins": observation.conformal_quality_bins,
            "device_type": observation.device_type,
            "device_context": observation.device_context,
            "profile_hash": self.profile.profile_hash,
        }
        if kind == "anon":
            kwargs["pipeline_id"] = action.pipeline_id
        elif kind == "local":
            kwargs["model_id"] = action.local_model_id
        else:
            rsu = _as_mapping(observation.rsus.get(action.rsu_id or ""))
            kwargs.update(
                {
                    "rsu_id": action.rsu_id,
                    "model_id": action.edge_model_id,
                    "pipeline_id": observation.selected_pipeline,
                    "artifact_token": observation.artifact_token,
                    "evaluation_pair_supported": evaluation_pair_supported,
                    "rsu_context": rsu.get("device_context"),
                }
            )
        if provider is not None:
            for name in method_names:
                method = getattr(provider, name, None)
                if callable(method):
                    try:
                        signature = inspect.signature(method)
                    except (TypeError, ValueError):
                        return _support_result(method(**kwargs))
                    try:
                        signature.bind(**kwargs)
                    except TypeError as keyword_error:
                        # Select a legacy fixture signature before invocation;
                        # a TypeError raised inside a provider must propagate.
                        identifier = (
                            action.pipeline_id
                            or action.local_model_id
                            or action.edge_model_id
                        )
                        try:
                            signature.bind(identifier)
                        except TypeError:
                            raise keyword_error
                        return _support_result(method(identifier))
                    return _support_result(method(**kwargs))
            rows = _trace_bundle_rows(
                provider,
                action,
                observation,
                pairing_key=observation.artifact_token,
            )
            if any(
                hasattr(provider, name)
                for name in ("anon_rows", "local_rows", "edge_rows")
            ):
                if kind in {"anon", "local"}:
                    covered_bins = {
                        str(getattr(row, "quality_bin", "")) for row in rows
                    }
                    return bool(rows) and set(
                        observation.conformal_quality_bins
                    ).issubset(covered_bins)
                return bool(rows)

        rows = observation.support
        section = rows.get(kind) or rows.get(f"{kind}_support")
        if isinstance(section, bool):
            return section
        if isinstance(section, Mapping):
            keys = [action.canonical_id]
            if kind == "anon":
                keys.extend([action.pipeline_id or ""])
            elif kind == "local":
                keys.extend([action.local_model_id or ""])
            else:
                keys.extend(
                    [
                        "|".join(
                            filter(
                                None,
                                (
                                    action.rsu_id,
                                    action.edge_model_id,
                                    observation.selected_pipeline,
                                    observation.artifact_token,
                                ),
                            )
                        ),
                        "|".join(filter(None, (action.rsu_id, action.edge_model_id))),
                    ]
                )
            for key in keys:
                if key in section:
                    return _support_result(section[key])
        return None

    def _version_reasons(
        self, action: Action, observation: Observation
    ) -> list[ReasonCode]:
        reasons: list[ReasonCode] = []
        versions = observation.versions
        protocol = versions.get("protocol_version")
        profile_hash = versions.get("profile_hash")
        if protocol is not None and protocol != self.profile.protocol_version:
            reasons.append(ReasonCode.PROTOCOL_MISMATCH)
        if profile_hash is not None and profile_hash != self.profile.profile_hash:
            reasons.append(ReasonCode.PROFILE_MISMATCH)

        if action.kind is ActionKind.LOCAL:
            model = self.profile.local_models.get(action.local_model_id or "")
            if model is None:
                return reasons + [ReasonCode.VERSION_MISMATCH]
            if observation.device_type not in model.supported_devices:
                reasons.append(ReasonCode.DEVICE_UNSUPPORTED)
            local_versions = _as_mapping(versions.get("local_models"))
            actual = local_versions.get(model.model_id)
            if isinstance(actual, Mapping):
                if actual.get("model_hash") != model.model_hash:
                    reasons.append(ReasonCode.VERSION_MISMATCH)
                if (
                    actual.get("protocol_version", model.protocol_version)
                    != model.protocol_version
                ):
                    reasons.append(ReasonCode.PROTOCOL_MISMATCH)
            elif isinstance(actual, str) and actual != model.model_hash:
                reasons.append(ReasonCode.VERSION_MISMATCH)
            if model.protocol_version != self.profile.protocol_version:
                reasons.append(ReasonCode.PROTOCOL_MISMATCH)

        if action.kind is ActionKind.PIPE:
            pipeline = self.profile.pipelines.get(action.pipeline_id or "")
            if pipeline is None:
                return reasons + [ReasonCode.VERSION_MISMATCH]
            if observation.device_type not in pipeline.supported_devices:
                reasons.append(ReasonCode.DEVICE_UNSUPPORTED)
            pipeline_versions = _as_mapping(versions.get("pipelines"))
            actual = pipeline_versions.get(pipeline.pipeline_id)
            if isinstance(actual, Mapping):
                expected = {
                    "pipeline_hash": pipeline.pipeline_hash,
                    "guard_hash": pipeline.guard_hash,
                    "encoder_hash": pipeline.encoder_hash,
                }
                if any(actual.get(key) != value for key, value in expected.items()):
                    reasons.append(ReasonCode.VERSION_MISMATCH)
                if (
                    actual.get("protocol_version", pipeline.protocol_version)
                    != pipeline.protocol_version
                ):
                    reasons.append(ReasonCode.PROTOCOL_MISMATCH)
            if pipeline.protocol_version != self.profile.protocol_version:
                reasons.append(ReasonCode.PROTOCOL_MISMATCH)

        if action.kind is ActionKind.EDGE:
            model = self.profile.edge_models.get(action.edge_model_id or "")
            pipeline = self.profile.pipelines.get(observation.selected_pipeline or "")
            if model is None or pipeline is None:
                return reasons + [ReasonCode.VERSION_MISMATCH]
            edge_versions = _as_mapping(versions.get("edge_models"))
            active = edge_versions.get(model.model_id)
            if isinstance(active, Mapping):
                if active.get("model_hash") != model.model_hash:
                    reasons.append(ReasonCode.VERSION_MISMATCH)
                if (
                    active.get("protocol_version", model.protocol_version)
                    != model.protocol_version
                ):
                    reasons.append(ReasonCode.PROTOCOL_MISMATCH)
            elif isinstance(active, str) and active != model.model_hash:
                reasons.append(ReasonCode.VERSION_MISMATCH)
            encoded = observation.encoded_evidence
            result = self.profile.validate_compatibility(
                protocol_version=str(protocol or self.profile.protocol_version),
                profile_hash=str(encoded.get("profile_hash", profile_hash or "")),
                pipeline_id=pipeline.pipeline_id,
                pipeline_hash=encoded.get("pipeline_hash"),
                guard_hash=encoded.get("guard_hash"),
                encoder_hash=encoded.get("encoder_hash"),
                edge_model_id=model.model_id,
                edge_model_hash=model.model_hash,
                device_type=observation.device_type,
                rsu_id=action.rsu_id,
            )
            reasons.extend(result.reasons)
        return reasons

    def _privacy_reasons(
        self, pipeline_id: str | None, observation: Observation, details: dict[str, Any]
    ) -> list[ReasonCode]:
        if not pipeline_id:
            return [ReasonCode.VERSION_MISMATCH]
        if observation.ood:
            return [ReasonCode.OOD]
        decision = self.profile.query_privacy(
            pipeline_id,
            observation.conformal_quality_bins,
            observation.device_type,
            **(
                {
                    "risk_threshold": self.config.privacy.risk_threshold,
                    "min_subjects": self.config.privacy.min_subjects,
                    "min_emission_lcb": self.config.privacy.min_emission_lcb,
                }
                if self.config is not None
                else {}
            ),
        )
        failed_risks = [
            risk
            for risk in PRIVACY_RISK_TYPES
            if decision.per_risk_ucb.get(risk, 1.0)
            > (
                self.config.privacy.risk_threshold
                if self.config is not None
                else self.profile.risk_threshold
            )
        ]
        details["privacy"] = {
            "safe": decision.safe,
            "candidate_quality_bins": tuple(decision.quality_bins),
            "per_risk_ucb": dict(decision.per_risk_ucb),
            "failed_risk_types": tuple(failed_risks),
            "worst_ucb": decision.worst_ucb,
            "min_subject_count": decision.min_subject_count,
            "min_emission_lcb": decision.min_emission_lcb,
        }
        return list(decision.reasons)

    def _bound_reasons(
        self,
        action: Action,
        observation: Observation,
        details: dict[str, Any],
    ) -> list[ReasonCode]:
        if action.kind is ActionKind.FAIL:
            return []
        row = action_estimate(action, observation, self.trace_support)
        bounds = dict(row)
        if self.config is not None and bounds:
            cost = self.config.cost
            weights = cost.weights
            duration = _finite_number(bounds.get("expected_duration_s"))
            vehicle_energy = _finite_number(bounds.get("expected_vehicle_energy_j"))
            rsu_energy = _finite_number(bounds.get("expected_rsu_energy_j"))
            utility = _finite_number(
                bounds.get("expected_fer_loss", bounds.get("expected_loss"))
            )
            failure = _finite_number(
                bounds.get("failure_probability", bounds.get("expected_failure"))
            )
            timeout = _finite_number(bounds.get("timeout_probability"))
            if None not in (duration, vehicle_energy, utility, failure):
                bounds["expected_cost"] = (
                    float(weights.get("latency", 1.0))
                    * float(duration)
                    / cost.latency_scale_s
                    + float(weights.get("vehicle_energy", 1.0))
                    * float(vehicle_energy)
                    / cost.vehicle_energy_scale_j
                    + float(weights.get("rsu_energy", 1.0))
                    * float(rsu_energy or 0.0)
                    / cost.rsu_energy_scale_j
                    + float(weights.get("utility", 1.0))
                    * float(utility)
                    / cost.utility_scale
                    + float(weights.get("failure", 1.0))
                    * cost.failure_loss
                    * float(failure)
                    + float(weights.get("timeout", 1.0))
                    * cost.failure_loss
                    * float(timeout or 0.0)
                )
        details["bounds"] = bounds
        if action.kind is ActionKind.EDGE:
            method = getattr(self.trace_support, "information_ablation_bounds", None)
            if callable(method):
                ablation = method(action=action, observation=observation)
                if isinstance(ablation, Mapping) and ablation:
                    details["information_ablation"] = dict(ablation)
        missing_reason = (
            ReasonCode.JOINT_TRACE_MISSING
            if action.kind is ActionKind.PIPE
            else ReasonCode.PAIRED_MEASUREMENT_MISSING
        )
        if not row:
            return [missing_reason]
        reasons: list[ReasonCode] = []
        duration = _finite_number(
            row.get(
                "optimistic_duration_s",
                row.get("duration_lower_bound_s", row.get("expected_duration_s")),
            )
        )
        if duration is None or duration < 0:
            reasons.append(missing_reason)
        elif duration > observation.slack_s + _EPS:
            reasons.append(ReasonCode.DEADLINE_IMPOSSIBLE)

        energy = _finite_number(
            row.get(
                "vehicle_energy_upper_j",
                row.get("vehicle_energy_j", row.get("uplink_start_energy_j")),
            )
        )
        if energy is None or energy < 0:
            reasons.append(missing_reason)
        elif energy > float(observation.vehicle.get("battery_j", 0.0)) + _EPS:
            reasons.append(ReasonCode.BATTERY)

        memory = _finite_number(
            row.get("vehicle_memory_upper_bytes", row.get("memory_bytes"))
        )
        if action.kind in {ActionKind.LOCAL, ActionKind.PIPE}:
            # A RAW/READY decision replaces the focal task's existing trusted
            # buffer reservation atomically; it does not allocate the new
            # envelope on top of that same task-owned reservation.  Other
            # tasks remain fully deducted from the public remaining capacity.
            focal_row = next(
                (
                    _as_mapping(item)
                    for item in observation.vehicle.get("active_tasks", ())
                    if bool(_as_mapping(item).get("is_focal", False))
                ),
                {},
            )
            focal_memory = max(
                0.0,
                float(focal_row.get("memory_reservation_bytes", 0.0)),
            )
            effective_memory_remaining = (
                float(observation.vehicle.get("memory_remaining_bytes", 0.0))
                + focal_memory
            )
            if memory is None or memory < 0:
                reasons.append(missing_reason)
            elif memory > effective_memory_remaining:
                reasons.append(ReasonCode.VEHICLE_MEMORY)
            token_defaults = (
                {"accelerator": 1}
                if action.kind is ActionKind.LOCAL
                else {"accelerator": 1, "cpu": 1, "encoder": 1}
            )
            tokens = row.get("descriptor_tokens", token_defaults)
            if not isinstance(tokens, Mapping):
                reasons.append(ReasonCode.VEHICLE_CAPACITY)
            else:
                remaining = _as_mapping(observation.vehicle.get("descriptor_remaining"))
                focal_tokens = _as_mapping(focal_row.get("reservation_tokens"))
                for name, count in sorted(
                    tokens.items(), key=lambda item: str(item[0])
                ):
                    numeric = _finite_number(count)
                    available = float(remaining.get(str(name), 0)) + float(
                        focal_tokens.get(str(name), 0)
                    )
                    if numeric is None or numeric < 0 or numeric > available:
                        reasons.append(ReasonCode.VEHICLE_CAPACITY)
                        break
        return reasons

    def _local_reasons(
        self, action: Action, observation: Observation, details: dict[str, Any]
    ) -> list[ReasonCode]:
        reasons = self._version_reasons(action, observation)
        if self._call_support("local", action, observation) is not True:
            reasons.append(ReasonCode.PAIRED_MEASUREMENT_MISSING)
        reasons.extend(self._bound_reasons(action, observation, details))
        return reasons

    def _pipeline_reasons(
        self, action: Action, observation: Observation, details: dict[str, Any]
    ) -> list[ReasonCode]:
        reasons = self._version_reasons(action, observation)
        reasons.extend(self._privacy_reasons(action.pipeline_id, observation, details))
        pipeline = self.profile.pipelines.get(action.pipeline_id or "")
        if (
            pipeline is not None
            and observation.attempt_started_count >= pipeline.max_attempts
        ):
            reasons.append(ReasonCode.RETRY_EXHAUSTED)
        support = self._call_support("anon", action, observation)
        # query_privacy already proves joint trace support in every candidate
        # quality cell.  An explicit trace index can only tighten that result.
        if support is False:
            reasons.append(ReasonCode.JOINT_TRACE_MISSING)
        reasons.extend(self._bound_reasons(action, observation, details))

        if pipeline is not None:
            fallback = pipeline.fallback_local_model
            fallback_possible = False
            if fallback:
                local = Action.local(ActionStage.RAW, fallback)
                local_details: dict[str, Any] = {}
                fallback_possible = not self._local_reasons(
                    local, observation, local_details
                )
            reachable_edge = False
            for rsu_id, link_row in sorted(observation.links.items()):
                if not bool(_as_mapping(link_row).get("connected", False)):
                    continue
                rsu = _as_mapping(observation.rsus.get(rsu_id))
                if not rsu or bool(rsu.get("failed", True)):
                    continue
                cached = _as_mapping(rsu.get("cached_models"))
                for model in self.profile.edge_models.values():
                    if (
                        model.supported_pipelines
                        and pipeline.pipeline_id not in model.supported_pipelines
                    ):
                        continue
                    if rsu_id not in model.supported_rsus:
                        continue
                    if cached.get(model.model_id) == model.model_hash:
                        reachable_edge = True
                        break
                if reachable_edge:
                    break
            if not reachable_edge and not fallback_possible:
                reasons.append(ReasonCode.CONNECTIVITY)
        return reasons

    def _edge_reasons(
        self,
        action: Action,
        observation: Observation,
        details: dict[str, Any],
    ) -> list[ReasonCode]:
        reasons: list[ReasonCode] = []
        encoded = observation.encoded_evidence
        if (
            encoded.get("message_source_type") != "EncodedAnon"
            or not encoded.get("artifact_token")
            or encoded.get("artifact_token") != observation.artifact_token
            or encoded.get("pipeline_id") != observation.selected_pipeline
            or not observation.encoded_size_bytes
        ):
            reasons.append(ReasonCode.MESSAGE_EVIDENCE_MISSING)
        reasons.extend(self._version_reasons(action, observation))
        reasons.extend(
            self._privacy_reasons(observation.selected_pipeline, observation, details)
        )
        if (
            self._call_support(
                "edge",
                action,
                observation,
                evaluation_pair_supported=self._evaluation_pair_supported(
                    action, observation
                ),
            )
            is not True
        ):
            reasons.append(ReasonCode.PAIRED_MEASUREMENT_MISSING)
        reasons.extend(self._bound_reasons(action, observation, details))

        link = _as_mapping(observation.links.get(action.rsu_id or ""))
        if not link or not bool(link.get("connected", False)):
            reasons.append(ReasonCode.CONNECTIVITY)
        rsu = _as_mapping(observation.rsus.get(action.rsu_id or ""))
        if not rsu or bool(rsu.get("failed", True)):
            reasons.append(ReasonCode.CONNECTIVITY)
        max_age = (
            self.config.max_snapshot_age_s
            if self.config is not None
            else _finite_number(observation.metadata.get("max_snapshot_age_s"), 0.0)
        )
        age = _finite_number(rsu.get("snapshot_age_s"))
        if age is None or max_age is None or age > max_age + _EPS:
            reasons.append(ReasonCode.SNAPSHOT_STALE)

        row = action_estimate(action, observation, self.trace_support)
        descriptors = _finite_number(row.get("rsu_descriptor_count"))
        vram = _finite_number(row.get("rsu_vram_upper_bytes"))
        work = _finite_number(
            row.get("rsu_work_upper_gpu_s", row.get("conservative_work_gpu_s"))
        )
        if None in {descriptors, vram, work}:
            reasons.append(ReasonCode.PAIRED_MEASUREMENT_MISSING)
        else:
            assert descriptors is not None and vram is not None and work is not None
            if descriptors < 0 or vram < 0 or work < 0:
                reasons.append(ReasonCode.PAIRED_MEASUREMENT_MISSING)
            if descriptors + float(rsu.get("descriptors", 0)) > float(
                rsu.get("descriptor_capacity", 0)
            ):
                reasons.append(ReasonCode.SNAPSHOT_CAPACITY)
            if vram + float(rsu.get("vram_bytes", 0)) > float(
                rsu.get("vram_capacity_bytes", 0)
            ):
                reasons.append(ReasonCode.SNAPSHOT_CAPACITY)
            if (
                work + float(rsu.get("reserved_work_gpu_s", 0.0))
                > float(rsu.get("workload_capacity_gpu_s", 0.0)) + _EPS
            ):
                reasons.append(ReasonCode.SNAPSHOT_CAPACITY)
        model = self.profile.edge_models.get(action.edge_model_id or "")
        cached = _as_mapping(rsu.get("cached_models"))
        if model is None or cached.get(action.edge_model_id or "") != model.model_hash:
            reasons.append(ReasonCode.MODEL_CACHE_MISSING)
        return reasons

    def assess(
        self,
        action: Action,
        observation: Observation,
    ) -> RemovalRecord:
        details: dict[str, Any] = {}
        reasons: list[ReasonCode] = []
        if action.stage is not observation.stage:
            reasons.append(ReasonCode.STAGE_ILLEGAL)
        elif action.kind is ActionKind.FAIL:
            pass
        elif action.kind is ActionKind.LOCAL:
            reasons.extend(self._local_reasons(action, observation, details))
        elif action.kind is ActionKind.PIPE:
            reasons.extend(self._pipeline_reasons(action, observation, details))
        elif action.kind is ActionKind.EDGE:
            reasons.extend(self._edge_reasons(action, observation, details))
        return RemovalRecord(action, _stable_reasons(reasons), deep_freeze(details))

    def enumerate(
        self,
        task: TaskRecord,
        observation: Observation,
        state: SimulationState | None = None,
        *,
        candidates: Iterable[Action] | None = None,
    ) -> MaskResult:
        """Return all hard-safe actions and reasons for every deletion.

        The policy-visible observation carries only an opaque artifact token.
        Its session-local registry stores only sanitized action capabilities;
        no evaluation artifact key is present in this object graph.
        """

        if (
            task.task_id != observation.task_id
            or task.vehicle_id != observation.vehicle_id
        ):
            raise ValueError("task and observation identity mismatch")
        if state is not None and state.clock_s != observation.time_s:
            raise ValueError(
                "hard mask requires an observation from the current event time"
            )
        rows = tuple(
            sorted(
                set(
                    candidates
                    if candidates is not None
                    else self._candidates(observation)
                )
            )
        )
        records: dict[Action, RemovalRecord] = {}
        allowed: list[Action] = []
        removed: dict[Action, tuple[ReasonCode, ...]] = {}
        for action in rows:
            record = self.assess(action, observation)
            records[action] = record
            if record.reasons:
                removed[action] = record.reasons
            else:
                allowed.append(action)
        return MaskResult(
            observation.stage,
            rows,
            tuple(sorted(allowed)),
            MappingProxyType(dict(sorted(removed.items()))),
            MappingProxyType(dict(sorted(records.items()))),
        )

    filter_candidates = enumerate


_FROZEN_FALLBACK_FAILURES = frozenset(
    {
        FailureReason.ANON_FAILED.value,
        FailureReason.ANON_OOM.value,
        FailureReason.GUARD_REJECTED.value,
        FailureReason.ENCODE_FAILED.value,
        FailureReason.ENCODE_SIZE_OOD.value,
        FailureReason.UL_FAILED.value,
        FailureReason.PERMANENT_LINK_LOSS.value,
        FailureReason.ADMISSION_REJECTED.value,
        FailureReason.EDGE_FAILED.value,
        FailureReason.DL_FAILED.value,
    }
)


@dataclass(frozen=True, slots=True)
class RepairDecision:
    proposed: Action
    executed: Action
    changed: bool
    trigger: str | None
    proposed_reasons: tuple[ReasonCode, ...]
    scores: Mapping[Action, float]
    mask: MaskResult

    def audit_row(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed.to_dict(),
            "executed": self.executed.to_dict(),
            "changed": self.changed,
            "trigger": self.trigger,
            "proposed_reason_codes": [reason.value for reason in self.proposed_reasons],
            "scores": {
                action.canonical_id: score
                for action, score in sorted(self.scores.items())
            },
        }


class DeterministicRepair:
    """Execution gate that cannot bypass privacy/type/physical hard masks."""

    def __init__(self, mask_engine: HardMaskEngine) -> None:
        self.mask_engine = mask_engine

    @staticmethod
    def _trigger_value(failure_reason: FailureReason | str | None) -> str | None:
        if failure_reason is None:
            return None
        return (
            failure_reason.value
            if isinstance(failure_reason, Enum)
            else str(failure_reason)
        )

    @staticmethod
    def _default_score(
        action: Action, observation: Observation, provider: Any
    ) -> float:
        row = action_estimate(action, observation, provider)
        value = _finite_number(row.get("repair_score", row.get("expected_cost")))
        if value is not None:
            return value
        # Fixed and documented fallback ranking; FAIL is last unless it is the
        # only hard-safe result.  This ranking contains no private truth.
        return {
            ActionKind.LOCAL: 0.0,
            ActionKind.PIPE: 1.0,
            ActionKind.EDGE: 1.0,
            ActionKind.FAIL: 1e30,
        }[action.kind]

    def repair(
        self,
        proposed: Action,
        task: TaskRecord,
        actual_observation: Observation,
        state: SimulationState | None = None,
        *,
        failure_reason: FailureReason | str | None = None,
        score: Callable[[Action], float] | Mapping[Action, float] | None = None,
    ) -> RepairDecision:
        mask = self.mask_engine.enumerate(task, actual_observation, state)
        trigger = self._trigger_value(failure_reason)
        forced_fallback = trigger in _FROZEN_FALLBACK_FAILURES
        has_partial_packet = (
            task.ul_remaining_bits > _EPS
            or task.dl_remaining_bits > _EPS
            or task.current_transfer_id is not None
        )

        if (
            proposed in mask.allowed
            and not forced_fallback
            and not (has_partial_packet and proposed.kind is ActionKind.EDGE)
        ):
            return RepairDecision(
                proposed,
                proposed,
                False,
                trigger,
                (),
                MappingProxyType(
                    {proposed: self._score(proposed, actual_observation, score)}
                ),
                mask,
            )

        choices = list(mask.allowed)
        if forced_fallback or has_partial_packet:
            pipeline_id = task.selected_pipeline or proposed.pipeline_id
            pipeline = self.mask_engine.profile.pipelines.get(pipeline_id or "")
            fallback = None if pipeline is None else pipeline.fallback_local_model
            choices = [
                action
                for action in choices
                if action.kind is ActionKind.LOCAL and action.local_model_id == fallback
            ]
            # A failed/partial upload is never repaired by choosing another
            # RSU.  The complete anonymous packet may only be retried later by
            # an explicit new READY decision if the state machine permits it.
        if not choices:
            fail = Action.fail(actual_observation.stage)
            choices = [fail] if fail in mask.allowed else []
        if not choices:
            # FAIL is structurally generated and always hard-safe; reaching
            # this line indicates caller-supplied candidate corruption.
            raise RuntimeError("hard mask did not contain an explicit FAIL action")

        score_rows = {
            action: self._score(action, actual_observation, score) for action in choices
        }
        executed = min(
            choices, key=lambda action: (score_rows[action], action.sort_key)
        )
        return RepairDecision(
            proposed,
            executed,
            executed != proposed,
            trigger,
            mask.reasons_for(proposed),
            MappingProxyType(dict(sorted(score_rows.items()))),
            mask,
        )

    def _score(
        self,
        action: Action,
        observation: Observation,
        score: Callable[[Action], float] | Mapping[Action, float] | None,
    ) -> float:
        if callable(score):
            value = score(action)
        elif isinstance(score, Mapping):
            value = score.get(action, math.inf)
        else:
            value = self._default_score(
                action, observation, self.mask_engine.trace_support
            )
        result = _finite_number(value, math.inf)
        return math.inf if result is None else result


__all__ = [
    "Action",
    "DeterministicRepair",
    "HardMaskEngine",
    "MaskResult",
    "Observation",
    "ObservationBuilder",
    "OnlineDecisionConfigView",
    "RemovalRecord",
    "RepairDecision",
    "action_estimate",
]
