"""Task state machine and mutable physical simulation state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .enums import FailureReason, TaskState, TransferDirection, TransferStatus
from .errors import InvariantViolation, TransitionError
from .events import EventQueue, _same_representable_instant
from .packets import (
    AlignedTensorHandle,
    AnonFERRequest,
    EncodedAnon,
    FERResult,
    RawImageHandle,
)
from .resources import RSUAdmission, ResourcePool


LEGAL_SUCCESSORS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.RAW_BUF: frozenset({TaskState.PREP_WAIT, TaskState.FAIL}),
    TaskState.PREP_WAIT: frozenset({TaskState.PREP_RUN, TaskState.FAIL}),
    TaskState.PREP_RUN: frozenset({TaskState.RAW, TaskState.FAIL}),
    TaskState.RAW: frozenset(
        {TaskState.LOCAL_WAIT, TaskState.ANON_WAIT, TaskState.FAIL}
    ),
    TaskState.LOCAL_WAIT: frozenset({TaskState.LOCAL_RUN, TaskState.FAIL}),
    TaskState.LOCAL_RUN: frozenset({TaskState.DONE, TaskState.FAIL}),
    TaskState.ANON_WAIT: frozenset(
        {TaskState.ANON_RUN, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.ANON_RUN: frozenset(
        {
            TaskState.GUARD_WAIT,
            TaskState.ANON_WAIT,
            TaskState.LOCAL_WAIT,
            TaskState.FAIL,
        }
    ),
    TaskState.GUARD_WAIT: frozenset(
        {TaskState.GUARD_RUN, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.GUARD_RUN: frozenset(
        {
            TaskState.ENCODE_WAIT,
            TaskState.ANON_WAIT,
            TaskState.LOCAL_WAIT,
            TaskState.FAIL,
        }
    ),
    TaskState.ENCODE_WAIT: frozenset(
        {TaskState.ENCODE_RUN, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.ENCODE_RUN: frozenset(
        {TaskState.READY, TaskState.ANON_WAIT, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.READY: frozenset({TaskState.UL, TaskState.LOCAL_WAIT, TaskState.FAIL}),
    TaskState.UL: frozenset(
        {TaskState.EDGE_WAIT, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.EDGE_WAIT: frozenset(
        {TaskState.EDGE_RUN, TaskState.LOCAL_WAIT, TaskState.FAIL}
    ),
    TaskState.EDGE_RUN: frozenset({TaskState.DL, TaskState.LOCAL_WAIT, TaskState.FAIL}),
    TaskState.DL: frozenset({TaskState.DONE, TaskState.LOCAL_WAIT, TaskState.FAIL}),
    TaskState.DONE: frozenset({TaskState.DONE}),
    TaskState.FAIL: frozenset({TaskState.FAIL}),
}


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    time_s: float
    previous: TaskState | None
    current: TaskState
    trigger: str
    detail: str | None = None


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    vehicle_id: str
    arrival_time_s: float
    relative_deadline_s: float
    absolute_deadline_s: float
    raw_handle: RawImageHandle | None
    aligned_handle: AlignedTensorHandle | None = None
    quality_features: tuple[float, ...] = ()
    quality_probabilities: tuple[tuple[str, float], ...] = ()
    conformal_quality_bins: tuple[str, ...] = ()
    ood: bool = False
    device_context: str = "nominal"
    selected_pipeline: str | None = None
    selected_local_model: str | None = None
    selected_rsu: str | None = None
    selected_edge_model: str | None = None
    attempt_started_count: int = 0
    max_attempts: int = 0
    trace_row_id: str | None = None
    current_attempt_index: int = -1
    artifact_key: str | None = None
    encoded_anon: EncodedAnon | None = None
    encoded_size_bytes: int | None = None
    current_job_id: str | None = None
    current_transfer_id: str | None = None
    ul_remaining_bits: float = 0.0
    dl_remaining_bits: float = 0.0
    vehicle_energy_j: float = 0.0
    rsu_energy_j: float = 0.0
    hold_vehicle_energy_j: float = 0.0
    hold_rsu_energy_j: float = 0.0
    failure_reason: FailureReason = FailureReason.NONE
    terminal_time_s: float | None = None
    result_valid: bool = False
    realized_fer_loss: float | None = None
    realized_fer_true_label: str | None = None
    realized_fer_class_probabilities: tuple[tuple[str, float], ...] = ()
    evaluation_subject_cluster_id: str | None = None
    actual_path: list[str] = field(default_factory=list)
    enqueue_times: dict[str, list[float]] = field(default_factory=dict)
    start_times: dict[str, list[float]] = field(default_factory=dict)
    end_times: dict[str, list[float]] = field(default_factory=dict)
    phase_history: list[PhaseTransition] = field(default_factory=list)
    mask_audit: list[dict[str, Any]] = field(default_factory=list)
    action_audit: list[dict[str, Any]] = field(default_factory=list)
    compute_audit: list[dict[str, Any]] = field(default_factory=list)
    anon_attempt_audit: list[dict[str, Any]] = field(default_factory=list)
    network_audit: list[dict[str, Any]] = field(default_factory=list)
    rsu_audit: list[dict[str, Any]] = field(default_factory=list)
    reservation_tokens: dict[str, int] = field(default_factory=dict)
    memory_reservation_bytes: int = 0
    rsu_reserved: bool = False
    true_identity: str | None = None
    true_expression_label: str | None = None
    # Simulator-only offline region g*.  Observation builders must never copy
    # this field into a policy-visible object.
    true_quality_region: str | None = None
    realized_attack_outcomes: dict[str, bool] = field(default_factory=dict)
    _state: TaskState = field(default=TaskState.RAW_BUF, repr=False)
    _event_versions: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.vehicle_id:
            raise ValueError("task and vehicle IDs are required")
        if not math.isfinite(self.arrival_time_s) or self.arrival_time_s < 0:
            raise ValueError("arrival time must be finite and nonnegative")
        if not math.isfinite(self.relative_deadline_s) or self.relative_deadline_s <= 0:
            raise ValueError("relative deadline must be finite and positive")
        expected = self.arrival_time_s + self.relative_deadline_s
        if (
            not math.isfinite(expected)
            or not math.isfinite(self.absolute_deadline_s)
            or not _same_representable_instant(self.absolute_deadline_s, expected)
        ):
            raise ValueError("absolute deadline must equal arrival + relative deadline")
        self.phase_history.append(
            PhaseTransition(self.arrival_time_s, None, self._state, "ARRIVAL")
        )

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._state.terminal

    def bump_event_version(self, object_id: str) -> int:
        value = self._event_versions.get(object_id, 0) + 1
        self._event_versions[object_id] = value
        return value

    def event_version(self, object_id: str) -> int:
        return self._event_versions.get(object_id, 0)

    def record_time(self, table: str, operation: str, time_s: float) -> None:
        target = {
            "enqueue": self.enqueue_times,
            "start": self.start_times,
            "end": self.end_times,
        }[table]
        target.setdefault(operation, []).append(time_s)

    def mark_anon_enqueued(self, max_attempts: int) -> int:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.attempt_started_count >= max_attempts:
            raise InvariantViolation(
                "RETRY_LIMIT_EXCEEDED",
                "attempt counter would exceed frozen total-attempt limit",
                task_id=self.task_id,
                attempts=self.attempt_started_count,
                max_attempts=max_attempts,
            )
        self.max_attempts = max_attempts
        self.attempt_started_count += 1
        self.current_attempt_index = self.attempt_started_count - 1
        return self.attempt_started_count


class TaskStateMachine:
    """The only component permitted to mutate a task's state."""

    @staticmethod
    def transition(
        task: TaskRecord,
        target: TaskState,
        *,
        time_s: float,
        trigger: str,
        detail: str | None = None,
        failure_reason: FailureReason | None = None,
    ) -> bool:
        current = task.state
        if current.terminal:
            if target is current:
                return False
            raise TransitionError(
                "TERMINAL_STATE_ABSORBING",
                "terminal task cannot transition",
                task_id=task.task_id,
                current=current.value,
                target=target.value,
            )
        if target not in LEGAL_SUCCESSORS[current]:
            raise TransitionError(
                "ILLEGAL_TASK_TRANSITION",
                "state transition is not in the closed transition relation",
                task_id=task.task_id,
                current=current.value,
                target=target.value,
                trigger=trigger,
            )
        if time_s < task.arrival_time_s or (
            task.phase_history and time_s < task.phase_history[-1].time_s
        ):
            raise InvariantViolation(
                "TASK_TIME_REGRESSION",
                "task phase timestamp moved backward",
                task_id=task.task_id,
            )
        if target is TaskState.DONE:
            if current not in {TaskState.LOCAL_RUN, TaskState.DL}:
                raise TransitionError(
                    "DONE_SOURCE_INVALID",
                    "DONE is only reachable from local result or downlink",
                )
            late = (
                time_s > task.absolute_deadline_s
                and not _same_representable_instant(time_s, task.absolute_deadline_s)
            )
            if late or not task.result_valid:
                raise TransitionError(
                    "DONE_RESULT_INVALID",
                    "DONE requires a valid result no later than deadline",
                )
        task._state = target
        task.phase_history.append(
            PhaseTransition(time_s, current, target, trigger, detail)
        )
        if target.terminal:
            task.terminal_time_s = time_s
            task.failure_reason = failure_reason or (
                FailureReason.NONE
                if target is TaskState.DONE
                else FailureReason.UNSUPPORTED
            )
        return True


@dataclass(slots=True)
class Transfer:
    transfer_id: str
    task_id: str
    vehicle_id: str
    rsu_id: str
    direction: TransferDirection
    packet: AnonFERRequest | FERResult
    total_bits: float
    remaining_bits: float
    start_time_s: float
    last_update_time_s: float
    status: TransferStatus = TransferStatus.ACTIVE
    paused_since_s: float | None = None
    completion_version: int = 1
    vehicle_energy_j: float = 0.0
    rsu_energy_j: float = 0.0
    delivered_bits: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.total_bits <= 0
            or self.remaining_bits <= 0
            or self.remaining_bits > self.total_bits
        ):
            raise ValueError("transfer bits must be positive and remaining <= total")
        if self.direction is TransferDirection.UL and not isinstance(
            self.packet, AnonFERRequest
        ):
            raise TypeError("uplink transfer only accepts AnonFERRequest")
        if self.direction is TransferDirection.DL and not isinstance(
            self.packet, FERResult
        ):
            raise TypeError("downlink transfer only accepts FERResult")

    def advance(
        self,
        dt_s: float,
        goodput_bps: float,
        vehicle_power_w: float,
        rsu_power_w: float,
    ) -> tuple[float, float, float]:
        if (
            self.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
            or dt_s <= 0
        ):
            return 0.0, 0.0, 0.0
        quantities = (goodput_bps, vehicle_power_w, rsu_power_w)
        if any(not math.isfinite(x) or x < 0 for x in quantities):
            raise InvariantViolation(
                "WIRELESS_TRACE_INVALID",
                "wireless rates and powers must be finite and nonnegative",
            )
        rate = goodput_bps if self.status is TransferStatus.ACTIVE else 0.0
        delivered = min(self.remaining_bits, rate * dt_s)
        self.remaining_bits = max(0.0, self.remaining_bits - delivered)
        if self.remaining_bits <= 1e-9:
            self.remaining_bits = 0.0
        self.delivered_bits += delivered
        vehicle_energy = vehicle_power_w * dt_s
        rsu_energy = rsu_power_w * dt_s
        self.vehicle_energy_j += vehicle_energy
        self.rsu_energy_j += rsu_energy
        self.last_update_time_s += dt_s
        return delivered, vehicle_energy, rsu_energy


@dataclass(slots=True)
class VehicleRuntime:
    vehicle_id: str
    device_type: str
    battery_capacity_j: float
    battery_j: float
    memory_capacity_bytes: int
    memory_reserved_bytes: int
    descriptor_capacity: dict[str, int]
    descriptors_reserved: dict[str, int]
    resources: dict[str, ResourcePool]
    idle_power_w: float
    physical_energy_j: float = 0.0
    failed: bool = False
    battery_depleted: bool = False
    hold_power_w: float = 0.0

    def can_reserve(
        self, tokens: dict[str, int], memory_bytes: int
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if self.memory_reserved_bytes + memory_bytes > self.memory_capacity_bytes:
            reasons.append("VEHICLE_MEMORY")
        for key, count in tokens.items():
            if self.descriptors_reserved.get(
                key, 0
            ) + count > self.descriptor_capacity.get(key, 0):
                reasons.append(f"VEHICLE_DESCRIPTOR_{key}")
        return not reasons, tuple(reasons)

    def reserve(
        self, task: TaskRecord, tokens: dict[str, int], memory_bytes: int
    ) -> bool:
        if task.reservation_tokens or task.memory_reservation_bytes:
            raise InvariantViolation(
                "VEHICLE_RESERVATION_DUPLICATE",
                "new reservation attempted while the task already owns shadow resources",
                task_id=task.task_id,
            )
        ok, _ = self.can_reserve(tokens, memory_bytes)
        if not ok:
            return False
        for key, count in tokens.items():
            self.descriptors_reserved[key] = (
                self.descriptors_reserved.get(key, 0) + count
            )
        self.memory_reserved_bytes += memory_bytes
        task.reservation_tokens = dict(tokens)
        task.memory_reservation_bytes = memory_bytes
        return True

    def reconcile_reservation(
        self, task: TaskRecord, tokens: dict[str, int], memory_bytes: int
    ) -> bool:
        """Atomically replace one task's shadow reservation without oversubscription."""

        if (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, int)
            or memory_bytes < 0
            or any(
                not isinstance(key, str)
                or not key
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for key, count in tokens.items()
            )
        ):
            raise InvariantViolation(
                "VEHICLE_RESERVATION_INVALID",
                "replacement shadow reservation must use finite nonnegative integers",
                task_id=task.task_id,
            )
        unknown = set(tokens) - set(self.descriptor_capacity)
        if unknown:
            raise InvariantViolation(
                "VEHICLE_RESERVATION_RESOURCE",
                "replacement shadow reservation names an unknown logical resource",
                task_id=task.task_id,
                unknown=sorted(unknown),
            )
        next_memory = (
            self.memory_reserved_bytes - task.memory_reservation_bytes + memory_bytes
        )
        if next_memory < 0 or next_memory > self.memory_capacity_bytes:
            return False
        next_descriptors: dict[str, int] = {}
        for key, capacity in self.descriptor_capacity.items():
            value = (
                self.descriptors_reserved.get(key, 0)
                - task.reservation_tokens.get(key, 0)
                + tokens.get(key, 0)
            )
            if value < 0 or value > capacity:
                return False
            next_descriptors[key] = value
        self.memory_reserved_bytes = next_memory
        self.descriptors_reserved.update(next_descriptors)
        task.reservation_tokens = dict(tokens)
        task.memory_reservation_bytes = memory_bytes
        return True

    def release(self, task: TaskRecord) -> None:
        for key, count in task.reservation_tokens.items():
            self.descriptors_reserved[key] = max(
                0, self.descriptors_reserved.get(key, 0) - count
            )
        self.memory_reserved_bytes = max(
            0, self.memory_reserved_bytes - task.memory_reservation_bytes
        )
        task.reservation_tokens.clear()
        task.memory_reservation_bytes = 0


@dataclass(slots=True)
class RSURuntime:
    rsu_id: str
    admission: RSUAdmission
    ingress: ResourcePool
    gpu: ResourcePool
    idle_power_w: float
    physical_energy_j: float = 0.0
    system_maintenance_energy_j: float = 0.0
    current_snapshot_time_s: float = 0.0
    failed: bool = False
    hold_power_w: float = 0.0
    # Scheduler-visible telemetry is deliberately detached from the live RSU
    # state.  A dated observation is not a reservation and may be stale by the
    # time the complete packet reaches atomic admission.
    public_snapshot: dict[str, Any] | None = None


@dataclass(slots=True)
class VirtualQueueBank:
    vehicle_power: dict[str, float]
    rsu_power: dict[str, float]
    timeout: float = 0.0
    failure: float = 0.0
    coverage: float = 0.0
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        *,
        time_s: float,
        dt_s: float,
        vehicle_energy: dict[str, float],
        rsu_energy: dict[str, float],
        vehicle_power_budget: dict[str, float],
        rsu_power_budget: dict[str, float],
        arrivals: int,
        timeouts: int,
        failures: int,
        completed: int,
        beta_timeout: float,
        beta_failure: float,
        beta_coverage: float,
    ) -> None:
        for key in self.vehicle_power:
            self.vehicle_power[key] = max(
                0.0,
                self.vehicle_power[key]
                + vehicle_energy.get(key, 0.0)
                - vehicle_power_budget[key] * dt_s,
            )
        for key in self.rsu_power:
            self.rsu_power[key] = max(
                0.0,
                self.rsu_power[key]
                + rsu_energy.get(key, 0.0)
                - rsu_power_budget[key] * dt_s,
            )
        self.timeout = max(0.0, self.timeout + timeouts - beta_timeout * arrivals)
        self.failure = max(0.0, self.failure + failures - beta_failure * arrivals)
        self.coverage = max(0.0, self.coverage + beta_coverage * arrivals - completed)
        self.trajectory.append(
            {
                "time_s": time_s,
                "vehicle_power": dict(self.vehicle_power),
                "rsu_power": dict(self.rsu_power),
                "timeout": self.timeout,
                "failure": self.failure,
                "coverage": self.coverage,
            }
        )


@dataclass(slots=True)
class SimulationState:
    clock_s: float
    events: EventQueue
    tasks: dict[str, TaskRecord]
    vehicles: dict[str, VehicleRuntime]
    rsus: dict[str, RSURuntime]
    transfers: dict[str, Transfer]
    virtual_queues: VirtualQueueBank
    event_log: list[dict[str, Any]] = field(default_factory=list)
    invariant_checks: int = 0
    last_interval_vehicle_energy: dict[str, float] = field(default_factory=dict)
    last_interval_rsu_energy: dict[str, float] = field(default_factory=dict)
