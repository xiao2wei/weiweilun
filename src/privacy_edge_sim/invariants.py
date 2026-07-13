"""Fatal per-compound-event invariant checks with compact state snapshots."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from .enums import JobStatus, Operation, TaskState, TransferDirection, TransferStatus
from .errors import InvariantViolation
from .events import _same_representable_instant
from .packets import AlignedTensorHandle, AnonFERRequest, RawImageHandle
from .state import SimulationState


FORBIDDEN_OBSERVATION_FIELDS = {
    "artifact_key",
    "raw_handle",
    "aligned_handle",
    "true_identity",
    "true_expression_label",
    "true_quality_region",
    "realized_attack_outcomes",
    "realized_fer_loss",
    "future_trace",
    "trace_cursor",
}


def _contains_protected(value: Any, seen: set[int] | None = None) -> bool:
    if isinstance(value, (RawImageHandle, AlignedTensorHandle)):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return False
    seen = seen or set()
    marker = id(value)
    if marker in seen:
        return False
    seen.add(marker)
    if isinstance(value, Mapping):
        return any(
            _contains_protected(k, seen) or _contains_protected(v, seen)
            for k, v in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_protected(v, seen) for v in value)
    if is_dataclass(value):
        return any(
            _contains_protected(getattr(value, f.name), seen) for f in fields(value)
        )
    return False


def _snapshot(state: SimulationState, task_id: str | None = None) -> dict[str, Any]:
    tasks = state.tasks.values() if task_id is None else [state.tasks[task_id]]
    return {
        "clock_s": state.clock_s,
        "tasks": {
            task.task_id: {
                "state": task.state.value,
                "attempts": task.attempt_started_count,
                "job": task.current_job_id,
                "transfer": task.current_transfer_id,
                "battery_j": state.vehicles[task.vehicle_id].battery_j,
                "vehicle_energy_j": task.vehicle_energy_j,
                "rsu_energy_j": task.rsu_energy_j,
                "reservation": dict(task.reservation_tokens),
                "rsu_reserved": task.rsu_reserved,
            }
            for task in tasks
        },
        "transfers": {
            transfer_id: {
                "task_id": tr.task_id,
                "direction": tr.direction.value,
                "status": tr.status.value,
                "total_bits": tr.total_bits,
                "remaining_bits": tr.remaining_bits,
                "delivered_bits": tr.delivered_bits,
                "rsu_id": tr.rsu_id,
            }
            for transfer_id, tr in state.transfers.items()
        },
    }


def _fail(
    code: str,
    message: str,
    state: SimulationState,
    *,
    task_id: str | None = None,
    **context: Any,
) -> None:
    raise InvariantViolation(
        code, message, snapshot=_snapshot(state, task_id), **context
    )


def assert_observation_safe(observation: Any) -> None:
    if _contains_protected(observation):
        raise InvariantViolation(
            "OBSERVATION_PROTECTED_DATA",
            "policy observation contains vehicle-local data",
        )
    if is_dataclass(observation):
        names = {f.name for f in fields(observation)}
    elif isinstance(observation, dict):
        names = set(observation)
    else:
        names = set(vars(observation)) if hasattr(observation, "__dict__") else set()
    leaked = names & FORBIDDEN_OBSERVATION_FIELDS
    if leaked:
        raise InvariantViolation(
            "OBSERVATION_SIMULATOR_ONLY_FIELD",
            "policy observation exposes simulator-only fields",
            fields=sorted(leaked),
        )


def assert_all_invariants(
    state: SimulationState, profile: Any, *, previous_clock_s: float | None = None
) -> None:
    if previous_clock_s is not None and state.clock_s < previous_clock_s:
        _fail(
            "TIME_REGRESSION",
            "simulation time is not monotone",
            state,
            previous=previous_clock_s,
        )

    for vehicle in state.vehicles.values():
        if (
            vehicle.battery_j < -1e-9
            or vehicle.battery_j > vehicle.battery_capacity_j + 1e-9
        ):
            _fail(
                "BATTERY_RANGE",
                "vehicle battery left configured bounds",
                state,
                vehicle_id=vehicle.vehicle_id,
            )
        if (
            vehicle.memory_reserved_bytes < 0
            or vehicle.memory_reserved_bytes > vehicle.memory_capacity_bytes
        ):
            _fail(
                "VEHICLE_MEMORY_OVERFLOW",
                "vehicle memory reservation exceeds capacity",
                state,
                vehicle_id=vehicle.vehicle_id,
            )
        for key, value in vehicle.descriptors_reserved.items():
            if value < 0 or value > vehicle.descriptor_capacity[key]:
                _fail(
                    "VEHICLE_DESCRIPTOR_OVERFLOW",
                    "vehicle descriptor reservation exceeds capacity",
                    state,
                    vehicle_id=vehicle.vehicle_id,
                    resource=key,
                )
        for pool in vehicle.resources.values():
            if pool.running_count > pool.server_count:
                _fail(
                    "VEHICLE_CONCURRENCY",
                    "vehicle logical resource exceeded server count",
                    state,
                    resource=pool.resource_id,
                )
            for job in pool.jobs.values():
                if (
                    job.status in {JobStatus.WAITING, JobStatus.RUNNING}
                    and job.memory_need_bytes
                    > state.tasks[job.task_id].memory_reservation_bytes
                ):
                    _fail(
                        "VEHICLE_JOB_MEMORY_UNRESERVED",
                        "active vehicle job memory exceeds its task shadow reservation",
                        state,
                        task_id=job.task_id,
                        job_id=job.job_id,
                        need_bytes=job.memory_need_bytes,
                        reserved_bytes=state.tasks[
                            job.task_id
                        ].memory_reservation_bytes,
                    )
        if vehicle.battery_depleted and (
            vehicle.battery_j > 1e-9 or not vehicle.failed
        ):
            _fail(
                "BATTERY_DEPLETION_STATE",
                "a depleted vehicle must remain off with zero battery",
                state,
                vehicle_id=vehicle.vehicle_id,
            )
        expected_memory = sum(
            task.memory_reservation_bytes
            for task in state.tasks.values()
            if task.vehicle_id == vehicle.vehicle_id
        )
        if expected_memory != vehicle.memory_reserved_bytes:
            _fail(
                "VEHICLE_MEMORY_ACCOUNTING",
                "vehicle memory total does not equal task reservations",
                state,
                vehicle_id=vehicle.vehicle_id,
                expected=expected_memory,
                actual=vehicle.memory_reserved_bytes,
            )
        for key, actual in vehicle.descriptors_reserved.items():
            expected = sum(
                task.reservation_tokens.get(key, 0)
                for task in state.tasks.values()
                if task.vehicle_id == vehicle.vehicle_id
            )
            if expected != actual:
                _fail(
                    "VEHICLE_DESCRIPTOR_ACCOUNTING",
                    "vehicle descriptor total does not equal task reservations",
                    state,
                    vehicle_id=vehicle.vehicle_id,
                    resource=key,
                    expected=expected,
                    actual=actual,
                )
        attributed = sum(
            task.vehicle_energy_j + task.hold_vehicle_energy_j
            for task in state.tasks.values()
            if task.vehicle_id == vehicle.vehicle_id
        )
        if attributed < -1e-12 or attributed > vehicle.physical_energy_j + 1e-8:
            _fail(
                "VEHICLE_ENERGY_ACCOUNTING",
                "task-attributed vehicle energy exceeds physical vehicle energy",
                state,
                vehicle_id=vehicle.vehicle_id,
                attributed_j=attributed,
                physical_j=vehicle.physical_energy_j,
            )

    for rsu in state.rsus.values():
        rsu.admission.assert_within_capacity()
        if (
            rsu.system_maintenance_energy_j < -1e-12
            or rsu.system_maintenance_energy_j > rsu.physical_energy_j + 1e-8
        ):
            _fail(
                "RSU_MAINTENANCE_ENERGY_ACCOUNTING",
                "system maintenance energy must be nonnegative and included in physical RSU energy",
                state,
                rsu_id=rsu.rsu_id,
                maintenance_j=rsu.system_maintenance_energy_j,
                physical_j=rsu.physical_energy_j,
            )
        if (
            rsu.ingress.running_count > rsu.ingress.server_count
            or rsu.gpu.running_count > rsu.gpu.server_count
        ):
            _fail(
                "RSU_CONCURRENCY",
                "RSU resource exceeded finite server count",
                state,
                rsu_id=rsu.rsu_id,
            )
        for job in rsu.gpu.jobs.values():
            if job.status not in {JobStatus.WAITING, JobStatus.RUNNING}:
                continue
            if job.operation is Operation.RSU_MODEL_MAINTENANCE:
                if job.task_id is not None or job.total_dynamic_energy_j <= 0:
                    _fail(
                        "RSU_MODEL_MAINTENANCE_INVALID",
                        "active model-maintenance job must be taskless with positive energy",
                        state,
                        job_id=job.job_id,
                        rsu_id=rsu.rsu_id,
                    )
                continue
            reservation = rsu.admission.reservation(job.task_id)
            if reservation is None:
                _fail(
                    "RSU_JOB_WITHOUT_ADMISSION",
                    "active RSU GPU job has no atomic admission reservation",
                    state,
                    task_id=job.task_id,
                    job_id=job.job_id,
                    rsu_id=rsu.rsu_id,
                )
            if (
                job.memory_need_bytes > reservation.vram_bytes
                or job.total_work_s > reservation.conservative_work_gpu_s + 1e-12
            ):
                _fail(
                    "RSU_JOB_EXCEEDS_ADMISSION",
                    "active RSU GPU job exceeds its atomic admission envelope",
                    state,
                    task_id=job.task_id,
                    job_id=job.job_id,
                    rsu_id=rsu.rsu_id,
                    job_vram_bytes=job.memory_need_bytes,
                    reserved_vram_bytes=reservation.vram_bytes,
                    job_work_s=job.total_work_s,
                    reserved_work_s=reservation.conservative_work_gpu_s,
                )
    attributed_rsu = sum(
        task.rsu_energy_j + task.hold_rsu_energy_j for task in state.tasks.values()
    )
    physical_rsu = sum(rsu.physical_energy_j for rsu in state.rsus.values())
    if attributed_rsu < -1e-12 or attributed_rsu > physical_rsu + 1e-8:
        _fail(
            "RSU_ENERGY_ACCOUNTING",
            "task-attributed RSU energy exceeds physical RSU energy",
            state,
            attributed_j=attributed_rsu,
            physical_j=physical_rsu,
        )

    active_job_owners: dict[str, int] = {}
    for vehicle in state.vehicles.values():
        for pool in vehicle.resources.values():
            for job in pool.jobs.values():
                if job.task_id is not None and job.status in {
                    JobStatus.WAITING,
                    JobStatus.RUNNING,
                }:
                    active_job_owners[job.task_id] = (
                        active_job_owners.get(job.task_id, 0) + 1
                    )

    active_transfer_owners: dict[str, int] = {}
    for transfer in state.transfers.values():
        if transfer.status in {TransferStatus.ACTIVE, TransferStatus.PAUSED}:
            active_transfer_owners[transfer.task_id] = (
                active_transfer_owners.get(transfer.task_id, 0) + 1
            )

    compute_states = {
        TaskState.PREP_WAIT,
        TaskState.PREP_RUN,
        TaskState.LOCAL_WAIT,
        TaskState.LOCAL_RUN,
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.EDGE_WAIT,
        TaskState.EDGE_RUN,
    }
    transfer_states = {TaskState.UL, TaskState.DL}
    for rsu in state.rsus.values():
        for pool in (rsu.ingress, rsu.gpu):
            for job in pool.jobs.values():
                if job.task_id is not None and job.status in {
                    JobStatus.WAITING,
                    JobStatus.RUNNING,
                }:
                    active_job_owners[job.task_id] = (
                        active_job_owners.get(job.task_id, 0) + 1
                    )

    for task in state.tasks.values():
        if not isinstance(task.state, TaskState):
            _fail(
                "TASK_STATE_INVALID",
                "task does not have exactly one legal state",
                state,
                task_id=task.task_id,
            )
        if task.attempt_started_count < 0 or (
            task.max_attempts and task.attempt_started_count > task.max_attempts
        ):
            _fail(
                "RETRY_LIMIT_EXCEEDED",
                "task exceeded total attempt limit",
                state,
                task_id=task.task_id,
            )
        if task.selected_pipeline and task.ood:
            _fail(
                "OOD_PIPELINE_SELECTED",
                "OOD task entered an anonymization pipeline",
                state,
                task_id=task.task_id,
            )
        if task.selected_pipeline and not task.ood and task.conformal_quality_bins:
            query = profile.query_privacy(
                task.selected_pipeline,
                task.conformal_quality_bins,
                state.vehicles[task.vehicle_id].device_type,
            )
            if task.state not in {TaskState.RAW, TaskState.FAIL} and not query.safe:
                _fail(
                    "PIPELINE_PRIVACY_UNSAFE",
                    "selected pipeline is not safe for all candidate bins",
                    state,
                    task_id=task.task_id,
                )
        if task.terminal:
            retained_transfer = any(
                tr.task_id == task.task_id for tr in state.transfers.values()
            )
            admission_held = any(
                rsu.admission.pinned_model(task.task_id) is not None
                for rsu in state.rsus.values()
            )
            if (
                active_job_owners.get(task.task_id, 0)
                or retained_transfer
                or task.current_job_id is not None
                or task.current_transfer_id is not None
                or task.reservation_tokens
                or task.memory_reservation_bytes
                or admission_held
                or task.rsu_reserved
            ):
                _fail(
                    "TERMINAL_RESOURCE_LEAK",
                    "terminal task still owns a job, transfer object or reservation",
                    state,
                    task_id=task.task_id,
                )
            if task.raw_handle is not None or task.aligned_handle is not None:
                _fail(
                    "TERMINAL_RAW_RETAINED",
                    "terminal cleanup retained protected handles",
                    state,
                    task_id=task.task_id,
                )
        elif active_job_owners.get(task.task_id, 0) > 1:
            _fail(
                "TASK_MULTIPLE_ACTIVE_JOBS",
                "one task owns more than one active compute job",
                state,
                task_id=task.task_id,
            )
        if active_transfer_owners.get(task.task_id, 0) > 1:
            _fail(
                "TASK_MULTIPLE_ACTIVE_TRANSFERS",
                "one task owns more than one active transfer",
                state,
                task_id=task.task_id,
            )
        if active_job_owners.get(task.task_id, 0) and active_transfer_owners.get(
            task.task_id, 0
        ):
            _fail(
                "TASK_JOB_TRANSFER_OVERLAP",
                "task owns compute and transfer simultaneously",
                state,
                task_id=task.task_id,
            )
        if not task.terminal:
            if (
                task.state in compute_states
                and active_job_owners.get(task.task_id, 0) != 1
            ):
                _fail(
                    "TASK_COMPUTE_STATE_ORPHAN",
                    "compute wait/run state lacks exactly one active job",
                    state,
                    task_id=task.task_id,
                )
            if (
                task.state in transfer_states
                and active_transfer_owners.get(task.task_id, 0) != 1
            ):
                _fail(
                    "TASK_TRANSFER_STATE_ORPHAN",
                    "transfer state lacks exactly one active transfer",
                    state,
                    task_id=task.task_id,
                )
            if task.state not in compute_states | transfer_states and (
                active_job_owners.get(task.task_id, 0)
                or active_transfer_owners.get(task.task_id, 0)
            ):
                _fail(
                    "TASK_STATE_RESOURCE_MISMATCH",
                    "task state and active resource disagree",
                    state,
                    task_id=task.task_id,
                )
        if task.state is TaskState.DONE:
            late = (
                task.terminal_time_s is not None
                and task.terminal_time_s > task.absolute_deadline_s
                and not _same_representable_instant(
                    task.terminal_time_s, task.absolute_deadline_s
                )
            )
            if not task.result_valid or task.terminal_time_s is None or late:
                _fail(
                    "LATE_OR_INVALID_DONE",
                    "DONE task lacks a timely valid result",
                    state,
                    task_id=task.task_id,
                )

    for transfer in state.transfers.values():
        if transfer.direction is TransferDirection.UL:
            if not isinstance(transfer.packet, AnonFERRequest) or _contains_protected(
                transfer.packet
            ):
                _fail(
                    "ILLEGAL_UPLINK_PACKET",
                    "uplink packet type/schema contains protected data",
                    state,
                    task_id=transfer.task_id,
                )
        if transfer.rsu_id not in state.rsus:
            _fail(
                "TRANSFER_RSU_UNKNOWN",
                "transfer targets unknown RSU",
                state,
                task_id=transfer.task_id,
            )
        task = state.tasks[transfer.task_id]
        if (
            transfer.status in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
            and task.selected_rsu != transfer.rsu_id
        ):
            _fail(
                "PARTIAL_PACKET_RSU_MISMATCH",
                "active partial packet moved away from its selected RSU",
                state,
                task_id=transfer.task_id,
            )
        if (
            transfer.direction is TransferDirection.UL
            and transfer.packet.protocol_version != profile.protocol_version
        ):
            _fail(
                "UPLINK_PROTOCOL_MISMATCH",
                "active uplink packet protocol differs from frozen profile",
                state,
                task_id=transfer.task_id,
            )
        if (
            abs(transfer.total_bits - transfer.remaining_bits - transfer.delivered_bits)
            > 1e-6
        ):
            _fail(
                "TRANSFER_BIT_ACCOUNTING",
                "cumulative delivered bits do not match remaining bits",
                state,
                task_id=transfer.task_id,
            )
        if transfer.remaining_bits < -1e-9:
            _fail(
                "TRANSFER_NEGATIVE_BITS",
                "transfer remaining bits became negative",
                state,
                task_id=transfer.task_id,
            )

    state.invariant_checks += 1
