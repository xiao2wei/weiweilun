"""Finite non-preemptive EDF resources and atomic RSU admission."""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

from .enums import JobStatus, Operation, ResourceKind
from .errors import InvariantViolation


EPS = 1e-12


@dataclass(slots=True)
class ComputeJob:
    job_id: str
    task_id: str | None
    owner_type: str
    owner_id: str
    operation: Operation
    resource_kind: ResourceKind
    model_or_pipeline_version: str
    enqueue_time_s: float
    absolute_deadline_s: float
    enqueue_seq: int
    total_work_s: float
    residual_work_s: float
    total_dynamic_energy_j: float
    consumed_dynamic_energy_j: float = 0.0
    memory_need_bytes: int = 0
    status: JobStatus = JobStatus.WAITING
    start_time_s: float | None = None
    end_time_s: float | None = None
    server_index: int | None = None
    completion_version: int = 0

    def __post_init__(self) -> None:
        numeric = (
            self.enqueue_time_s,
            self.absolute_deadline_s,
            self.total_work_s,
            self.residual_work_s,
            self.total_dynamic_energy_j,
        )
        if any(not math.isfinite(x) for x in numeric):
            raise ValueError("job quantities must be finite")
        if (
            self.total_work_s <= 0
            or self.residual_work_s <= 0
            or self.residual_work_s > self.total_work_s + EPS
        ):
            raise ValueError("job work must be positive and residual <= total")
        if self.total_dynamic_energy_j < 0 or self.memory_need_bytes < 0:
            raise ValueError("job energy and memory must be nonnegative")
        system_maintenance = self.operation is Operation.RSU_MODEL_MAINTENANCE
        if system_maintenance != (self.task_id is None):
            raise ValueError(
                "only RSU model-maintenance jobs may omit an ordinary task owner"
            )
        if system_maintenance and self.total_dynamic_energy_j <= 0:
            raise ValueError("RSU model maintenance must consume positive energy")

    def advance(
        self,
        dt_s: float,
        effective_rate: float,
    ) -> tuple[float, float]:
        if self.status is not JobStatus.RUNNING or dt_s <= 0:
            return 0.0, 0.0
        if effective_rate < 0 or not math.isfinite(effective_rate):
            raise InvariantViolation(
                "RESOURCE_RATE_INVALID",
                "resource service multiplier must be finite and nonnegative",
            )
        served = min(self.residual_work_s, effective_rate * dt_s)
        before = self.residual_work_s
        self.residual_work_s = max(0.0, before - served)
        if self.residual_work_s <= EPS:
            self.residual_work_s = 0.0
        # The frozen joint-trace row supplies the actual dynamic energy for the
        # complete service transaction in its paired device/thermal context.
        # Thermal service-rate changes wall time, but must not rescale that
        # paired energy a second time.  Partial execution consumes the same
        # fraction of energy as completed busy-work.
        energy = self.total_dynamic_energy_j * served / self.total_work_s
        self.consumed_dynamic_energy_j += energy
        return served, energy


class ResourcePool:
    """A finite set of identical logical servers with non-preemptive EDF."""

    def __init__(self, resource_id: str, kind: ResourceKind, server_count: int) -> None:
        if not isinstance(server_count, int) or server_count < 1:
            raise ValueError("server_count must be a finite integer >= 1")
        self.resource_id = resource_id
        self.kind = kind
        self.server_count = server_count
        self.jobs: dict[str, ComputeJob] = {}
        self._waiting: list[tuple[float, int, str]] = []
        self.running: list[str | None] = [None] * server_count
        self._enqueue_seq = itertools.count()
        self.busy_server_seconds = 0.0
        self.max_running_observed = 0

    def next_enqueue_seq(self) -> int:
        return next(self._enqueue_seq)

    def enqueue(self, job: ComputeJob) -> None:
        if job.resource_kind is not self.kind:
            raise InvariantViolation(
                "RESOURCE_KIND_MISMATCH",
                "job mapped to wrong logical resource",
                job_id=job.job_id,
                job_kind=job.resource_kind.value,
                pool_kind=self.kind.value,
            )
        if job.job_id in self.jobs:
            raise InvariantViolation(
                "DUPLICATE_JOB", "job ID already exists", job_id=job.job_id
            )
        self.jobs[job.job_id] = job
        heapq.heappush(
            self._waiting, (job.absolute_deadline_s, job.enqueue_seq, job.job_id)
        )

    def dispatch(
        self,
        now_s: float,
        *,
        eligible: Callable[[ComputeJob], bool] | None = None,
    ) -> list[ComputeJob]:
        """Start EDF jobs on idle servers, optionally respecting dependencies.

        ``eligible`` is a physical readiness gate, not a second scheduling
        policy.  Ineligible jobs retain their EDF entries and remain visible
        in queue/workload telemetry; the earliest eligible job is selected.
        This is used for ordered model-cache transactions whose successor may
        not execute before its predecessor commits.
        """

        started: list[ComputeJob] = []
        deferred: list[tuple[float, int, str]] = []
        try:
            for server in range(self.server_count):
                if self.running[server] is not None:
                    continue
                while self._waiting:
                    entry = heapq.heappop(self._waiting)
                    _, _, job_id = entry
                    job = self.jobs[job_id]
                    if job.status is not JobStatus.WAITING:
                        continue
                    if eligible is not None and not eligible(job):
                        deferred.append(entry)
                        continue
                    job.status = JobStatus.RUNNING
                    job.start_time_s = now_s
                    job.server_index = server
                    job.completion_version += 1
                    self.running[server] = job_id
                    started.append(job)
                    break
        finally:
            for entry in deferred:
                heapq.heappush(self._waiting, entry)
        self.max_running_observed = max(
            self.max_running_observed, sum(x is not None for x in self.running)
        )
        return started

    def advance(
        self,
        dt_s: float,
        effective_rate: float,
    ) -> list[tuple[ComputeJob, float, float]]:
        advanced: list[tuple[ComputeJob, float, float]] = []
        running_count = sum(job_id is not None for job_id in self.running)
        self.busy_server_seconds += running_count * dt_s
        for job_id in tuple(self.running):
            if job_id is None:
                continue
            job = self.jobs[job_id]
            served, energy = job.advance(dt_s, effective_rate)
            advanced.append((job, served, energy))
        return advanced

    def zero_residual_jobs(self) -> list[ComputeJob]:
        return [
            self.jobs[job_id]
            for job_id in self.running
            if job_id is not None and self.jobs[job_id].residual_work_s == 0.0
        ]

    def complete(
        self, job_id: str, now_s: float, version_token: int
    ) -> ComputeJob | None:
        job = self.jobs.get(job_id)
        if (
            job is None
            or job.status is not JobStatus.RUNNING
            or job.completion_version != version_token
        ):
            return None
        if job.residual_work_s > EPS:
            return None
        server = job.server_index
        if server is None or self.running[server] != job_id:
            raise InvariantViolation(
                "RESOURCE_SERVER_CORRUPT",
                "running job/server mapping is inconsistent",
                job_id=job_id,
            )
        job.residual_work_s = 0.0
        job.status = JobStatus.DONE
        job.end_time_s = now_s
        self.running[server] = None
        return job

    def cancel_task(self, task_id: str, now_s: float) -> list[ComputeJob]:
        cancelled: list[ComputeJob] = []
        for job in self.jobs.values():
            if job.task_id != task_id or job.status not in {
                JobStatus.WAITING,
                JobStatus.RUNNING,
            }:
                continue
            if job.status is JobStatus.RUNNING and job.server_index is not None:
                if self.running[job.server_index] == job.job_id:
                    self.running[job.server_index] = None
            job.status = JobStatus.CANCELLED
            job.end_time_s = now_s
            job.completion_version += 1
            cancelled.append(job)
        return cancelled

    @property
    def waiting_count(self) -> int:
        return sum(1 for job in self.jobs.values() if job.status is JobStatus.WAITING)

    @property
    def running_count(self) -> int:
        return sum(x is not None for x in self.running)

    @property
    def residual_work_s(self) -> float:
        return sum(
            job.residual_work_s
            for job in self.jobs.values()
            if job.status in {JobStatus.WAITING, JobStatus.RUNNING}
        )

    def active_jobs_for_task(self, task_id: str) -> tuple[ComputeJob, ...]:
        return tuple(
            job
            for job in self.jobs.values()
            if job.task_id == task_id
            and job.status in {JobStatus.WAITING, JobStatus.RUNNING}
        )


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    task_id: str
    descriptor_count: int
    vram_bytes: int
    conservative_work_gpu_s: float
    model_id: str
    model_hash: str
    protocol_version: str
    message_valid: bool

    def __post_init__(self) -> None:
        if not all(
            (self.task_id, self.model_id, self.model_hash, self.protocol_version)
        ):
            raise ValueError("admission identifiers and versions must be non-empty")
        if (
            isinstance(self.descriptor_count, bool)
            or not isinstance(self.descriptor_count, int)
            or self.descriptor_count < 1
        ):
            raise ValueError("admission descriptor_count must be an integer >= 1")
        if (
            isinstance(self.vram_bytes, bool)
            or not isinstance(self.vram_bytes, int)
            or self.vram_bytes < 1
        ):
            raise ValueError("admission vram_bytes must be an integer >= 1")
        if (
            not math.isfinite(self.conservative_work_gpu_s)
            or self.conservative_work_gpu_s <= 0
        ):
            raise ValueError(
                "admission conservative_work_gpu_s must be finite and positive"
            )
        if not isinstance(self.message_valid, bool):
            raise ValueError("admission message_valid must be boolean")


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    descriptors: int
    vram_bytes: int
    reserved_work_gpu_s: float
    reservations: tuple[tuple[str, int, int, float, str, str], ...]
    cached_models: tuple[tuple[str, str], ...]


class RSUAdmission:
    """Check-then-commit admission; a reject has exactly zero side effects."""

    def __init__(
        self,
        *,
        descriptor_capacity: int,
        vram_capacity_bytes: int,
        workload_capacity_gpu_s: float,
        protocol_version: str,
        cached_models: dict[str, str],
    ) -> None:
        if (
            isinstance(descriptor_capacity, bool)
            or not isinstance(descriptor_capacity, int)
            or descriptor_capacity < 1
            or isinstance(vram_capacity_bytes, bool)
            or not isinstance(vram_capacity_bytes, int)
            or vram_capacity_bytes < 1
            or not math.isfinite(workload_capacity_gpu_s)
            or workload_capacity_gpu_s <= 0
        ):
            raise ValueError("RSU admission capacities must be finite and positive")
        if not protocol_version or any(
            not key or not value for key, value in cached_models.items()
        ):
            raise ValueError(
                "RSU protocol and cached model identities must be non-empty"
            )
        self.descriptor_capacity = descriptor_capacity
        self.vram_capacity_bytes = vram_capacity_bytes
        self.workload_capacity_gpu_s = workload_capacity_gpu_s
        self.protocol_version = protocol_version
        self.cached_models = dict(cached_models)
        self._reservations: dict[str, AdmissionRequest] = {}
        self.descriptors = 0
        self.vram_bytes = 0
        self.reserved_work_gpu_s = 0.0

    def snapshot(self) -> AdmissionSnapshot:
        rows = tuple(
            sorted(
                (
                    task_id,
                    req.descriptor_count,
                    req.vram_bytes,
                    req.conservative_work_gpu_s,
                    req.model_id,
                    req.model_hash,
                )
                for task_id, req in self._reservations.items()
            )
        )
        return AdmissionSnapshot(
            self.descriptors,
            self.vram_bytes,
            self.reserved_work_gpu_s,
            rows,
            tuple(sorted(self.cached_models.items())),
        )

    def can_admit(self, req: AdmissionRequest) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if not req.message_valid:
            reasons.append("MESSAGE_EVIDENCE_MISSING")
        if req.protocol_version != self.protocol_version:
            reasons.append("PROTOCOL_MISMATCH")
        if self.cached_models.get(req.model_id) != req.model_hash:
            reasons.append("MODEL_CACHE_MISSING")
        if req.task_id in self._reservations:
            reasons.append("DUPLICATE_ADMISSION")
        if self.descriptors + req.descriptor_count > self.descriptor_capacity:
            reasons.append("RSU_DESCRIPTOR_CAPACITY")
        if self.vram_bytes + req.vram_bytes > self.vram_capacity_bytes:
            reasons.append("RSU_VRAM_CAPACITY")
        if (
            self.reserved_work_gpu_s + req.conservative_work_gpu_s
            > self.workload_capacity_gpu_s + EPS
        ):
            reasons.append("RSU_WORKLOAD_CAPACITY")
        return not reasons, tuple(reasons)

    def admit(self, req: AdmissionRequest) -> tuple[bool, tuple[str, ...]]:
        accepted, reasons = self.can_admit(req)
        if not accepted:
            return False, reasons
        self._reservations[req.task_id] = req
        self.descriptors += req.descriptor_count
        self.vram_bytes += req.vram_bytes
        self.reserved_work_gpu_s += req.conservative_work_gpu_s
        self.assert_within_capacity()
        return True, ()

    def release(self, task_id: str) -> bool:
        req = self._reservations.pop(task_id, None)
        if req is None:
            return False
        self.descriptors -= req.descriptor_count
        self.vram_bytes -= req.vram_bytes
        self.reserved_work_gpu_s = max(
            0.0, self.reserved_work_gpu_s - req.conservative_work_gpu_s
        )
        return True

    def pinned_model(self, task_id: str) -> tuple[str, str] | None:
        req = self._reservations.get(task_id)
        return None if req is None else (req.model_id, req.model_hash)

    def reservation(self, task_id: str) -> AdmissionRequest | None:
        """Return the immutable atomic request held for one admitted task."""

        return self._reservations.get(task_id)

    def update_cache(self, cached_models: dict[str, str]) -> None:
        self.cached_models = dict(cached_models)

    def assert_within_capacity(self) -> None:
        if not (0 <= self.descriptors <= self.descriptor_capacity):
            raise InvariantViolation(
                "RSU_DESCRIPTOR_OVERFLOW", "RSU descriptor capacity exceeded"
            )
        if not (0 <= self.vram_bytes <= self.vram_capacity_bytes):
            raise InvariantViolation("RSU_VRAM_OVERFLOW", "RSU VRAM capacity exceeded")
        if not (0 <= self.reserved_work_gpu_s <= self.workload_capacity_gpu_s + EPS):
            raise InvariantViolation(
                "RSU_WORKLOAD_OVERFLOW", "RSU workload reservation exceeded"
            )
