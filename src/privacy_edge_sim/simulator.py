"""Continuous-real-time discrete-event simulator.

The clock jumps directly to the next event.  Every timestamp is processed as
one compound event with completion before faults/versions, deadline, arrival,
dispatch and decision.  There is no fixed-step polling loop.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .config import SimulationConfig, load_config
from .enums import (
    ActionKind,
    ActionStage,
    EventKind,
    EventPriority,
    FailureReason,
    JobStatus,
    Operation,
    ResourceKind,
    TaskState,
    TransferDirection,
    TransferStatus,
)
from .errors import ConfigError, InvariantViolation, TransitionError
from .estimation import FrozenEstimateProvider, encode_context
from .events import Event, EventQueue, _strict_future_instant
from .invariants import assert_all_invariants, assert_observation_safe
from .metrics import MetricLedger
from .packets import (
    AnonFERRequest,
    FERResult,
    RawImageHandle,
    AlignedTensorHandle,
    _finalize_encoded_anon,
    _replay_anonymization_success,
    _replay_encoding_success,
    _replay_guard_success,
)
from .profiles import FrozenProfileBundle, canonical_json_bytes, load_profile
from .resources import AdmissionRequest, ComputeJob, RSUAdmission, ResourcePool
from .safety import (
    Action,
    DeterministicRepair,
    HardMaskEngine,
    Observation,
    ObservationBuilder,
    _snapshot_task_tokens,
)
from .state import (
    RSURuntime,
    SimulationState,
    TaskRecord,
    TaskStateMachine,
    Transfer,
    VehicleRuntime,
    VirtualQueueBank,
)
from .traces import (
    AnonTraceRow,
    DeviceContext,
    EdgeFERTraceRow,
    ExogenousEvent,
    LocalFERTraceRow,
    PrepTraceRow,
    ScenarioLibrary,
    ThermalSegment,
    TraceBundle,
    WirelessSegment,
    load_trace,
)


EPS = 1e-10


def _strict_future_completion_time(
    now_s: float, remaining_service: float, service_rate: float
) -> float:
    """Return a representable completion timestamp strictly after ``now_s``.

    At large absolute timestamps, a positive residual divided by a high rate
    can be smaller than half an IEEE-754 ulp.  Plain ``now + residual/rate``
    then rounds back to ``now`` and creates an infinite same-time completion
    loop.  One ulp is the smallest representable physical-time advance and its
    service integral is at least the requested residual in that case.
    """

    if remaining_service <= 0 or service_rate <= 0:
        raise ValueError("remaining service and service rate must be positive")
    return _strict_future_instant(now_s, remaining_service / service_rate)


@dataclass(frozen=True, slots=True)
class RunResult:
    state: SimulationState
    profile: FrozenProfileBundle
    trace: TraceBundle
    scenario_trace: TraceBundle
    config: SimulationConfig
    policy_name: str
    invariant_failures: tuple[dict[str, Any], ...]
    ledger: MetricLedger


@dataclass(frozen=True, slots=True)
class PolicyTaskView:
    """Minimal task facts exposed to a controller implementation."""

    task_id: str
    vehicle_id: str
    selected_pipeline: str | None
    ul_remaining_bits: float
    dl_remaining_bits: float
    current_transfer_id: str | None


@dataclass(frozen=True, slots=True)
class PolicyStateView:
    """Clock-only state view; all usable system facts live in Observation."""

    clock_s: float


class DiscreteEventSimulator:
    """One isolated simulator instance for one policy and one frozen trace."""

    def __init__(
        self,
        config: SimulationConfig,
        profile: FrozenProfileBundle,
        trace: TraceBundle,
        policy: Any,
        *,
        policy_name: str | None = None,
        scenario_trace: TraceBundle | None = None,
    ) -> None:
        if profile.profile_hash != trace.profile_hash:
            raise ConfigError(
                "PROFILE_TRACE_HASH_MISMATCH",
                "trace was not validated against the loaded frozen profile",
                profile_hash=profile.profile_hash,
                trace_profile_hash=trace.profile_hash,
            )
        if (
            config.protocol_version != profile.protocol_version
            or trace.protocol_version != profile.protocol_version
        ):
            raise ConfigError(
                "PROTOCOL_VERSION_MISMATCH",
                "config, profile and trace protocols must match",
            )
        if profile.online_mutable:
            raise ConfigError(
                "ONLINE_PROFILE_MUTABLE", "online runs require an immutable profile"
            )
        if scenario_trace is None:
            scenario_trace = load_trace(config.scenario_trace_path, profile)
        if scenario_trace.profile_hash != profile.profile_hash:
            raise ConfigError(
                "SCENARIO_PROFILE_HASH_MISMATCH",
                "controller scenario trace must use the loaded frozen profile",
                scenario_profile_hash=scenario_trace.profile_hash,
                profile_hash=profile.profile_hash,
            )
        if scenario_trace.protocol_version != profile.protocol_version:
            raise ConfigError(
                "SCENARIO_PROTOCOL_MISMATCH",
                "controller scenario trace protocol must match the frozen profile",
            )
        if scenario_trace.source_path.resolve() == trace.source_path.resolve():
            raise ConfigError(
                "SCENARIO_EVALUATION_OVERLAP",
                "online runs require a separate training/validation scenario trace",
            )
        if scenario_trace.data_kind != trace.data_kind:
            raise ConfigError(
                "SCENARIO_EVALUATION_KIND_MISMATCH",
                "evaluation and controller scenario traces must use the same evidence kind",
                evaluation_data_kind=trace.data_kind,
                scenario_data_kind=scenario_trace.data_kind,
            )
        scenario_split = scenario_trace.metadata.get("data_split", {})
        evaluation_split = trace.metadata.get("data_split", {})
        scenario_role = (
            str(scenario_split.get("role", ""))
            if isinstance(scenario_split, Mapping)
            else ""
        )
        evaluation_role = (
            str(evaluation_split.get("role", ""))
            if isinstance(evaluation_split, Mapping)
            else ""
        )
        if scenario_role not in {"training", "validation", "training_validation"}:
            raise ConfigError(
                "SCENARIO_SPLIT_UNDECLARED",
                "scenario trace must declare a training/validation split role",
                role=scenario_role,
            )
        if evaluation_role not in {"evaluation", "test"}:
            raise ConfigError(
                "EVALUATION_SPLIT_UNDECLARED",
                "evaluation trace must declare an evaluation/test split role",
                role=evaluation_role,
            )
        self.config = config
        self.profile = profile
        self.trace = trace
        self.scenario_trace = scenario_trace
        self.scenario_library = ScenarioLibrary.from_trace(
            scenario_trace,
            rsu_snapshot_period_s=config.rsu_snapshot_period_s,
            rsu_telemetry_delay_s=config.rsu_telemetry_delay_s,
            rsu_telemetry_quantum_work_s=config.rsu_telemetry_quantum_work_s,
            rsu_telemetry_drop_every=config.rsu_telemetry_drop_every,
            metadata_bits=config.metadata_bits,
            uplink_pause_limit_s=config.uplink_pause_limit_s,
            downlink_pause_limit_s=config.downlink_pause_limit_s,
            vehicle_anchor_parameters={
                row.vehicle_id: {
                    "device_type": row.device_type,
                    "initial_battery_j": row.initial_battery_j,
                    "battery_capacity_j": row.battery_capacity_j,
                    "memory_capacity_bytes": row.memory_capacity_bytes,
                    "idle_power_w": row.idle_power_w,
                    "hold_power_w": row.hold_power_w,
                    "controller_overhead_s": config.controller.controller_overhead_s,
                    "controller_energy_j": config.controller.controller_energy_j,
                    "descriptor_capacity": {
                        "accelerator": row.accelerator_descriptors,
                        "cpu": row.cpu_descriptors,
                        "encoder": row.encoder_descriptors,
                    },
                    "server_count": {
                        "accelerator": 1,
                        "cpu": 1,
                        "encoder": 1,
                    },
                }
                for row in config.vehicles
            },
            rsu_anchor_parameters={
                row.rsu_id: {
                    "descriptor_capacity": row.descriptor_capacity,
                    "vram_capacity_bytes": row.vram_capacity_bytes,
                    "workload_capacity_gpu_s": row.workload_capacity_gpu_s,
                    "ingress_servers": 1,
                    "gpu_servers": row.gpu_servers,
                    "idle_power_w": row.idle_power_w,
                    "hold_power_w": row.hold_power_w,
                    "cached_models": {
                        model_id: profile.edge_models[model_id].model_hash
                        for model_id in row.cached_models
                    },
                }
                for row in config.rsus
            },
        )
        evaluation_edge_support = frozenset(
            (row.rsu_id, row.model_id, row.pipeline_id, row.artifact_key)
            for row in trace.edge_rows
        )
        # This is the only online index that retains evaluation artifact keys.
        # It belongs to the simulator trusted domain and is never attached to
        # the policy, HardMaskEngine, estimate provider or Observation.
        self._evaluation_edge_support = evaluation_edge_support
        self.estimator = FrozenEstimateProvider(
            profile,
            self.scenario_library,
            config,
            requires_evaluation_pair=True,
        )
        self.mask_engine = HardMaskEngine(profile, config, trace_support=self.estimator)
        self.repairer = DeterministicRepair(self.mask_engine)
        if isinstance(policy, str):
            from .policies import POLICY_REGISTRY

            if policy not in POLICY_REGISTRY:
                raise ConfigError(
                    "POLICY_UNKNOWN", "unknown configured policy", policy=policy
                )
            policy_type = POLICY_REGISTRY[policy]
            kwargs: dict[str, Any] = {}
            if policy in {"safe_lyapunov_h1", "esl_smpc"}:
                kwargs["scenario_source"] = self.scenario_library
            self.policy = policy_type(self.mask_engine, self.repairer, **kwargs)
        else:
            self.policy = policy
        self.policy_name = policy_name or getattr(
            self.policy, "name", type(self.policy).__name__
        )
        self.state = self._initial_state()
        self.ledger = MetricLedger(simulation_start_s=trace.horizon_start_s)
        self._arrivals = {row.task_id: row for row in trace.arrivals}
        self._decision_score_buffer: dict[str, Mapping[Action, float]] = {}
        self._pending_decisions: dict[str, Action] = {}
        self._pending_decision_scores: dict[str, Mapping[Action, float]] = {}
        self._pending_decision_due_s: dict[str, float] = {}
        self._anon_rows: dict[str, AnonTraceRow] = {}
        self._prep_rows: dict[str, PrepTraceRow] = {}
        self._local_rows: dict[str, LocalFERTraceRow] = {}
        self._edge_rows: dict[str, EdgeFERTraceRow] = {}
        self._link_versions: dict[tuple[str, str, str], int] = {}
        self._battery_versions: dict[str, int] = {
            vehicle_id: 0 for vehicle_id in self.state.vehicles
        }
        self._telemetry_sample_sequence: dict[str, int] = {
            rsu_id: 0 for rsu_id in self.state.rsus
        }
        self._job_counter = 0
        self._transfer_counter = 0
        self._rsu_maintenance_events: dict[str, ExogenousEvent] = {}
        self._rsu_maintenance_job_keys: dict[str, tuple[str, str]] = {}
        self._rsu_maintenance_active: dict[tuple[str, str], str] = {}
        self._rsu_maintenance_waiting: dict[tuple[str, str], list[str]] = {}
        self._invariant_failures: list[dict[str, Any]] = []
        self._replay_hasher = hashlib.sha256()
        self._compound_events = 0
        self._initial_profile_hash = profile.profile_hash
        self._active_profile_hash = profile.profile_hash
        self._active_protocol_version = config.protocol_version
        self._active_local_model_hashes = {
            (vehicle_id, model.model_id): model.model_hash
            for vehicle_id, vehicle in self.state.vehicles.items()
            for model in profile.local_models.values()
            if vehicle.device_type in model.supported_devices
        }
        self._schedule_external_events()
        # Battery depletion is itself a physical event.  It must be present
        # before the first arrival/link event so idle draw cannot cross zero
        # while the event loop advances to its first externally supplied time.
        self._schedule_completions_and_battery_guards()

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        policy: Any,
        *,
        policy_name: str | None = None,
    ) -> "DiscreteEventSimulator":
        config = load_config(config_path)
        profile = load_profile(config.profile_path)
        trace = load_trace(config.trace_path, profile)
        scenario_trace = load_trace(config.scenario_trace_path, profile)
        return cls(
            config,
            profile,
            trace,
            policy,
            policy_name=policy_name,
            scenario_trace=scenario_trace,
        )

    def _initial_state(self) -> SimulationState:
        events = EventQueue()
        vehicles: dict[str, VehicleRuntime] = {}
        for row in self.config.vehicles:
            pools = {
                "accelerator": ResourcePool(
                    f"{row.vehicle_id}:accelerator", ResourceKind.ACCELERATOR, 1
                ),
                "cpu": ResourcePool(f"{row.vehicle_id}:cpu", ResourceKind.CPU, 1),
                "encoder": ResourcePool(
                    f"{row.vehicle_id}:encoder", ResourceKind.ENCODER, 1
                ),
            }
            vehicles[row.vehicle_id] = VehicleRuntime(
                row.vehicle_id,
                row.device_type,
                row.battery_capacity_j,
                row.initial_battery_j,
                row.memory_capacity_bytes,
                0,
                {
                    "accelerator": row.accelerator_descriptors,
                    "cpu": row.cpu_descriptors,
                    "encoder": row.encoder_descriptors,
                },
                {"accelerator": 0, "cpu": 0, "encoder": 0},
                pools,
                row.idle_power_w,
                hold_power_w=row.hold_power_w,
            )
        rsus: dict[str, RSURuntime] = {}
        for row in self.config.rsus:
            cached: dict[str, str] = {}
            for model_id in row.cached_models:
                model = self.profile.edge_models.get(model_id)
                if model is None or row.rsu_id not in model.supported_rsus:
                    raise ConfigError(
                        "RSU_MODEL_CACHE_CONFIG",
                        "configured RSU cache contains unsupported model",
                        rsu_id=row.rsu_id,
                        model_id=model_id,
                    )
                cached[model_id] = model.model_hash
            rsus[row.rsu_id] = RSURuntime(
                row.rsu_id,
                RSUAdmission(
                    descriptor_capacity=row.descriptor_capacity,
                    vram_capacity_bytes=row.vram_capacity_bytes,
                    workload_capacity_gpu_s=row.workload_capacity_gpu_s,
                    protocol_version=self.config.protocol_version,
                    cached_models=cached,
                ),
                ResourcePool(f"{row.rsu_id}:ingress", ResourceKind.RSU_INGRESS_CPU, 1),
                ResourcePool(
                    f"{row.rsu_id}:gpu", ResourceKind.RSU_GPU, row.gpu_servers
                ),
                row.idle_power_w,
                hold_power_w=row.hold_power_w,
            )
            runtime = rsus[row.rsu_id]
            runtime.current_snapshot_time_s = self.trace.horizon_start_s
            runtime.public_snapshot = self._rsu_public_snapshot(
                runtime,
                row.rsu_id,
                self.trace.horizon_start_s,
            )
        queues = VirtualQueueBank(
            {vehicle_id: 0.0 for vehicle_id in vehicles},
            {rsu_id: 0.0 for rsu_id in rsus},
        )
        return SimulationState(
            clock_s=self.trace.horizon_start_s,
            events=events,
            tasks={},
            vehicles=vehicles,
            rsus=rsus,
            transfers={},
            virtual_queues=queues,
        )

    def _rsu_public_snapshot(
        self,
        runtime: RSURuntime,
        rsu_id: str,
        time_s: float,
    ) -> dict[str, Any]:
        admission = runtime.admission.snapshot()

        def remaining_dynamic_energy(pool: ResourcePool) -> float:
            return sum(
                job.total_dynamic_energy_j * job.residual_work_s / job.total_work_s
                for job in pool.jobs.values()
                if job.status in {JobStatus.WAITING, JobStatus.RUNNING}
            )

        return {
            "device_context": encode_context(
                self._device_context("rsu", rsu_id, time_s)
            ),
            "failed": runtime.failed,
            "descriptors": admission.descriptors,
            "descriptor_capacity": runtime.admission.descriptor_capacity,
            "vram_bytes": admission.vram_bytes,
            "vram_capacity_bytes": runtime.admission.vram_capacity_bytes,
            "reserved_work_gpu_s": admission.reserved_work_gpu_s,
            "hold_participant_count": len(admission.reservations),
            "workload_capacity_gpu_s": runtime.admission.workload_capacity_gpu_s,
            "cached_models": dict(admission.cached_models),
            "ingress_waiting": runtime.ingress.waiting_count,
            "ingress_running": runtime.ingress.running_count,
            "ingress_residual_work_s": runtime.ingress.residual_work_s,
            "ingress_remaining_dynamic_energy_j": remaining_dynamic_energy(
                runtime.ingress
            ),
            "gpu_waiting": runtime.gpu.waiting_count,
            "gpu_running": runtime.gpu.running_count,
            "gpu_servers": runtime.gpu.server_count,
            "gpu_residual_work_s": runtime.gpu.residual_work_s,
            "gpu_remaining_dynamic_energy_j": remaining_dynamic_energy(runtime.gpu),
        }

    def _schedule_external_events(self) -> None:
        for arrival in self.trace.arrivals:
            if arrival.vehicle_id not in self.state.vehicles:
                raise ConfigError(
                    "TRACE_VEHICLE_UNKNOWN",
                    "arrival references vehicle absent from config",
                    task_id=arrival.task_id,
                    vehicle_id=arrival.vehicle_id,
                )
            self.state.events.push(
                arrival.arrival_time_s, EventKind.ARRIVAL, task_id=arrival.task_id
            )
            self.state.events.push(
                arrival.absolute_deadline_s, EventKind.DEADLINE, task_id=arrival.task_id
            )
        boundary_keys: set[tuple[float, str, str, str]] = set()
        for segment in self.trace.wireless:
            for time_s in (segment.start_time_s, segment.end_time_s):
                if time_s <= self.state.clock_s or time_s > self.trace.horizon_end_s:
                    continue
                key = (
                    time_s,
                    segment.vehicle_id,
                    segment.rsu_id,
                    segment.direction.value,
                )
                if key in boundary_keys:
                    continue
                boundary_keys.add(key)
                self.state.events.push(
                    time_s,
                    EventKind.LINK_CHANGE,
                    object_id=f"{segment.vehicle_id}|{segment.rsu_id}|{segment.direction.value}",
                    payload={
                        "boundary": True,
                        "vehicle_id": segment.vehicle_id,
                        "rsu_id": segment.rsu_id,
                        "direction": segment.direction.value,
                    },
                )
        thermal_keys: set[tuple[float, str, str, str]] = set()
        for segment in self.trace.thermal:
            for time_s in (segment.start_time_s, segment.end_time_s):
                if time_s <= self.state.clock_s or time_s > self.trace.horizon_end_s:
                    continue
                key = (time_s, segment.owner_type, segment.owner_id, segment.resource)
                if key in thermal_keys:
                    continue
                thermal_keys.add(key)
                self.state.events.push(
                    time_s,
                    EventKind.THERMAL_CHANGE,
                    object_id=f"{segment.owner_type}|{segment.owner_id}|{segment.resource}",
                    payload={
                        "owner_type": segment.owner_type,
                        "owner_id": segment.owner_id,
                        "resource": segment.resource,
                    },
                )
        for row in self.trace.exogenous_events:
            kind = {
                "DEVICE_FAULT_START": EventKind.DEVICE_FAULT,
                "DEVICE_FAULT_END": EventKind.DEVICE_FAULT,
                "DEVICE_FAULT_PERMANENT": EventKind.DEVICE_FAULT,
                "LINK_CHANGE": EventKind.LINK_CHANGE,
                "MODEL_VERSION": EventKind.MODEL_VERSION,
                "PROFILE_VERSION": EventKind.PROFILE_VERSION,
                "PROTOCOL_VERSION": EventKind.PROFILE_VERSION,
                "MODEL_CACHE": EventKind.MODEL_VERSION,
            }[row.event_type]
            self.state.events.push(
                row.time_s, kind, object_id=row.event_id, payload=row
            )
        snapshot_index = 1
        while True:
            snapshot_time = (
                self.trace.horizon_start_s
                + snapshot_index * self.config.rsu_snapshot_period_s
            )
            if snapshot_time > self.trace.horizon_end_s + EPS:
                break
            for rsu_id in sorted(self.state.rsus):
                self.state.events.push(
                    snapshot_time,
                    EventKind.RSU_SNAPSHOT,
                    object_id=rsu_id,
                )
            snapshot_index += 1

    def _rng(self, purpose: str, task_id: str, conditional_key: str) -> random.Random:
        stream_name = {
            "quality": "arrivals",
            "prep": "vehicle",
            "local": "vehicle",
            "anon": "vehicle",
            "edge": "rsu",
        }.get(purpose, "environment")
        material = f"{self.config.seeds[stream_name]}|{stream_name}|{purpose}|{task_id}|{conditional_key}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(seed)

    def _artifact_capabilities(
        self, artifact_key: str | None
    ) -> frozenset[tuple[str, str, str]]:
        """Project a trusted evaluation key into a non-identifying capability."""

        if artifact_key is None:
            return frozenset()
        return frozenset(
            (rsu_id, model_id, pipeline_id)
            for rsu_id, model_id, pipeline_id, candidate_key in self._evaluation_edge_support
            if candidate_key == artifact_key
        )

    def _runtime_evaluation_pair_supported(
        self,
        task: TaskRecord,
        *,
        rsu_id: str,
        model_id: str,
        pipeline_id: str,
    ) -> bool:
        """Trusted atomic check against the actual received artifact."""

        return (
            task.artifact_key is not None
            and (
                rsu_id,
                model_id,
                pipeline_id,
                task.artifact_key,
            )
            in self._evaluation_edge_support
        )

    def _thermal_segment(
        self, owner_type: str, owner_id: str, resource: str, time_s: float
    ) -> ThermalSegment | None:
        candidates = [
            row
            for row in self.trace.thermal
            if row.owner_type == owner_type
            and row.owner_id == owner_id
            and row.resource in {resource, "all"}
            and row.start_time_s <= time_s < row.end_time_s
        ]
        if not candidates and math.isclose(
            time_s, self.trace.horizon_end_s, abs_tol=EPS
        ):
            candidates = [
                row
                for row in self.trace.thermal
                if row.owner_type == owner_type
                and row.owner_id == owner_id
                and row.resource in {resource, "all"}
                and math.isclose(row.end_time_s, time_s, abs_tol=EPS)
            ]
        exact = [row for row in candidates if row.resource == resource]
        rows = exact or candidates
        return max(rows, key=lambda row: row.start_time_s) if rows else None

    def _device_context(
        self, owner_type: str, owner_id: str, time_s: float
    ) -> DeviceContext:
        segment = self._thermal_segment(owner_type, owner_id, "all", time_s)
        state = "nominal" if segment is None else segment.state
        return DeviceContext(state, "nominal", "normal")

    def _resource_rate(
        self, owner_type: str, owner_id: str, resource: str, time_s: float
    ) -> float:
        segment = self._thermal_segment(owner_type, owner_id, resource, time_s)
        return 1.0 if segment is None else segment.service_rate_multiplier

    def _remaining_job_energy_upper(
        self,
        job: ComputeJob,
        resource: str,
        now_s: float,
    ) -> float:
        """Return paired dynamic energy remaining for residual busy-work."""

        return job.total_dynamic_energy_j * job.residual_work_s / job.total_work_s

    def _wireless_segment(
        self, vehicle_id: str, rsu_id: str, direction: TransferDirection, time_s: float
    ) -> WirelessSegment | None:
        rows = [
            row
            for row in self.trace.wireless
            if row.vehicle_id == vehicle_id
            and row.rsu_id == rsu_id
            and row.direction is direction
            and row.start_time_s <= time_s < row.end_time_s
        ]
        return rows[0] if rows else None

    @staticmethod
    def _transfer_link_key(
        transfer: Transfer,
    ) -> tuple[str, str, TransferDirection]:
        """Return the aggregate net-service process shared by packets.

        A wireless trace row is a link-level application-payload service
        process, not a fresh copy of capacity for every packet.  Concurrent
        packets on the same vehicle/RSU/direction therefore time-share that
        row deterministically.  Different vehicles keep their separately
        measured/simulated link processes.
        """

        return transfer.vehicle_id, transfer.rsu_id, transfer.direction

    def _active_link_counts(
        self, time_s: float
    ) -> dict[tuple[str, str, TransferDirection], int]:
        """Count packets sharing a link-level service/power process."""

        counts: dict[tuple[str, str, TransferDirection], int] = {}
        for transfer in self.state.transfers.values():
            if transfer.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}:
                continue
            segment = self._wireless_segment(
                transfer.vehicle_id, transfer.rsu_id, transfer.direction, time_s
            )
            if segment is None or segment.link_state not in {
                "connected",
                "temporary_outage",
            }:
                continue
            if segment.link_state == "connected" and (
                transfer.status is not TransferStatus.ACTIVE or segment.goodput_bps <= 0
            ):
                continue
            key = self._transfer_link_key(transfer)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _transfer_service(
        self,
        transfer: Transfer,
        time_s: float,
        active_counts: Mapping[tuple[str, str, TransferDirection], int],
    ) -> tuple[float, float, float]:
        """Return this packet's goodput and paired endpoint powers.

        Connected packets use equal deterministic time sharing.  Both
        goodput and active radio power are divided by the same share, so the
        aggregate service and aggregate paired energy cannot be duplicated.
        Paused packets receive zero service and split the trace's link-level
        outage/listen power, preserving attribution without cloning physical
        system energy.
        """

        segment = self._wireless_segment(
            transfer.vehicle_id, transfer.rsu_id, transfer.direction, time_s
        )
        if segment is None:
            return 0.0, 0.0, 0.0
        count = active_counts.get(self._transfer_link_key(transfer), 0)
        if count < 1:
            return 0.0, 0.0, 0.0
        share = 1.0 / count
        if (
            transfer.status is TransferStatus.PAUSED
            or segment.link_state == "temporary_outage"
        ):
            return (
                0.0,
                segment.transmitter_power_w * share,
                segment.receiver_power_w * share,
            )
        if (
            transfer.status is not TransferStatus.ACTIVE
            or segment.link_state != "connected"
        ):
            return 0.0, 0.0, 0.0
        return (
            segment.goodput_bps * share,
            segment.transmitter_power_w * share,
            segment.receiver_power_w * share,
        )

    def _next_thermal_boundary(
        self, owner_type: str, owner_id: str, resource: str, now_s: float
    ) -> float:
        values = [
            value
            for row in self.trace.thermal
            if row.owner_type == owner_type
            and row.owner_id == owner_id
            and row.resource in {resource, "all"}
            for value in (row.start_time_s, row.end_time_s)
            if value > now_s + EPS
        ]
        return min(values) if values else math.inf

    def _next_link_boundary(self, transfer: Transfer, now_s: float) -> float:
        values = [
            value
            for row in self.trace.wireless
            if row.vehicle_id == transfer.vehicle_id
            and row.rsu_id == transfer.rsu_id
            and row.direction is transfer.direction
            for value in (row.start_time_s, row.end_time_s)
            if value > now_s + EPS
        ]
        return min(values) if values else math.inf

    def _update_replay_prefix(self) -> str:
        """Hash one stable, raw-free compound-event state snapshot."""

        resource_tail = [
            row
            for row in self.ledger.resource_rows
            if abs(float(row["time_s"]) - self.state.clock_s) <= EPS
        ]
        virtual_tail = [
            row
            for row in self.ledger.virtual_queue_rows
            if abs(float(row["time_s"]) - self.state.clock_s) <= EPS
        ]
        payload = {
            "compound_event": self.state.event_log[-1],
            "tasks": MetricLedger.task_rows(self.state, self.config),
            "resources": resource_tail,
            "virtual_queues": virtual_tail,
        }
        self._replay_hasher.update(canonical_json_bytes(payload))
        self._compound_events += 1
        return self._replay_hasher.hexdigest()

    def run(
        self,
        *,
        replay_checkpoint: Mapping[str, Any] | None = None,
        checkpoint_callback: Callable[[int, float, str, bool], None] | None = None,
    ) -> RunResult:
        previous_clock = self.state.clock_s
        replay_count = (
            0
            if replay_checkpoint is None
            else int(replay_checkpoint["compound_events"])
        )
        replay_digest = (
            None
            if replay_checkpoint is None
            else str(replay_checkpoint["prefix_sha256"])
        )
        replay_clock = (
            None if replay_checkpoint is None else float(replay_checkpoint["clock_s"])
        )
        replay_verified = replay_checkpoint is None
        if replay_count == 0 and replay_checkpoint is not None:
            empty_digest = self._replay_hasher.hexdigest()
            if (
                replay_digest != empty_digest
                or abs((replay_clock or 0.0) - self.state.clock_s) > EPS
            ):
                raise ConfigError(
                    "REPLAY_CHECKPOINT_PREFIX",
                    "zero-event replay checkpoint does not match simulator origin",
                )
            replay_verified = True
        try:
            while len(self.state.events):
                time_s, batch = self.state.events.pop_compound(
                    current_time_s=self.state.clock_s
                )
                self._advance_to(time_s)
                failures_before = sum(
                    task.state is TaskState.FAIL for task in self.state.tasks.values()
                )
                done_before = sum(
                    task.state is TaskState.DONE for task in self.state.tasks.values()
                )
                arrivals, timeouts = self._process_compound(time_s, batch)
                self._dispatch_all()
                self._make_decisions()
                # Decisions with zero configured overhead can enqueue work at
                # this same timestamp; close dispatch locally without polling.
                self._dispatch_all()
                failures_after = sum(
                    task.state is TaskState.FAIL for task in self.state.tasks.values()
                )
                done_after = sum(
                    task.state is TaskState.DONE for task in self.state.tasks.values()
                )
                # One virtual-queue update closes the entire compound event,
                # including dispatch guards and zero-overhead decisions.  The
                # before/after delta makes every new terminal outcome visible
                # exactly once and retains the interval energy integrated by
                # _advance_to above.
                self._update_virtual_queues(
                    arrivals=arrivals,
                    timeouts=timeouts,
                    failures=max(0, failures_after - failures_before),
                    completed=max(0, done_after - done_before),
                )
                self._schedule_completions_and_battery_guards()
                assert_all_invariants(
                    self.state, self.profile, previous_clock_s=previous_clock
                )
                previous_clock = self.state.clock_s
                self._log_compound(batch)
                self.ledger.snapshot(self.state, batch)
                prefix_digest = self._update_replay_prefix()
                if (
                    self._compound_events == replay_count
                    and replay_checkpoint is not None
                ):
                    if (
                        prefix_digest != replay_digest
                        or replay_clock is None
                        or abs(replay_clock - self.state.clock_s) > EPS
                    ):
                        raise InvariantViolation(
                            "REPLAY_CHECKPOINT_DIVERGED",
                            "deterministic replay prefix differs from checkpoint",
                            expected_events=replay_count,
                            expected_clock_s=replay_clock,
                            actual_clock_s=self.state.clock_s,
                            expected_digest=replay_digest,
                            actual_digest=prefix_digest,
                        )
                    replay_verified = True
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        self._compound_events,
                        self.state.clock_s,
                        prefix_digest,
                        False,
                    )
                if (
                    self.state.tasks
                    and len(self.state.tasks) == len(self.trace.arrivals)
                    and all(task.terminal for task in self.state.tasks.values())
                ):
                    break
            if not replay_verified:
                raise InvariantViolation(
                    "REPLAY_CHECKPOINT_UNREACHABLE",
                    "run ended before the checkpoint compound-event boundary",
                    expected_events=replay_count,
                    actual_events=self._compound_events,
                )
            if len(self.state.tasks) != len(self.trace.arrivals) or any(
                not task.terminal for task in self.state.tasks.values()
            ):
                raise InvariantViolation(
                    "RUN_NOT_DRAINED",
                    "finite trace ended before every arrival reached DONE/FAIL",
                    arrived=len(self.state.tasks),
                    expected=len(self.trace.arrivals),
                    nonterminal=[
                        task.task_id
                        for task in self.state.tasks.values()
                        if not task.terminal
                    ],
                )
            if self.profile.profile_hash != self._initial_profile_hash:
                raise InvariantViolation(
                    "ONLINE_PROFILE_MUTATED", "frozen profile hash changed during run"
                )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    self._compound_events,
                    self.state.clock_s,
                    self._replay_hasher.hexdigest(),
                    True,
                )
        except (InvariantViolation, TransitionError) as exc:
            self._invariant_failures.append(
                {
                    "time_s": self.state.clock_s,
                    "code": exc.detail.code,
                    "message": exc.detail.message,
                    "context": exc.detail.context,
                    "recent_events": self.state.event_log[-20:],
                }
            )
            if isinstance(exc, TransitionError):
                raise InvariantViolation(
                    "STATE_TRANSITION_FATAL",
                    "state-machine transition failed during simulation",
                    cause_code=exc.detail.code,
                    cause_message=exc.detail.message,
                    cause_context=exc.detail.context,
                    time_s=self.state.clock_s,
                    recent_events=self.state.event_log[-20:],
                ) from exc
            raise
        return RunResult(
            self.state,
            self.profile,
            self.trace,
            self.scenario_trace,
            self.config,
            self.policy_name,
            tuple(self._invariant_failures),
            self.ledger,
        )

    def _advance_to(self, time_s: float) -> None:
        dt_s = time_s - self.state.clock_s
        if dt_s < -EPS:
            raise InvariantViolation(
                "TIME_REGRESSION", "advance_to received a past timestamp"
            )
        if dt_s == 0.0:
            self.state.clock_s = time_s
            self.state.last_interval_vehicle_energy = {
                key: 0.0 for key in self.state.vehicles
            }
            self.state.last_interval_rsu_energy = {key: 0.0 for key in self.state.rsus}
            return
        start_s = self.state.clock_s
        vehicle_interval = {key: 0.0 for key in self.state.vehicles}
        rsu_interval = {key: 0.0 for key in self.state.rsus}
        for vehicle in self.state.vehicles.values():
            baseline = 0.0 if vehicle.failed else vehicle.idle_power_w * dt_s
            vehicle.physical_energy_j += baseline
            vehicle.battery_j -= baseline
            vehicle_interval[vehicle.vehicle_id] += baseline
            for name, pool in vehicle.resources.items():
                rate = self._resource_rate("vehicle", vehicle.vehicle_id, name, start_s)
                for job, _, energy in pool.advance(dt_s, rate):
                    task = self.state.tasks[job.task_id]
                    task.vehicle_energy_j += energy
                    vehicle.physical_energy_j += energy
                    vehicle.battery_j -= energy
                    vehicle_interval[vehicle.vehicle_id] += energy
            if not vehicle.failed and vehicle.hold_power_w > 0:
                for task in self.state.tasks.values():
                    if task.vehicle_id != vehicle.vehicle_id or task.terminal:
                        continue
                    energy = vehicle.hold_power_w * dt_s
                    task.hold_vehicle_energy_j += energy
                    vehicle.physical_energy_j += energy
                    vehicle.battery_j -= energy
                    vehicle_interval[vehicle.vehicle_id] += energy
        for rsu in self.state.rsus.values():
            baseline = rsu.idle_power_w * dt_s
            rsu.physical_energy_j += baseline
            rsu_interval[rsu.rsu_id] += baseline
            for name, pool in (("ingress", rsu.ingress), ("gpu", rsu.gpu)):
                rate = (
                    0.0
                    if rsu.failed
                    else self._resource_rate("rsu", rsu.rsu_id, name, start_s)
                )
                for job, _, energy in pool.advance(dt_s, rate):
                    if job.task_id is None:
                        if job.operation is not Operation.RSU_MODEL_MAINTENANCE:
                            raise InvariantViolation(
                                "SYSTEM_JOB_KIND_INVALID",
                                "taskless RSU job is not a model-maintenance operation",
                                job_id=job.job_id,
                            )
                        rsu.system_maintenance_energy_j += energy
                    else:
                        task = self.state.tasks[job.task_id]
                        task.rsu_energy_j += energy
                    rsu.physical_energy_j += energy
                    rsu_interval[rsu.rsu_id] += energy
            if rsu.hold_power_w > 0:
                for task in self.state.tasks.values():
                    if (
                        task.terminal
                        or not task.rsu_reserved
                        or task.selected_rsu != rsu.rsu_id
                    ):
                        continue
                    energy = rsu.hold_power_w * dt_s
                    task.hold_rsu_energy_j += energy
                    rsu.physical_energy_j += energy
                    rsu_interval[rsu.rsu_id] += energy
        active_link_counts = self._active_link_counts(start_s)
        for transfer in self.state.transfers.values():
            if transfer.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}:
                continue
            goodput, tx_power, rx_power = self._transfer_service(
                transfer, start_s, active_link_counts
            )
            if transfer.direction is TransferDirection.UL:
                vehicle_power, rsu_power = tx_power, rx_power
            else:
                vehicle_power, rsu_power = rx_power, tx_power
            _, vehicle_energy, rsu_energy = transfer.advance(
                dt_s, goodput, vehicle_power, rsu_power
            )
            task = self.state.tasks[transfer.task_id]
            vehicle = self.state.vehicles[transfer.vehicle_id]
            rsu = self.state.rsus[transfer.rsu_id]
            task.vehicle_energy_j += vehicle_energy
            task.rsu_energy_j += rsu_energy
            vehicle.physical_energy_j += vehicle_energy
            vehicle.battery_j -= vehicle_energy
            rsu.physical_energy_j += rsu_energy
            vehicle_interval[vehicle.vehicle_id] += vehicle_energy
            rsu_interval[rsu.rsu_id] += rsu_energy
            if transfer.direction is TransferDirection.UL:
                task.ul_remaining_bits = transfer.remaining_bits
            else:
                task.dl_remaining_bits = transfer.remaining_bits
        for vehicle in self.state.vehicles.values():
            if vehicle.battery_j < -1e-7:
                raise InvariantViolation(
                    "BATTERY_NEGATIVE_DURING_ADVANCE",
                    "battery guard did not precede depletion",
                    vehicle_id=vehicle.vehicle_id,
                    battery_j=vehicle.battery_j,
                )
            if vehicle.battery_j < 0:
                vehicle.battery_j = 0.0
        self.state.clock_s = time_s
        self.state.last_interval_vehicle_energy = vehicle_interval
        self.state.last_interval_rsu_energy = rsu_interval

    def _all_pools(self) -> Iterable[tuple[str, str, str, ResourcePool]]:
        for vehicle_id, runtime in sorted(self.state.vehicles.items()):
            for name, pool in sorted(runtime.resources.items()):
                yield "vehicle", vehicle_id, name, pool
        for rsu_id, runtime in sorted(self.state.rsus.items()):
            yield "rsu", rsu_id, "ingress", runtime.ingress
            yield "rsu", rsu_id, "gpu", runtime.gpu

    def _find_job_pool(self, job_id: str) -> tuple[str, str, str, ResourcePool] | None:
        for owner_type, owner_id, name, pool in self._all_pools():
            if job_id in pool.jobs:
                return owner_type, owner_id, name, pool
        return None

    def _materialized_completions(self, batch: list[Event]) -> list[Event]:
        by_key: dict[tuple[EventKind, str], Event] = {
            (event.kind, event.object_id or ""): event
            for event in batch
            if event.kind in {EventKind.COMPUTE_COMPLETE, EventKind.TRANSFER_COMPLETE}
        }
        synthetic_seq = max((event.seq for event in batch), default=0) + 1
        for _, _, _, pool in self._all_pools():
            for job in pool.zero_residual_jobs():
                key = (EventKind.COMPUTE_COMPLETE, job.job_id)
                existing = by_key.get(key)
                if existing is None or existing.version_token != job.completion_version:
                    by_key[key] = Event(
                        self.state.clock_s,
                        int(EventPriority.COMPLETION),
                        synthetic_seq,
                        EventKind.COMPUTE_COMPLETE,
                        job.task_id,
                        job.job_id,
                        job.completion_version,
                    )
                    synthetic_seq += 1
        for transfer in self.state.transfers.values():
            if (
                transfer.status in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
                and transfer.remaining_bits == 0.0
            ):
                key = (EventKind.TRANSFER_COMPLETE, transfer.transfer_id)
                existing = by_key.get(key)
                if (
                    existing is None
                    or existing.version_token != transfer.completion_version
                ):
                    by_key[key] = Event(
                        self.state.clock_s,
                        int(EventPriority.COMPLETION),
                        synthetic_seq,
                        EventKind.TRANSFER_COMPLETE,
                        transfer.task_id,
                        transfer.transfer_id,
                        transfer.completion_version,
                    )
                    synthetic_seq += 1
        return sorted(
            by_key.values(), key=lambda event: (event.seq, event.object_id or "")
        )

    def _process_compound(self, time_s: float, batch: list[Event]) -> tuple[int, int]:
        """Apply the fixed event phases and return arrival/timeout increments.

        Terminal deltas are deliberately consumed by ``run`` only after the
        subsequent dispatch/decision closure at the same timestamp.  Updating
        them here would miss failures caused by dispatch guards or by a
        zero-overhead controller decision.
        """

        arrivals = 0
        timeouts = 0

        # Phase 1/2: interval service was already integrated; all zero residual
        # completions now beat same-timestamp faults and deadline events.
        for event in self._materialized_completions(batch):
            if event.kind is EventKind.COMPUTE_COMPLETE:
                self._handle_compute_completion(event)
            else:
                self._handle_transfer_completion(event)

        # Phase 3: faults, link/mobility, thermal and version/cache changes.
        for event in sorted(batch, key=lambda item: (item.priority, item.seq)):
            if event.kind in {EventKind.COMPUTE_COMPLETE, EventKind.TRANSFER_COMPLETE}:
                continue
            if event.kind is EventKind.LINK_CHANGE:
                self._handle_link_change(event)
            elif event.kind is EventKind.THERMAL_CHANGE:
                self._handle_thermal_change(event)
            elif event.kind is EventKind.RSU_SNAPSHOT:
                self._handle_rsu_snapshot(event)
            elif event.kind is EventKind.DEVICE_FAULT:
                self._handle_device_fault(event)
            elif event.kind in {EventKind.MODEL_VERSION, EventKind.PROFILE_VERSION}:
                self._handle_version_event(event)
            elif event.kind is EventKind.BATTERY_GUARD:
                self._handle_battery_guard(event)

        # Phase 4: deadline only sees tasks that did not validly complete above.
        for event in sorted(batch, key=lambda item: item.seq):
            if event.kind is EventKind.DEADLINE:
                task = self.state.tasks.get(event.task_id or "")
                if task is not None and not task.terminal:
                    self._terminate_fail(task, FailureReason.TIMEOUT, "DEADLINE")
                    timeouts += 1

        # Phase 5: arrivals.
        for event in sorted(batch, key=lambda item: item.seq):
            if event.kind is EventKind.ARRIVAL:
                self._handle_arrival(event)
                arrivals += 1

        # Phase 6/7 prelude: deterministic controller-overhead completions commit
        # before idle-resource dispatch and new decision construction.
        for event in sorted(batch, key=lambda item: item.seq):
            if event.kind is EventKind.DISPATCH_DECISION:
                self._handle_decision_commit(event)
        return arrivals, timeouts

    def _update_virtual_queues(
        self, *, arrivals: int, timeouts: int, failures: int, completed: int
    ) -> None:
        vehicle_budget = {
            row.vehicle_id: row.average_power_budget_w for row in self.config.vehicles
        }
        rsu_budget = {
            row.rsu_id: row.average_power_budget_w for row in self.config.rsus
        }
        previous_time = (
            self.state.virtual_queues.trajectory[-1]["time_s"]
            if self.state.virtual_queues.trajectory
            else self.trace.horizon_start_s
        )
        self.state.virtual_queues.update(
            time_s=self.state.clock_s,
            dt_s=max(0.0, self.state.clock_s - previous_time),
            vehicle_energy=self.state.last_interval_vehicle_energy,
            rsu_energy=self.state.last_interval_rsu_energy,
            vehicle_power_budget=vehicle_budget,
            rsu_power_budget=rsu_budget,
            arrivals=arrivals,
            timeouts=timeouts,
            failures=failures,
            completed=completed,
            beta_timeout=self.config.long_term.timeout_rate_limit,
            beta_failure=self.config.long_term.failure_rate_limit,
            beta_coverage=self.config.long_term.coverage_rate_minimum,
        )

    def _handle_arrival(self, event: Event) -> None:
        arrival = self._arrivals[event.task_id or ""]
        if arrival.task_id in self.state.tasks:
            raise InvariantViolation(
                "DUPLICATE_ARRIVAL",
                "task arrived more than once",
                task_id=arrival.task_id,
            )
        task = TaskRecord(
            task_id=arrival.task_id,
            vehicle_id=arrival.vehicle_id,
            arrival_time_s=arrival.arrival_time_s,
            relative_deadline_s=arrival.relative_deadline_s,
            absolute_deadline_s=arrival.absolute_deadline_s,
            raw_handle=RawImageHandle(f"raw:{arrival.fixture_key}:{arrival.task_id}"),
            quality_features=tuple(
                value for _, value in sorted(arrival.quality_features.items())
            ),
            quality_probabilities=arrival.quality_probabilities,
            conformal_quality_bins=arrival.quality_candidates,
            ood=arrival.ood,
            true_identity=f"synthetic-subject:{arrival.fixture_key}",
            true_quality_region=arrival.true_quality_region,
        )
        self.state.tasks[task.task_id] = task
        vehicle = self.state.vehicles[task.vehicle_id]
        if vehicle.battery_depleted or vehicle.battery_j <= EPS:
            self._terminate_fail(
                task, FailureReason.BATTERY_GUARD, "ARRIVAL_WITH_DEPLETED_BATTERY"
            )
            return
        if vehicle.failed:
            self._terminate_fail(
                task, FailureReason.DEVICE_FAULT, "ARRIVAL_DURING_DEVICE_FAULT"
            )
            return
        context = self._device_context("vehicle", task.vehicle_id, self.state.clock_s)
        task.device_context = encode_context(context)
        true_quality = arrival.true_quality_region
        prep_result = self.trace.sample_prep(
            arrival.fixture_key,
            true_quality,
            vehicle.device_type,
            context,
            self._rng("prep", task.task_id, arrival.fixture_key),
        )
        if not prep_result.supported or prep_result.value is None:
            self._terminate_fail(task, FailureReason.UNSUPPORTED, "PREP_UNSUPPORTED")
            return
        row = prep_result.value
        self._prep_rows[task.task_id] = row
        prep_bounds = self.profile.preprocessing_resource_bounds
        if not vehicle.reserve(
            task,
            {"accelerator": 1},
            int(prep_bounds["max_memory_bytes"]),
        ):
            self._terminate_fail(
                task, FailureReason.BUFFER_CAPACITY, "PREP_RESERVATION"
            )
            return
        TaskStateMachine.transition(
            task, TaskState.PREP_WAIT, time_s=self.state.clock_s, trigger="PREP_ENQUEUE"
        )
        self._enqueue_job(
            task,
            operation=Operation.PREP,
            resource_kind=ResourceKind.ACCELERATOR,
            owner_type="vehicle",
            owner_id=task.vehicle_id,
            work_s=row.service_work_s,
            energy_j=row.dynamic_energy_j,
            memory_bytes=row.memory_bytes,
            version="prep-profile-v1",
        )

    def _handle_compute_completion(self, event: Event) -> None:
        found = self._find_job_pool(event.object_id or "")
        if found is None:
            return
        owner_type, owner_id, _, pool = found
        job = pool.complete(
            event.object_id or "", self.state.clock_s, event.version_token or -1
        )
        if job is None:
            return  # stale completion token
        if job.operation is Operation.RSU_MODEL_MAINTENANCE:
            self._complete_rsu_model_maintenance(job)
            return
        if job.task_id is None:
            raise InvariantViolation(
                "TASK_JOB_OWNER_MISSING",
                "ordinary compute completion has no task owner",
                job_id=job.job_id,
            )
        task = self.state.tasks[job.task_id]
        if task.terminal:
            return
        task.current_job_id = None
        task.record_time("end", job.operation.value, self.state.clock_s)
        if job.operation is Operation.PREP:
            row = self._prep_rows[task.task_id]
            vehicle = self.state.vehicles[task.vehicle_id]
            if not vehicle.reconcile_reservation(
                task,
                {},
                int(self.profile.preprocessing_resource_bounds["max_memory_bytes"]),
            ):
                raise InvariantViolation(
                    "PREP_BUFFER_RESERVATION_LOST",
                    "completed preprocessing could not retain its trusted buffer reservation",
                    task_id=task.task_id,
                )
            if row.failed:
                self._terminate_fail(task, FailureReason.PREP_FAILED, "PREP_FAILED")
                return
            task.aligned_handle = AlignedTensorHandle(f"aligned:{task.task_id}")
            task.actual_path.append("PREP")
            TaskStateMachine.transition(
                task, TaskState.RAW, time_s=self.state.clock_s, trigger="PREP_DONE"
            )
        elif job.operation is Operation.LOCAL_FER:
            row = self._local_rows[task.task_id]
            task.actual_path.append("LOCAL_FER")
            if row.failed or row.fer_loss is None:
                self._terminate_fail(
                    task, FailureReason.LOCAL_FAILED, "LOCAL_DONE_INVALID"
                )
                return
            task.realized_fer_loss = row.fer_loss
            task.realized_fer_true_label = row.true_label
            task.realized_fer_class_probabilities = row.class_probabilities
            task.evaluation_subject_cluster_id = row.subject_cluster_id
            task.result_valid = True
            TaskStateMachine.transition(
                task, TaskState.DONE, time_s=self.state.clock_s, trigger="LOCAL_DONE"
            )
            self._cleanup_terminal(task)
        elif job.operation is Operation.ANON:
            row = self._anon_rows[task.task_id]
            attempt = row.attempts[task.current_attempt_index]
            task.anon_attempt_audit.append(
                {
                    "attempt": attempt.attempt_index,
                    "time_s": self.state.clock_s,
                    "attempt_started_time_s": task.start_times[Operation.ANON.value][
                        -1
                    ],
                    "anon_completed_time_s": self.state.clock_s,
                    "anon_oom": attempt.anon_oom,
                    "anon_work_s": attempt.anon_work_s,
                    "anon_energy_j": attempt.anon_energy_j,
                }
            )
            task.actual_path.append(f"ANON#{attempt.attempt_index}")
            if attempt.anon_oom:
                self._retry_or_fallback(task, FailureReason.ANON_OOM)
                return
            if attempt.guard_work_s is None or attempt.guard_energy_j is None:
                self._retry_or_fallback(task, FailureReason.ANON_FAILED)
                return
            TaskStateMachine.transition(
                task,
                TaskState.GUARD_WAIT,
                time_s=self.state.clock_s,
                trigger="ANON_DONE",
            )
            self._enqueue_job(
                task,
                operation=Operation.GUARD,
                resource_kind=ResourceKind.CPU,
                owner_type="vehicle",
                owner_id=task.vehicle_id,
                work_s=attempt.guard_work_s,
                energy_j=attempt.guard_energy_j,
                memory_bytes=attempt.peak_memory_bytes,
                version=self.profile.pipelines[task.selected_pipeline or ""].guard_hash,
            )
        elif job.operation is Operation.GUARD:
            row = self._anon_rows[task.task_id]
            attempt = row.attempts[task.current_attempt_index]
            task.anon_attempt_audit[-1]["guard_passed"] = attempt.guard_passed
            task.anon_attempt_audit[-1]["guard_completed_time_s"] = self.state.clock_s
            task.actual_path.append(f"GUARD#{attempt.attempt_index}")
            if not attempt.guard_passed:
                self._retry_or_fallback(task, FailureReason.GUARD_REJECTED)
                return
            if attempt.encode_work_s is None or attempt.encode_energy_j is None:
                self._retry_or_fallback(task, FailureReason.ENCODE_FAILED)
                return
            TaskStateMachine.transition(
                task,
                TaskState.ENCODE_WAIT,
                time_s=self.state.clock_s,
                trigger="GUARD_DONE",
            )
            self._enqueue_job(
                task,
                operation=Operation.ENCODE,
                resource_kind=ResourceKind.ENCODER,
                owner_type="vehicle",
                owner_id=task.vehicle_id,
                work_s=attempt.encode_work_s,
                energy_j=attempt.encode_energy_j,
                memory_bytes=attempt.peak_memory_bytes,
                version=self.profile.pipelines[
                    task.selected_pipeline or ""
                ].encoder_hash,
            )
        elif job.operation is Operation.ENCODE:
            self._finish_encode(task)
        elif job.operation is Operation.RSU_INGRESS:
            row = self._edge_rows[task.task_id]
            task.rsu_audit.append(
                {
                    "phase": "ingress_done",
                    "time_s": self.state.clock_s,
                    "valid": not row.ingress_failed,
                }
            )
            if row.ingress_failed:
                # The paired joint row makes this a realized ingress outcome,
                # not a separately sampled failure.  Admission and all upload /
                # ingress costs remain charged; GPU work is never enqueued.
                task.actual_path.append(f"RSU_INGRESS_FAIL:{owner_id}")
                self._fallback_or_fail(task, FailureReason.EDGE_FAILED)
                return
            self._enqueue_job(
                task,
                operation=Operation.EDGE_FER,
                resource_kind=ResourceKind.RSU_GPU,
                owner_type="rsu",
                owner_id=task.selected_rsu or owner_id,
                work_s=row.gpu_work_s,
                energy_j=row.gpu_energy_j,
                memory_bytes=row.vram_bytes,
                version=self.profile.edge_models[
                    task.selected_edge_model or ""
                ].model_hash,
            )
        elif job.operation is Operation.EDGE_FER:
            self._finish_edge(task)

    def _handle_transfer_completion(self, event: Event) -> None:
        transfer = self.state.transfers.get(event.object_id or "")
        if (
            transfer is None
            or transfer.completion_version != event.version_token
            or transfer.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
            or transfer.remaining_bits > EPS
        ):
            return
        task = self.state.tasks[transfer.task_id]
        if task.terminal:
            return
        transfer.remaining_bits = 0.0
        transfer.status = TransferStatus.DONE
        task.current_transfer_id = None
        if transfer.direction is TransferDirection.UL:
            task.ul_remaining_bits = 0.0
            task.network_audit.append(
                {
                    "direction": "UL",
                    "status": "DONE",
                    "time_s": self.state.clock_s,
                    "delivered_bits": transfer.delivered_bits,
                    "vehicle_energy_j": transfer.vehicle_energy_j,
                    "rsu_energy_j": transfer.rsu_energy_j,
                }
            )
            self._admit_at_rsu(task, transfer)
        else:
            task.dl_remaining_bits = 0.0
            task.network_audit.append(
                {
                    "direction": "DL",
                    "status": "DONE",
                    "time_s": self.state.clock_s,
                    "delivered_bits": transfer.delivered_bits,
                    "vehicle_energy_j": transfer.vehicle_energy_j,
                    "rsu_energy_j": transfer.rsu_energy_j,
                }
            )
            if not self._frozen_versions_active():
                self._fallback_or_fail(task, FailureReason.VERSION_MISMATCH)
                return
            task.result_valid = True
            TaskStateMachine.transition(
                task, TaskState.DONE, time_s=self.state.clock_s, trigger="DL_DONE"
            )
            self._cleanup_terminal(task)

    def _handle_link_change(self, event: Event) -> None:
        payload = event.payload
        if isinstance(payload, ExogenousEvent):
            details = dict(payload.details)
            vehicle_id = str(
                details.get(
                    "vehicle_id",
                    payload.target_id if payload.target_type == "vehicle" else "",
                )
            )
            rsu_id = str(
                details.get(
                    "rsu_id", payload.target_id if payload.target_type == "rsu" else ""
                )
            )
            direction_value = str(details.get("direction", "UL"))
        elif isinstance(payload, Mapping):
            vehicle_id = str(payload.get("vehicle_id", ""))
            rsu_id = str(payload.get("rsu_id", ""))
            direction_value = str(payload.get("direction", "UL"))
            if payload.get("pause_expiry"):
                transfer = self.state.transfers.get(str(payload.get("transfer_id", "")))
                if (
                    transfer is not None
                    and transfer.status is TransferStatus.PAUSED
                    and transfer.completion_version == event.version_token
                ):
                    self._fail_transfer(
                        transfer,
                        FailureReason.UL_FAILED
                        if transfer.direction is TransferDirection.UL
                        else FailureReason.DL_FAILED,
                    )
                return
        else:
            return
        try:
            direction = TransferDirection(direction_value)
        except ValueError:
            return
        current = self._wireless_segment(
            vehicle_id, rsu_id, direction, self.state.clock_s
        )
        for transfer in sorted(
            self.state.transfers.values(), key=lambda item: item.transfer_id
        ):
            if (
                transfer.vehicle_id != vehicle_id
                or transfer.rsu_id != rsu_id
                or transfer.direction is not direction
                or transfer.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
            ):
                continue
            if current is not None and current.link_state == "connected":
                transfer.status = TransferStatus.ACTIVE
                transfer.paused_since_s = None
                transfer.completion_version += 1
                self.state.tasks[transfer.task_id].network_audit.append(
                    {
                        "direction": direction.value,
                        "status": "RESUMED",
                        "time_s": self.state.clock_s,
                    }
                )
            elif current is not None and current.link_state == "temporary_outage":
                if transfer.status is not TransferStatus.PAUSED:
                    transfer.status = TransferStatus.PAUSED
                    transfer.paused_since_s = self.state.clock_s
                    transfer.completion_version += 1
                    limit = (
                        self.config.uplink_pause_limit_s
                        if direction is TransferDirection.UL
                        else self.config.downlink_pause_limit_s
                    )
                    self.state.tasks[transfer.task_id].network_audit.append(
                        {
                            "direction": direction.value,
                            "status": "PAUSED",
                            "time_s": self.state.clock_s,
                        }
                    )
                    if limit <= EPS:
                        self._fail_transfer(
                            transfer,
                            FailureReason.UL_FAILED
                            if direction is TransferDirection.UL
                            else FailureReason.DL_FAILED,
                        )
                    else:
                        self.state.events.push(
                            self.state.clock_s + limit,
                            EventKind.LINK_CHANGE,
                            task_id=transfer.task_id,
                            object_id=transfer.transfer_id,
                            version_token=transfer.completion_version,
                            payload={
                                "pause_expiry": True,
                                "transfer_id": transfer.transfer_id,
                            },
                        )
            else:
                reason = FailureReason.PERMANENT_LINK_LOSS
                self._fail_transfer(transfer, reason)

    def _handle_thermal_change(self, event: Event) -> None:
        raw_payload = getattr(event, "payload", None)
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        owner_type = str(payload.get("owner_type", ""))
        owner_id = str(payload.get("owner_id", ""))
        resource = str(payload.get("resource", "all"))
        for current_owner_type, current_owner_id, name, pool in self._all_pools():
            if current_owner_type != owner_type or current_owner_id != owner_id:
                continue
            if resource not in {"all", name}:
                continue
            for job_id in pool.running:
                if job_id is not None:
                    pool.jobs[job_id].completion_version += 1

    def _handle_device_fault(self, event: Event) -> None:
        row = event.payload
        if not isinstance(row, ExogenousEvent):
            return
        failed = row.event_type != "DEVICE_FAULT_END"
        if row.target_type == "vehicle" and row.target_id in self.state.vehicles:
            runtime = self.state.vehicles[row.target_id]
            runtime.failed = failed or runtime.battery_depleted
            if failed:
                for task in sorted(
                    self.state.tasks.values(), key=lambda item: item.task_id
                ):
                    if task.vehicle_id == row.target_id and not task.terminal:
                        self._terminate_fail(
                            task, FailureReason.DEVICE_FAULT, "VEHICLE_DEVICE_FAULT"
                        )
        elif row.target_type == "rsu" and row.target_id in self.state.rsus:
            runtime = self.state.rsus[row.target_id]
            runtime.failed = failed
            if failed:
                affected: set[str] = set()
                for transfer in sorted(
                    self.state.transfers.values(), key=lambda item: item.transfer_id
                ):
                    if transfer.rsu_id != row.target_id or transfer.status not in {
                        TransferStatus.ACTIVE,
                        TransferStatus.PAUSED,
                    }:
                        continue
                    self._fail_transfer(
                        transfer,
                        FailureReason.UL_FAILED
                        if transfer.direction is TransferDirection.UL
                        else FailureReason.DL_FAILED,
                    )
                for pool in (runtime.ingress, runtime.gpu):
                    for job in list(pool.jobs.values()):
                        if job.task_id is not None and job.status in {
                            JobStatus.WAITING,
                            JobStatus.RUNNING,
                        }:
                            affected.add(job.task_id)
                for task_id in sorted(affected):
                    task = self.state.tasks[task_id]
                    self._cancel_compute_jobs_with_audit(
                        task, FailureReason.EDGE_FAILED, "RSU_DEVICE_FAULT"
                    )
                    task.current_job_id = None
                    runtime.admission.release(task_id)
                    task.rsu_reserved = False
                    self._fallback_or_fail(task, FailureReason.EDGE_FAILED)

    def _handle_version_event(self, event: Event) -> None:
        row = event.payload
        if not isinstance(row, ExogenousEvent):
            return
        if (
            row.event_type in {"MODEL_VERSION", "MODEL_CACHE"}
            and row.target_id in self.state.rsus
        ):
            self._enqueue_rsu_model_maintenance(row)
        elif row.event_type == "MODEL_VERSION" and row.target_id in self.state.vehicles:
            model_id = str(row.details.get("model_id", ""))
            if model_id:
                self._active_local_model_hashes[(row.target_id, model_id)] = (
                    row.new_version or "expired"
                )
        elif row.event_type == "PROFILE_VERSION":
            self._active_profile_hash = row.new_version or "expired"
        elif row.event_type == "PROTOCOL_VERSION":
            self._active_protocol_version = row.new_version or "expired"

    def _enqueue_rsu_model_maintenance(self, row: ExogenousEvent) -> ComputeJob:
        """Queue a taskless, non-preemptive GPU maintenance transaction.

        The event arrival has no cache side effect.  Its frozen work and
        dynamic energy compete with inference on the same finite RSU GPU;
        cache mutation is committed atomically only by the completion handler.
        """

        if (
            row.target_type != "rsu"
            or row.target_id not in self.state.rsus
            or row.event_type not in {"MODEL_VERSION", "MODEL_CACHE"}
            or row.maintenance_work_s is None
            or row.maintenance_work_s <= 0
            or row.maintenance_energy_j is None
            or row.maintenance_energy_j <= 0
        ):
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_INVALID",
                "RSU model/cache event lacks positive frozen maintenance work and energy",
                event_id=row.event_id,
                rsu_id=row.target_id,
            )
        model_id = str(row.details.get("model_id", ""))
        if not model_id:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_MODEL_MISSING",
                "RSU model/cache maintenance event lacks details.model_id",
                event_id=row.event_id,
            )
        runtime = self.state.rsus[row.target_id]
        self._job_counter += 1
        job_id = f"job-{self._job_counter:08d}"
        job = ComputeJob(
            job_id=job_id,
            task_id=None,
            owner_type="rsu",
            owner_id=row.target_id,
            operation=Operation.RSU_MODEL_MAINTENANCE,
            resource_kind=ResourceKind.RSU_GPU,
            model_or_pipeline_version=row.new_version or f"remove:{model_id}",
            enqueue_time_s=self.state.clock_s,
            absolute_deadline_s=max(self.state.clock_s, self.trace.horizon_end_s),
            enqueue_seq=runtime.gpu.next_enqueue_seq(),
            total_work_s=row.maintenance_work_s,
            residual_work_s=row.maintenance_work_s,
            total_dynamic_energy_j=row.maintenance_energy_j,
        )
        runtime.gpu.enqueue(job)
        self._rsu_maintenance_events[job_id] = row
        key = (row.target_id, model_id)
        self._rsu_maintenance_job_keys[job_id] = key
        if key in self._rsu_maintenance_active:
            self._rsu_maintenance_waiting.setdefault(key, []).append(job_id)
        else:
            self._rsu_maintenance_active[key] = job_id
        self.ledger.record_event(
            {
                "event_kind": "RSU_MODEL_MAINTENANCE_ENQUEUE",
                "event_id": row.event_id,
                "job_id": job_id,
                "rsu_id": row.target_id,
                "model_id": model_id,
                "model_event_type": row.event_type,
                "time_s": self.state.clock_s,
                "work_s": row.maintenance_work_s,
                "energy_j": row.maintenance_energy_j,
            }
        )
        return job

    def _rsu_model_maintenance_dispatchable(self, job: ComputeJob) -> bool:
        if job.operation is not Operation.RSU_MODEL_MAINTENANCE:
            return True
        key = self._rsu_maintenance_job_keys.get(job.job_id)
        if key is None:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_CHAIN_MISSING",
                "queued model maintenance has no serialization key",
                job_id=job.job_id,
            )
        return self._rsu_maintenance_active.get(key) == job.job_id

    def _release_next_rsu_model_maintenance(self, job: ComputeJob) -> None:
        key = self._rsu_maintenance_job_keys.pop(job.job_id, None)
        if key is None or self._rsu_maintenance_active.get(key) != job.job_id:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_CHAIN_CORRUPT",
                "completed model maintenance is not the active chain head",
                job_id=job.job_id,
                key=key,
                active_job_id=None
                if key is None
                else self._rsu_maintenance_active.get(key),
            )
        waiting = self._rsu_maintenance_waiting.get(key, [])
        if waiting:
            self._rsu_maintenance_active[key] = waiting.pop(0)
            if not waiting:
                self._rsu_maintenance_waiting.pop(key, None)
        else:
            self._rsu_maintenance_active.pop(key, None)

    def _complete_rsu_model_maintenance(self, job: ComputeJob) -> None:
        row = self._rsu_maintenance_events.get(job.job_id)
        if row is None:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_EVENT_LOST",
                "completed RSU maintenance job has no frozen source event",
                job_id=job.job_id,
            )
        key = self._rsu_maintenance_job_keys.get(job.job_id)
        if key is None or self._rsu_maintenance_active.get(key) != job.job_id:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_CHAIN_CORRUPT",
                "completed model maintenance is not the active chain head",
                job_id=job.job_id,
                key=key,
                active_job_id=None
                if key is None
                else self._rsu_maintenance_active.get(key),
            )
        runtime = self.state.rsus[job.owner_id]
        model_id = str(row.details.get("model_id", ""))
        cache_before = dict(runtime.admission.cached_models)
        if row.old_version is not None:
            if cache_before.get(model_id) != row.old_version:
                raise InvariantViolation(
                    "RSU_MODEL_MAINTENANCE_VERSION_PRECONDITION",
                    "model cache changed before a queued version transaction could commit",
                    event_id=row.event_id,
                    model_id=model_id,
                    expected_old_version=row.old_version,
                    actual_version=cache_before.get(model_id),
                )
        cache_after = dict(cache_before)
        if bool(row.details.get("remove", False)):
            cache_after.pop(model_id, None)
        elif row.new_version:
            cache_after[model_id] = row.new_version
        else:
            raise InvariantViolation(
                "RSU_MODEL_MAINTENANCE_UPDATE_MISSING",
                "model/cache maintenance has neither remove nor new_version",
                event_id=row.event_id,
            )
        runtime.admission.update_cache(cache_after)
        self._rsu_maintenance_events.pop(job.job_id, None)
        self._release_next_rsu_model_maintenance(job)
        self.ledger.record_event(
            {
                "event_kind": "RSU_MODEL_MAINTENANCE_COMPLETE",
                "event_id": row.event_id,
                "job_id": job.job_id,
                "rsu_id": row.target_id,
                "model_id": model_id,
                "model_event_type": row.event_type,
                "enqueue_time_s": job.enqueue_time_s,
                "start_time_s": job.start_time_s,
                "time_s": self.state.clock_s,
                "wait_time_s": (
                    None
                    if job.start_time_s is None
                    else job.start_time_s - job.enqueue_time_s
                ),
                "service_elapsed_s": (
                    None
                    if job.start_time_s is None
                    else self.state.clock_s - job.start_time_s
                ),
                "work_s": job.total_work_s,
                "dynamic_energy_j": job.consumed_dynamic_energy_j,
                "old_version": cache_before.get(model_id),
                "new_version": cache_after.get(model_id),
            }
        )
        # Existing admissions remain pinned in their immutable reservations;
        # only subsequent atomic admission reads the completed cache state.

    def _handle_rsu_snapshot(self, event: Event) -> None:
        runtime = self.state.rsus.get(event.object_id or "")
        if runtime is None:
            raise InvariantViolation(
                "RSU_SNAPSHOT_TARGET_UNKNOWN",
                "telemetry event references an unknown RSU",
                rsu_id=event.object_id,
            )
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        if payload.get("phase") == "delivery":
            snapshot = payload.get("snapshot")
            sample_time_s = payload.get("sample_time_s")
            if not isinstance(snapshot, Mapping) or not isinstance(
                sample_time_s, (int, float)
            ):
                raise InvariantViolation(
                    "RSU_TELEMETRY_PAYLOAD",
                    "telemetry delivery lacks a valid frozen snapshot",
                    rsu_id=event.object_id,
                )
            runtime.current_snapshot_time_s = float(sample_time_s)
            runtime.public_snapshot = dict(snapshot)
            return

        rsu_id = event.object_id or ""
        sequence = self._telemetry_sample_sequence[rsu_id] + 1
        self._telemetry_sample_sequence[rsu_id] = sequence
        drop_every = self.config.rsu_telemetry_drop_every
        if drop_every > 0 and sequence % drop_every == 0:
            self.ledger.record_event(
                time_s=self.state.clock_s,
                event_kind="RSU_TELEMETRY_DROP",
                rsu_id=rsu_id,
                sample_sequence=sequence,
            )
            return
        snapshot = self._rsu_public_snapshot(runtime, rsu_id, self.state.clock_s)
        quantum = self.config.rsu_telemetry_quantum_work_s
        if quantum > 0:
            for field in (
                "reserved_work_gpu_s",
                "ingress_residual_work_s",
                "gpu_residual_work_s",
            ):
                snapshot[field] = round(float(snapshot[field]) / quantum) * quantum
        if self.config.rsu_telemetry_delay_s <= EPS:
            runtime.current_snapshot_time_s = self.state.clock_s
            runtime.public_snapshot = snapshot
            return
        delivery_time = self.state.clock_s + self.config.rsu_telemetry_delay_s
        if delivery_time <= self.trace.horizon_end_s + EPS:
            self.state.events.push(
                delivery_time,
                EventKind.RSU_SNAPSHOT,
                object_id=rsu_id,
                payload={
                    "phase": "delivery",
                    "sample_time_s": self.state.clock_s,
                    "sample_sequence": sequence,
                    "snapshot": snapshot,
                },
            )

    def _handle_battery_guard(self, event: Event) -> None:
        vehicle_id = event.object_id or ""
        if event.version_token != self._battery_versions.get(vehicle_id):
            return
        vehicle = self.state.vehicles.get(vehicle_id)
        if vehicle is None:
            return
        vehicle.battery_j = 0.0
        vehicle.battery_depleted = True
        vehicle.failed = True
        for task in sorted(self.state.tasks.values(), key=lambda item: item.task_id):
            if task.vehicle_id == vehicle_id and not task.terminal:
                self._terminate_fail(task, FailureReason.BATTERY_GUARD, "BATTERY_GUARD")

    def _dispatch_all(self) -> None:
        changed = True
        while changed:
            changed = False
            for owner_type, owner_id, resource_name, pool in self._all_pools():
                eligible = (
                    self._rsu_model_maintenance_dispatchable
                    if owner_type == "rsu" and resource_name == "gpu"
                    else None
                )
                started = pool.dispatch(self.state.clock_s, eligible=eligible)
                if started:
                    changed = True
                for job in started:
                    if job.operation is Operation.RSU_MODEL_MAINTENANCE:
                        if owner_type != "rsu" or job.task_id is not None:
                            raise InvariantViolation(
                                "RSU_MODEL_MAINTENANCE_OWNER",
                                "model maintenance must be a taskless RSU GPU job",
                                job_id=job.job_id,
                            )
                        source = self._rsu_maintenance_events.get(job.job_id)
                        self.ledger.record_event(
                            {
                                "event_kind": "RSU_MODEL_MAINTENANCE_START",
                                "event_id": (
                                    None if source is None else source.event_id
                                ),
                                "job_id": job.job_id,
                                "rsu_id": owner_id,
                                "time_s": self.state.clock_s,
                                "wait_time_s": self.state.clock_s - job.enqueue_time_s,
                            }
                        )
                        continue
                    if job.task_id is None:
                        raise InvariantViolation(
                            "TASK_JOB_OWNER_MISSING",
                            "ordinary dispatched job has no task owner",
                            job_id=job.job_id,
                        )
                    task = self.state.tasks[job.task_id]
                    if task.terminal:
                        pool.cancel_task(task.task_id, self.state.clock_s)
                        continue
                    if owner_type == "vehicle":
                        runtime = self.state.vehicles[owner_id]
                        remaining_energy = self._remaining_job_energy_upper(
                            job,
                            resource_name,
                            self.state.clock_s,
                        )
                        if runtime.failed or runtime.battery_j + EPS < remaining_energy:
                            self._cancel_compute_jobs_with_audit(
                                task,
                                FailureReason.BATTERY_GUARD,
                                "DISPATCH_BATTERY_GUARD",
                            )
                            task.current_job_id = None
                            self._fallback_or_fail(task, FailureReason.BATTERY_GUARD)
                            continue
                    else:
                        runtime_rsu = self.state.rsus[owner_id]
                        if runtime_rsu.failed:
                            self._cancel_compute_jobs_with_audit(
                                task, FailureReason.EDGE_FAILED, "RSU_DISPATCH_FAILED"
                            )
                            task.current_job_id = None
                            runtime_rsu.admission.release(task.task_id)
                            task.rsu_reserved = False
                            self._fallback_or_fail(task, FailureReason.EDGE_FAILED)
                            continue
                    task.record_time("start", job.operation.value, self.state.clock_s)
                    if job.operation is Operation.PREP:
                        TaskStateMachine.transition(
                            task,
                            TaskState.PREP_RUN,
                            time_s=self.state.clock_s,
                            trigger="PREP_START",
                        )
                    elif job.operation is Operation.LOCAL_FER:
                        TaskStateMachine.transition(
                            task,
                            TaskState.LOCAL_RUN,
                            time_s=self.state.clock_s,
                            trigger="LOCAL_START",
                        )
                    elif job.operation is Operation.ANON:
                        TaskStateMachine.transition(
                            task,
                            TaskState.ANON_RUN,
                            time_s=self.state.clock_s,
                            trigger="ANON_START",
                        )
                    elif job.operation is Operation.GUARD:
                        TaskStateMachine.transition(
                            task,
                            TaskState.GUARD_RUN,
                            time_s=self.state.clock_s,
                            trigger="GUARD_START",
                        )
                    elif job.operation is Operation.ENCODE:
                        TaskStateMachine.transition(
                            task,
                            TaskState.ENCODE_RUN,
                            time_s=self.state.clock_s,
                            trigger="ENCODE_START",
                        )
                    elif job.operation is Operation.RSU_INGRESS:
                        task.rsu_audit.append(
                            {"phase": "ingress_start", "time_s": self.state.clock_s}
                        )
                    elif job.operation is Operation.EDGE_FER:
                        TaskStateMachine.transition(
                            task,
                            TaskState.EDGE_RUN,
                            time_s=self.state.clock_s,
                            trigger="EDGE_START",
                        )

    def _schedule_completions_and_battery_guards(self) -> None:
        now_s = self.state.clock_s
        for owner_type, owner_id, resource, pool in self._all_pools():
            rate = self._resource_rate(owner_type, owner_id, resource, now_s)
            if owner_type == "rsu" and self.state.rsus[owner_id].failed:
                rate = 0.0
            boundary = self._next_thermal_boundary(
                owner_type, owner_id, resource, now_s
            )
            if rate <= 0:
                continue
            for job_id in pool.running:
                if job_id is None:
                    continue
                job = pool.jobs[job_id]
                finish = _strict_future_completion_time(
                    now_s, job.residual_work_s, rate
                )
                if finish <= boundary + EPS:
                    self.state.events.push(
                        finish,
                        EventKind.COMPUTE_COMPLETE,
                        task_id=job.task_id,
                        object_id=job.job_id,
                        version_token=job.completion_version,
                    )
        active_link_counts = self._active_link_counts(now_s)
        for transfer in self.state.transfers.values():
            if transfer.status is not TransferStatus.ACTIVE:
                continue
            goodput, _, _ = self._transfer_service(transfer, now_s, active_link_counts)
            if goodput <= 0:
                continue
            finish = _strict_future_completion_time(
                now_s, transfer.remaining_bits, goodput
            )
            boundary = self._next_link_boundary(transfer, now_s)
            if finish <= boundary + EPS:
                self.state.events.push(
                    finish,
                    EventKind.TRANSFER_COMPLETE,
                    task_id=transfer.task_id,
                    object_id=transfer.transfer_id,
                    version_token=transfer.completion_version,
                )
        for vehicle_id, vehicle in self.state.vehicles.items():
            power = 0.0 if vehicle.failed else vehicle.idle_power_w
            if not vehicle.failed:
                power += vehicle.hold_power_w * sum(
                    task.vehicle_id == vehicle_id and not task.terminal
                    for task in self.state.tasks.values()
                )
            for name, pool in vehicle.resources.items():
                rate = self._resource_rate("vehicle", vehicle_id, name, now_s)
                for job_id in pool.running:
                    if job_id is not None and not vehicle.failed:
                        job = pool.jobs[job_id]
                        if rate > 0:
                            power += (
                                job.total_dynamic_energy_j / job.total_work_s * rate
                            )
            for transfer in self.state.transfers.values():
                if (
                    vehicle.failed
                    or transfer.vehicle_id != vehicle_id
                    or transfer.status
                    not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
                ):
                    continue
                _, tx_power, rx_power = self._transfer_service(
                    transfer, now_s, active_link_counts
                )
                power += (
                    tx_power if transfer.direction is TransferDirection.UL else rx_power
                )
            self._battery_versions[vehicle_id] += 1
            if power > 0 and vehicle.battery_j > 0:
                depletion = now_s + vehicle.battery_j / power
                if depletion <= self.trace.horizon_end_s + EPS:
                    self.state.events.push(
                        depletion,
                        EventKind.BATTERY_GUARD,
                        object_id=vehicle_id,
                        version_token=self._battery_versions[vehicle_id],
                    )

    def _observation(
        self, task: TaskRecord, *, stage: ActionStage | None = None
    ) -> Observation:
        context = self._device_context("vehicle", task.vehicle_id, self.state.clock_s)
        task.device_context = encode_context(context)
        snapshot_tokens = _snapshot_task_tokens(self.state, task.vehicle_id)
        links: dict[str, Any] = {}
        for rsu_id in sorted(self.state.rsus):
            ul = self._wireless_segment(
                task.vehicle_id, rsu_id, TransferDirection.UL, self.state.clock_s
            )
            dl = self._wireless_segment(
                task.vehicle_id, rsu_id, TransferDirection.DL, self.state.clock_s
            )
            active_transfers = tuple(
                {
                    # Observation-local identity is deliberately unrelated to
                    # the simulator transfer/task IDs.  These causal rows let
                    # an isolated rollout preserve shared-link contention
                    # without exposing packet contents or task identities.
                    "queue_rank": rank,
                    "task_token": snapshot_tokens.get(transfer.task_id),
                    "direction": transfer.direction.value,
                    "total_bits": transfer.total_bits,
                    "remaining_bits": transfer.remaining_bits,
                    "status": transfer.status.value,
                    "start_age_s": max(0.0, self.state.clock_s - transfer.start_time_s),
                    "pause_age_s": (
                        None
                        if transfer.paused_since_s is None
                        else max(
                            0.0,
                            self.state.clock_s - transfer.paused_since_s,
                        )
                    ),
                }
                for rank, transfer in enumerate(
                    sorted(
                        (
                            transfer
                            for transfer in self.state.transfers.values()
                            if transfer.vehicle_id == task.vehicle_id
                            and transfer.rsu_id == rsu_id
                            and transfer.status
                            in {TransferStatus.ACTIVE, TransferStatus.PAUSED}
                        ),
                        key=lambda transfer: (
                            transfer.direction.value,
                            transfer.start_time_s,
                            transfer.transfer_id,
                        ),
                    )
                )
            )
            links[rsu_id] = {
                "connected": ul is not None and ul.link_state == "connected",
                "ul_goodput_bps": 0.0 if ul is None else ul.goodput_bps,
                "dl_goodput_bps": 0.0 if dl is None else dl.goodput_bps,
                "ul_link_state": "missing" if ul is None else ul.link_state,
                "dl_link_state": "missing" if dl is None else dl.link_state,
                # Current-segment powers are causal public link telemetry.  No
                # future boundary or future segment value is exposed.
                "ul_transmitter_power_w": (
                    0.0 if ul is None else ul.transmitter_power_w
                ),
                "ul_receiver_power_w": 0.0 if ul is None else ul.receiver_power_w,
                "dl_transmitter_power_w": (
                    0.0 if dl is None else dl.transmitter_power_w
                ),
                "dl_receiver_power_w": 0.0 if dl is None else dl.receiver_power_w,
                "uplink_start_energy_j": 0.001,
                "active_transfers": active_transfers,
            }
        versions = {
            "protocol_version": self._active_protocol_version,
            "profile_hash": self._active_profile_hash,
            "local_models": {
                model_id: {
                    "model_hash": model_hash,
                    "protocol_version": self._active_protocol_version,
                }
                for (vehicle_id, model_id), model_hash in sorted(
                    self._active_local_model_hashes.items()
                )
                if vehicle_id == task.vehicle_id
            },
        }
        observation = ObservationBuilder.build(
            task,
            self.state,
            profile=self.profile,
            stage=stage,
            links=links,
            versions=versions,
            metadata={"max_snapshot_age_s": self.config.max_snapshot_age_s},
            pending_decisions={
                task_id: (action, self._pending_decision_due_s[task_id])
                for task_id, action in self._pending_decisions.items()
                if task_id in self._pending_decision_due_s
            },
            pairing_tokens=self.mask_engine._pairing_tokens,
            artifact_capabilities={
                item.task_id: self._artifact_capabilities(item.artifact_key)
                for item in self.state.tasks.values()
                if item.vehicle_id == task.vehicle_id and not item.terminal
            },
        )
        assert_observation_safe(observation)
        return observation

    def _make_decisions(self) -> None:
        tasks = [
            task
            for task in self.state.tasks.values()
            if task.state in {TaskState.RAW, TaskState.READY}
            and task.task_id not in self._pending_decisions
            and not task.terminal
        ]
        for task in sorted(
            tasks,
            key=lambda item: (item.absolute_deadline_s, item.vehicle_id, item.task_id),
        ):
            observation = self._observation(task)
            mask = self.mask_engine.enumerate(task, observation, self.state)
            task.mask_audit.append(
                {
                    "time_s": self.state.clock_s,
                    "stage": observation.stage.value,
                    "mask_epoch": "DECISION",
                    "rows": list(mask.audit_rows()),
                }
            )
            start_ns = time.perf_counter_ns()
            self._decision_score_buffer.pop(task.task_id, None)
            proposed = self._policy_choose(task, observation, mask)
            policy_scores = self._decision_score_buffer.pop(task.task_id, None)
            wall_clock_s = (time.perf_counter_ns() - start_ns) / 1e9
            if proposed not in mask.candidates:
                proposed = Action.fail(observation.stage)
            elif proposed not in mask.allowed:
                proposed = self.repairer.repair(
                    proposed,
                    task,
                    observation,
                    self.state,
                    score=policy_scores,
                ).executed
            task.action_audit.append(
                {
                    "time_s": self.state.clock_s,
                    "stage": observation.stage.value,
                    "proposed": proposed.to_dict(),
                    "controller_wall_clock_s_diagnostic": wall_clock_s,
                    "simulated_controller_overhead_s": self.config.controller.controller_overhead_s,
                }
            )
            controller_energy = self.config.controller.controller_energy_j
            vehicle = self.state.vehicles[task.vehicle_id]
            if vehicle.battery_j + EPS < controller_energy:
                self._terminate_fail(
                    task, FailureReason.BATTERY_GUARD, "CONTROLLER_ENERGY_GUARD"
                )
                continue
            task.vehicle_energy_j += controller_energy
            vehicle.physical_energy_j += controller_energy
            vehicle.battery_j -= controller_energy
            self.state.virtual_queues.vehicle_power[task.vehicle_id] = max(
                0.0,
                self.state.virtual_queues.vehicle_power[task.vehicle_id]
                + controller_energy,
            )
            planned_shadow = self._planned_vehicle_shadow(task, proposed)
            if planned_shadow is not None:
                tokens, memory_bytes = planned_shadow
                if not vehicle.reconcile_reservation(task, tokens, memory_bytes):
                    raise InvariantViolation(
                        "PLANNED_SHADOW_MASK_DIVERGENCE",
                        "hard-safe action could not acquire its sequential planned shadow",
                        task_id=task.task_id,
                        action=proposed.canonical_id,
                    )
                task.action_audit.append(
                    {
                        "time_s": self.state.clock_s,
                        "stage": observation.stage.value,
                        "planned_shadow": {
                            "action_id": proposed.canonical_id,
                            "descriptor_tokens": dict(tokens),
                            "memory_bytes": memory_bytes,
                        },
                    }
                )
            overhead = self.config.controller.controller_overhead_s
            if overhead <= EPS:
                self._execute_proposed(task, proposed, score=policy_scores)
            else:
                self._pending_decisions[task.task_id] = proposed
                if policy_scores is not None:
                    self._pending_decision_scores[task.task_id] = policy_scores
                self._pending_decision_due_s[task.task_id] = (
                    self.state.clock_s + overhead
                )
                self.state.events.push(
                    self.state.clock_s + overhead,
                    EventKind.DISPATCH_DECISION,
                    task_id=task.task_id,
                    payload=proposed,
                )

    def _planned_vehicle_shadow(
        self, task: TaskRecord, action: Action
    ) -> tuple[dict[str, int], int] | None:
        """Return the preregistered reservation committed during joint ordering."""

        if action.kind is ActionKind.LOCAL:
            model = self.profile.local_models.get(action.local_model_id or "")
            if model is None:
                return None
            return (
                {"accelerator": 1},
                int(model.deployment_resource_bounds["max_memory_bytes"]),
            )
        if action.kind is ActionKind.PIPE:
            pipeline = self.profile.pipelines.get(action.pipeline_id or "")
            if pipeline is None:
                return None
            memory_bytes = int(
                pipeline.deployment_resource_bounds["max_peak_memory_bytes"]
            )
            if pipeline.fallback_local_model:
                fallback = self.profile.local_models[pipeline.fallback_local_model]
                memory_bytes = max(
                    memory_bytes,
                    int(fallback.deployment_resource_bounds["max_memory_bytes"]),
                )
            return {"accelerator": 1, "cpu": 1, "encoder": 1}, memory_bytes
        return None

    def _policy_choose(
        self, task: TaskRecord, observation: Observation, mask: Any
    ) -> Action:
        """Return the selected action and retain scores for the caller.

        The action-only signature is intentionally stable for external policy
        isolation tests and legacy integrations.  ``_make_decisions`` clears
        and consumes the task-keyed one-call buffer immediately, so a legacy
        override that supplies no scores cannot inherit stale policy state.
        """

        action, scores = self._policy_choose_with_scores(task, observation, mask)
        if scores is not None:
            self._decision_score_buffer[task.task_id] = scores
        return action

    def _policy_choose_with_scores(
        self, task: TaskRecord, observation: Observation, mask: Any
    ) -> tuple[Action, Mapping[Action, float] | None]:
        """Select an action and freeze any policy-specific alternative scores.

        H=1 exposes drift-cost scores and ESL-SMPC exposes scenario-rollout
        scores through ``PolicyDecision``.  A nonzero simulated controller
        overhead can invalidate the selected action before physical commit.
        Keeping a defensive immutable snapshot lets the execution-time repair
        rank the remaining hard-safe, same-stage alternatives with the same
        policy objective instead of silently changing to the generic repair
        estimate.  Legacy policy interfaces have no such score map and retain
        the documented generic deterministic ranking.
        """

        policy_task = PolicyTaskView(
            task_id=task.task_id,
            vehicle_id=task.vehicle_id,
            selected_pipeline=task.selected_pipeline,
            ul_remaining_bits=task.ul_remaining_bits,
            dl_remaining_bits=task.dl_remaining_bits,
            current_transfer_id=task.current_transfer_id,
        )
        policy_state = PolicyStateView(clock_s=self.state.clock_s)
        decide = getattr(self.policy, "decide", None)
        if callable(decide):
            decision = decide(policy_task, observation, policy_state)
            task.action_audit.append(decision.audit_row())
            raw_scores = getattr(decision, "scores", None)
            frozen_scores = (
                MappingProxyType(dict(raw_scores))
                if isinstance(raw_scores, Mapping)
                else None
            )
            return decision.executed, frozen_scores
        method = getattr(self.policy, "choose_action", None)
        if callable(method):
            return method(policy_task, observation, policy_state), None
        select = getattr(self.policy, "select", None)
        if callable(select):
            return select(mask.allowed, observation), None
        if callable(self.policy):
            return self.policy(mask.allowed, observation), None
        raise TypeError("policy must expose choose_action, select, or be callable")

    def _policy_rescore_current(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: Any,
    ) -> Mapping[Action, float] | None:
        """Causally rescore the current safe set through the policy boundary."""

        rescore = getattr(self.policy, "score_current_actions", None)
        if not callable(rescore):
            return None
        policy_task = PolicyTaskView(
            task_id=task.task_id,
            vehicle_id=task.vehicle_id,
            selected_pipeline=task.selected_pipeline,
            ul_remaining_bits=task.ul_remaining_bits,
            dl_remaining_bits=task.dl_remaining_bits,
            current_transfer_id=task.current_transfer_id,
        )
        policy_state = PolicyStateView(clock_s=self.state.clock_s)
        raw_scores = rescore(policy_task, observation, policy_state, mask=mask)
        if not isinstance(raw_scores, Mapping):
            raise TypeError("policy current-score API must return a mapping")
        missing = tuple(action for action in mask.allowed if action not in raw_scores)
        if missing:
            raise RuntimeError(
                "policy current-score API omitted hard-safe actions: "
                + ", ".join(action.canonical_id for action in missing)
            )
        return MappingProxyType({action: raw_scores[action] for action in mask.allowed})

    def _handle_decision_commit(self, event: Event) -> None:
        task = self.state.tasks.get(event.task_id or "")
        proposed = self._pending_decisions.pop(event.task_id or "", None)
        policy_scores = self._pending_decision_scores.pop(event.task_id or "", None)
        self._pending_decision_due_s.pop(event.task_id or "", None)
        if (
            task is None
            or proposed is None
            or task.terminal
            or task.state not in {TaskState.RAW, TaskState.READY}
        ):
            return
        self._execute_proposed(task, proposed, score=policy_scores)

    def _execute_proposed(
        self,
        task: TaskRecord,
        proposed: Action,
        *,
        score: Mapping[Action, float] | None = None,
    ) -> None:
        observation = self._observation(task)
        current_mask = self.mask_engine.enumerate(task, observation, self.state)
        execution_check_id = (
            f"{task.task_id}|{observation.stage.value}|exec-{len(task.mask_audit):06d}"
        )
        task.mask_audit.append(
            {
                "time_s": self.state.clock_s,
                "stage": observation.stage.value,
                "mask_epoch": "EXECUTION_RECHECK",
                "execution_check_id": execution_check_id,
                "rows": list(current_mask.audit_rows()),
            }
        )
        score_source = (
            "decision_epoch_policy_scores" if score is not None else "generic"
        )
        if proposed not in current_mask.allowed:
            current_scores = self._policy_rescore_current(
                task, observation, current_mask
            )
            if current_scores is not None:
                score = current_scores
                score_source = "current_policy_rescore"
        repair = self.repairer.repair(
            proposed, task, observation, self.state, score=score
        )
        audit = repair.audit_row()
        audit.update(
            {
                "time_s": self.state.clock_s,
                "executed_stage": observation.stage.value,
                "execution_check_id": execution_check_id,
                "repair_score_source": score_source,
            }
        )
        task.action_audit.append(audit)
        self._commit_action(task, repair.executed)

    def _commit_action(self, task: TaskRecord, action: Action) -> None:
        if action.kind is ActionKind.FAIL:
            self._terminate_fail(
                task, FailureReason.POLICY_EXPLICIT_FAIL, "POLICY_FAIL"
            )
        elif action.kind is ActionKind.LOCAL:
            self._start_local_action(task, action.local_model_id or "")
        elif action.kind is ActionKind.PIPE:
            self._start_pipeline_action(task, action.pipeline_id or "")
        elif action.kind is ActionKind.EDGE:
            self._start_edge_action(
                task, action.rsu_id or "", action.edge_model_id or ""
            )

    def _start_local_action(self, task: TaskRecord, model_id: str) -> None:
        if not self._local_model_active(task.vehicle_id, model_id):
            self._terminate_fail(
                task, FailureReason.VERSION_MISMATCH, "LOCAL_VERSION_RECHECK"
            )
            return
        vehicle = self.state.vehicles[task.vehicle_id]
        true_quality = task.true_quality_region
        if true_quality is None:
            raise InvariantViolation(
                "TRUE_QUALITY_REGION_MISSING",
                "actual local trace replay requires a simulator-only true quality region",
                task_id=task.task_id,
            )
        context = self._device_context("vehicle", task.vehicle_id, self.state.clock_s)
        result = self.trace.sample_local_fer(
            model_id,
            true_quality,
            vehicle.device_type,
            context,
            self._rng("local", task.task_id, model_id),
        )
        if not result.supported or result.value is None:
            self._terminate_fail(
                task, FailureReason.UNSUPPORTED, "LOCAL_TRACE_UNSUPPORTED"
            )
            return
        row = result.value
        model_bounds = self.profile.local_models[model_id].deployment_resource_bounds
        reserved_memory = int(model_bounds["max_memory_bytes"])
        if not task.reservation_tokens:
            if not vehicle.reconcile_reservation(
                task, {"accelerator": 1}, reserved_memory
            ):
                self._terminate_fail(
                    task, FailureReason.VEHICLE_CAPACITY, "LOCAL_RESERVATION"
                )
                return
        elif not vehicle.reconcile_reservation(
            task, {"accelerator": 1}, reserved_memory
        ):
            self._terminate_fail(
                task, FailureReason.VEHICLE_CAPACITY, "LOCAL_FALLBACK_MEMORY"
            )
            return
        task.selected_local_model = model_id
        self._local_rows[task.task_id] = row
        TaskStateMachine.transition(
            task,
            TaskState.LOCAL_WAIT,
            time_s=self.state.clock_s,
            trigger="LOCAL_ENQUEUE",
        )
        self._enqueue_job(
            task,
            operation=Operation.LOCAL_FER,
            resource_kind=ResourceKind.ACCELERATOR,
            owner_type="vehicle",
            owner_id=task.vehicle_id,
            work_s=row.service_work_s,
            energy_j=row.dynamic_energy_j,
            memory_bytes=row.memory_bytes,
            version=row.model_hash,
        )

    def _start_pipeline_action(self, task: TaskRecord, pipeline_id: str) -> None:
        if not self._frozen_versions_active():
            self._terminate_fail(
                task, FailureReason.VERSION_MISMATCH, "PIPELINE_VERSION_RECHECK"
            )
            return
        pipeline = self.profile.pipelines[pipeline_id]
        vehicle = self.state.vehicles[task.vehicle_id]
        context = self._device_context("vehicle", task.vehicle_id, self.state.clock_s)
        bounds = pipeline.deployment_resource_bounds
        fallback_memory = 0
        if pipeline.fallback_local_model:
            fallback_memory = int(
                self.profile.local_models[
                    pipeline.fallback_local_model
                ].deployment_resource_bounds["max_memory_bytes"]
            )
        reserved_memory = max(int(bounds["max_peak_memory_bytes"]), fallback_memory)
        if not vehicle.reconcile_reservation(
            task,
            {"accelerator": 1, "cpu": 1, "encoder": 1},
            reserved_memory,
        ):
            self._terminate_fail(
                task, FailureReason.VEHICLE_CAPACITY, "PIPE_RESERVATION"
            )
            return
        if task.true_quality_region is None:
            raise InvariantViolation(
                "TRUE_QUALITY_REGION_MISSING",
                "actual anonymous trace replay requires a simulator-only true quality region",
                task_id=task.task_id,
            )
        actual_result = self.trace.sample_anon_transaction(
            pipeline_id,
            (task.true_quality_region,),
            vehicle.device_type,
            context,
            self._rng("anon", task.task_id, pipeline_id),
        )
        if not actual_result.supported or actual_result.value is None:
            self._terminate_fail(
                task, FailureReason.UNSUPPORTED, "ANON_TRACE_UNSUPPORTED"
            )
            return
        row = actual_result.value
        if row.attempt_count > pipeline.max_attempts:
            raise InvariantViolation(
                "TRACE_RETRY_BOUND",
                "joint trace row exceeds frozen pipeline attempt limit",
                row_id=row.row_id,
                attempts=row.attempt_count,
                max_attempts=pipeline.max_attempts,
            )
        task.selected_pipeline = pipeline_id
        task.trace_row_id = row.row_id
        task.max_attempts = pipeline.max_attempts
        self._anon_rows[task.task_id] = row
        TaskStateMachine.transition(
            task, TaskState.ANON_WAIT, time_s=self.state.clock_s, trigger="PIPE_COMMIT"
        )
        self._enqueue_anon_attempt(task)

    def _enqueue_anon_attempt(self, task: TaskRecord) -> None:
        row = self._anon_rows[task.task_id]
        pipeline = self.profile.pipelines[task.selected_pipeline or ""]
        if task.attempt_started_count >= min(pipeline.max_attempts, row.attempt_count):
            self._fallback_or_fail(task, FailureReason.ANON_FAILED)
            return
        # Counter increases only after this concrete ANON job is successfully
        # inserted into the finite accelerator queue.
        attempt = row.attempts[task.attempt_started_count]
        self._enqueue_job(
            task,
            operation=Operation.ANON,
            resource_kind=ResourceKind.ACCELERATOR,
            owner_type="vehicle",
            owner_id=task.vehicle_id,
            work_s=attempt.anon_work_s,
            energy_j=attempt.anon_energy_j,
            memory_bytes=attempt.peak_memory_bytes,
            version=pipeline.pipeline_hash,
        )
        task.mark_anon_enqueued(pipeline.max_attempts)

    def _retry_or_fallback(self, task: TaskRecord, reason: FailureReason) -> None:
        self._finalize_anon_attempt_audit(task, failure_reason=reason)
        row = self._anon_rows[task.task_id]
        pipeline = self.profile.pipelines[task.selected_pipeline or ""]
        retryable = reason.value in set(pipeline.retryable_reasons)
        if (
            retryable
            and task.attempt_started_count < pipeline.max_attempts
            and task.attempt_started_count < row.attempt_count
            and self.state.clock_s < task.absolute_deadline_s
        ):
            TaskStateMachine.transition(
                task,
                TaskState.ANON_WAIT,
                time_s=self.state.clock_s,
                trigger="ANON_RETRY",
                detail=reason.value,
            )
            self._enqueue_anon_attempt(task)
            return
        self._fallback_or_fail(task, reason)

    def _finalize_anon_attempt_audit(
        self, task: TaskRecord, *, failure_reason: FailureReason | None
    ) -> None:
        """Close one measured joint attempt without dropping failed work."""

        if not task.anon_attempt_audit:
            return
        audit = task.anon_attempt_audit[-1]
        if "attempt_terminal_time_s" in audit:
            return
        row = self._anon_rows[task.task_id]
        attempt = row.attempts[task.current_attempt_index]
        start = float(audit["attempt_started_time_s"])
        work_s = attempt.anon_work_s
        energy_j = attempt.anon_energy_j
        if "guard_completed_time_s" in audit:
            work_s += attempt.guard_work_s or 0.0
            energy_j += attempt.guard_energy_j or 0.0
        if "encode_completed_time_s" in audit:
            work_s += attempt.encode_work_s or 0.0
            energy_j += attempt.encode_energy_j or 0.0
        audit.update(
            {
                "attempt_terminal_time_s": self.state.clock_s,
                "latency_s": max(0.0, self.state.clock_s - start),
                "executed_work_s": work_s,
                "vehicle_energy_j": energy_j,
                "failure_reason": (
                    None if failure_reason is None else failure_reason.value
                ),
            }
        )

    def _finish_encode(self, task: TaskRecord) -> None:
        row = self._anon_rows[task.task_id]
        attempt = row.attempts[task.current_attempt_index]
        pipeline = self.profile.pipelines[task.selected_pipeline or ""]
        task.anon_attempt_audit[-1]["encode_success"] = attempt.encode_success
        task.anon_attempt_audit[-1]["encoded_size_bytes"] = attempt.encoded_size_bytes
        task.anon_attempt_audit[-1]["encode_completed_time_s"] = self.state.clock_s
        task.actual_path.append(f"ENCODE#{attempt.attempt_index}")
        if (
            not attempt.encode_success
            or not attempt.encoded_size_bytes
            or not attempt.artifact_key
            or not row.formed_packet
        ):
            self._retry_or_fallback(task, FailureReason.ENCODE_FAILED)
            return
        if (
            attempt.encoded_size_bytes != row.final_encoded_size_bytes
            or attempt.artifact_key != row.artifact_key
            or attempt.encoded_size_bytes
            > int(pipeline.deployment_resource_bounds["max_output_bytes"])
        ):
            self._finalize_anon_attempt_audit(
                task, failure_reason=FailureReason.ENCODE_SIZE_OOD
            )
            self._fallback_or_fail(task, FailureReason.ENCODE_SIZE_OOD)
            return
        self._finalize_anon_attempt_audit(task, failure_reason=None)
        payload_seed = hashlib.sha256(
            f"{task.task_id}|{attempt.artifact_key}".encode()
        ).digest()
        payload = (
            payload_seed * ((attempt.encoded_size_bytes // len(payload_seed)) + 1)
        )[: attempt.encoded_size_bytes]
        anonymized = _replay_anonymization_success(
            aligned=task.aligned_handle,
            task_id=task.task_id,
            pipeline_id=pipeline.pipeline_id,
            pipeline_hash=pipeline.pipeline_hash,
            artifact_key=attempt.artifact_key,
            attempt=attempt.attempt_index,
        )
        guarded = _replay_guard_success(
            anonymized,
            guard_hash=pipeline.guard_hash,
            guard_certificate_id=f"guard:{task.task_id}:{attempt.attempt_index}",
        )
        encoding = _replay_encoding_success(
            guarded,
            payload=payload,
            encoder_hash=pipeline.encoder_hash,
            encoded_size_bytes=attempt.encoded_size_bytes,
        )
        task.encoded_anon = _finalize_encoded_anon(
            encoding,
            profile_hash=self.profile.profile_hash,
            quality_bins=task.conformal_quality_bins,
        )
        task.artifact_key = attempt.artifact_key
        task.encoded_size_bytes = attempt.encoded_size_bytes
        fallback = pipeline.fallback_local_model
        ready_tokens = {"accelerator": 1} if fallback else {}
        ready_memory = (
            int(
                self.profile.local_models[fallback].deployment_resource_bounds[
                    "max_memory_bytes"
                ]
            )
            if fallback
            else attempt.encoded_size_bytes
        )
        if not self.state.vehicles[task.vehicle_id].reconcile_reservation(
            task, ready_tokens, ready_memory
        ):
            raise InvariantViolation(
                "READY_SHADOW_RESERVATION_LOST",
                "pipeline completion could not retain its preregistered fallback/buffer reservation",
                task_id=task.task_id,
            )
        if not fallback:
            task.raw_handle = None
            task.aligned_handle = None
        TaskStateMachine.transition(
            task, TaskState.READY, time_s=self.state.clock_s, trigger="ENCODE_DONE"
        )

    def _start_edge_action(self, task: TaskRecord, rsu_id: str, model_id: str) -> None:
        if task.encoded_anon is None:
            self._terminate_fail(
                task, FailureReason.UNSUPPORTED, "EDGE_WITHOUT_ENCODED_ANON"
            )
            return
        model = self.profile.edge_models[model_id]
        live_rsu = self.state.rsus[rsu_id]
        if live_rsu.failed:
            self._fallback_or_fail(task, FailureReason.EDGE_FAILED)
            return
        if (
            self._active_protocol_version != self.config.protocol_version
            or self._active_profile_hash != self.profile.profile_hash
        ):
            self._fallback_or_fail(task, FailureReason.VERSION_MISMATCH)
            return
        compatibility = self.profile.validate_compatibility(
            protocol_version=self.config.protocol_version,
            profile_hash=task.encoded_anon.profile_hash,
            pipeline_id=task.encoded_anon.pipeline_id,
            pipeline_hash=task.encoded_anon.pipeline_hash,
            guard_hash=task.encoded_anon.guard_hash,
            encoder_hash=task.encoded_anon.encoder_hash,
            edge_model_id=model_id,
            edge_model_hash=model.model_hash,
            device_type=self.state.vehicles[task.vehicle_id].device_type,
            rsu_id=rsu_id,
        )
        segment = self._wireless_segment(
            task.vehicle_id, rsu_id, TransferDirection.UL, self.state.clock_s
        )
        if (
            not compatibility.compatible
            or live_rsu.admission.cached_models.get(model_id) != model.model_hash
        ):
            self._fallback_or_fail(task, FailureReason.VERSION_MISMATCH)
            return
        if segment is None:
            self._fallback_or_fail(task, FailureReason.UL_FAILED)
            return
        if segment.link_state != "connected":
            self._fallback_or_fail(
                task,
                FailureReason.PERMANENT_LINK_LOSS
                if segment.link_state == "permanent_loss"
                else FailureReason.UL_FAILED,
            )
            return
        task.selected_rsu = rsu_id
        task.selected_edge_model = model_id
        packet = AnonFERRequest.from_encoded(
            task.encoded_anon,
            protocol_version=self.config.protocol_version,
            requested_edge_model=model_id,
            requested_edge_model_hash=model.model_hash,
            vehicle_id=task.vehicle_id,
            task_id=task.task_id,
        )
        total_bits = float(packet.payload_bits + self.config.metadata_bits)
        self._transfer_counter += 1
        transfer_id = f"transfer-{self._transfer_counter:08d}"
        transfer = Transfer(
            transfer_id,
            task.task_id,
            task.vehicle_id,
            rsu_id,
            TransferDirection.UL,
            packet,
            total_bits,
            total_bits,
            self.state.clock_s,
            self.state.clock_s,
        )
        self.state.transfers[transfer_id] = transfer
        task.current_transfer_id = transfer_id
        task.ul_remaining_bits = total_bits
        task.actual_path.append(f"UL:{rsu_id}")
        task.network_audit.append(
            {
                "direction": "UL",
                "status": "START",
                "time_s": self.state.clock_s,
                "total_bits": total_bits,
                "rsu_id": rsu_id,
            }
        )
        TaskStateMachine.transition(
            task, TaskState.UL, time_s=self.state.clock_s, trigger="UL_START"
        )

    def _admit_at_rsu(self, task: TaskRecord, transfer: Transfer) -> None:
        rsu = self.state.rsus[transfer.rsu_id]
        if rsu.failed:
            before = rsu.admission.snapshot()
            task.rsu_audit.append(
                {
                    "admission": "REJECT",
                    "time_s": self.state.clock_s,
                    "reason_codes": ["RSU_FAILED"],
                }
            )
            if before != rsu.admission.snapshot():
                raise InvariantViolation(
                    "ADMISSION_REJECT_SIDE_EFFECT",
                    "failed-RSU admission rejection mutated RSU state",
                    task_id=task.task_id,
                )
            self._fallback_or_fail(task, FailureReason.ADMISSION_REJECTED)
            return
        model = self.profile.edge_models[task.selected_edge_model or ""]
        pipeline = self.profile.pipelines.get(task.selected_pipeline or "")
        current_rsu_context = self._device_context(
            "rsu", transfer.rsu_id, self.state.clock_s
        )
        admission_observation = self._observation(task, stage=ActionStage.READY)
        bounds = self.estimator.runtime_admission_bounds(
            action=Action.edge(transfer.rsu_id, model.model_id),
            observation=admission_observation,
            evaluation_pair_supported=self._runtime_evaluation_pair_supported(
                task,
                rsu_id=transfer.rsu_id,
                model_id=model.model_id,
                pipeline_id=task.selected_pipeline or "",
            ),
            rsu_context=current_rsu_context,
        )
        if bounds is None:
            before = rsu.admission.snapshot()
            task.rsu_audit.append(
                {
                    "admission": "REJECT",
                    "time_s": self.state.clock_s,
                    "reason_codes": ["PAIRED_MEASUREMENT_MISSING"],
                    "certificate_source": "profile_plus_identity_free_scenario",
                }
            )
            if before != rsu.admission.snapshot():
                raise InvariantViolation(
                    "ADMISSION_REJECT_SIDE_EFFECT",
                    "unsupported admission certificate mutated RSU state",
                    task_id=task.task_id,
                )
            self._fallback_or_fail(task, FailureReason.ADMISSION_REJECTED)
            return
        packet = transfer.packet
        message_valid = (
            isinstance(packet, AnonFERRequest)
            and pipeline is not None
            and packet.artifact_key == task.artifact_key
            and packet.pipeline_id == pipeline.pipeline_id
            and packet.pipeline_hash == pipeline.pipeline_hash
            and packet.guard_hash == pipeline.guard_hash
            and packet.encoder_hash == pipeline.encoder_hash
            and packet.profile_hash
            == self.profile.profile_hash
            == self._active_profile_hash
            and packet.protocol_version
            == self.config.protocol_version
            == self._active_protocol_version
            and packet.requested_edge_model == model.model_id
            and packet.requested_edge_model_hash == model.model_hash
            and packet.quality_bins == task.conformal_quality_bins
            and packet.payload_size_bytes == task.encoded_size_bytes
            and bounds.rsu_id == transfer.rsu_id
            and bounds.model_id == model.model_id
            and bounds.model_hash == model.model_hash
            and bounds.pipeline_id == task.selected_pipeline
            and bounds.protocol_version == self.config.protocol_version
            and bounds.profile_hash == self.profile.profile_hash
            and bounds.candidate_quality_bins == task.conformal_quality_bins
        )
        request = AdmissionRequest(
            task.task_id,
            bounds.descriptor_count,
            bounds.max_vram_bytes,
            bounds.max_gpu_work_s,
            bounds.model_id,
            bounds.model_hash,
            bounds.protocol_version,
            message_valid,
        )
        request_audit = {
            "descriptor_count": request.descriptor_count,
            "vram_bytes": request.vram_bytes,
            "conservative_work_gpu_s": request.conservative_work_gpu_s,
            "model_id": request.model_id,
            "model_hash": request.model_hash,
            "protocol_version": request.protocol_version,
            "profile_hash": bounds.profile_hash,
            "pipeline_id": bounds.pipeline_id,
            "candidate_quality_bins": list(bounds.candidate_quality_bins),
            "message_valid": request.message_valid,
            "certificate_source": "profile_deployment_resource_bounds",
            "scenario_trace_version": bounds.scenario_trace_version,
            "scenario_trace_hash": bounds.scenario_trace_hash,
            "scenario_split_role": bounds.scenario_split_role,
        }

        # Only the post-certificate evaluation realization uses simulator-only
        # g*.  It never contributes a numeric value to the admission request.
        # Loader validation already proves every evaluation row lies inside the
        # profile bounds; these checks remain as a fail-closed runtime guard.
        if task.true_quality_region is None:
            raise InvariantViolation(
                "TRUE_QUALITY_REGION_MISSING",
                "actual edge FER replay requires a simulator-only true quality region",
                task_id=task.task_id,
            )
        edge_result = self.trace.sample_edge_fer(
            transfer.rsu_id,
            model.model_id,
            task.selected_pipeline or "",
            task.artifact_key or "",
            task.true_quality_region,
            current_rsu_context,
            self._rng(
                "edge",
                task.task_id,
                f"{transfer.rsu_id}|{model.model_id}|{task.artifact_key}",
            ),
        )
        if not edge_result.supported or edge_result.value is None:
            before = rsu.admission.snapshot()
            task.rsu_audit.append(
                {
                    "admission": "REJECT",
                    "time_s": self.state.clock_s,
                    "reason_codes": ["PAIRED_MEASUREMENT_MISSING"],
                    "request": request_audit,
                }
            )
            if before != rsu.admission.snapshot():
                raise InvariantViolation(
                    "ADMISSION_REJECT_SIDE_EFFECT",
                    "missing realized edge pairing mutated RSU state",
                    task_id=task.task_id,
                )
            self._fallback_or_fail(task, FailureReason.ADMISSION_REJECTED)
            return
        row = edge_result.value
        within_certificate = bool(
            row.rsu_id == bounds.rsu_id
            and row.model_id == bounds.model_id
            and row.model_hash == bounds.model_hash
            and row.pipeline_id == bounds.pipeline_id
            and row.quality_bin in bounds.candidate_quality_bins
            and row.context == current_rsu_context
            and row.vram_bytes <= bounds.max_vram_bytes
            and row.ingress_work_s <= bounds.max_ingress_work_s + EPS
            and row.ingress_energy_j <= bounds.max_ingress_energy_j + EPS
            and row.gpu_work_s <= bounds.max_gpu_work_s + EPS
            and row.gpu_energy_j <= bounds.max_gpu_energy_j + EPS
            and row.result_size_bits <= bounds.max_result_size_bits
        )
        if not within_certificate:
            before = rsu.admission.snapshot()
            task.rsu_audit.append(
                {
                    "admission": "REJECT",
                    "time_s": self.state.clock_s,
                    "reason_codes": ["OOD", "DEPLOYMENT_RESOURCE_BOUND_EXCEEDED"],
                    "request": request_audit,
                }
            )
            if before != rsu.admission.snapshot():
                raise InvariantViolation(
                    "ADMISSION_REJECT_SIDE_EFFECT",
                    "out-of-profile edge row mutated RSU state",
                    task_id=task.task_id,
                )
            self._fallback_or_fail(task, FailureReason.OOD)
            return
        self._edge_rows[task.task_id] = row
        before = rsu.admission.snapshot()
        accepted, reasons = rsu.admission.admit(request)
        if not accepted:
            after = rsu.admission.snapshot()
            if before != after:
                raise InvariantViolation(
                    "ADMISSION_REJECT_SIDE_EFFECT",
                    "rejected admission mutated RSU state",
                    task_id=task.task_id,
                )
            task.rsu_audit.append(
                {
                    "admission": "REJECT",
                    "time_s": self.state.clock_s,
                    "reason_codes": list(reasons),
                    "request": request_audit,
                }
            )
            self._fallback_or_fail(task, FailureReason.ADMISSION_REJECTED)
            return
        task.rsu_reserved = True
        task.rsu_audit.append(
            {
                "admission": "ACCEPT",
                "time_s": self.state.clock_s,
                "vram_bytes": bounds.max_vram_bytes,
                "work_gpu_s": bounds.max_gpu_work_s,
                "pinned_model_hash": model.model_hash,
                "request": request_audit,
            }
        )
        TaskStateMachine.transition(
            task, TaskState.EDGE_WAIT, time_s=self.state.clock_s, trigger="RSU_ADMIT"
        )
        self._enqueue_job(
            task,
            operation=Operation.RSU_INGRESS,
            resource_kind=ResourceKind.RSU_INGRESS_CPU,
            owner_type="rsu",
            owner_id=transfer.rsu_id,
            work_s=row.ingress_work_s,
            energy_j=row.ingress_energy_j,
            memory_bytes=0,
            version=self.config.protocol_version,
        )

    def _finish_edge(self, task: TaskRecord) -> None:
        row = self._edge_rows[task.task_id]
        rsu_id = task.selected_rsu or ""
        rsu = self.state.rsus[rsu_id]
        pinned = rsu.admission.pinned_model(task.task_id)
        expected_hash = self.profile.edge_models[
            task.selected_edge_model or ""
        ].model_hash
        if pinned != (task.selected_edge_model, expected_hash):
            raise InvariantViolation(
                "MODEL_PIN_LOST",
                "admitted task did not retain its pinned model",
                task_id=task.task_id,
            )
        # GPU/VRAM/model reservation is released when inference has produced a
        # standalone result packet; downlink no longer needs those resources.
        rsu.admission.release(task.task_id)
        task.rsu_reserved = False
        task.actual_path.append(f"EDGE:{rsu_id}")
        if row.failed or row.fer_loss is None:
            self._fallback_or_fail(task, FailureReason.EDGE_FAILED)
            return
        result = FERResult(
            task.task_id,
            task.selected_edge_model or "",
            expected_hash,
            self.config.protocol_version,
            result_code=0,
            valid=True,
            size_bits=row.result_size_bits,
        )
        task.realized_fer_loss = row.fer_loss
        task.realized_fer_true_label = row.true_label
        task.realized_fer_class_probabilities = row.class_probabilities
        task.evaluation_subject_cluster_id = self._anon_rows[
            task.task_id
        ].subject_cluster_id
        self._transfer_counter += 1
        transfer_id = f"transfer-{self._transfer_counter:08d}"
        transfer = Transfer(
            transfer_id,
            task.task_id,
            task.vehicle_id,
            rsu_id,
            TransferDirection.DL,
            result,
            float(row.result_size_bits),
            float(row.result_size_bits),
            self.state.clock_s,
            self.state.clock_s,
        )
        self.state.transfers[transfer_id] = transfer
        task.current_transfer_id = transfer_id
        task.dl_remaining_bits = float(row.result_size_bits)
        task.actual_path.append(f"DL:{rsu_id}")
        task.network_audit.append(
            {
                "direction": "DL",
                "status": "START",
                "time_s": self.state.clock_s,
                "total_bits": row.result_size_bits,
                "rsu_id": rsu_id,
            }
        )
        TaskStateMachine.transition(
            task, TaskState.DL, time_s=self.state.clock_s, trigger="EDGE_DONE"
        )

    def _fail_transfer(self, transfer: Transfer, reason: FailureReason) -> None:
        if transfer.status not in {TransferStatus.ACTIVE, TransferStatus.PAUSED}:
            return
        transfer.status = TransferStatus.CANCELLED
        transfer.completion_version += 1
        task = self.state.tasks[transfer.task_id]
        task.current_transfer_id = None
        task.network_audit.append(
            {
                "direction": transfer.direction.value,
                "status": "FAIL",
                "time_s": self.state.clock_s,
                "reason": reason.value,
                "delivered_bits": transfer.delivered_bits,
                "remaining_bits": transfer.remaining_bits,
                "rsu_id": transfer.rsu_id,
                "vehicle_energy_j": transfer.vehicle_energy_j,
                "rsu_energy_j": transfer.rsu_energy_j,
            }
        )
        # The old partial packet is deleted and never assigned to a new RSU.
        if transfer.direction is TransferDirection.UL:
            task.ul_remaining_bits = 0.0
        else:
            task.dl_remaining_bits = 0.0
        self._fallback_or_fail(task, reason)

    def _fallback_or_fail(self, task: TaskRecord, reason: FailureReason) -> None:
        if task.terminal:
            return
        # A local path is already the terminal technical fallback.  Retrying
        # the same model from LOCAL_WAIT/RUN would either create a loop or
        # violate the closed state relation (LOCAL_RUN -> LOCAL_WAIT).
        if task.state in {TaskState.LOCAL_WAIT, TaskState.LOCAL_RUN}:
            self._terminate_fail(task, reason, "LOCAL_FALLBACK_EXHAUSTED")
            return
        if task.current_transfer_id:
            transfer = self.state.transfers.get(task.current_transfer_id)
            if transfer is not None and transfer.status in {
                TransferStatus.ACTIVE,
                TransferStatus.PAUSED,
            }:
                self._cancel_transfers_with_audit(task, reason, "FALLBACK")
            task.current_transfer_id = None
            task.ul_remaining_bits = 0.0
            task.dl_remaining_bits = 0.0
        if task.selected_rsu and task.selected_rsu in self.state.rsus:
            self.state.rsus[task.selected_rsu].admission.release(task.task_id)
            task.rsu_reserved = False
        pipeline = self.profile.pipelines.get(task.selected_pipeline or "")
        fallback = None if pipeline is None else pipeline.fallback_local_model
        if fallback and self._local_fallback_feasible(task, fallback):
            self._start_local_action(task, fallback)
            task.action_audit.append(
                {
                    "time_s": self.state.clock_s,
                    "repair": "FROZEN_LOCAL_FALLBACK",
                    "trigger": reason.value,
                    "model_id": fallback,
                }
            )
            return
        self._terminate_fail(task, reason, "FALLBACK_UNAVAILABLE")

    def _local_fallback_feasible(self, task: TaskRecord, model_id: str) -> bool:
        if self.state.clock_s >= task.absolute_deadline_s - EPS:
            return False
        if not self._local_model_active(task.vehicle_id, model_id):
            return False
        vehicle = self.state.vehicles[task.vehicle_id]
        if vehicle.failed:
            return False
        context = self._device_context("vehicle", task.vehicle_id, self.state.clock_s)
        rows = [
            row
            for row in self.scenario_trace.local_rows
            if row.model_id == model_id
            and row.device_type == vehicle.device_type
            and row.quality_bin in task.conformal_quality_bins
            and row.context == context
        ]
        if {row.quality_bin for row in rows} != set(task.conformal_quality_bins):
            return False
        min_work = min(row.service_work_s for row in rows)
        max_energy = max(row.dynamic_energy_j for row in rows)
        model_bounds = self.profile.local_models[model_id].deployment_resource_bounds
        max_memory = int(model_bounds["max_memory_bytes"])
        rate = self._resource_rate(
            "vehicle", task.vehicle_id, "accelerator", self.state.clock_s
        )
        if (
            rate <= 0
            or min_work / rate > task.absolute_deadline_s - self.state.clock_s + EPS
        ):
            return False
        if max_energy > vehicle.battery_j + EPS:
            return False
        effective_reserved = max(task.memory_reservation_bytes, max_memory)
        extra = effective_reserved - task.memory_reservation_bytes
        if vehicle.memory_reserved_bytes + extra > vehicle.memory_capacity_bytes:
            return False
        if not task.reservation_tokens:
            reserve_ok, _ = vehicle.can_reserve({"accelerator": 1}, max_memory)
            if not reserve_ok:
                return False
        return True

    def _frozen_versions_active(self) -> bool:
        """Return whether the loaded frozen bundle is still deployable now.

        A profile/protocol transition does not mutate the frozen object.  It
        invalidates new work and deterministic fallback that would otherwise
        bypass the normal hard-mask/repair recheck.
        """

        return (
            self._active_profile_hash == self.profile.profile_hash
            and self._active_protocol_version == self.config.protocol_version
        )

    def _local_model_active(self, vehicle_id: str, model_id: str) -> bool:
        model = self.profile.local_models.get(model_id)
        return (
            self._frozen_versions_active()
            and model is not None
            and self._active_local_model_hashes.get((vehicle_id, model_id))
            == model.model_hash
        )

    def _enqueue_job(
        self,
        task: TaskRecord,
        *,
        operation: Operation,
        resource_kind: ResourceKind,
        owner_type: str,
        owner_id: str,
        work_s: float,
        energy_j: float,
        memory_bytes: int,
        version: str,
    ) -> ComputeJob:
        if task.current_job_id is not None:
            raise InvariantViolation(
                "TASK_JOB_OVERLAP",
                "task attempted to own two compute jobs",
                task_id=task.task_id,
            )
        if work_s <= 0 or energy_j < 0:
            raise InvariantViolation(
                "TRACE_JOB_PHYSICS",
                "trace job work/energy is invalid",
                task_id=task.task_id,
            )
        if owner_type == "vehicle":
            if memory_bytes > task.memory_reservation_bytes:
                raise InvariantViolation(
                    "VEHICLE_JOB_MEMORY_UNRESERVED",
                    "vehicle job memory exceeds the task's preregistered shadow reservation",
                    task_id=task.task_id,
                    operation=operation.value,
                    job_memory_bytes=memory_bytes,
                    reserved_memory_bytes=task.memory_reservation_bytes,
                )
            key = {
                ResourceKind.ACCELERATOR: "accelerator",
                ResourceKind.CPU: "cpu",
                ResourceKind.ENCODER: "encoder",
            }[resource_kind]
            pool = self.state.vehicles[owner_id].resources[key]
        else:
            runtime = self.state.rsus[owner_id]
            pool = (
                runtime.ingress
                if resource_kind is ResourceKind.RSU_INGRESS_CPU
                else runtime.gpu
            )
        self._job_counter += 1
        job_id = f"job-{self._job_counter:08d}"
        job = ComputeJob(
            job_id,
            task.task_id,
            owner_type,
            owner_id,
            operation,
            resource_kind,
            version,
            self.state.clock_s,
            task.absolute_deadline_s,
            pool.next_enqueue_seq(),
            work_s,
            work_s,
            energy_j,
            memory_need_bytes=memory_bytes,
        )
        pool.enqueue(job)
        task.current_job_id = job_id
        task.record_time("enqueue", operation.value, self.state.clock_s)
        return job

    def _close_interrupted_anon_attempt(
        self,
        task: TaskRecord,
        job: ComputeJob,
        reason: FailureReason,
        trigger: str,
    ) -> None:
        """Close an interrupted joint attempt using only work already served."""

        if job.operation not in {Operation.ANON, Operation.GUARD, Operation.ENCODE}:
            return
        row = self._anon_rows.get(task.task_id)
        index = task.current_attempt_index
        if row is None or index < 0 or index >= len(row.attempts):
            raise InvariantViolation(
                "ANON_AUDIT_TRACE_MISSING",
                "an interrupted anonymization job lacks its paired joint trace row",
                task_id=task.task_id,
                operation=job.operation.value,
            )
        attempt = row.attempts[index]
        audit = next(
            (
                item
                for item in reversed(task.anon_attempt_audit)
                if item.get("attempt") == attempt.attempt_index
                and "attempt_terminal_time_s" not in item
            ),
            None,
        )
        if audit is None:
            starts = task.start_times.get(Operation.ANON.value, [])
            start = starts[index] if index < len(starts) else job.start_time_s
            audit = {
                "attempt": attempt.attempt_index,
                "time_s": self.state.clock_s,
                "attempt_enqueued_time_s": job.enqueue_time_s,
                "attempt_started_time_s": start,
            }
            task.anon_attempt_audit.append(audit)

        phase_work_s = max(0.0, job.total_work_s - job.residual_work_s)
        phase_energy_j = max(0.0, job.consumed_dynamic_energy_j)
        completed_work_s = 0.0
        completed_energy_j = 0.0
        if job.operation in {Operation.GUARD, Operation.ENCODE}:
            completed_work_s += attempt.anon_work_s
            completed_energy_j += attempt.anon_energy_j
        if job.operation is Operation.ENCODE:
            completed_work_s += attempt.guard_work_s or 0.0
            completed_energy_j += attempt.guard_energy_j or 0.0
        start_value = audit.get("attempt_started_time_s")
        latency_s = (
            0.0
            if start_value is None
            else max(0.0, self.state.clock_s - float(start_value))
        )
        audit.update(
            {
                "attempt_terminal_time_s": self.state.clock_s,
                "latency_s": latency_s,
                "executed_work_s": completed_work_s + phase_work_s,
                "vehicle_energy_j": completed_energy_j + phase_energy_j,
                "failure_reason": reason.value,
                "termination_trigger": trigger,
                "interrupted_phase": job.operation.value,
                "phase_executed_work_s": phase_work_s,
                "phase_dynamic_energy_j": phase_energy_j,
                "phase_residual_work_s": job.residual_work_s,
                "attempt_cancelled": True,
            }
        )

    def _cancel_compute_jobs_with_audit(
        self,
        task: TaskRecord,
        reason: FailureReason,
        trigger: str,
    ) -> None:
        """Cancel task compute atomically after recording its partial service."""

        for owner_type, owner_id, resource_name, pool in self._all_pools():
            active = pool.active_jobs_for_task(task.task_id)
            for job in active:
                executed_work_s = max(0.0, job.total_work_s - job.residual_work_s)
                queue_end_s = (
                    self.state.clock_s if job.start_time_s is None else job.start_time_s
                )
                audit = {
                    "time_s": self.state.clock_s,
                    "status": "CANCELLED",
                    "reason": reason.value,
                    "trigger": trigger,
                    "job_id": job.job_id,
                    "operation": job.operation.value,
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "resource": resource_name,
                    "status_before_cancel": job.status.value,
                    "enqueue_time_s": job.enqueue_time_s,
                    "start_time_s": job.start_time_s,
                    "queue_wait_s": max(0.0, queue_end_s - job.enqueue_time_s),
                    "service_elapsed_s": (
                        0.0
                        if job.start_time_s is None
                        else max(0.0, self.state.clock_s - job.start_time_s)
                    ),
                    "total_work_s": job.total_work_s,
                    "executed_work_s": executed_work_s,
                    "residual_work_s": job.residual_work_s,
                    "dynamic_energy_j": job.consumed_dynamic_energy_j,
                }
                task.compute_audit.append(audit)
                task.record_time("end", job.operation.value, self.state.clock_s)
                self._close_interrupted_anon_attempt(task, job, reason, trigger)
                if owner_type == "rsu":
                    task.rsu_audit.append(
                        {
                            "phase": f"{resource_name}_cancel",
                            **audit,
                        }
                    )
            if active:
                pool.cancel_task(task.task_id, self.state.clock_s)

    def _cancel_transfers_with_audit(
        self,
        task: TaskRecord,
        reason: FailureReason,
        trigger: str,
    ) -> None:
        """Cancel active radio objects while preserving paired partial costs."""

        for transfer in sorted(
            self.state.transfers.values(), key=lambda item: item.transfer_id
        ):
            if transfer.task_id != task.task_id or transfer.status not in {
                TransferStatus.ACTIVE,
                TransferStatus.PAUSED,
            }:
                continue
            task.network_audit.append(
                {
                    "direction": transfer.direction.value,
                    "status": "FAIL",
                    "time_s": self.state.clock_s,
                    "reason": reason.value,
                    "trigger": trigger,
                    "status_before_cancel": transfer.status.value,
                    "start_time_s": transfer.start_time_s,
                    "elapsed_s": max(0.0, self.state.clock_s - transfer.start_time_s),
                    "total_bits": transfer.total_bits,
                    "delivered_bits": transfer.delivered_bits,
                    "remaining_bits": transfer.remaining_bits,
                    "rsu_id": transfer.rsu_id,
                    "vehicle_energy_j": transfer.vehicle_energy_j,
                    "rsu_energy_j": transfer.rsu_energy_j,
                }
            )
            transfer.status = TransferStatus.CANCELLED
            transfer.completion_version += 1

    def _terminate_fail(
        self, task: TaskRecord, reason: FailureReason, trigger: str
    ) -> None:
        if task.terminal:
            return
        self._cancel_compute_jobs_with_audit(task, reason, trigger)
        self._cancel_transfers_with_audit(task, reason, trigger)
        for rsu in self.state.rsus.values():
            rsu.admission.release(task.task_id)
        task.current_job_id = None
        task.current_transfer_id = None
        task.ul_remaining_bits = 0.0
        task.dl_remaining_bits = 0.0
        task.rsu_reserved = False
        task.result_valid = False
        TaskStateMachine.transition(
            task,
            TaskState.FAIL,
            time_s=self.state.clock_s,
            trigger=trigger,
            failure_reason=reason,
        )
        self._cleanup_terminal(task)

    def _cleanup_terminal(self, task: TaskRecord) -> None:
        self.state.events.cancel_task(task.task_id)
        self.state.vehicles[task.vehicle_id].release(task)
        for transfer_id in sorted(
            transfer_id
            for transfer_id, transfer in self.state.transfers.items()
            if transfer.task_id == task.task_id
        ):
            del self.state.transfers[transfer_id]
        task.raw_handle = None
        task.aligned_handle = None
        task.encoded_anon = None
        self._decision_score_buffer.pop(task.task_id, None)
        self._pending_decisions.pop(task.task_id, None)
        self._pending_decision_scores.pop(task.task_id, None)
        self._pending_decision_due_s.pop(task.task_id, None)
        callback = getattr(self.policy, "on_task_terminal", None)
        if callable(callback):
            callback(task.task_id)

    def _log_compound(self, batch: list[Event]) -> None:
        self.state.event_log.append(
            {
                "time_s": self.state.clock_s,
                "events": [
                    {
                        "kind": event.kind.value,
                        "priority": event.priority,
                        "seq": event.seq,
                        "task_id": event.task_id,
                        "object_id": event.object_id,
                    }
                    for event in sorted(
                        batch, key=lambda item: (item.priority, item.seq)
                    )
                ],
                "task_states": {
                    task_id: task.state.value
                    for task_id, task in sorted(self.state.tasks.items())
                },
            }
        )


def run_from_config(
    config_path: str | Path, policy: Any, *, policy_name: str | None = None
) -> RunResult:
    simulator = DiscreteEventSimulator.from_config_path(
        config_path, policy, policy_name=policy_name
    )
    return simulator.run()


__all__ = ["DiscreteEventSimulator", "RunResult", "run_from_config"]
