"""Auditable task, event, resource and system metrics.

The ledger never serializes protected vehicle-domain handles or anonymous
payload bytes.  It records identifiers, sizes, service, energy and evidence
metadata only.  All deterministic simulation outputs participate in a
canonical core digest; wall-clock controller diagnostics do not.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import SimulationConfig
from .enums import FailureReason, JobStatus, TaskState
from .manifest import canonical_core_digest, prepare_output_directory, sha256_file
from .packets import (
    AlignedTensorHandle,
    AnonFERRequest,
    EncodedAnon,
    FERResult,
    RawImageHandle,
)
from .profiles import canonical_json_bytes
from .resources import ResourcePool
from .state import SimulationState


TASK_COLUMNS = (
    "task_id",
    "vehicle_id",
    "state",
    "arrival_time_s",
    "relative_deadline_s",
    "absolute_deadline_s",
    "terminal_time_s",
    "terminal_latency_s",
    "cost_latency_s",
    "done",
    "failed",
    "timeout",
    "result_valid",
    "failure_reason",
    "realized_fer_loss",
    "all_task_loss",
    "failure_penalty_cost",
    "selected_pipeline",
    "selected_local_model",
    "selected_rsu",
    "selected_edge_model",
    "attempt_started_count",
    "max_attempts",
    "trace_row_id",
    "artifact_key",
    "encoded_size_bytes",
    "ul_remaining_bits",
    "dl_remaining_bits",
    "vehicle_compute_radio_energy_j",
    "vehicle_hold_energy_j",
    "vehicle_attributed_energy_j",
    "rsu_compute_radio_energy_j",
    "rsu_hold_energy_j",
    "rsu_attributed_energy_j",
    "actual_path_json",
    "phase_history_json",
    "enqueue_times_json",
    "start_times_json",
    "end_times_json",
    "compute_audit_json",
    "anon_attempts_json",
    "network_audit_json",
    "rsu_audit_json",
)

RESOURCE_COLUMNS = (
    "time_s",
    "owner_type",
    "owner_id",
    "resource_id",
    "resource_kind",
    "server_count",
    "waiting_jobs",
    "running_jobs",
    "residual_work_s",
    "busy_server_seconds",
    "utilization",
    "max_running_observed",
    "physical_energy_j",
    "system_maintenance_energy_j",
    "memory_reserved_bytes",
    "memory_capacity_bytes",
    "descriptor_reserved_json",
    "descriptor_capacity_json",
    "admitted_descriptors",
    "descriptor_capacity",
    "reserved_vram_bytes",
    "vram_capacity_bytes",
    "reserved_work_gpu_s",
    "workload_capacity_gpu_s",
    "rsu_failed",
)

VIRTUAL_QUEUE_COLUMNS = ("time_s", "queue_family", "owner_id", "value")

_PROTECTED_LOG_KEYS = frozenset(
    {
        "raw_handle",
        "aligned_handle",
        "true_identity",
        "true_expression_label",
        "true_quality_region",
        "realized_attack_outcomes",
        "realized_fer_true_label",
        "realized_fer_class_probabilities",
        "evaluation_subject_cluster_id",
        "payload",
        "payload_b64",
        "packet",
        "_payload",
    }
)


def _safe_value(value: Any) -> Any:
    """Convert an audit value to strict JSON without protected payloads."""

    if isinstance(value, (RawImageHandle, AlignedTensorHandle)):
        raise TypeError("protected vehicle-domain handles cannot enter metric outputs")
    if isinstance(value, bytes):
        raise TypeError(
            "binary payloads cannot enter metric outputs; record size/hash metadata instead"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metric values must be finite")
        return value
    if isinstance(value, Enum):
        return _safe_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, EncodedAnon):
        return {
            "artifact_key": value.artifact_key,
            "pipeline_id": value.pipeline_id,
            "pipeline_hash": value.pipeline_hash,
            "guard_hash": value.guard_hash,
            "encoder_hash": value.encoder_hash,
            "profile_hash": value.profile_hash,
            "quality_bins": list(value.quality_bins),
            "size_bytes": value.size_bytes,
        }
    if isinstance(value, AnonFERRequest):
        return {
            "message_type": "AnonFERRequest",
            "task_id": value.task_id,
            "vehicle_id": value.vehicle_id,
            "artifact_key": value.artifact_key,
            "pipeline_id": value.pipeline_id,
            "payload_size_bytes": value.payload_size_bytes,
            "protocol_version": value.protocol_version,
            "requested_edge_model": value.requested_edge_model,
        }
    if isinstance(value, FERResult):
        return {
            "message_type": "FERResult",
            "task_id": value.task_id,
            "model_id": value.model_id,
            "model_hash": value.model_hash,
            "protocol_version": value.protocol_version,
            "result_code": value.result_code,
            "valid": value.valid,
            "size_bits": value.size_bits,
        }
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key).lower() not in _PROTECTED_LOG_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_safe_value(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if is_dataclass(value):
        return {
            item.name: _safe_value(getattr(value, item.name))
            for item in fields(value)
            if item.name.lower() not in _PROTECTED_LOG_KEYS
        }
    raise TypeError(f"unsupported metric value type: {type(value).__name__}")


def _json_text(value: Any) -> str:
    return canonical_json_bytes(_safe_value(value)).decode("utf-8")


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _fer_classification_metrics(tasks: Sequence[Any]) -> dict[str, Any]:
    """Aggregate only realized, selected-path FER predictions for DONE tasks.

    Labels and probability vectors remain simulator-only task state: this
    function emits aggregate statistics and never task-level ground truth.
    """

    terminal_done = [task for task in tasks if task.state is TaskState.DONE]
    records: list[tuple[str, dict[str, float], str | None, str]] = []
    for task in terminal_done:
        if (
            task.realized_fer_true_label is None
            or not task.realized_fer_class_probabilities
        ):
            continue
        route = "local" if task.selected_local_model is not None else "edge"
        records.append(
            (
                task.realized_fer_true_label,
                dict(task.realized_fer_class_probabilities),
                task.evaluation_subject_cluster_id,
                route,
            )
        )

    def calculate(
        values: Sequence[tuple[str, dict[str, float], str | None, str]],
    ) -> dict[str, Any]:
        if not values:
            return {
                "sample_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "balanced_accuracy": None,
                "per_class_recall": {},
                "negative_log_likelihood": None,
                "expected_calibration_error": None,
            }
        classes = sorted(
            {label for label, _, _, _ in values}
            | {name for _, probabilities, _, _ in values for name in probabilities}
        )
        predictions: list[tuple[str, str, float, float]] = []
        for true_label, probabilities, _, _ in values:
            predicted = max(sorted(probabilities), key=lambda name: probabilities[name])
            predictions.append(
                (
                    true_label,
                    predicted,
                    probabilities[predicted],
                    probabilities[true_label],
                )
            )
        correct = sum(true == predicted for true, predicted, _, _ in predictions)
        recalls: list[float] = []
        per_class_recall: dict[str, float | None] = {}
        f1_values: list[float] = []
        for class_name in classes:
            tp = sum(
                true == class_name and predicted == class_name
                for true, predicted, _, _ in predictions
            )
            fp = sum(
                true != class_name and predicted == class_name
                for true, predicted, _, _ in predictions
            )
            fn = sum(
                true == class_name and predicted != class_name
                for true, predicted, _, _ in predictions
            )
            support = tp + fn
            if support:
                recall = tp / support
                recalls.append(recall)
                per_class_recall[class_name] = recall
            else:
                per_class_recall[class_name] = None
            denominator = 2 * tp + fp + fn
            f1_values.append(2 * tp / denominator if denominator else 0.0)
        bins: list[list[tuple[bool, float]]] = [[] for _ in range(10)]
        for true, predicted, confidence, _ in predictions:
            index = min(9, int(confidence * 10.0))
            bins[index].append((true == predicted, confidence))
        ece = sum(
            len(bin_rows)
            / len(predictions)
            * abs(
                sum(correct_flag for correct_flag, _ in bin_rows) / len(bin_rows)
                - sum(confidence for _, confidence in bin_rows) / len(bin_rows)
            )
            for bin_rows in bins
            if bin_rows
        )
        nll = -sum(
            math.log(max(true_probability, 1e-15))
            for _, _, _, true_probability in predictions
        ) / len(predictions)
        return {
            "sample_count": len(predictions),
            "class_labels": classes,
            "accuracy": correct / len(predictions),
            "macro_f1": sum(f1_values) / len(f1_values),
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            "per_class_recall": per_class_recall,
            "negative_log_likelihood": nll,
            "expected_calibration_error": ece,
            "ece_bins": 10,
        }

    result = calculate(records)
    result["missing_done_prediction_count"] = len(terminal_done) - len(records)
    result["scope"] = (
        "DONE tasks using the FER prediction on the actually selected path"
    )
    result["by_route"] = {
        route: calculate([record for record in records if record[3] == route])
        for route in ("local", "edge")
    }
    clusters = sorted({cluster for _, _, cluster, _ in records if cluster is not None})
    cluster_accuracies = []
    for cluster in clusters:
        rows = [record for record in records if record[2] == cluster]
        metric = calculate(rows)
        if metric["accuracy"] is not None:
            cluster_accuracies.append(float(metric["accuracy"]))
    result["subject_cluster_report"] = {
        "cluster_count": len(clusters),
        "macro_accuracy": (
            sum(cluster_accuracies) / len(cluster_accuracies)
            if cluster_accuracies
            else None
        ),
        "unit": "subject_cluster",
        "task_level_identifiers_emitted": False,
    }
    return result


def _config_core(config: SimulationConfig) -> dict[str, Any]:
    """Configuration semantics excluding input/output locations and formats."""

    return {
        "schema_version": config.schema_version,
        "protocol_version": config.protocol_version,
        "max_snapshot_age_s": config.max_snapshot_age_s,
        "rsu_snapshot_period_s": config.rsu_snapshot_period_s,
        "uplink_pause_limit_s": config.uplink_pause_limit_s,
        "downlink_pause_limit_s": config.downlink_pause_limit_s,
        "metadata_bits": config.metadata_bits,
        "vehicles": _safe_value(config.vehicles),
        "rsus": _safe_value(config.rsus),
        "controller": _safe_value(config.controller),
        "privacy": _safe_value(config.privacy),
        "cost": _safe_value(config.cost),
        "long_term": _safe_value(config.long_term),
        "seeds": _safe_value(config.seeds),
        "parameter_sources": _safe_value(config.parameter_sources),
    }


@dataclass(frozen=True, slots=True)
class OutputFile:
    filename: str
    sha256: str
    size_bytes: int
    row_count: int | None


@dataclass(frozen=True, slots=True)
class ParquetStatus:
    requested: bool
    available: bool
    generated: bool
    status: str
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class MetricsArtifacts:
    files: Mapping[str, OutputFile]
    parquet_status: ParquetStatus
    core_digest: str
    controller_diagnostics: Mapping[str, Any]


@dataclass(slots=True)
class MetricLedger:
    """Mutable event-time ledger; policies and physics remain independent."""

    simulation_start_s: float = 0.0
    event_rows: list[dict[str, Any]] = field(default_factory=list)
    action_rows: list[dict[str, Any]] = field(default_factory=list)
    resource_rows: list[dict[str, Any]] = field(default_factory=list)
    virtual_queue_rows: list[dict[str, Any]] = field(default_factory=list)
    controller_samples: list[dict[str, Any]] = field(default_factory=list)
    invariant_failures: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.simulation_start_s) or self.simulation_start_s < 0:
            raise ValueError("simulation_start_s must be finite and nonnegative")

    def _elapsed_s(self, time_s: float) -> float:
        if not math.isfinite(time_s):
            raise ValueError("metric time must be finite")
        return max(0.0, time_s - self.simulation_start_s)

    def record_event(
        self, record: Mapping[str, Any] | None = None, **values: Any
    ) -> None:
        row = dict(record or {})
        row.update(values)
        row.setdefault("record_kind", "EVENT_AUDIT")
        row.setdefault("record_index", len(self.event_rows))
        self.event_rows.append(_safe_value(row))

    def record_action(
        self, record: Mapping[str, Any] | None = None, **values: Any
    ) -> None:
        row = dict(record or {})
        row.update(values)
        row.setdefault("record_index", len(self.action_rows))
        self.action_rows.append(_safe_value(row))

    def record_controller_time(
        self,
        wall_clock_s: float,
        *,
        task_id: str | None = None,
        stage: str | None = None,
        policy: str | None = None,
    ) -> None:
        if not math.isfinite(wall_clock_s) or wall_clock_s < 0:
            raise ValueError(
                "controller wall-clock duration must be finite and nonnegative"
            )
        self.controller_samples.append(
            {
                "wall_clock_s": wall_clock_s,
                "task_id": task_id,
                "stage": stage,
                "policy": policy,
            }
        )

    def record_invariant_failure(self, record: Mapping[str, Any]) -> None:
        self.invariant_failures.append(_safe_value(record))

    def _pool_row(
        self,
        pool: ResourcePool,
        *,
        time_s: float,
        owner_type: str,
        owner_id: str,
        physical_energy_j: float,
        extras: Mapping[str, Any],
    ) -> dict[str, Any]:
        denominator = self._elapsed_s(time_s) * pool.server_count
        utilization = pool.busy_server_seconds / denominator if denominator > 0 else 0.0
        if utilization > 1.0 and utilization <= 1.0 + 1e-12:
            utilization = 1.0
        return {
            "time_s": time_s,
            "owner_type": owner_type,
            "owner_id": owner_id,
            "resource_id": pool.resource_id,
            "resource_kind": pool.kind.value,
            "server_count": pool.server_count,
            "waiting_jobs": pool.waiting_count,
            "running_jobs": pool.running_count,
            "residual_work_s": pool.residual_work_s,
            "busy_server_seconds": pool.busy_server_seconds,
            "utilization": utilization,
            "max_running_observed": pool.max_running_observed,
            "physical_energy_j": physical_energy_j,
            **dict(extras),
        }

    def snapshot_resources(
        self, state: SimulationState, *, time_s: float | None = None
    ) -> None:
        now = state.clock_s if time_s is None else float(time_s)
        if not math.isfinite(now) or now < 0:
            raise ValueError("resource snapshot time must be finite and nonnegative")
        for vehicle_id, vehicle in sorted(state.vehicles.items()):
            extras = {
                "memory_reserved_bytes": vehicle.memory_reserved_bytes,
                "memory_capacity_bytes": vehicle.memory_capacity_bytes,
                "descriptor_reserved_json": _json_text(vehicle.descriptors_reserved),
                "descriptor_capacity_json": _json_text(vehicle.descriptor_capacity),
                "admitted_descriptors": None,
                "descriptor_capacity": None,
                "reserved_vram_bytes": None,
                "vram_capacity_bytes": None,
                "reserved_work_gpu_s": None,
                "workload_capacity_gpu_s": None,
                "rsu_failed": None,
                "system_maintenance_energy_j": None,
            }
            for _, pool in sorted(vehicle.resources.items()):
                self.resource_rows.append(
                    self._pool_row(
                        pool,
                        time_s=now,
                        owner_type="vehicle",
                        owner_id=vehicle_id,
                        physical_energy_j=vehicle.physical_energy_j,
                        extras=extras,
                    )
                )
        for rsu_id, rsu in sorted(state.rsus.items()):
            admission = rsu.admission
            extras = {
                "memory_reserved_bytes": None,
                "memory_capacity_bytes": None,
                "descriptor_reserved_json": None,
                "descriptor_capacity_json": None,
                "admitted_descriptors": admission.descriptors,
                "descriptor_capacity": admission.descriptor_capacity,
                "reserved_vram_bytes": admission.vram_bytes,
                "vram_capacity_bytes": admission.vram_capacity_bytes,
                "reserved_work_gpu_s": admission.reserved_work_gpu_s,
                "workload_capacity_gpu_s": admission.workload_capacity_gpu_s,
                "rsu_failed": rsu.failed,
                "system_maintenance_energy_j": rsu.system_maintenance_energy_j,
            }
            for pool in (rsu.ingress, rsu.gpu):
                self.resource_rows.append(
                    self._pool_row(
                        pool,
                        time_s=now,
                        owner_type="rsu",
                        owner_id=rsu_id,
                        physical_energy_j=rsu.physical_energy_j,
                        extras=extras,
                    )
                )

    def snapshot_virtual_queues(
        self, state: SimulationState, *, time_s: float | None = None
    ) -> None:
        now = state.clock_s if time_s is None else float(time_s)
        bank = state.virtual_queues
        for owner_id, value in sorted(bank.vehicle_power.items()):
            self.virtual_queue_rows.append(
                {
                    "time_s": now,
                    "queue_family": "vehicle_power",
                    "owner_id": owner_id,
                    "value": value,
                }
            )
        for owner_id, value in sorted(bank.rsu_power.items()):
            self.virtual_queue_rows.append(
                {
                    "time_s": now,
                    "queue_family": "rsu_power",
                    "owner_id": owner_id,
                    "value": value,
                }
            )
        for family in ("timeout", "failure", "coverage"):
            self.virtual_queue_rows.append(
                {
                    "time_s": now,
                    "queue_family": family,
                    "owner_id": "",
                    "value": getattr(bank, family),
                }
            )

    def snapshot(self, state: SimulationState, _batch: Any | None = None) -> None:
        """Capture resource and virtual-queue state after one compound event."""

        self.snapshot_resources(state)
        self.snapshot_virtual_queues(state)

    @staticmethod
    def task_rows(
        state: SimulationState, config: SimulationConfig
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task in sorted(state.tasks.values(), key=lambda item: item.task_id):
            done = task.state is TaskState.DONE and task.result_valid
            failed = task.state is TaskState.FAIL
            terminal_time = task.terminal_time_s
            latency = (
                None
                if terminal_time is None
                else max(0.0, terminal_time - task.arrival_time_s)
            )
            cost_latency = latency if done else task.relative_deadline_s
            if done and task.realized_fer_loss is not None:
                all_task_loss: float | None = task.realized_fer_loss
            elif failed:
                all_task_loss = config.cost.failure_loss
            else:
                all_task_loss = None
            vehicle_total = task.vehicle_energy_j + task.hold_vehicle_energy_j
            rsu_total = task.rsu_energy_j + task.hold_rsu_energy_j
            row = {
                "task_id": task.task_id,
                "vehicle_id": task.vehicle_id,
                "state": task.state.value,
                "arrival_time_s": task.arrival_time_s,
                "relative_deadline_s": task.relative_deadline_s,
                "absolute_deadline_s": task.absolute_deadline_s,
                "terminal_time_s": terminal_time,
                "terminal_latency_s": latency,
                "cost_latency_s": cost_latency,
                "done": done,
                "failed": failed,
                "timeout": task.failure_reason is FailureReason.TIMEOUT,
                "result_valid": task.result_valid,
                "failure_reason": task.failure_reason.value,
                "realized_fer_loss": task.realized_fer_loss,
                "all_task_loss": all_task_loss,
                "failure_penalty_cost": config.cost.failure_loss if failed else 0.0,
                "selected_pipeline": task.selected_pipeline,
                "selected_local_model": task.selected_local_model,
                "selected_rsu": task.selected_rsu,
                "selected_edge_model": task.selected_edge_model,
                "attempt_started_count": task.attempt_started_count,
                "max_attempts": task.max_attempts,
                "trace_row_id": task.trace_row_id,
                "artifact_key": task.artifact_key,
                "encoded_size_bytes": task.encoded_size_bytes,
                "ul_remaining_bits": task.ul_remaining_bits,
                "dl_remaining_bits": task.dl_remaining_bits,
                "vehicle_compute_radio_energy_j": task.vehicle_energy_j,
                "vehicle_hold_energy_j": task.hold_vehicle_energy_j,
                "vehicle_attributed_energy_j": vehicle_total,
                "rsu_compute_radio_energy_j": task.rsu_energy_j,
                "rsu_hold_energy_j": task.hold_rsu_energy_j,
                "rsu_attributed_energy_j": rsu_total,
                "actual_path_json": _json_text(task.actual_path),
                "phase_history_json": _json_text(task.phase_history),
                "enqueue_times_json": _json_text(task.enqueue_times),
                "start_times_json": _json_text(task.start_times),
                "end_times_json": _json_text(task.end_times),
                "compute_audit_json": _json_text(task.compute_audit),
                "anon_attempts_json": _json_text(task.anon_attempt_audit),
                "network_audit_json": _json_text(task.network_audit),
                "rsu_audit_json": _json_text(task.rsu_audit),
            }
            rows.append(_safe_value(row))
        return rows

    def _event_output_rows(self, state: SimulationState) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event_batch in state.event_log:
            row = dict(event_batch)
            row.setdefault("record_kind", "EVENT_BATCH")
            rows.append(row)
        rows.extend(self.event_rows)
        for task in sorted(state.tasks.values(), key=lambda item: item.task_id):
            for audit_type, audits in (
                ("COMPUTE", task.compute_audit),
                ("ANON_ATTEMPT", task.anon_attempt_audit),
                ("NETWORK", task.network_audit),
                ("RSU", task.rsu_audit),
            ):
                for index, audit in enumerate(audits):
                    rows.append(
                        {
                            "audit_type": audit_type,
                            "record_kind": audit_type,
                            "task_id": task.task_id,
                            "vehicle_id": task.vehicle_id,
                            "audit_index": index,
                            **dict(audit),
                        }
                    )
        # Do not deduplicate: two physically distinct events may intentionally
        # have identical public fields.  Stable event sequence numbers/audit
        # indices are the producer's responsibility.
        return [_safe_value(row) for row in rows]

    def _action_output_rows(self, state: SimulationState) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = list(self.action_rows)
        for task in sorted(state.tasks.values(), key=lambda item: item.task_id):
            for audit_type, audits in (
                ("HARD_MASK", task.mask_audit),
                ("ACTION", task.action_audit),
            ):
                for index, audit in enumerate(audits):
                    if audit_type == "ACTION":
                        if audit.get("repair") == "FROZEN_LOCAL_FALLBACK":
                            record_kind = "EXECUTION_REPAIR"
                        elif "policy" in audit:
                            record_kind = "POLICY_DECISION"
                        elif "executed_stage" in audit:
                            record_kind = "EXECUTION_REPAIR"
                        else:
                            record_kind = "CONTROLLER_PROPOSAL"
                    else:
                        record_kind = "HARD_MASK"
                    rows.append(
                        {
                            "audit_type": audit_type,
                            "record_kind": record_kind,
                            "task_id": task.task_id,
                            "vehicle_id": task.vehicle_id,
                            "audit_index": index,
                            **dict(audit),
                        }
                    )
        for index, sample in enumerate(self.controller_samples):
            rows.append(
                {
                    "audit_type": "CONTROLLER_DIAGNOSTIC",
                    "record_kind": "CONTROLLER_DIAGNOSTIC",
                    "audit_index": index,
                    **sample,
                }
            )
        return [_safe_value(row) for row in rows]

    def _resource_output_rows(self, state: SimulationState) -> list[dict[str, Any]]:
        if not self.resource_rows:
            self.snapshot_resources(state)
        return [_safe_value(row) for row in self.resource_rows]

    def _virtual_output_rows(self, state: SimulationState) -> list[dict[str, Any]]:
        if self.virtual_queue_rows:
            return [_safe_value(row) for row in self.virtual_queue_rows]
        rows: list[dict[str, Any]] = []
        for snapshot in state.virtual_queues.trajectory:
            time_s = snapshot["time_s"]
            for owner_id, value in sorted(snapshot.get("vehicle_power", {}).items()):
                rows.append(
                    {
                        "time_s": time_s,
                        "queue_family": "vehicle_power",
                        "owner_id": owner_id,
                        "value": value,
                    }
                )
            for owner_id, value in sorted(snapshot.get("rsu_power", {}).items()):
                rows.append(
                    {
                        "time_s": time_s,
                        "queue_family": "rsu_power",
                        "owner_id": owner_id,
                        "value": value,
                    }
                )
            for family in ("timeout", "failure", "coverage"):
                rows.append(
                    {
                        "time_s": time_s,
                        "queue_family": family,
                        "owner_id": "",
                        "value": snapshot[family],
                    }
                )
        if not rows:
            self.snapshot_virtual_queues(state)
            rows = list(self.virtual_queue_rows)
        return [_safe_value(row) for row in rows]

    def controller_diagnostics(
        self, state: SimulationState | None = None
    ) -> dict[str, Any]:
        if self.controller_samples:
            durations = [
                float(item["wall_clock_s"]) for item in self.controller_samples
            ]
        elif state is not None:
            # The simulator stores the diagnostic beside its proposed action;
            # use it when the caller did not separately feed the ledger.
            durations = [
                float(audit["controller_wall_clock_s_diagnostic"])
                for task in state.tasks.values()
                for audit in task.action_audit
                if audit.get("controller_wall_clock_s_diagnostic") is not None
            ]
        else:
            durations = []
        return {
            "sample_count": len(durations),
            "wall_clock_total_s": sum(durations),
            "wall_clock_mean_s": sum(durations) / len(durations) if durations else None,
            "wall_clock_p50_s": _percentile(durations, 0.50),
            "wall_clock_p95_s": _percentile(durations, 0.95),
            "wall_clock_max_s": max(durations) if durations else None,
            "note": "engineering diagnostic only; excluded from simulated time and core_digest",
        }

    def _pool_summaries(
        self,
        state: SimulationState,
        resource_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        row_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        for row in resource_rows:
            key = (
                str(row["owner_type"]),
                str(row["owner_id"]),
                str(row["resource_id"]),
            )
            row_groups.setdefault(key, []).append(row)
        pools: list[tuple[str, str, ResourcePool]] = []
        for vehicle_id, vehicle in sorted(state.vehicles.items()):
            pools.extend(
                ("vehicle", vehicle_id, pool)
                for _, pool in sorted(vehicle.resources.items())
            )
        for rsu_id, rsu in sorted(state.rsus.items()):
            pools.extend(("rsu", rsu_id, pool) for pool in (rsu.ingress, rsu.gpu))
        elapsed = self._elapsed_s(state.clock_s)
        result: dict[str, Any] = {}
        for owner_type, owner_id, pool in pools:
            key = (owner_type, owner_id, pool.resource_id)
            history = row_groups.get(key, [])
            waits = [
                job.start_time_s - job.enqueue_time_s
                for job in pool.jobs.values()
                if job.start_time_s is not None
            ]
            services = [
                job.end_time_s - job.start_time_s
                for job in pool.jobs.values()
                if job.start_time_s is not None and job.end_time_s is not None
            ]
            denominator = elapsed * pool.server_count
            utilization = (
                pool.busy_server_seconds / denominator if denominator > 0 else 0.0
            )
            result["/".join(key)] = {
                "owner_type": owner_type,
                "owner_id": owner_id,
                "resource_id": pool.resource_id,
                "resource_kind": pool.kind.value,
                "server_count": pool.server_count,
                "job_count": len(pool.jobs),
                "completed_job_count": sum(
                    job.status is JobStatus.DONE for job in pool.jobs.values()
                ),
                "cancelled_job_count": sum(
                    job.status is JobStatus.CANCELLED for job in pool.jobs.values()
                ),
                "busy_server_seconds": pool.busy_server_seconds,
                "utilization": utilization,
                "max_running_observed": pool.max_running_observed,
                "max_waiting_jobs": max(
                    (int(row["waiting_jobs"]) for row in history),
                    default=pool.waiting_count,
                ),
                "max_residual_work_s": max(
                    (float(row["residual_work_s"]) for row in history),
                    default=pool.residual_work_s,
                ),
                "wait_s": _distribution(waits),
                "service_s": _distribution(services),
            }
        return result

    def summarize(
        self,
        state: SimulationState,
        config: SimulationConfig,
        *,
        task_rows: Sequence[Mapping[str, Any]] | None = None,
        resource_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tasks = (
            list(task_rows) if task_rows is not None else self.task_rows(state, config)
        )
        resources = (
            list(resource_rows)
            if resource_rows is not None
            else self._resource_output_rows(state)
        )
        done = [row for row in tasks if row["done"]]
        failed = [row for row in tasks if row["failed"]]
        timeouts = [row for row in tasks if row["timeout"]]
        nonterminal = [row for row in tasks if not row["done"] and not row["failed"]]
        terminal_latencies = [
            float(row["terminal_latency_s"])
            for row in tasks
            if row["terminal_latency_s"] is not None
        ]
        successful_latencies = [
            float(row["terminal_latency_s"])
            for row in done
            if row["terminal_latency_s"] is not None
        ]
        cost_latencies = [
            float(row["cost_latency_s"])
            for row in tasks
            if row["cost_latency_s"] is not None
        ]
        successful_losses = [
            float(row["realized_fer_loss"])
            for row in done
            if row["realized_fer_loss"] is not None
        ]
        all_losses = [
            float(row["all_task_loss"])
            for row in tasks
            if row["all_task_loss"] is not None
        ]
        reasons: dict[str, int] = {}
        for row in failed:
            reason = str(row["failure_reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
        task_vehicle_energy = sum(
            float(row["vehicle_attributed_energy_j"]) for row in tasks
        )
        task_rsu_energy = sum(float(row["rsu_attributed_energy_j"]) for row in tasks)
        physical_vehicle = {
            key: value.physical_energy_j
            for key, value in sorted(state.vehicles.items())
        }
        physical_rsu = {
            key: value.physical_energy_j for key, value in sorted(state.rsus.items())
        }
        physical_vehicle_total = sum(physical_vehicle.values())
        physical_rsu_total = sum(physical_rsu.values())
        pool_summaries = self._pool_summaries(state, resources)
        pool_utilizations = [
            float(item["utilization"]) for item in pool_summaries.values()
        ]
        count = len(tasks)
        coverage = len(done) / count if count else None
        success_fer = (
            sum(successful_losses) / len(successful_losses)
            if successful_losses
            else None
        )
        all_task_loss = (
            sum(all_losses) / len(all_losses)
            if len(all_losses) == count and count
            else None
        )
        terminal_dist = _distribution(terminal_latencies)
        summary = {
            "task_count": count,
            "done_count": len(done),
            "fail_count": len(failed),
            "timeout_count": len(timeouts),
            "nonterminal_count": len(nonterminal),
            "coverage": coverage,
            "failure_rate": len(failed) / count if count else None,
            "timeout_rate": len(timeouts) / count if count else None,
            "success_conditional_fer_loss": success_fer,
            "success_conditional_fer_available_count": len(successful_losses),
            "success_conditional_fer_missing_count": len(done) - len(successful_losses),
            "selected_path_fer_classification": _fer_classification_metrics(
                list(state.tasks.values())
            ),
            "all_task_loss": all_task_loss,
            "all_task_loss_available_count": len(all_losses),
            "all_task_loss_complete": len(all_losses) == count,
            "latency_p50_s": terminal_dist["p50"],
            "latency_p95_s": terminal_dist["p95"],
            "latency_p99_s": terminal_dist["p99"],
            "terminal_latency_s": terminal_dist,
            "successful_latency_s": _distribution(successful_latencies),
            "cost_latency_s": _distribution(cost_latencies),
            "failure_reasons": dict(sorted(reasons.items())),
            "energy_j": {
                "task_attributed": {
                    "vehicle": task_vehicle_energy,
                    "rsu": task_rsu_energy,
                    "total": task_vehicle_energy + task_rsu_energy,
                },
                "physical_system": {
                    "vehicle_by_id": physical_vehicle,
                    "rsu_by_id": physical_rsu,
                    "vehicle": physical_vehicle_total,
                    "rsu": physical_rsu_total,
                    "total": physical_vehicle_total + physical_rsu_total,
                },
            },
            "resources": {
                "pools": pool_summaries,
                "pool_count": len(pool_summaries),
                "mean_utilization": sum(pool_utilizations) / len(pool_utilizations)
                if pool_utilizations
                else None,
                "max_utilization": max(pool_utilizations)
                if pool_utilizations
                else None,
                "max_waiting_jobs": max(
                    (item["max_waiting_jobs"] for item in pool_summaries.values()),
                    default=0,
                ),
                "max_residual_work_s": max(
                    (item["max_residual_work_s"] for item in pool_summaries.values()),
                    default=0.0,
                ),
            },
            "virtual_queues_final": {
                "vehicle_power": dict(
                    sorted(state.virtual_queues.vehicle_power.items())
                ),
                "rsu_power": dict(sorted(state.virtual_queues.rsu_power.items())),
                "timeout": state.virtual_queues.timeout,
                "failure": state.virtual_queues.failure,
                "coverage": state.virtual_queues.coverage,
            },
            "controller_diagnostics": self.controller_diagnostics(state),
            "invariant_check_count": state.invariant_checks,
            "invariant_failure_count": len(self.invariant_failures),
            "metric_semantics": {
                "terminal_latency_s": "terminal_time-arrival_time for every terminal task",
                "cost_latency_s": "DONE latency, otherwise the configured relative deadline",
                "all_task_loss": "DONE FER loss; FAIL uses frozen config.cost.failure_loss",
                "energy": "task attribution is distinct from independent physical system energy",
            },
        }
        return _safe_value(summary)

    def finalize(
        self, state: SimulationState, config: SimulationConfig
    ) -> dict[str, Any]:
        """Return the final in-memory summary without writing files."""

        return self.summarize(state, config)

    @staticmethod
    def _write_csv(
        path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]
    ) -> None:
        extras = sorted({str(key) for row in rows for key in row} - set(preferred))
        fieldnames = list(preferred) + extras
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row).decode("utf-8"))
                handle.write("\n")

    @staticmethod
    def _file_record(path: Path, rows: int | None) -> OutputFile:
        return OutputFile(path.name, sha256_file(path), path.stat().st_size, rows)

    def write_outputs(
        self,
        output_dir: str | Path,
        state: SimulationState,
        config: SimulationConfig,
        *,
        overwrite: bool = False,
    ) -> MetricsArtifacts:
        """Write the six required machine-readable files and optional Parquet."""

        output = prepare_output_directory(output_dir, overwrite=overwrite)
        tasks = self.task_rows(state, config)
        events = self._event_output_rows(state)
        actions = self._action_output_rows(state)
        resources = self._resource_output_rows(state)
        virtual = self._virtual_output_rows(state)
        summary = self.summarize(
            state, config, task_rows=tasks, resource_rows=resources
        )

        paths = {
            "tasks.csv": output / "tasks.csv",
            "events.jsonl": output / "events.jsonl",
            "actions.jsonl": output / "actions.jsonl",
            "resources.csv": output / "resources.csv",
            "virtual_queues.csv": output / "virtual_queues.csv",
            "summary.json": output / "summary.json",
        }
        self._write_csv(paths["tasks.csv"], tasks, TASK_COLUMNS)
        self._write_jsonl(paths["events.jsonl"], events)
        self._write_jsonl(paths["actions.jsonl"], actions)
        self._write_csv(paths["resources.csv"], resources, RESOURCE_COLUMNS)
        self._write_csv(paths["virtual_queues.csv"], virtual, VIRTUAL_QUEUE_COLUMNS)
        paths["summary.json"].write_text(
            json.dumps(
                summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        core_payload = {
            "configuration": _config_core(config),
            "tasks": tasks,
            "events": events,
            "actions": actions,
            "resources": resources,
            "virtual_queues": virtual,
            "summary": summary,
        }
        core_digest = canonical_core_digest(core_payload)
        row_counts = {
            "tasks.csv": len(tasks),
            "events.jsonl": len(events),
            "actions.jsonl": len(actions),
            "resources.csv": len(resources),
            "virtual_queues.csv": len(virtual),
            "summary.json": None,
        }
        file_records: dict[str, OutputFile] = {
            name: self._file_record(path, row_counts[name])
            for name, path in paths.items()
        }

        requested = config.output_parquet
        if not requested:
            parquet = ParquetStatus(
                requested=False,
                available=importlib.util.find_spec("pyarrow") is not None,
                generated=False,
                status="not_requested",
            )
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError:
                parquet = ParquetStatus(
                    requested=True,
                    available=False,
                    generated=False,
                    status="optional_dependency_unavailable:pyarrow",
                )
            else:
                parquet_path = output / "tasks.parquet"
                table = pa.Table.from_pylist(tasks)
                pq.write_table(table, parquet_path)
                file_records[parquet_path.name] = self._file_record(
                    parquet_path, len(tasks)
                )
                parquet = ParquetStatus(
                    requested=True,
                    available=True,
                    generated=True,
                    status="generated",
                    filename=parquet_path.name,
                )

        return MetricsArtifacts(
            files=dict(sorted(file_records.items())),
            parquet_status=parquet,
            core_digest=core_digest,
            controller_diagnostics=self.controller_diagnostics(state),
        )


__all__ = [
    "MetricLedger",
    "MetricsArtifacts",
    "OutputFile",
    "ParquetStatus",
    "RESOURCE_COLUMNS",
    "TASK_COLUMNS",
    "VIRTUAL_QUEUE_COLUMNS",
]
