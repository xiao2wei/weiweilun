"""Strict offline audits and a finite exact oracle for paper experiments.

Every function is pure and counterfactual.  In particular, hard-mask audit
rows are never converted into executable actions.  Missing cost-accounting
fields cause a structured refusal instead of an inferred or zero-filled cost.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .profiles import canonical_document_sha256, canonical_json_bytes


class AuditValidationError(ValueError):
    """Machine-readable refusal for an incomplete or invalid audit input."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "REFUSED",
            "error_code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditValidationError("AUDIT_TEXT", f"{field} must be non-empty text")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    strict_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditValidationError("AUDIT_NUMBER", f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AuditValidationError("AUDIT_NUMBER", f"{field} must be finite")
    if strict_positive and number <= 0.0:
        raise AuditValidationError("AUDIT_NUMBER", f"{field} must be positive")
    if minimum is not None and number < minimum:
        raise AuditValidationError(
            "AUDIT_NUMBER", f"{field} is below its minimum", minimum=minimum
        )
    return number


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditValidationError("AUDIT_SEQUENCE", f"{field} must be an array")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditValidationError("AUDIT_OBJECT", f"{field} must be an object")
    return value


def _hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finish(report: dict[str, Any], inputs: Any) -> dict[str, Any]:
    report["input_sha256"] = _hash(inputs)
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    return report


def _mask_records(records: Sequence[Mapping[str, Any]], stage: str | None = None):
    masks = []
    for index, raw in enumerate(_sequence(records, "action_records")):
        record = _mapping(raw, f"action_records[{index}]")
        if record.get("record_kind") != "HARD_MASK":
            continue
        record_stage = _text(record.get("stage"), f"action_records[{index}].stage")
        if stage is not None and record_stage != stage:
            continue
        task_id = _text(record.get("task_id"), f"action_records[{index}].task_id")
        rows = _sequence(record.get("rows"), f"action_records[{index}].rows")
        if not rows:
            raise AuditValidationError(
                "AUDIT_EMPTY_MASK",
                "hard-mask record has no action rows",
                task_id=task_id,
            )
        masks.append((index, task_id, record_stage, record, rows))
    if not masks:
        raise AuditValidationError("AUDIT_NO_MASK", "no matching HARD_MASK records")
    return masks


def _expected_cost(row: Mapping[str, Any], field: str) -> float | None:
    details = _mapping(row.get("details", {}), f"{field}.details")
    bounds = _mapping(details.get("bounds", {}), f"{field}.details.bounds")
    value = bounds.get("expected_cost")
    if value is None:
        return None
    return _number(value, f"{field}.details.bounds.expected_cost", minimum=0.0)


def audit_hard_mask_counterfactual(
    action_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count rejected actions that look cheaper without ever executing them."""

    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    rejected_count = 0
    seemingly_better_count = 0
    unavailable_count = 0
    for index, task_id, stage, _, raw_rows in _mask_records(action_records):
        rows = [
            _mapping(row, f"action_records[{index}].rows[{row_index}]")
            for row_index, row in enumerate(raw_rows)
        ]
        action_ids = [
            _text(row.get("action_id"), f"action_records[{index}].action_id")
            for row in rows
        ]
        if len(action_ids) != len(set(action_ids)):
            raise AuditValidationError(
                "AUDIT_DUPLICATE_ACTION", "hard-mask action IDs are not unique"
            )
        allowed = []
        for row_index, row in enumerate(rows):
            if row.get("allowed") is True:
                cost = _expected_cost(row, f"action_records[{index}].rows[{row_index}]")
                if cost is not None:
                    allowed.append((cost, action_ids[row_index]))
        allowed_best = min(allowed, key=lambda item: item) if allowed else None
        allowed_best_cost = None if allowed_best is None else allowed_best[0]
        allowed_best_id = None if allowed_best is None else allowed_best[1]
        better: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            if row.get("allowed") is not False:
                continue
            rejected_count += 1
            reason_codes = _sequence(
                row.get("reason_codes"),
                f"action_records[{index}].rows[{row_index}].reason_codes",
            )
            if not reason_codes:
                raise AuditValidationError(
                    "AUDIT_REJECTION_REASON_MISSING",
                    "rejected action has no hard-mask reason",
                    action_id=action_ids[row_index],
                )
            normalized_reasons = sorted(
                _text(value, "reason_code") for value in reason_codes
            )
            reasons.update(normalized_reasons)
            cost = _expected_cost(row, f"action_records[{index}].rows[{row_index}]")
            if cost is None:
                unavailable_count += 1
                continue
            if allowed_best_cost is None:
                unavailable_count += 1
                continue
            if cost < allowed_best_cost:
                seemingly_better_count += 1
                better.append(
                    {
                        "action_id": action_ids[row_index],
                        "expected_cost": cost,
                        "cost_advantage": allowed_best_cost - cost,
                        "reason_codes": normalized_reasons,
                    }
                )
        decisions.append(
            {
                "task_id": task_id,
                "stage": stage,
                "allowed_reference_action_id": allowed_best_id,
                "allowed_reference_expected_cost": allowed_best_cost,
                "comparison_status": (
                    "NO_COSTED_ALLOWED_ACTION"
                    if allowed_best_cost is None
                    else "COMPARED"
                ),
                "rejected_seemingly_better": sorted(
                    better, key=lambda row: (row["expected_cost"], row["action_id"])
                ),
            }
        )
    return _finish(
        {
            "schema_version": "1.0",
            "analysis": "hard_mask_safety_counterfactual",
            "counterfactual_only": True,
            "unsafe_actions_executed_by_audit": 0,
            "hard_mask_bypassed": False,
            "units": {"expected_cost": "normalized_cost"},
            "mask_count": len(decisions),
            "rejected_action_count": rejected_count,
            "rejected_with_cost_unavailable": unavailable_count,
            "rejected_seemingly_better_count": seemingly_better_count,
            "reason_counts": dict(sorted(reasons.items())),
            "decisions": decisions,
        },
        action_records,
    )


def _ablation_action(row: Mapping[str, Any], field: str) -> dict[str, Any] | None:
    action = _mapping(row.get("action"), f"{field}.action")
    if row.get("allowed") is not True or action.get("kind") == "FAIL":
        return None
    action_id = _text(row.get("action_id"), f"{field}.action_id")
    cost = _expected_cost(row, field)
    if cost is None:
        raise AuditValidationError(
            "ABLATION_COST_MISSING",
            "allowed action has no expected cost",
            action_id=action_id,
        )
    if action.get("kind") != "EDGE":
        return {
            "action_id": action_id,
            "expected_cost": cost,
            "without_output_size_cost": cost,
            "without_fresh_queue_cost": cost,
        }
    details = _mapping(row.get("details"), f"{field}.details")
    ablation = _mapping(
        details.get("information_ablation"), f"{field}.details.information_ablation"
    )
    output = _number(
        ablation.get("observed_output_size_cost"),
        f"{field}.information_ablation.observed_output_size_cost",
        minimum=0.0,
    )
    output_conservative = _number(
        ablation.get("conservative_output_size_cost"),
        f"{field}.information_ablation.conservative_output_size_cost",
        minimum=output,
    )
    queue = _number(
        ablation.get("observed_fresh_queue_cost"),
        f"{field}.information_ablation.observed_fresh_queue_cost",
        minimum=0.0,
    )
    queue_conservative = _number(
        ablation.get("conservative_stale_queue_cost"),
        f"{field}.information_ablation.conservative_stale_queue_cost",
        minimum=queue,
    )
    return {
        "action_id": action_id,
        "expected_cost": cost,
        "without_output_size_cost": cost - output + output_conservative,
        "without_fresh_queue_cost": cost - queue + queue_conservative,
    }


def build_preregistered_one_shot_commitments(
    action_records: Sequence[Mapping[str, Any]],
    *,
    registered_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a frozen READY priority plan using RAW-visible task identity only.

    READY masks, costs, queues, encoded sizes and outcomes are deliberately not
    read.  At evaluation time the hard mask may deterministically advance to a
    later preregistered fallback, but it may never rank READY actions by cost.
    """

    plan = _mapping(registered_plan, "registered_plan")
    default_priority = tuple(
        _text(value, "registered_plan.default_action_priority")
        for value in _sequence(
            plan.get("default_action_priority"),
            "registered_plan.default_action_priority",
        )
    )
    if not default_priority or len(default_priority) != len(set(default_priority)):
        raise AuditValidationError(
            "COMMITMENT_PLAN_PRIORITY",
            "default action priority must be non-empty and unique",
        )
    if any(not value.startswith("READY|") for value in default_priority):
        raise AuditValidationError(
            "COMMITMENT_PLAN_STAGE", "one-shot actions must target READY"
        )
    by_vehicle_raw = _mapping(plan.get("by_vehicle", {}), "registered_plan.by_vehicle")
    by_vehicle: dict[str, tuple[str, ...]] = {}
    for vehicle, raw_priority in by_vehicle_raw.items():
        vehicle_id = _text(vehicle, "registered_plan.by_vehicle key")
        priority = tuple(
            _text(value, f"registered_plan.by_vehicle.{vehicle_id}")
            for value in _sequence(
                raw_priority, f"registered_plan.by_vehicle.{vehicle_id}"
            )
        )
        if (
            not priority
            or len(priority) != len(set(priority))
            or any(not value.startswith("READY|") for value in priority)
        ):
            raise AuditValidationError(
                "COMMITMENT_PLAN_PRIORITY",
                "vehicle action priority must be unique, non-empty and READY-only",
                vehicle_id=vehicle_id,
            )
        by_vehicle[vehicle_id] = priority

    raw_projection: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for index, task_id, stage, record, _ in _mask_records(action_records, "RAW"):
        if stage != "RAW":
            continue
        if task_id in seen_tasks:
            raise AuditValidationError(
                "COMMITMENT_DUPLICATE_TASK",
                "RAW audit contains duplicate task decisions",
                task_id=task_id,
            )
        seen_tasks.add(task_id)
        vehicle_id = _text(
            record.get("vehicle_id"), f"action_records[{index}].vehicle_id"
        )
        raw_projection.append({"task_id": task_id, "vehicle_id": vehicle_id})
    commitments = {
        row["task_id"]: {
            "action_priority": list(
                by_vehicle.get(row["vehicle_id"], default_priority)
            ),
            "source_vehicle_id": row["vehicle_id"],
        }
        for row in sorted(raw_projection, key=lambda value: value["task_id"])
    }
    report = {
        "schema_version": "1.0",
        "analysis": "preregistered_one_shot_commitments",
        "policy": "static_ready_action_priority_with_hard_mask_fallback",
        "source_stage": "RAW",
        "uses_ready_records": False,
        "uses_ready_expected_cost": False,
        "uses_output_size": False,
        "uses_queue_freshness": False,
        "hard_mask_fallback_only": True,
        "registered_plan": {
            "default_action_priority": list(default_priority),
            "by_vehicle": {
                key: list(value) for key, value in sorted(by_vehicle.items())
            },
        },
        "registered_plan_sha256": _hash(
            {
                "default_action_priority": list(default_priority),
                "by_vehicle": {
                    key: list(value) for key, value in sorted(by_vehicle.items())
                },
            }
        ),
        "raw_visible_input_sha256": _hash(raw_projection),
        "commitments": commitments,
    }
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    return report


def _commitment_priority(value: Any, task_id: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_text(value, f"one_shot_commitments.{task_id}"),)
    commitment = _mapping(value, f"one_shot_commitments.{task_id}")
    priority = tuple(
        _text(item, f"one_shot_commitments.{task_id}.action_priority")
        for item in _sequence(
            commitment.get("action_priority"),
            f"one_shot_commitments.{task_id}.action_priority",
        )
    )
    if not priority or len(priority) != len(set(priority)):
        raise AuditValidationError(
            "ABLATION_COMMITMENT_PRIORITY",
            "one-shot action priority must be non-empty and unique",
            task_id=task_id,
        )
    return priority


def evaluate_two_stage_information_ablation(
    action_records: Sequence[Mapping[str, Any]],
    *,
    one_shot_commitments: Mapping[str, str],
) -> dict[str, Any]:
    """Pair READY recourse with safe one-shot and conservative information ablations."""

    commitment_document = _mapping(one_shot_commitments, "one_shot_commitments")
    commitment_provenance: dict[str, Any] = {"legacy_unregistered_mapping": True}
    if "commitments" in commitment_document:
        if commitment_document.get("analysis") != "preregistered_one_shot_commitments":
            raise AuditValidationError(
                "ABLATION_COMMITMENT_DOCUMENT",
                "commitment document has an unexpected analysis type",
            )
        if commitment_document.get("uses_ready_records") is not False:
            raise AuditValidationError(
                "ABLATION_READY_LEAKAGE",
                "one-shot commitments must not use READY records",
            )
        expected_hash = commitment_document.get("report_sha256")
        actual_hash = canonical_document_sha256(commitment_document, "report_sha256")
        if expected_hash != actual_hash:
            raise AuditValidationError(
                "ABLATION_COMMITMENT_HASH", "commitment document hash is invalid"
            )
        commitment_provenance = {
            "legacy_unregistered_mapping": False,
            "commitment_report_sha256": expected_hash,
            "registered_plan_sha256": commitment_document.get("registered_plan_sha256"),
            "raw_visible_input_sha256": commitment_document.get(
                "raw_visible_input_sha256"
            ),
            "uses_ready_records": False,
            "uses_ready_expected_cost": False,
        }
    commitments = _mapping(
        commitment_document.get("commitments", commitment_document),
        "one_shot_commitments.commitments",
    )
    pairs: list[dict[str, Any]] = []
    for index, task_id, _, _, raw_rows in _mask_records(action_records, "READY"):
        actions = []
        for row_index, raw in enumerate(raw_rows):
            action = _ablation_action(
                _mapping(raw, f"action_records[{index}].rows[{row_index}]"),
                f"action_records[{index}].rows[{row_index}]",
            )
            if action is not None:
                actions.append(action)
        if not actions:
            raise AuditValidationError(
                "ABLATION_NO_SAFE_ACTION",
                "READY mask has no costed safe action",
                task_id=task_id,
            )
        by_id = {str(row["action_id"]): row for row in actions}
        priority = _commitment_priority(commitments.get(task_id), task_id)
        commitment = next(
            (action_id for action_id in priority if action_id in by_id), None
        )
        if commitment is None:
            raise AuditValidationError(
                "ABLATION_UNSAFE_COMMITMENT",
                "all preregistered one-shot actions are absent or hard-mask rejected",
                task_id=task_id,
                action_priority=list(priority),
            )
        recourse = min(
            actions, key=lambda row: (row["expected_cost"], row["action_id"])
        )
        no_output = min(
            actions,
            key=lambda row: (row["without_output_size_cost"], row["action_id"]),
        )
        no_queue = min(
            actions,
            key=lambda row: (row["without_fresh_queue_cost"], row["action_id"]),
        )
        committed = by_id[commitment]
        pairs.append(
            {
                "task_id": task_id,
                "ready_recourse": {
                    "action_id": recourse["action_id"],
                    "expected_cost": recourse["expected_cost"],
                },
                "one_shot": {
                    "primary_action_id": priority[0],
                    "action_id": commitment,
                    "expected_cost": committed["expected_cost"],
                    "hard_mask_fallback_applied": commitment != priority[0],
                    "paired_excess_cost": committed["expected_cost"]
                    - recourse["expected_cost"],
                },
                "without_output_size": {
                    "action_id": no_output["action_id"],
                    "expected_cost": no_output["without_output_size_cost"],
                    "paired_excess_cost": no_output["without_output_size_cost"]
                    - recourse["expected_cost"],
                    "replacement": "registered_conservative_output_size_bound",
                },
                "without_fresh_queue": {
                    "action_id": no_queue["action_id"],
                    "expected_cost": no_queue["without_fresh_queue_cost"],
                    "paired_excess_cost": no_queue["without_fresh_queue_cost"]
                    - recourse["expected_cost"],
                    "replacement": "registered_conservative_stale_queue_bound",
                },
            }
        )
    return _finish(
        {
            "schema_version": "1.0",
            "analysis": "two_stage_information_ablation",
            "paired_unit": "task",
            "hard_mask_bypassed": False,
            "only_allowed_actions_compared": True,
            "one_shot_commitment_provenance": commitment_provenance,
            "units": {"expected_cost": "normalized_cost"},
            "pair_count": len(pairs),
            "pairs": pairs,
        },
        {"actions": action_records, "one_shot_commitments": commitments},
    )


def _task_retry_cost(task: Mapping[str, Any], task_id: str) -> tuple[float, float]:
    count = task.get("attempt_started_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise AuditValidationError(
            "FAILURE_ATTEMPT_COUNT", "attempt_started_count must be an integer >= 0"
        )
    attempts = _sequence(task.get("anon_attempts"), f"task[{task_id}].anon_attempts")
    if len(attempts) != count:
        raise AuditValidationError(
            "FAILURE_ATTEMPT_MISMATCH",
            "attempt rows do not match attempt_started_count",
            task_id=task_id,
        )
    latency = 0.0
    energy = 0.0
    for index, raw in enumerate(attempts):
        attempt = _mapping(raw, f"task[{task_id}].anon_attempts[{index}]")
        attempt_number = attempt.get("attempt")
        if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
            raise AuditValidationError(
                "FAILURE_ATTEMPT_INDEX", "attempt index must be an integer"
            )
        if attempt_number <= 1:
            continue
        if "latency_s" not in attempt or "vehicle_energy_j" not in attempt:
            raise AuditValidationError(
                "FAILURE_RETRY_FIELDS_MISSING",
                "retry accounting requires measured latency_s and vehicle_energy_j",
                task_id=task_id,
                attempt=attempt_number,
            )
        latency += _number(attempt["latency_s"], "retry.latency_s", minimum=0.0)
        energy += _number(
            attempt["vehicle_energy_j"], "retry.vehicle_energy_j", minimum=0.0
        )
    return latency, energy


def _task_downlink_cost(
    task: Mapping[str, Any], task_id: str
) -> tuple[float, float, float]:
    audit = _sequence(task.get("network_audit"), f"task[{task_id}].network_audit")
    start: float | None = None
    latency = vehicle_energy = rsu_energy = 0.0
    for index, raw in enumerate(audit):
        row = _mapping(raw, f"task[{task_id}].network_audit[{index}]")
        if row.get("direction") != "DL":
            continue
        status = _text(row.get("status"), "network_audit.status")
        time_s = _number(row.get("time_s"), "network_audit.time_s", minimum=0.0)
        if status == "START":
            if start is not None:
                raise AuditValidationError(
                    "FAILURE_DL_OVERLAP", "overlapping downlink attempts are ambiguous"
                )
            start = time_s
        elif status in {"DONE", "FAIL"}:
            if start is None or time_s < start:
                raise AuditValidationError(
                    "FAILURE_DL_PAIR", "downlink terminal record has no valid start"
                )
            for field in ("vehicle_energy_j", "rsu_energy_j"):
                if field not in row:
                    raise AuditValidationError(
                        "FAILURE_DL_FIELDS_MISSING",
                        "downlink terminal record lacks paired energy",
                        task_id=task_id,
                        field=field,
                    )
            latency += time_s - start
            vehicle_energy += _number(
                row["vehicle_energy_j"], "downlink.vehicle_energy_j", minimum=0.0
            )
            rsu_energy += _number(
                row["rsu_energy_j"], "downlink.rsu_energy_j", minimum=0.0
            )
            start = None
    if start is not None:
        raise AuditValidationError(
            "FAILURE_DL_INCOMPLETE", "downlink start has no terminal accounting record"
        )
    return latency, vehicle_energy, rsu_energy


def _task_actual_path(task: Mapping[str, Any], task_id: str) -> tuple[str, ...]:
    raw = task.get("actual_path")
    if raw is None and isinstance(task.get("actual_path_json"), str):
        import json

        try:
            raw = json.loads(str(task["actual_path_json"]))
        except json.JSONDecodeError as exc:
            raise AuditValidationError(
                "FAILURE_PATH_JSON", "actual_path_json is invalid", task_id=task_id
            ) from exc
    path = tuple(
        _text(value, f"task[{task_id}].actual_path")
        for value in _sequence(raw, f"task[{task_id}].actual_path")
    )
    return path


def _failed_anon_attempts(
    task: Mapping[str, Any], task_id: str
) -> list[dict[str, Any]]:
    attempts = _sequence(task.get("anon_attempts"), f"task[{task_id}].anon_attempts")
    failed: list[dict[str, Any]] = []
    for index, raw in enumerate(attempts):
        attempt = _mapping(raw, f"task[{task_id}].anon_attempts[{index}]")
        reason = attempt.get("failure_reason")
        if reason in (None, "", "NONE"):
            continue
        reason_text = _text(reason, "anon_attempt.failure_reason")
        for field in ("executed_work_s", "vehicle_energy_j"):
            if field not in attempt:
                raise AuditValidationError(
                    "FAILURE_ANON_FIELDS_MISSING",
                    "failed anonymization attempts require measured work and energy",
                    task_id=task_id,
                    attempt=attempt.get("attempt"),
                    field=field,
                )
        failed.append(
            {
                "reason": reason_text,
                "executed_work_s": _number(
                    attempt["executed_work_s"], "anon.executed_work_s", minimum=0.0
                ),
                "vehicle_energy_j": _number(
                    attempt["vehicle_energy_j"], "anon.vehicle_energy_j", minimum=0.0
                ),
            }
        )
    return failed


def _explicit_fail_actions(
    action_records: Sequence[Mapping[str, Any]],
) -> tuple[int, tuple[str, ...]]:
    task_ids: set[str] = set()
    count = 0
    for index, raw in enumerate(action_records):
        record = _mapping(raw, f"action_records[{index}]")
        if record.get("record_kind") != "POLICY_DECISION":
            continue
        executed = _mapping(record.get("executed"), f"action_records[{index}].executed")
        if executed.get("kind") != "FAIL":
            continue
        count += 1
        task_ids.add(_text(record.get("task_id"), "policy_decision.task_id"))
    return count, tuple(sorted(task_ids))


def audit_failure_cost_completeness(
    task_rows: Sequence[Mapping[str, Any]],
    action_records: Sequence[Mapping[str, Any]],
    event_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Quantify omitted physical/cost terms, refusing incomplete accounting fields."""

    tasks = _sequence(task_rows, "task_rows")
    actions = _sequence(action_records, "action_records")
    events = _sequence(event_records, "event_records")
    if not tasks:
        raise AuditValidationError("FAILURE_NO_TASKS", "task audit is empty")
    omissions = {
        "retry": {"latency_s": 0.0, "vehicle_energy_j": 0.0, "tasks_affected": 0},
        "downlink": {
            "latency_s": 0.0,
            "vehicle_energy_j": 0.0,
            "rsu_energy_j": 0.0,
            "tasks_affected": 0,
        },
        "rsu_energy": {"rsu_energy_j": 0.0, "tasks_affected": 0},
        "failure": {"normalized_cost": 0.0, "tasks_affected": 0},
        "anonymization_failure": {
            "failed_attempt_count": 0,
            "executed_work_s": 0.0,
            "vehicle_energy_j": 0.0,
            "tasks_affected": 0,
            "reason_counts": {},
            "accounting": "measured_failed_attempt_components",
        },
        "local_fallback": {
            "task_count": 0,
            "task_ids": [],
            "accounting": "exact_structural_count",
        },
        "explicit_fail_action": {
            "decision_count": 0,
            "task_count": 0,
            "task_ids": [],
            "accounting": "exact_POLICY_DECISION_executed_count",
        },
    }
    anon_failure_reasons: Counter[str] = Counter()
    fallback_task_ids: list[str] = []
    task_details = []
    for index, raw in enumerate(tasks):
        task = _mapping(raw, f"task_rows[{index}]")
        task_id = _text(task.get("task_id"), f"task_rows[{index}].task_id")
        if "failure_penalty_cost" not in task:
            raise AuditValidationError(
                "FAILURE_PENALTY_FIELD_MISSING",
                "failure term cannot be isolated from all-task loss",
                task_id=task_id,
            )
        retry_latency, retry_energy = _task_retry_cost(task, task_id)
        dl_latency, dl_vehicle, dl_rsu = _task_downlink_cost(task, task_id)
        rsu_total = _number(
            task.get("rsu_attributed_energy_j"),
            f"task[{task_id}].rsu_attributed_energy_j",
            minimum=0.0,
        )
        failure = _number(
            task["failure_penalty_cost"],
            f"task[{task_id}].failure_penalty_cost",
            minimum=0.0,
        )
        failed_attempts = _failed_anon_attempts(task, task_id)
        actual_path = _task_actual_path(task, task_id)
        local_fallback = bool(
            actual_path
            and actual_path[-1] == "LOCAL_FER"
            and any(
                stage.startswith("ANON#")
                or stage.startswith("UL:")
                or stage.startswith("EDGE:")
                or stage.startswith("DL:")
                for stage in actual_path[:-1]
            )
        )
        if retry_latency > 0.0 or retry_energy > 0.0:
            omissions["retry"]["tasks_affected"] += 1
        omissions["retry"]["latency_s"] += retry_latency
        omissions["retry"]["vehicle_energy_j"] += retry_energy
        if dl_latency > 0.0 or dl_vehicle > 0.0 or dl_rsu > 0.0:
            omissions["downlink"]["tasks_affected"] += 1
        omissions["downlink"]["latency_s"] += dl_latency
        omissions["downlink"]["vehicle_energy_j"] += dl_vehicle
        omissions["downlink"]["rsu_energy_j"] += dl_rsu
        if rsu_total > 0.0:
            omissions["rsu_energy"]["tasks_affected"] += 1
        omissions["rsu_energy"]["rsu_energy_j"] += rsu_total
        if failure > 0.0:
            omissions["failure"]["tasks_affected"] += 1
        omissions["failure"]["normalized_cost"] += failure
        if failed_attempts:
            omissions["anonymization_failure"]["tasks_affected"] += 1
        omissions["anonymization_failure"]["failed_attempt_count"] += len(
            failed_attempts
        )
        omissions["anonymization_failure"]["executed_work_s"] += sum(
            row["executed_work_s"] for row in failed_attempts
        )
        omissions["anonymization_failure"]["vehicle_energy_j"] += sum(
            row["vehicle_energy_j"] for row in failed_attempts
        )
        anon_failure_reasons.update(row["reason"] for row in failed_attempts)
        if local_fallback:
            fallback_task_ids.append(task_id)
        task_details.append(
            {
                "task_id": task_id,
                "delete_retry": {
                    "omitted_latency_s": retry_latency,
                    "omitted_vehicle_energy_j": retry_energy,
                },
                "delete_downlink": {
                    "omitted_latency_s": dl_latency,
                    "omitted_vehicle_energy_j": dl_vehicle,
                    "omitted_rsu_energy_j": dl_rsu,
                },
                "delete_rsu_energy": {"omitted_rsu_energy_j": rsu_total},
                "delete_failure_term": {"omitted_normalized_cost": failure},
                "delete_anonymization_failures": {
                    "failed_attempt_count": len(failed_attempts),
                    "omitted_executed_work_s": sum(
                        row["executed_work_s"] for row in failed_attempts
                    ),
                    "omitted_vehicle_energy_j": sum(
                        row["vehicle_energy_j"] for row in failed_attempts
                    ),
                    "reason_counts": dict(
                        sorted(
                            Counter(row["reason"] for row in failed_attempts).items()
                        )
                    ),
                },
                "delete_local_fallback": {"fallback_path_present": local_fallback},
            }
        )
    explicit_fail_count, explicit_fail_task_ids = _explicit_fail_actions(actions)
    omissions["anonymization_failure"]["reason_counts"] = dict(
        sorted(anon_failure_reasons.items())
    )
    omissions["local_fallback"].update(
        {
            "task_count": len(fallback_task_ids),
            "task_ids": sorted(fallback_task_ids),
        }
    )
    omissions["explicit_fail_action"].update(
        {
            "decision_count": explicit_fail_count,
            "task_count": len(explicit_fail_task_ids),
            "task_ids": list(explicit_fail_task_ids),
        }
    )
    return _finish(
        {
            "schema_version": "1.0",
            "analysis": "failure_cost_completeness",
            "status": "COMPLETE",
            "units": {
                "latency": "s",
                "vehicle_energy": "J",
                "rsu_energy": "J",
                "failure_penalty": "normalized_cost",
            },
            "task_count": len(tasks),
            "action_record_count": len(actions),
            "event_record_count": len(events),
            "omissions": omissions,
            "tasks": task_details,
        },
        {"tasks": tasks, "actions": actions, "events": events},
    )


def exact_finite_scenario_oracle(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    esl_action_sequence: Sequence[str],
    max_sequences: int = 100_000,
) -> dict[str, Any]:
    """Exhaustively solve a bounded hard-safe open-loop scenario ratio problem."""

    raw_scenarios = _sequence(scenarios, "scenarios")
    if not raw_scenarios:
        raise AuditValidationError("ORACLE_NO_SCENARIOS", "scenario set is empty")
    parsed = []
    horizon: int | None = None
    action_ids: set[str] | None = None
    probability_sum = 0.0
    for scenario_index, raw in enumerate(raw_scenarios):
        scenario = _mapping(raw, f"scenarios[{scenario_index}]")
        scenario_id = _text(scenario.get("scenario_id"), "scenario_id")
        probability = _number(
            scenario.get("probability"), "scenario.probability", minimum=0.0
        )
        stages = _sequence(scenario.get("stages"), "scenario.stages")
        if not stages:
            raise AuditValidationError("ORACLE_EMPTY_HORIZON", "scenario has no stages")
        if horizon is None:
            horizon = len(stages)
        elif len(stages) != horizon:
            raise AuditValidationError(
                "ORACLE_HORIZON_MISMATCH", "scenario horizons do not match"
            )
        parsed_stages = []
        for stage_index, raw_stage in enumerate(stages):
            stage = _mapping(raw_stage, f"scenario.stages[{stage_index}]")
            normalized = {}
            for raw_action_id, raw_outcome in stage.items():
                action_id = _text(raw_action_id, "action_id")
                outcome = _mapping(raw_outcome, "scenario action outcome")
                normalized[action_id] = {
                    "hard_safe": outcome.get("hard_safe") is True,
                    "cost": _number(outcome.get("cost"), "outcome.cost", minimum=0.0),
                    "duration_s": _number(
                        outcome.get("duration_s"),
                        "outcome.duration_s",
                        strict_positive=True,
                    ),
                }
            if not normalized:
                raise AuditValidationError(
                    "ORACLE_NO_ACTIONS", "scenario stage has no actions"
                )
            current_ids = set(normalized)
            if action_ids is None:
                action_ids = current_ids
            elif current_ids != action_ids:
                raise AuditValidationError(
                    "ORACLE_ACTION_MISMATCH",
                    "all scenario stages need identical actions",
                )
            parsed_stages.append(normalized)
        parsed.append((scenario_id, probability, parsed_stages))
        probability_sum += probability
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
        raise AuditValidationError(
            "ORACLE_PROBABILITY", "scenario probabilities must sum to one"
        )
    assert horizon is not None and action_ids is not None
    ordered_actions = tuple(sorted(action_ids))
    sequence_count = len(ordered_actions) ** horizon
    if (
        isinstance(max_sequences, bool)
        or not isinstance(max_sequences, int)
        or max_sequences < 1
    ):
        raise AuditValidationError("ORACLE_LIMIT", "max_sequences must be positive")
    if sequence_count > max_sequences:
        raise AuditValidationError(
            "ORACLE_COMPLEXITY_LIMIT",
            "exact enumeration exceeds the registered cap",
            candidate_sequences=sequence_count,
            max_sequences=max_sequences,
        )

    def evaluate(sequence: Sequence[str]) -> tuple[float, float, float] | None:
        expected_cost = 0.0
        expected_duration = 0.0
        for _, probability, stages in parsed:
            scenario_cost = 0.0
            scenario_duration = 0.0
            for stage_index, action_id in enumerate(sequence):
                outcome = stages[stage_index][action_id]
                if not outcome["hard_safe"]:
                    return None
                scenario_cost += outcome["cost"]
                scenario_duration += outcome["duration_s"]
            expected_cost += probability * scenario_cost
            expected_duration += probability * scenario_duration
        return expected_cost / expected_duration, expected_cost, expected_duration

    best: tuple[float, tuple[str, ...], float, float] | None = None
    feasible_count = 0
    for sequence in itertools.product(ordered_actions, repeat=horizon):
        result = evaluate(sequence)
        if result is None:
            continue
        feasible_count += 1
        ratio, cost, duration = result
        candidate = (ratio, sequence, cost, duration)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise AuditValidationError(
            "ORACLE_NO_SAFE_SEQUENCE", "no hard-safe sequence exists"
        )
    esl_sequence = tuple(
        _text(action_id, "esl_action_sequence")
        for action_id in _sequence(esl_action_sequence, "esl_action_sequence")
    )
    if len(esl_sequence) != horizon or any(
        action_id not in action_ids for action_id in esl_sequence
    ):
        raise AuditValidationError(
            "ORACLE_ESL_SEQUENCE", "ESL sequence has invalid length or action"
        )
    esl = evaluate(esl_sequence)
    if esl is None:
        raise AuditValidationError(
            "ORACLE_ESL_UNSAFE", "ESL sequence violates a scenario hard mask"
        )
    optimum_ratio, optimum_sequence, optimum_cost, optimum_duration = best
    esl_ratio, esl_cost, esl_duration = esl
    return _finish(
        {
            "schema_version": "1.0",
            "analysis": "exact_finite_scenario_ratio_oracle",
            "oracle_scope": "bounded_open_loop_action_sequence",
            "hard_mask_bypassed": False,
            "units": {
                "cost": "normalized_cost",
                "duration": "s",
                "ratio": "normalized_cost/s",
            },
            "optimum": {
                "action_sequence": list(optimum_sequence),
                "expected_cost": optimum_cost,
                "expected_duration_s": optimum_duration,
                "exact_ratio": optimum_ratio,
            },
            "esl": {
                "action_sequence": list(esl_sequence),
                "expected_cost": esl_cost,
                "expected_duration_s": esl_duration,
                "ratio": esl_ratio,
                "absolute_gap": esl_ratio - optimum_ratio,
                "relative_gap": (
                    (esl_ratio - optimum_ratio) / optimum_ratio
                    if optimum_ratio > 0.0
                    else 0.0
                ),
            },
            "complexity": {
                "scenario_count": len(parsed),
                "horizon": horizon,
                "action_count": len(ordered_actions),
                "candidate_sequences": sequence_count,
                "evaluated_sequences": sequence_count,
                "hard_safe_sequences": feasible_count,
                "max_sequences": max_sequences,
                "wallclock_used_in_simulation": False,
                "wallclock_reported": False,
            },
        },
        {
            "scenarios": raw_scenarios,
            "esl_action_sequence": esl_sequence,
            "max_sequences": max_sequences,
        },
    )


def exact_adaptive_scenario_tree_oracle(
    scenario_tree: Mapping[str, Any],
    *,
    esl_contingent_policy: Mapping[str, str],
    max_policies: int = 100_000,
) -> dict[str, Any]:
    """Enumerate contingent actions at explicit observation nodes exactly."""

    tree = _mapping(scenario_tree, "scenario_tree")
    root_id = _text(tree.get("root_id"), "scenario_tree.root_id")
    raw_nodes = _sequence(tree.get("nodes"), "scenario_tree.nodes")
    if not raw_nodes:
        raise AuditValidationError("TREE_NO_NODES", "adaptive scenario tree is empty")
    nodes: dict[str, dict[str, Any]] = {}
    for node_index, raw_node in enumerate(raw_nodes):
        node = _mapping(raw_node, f"scenario_tree.nodes[{node_index}]")
        node_id = _text(node.get("node_id"), "node.node_id")
        if node_id in nodes:
            raise AuditValidationError(
                "TREE_DUPLICATE_NODE", "scenario observation node IDs must be unique"
            )
        actions_raw = _mapping(node.get("actions"), f"node[{node_id}].actions")
        if not actions_raw:
            raise AuditValidationError(
                "TREE_NO_ACTIONS",
                "each observation node requires actions",
                node_id=node_id,
            )
        actions: dict[str, Any] = {}
        for raw_action_id, raw_outcome in actions_raw.items():
            action_id = _text(raw_action_id, f"node[{node_id}].action_id")
            outcome = _mapping(raw_outcome, f"node[{node_id}].actions[{action_id}]")
            branches = []
            probability_sum = 0.0
            for branch_index, raw_branch in enumerate(
                _sequence(
                    outcome.get("branches", ()),
                    f"node[{node_id}].actions[{action_id}].branches",
                )
            ):
                branch = _mapping(raw_branch, "scenario branch")
                probability = _number(
                    branch.get("probability"), "branch.probability", minimum=0.0
                )
                if probability <= 0.0:
                    raise AuditValidationError(
                        "TREE_BRANCH_PROBABILITY",
                        "branch probabilities must be strictly positive",
                    )
                branches.append(
                    {
                        "probability": probability,
                        "next_node_id": _text(
                            branch.get("next_node_id"), "branch.next_node_id"
                        ),
                    }
                )
                probability_sum += probability
            if branches and not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
                raise AuditValidationError(
                    "TREE_BRANCH_PROBABILITY",
                    "non-terminal branch probabilities must sum to one",
                    node_id=node_id,
                    action_id=action_id,
                )
            actions[action_id] = {
                "hard_safe": outcome.get("hard_safe") is True,
                "cost": _number(outcome.get("cost"), "outcome.cost", minimum=0.0),
                "duration_s": _number(
                    outcome.get("duration_s"),
                    "outcome.duration_s",
                    strict_positive=True,
                ),
                "branches": tuple(branches),
            }
        nodes[node_id] = {"actions": actions}
    if root_id not in nodes:
        raise AuditValidationError("TREE_ROOT", "root observation node is missing")
    for node_id, node in nodes.items():
        for action_id, outcome in node["actions"].items():
            for branch in outcome["branches"]:
                if branch["next_node_id"] not in nodes:
                    raise AuditValidationError(
                        "TREE_BRANCH_TARGET",
                        "branch references an unknown observation node",
                        node_id=node_id,
                        action_id=action_id,
                        target=branch["next_node_id"],
                    )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise AuditValidationError("TREE_CYCLE", "scenario tree contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for outcome in nodes[node_id]["actions"].values():
            for branch in outcome["branches"]:
                visit(branch["next_node_id"])
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root_id)
    if visited != set(nodes):
        raise AuditValidationError(
            "TREE_UNREACHABLE_NODE",
            "scenario tree contains nodes unreachable from the root",
            unreachable=sorted(set(nodes) - visited),
        )
    ordered_nodes = tuple(sorted(nodes))
    action_options = tuple(
        tuple(sorted(nodes[node_id]["actions"])) for node_id in ordered_nodes
    )
    candidate_policies = math.prod(len(options) for options in action_options)
    if (
        isinstance(max_policies, bool)
        or not isinstance(max_policies, int)
        or max_policies < 1
    ):
        raise AuditValidationError("TREE_LIMIT", "max_policies must be positive")
    if candidate_policies > max_policies:
        raise AuditValidationError(
            "TREE_COMPLEXITY_LIMIT",
            "contingent policy enumeration exceeds the registered cap",
            candidate_policies=candidate_policies,
            max_policies=max_policies,
        )

    def evaluate(policy: Mapping[str, str]) -> tuple[float, float, float] | None:
        memo: dict[str, tuple[float, float] | None] = {}

        def node_value(node_id: str) -> tuple[float, float] | None:
            if node_id in memo:
                return memo[node_id]
            action_id = policy[node_id]
            outcome = nodes[node_id]["actions"].get(action_id)
            if outcome is None or not outcome["hard_safe"]:
                memo[node_id] = None
                return None
            cost = float(outcome["cost"])
            duration = float(outcome["duration_s"])
            for branch in outcome["branches"]:
                child = node_value(branch["next_node_id"])
                if child is None:
                    memo[node_id] = None
                    return None
                cost += branch["probability"] * child[0]
                duration += branch["probability"] * child[1]
            memo[node_id] = (cost, duration)
            return memo[node_id]

        root = node_value(root_id)
        if root is None:
            return None
        return root[0] / root[1], root[0], root[1]

    best: tuple[float, tuple[str, ...], float, float] | None = None
    hard_safe_policies = 0
    for action_tuple in itertools.product(*action_options):
        policy = dict(zip(ordered_nodes, action_tuple, strict=True))
        result = evaluate(policy)
        if result is None:
            continue
        hard_safe_policies += 1
        ratio, cost, duration = result
        candidate = (ratio, action_tuple, cost, duration)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise AuditValidationError(
            "TREE_NO_SAFE_POLICY", "no contingent policy is hard-safe"
        )
    esl_policy_raw = _mapping(esl_contingent_policy, "esl_contingent_policy")
    if set(esl_policy_raw) != set(ordered_nodes):
        raise AuditValidationError(
            "TREE_ESL_POLICY",
            "ESL contingent policy must assign every observation node",
        )
    esl_policy = {
        node_id: _text(esl_policy_raw[node_id], f"esl_policy.{node_id}")
        for node_id in ordered_nodes
    }
    esl = evaluate(esl_policy)
    if esl is None:
        raise AuditValidationError(
            "TREE_ESL_UNSAFE", "ESL contingent policy is not hard-safe"
        )
    optimum_ratio, optimum_actions, optimum_cost, optimum_duration = best
    optimum_policy = dict(zip(ordered_nodes, optimum_actions, strict=True))
    esl_ratio, esl_cost, esl_duration = esl
    terminal_action_count = sum(
        not outcome["branches"]
        for node in nodes.values()
        for outcome in node["actions"].values()
    )
    return _finish(
        {
            "schema_version": "1.0",
            "analysis": "exact_adaptive_scenario_tree_ratio_oracle",
            "oracle_scope": "explicit_observation_node_contingent_policy",
            "hard_mask_bypassed": False,
            "units": {
                "cost": "normalized_cost",
                "duration": "s",
                "ratio": "normalized_cost/s",
            },
            "optimum": {
                "contingent_policy": optimum_policy,
                "expected_cost": optimum_cost,
                "expected_duration_s": optimum_duration,
                "exact_ratio": optimum_ratio,
            },
            "esl": {
                "contingent_policy": esl_policy,
                "expected_cost": esl_cost,
                "expected_duration_s": esl_duration,
                "ratio": esl_ratio,
                "absolute_gap": esl_ratio - optimum_ratio,
                "relative_gap": (
                    (esl_ratio - optimum_ratio) / optimum_ratio
                    if optimum_ratio > 0.0
                    else 0.0
                ),
            },
            "complexity": {
                "observation_node_count": len(nodes),
                "terminal_action_count": terminal_action_count,
                "candidate_policies": candidate_policies,
                "evaluated_policies": candidate_policies,
                "hard_safe_policies": hard_safe_policies,
                "max_policies": max_policies,
                "stable_node_order": list(ordered_nodes),
                "wallclock_used_in_simulation": False,
                "wallclock_reported": False,
            },
        },
        {
            "scenario_tree": tree,
            "esl_contingent_policy": esl_policy,
            "max_policies": max_policies,
        },
    )


__all__ = [
    "AuditValidationError",
    "audit_failure_cost_completeness",
    "audit_hard_mask_counterfactual",
    "build_preregistered_one_shot_commitments",
    "evaluate_two_stage_information_ablation",
    "exact_adaptive_scenario_tree_oracle",
    "exact_finite_scenario_oracle",
]
