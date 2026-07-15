"""Strict offline audits and a finite exact oracle for paper experiments.

Every function is pure and read-only.  The hard-mask audit validates logged
physical decisions against their causal mask, while rejected rows remain
counterfactual and are never converted into executable actions.  Missing
pairing or cost-accounting fields cause a structured refusal instead of an
inferred or zero-filled result.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .profiles import canonical_document_sha256, canonical_json_bytes


_FAILURE_COVERAGE_CATEGORIES = (
    "retry",
    "anonymization_failure",
    "uplink",
    "uplink_failure",
    "admission_accept",
    "admission_reject",
    "rsu_ingress",
    "edge_execution",
    "rsu_failure",
    "rsu_attributed_energy",
    "downlink",
    "downlink_failure",
    "local_fallback",
    "explicit_fail_action",
    "failure_penalty",
)

_FAILURE_ACCOUNTING_CATEGORIES = frozenset(
    {
        "retry",
        "anonymization_failure",
        "uplink",
        "uplink_failure",
        "rsu_attributed_energy",
        "downlink",
        "downlink_failure",
        "failure_penalty",
    }
)

_FAILURE_COVERAGE_SCOPES = {
    "retry": "measured retry latency and vehicle energy",
    "anonymization_failure": "measured failed-attempt work and vehicle energy",
    "uplink": "paired start/terminal records and paired endpoint energy",
    "uplink_failure": "paired failed-uplink terminal record and endpoint energy",
    "admission_accept": "recorded admission outcome only; not an atomicity proof",
    "admission_reject": "recorded admission outcome only; not an atomicity proof",
    "rsu_ingress": "paired ingress start/terminal structure",
    "edge_execution": "executed path contains an edge stage",
    "rsu_failure": "recorded RSU/edge failure marker",
    "rsu_attributed_energy": "nonzero numeric RSU attributed energy",
    "downlink": "paired start/terminal records and paired endpoint energy",
    "downlink_failure": "paired failed-downlink terminal record and endpoint energy",
    "local_fallback": "executed path contains a post-attempt local fallback",
    "explicit_fail_action": "executed POLICY_DECISION fail action",
    "failure_penalty": "nonzero numeric all-task failure penalty",
}


def _coverage_status(category: str, observed: int) -> str:
    if not observed:
        return "NOT_OBSERVED"
    if category in _FAILURE_ACCOUNTING_CATEGORIES:
        return "OBSERVED_ACCOUNTING_VALIDATED"
    return "OBSERVED_STRUCTURE_VALIDATED"


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


def _logged_action_id(value: Any, field: str) -> tuple[str, str]:
    """Return the canonical ID and stage of one logged closed action.

    This intentionally mirrors the small, closed wire representation instead
    of accepting an arbitrary identifier supplied by an audit producer.
    """

    action = _mapping(value, field)
    stage = _text(action.get("stage"), f"{field}.stage")
    kind = _text(action.get("kind"), f"{field}.kind")
    if stage not in {"RAW", "READY"}:
        raise AuditValidationError(
            "AUDIT_EXECUTED_ACTION_INVALID",
            "executed action has an invalid stage",
            field=field,
            stage=stage,
        )
    expected: set[str]
    if kind == "FAIL":
        expected = set()
    elif kind == "LOCAL":
        expected = {"local_model_id"}
    elif kind == "PIPE" and stage == "RAW":
        expected = {"pipeline_id"}
    elif kind == "EDGE" and stage == "READY":
        expected = {"rsu_id", "edge_model_id"}
    else:
        raise AuditValidationError(
            "AUDIT_EXECUTED_ACTION_INVALID",
            "executed action kind is illegal at its stage",
            field=field,
            stage=stage,
            kind=kind,
        )
    identifiers: dict[str, str] = {}
    for name in ("local_model_id", "pipeline_id", "rsu_id", "edge_model_id"):
        raw = action.get(name)
        if raw is not None:
            identifiers[name] = _text(raw, f"{field}.{name}")
    if set(identifiers) != expected:
        raise AuditValidationError(
            "AUDIT_EXECUTED_ACTION_INVALID",
            "executed action identifiers do not match its kind",
            field=field,
            stage=stage,
            kind=kind,
            expected_identifiers=sorted(expected),
            present_identifiers=sorted(identifiers),
        )
    action_id = "|".join(
        (
            stage,
            kind,
            identifiers.get("local_model_id", ""),
            identifiers.get("pipeline_id", ""),
            identifiers.get("rsu_id", ""),
            identifiers.get("edge_model_id", ""),
        )
    )
    return action_id, stage


def _selected_execution_records(
    action_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Select committed repair rows, falling back to policy rows per stage."""

    repairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    policies: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ignored_automatic_repairs = 0
    for index, raw in enumerate(_sequence(action_records, "action_records")):
        record = _mapping(raw, f"action_records[{index}]")
        kind = record.get("record_kind")
        if kind not in {"EXECUTION_REPAIR", "POLICY_DECISION"}:
            continue
        if "executed" not in record:
            if (
                kind == "EXECUTION_REPAIR"
                and record.get("repair") == "FROZEN_LOCAL_FALLBACK"
            ):
                # This is an automatic post-failure transition, not a RAW or
                # READY decision member.  It has no corresponding decision mask.
                ignored_automatic_repairs += 1
                continue
            raise AuditValidationError(
                "AUDIT_EXECUTED_ACTION_MISSING",
                "decision record has no executed action",
                record_index=index,
                record_kind=kind,
            )
        action_id, stage = _logged_action_id(
            record.get("executed"), f"action_records[{index}].executed"
        )
        task_id = _text(record.get("task_id"), f"action_records[{index}].task_id")
        declared_stage = record.get("executed_stage", record.get("stage"))
        if declared_stage is not None and declared_stage != stage:
            raise AuditValidationError(
                "AUDIT_EXECUTION_STAGE_MISMATCH",
                "execution record stage disagrees with executed action",
                record_index=index,
                task_id=task_id,
                declared_stage=declared_stage,
                executed_stage=stage,
            )
        raw_time = record.get("time_s")
        execution_time_s = (
            None
            if raw_time is None
            else _number(raw_time, f"action_records[{index}].time_s", minimum=0.0)
        )
        normalized = {
            "record_index": index,
            "record_kind": kind,
            "task_id": task_id,
            "stage": stage,
            "action_id": action_id,
            "time_s": execution_time_s,
            "execution_check_id": record.get("execution_check_id"),
        }
        if kind == "EXECUTION_REPAIR":
            if record.get("execution_check_id") is None:
                raise AuditValidationError(
                    "AUDIT_EXECUTION_BINDING_MISSING",
                    "execution repair has no execution-time mask binding",
                    record_index=index,
                    task_id=task_id,
                    stage=stage,
                )
            normalized["execution_check_id"] = _text(
                record.get("execution_check_id"),
                f"action_records[{index}].execution_check_id",
            )
        elif normalized["execution_check_id"] is not None:
            normalized["execution_check_id"] = _text(
                normalized["execution_check_id"],
                f"action_records[{index}].execution_check_id",
            )
        target = repairs if kind == "EXECUTION_REPAIR" else policies
        target.setdefault((task_id, stage), []).append(normalized)

    selected: list[dict[str, Any]] = []
    ignored_policy_count = 0
    for key in sorted(set(repairs) | set(policies)):
        if repairs.get(key):
            selected.extend(repairs[key])
            ignored_policy_count += len(policies.get(key, ()))
        else:
            selected.extend(policies[key])

    by_moment: set[tuple[str, str, float | None]] = set()
    for record in selected:
        moment = (record["task_id"], record["stage"], record["time_s"])
        if moment in by_moment:
            raise AuditValidationError(
                "AUDIT_EXECUTION_AMBIGUOUS",
                "multiple selected execution records occupy the same task/stage moment",
                task_id=record["task_id"],
                stage=record["stage"],
                time_s=record["time_s"],
            )
        by_moment.add(moment)
    selected.sort(
        key=lambda row: (
            row["task_id"],
            row["stage"],
            math.inf if row["time_s"] is None else row["time_s"],
            row["record_index"],
        )
    )
    return selected, ignored_policy_count, ignored_automatic_repairs


def _validate_executed_mask_membership(
    action_records: Sequence[Mapping[str, Any]],
    masks: Sequence[tuple[int, str, str, Mapping[str, Any], Sequence[Any]]],
) -> dict[str, Any]:
    grouped_masks: dict[
        tuple[str, str],
        list[tuple[int, float | None, Mapping[str, Any], Sequence[Any]]],
    ] = {}
    for index, task_id, stage, record, rows in masks:
        raw_time = record.get("time_s")
        mask_time_s = (
            None
            if raw_time is None
            else _number(raw_time, f"action_records[{index}].time_s", minimum=0.0)
        )
        grouped_masks.setdefault((task_id, stage), []).append(
            (index, mask_time_s, record, rows)
        )

    executions, ignored_policy_count, ignored_automatic_repairs = (
        _selected_execution_records(action_records)
    )
    pairings: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    validated_count = 0
    for execution in executions:
        key = (execution["task_id"], execution["stage"])
        candidates = grouped_masks.get(key, [])
        if not candidates:
            raise AuditValidationError(
                "AUDIT_EXECUTION_MASK_MISSING",
                "executed action has no same-task, same-stage hard mask",
                task_id=execution["task_id"],
                stage=execution["stage"],
                action_id=execution["action_id"],
            )
        execution_time_s = execution["time_s"]
        execution_check_id = execution["execution_check_id"]
        if execution_check_id is not None:
            bound = [
                mask
                for mask in candidates
                if mask[2].get("execution_check_id") == execution_check_id
            ]
            if len(bound) != 1:
                raise AuditValidationError(
                    "AUDIT_EXECUTION_BINDING_INVALID",
                    "execution check ID does not bind exactly one hard mask",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                    execution_check_id=execution_check_id,
                    bound_mask_count=len(bound),
                )
            matched = bound[0]
            if matched[2].get("mask_epoch") != "EXECUTION_RECHECK":
                raise AuditValidationError(
                    "AUDIT_EXECUTION_BINDING_INVALID",
                    "execution check ID is not bound to an execution-time mask",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                    execution_check_id=execution_check_id,
                    mask_epoch=matched[2].get("mask_epoch"),
                )
            if execution_time_s is None or matched[1] is None:
                raise AuditValidationError(
                    "AUDIT_EXECUTION_TIME_MISSING",
                    "bound execution and execution-time mask require timestamps",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                    execution_check_id=execution_check_id,
                )
            if matched[1] != execution_time_s:
                raise AuditValidationError(
                    "AUDIT_EXECUTION_TIME_MISMATCH",
                    "bound execution-time mask and repair have different timestamps",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                    execution_check_id=execution_check_id,
                    mask_time_s=matched[1],
                    execution_time_s=execution_time_s,
                )
        else:
            # Legacy POLICY_DECISION fallback is allowed only when its
            # decision-epoch mask can still be identified uniquely.  Repair
            # records never take this path: they require an exact execution
            # check binding above.
            decision_candidates = [
                mask
                for mask in candidates
                if mask[2].get("mask_epoch") != "EXECUTION_RECHECK"
            ]
            if execution_time_s is None:
                if len(decision_candidates) != 1:
                    raise AuditValidationError(
                        "AUDIT_EXECUTION_TIME_MISSING",
                        "untimed policy execution cannot be paired uniquely",
                        task_id=execution["task_id"],
                        stage=execution["stage"],
                        mask_count=len(decision_candidates),
                    )
                matched = decision_candidates[0]
            elif any(mask[1] is None for mask in decision_candidates):
                raise AuditValidationError(
                    "AUDIT_MASK_TIME_MISSING",
                    "an untimed decision mask prevents causal policy pairing",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                )
            else:
                prior = [
                    mask
                    for mask in decision_candidates
                    if mask[1] <= execution_time_s
                ]
                if not prior:
                    raise AuditValidationError(
                        "AUDIT_CAUSAL_MASK_MISSING",
                        "no decision mask precedes the policy execution",
                        task_id=execution["task_id"],
                        stage=execution["stage"],
                        execution_time_s=execution_time_s,
                    )
                latest_time = max(float(mask[1]) for mask in prior)
                nearest = [mask for mask in prior if mask[1] == latest_time]
                if len(nearest) != 1:
                    raise AuditValidationError(
                        "AUDIT_MASK_PAIR_AMBIGUOUS",
                        "multiple masks are equally recent before policy execution",
                        task_id=execution["task_id"],
                        stage=execution["stage"],
                        execution_time_s=execution_time_s,
                        mask_time_s=latest_time,
                    )
                matched = nearest[0]

        mask_index, mask_time_s, _, raw_rows = matched
        rows = [
            _mapping(row, f"action_records[{mask_index}].rows[{row_index}]")
            for row_index, row in enumerate(raw_rows)
        ]
        matching_rows = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if row.get("action_id") == execution["action_id"]
        ]
        pairing = {
            "task_id": execution["task_id"],
            "stage": execution["stage"],
            "action_id": execution["action_id"],
            "execution_record_kind": execution["record_kind"],
            "execution_record_index": execution["record_index"],
            "execution_time_s": execution_time_s,
            "execution_check_id": execution_check_id,
            "mask_record_index": mask_index,
            "mask_time_s": mask_time_s,
        }
        if not matching_rows:
            violation = {
                **pairing,
                "violation_code": "EXECUTED_ACTION_ABSENT_FROM_MASK",
                "reason_codes": [],
            }
            violations.append(violation)
            pairings.append({**pairing, "validation_status": "VIOLATION"})
            continue
        if len(matching_rows) != 1:
            raise AuditValidationError(
                "AUDIT_MASK_ACTION_AMBIGUOUS",
                "executed action appears more than once in its paired mask",
                task_id=execution["task_id"],
                stage=execution["stage"],
                action_id=execution["action_id"],
                mask_record_index=mask_index,
            )
        row_index, row = matching_rows[0]
        logged_mask_id, logged_mask_stage = _logged_action_id(
            row.get("action"),
            f"action_records[{mask_index}].rows[{row_index}].action",
        )
        if logged_mask_id != execution["action_id"] or logged_mask_stage != execution["stage"]:
            raise AuditValidationError(
                "AUDIT_MASK_ACTION_ID_MISMATCH",
                "hard-mask action_id disagrees with its structured action",
                task_id=execution["task_id"],
                stage=execution["stage"],
                action_id=execution["action_id"],
                structured_action_id=logged_mask_id,
                mask_record_index=mask_index,
            )
        allowed = row.get("allowed")
        if not isinstance(allowed, bool):
            raise AuditValidationError(
                "AUDIT_MASK_ALLOWED_INVALID",
                "paired hard-mask row has no Boolean allowed decision",
                task_id=execution["task_id"],
                stage=execution["stage"],
                action_id=execution["action_id"],
                mask_record_index=mask_index,
            )
        if not allowed:
            reason_codes = sorted(
                _text(value, "reason_code")
                for value in _sequence(
                    row.get("reason_codes"),
                    f"action_records[{mask_index}].rows[{row_index}].reason_codes",
                )
            )
            if not reason_codes:
                raise AuditValidationError(
                    "AUDIT_REJECTION_REASON_MISSING",
                    "executed action's rejected hard-mask row has no reason",
                    task_id=execution["task_id"],
                    stage=execution["stage"],
                    action_id=execution["action_id"],
                    mask_record_index=mask_index,
                )
            violation = {
                **pairing,
                "violation_code": "EXECUTED_ACTION_REJECTED_BY_MASK",
                "reason_codes": reason_codes,
            }
            violations.append(violation)
            pairings.append({**pairing, "validation_status": "VIOLATION"})
            continue
        validated_count += 1
        pairings.append({**pairing, "validation_status": "VALIDATED_ALLOWED_MEMBER"})

    violation_count = len(violations)
    if not executions:
        status = "NOT_OBSERVED"
        hard_mask_bypassed: bool | None = None
    elif violation_count:
        status = "VIOLATION"
        hard_mask_bypassed = True
    else:
        status = "VALIDATED"
        hard_mask_bypassed = False
    return {
        "execution_validation_status": status,
        "execution_validation_scope": (
            "RAW_READY_POLICY_DECISIONS_AND_EXECUTION_TIME_REPAIRS_ONLY"
        ),
        "automatic_failure_successor_rule": (
            "FROZEN_LOCAL_FALLBACK_IS_SEPARATELY_GUARDED_BY_"
            "LOCAL_FALLBACK_FEASIBILITY_AND_RUNTIME_INVARIANTS"
        ),
        "execution_source_rule": "EXECUTION_REPAIR_PER_TASK_STAGE_ELSE_POLICY_DECISION",
        "pairing_rule": (
            "EXACT_EXECUTION_CHECK_ID_FOR_REPAIR;"
            "LATEST_CAUSAL_DECISION_MASK_ONLY_FOR_POLICY_FALLBACK"
        ),
        "executed_action_count": len(executions),
        "validated_count": validated_count,
        "violation_count": violation_count,
        "hard_mask_bypassed": hard_mask_bypassed,
        "ignored_policy_decision_count": ignored_policy_count,
        "ignored_automatic_fallback_repair_count": ignored_automatic_repairs,
        "execution_pairings": pairings,
        "violations": violations,
    }


def audit_hard_mask_counterfactual(
    action_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate logged executions and count cheaper rejected counterfactuals."""

    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    rejected_count = 0
    seemingly_better_count = 0
    unavailable_count = 0
    masks = _mask_records(action_records)
    decision_masks = [
        mask for mask in masks if mask[3].get("mask_epoch") != "EXECUTION_RECHECK"
    ]
    for index, task_id, stage, _, raw_rows in decision_masks:
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
    execution_validation = _validate_executed_mask_membership(action_records, masks)
    return _finish(
        {
            "schema_version": "1.1",
            "analysis": "hard_mask_safety_counterfactual",
            "counterfactual_only": False,
            "rejected_action_analysis_counterfactual_only": True,
            "unsafe_actions_executed_by_audit": 0,
            "units": {"expected_cost": "normalized_cost"},
            "mask_count": len(decisions),
            "execution_recheck_mask_count": len(masks) - len(decision_masks),
            "rejected_action_count": rejected_count,
            "rejected_with_cost_unavailable": unavailable_count,
            "rejected_seemingly_better_count": seemingly_better_count,
            "reason_counts": dict(sorted(reasons.items())),
            "decisions": decisions,
            **execution_validation,
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


def _optional_task_array(
    task: Mapping[str, Any], task_id: str, field: str
) -> Sequence[Any]:
    """Read an optional parsed array or its CSV JSON representation."""

    raw = task.get(field)
    if raw is None:
        encoded = task.get(f"{field}_json")
        if encoded in (None, ""):
            return ()
        if not isinstance(encoded, str):
            raise AuditValidationError(
                "FAILURE_OPTIONAL_ARRAY",
                f"{field}_json must be text",
                task_id=task_id,
            )
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise AuditValidationError(
                "FAILURE_OPTIONAL_ARRAY_JSON",
                f"{field}_json is invalid",
                task_id=task_id,
            ) from exc
    return _sequence(raw, f"task[{task_id}].{field}")


def _task_network_observations(
    task: Mapping[str, Any], task_id: str
) -> dict[str, dict[str, int]]:
    """Validate observed UL/DL transaction pairing and count terminal outcomes."""

    audit = _sequence(task.get("network_audit"), f"task[{task_id}].network_audit")
    open_attempt = {"UL": False, "DL": False}
    counts = {
        "UL": {"started": 0, "done": 0, "failed": 0},
        "DL": {"started": 0, "done": 0, "failed": 0},
    }
    for index, raw in enumerate(audit):
        row = _mapping(raw, f"task[{task_id}].network_audit[{index}]")
        direction = row.get("direction")
        if direction not in counts:
            continue
        status = _text(row.get("status"), "network_audit.status")
        _number(row.get("time_s"), "network_audit.time_s", minimum=0.0)
        if status == "START":
            if open_attempt[direction]:
                raise AuditValidationError(
                    "FAILURE_NETWORK_OVERLAP",
                    "overlapping network attempts are ambiguous",
                    task_id=task_id,
                    direction=direction,
                )
            open_attempt[direction] = True
            counts[direction]["started"] += 1
        elif status in {"PAUSED", "RESUMED"}:
            if not open_attempt[direction]:
                raise AuditValidationError(
                    "FAILURE_NETWORK_STATE",
                    "pause/resume record has no active network attempt",
                    task_id=task_id,
                    direction=direction,
                    status=status,
                )
        elif status in {"DONE", "FAIL"}:
            if not open_attempt[direction]:
                raise AuditValidationError(
                    "FAILURE_NETWORK_PAIR",
                    "network terminal record has no active start",
                    task_id=task_id,
                    direction=direction,
                )
            for field in ("vehicle_energy_j", "rsu_energy_j"):
                if field not in row:
                    raise AuditValidationError(
                        "FAILURE_NETWORK_FIELDS_MISSING",
                        "network terminal record lacks paired energy",
                        task_id=task_id,
                        direction=direction,
                        field=field,
                    )
                _number(row[field], f"network.{field}", minimum=0.0)
            counts[direction]["done" if status == "DONE" else "failed"] += 1
            open_attempt[direction] = False
        else:
            raise AuditValidationError(
                "FAILURE_NETWORK_STATUS",
                "network audit contains an unknown status",
                task_id=task_id,
                direction=direction,
                status=status,
            )
    incomplete = [direction for direction, active in open_attempt.items() if active]
    if incomplete:
        raise AuditValidationError(
            "FAILURE_NETWORK_INCOMPLETE",
            "network start has no terminal accounting record",
            task_id=task_id,
            directions=incomplete,
        )
    return counts


def _task_rsu_observations(
    task: Mapping[str, Any], task_id: str
) -> dict[str, Any]:
    """Validate the observed admission/ingress sequence without claiming atomicity."""

    counts = {
        "admission_accept": 0,
        "admission_reject": 0,
        "rsu_ingress": 0,
        "rsu_failure": 0,
    }
    admission_reject_reasons: Counter[str] = Counter()
    ingress_open = False
    for index, raw in enumerate(_optional_task_array(task, task_id, "rsu_audit")):
        row = _mapping(raw, f"task[{task_id}].rsu_audit[{index}]")
        if "admission" in row:
            admission = _text(row.get("admission"), "rsu_audit.admission")
            if admission == "ACCEPT":
                counts["admission_accept"] += 1
            elif admission == "REJECT":
                counts["admission_reject"] += 1
                if "reason_codes" not in row:
                    raise AuditValidationError(
                        "FAILURE_ADMISSION_REASON_MISSING",
                        "rejected RSU admission has no reason code",
                        task_id=task_id,
                        audit_index=index,
                    )
                reasons = [
                    _text(value, "rsu_audit.reason_code")
                    for value in _sequence(
                        row.get("reason_codes"), "rsu_audit.reason_codes"
                    )
                ]
                if not reasons:
                    raise AuditValidationError(
                        "FAILURE_ADMISSION_REASON_MISSING",
                        "rejected RSU admission has no reason code",
                        task_id=task_id,
                        audit_index=index,
                    )
                if len(reasons) != len(set(reasons)):
                    raise AuditValidationError(
                        "FAILURE_ADMISSION_REASON_DUPLICATE",
                        "rejected RSU admission repeats a reason code",
                        task_id=task_id,
                        audit_index=index,
                        reason_codes=reasons,
                    )
                admission_reject_reasons.update(reasons)
            else:
                raise AuditValidationError(
                    "FAILURE_ADMISSION_STATUS",
                    "RSU audit contains an unknown admission outcome",
                    task_id=task_id,
                    admission=admission,
                )
            continue
        phase = row.get("phase")
        if phase is None:
            continue
        phase_text = _text(phase, "rsu_audit.phase")
        if phase_text == "ingress_start":
            if ingress_open:
                raise AuditValidationError(
                    "FAILURE_RSU_INGRESS_OVERLAP",
                    "overlapping RSU ingress attempts are ambiguous",
                    task_id=task_id,
                )
            ingress_open = True
        elif phase_text == "ingress_done":
            if not ingress_open:
                raise AuditValidationError(
                    "FAILURE_RSU_INGRESS_PAIR",
                    "RSU ingress terminal record has no start",
                    task_id=task_id,
                )
            counts["rsu_ingress"] += 1
            if row.get("valid") is False:
                counts["rsu_failure"] += 1
            ingress_open = False
        elif "fail" in phase_text.lower():
            counts["rsu_failure"] += 1
            ingress_open = False
    if ingress_open:
        raise AuditValidationError(
            "FAILURE_RSU_INGRESS_INCOMPLETE",
            "RSU ingress start has no terminal accounting record",
            task_id=task_id,
        )
    failure_reason = task.get("failure_reason")
    if (
        counts["rsu_failure"] == 0
        and isinstance(failure_reason, str)
        and (
            "RSU" in failure_reason.upper()
            or "EDGE" in failure_reason.upper()
        )
    ):
        # The task-level reason is a conservative fallback for older rows that
        # lack an explicit RSU failure marker.  It must not count the same
        # realized failure a second time when rsu_audit already recorded it.
        counts["rsu_failure"] += 1
    counts["admission_reject_reason_counts"] = dict(
        sorted(admission_reject_reasons.items())
    )
    return counts


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
    admission_reject_reasons: Counter[str] = Counter()
    fallback_task_ids: list[str] = []
    coverage_task_ids = {
        category: set() for category in _FAILURE_COVERAGE_CATEGORIES
    }
    coverage_observations: Counter[str] = Counter()
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
        network = _task_network_observations(task, task_id)
        rsu = _task_rsu_observations(task, task_id)
        admission_reject_reasons.update(rsu["admission_reject_reason_counts"])
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
        if int(task["attempt_started_count"]) > 1:
            coverage_task_ids["retry"].add(task_id)
            coverage_observations["retry"] += int(task["attempt_started_count"]) - 1
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
            coverage_task_ids["anonymization_failure"].add(task_id)
            coverage_observations["anonymization_failure"] += len(failed_attempts)
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
            coverage_task_ids["local_fallback"].add(task_id)
            coverage_observations["local_fallback"] += 1
        for direction, category, failure_category in (
            ("UL", "uplink", "uplink_failure"),
            ("DL", "downlink", "downlink_failure"),
        ):
            if network[direction]["started"]:
                coverage_task_ids[category].add(task_id)
                coverage_observations[category] += network[direction]["started"]
            if network[direction]["failed"]:
                coverage_task_ids[failure_category].add(task_id)
                coverage_observations[failure_category] += network[direction]["failed"]
        for category in (
            "admission_accept",
            "admission_reject",
            "rsu_ingress",
            "rsu_failure",
        ):
            if rsu[category]:
                coverage_task_ids[category].add(task_id)
                coverage_observations[category] += rsu[category]
        if any(stage.startswith("EDGE:") for stage in actual_path):
            coverage_task_ids["edge_execution"].add(task_id)
            coverage_observations["edge_execution"] += 1
        if rsu_total > 0.0:
            coverage_task_ids["rsu_attributed_energy"].add(task_id)
            coverage_observations["rsu_attributed_energy"] += 1
        if failure > 0.0:
            coverage_task_ids["failure_penalty"].add(task_id)
            coverage_observations["failure_penalty"] += 1
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
    coverage_task_ids["explicit_fail_action"].update(explicit_fail_task_ids)
    coverage_observations["explicit_fail_action"] += explicit_fail_count
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
    coverage_categories = {}
    for category in _FAILURE_COVERAGE_CATEGORIES:
        observed = coverage_observations[category]
        coverage_categories[category] = {
            "observation_count": observed,
            "task_count": len(coverage_task_ids[category]),
            "task_ids": sorted(coverage_task_ids[category]),
            "status": _coverage_status(category, observed),
            "validation_scope": _FAILURE_COVERAGE_SCOPES[category],
        }
    coverage_categories["admission_reject"]["reason_counts"] = dict(
        sorted(admission_reject_reasons.items())
    )
    observed_categories = [
        category
        for category in _FAILURE_COVERAGE_CATEGORIES
        if coverage_observations[category]
    ]
    not_observed_categories = [
        category
        for category in _FAILURE_COVERAGE_CATEGORIES
        if not coverage_observations[category]
    ]
    coverage_complete = not not_observed_categories
    coverage_status = (
        "COMPLETE_OBSERVED_CATEGORY_COVERAGE"
        if coverage_complete
        else "PARTIAL_OBSERVED_CATEGORY_COVERAGE"
    )
    return _finish(
        {
            "schema_version": "1.0",
            "coverage_schema_version": "1.1",
            "analysis": "failure_cost_completeness",
            # Preserve the pre-coverage API contract for existing consumers.
            # Coverage completeness is deliberately reported separately: the
            # accounting audit can be complete for every observed path without
            # claiming that every registered failure category occurred.
            "status": "COMPLETE",
            "coverage_status": coverage_status,
            "accounting_validation_status": "COMPLETE_FOR_OBSERVED_PATHS",
            "coverage_scope": {
                "claim": "empirical coverage of categories observed in supplied runs",
                "does_not_claim_unobserved_categories": True,
                "does_not_prove_admission_atomicity": True,
                "registered_categories": list(_FAILURE_COVERAGE_CATEGORIES),
                "observed_categories": observed_categories,
                "not_observed_categories": not_observed_categories,
                "all_registered_categories_observed": coverage_complete,
            },
            "observed_coverage": coverage_categories,
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


def audit_failure_cost_coverage(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate strict failure-cost coverage across one or more independent runs.

    Each run must contain ``task_rows``, ``action_records`` and ``event_records``.
    A run may also supply a stable ``run_id``.  The aggregate never upgrades an
    unobserved category to validated merely because another cost term was present.
    """

    raw_runs = _sequence(runs, "runs")
    if not raw_runs:
        raise AuditValidationError("FAILURE_NO_RUNS", "run audit is empty")
    seen_ids: set[str] = set()
    run_summaries: list[dict[str, Any]] = []
    observed_run_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    admission_reject_reasons: Counter[str] = Counter()
    aggregate_costs = {
        "retry_latency_s": 0.0,
        "retry_vehicle_energy_j": 0.0,
        "downlink_latency_s": 0.0,
        "downlink_vehicle_energy_j": 0.0,
        "downlink_rsu_energy_j": 0.0,
        "rsu_attributed_energy_j": 0.0,
        "failure_penalty_normalized_cost": 0.0,
        "failed_anonymization_executed_work_s": 0.0,
        "failed_anonymization_vehicle_energy_j": 0.0,
    }
    total_tasks = 0
    for index, raw in enumerate(raw_runs):
        run = _mapping(raw, f"runs[{index}]")
        raw_id = run.get("run_id", f"run-{index:05d}")
        run_id = _text(raw_id, f"runs[{index}].run_id")
        if run_id in seen_ids:
            raise AuditValidationError(
                "FAILURE_DUPLICATE_RUN", "run IDs must be unique", run_id=run_id
            )
        seen_ids.add(run_id)
        report = audit_failure_cost_completeness(
            _sequence(run.get("task_rows"), f"runs[{index}].task_rows"),
            _sequence(run.get("action_records"), f"runs[{index}].action_records"),
            _sequence(run.get("event_records"), f"runs[{index}].event_records"),
        )
        total_tasks += int(report["task_count"])
        observed = report["coverage_scope"]["observed_categories"]
        for category in observed:
            observed_run_counts[category] += 1
            entry = report["observed_coverage"][category]
            observation_counts[category] += int(entry["observation_count"])
            task_counts[category] += int(entry["task_count"])
        admission_reject_reasons.update(
            report["observed_coverage"]["admission_reject"]["reason_counts"]
        )
        omissions = report["omissions"]
        aggregate_costs["retry_latency_s"] += omissions["retry"]["latency_s"]
        aggregate_costs["retry_vehicle_energy_j"] += omissions["retry"][
            "vehicle_energy_j"
        ]
        aggregate_costs["downlink_latency_s"] += omissions["downlink"]["latency_s"]
        aggregate_costs["downlink_vehicle_energy_j"] += omissions["downlink"][
            "vehicle_energy_j"
        ]
        aggregate_costs["downlink_rsu_energy_j"] += omissions["downlink"][
            "rsu_energy_j"
        ]
        aggregate_costs["rsu_attributed_energy_j"] += omissions["rsu_energy"][
            "rsu_energy_j"
        ]
        aggregate_costs["failure_penalty_normalized_cost"] += omissions["failure"][
            "normalized_cost"
        ]
        aggregate_costs[
            "failed_anonymization_executed_work_s"
        ] += omissions["anonymization_failure"]["executed_work_s"]
        aggregate_costs[
            "failed_anonymization_vehicle_energy_j"
        ] += omissions["anonymization_failure"]["vehicle_energy_j"]
        run_summaries.append(
            {
                "run_id": run_id,
                "status": report["status"],
                "task_count": report["task_count"],
                "observed_categories": list(observed),
                "not_observed_categories": list(
                    report["coverage_scope"]["not_observed_categories"]
                ),
                "report_sha256": report["report_sha256"],
            }
        )
    aggregate_coverage = {}
    for category in _FAILURE_COVERAGE_CATEGORIES:
        count = observation_counts[category]
        aggregate_coverage[category] = {
            "observed_run_count": observed_run_counts[category],
            "observation_count": count,
            "task_count": task_counts[category],
            "status": _coverage_status(category, count),
            "validation_scope": _FAILURE_COVERAGE_SCOPES[category],
        }
    aggregate_coverage["admission_reject"]["reason_counts"] = dict(
        sorted(admission_reject_reasons.items())
    )
    not_observed = [
        category
        for category in _FAILURE_COVERAGE_CATEGORIES
        if not observation_counts[category]
    ]
    coverage_status = (
        "COMPLETE_OBSERVED_CATEGORY_COVERAGE"
        if not not_observed
        else "PARTIAL_OBSERVED_CATEGORY_COVERAGE"
    )
    return _finish(
        {
            "schema_version": "1.0",
            "coverage_schema_version": "1.1",
            "analysis": "failure_cost_coverage_aggregate",
            "status": coverage_status,
            "coverage_status": coverage_status,
            "accounting_validation_status": "COMPLETE_FOR_OBSERVED_PATHS",
            "run_count": len(raw_runs),
            "task_count": total_tasks,
            "coverage_scope": {
                "claim": "aggregate empirical coverage of supplied runs",
                "does_not_claim_unobserved_categories": True,
                "does_not_prove_admission_atomicity": True,
                "registered_categories": list(_FAILURE_COVERAGE_CATEGORIES),
                "not_observed_categories": not_observed,
                "all_registered_categories_observed": not not_observed,
            },
            "observed_coverage": aggregate_coverage,
            "aggregate_costs": aggregate_costs,
            "runs": run_summaries,
        },
        raw_runs,
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
    "audit_failure_cost_coverage",
    "audit_hard_mask_counterfactual",
    "build_preregistered_one_shot_commitments",
    "evaluate_two_stage_information_ablation",
    "exact_adaptive_scenario_tree_oracle",
    "exact_finite_scenario_oracle",
]
