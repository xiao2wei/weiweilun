from __future__ import annotations

import copy

import pytest

from privacy_edge_sim.paper_experiment_audits import (
    AuditValidationError,
    audit_failure_cost_completeness,
    audit_hard_mask_counterfactual,
    build_preregistered_one_shot_commitments,
    evaluate_two_stage_information_ablation,
    exact_adaptive_scenario_tree_oracle,
    exact_finite_scenario_oracle,
)
from privacy_edge_sim.profiles import canonical_document_sha256


def _mask_row(
    action_id,
    kind,
    allowed,
    cost,
    *,
    reasons=(),
    information_ablation=None,
):
    details = {}
    if cost is not None:
        details["bounds"] = {"expected_cost": cost}
    if information_ablation is not None:
        details["information_ablation"] = information_ablation
    return {
        "action_id": action_id,
        "action": {"stage": "READY", "kind": kind},
        "allowed": allowed,
        "reason_codes": list(reasons),
        "details": details,
    }


def test_hard_mask_counterfactual_counts_cheaper_rejections_without_execution():
    actions = [
        {
            "record_kind": "HARD_MASK",
            "task_id": "task-1",
            "stage": "RAW",
            "rows": [
                _mask_row("local", "LOCAL", True, 1.0),
                _mask_row(
                    "unsafe-cheap", "PIPE", False, 0.4, reasons=("PRIVACY_RISK",)
                ),
                _mask_row("unsupported", "PIPE", False, None, reasons=("UNSUPPORTED",)),
            ],
        }
    ]
    original = copy.deepcopy(actions)
    report = audit_hard_mask_counterfactual(actions)
    assert actions == original
    assert report["rejected_action_count"] == 2
    assert report["rejected_seemingly_better_count"] == 1
    assert report["rejected_with_cost_unavailable"] == 1
    assert report["reason_counts"] == {"PRIVACY_RISK": 1, "UNSUPPORTED": 1}
    assert report["unsafe_actions_executed_by_audit"] == 0
    assert report["hard_mask_bypassed"] is False
    assert report["report_sha256"] == canonical_document_sha256(report, "report_sha256")


def test_hard_mask_rejected_action_requires_reason_code():
    actions = [
        {
            "record_kind": "HARD_MASK",
            "task_id": "task-1",
            "stage": "RAW",
            "rows": [
                _mask_row("local", "LOCAL", True, 1.0),
                _mask_row("bad", "PIPE", False, 0.5),
            ],
        }
    ]
    with pytest.raises(AuditValidationError) as captured:
        audit_hard_mask_counterfactual(actions)
    assert captured.value.as_dict()["error_code"] == ("AUDIT_REJECTION_REASON_MISSING")


def test_two_stage_ablation_uses_only_safe_actions_and_conservative_replacements():
    ablation = {
        "observed_output_size_cost": 0.10,
        "conservative_output_size_cost": 0.40,
        "observed_fresh_queue_cost": 0.05,
        "conservative_stale_queue_cost": 0.30,
    }
    actions = [
        {
            "record_kind": "HARD_MASK",
            "task_id": "task-1",
            "stage": "READY",
            "rows": [
                _mask_row("local", "LOCAL", True, 1.0),
                _mask_row("edge", "EDGE", True, 0.8, information_ablation=ablation),
                _mask_row("unsafe", "EDGE", False, 0.1, reasons=("PRIVACY_RISK",)),
            ],
        }
    ]
    report = evaluate_two_stage_information_ablation(
        actions, one_shot_commitments={"task-1": "local"}
    )
    pair = report["pairs"][0]
    assert pair["ready_recourse"] == {"action_id": "edge", "expected_cost": 0.8}
    assert pair["one_shot"]["paired_excess_cost"] == pytest.approx(0.2)
    assert pair["without_output_size"]["action_id"] == "local"
    assert pair["without_fresh_queue"]["action_id"] == "local"
    assert report["only_allowed_actions_compared"] is True
    assert "unsafe" not in str(report["pairs"])


def test_two_stage_ablation_refuses_unsafe_one_shot_commitment():
    actions = [
        {
            "record_kind": "HARD_MASK",
            "task_id": "task-1",
            "stage": "READY",
            "rows": [
                _mask_row("local", "LOCAL", True, 1.0),
                _mask_row("unsafe", "EDGE", False, 0.1, reasons=("OOD",)),
            ],
        }
    ]
    with pytest.raises(AuditValidationError) as captured:
        evaluate_two_stage_information_ablation(
            actions, one_shot_commitments={"task-1": "unsafe"}
        )
    assert captured.value.code == "ABLATION_UNSAFE_COMMITMENT"


def test_one_shot_commitments_use_only_raw_visible_identity_and_frozen_plan():
    raw = {
        "record_kind": "HARD_MASK",
        "task_id": "task-1",
        "vehicle_id": "veh-1",
        "stage": "RAW",
        "rows": [_mask_row("local", "LOCAL", True, 1.0)],
    }
    ready = {
        "record_kind": "HARD_MASK",
        "task_id": "task-1",
        "vehicle_id": "veh-1",
        "stage": "READY",
        "rows": [_mask_row("edge", "EDGE", True, 0.1)],
    }
    plan = {
        "default_action_priority": [
            "READY|EDGE|||rsu-1|edge",
            "READY|LOCAL|local|||",
        ]
    }
    first = build_preregistered_one_shot_commitments([raw, ready], registered_plan=plan)
    changed = copy.deepcopy(ready)
    changed["rows"][0]["details"]["bounds"]["expected_cost"] = 999.0
    second = build_preregistered_one_shot_commitments(
        [raw, changed], registered_plan=plan
    )
    assert first == second
    assert first["uses_ready_records"] is False
    assert first["uses_ready_expected_cost"] is False
    assert (
        first["commitments"]["task-1"]["action_priority"]
        == (plan["default_action_priority"])
    )


def _complete_task():
    return {
        "task_id": "task-1",
        "attempt_started_count": 2,
        "anon_attempts": [
            {
                "attempt": 1,
                "failure_reason": "GUARD_REJECTED",
                "executed_work_s": 0.08,
                "vehicle_energy_j": 0.5,
            },
            {"attempt": 2, "latency_s": 0.12, "vehicle_energy_j": 0.8},
        ],
        "network_audit": [
            {"direction": "DL", "status": "START", "time_s": 1.0},
            {
                "direction": "DL",
                "status": "DONE",
                "time_s": 1.3,
                "vehicle_energy_j": 0.2,
                "rsu_energy_j": 0.4,
            },
        ],
        "rsu_attributed_energy_j": 3.5,
        "failure_penalty_cost": 2.0,
        "actual_path": ["ANON#1", "ANON#2", "LOCAL_FER"],
    }


def test_failure_completeness_quantifies_retry_downlink_rsu_and_failure_terms():
    report = audit_failure_cost_completeness(
        [_complete_task()],
        [
            {
                "record_kind": "POLICY_DECISION",
                "task_id": "task-1",
                "executed": {"kind": "FAIL", "stage": "READY"},
            }
        ],
        [{"record_kind": "EVENT_BATCH"}],
    )
    assert report["status"] == "COMPLETE"
    assert report["omissions"]["retry"] == {
        "latency_s": pytest.approx(0.12),
        "vehicle_energy_j": pytest.approx(0.8),
        "tasks_affected": 1,
    }
    assert report["omissions"]["downlink"]["latency_s"] == pytest.approx(0.3)
    assert report["omissions"]["downlink"]["vehicle_energy_j"] == 0.2
    assert report["omissions"]["downlink"]["rsu_energy_j"] == 0.4
    assert report["omissions"]["rsu_energy"]["rsu_energy_j"] == 3.5
    assert report["omissions"]["failure"]["normalized_cost"] == 2.0
    assert report["omissions"]["anonymization_failure"] == {
        "failed_attempt_count": 1,
        "executed_work_s": pytest.approx(0.08),
        "vehicle_energy_j": pytest.approx(0.5),
        "tasks_affected": 1,
        "reason_counts": {"GUARD_REJECTED": 1},
        "accounting": "measured_failed_attempt_components",
    }
    assert report["omissions"]["local_fallback"]["task_ids"] == ["task-1"]
    assert report["omissions"]["explicit_fail_action"]["decision_count"] == 1


def test_failure_completeness_structurally_refuses_missing_fields():
    task = _complete_task()
    del task["anon_attempts"][1]["vehicle_energy_j"]
    with pytest.raises(AuditValidationError) as captured:
        audit_failure_cost_completeness([task], [], [])
    refusal = captured.value.as_dict()
    assert refusal["status"] == "REFUSED"
    assert refusal["error_code"] == "FAILURE_RETRY_FIELDS_MISSING"


def _scenarios():
    return [
        {
            "scenario_id": "s1",
            "probability": 0.5,
            "stages": [
                {
                    "a": {"hard_safe": True, "cost": 1.0, "duration_s": 1.0},
                    "b": {"hard_safe": True, "cost": 2.0, "duration_s": 1.0},
                },
                {
                    "a": {"hard_safe": True, "cost": 3.0, "duration_s": 1.0},
                    "b": {"hard_safe": True, "cost": 1.0, "duration_s": 1.0},
                },
            ],
        },
        {
            "scenario_id": "s2",
            "probability": 0.5,
            "stages": [
                {
                    "a": {"hard_safe": True, "cost": 2.0, "duration_s": 1.0},
                    "b": {"hard_safe": True, "cost": 1.0, "duration_s": 1.0},
                },
                {
                    "a": {"hard_safe": True, "cost": 2.0, "duration_s": 1.0},
                    "b": {"hard_safe": True, "cost": 2.0, "duration_s": 1.0},
                },
            ],
        },
    ]


def test_exact_oracle_enumerates_stably_and_reports_esl_gap_without_wallclock():
    report = exact_finite_scenario_oracle(
        _scenarios(), esl_action_sequence=("a", "a"), max_sequences=10
    )
    assert report["optimum"]["action_sequence"] == ["a", "b"]
    assert report["optimum"]["exact_ratio"] == pytest.approx(1.5)
    assert report["esl"]["ratio"] == pytest.approx(2.0)
    assert report["esl"]["absolute_gap"] == pytest.approx(0.5)
    assert report["complexity"] == {
        "scenario_count": 2,
        "horizon": 2,
        "action_count": 2,
        "candidate_sequences": 4,
        "evaluated_sequences": 4,
        "hard_safe_sequences": 4,
        "max_sequences": 10,
        "wallclock_used_in_simulation": False,
        "wallclock_reported": False,
    }


def test_exact_oracle_respects_hard_mask_and_complexity_cap():
    scenarios = _scenarios()
    scenarios[0]["stages"][0]["b"]["hard_safe"] = False
    report = exact_finite_scenario_oracle(
        scenarios, esl_action_sequence=("a", "a"), max_sequences=4
    )
    assert report["complexity"]["hard_safe_sequences"] == 2
    assert all(action == "a" for action in report["optimum"]["action_sequence"][:1])
    with pytest.raises(AuditValidationError) as captured:
        exact_finite_scenario_oracle(
            _scenarios(), esl_action_sequence=("a", "a"), max_sequences=3
        )
    assert captured.value.code == "ORACLE_COMPLEXITY_LIMIT"


def _adaptive_tree():
    return {
        "root_id": "root",
        "nodes": [
            {
                "node_id": "root",
                "actions": {
                    "start": {
                        "hard_safe": True,
                        "cost": 0.0,
                        "duration_s": 1.0,
                        "branches": [
                            {"probability": 0.5, "next_node_id": "good"},
                            {"probability": 0.5, "next_node_id": "bad"},
                        ],
                    }
                },
            },
            {
                "node_id": "good",
                "actions": {
                    "x": {
                        "hard_safe": True,
                        "cost": 1.0,
                        "duration_s": 1.0,
                        "branches": [],
                    },
                    "y": {
                        "hard_safe": True,
                        "cost": 3.0,
                        "duration_s": 1.0,
                        "branches": [],
                    },
                },
            },
            {
                "node_id": "bad",
                "actions": {
                    "x": {
                        "hard_safe": True,
                        "cost": 4.0,
                        "duration_s": 1.0,
                        "branches": [],
                    },
                    "y": {
                        "hard_safe": True,
                        "cost": 1.0,
                        "duration_s": 1.0,
                        "branches": [],
                    },
                },
            },
        ],
    }


def test_exact_adaptive_tree_oracle_finds_contingent_policy_and_esl_gap():
    report = exact_adaptive_scenario_tree_oracle(
        _adaptive_tree(),
        esl_contingent_policy={"root": "start", "good": "x", "bad": "x"},
        max_policies=4,
    )
    assert report["optimum"]["contingent_policy"] == {
        "bad": "y",
        "good": "x",
        "root": "start",
    }
    assert report["optimum"]["exact_ratio"] == pytest.approx(0.5)
    assert report["esl"]["ratio"] == pytest.approx(1.25)
    assert report["esl"]["absolute_gap"] == pytest.approx(0.75)
    assert report["complexity"]["candidate_policies"] == 4
    assert report["complexity"]["wallclock_used_in_simulation"] is False


def test_exact_adaptive_tree_oracle_rejects_cycles_and_complexity_excess():
    with pytest.raises(AuditValidationError) as limited:
        exact_adaptive_scenario_tree_oracle(
            _adaptive_tree(),
            esl_contingent_policy={"root": "start", "good": "x", "bad": "x"},
            max_policies=3,
        )
    assert limited.value.code == "TREE_COMPLEXITY_LIMIT"
    cyclic = _adaptive_tree()
    cyclic["nodes"][1]["actions"]["x"]["branches"] = [
        {"probability": 1.0, "next_node_id": "root"}
    ]
    with pytest.raises(AuditValidationError) as cycle:
        exact_adaptive_scenario_tree_oracle(
            cyclic,
            esl_contingent_policy={"root": "start", "good": "x", "bad": "x"},
        )
    assert cycle.value.code == "TREE_CYCLE"
