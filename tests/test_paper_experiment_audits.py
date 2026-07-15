from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from privacy_edge_sim.paper_experiment_audits import (
    AuditValidationError,
    audit_failure_cost_completeness,
    audit_failure_cost_coverage,
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
    assert report["execution_validation_status"] == "NOT_OBSERVED"
    assert (
        report["execution_validation_scope"]
        == "RAW_READY_POLICY_DECISIONS_AND_EXECUTION_TIME_REPAIRS_ONLY"
    )
    assert report["executed_action_count"] == 0
    assert report["validated_count"] == 0
    assert report["violation_count"] == 0
    assert report["hard_mask_bypassed"] is None
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


def _closed_action(stage, kind, **identifiers):
    return {"stage": stage, "kind": kind, **identifiers}


def _execution_mask(
    task_id,
    stage,
    time_s,
    rows,
    *,
    mask_epoch="DECISION",
    execution_check_id=None,
):
    result = {
        "record_kind": "HARD_MASK",
        "task_id": task_id,
        "stage": stage,
        "time_s": time_s,
        "mask_epoch": mask_epoch,
        "rows": rows,
    }
    if execution_check_id is not None:
        result["execution_check_id"] = execution_check_id
    return result


def _execution_row(action, allowed, *, reasons=()):
    action_id = "|".join(
        (
            action["stage"],
            action["kind"],
            action.get("local_model_id", ""),
            action.get("pipeline_id", ""),
            action.get("rsu_id", ""),
            action.get("edge_model_id", ""),
        )
    )
    return {
        "action_id": action_id,
        "action": action,
        "allowed": allowed,
        "reason_codes": list(reasons),
        "details": {"bounds": {"expected_cost": 1.0}},
    }


def _execution_record(
    kind, task_id, action, *, time_s=None, execution_check_id=None
):
    result = {
        "record_kind": kind,
        "task_id": task_id,
        "executed": action,
    }
    if kind == "EXECUTION_REPAIR":
        result["executed_stage"] = action["stage"]
        result["execution_check_id"] = execution_check_id
    elif execution_check_id is not None:
        result["execution_check_id"] = execution_check_id
    if time_s is not None:
        result["time_s"] = time_s
    return result


def _write_hard_mask_manifest(
    actions,
    *,
    source_clean=True,
    invariants_passed=True,
    corrupt_self_hash=False,
):
    actions_hash = hashlib.sha256(actions.read_bytes()).hexdigest()
    manifest = {
        "manifest_schema_version": "1.1",
        "code_version": {
            "source_commit_reproducible": source_clean,
            "source_git_dirty": not source_clean,
            "source_git_status": [] if source_clean else [" M src/privacy_edge_sim/x.py"],
        },
        "data_provenance": {"source_commit_reproducible": source_clean},
        "source_cleanliness_preflight": {
            "require_clean_source": True,
            "requirement_status": "passed" if source_clean else "failed",
            "source_commit_reproducible": source_clean,
            "source_git_dirty": not source_clean,
            "git_commit": "a" * 40,
        },
        "invariants": {
            "passed": invariants_passed,
            "status": "passed" if invariants_passed else "failed",
            "check_count": 7,
            "failure_count": 0 if invariants_passed else 1,
            "failures": [] if invariants_passed else [{"code": "BROKEN"}],
        },
        "outputs": {
            "files": {
                actions.name: {
                    "filename": actions.name,
                    "sha256": actions_hash,
                    "size_bytes": actions.stat().st_size,
                    "row_count": len(actions.read_text(encoding="utf-8").splitlines()),
                }
            }
        },
    }
    manifest["manifest_sha256"] = canonical_document_sha256(
        manifest, "manifest_sha256"
    )
    if corrupt_self_hash:
        manifest["manifest_sha256"] = "0" * 64
    (actions.parent / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_hard_mask_execution_uses_latest_causal_same_task_stage_mask():
    task_id = "task-pair"
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask(
            task_id,
            "RAW",
            1.0,
            [_execution_row(local, False, reasons=("BATTERY_GUARD",))],
        ),
        _execution_mask(
            task_id,
            "RAW",
            2.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-1",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            task_id,
            local,
            time_s=2.0,
            execution_check_id="check-1",
        ),
    ]

    report = audit_hard_mask_counterfactual(records)

    assert report["execution_validation_status"] == "VALIDATED"
    assert report["executed_action_count"] == 1
    assert report["validated_count"] == 1
    assert report["violation_count"] == 0
    assert report["hard_mask_bypassed"] is False
    pairing = report["execution_pairings"][0]
    assert pairing["mask_time_s"] == 2.0
    assert pairing["execution_time_s"] == 2.0
    assert pairing["validation_status"] == "VALIDATED_ALLOWED_MEMBER"


@pytest.mark.parametrize(
    ("mask_rows", "violation_code"),
    [
        (
            [
                _execution_row(
                    _closed_action("RAW", "LOCAL", local_model_id="local-v1"),
                    False,
                    reasons=("OOD",),
                )
            ],
            "EXECUTED_ACTION_REJECTED_BY_MASK",
        ),
        (
            [
                _execution_row(
                    _closed_action("RAW", "LOCAL", local_model_id="other"), True
                )
            ],
            "EXECUTED_ACTION_ABSENT_FROM_MASK",
        ),
    ],
)
def test_hard_mask_execution_reports_rejected_or_absent_membership_violation(
    mask_rows, violation_code
):
    executed = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask(
            "task-v",
            "RAW",
            1.0,
            mask_rows,
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-v",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-v",
            executed,
            time_s=1.0,
            execution_check_id="check-v",
        ),
    ]

    report = audit_hard_mask_counterfactual(records)

    assert report["execution_validation_status"] == "VIOLATION"
    assert report["executed_action_count"] == 1
    assert report["validated_count"] == 0
    assert report["violation_count"] == 1
    assert report["hard_mask_bypassed"] is True
    assert report["violations"][0]["violation_code"] == violation_code


def test_hard_mask_execution_prefers_repair_over_policy_decision_per_stage():
    local = _closed_action("READY", "LOCAL", local_model_id="local-v1")
    fail = _closed_action("READY", "FAIL")
    records = [
        _execution_mask(
            "task-repair",
            "READY",
            1.0,
            [_execution_row(local, True), _execution_row(fail, False, reasons=("X",))],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-repair",
        ),
        _execution_record("POLICY_DECISION", "task-repair", fail),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-repair",
            local,
            time_s=1.0,
            execution_check_id="check-repair",
        ),
    ]

    report = audit_hard_mask_counterfactual(records)

    assert report["executed_action_count"] == 1
    assert report["validated_count"] == 1
    assert report["violation_count"] == 0
    assert report["ignored_policy_decision_count"] == 1
    assert report["execution_pairings"][0]["execution_record_kind"] == (
        "EXECUTION_REPAIR"
    )


def test_hard_mask_execution_falls_back_to_unique_policy_decision_pair():
    pipeline = _closed_action("RAW", "PIPE", pipeline_id="pipe-v1")
    records = [
        _execution_mask("task-policy", "RAW", 1.0, [_execution_row(pipeline, True)]),
        _execution_record("POLICY_DECISION", "task-policy", pipeline),
    ]

    report = audit_hard_mask_counterfactual(records)

    assert report["execution_validation_status"] == "VALIDATED"
    assert report["execution_pairings"][0]["execution_record_kind"] == (
        "POLICY_DECISION"
    )


def test_hard_mask_execution_refuses_ambiguous_or_noncausal_pairing():
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    same_time_masks = [
        _execution_mask(
            "task-a",
            "RAW",
            1.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="duplicate",
        ),
        _execution_mask(
            "task-a",
            "RAW",
            1.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="duplicate",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-a",
            local,
            time_s=1.1,
            execution_check_id="duplicate",
        ),
    ]
    with pytest.raises(AuditValidationError) as ambiguous:
        audit_hard_mask_counterfactual(same_time_masks)
    assert ambiguous.value.code == "AUDIT_EXECUTION_BINDING_INVALID"

    future_mask = [
        _execution_mask(
            "task-b",
            "RAW",
            2.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="future",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-b",
            local,
            time_s=1.1,
            execution_check_id="future",
        ),
    ]
    with pytest.raises(AuditValidationError) as noncausal:
        audit_hard_mask_counterfactual(future_mask)
    assert noncausal.value.code == "AUDIT_EXECUTION_TIME_MISMATCH"


def test_hard_mask_execution_refuses_incomplete_decision_record():
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask("task-missing", "RAW", 1.0, [_execution_row(local, True)]),
        {
            "record_kind": "EXECUTION_REPAIR",
            "task_id": "task-missing",
            "time_s": 1.1,
        },
    ]
    with pytest.raises(AuditValidationError) as captured:
        audit_hard_mask_counterfactual(records)
    assert captured.value.code == "AUDIT_EXECUTED_ACTION_MISSING"


def test_hard_mask_execution_refuses_repair_without_exact_mask_binding():
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask("task-binding", "RAW", 1.0, [_execution_row(local, True)]),
        {
            "record_kind": "EXECUTION_REPAIR",
            "task_id": "task-binding",
            "executed_stage": "RAW",
            "executed": local,
            "time_s": 1.1,
        },
    ]

    with pytest.raises(AuditValidationError) as captured:
        audit_hard_mask_counterfactual(records)

    assert captured.value.code == "AUDIT_EXECUTION_BINDING_MISSING"


def test_cli_hard_mask_audit_validates_actual_execution_member(repo_root, tmp_path):
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask(
            "task-cli",
            "RAW",
            1.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-cli",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-cli",
            local,
            time_s=1.0,
            execution_check_id="check-cli",
        ),
    ]
    actions = tmp_path / "actions.jsonl"
    output = tmp_path / "hard-mask.json"
    actions.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    _write_hard_mask_manifest(actions)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "audit-hard-mask",
            "--actions",
            str(actions),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["execution_validation_status"] == "VALIDATED"
    assert stdout["executed_action_count"] == 1
    assert stdout["validated_count"] == 1
    assert stdout["violation_count"] == 0
    assert stdout["provenance_status"] == "VERIFIED_FORMAL_PROVENANCE"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["input_file_sha256"] == report["input_file_sha256"]
    assert len(report["input_file_sha256"]) == 64
    assert report["executed_action_count"] == 1
    assert report["validated_count"] == 1
    assert report["violation_count"] == 0
    assert report["hard_mask_bypassed"] is False
    assert report["input_artifact_verification"]["manifest_verified"] is True
    assert report["input_artifact_verification"]["invariants_passed"] is True
    assert report["report_sha256"] == canonical_document_sha256(
        report, "report_sha256"
    )


def test_cli_hard_mask_audit_writes_violation_evidence_before_nonzero_exit(
    repo_root, tmp_path
):
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask(
            "task-cli-violation",
            "RAW",
            1.0,
            [_execution_row(local, False, reasons=("OOD",))],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-violation",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-cli-violation",
            local,
            time_s=1.0,
            execution_check_id="check-violation",
        ),
    ]
    actions = tmp_path / "actions-violation.jsonl"
    output = tmp_path / "hard-mask-violation.json"
    actions.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    _write_hard_mask_manifest(actions)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "audit-hard-mask",
            "--actions",
            str(actions),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2, completed.stderr
    assert output.is_file()
    stdout = json.loads(completed.stdout)
    assert stdout["execution_validation_status"] == "VIOLATION"
    assert stdout["violation_count"] == 1
    assert stdout["hard_mask_bypassed"] is True
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["violation_count"] == 1
    assert report["hard_mask_bypassed"] is True


def test_cli_hard_mask_audit_writes_structured_refusal_before_nonzero_exit(
    repo_root, tmp_path
):
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask("task-refused", "RAW", 1.0, [_execution_row(local, True)]),
        {
            "record_kind": "EXECUTION_REPAIR",
            "task_id": "task-refused",
            "executed_stage": "RAW",
            "executed": local,
            "time_s": 1.1,
        },
    ]
    actions = tmp_path / "actions-refused.jsonl"
    output = tmp_path / "hard-mask-refused.json"
    actions.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    _write_hard_mask_manifest(actions)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "audit-hard-mask",
            "--actions",
            str(actions),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "REFUSED"
    assert report["error_code"] == "AUDIT_EXECUTION_BINDING_MISSING"
    assert report["execution_validation_status"] == "REFUSED"
    assert report["hard_mask_bypassed"] is None


@pytest.mark.parametrize(
    "failure_mode",
    ("manifest_hash", "actions_hash", "dirty_source", "invariants"),
)
def test_cli_hard_mask_strict_provenance_refuses_invalid_run_artifact(
    repo_root, tmp_path, failure_mode
):
    run = tmp_path / failure_mode
    run.mkdir()
    local = _closed_action("RAW", "LOCAL", local_model_id="local-v1")
    records = [
        _execution_mask(
            "task-provenance",
            "RAW",
            1.0,
            [_execution_row(local, True)],
            mask_epoch="EXECUTION_RECHECK",
            execution_check_id="check-provenance",
        ),
        _execution_record(
            "EXECUTION_REPAIR",
            "task-provenance",
            local,
            time_s=1.0,
            execution_check_id="check-provenance",
        ),
    ]
    actions = run / "actions.jsonl"
    output = tmp_path / f"{failure_mode}.json"
    actions.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    _write_hard_mask_manifest(
        actions,
        source_clean=failure_mode != "dirty_source",
        invariants_passed=failure_mode != "invariants",
        corrupt_self_hash=failure_mode == "manifest_hash",
    )
    if failure_mode == "actions_hash":
        actions.write_text(
            actions.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "audit-hard-mask",
            "--actions",
            str(actions),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "REFUSED"
    assert report["error_code"] == "AUDIT_INPUT_PROVENANCE_INVALID"
    assert report["execution_validation_status"] == "REFUSED"
    assert report["input_artifact_verification"]["status"] == "REFUSED"
    assert report["input_artifact_verification"]["manifest_verified"] is False


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
    assert report["coverage_schema_version"] == "1.1"
    assert report["coverage_status"] == "PARTIAL_OBSERVED_CATEGORY_COVERAGE"
    assert report["accounting_validation_status"] == "COMPLETE_FOR_OBSERVED_PATHS"
    assert report["observed_coverage"]["retry"]["status"] == (
        "OBSERVED_ACCOUNTING_VALIDATED"
    )
    assert report["observed_coverage"]["downlink"]["status"] == (
        "OBSERVED_ACCOUNTING_VALIDATED"
    )
    assert report["observed_coverage"]["edge_execution"]["status"] == (
        "NOT_OBSERVED"
    )
    assert report["coverage_scope"]["does_not_claim_unobserved_categories"] is True
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
    assert report["report_sha256"] == canonical_document_sha256(
        report, "report_sha256"
    )


def test_failure_completeness_structurally_refuses_missing_fields():
    task = _complete_task()
    del task["anon_attempts"][1]["vehicle_energy_j"]
    with pytest.raises(AuditValidationError) as captured:
        audit_failure_cost_completeness([task], [], [])
    refusal = captured.value.as_dict()
    assert refusal["status"] == "REFUSED"
    assert refusal["error_code"] == "FAILURE_RETRY_FIELDS_MISSING"


def _edge_failure_task():
    return {
        "task_id": "edge-failure",
        "attempt_started_count": 1,
        "anon_attempts": [
            {
                "attempt": 1,
                "latency_s": 0.1,
                "vehicle_energy_j": 0.6,
            }
        ],
        "network_audit": [
            {"direction": "UL", "status": "START", "time_s": 1.0},
            {
                "direction": "UL",
                "status": "DONE",
                "time_s": 1.2,
                "vehicle_energy_j": 0.3,
                "rsu_energy_j": 0.2,
            },
            {"direction": "DL", "status": "START", "time_s": 1.5},
            {
                "direction": "DL",
                "status": "FAIL",
                "time_s": 1.6,
                "vehicle_energy_j": 0.05,
                "rsu_energy_j": 0.1,
            },
        ],
        "rsu_audit": [
            {"admission": "ACCEPT", "time_s": 1.2},
            {"phase": "ingress_start", "time_s": 1.2},
            {"phase": "ingress_done", "time_s": 1.3, "valid": True},
        ],
        "rsu_attributed_energy_j": 1.5,
        "failure_penalty_cost": 2.0,
        "failure_reason": "RSU_FAILURE",
        "actual_path": ["ANON#1", "UL:rsu-1", "EDGE:rsu-1", "DL:rsu-1"],
    }


def _admission_reject_task():
    return {
        "task_id": "admission-reject",
        "attempt_started_count": 1,
        "anon_attempts": [
            {
                "attempt": 1,
                "latency_s": 0.1,
                "vehicle_energy_j": 0.6,
            }
        ],
        "network_audit": [
            {"direction": "UL", "status": "START", "time_s": 2.0},
            {
                "direction": "UL",
                "status": "DONE",
                "time_s": 2.2,
                "vehicle_energy_j": 0.3,
                "rsu_energy_j": 0.2,
            },
        ],
        "rsu_audit_json": (
            '[{"admission":"REJECT","reason_codes":'
            '["RSU_DESCRIPTOR_CAPACITY"],"time_s":2.2}]'
        ),
        "rsu_attributed_energy_j": 0.2,
        "failure_penalty_cost": 0.0,
        "failure_reason": "NONE",
        "actual_path": ["ANON#1", "UL:rsu-1", "LOCAL_FER"],
    }


def test_failure_coverage_aggregates_edge_admission_rsu_and_downlink_runs():
    aggregate = audit_failure_cost_coverage(
        [
            {
                "run_id": "edge-run",
                "task_rows": [_edge_failure_task()],
                "action_records": [],
                "event_records": [],
            },
            {
                "run_id": "reject-run",
                "task_rows": [_admission_reject_task()],
                "action_records": [],
                "event_records": [],
            },
        ]
    )
    assert aggregate["run_count"] == 2
    assert aggregate["task_count"] == 2
    assert aggregate["coverage_schema_version"] == "1.1"
    assert aggregate["coverage_status"] == aggregate["status"]
    for category in (
        "uplink",
        "rsu_attributed_energy",
        "downlink",
        "downlink_failure",
        "failure_penalty",
    ):
        assert aggregate["observed_coverage"][category]["status"] == (
            "OBSERVED_ACCOUNTING_VALIDATED"
        )
    for category in (
        "admission_accept",
        "admission_reject",
        "rsu_ingress",
        "edge_execution",
        "rsu_failure",
        "local_fallback",
    ):
        assert aggregate["observed_coverage"][category]["status"] == (
            "OBSERVED_STRUCTURE_VALIDATED"
        )
    assert aggregate["observed_coverage"]["edge_execution"]["observed_run_count"] == 1
    assert aggregate["observed_coverage"]["admission_reject"]["task_count"] == 1
    assert aggregate["observed_coverage"]["admission_reject"]["reason_counts"] == {
        "RSU_DESCRIPTOR_CAPACITY": 1
    }
    assert aggregate["coverage_scope"]["does_not_claim_unobserved_categories"] is True
    assert "uplink_failure" in aggregate["coverage_scope"]["not_observed_categories"]
    assert aggregate["report_sha256"] == canonical_document_sha256(
        aggregate, "report_sha256"
    )


def test_failure_coverage_does_not_double_count_explicit_rsu_failure_marker():
    task = {
        "task_id": "ingress-failure",
        "attempt_started_count": 0,
        "anon_attempts": [],
        "network_audit": [],
        "rsu_audit": [
            {"phase": "ingress_start", "time_s": 1.0},
            {
                "phase": "ingress_done",
                "time_s": 1.1,
                "valid": False,
            },
        ],
        "rsu_attributed_energy_j": 0.2,
        "failure_penalty_cost": 2.0,
        "failure_reason": "EDGE_FAILED",
        "actual_path": ["RSU_INGRESS_FAIL:rsu-1"],
    }

    report = audit_failure_cost_completeness([task], [], [])

    assert report["observed_coverage"]["rsu_failure"] == {
        "observation_count": 1,
        "task_count": 1,
        "task_ids": ["ingress-failure"],
        "status": "OBSERVED_STRUCTURE_VALIDATED",
        "validation_scope": "recorded RSU/edge failure marker",
    }
    assert report["report_sha256"] == canonical_document_sha256(
        report, "report_sha256"
    )


def test_failure_coverage_aggregates_distinct_admission_capacity_reasons():
    descriptor = _admission_reject_task()
    workload = copy.deepcopy(descriptor)
    workload["task_id"] = "workload-reject"
    workload["rsu_audit_json"] = (
        '[{"admission":"REJECT","reason_codes":'
        '["RSU_VRAM_CAPACITY","RSU_WORKLOAD_CAPACITY"],"time_s":2.2}]'
    )

    report = audit_failure_cost_coverage(
        [
            {
                "run_id": "capacity-reasons",
                "task_rows": [descriptor, workload],
                "action_records": [],
                "event_records": [],
            }
        ]
    )

    assert report["observed_coverage"]["admission_reject"]["reason_counts"] == {
        "RSU_DESCRIPTOR_CAPACITY": 1,
        "RSU_VRAM_CAPACITY": 1,
        "RSU_WORKLOAD_CAPACITY": 1,
    }


def test_failure_coverage_refuses_admission_reject_without_reason():
    task = _admission_reject_task()
    task["rsu_audit_json"] = '[{"admission":"REJECT","time_s":2.2}]'

    with pytest.raises(AuditValidationError) as captured:
        audit_failure_cost_completeness([task], [], [])

    assert captured.value.code == "FAILURE_ADMISSION_REASON_MISSING"


def test_failure_coverage_refuses_duplicate_run_ids():
    run = {
        "run_id": "same",
        "task_rows": [_admission_reject_task()],
        "action_records": [],
        "event_records": [],
    }
    with pytest.raises(AuditValidationError) as captured:
        audit_failure_cost_coverage([run, copy.deepcopy(run)])
    assert captured.value.code == "FAILURE_DUPLICATE_RUN"


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
