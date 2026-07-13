from __future__ import annotations

import argparse
import builtins
import copy
from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import pytest

from privacy_edge_sim.cli import (
    _execute,
    _parquet_safe_rows,
    command_aggregate,
    command_audit_failure_integrity,
    command_audit_hard_mask,
)
from privacy_edge_sim.config import load_config
from privacy_edge_sim.evidence import verify_run_evidence
from privacy_edge_sim.evidence_reports import build_subject_cluster_evidence_report
from privacy_edge_sim.errors import EvidenceValidationError
from privacy_edge_sim.manifest import sha256_file
from privacy_edge_sim.numerical import NumericalStudySpec, generate_numerical_study
from privacy_edge_sim.profiles import canonical_document_sha256, canonical_json_bytes
from privacy_edge_sim.profiles import load_profile
from privacy_edge_sim.safety import Observation
from privacy_edge_sim.traces import load_trace


REQUIRED_RUN_FILES = {
    "tasks.csv",
    "events.jsonl",
    "actions.jsonl",
    "resources.csv",
    "virtual_queues.csv",
    "summary.json",
    "manifest.json",
}


def _tamper_quality_support_count(document):
    support = document["quality_conformal"]["profile_evaluation_quality_support"]
    support["cells"][0]["subject_count"] += 1
    support["support_hash"] = canonical_document_sha256(support, "support_hash")


def _tamper_privacy_score_model(document):
    model = document["privacy_score_model"]
    model["quality_multiplier"]["intercept"] += 0.01
    model["model_hash"] = canonical_document_sha256(model, "model_hash")


@pytest.fixture(scope="module")
def numerical_cli_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("numerical-cli-evidence")
    paths = generate_numerical_study(
        root,
        spec=NumericalStudySpec(
            seed=1901,
            attack_train_subjects=8,
            threshold_calibration_subjects=12,
            quality_calibration_subjects=12,
            profile_evaluation_subjects=24,
            scenario_subjects=8,
            test_subjects=8,
            frames_per_subject=3,
            task_count=6,
            horizon_s=8.0,
            privacy_threshold=0.8,
        ),
    )
    config = load_config(paths.config_path)
    profile = load_profile(config.profile_path)
    evaluation = load_trace(config.trace_path, profile)
    scenario = load_trace(config.scenario_trace_path, profile)
    return paths, config, profile, evaluation, scenario


def test_numerical_evidence_is_required_and_tamper_evident(
    numerical_cli_bundle, tmp_path
):
    _, config, profile, evaluation, scenario = numerical_cli_bundle
    verified = verify_run_evidence(config, profile, evaluation, scenario)
    assert verified.required is verified.verified is True
    assert verified.file_sha256 and verified.evidence_hash

    with pytest.raises(EvidenceValidationError, match="EVIDENCE_PATH_REQUIRED"):
        verify_run_evidence(
            replace(config, evidence_path=None), profile, evaluation, scenario
        )

    tampered = tmp_path / "evidence.json"
    document = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    document["description"] = "tampered"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_HASH"):
        verify_run_evidence(
            replace(config, evidence_path=tampered), profile, evaluation, scenario
        )


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    (
        (
            lambda document: document["attacker_registry"][0].__setitem__(
                "training_subjects_hash", "0" * 64
            ),
            "EVIDENCE_ATTACK_TRAIN_SPLIT",
        ),
        (
            lambda document: document["attacker_registry"][0].__setitem__(
                "threshold_calibration_subjects_hash", "0" * 64
            ),
            "EVIDENCE_ATTACK_THRESHOLD_SPLIT",
        ),
        (
            lambda document: document["attacker_registry"][0]["thresholds"][
                "identity"
            ].__setitem__("used_for_success", True),
            "EVIDENCE_IDENTITY_RANK1",
        ),
        (
            lambda document: document["privacy_evidence"][0]["stage_statistics"][
                "single_attempt"
            ].__setitem__("joint_risk_ucb", 0.0),
            "EVIDENCE_PRIVACY_RECOMPUTE",
        ),
        (
            lambda document: document["privacy_evidence"][0].__setitem__(
                "threshold_id", "unregistered-threshold"
            ),
            "EVIDENCE_PRIVACY_THRESHOLD",
        ),
        (
            lambda document: document["privacy_protocol"][
                "registered_pipeline_ids"
            ].__setitem__(0, "unregistered-pipeline"),
            "EVIDENCE_PRIVACY_HYPOTHESIS_COUNT",
        ),
        (_tamper_quality_support_count, "EVIDENCE_QUALITY_SUPPORT_COUNT"),
        (_tamper_privacy_score_model, "EVIDENCE_ATTACK_SCORE_MODEL"),
        (
            lambda document: document["fer_paired_records"].pop(),
            "EVIDENCE_FER_FACTORIAL_SUPPORT",
        ),
    ),
)
def test_numerical_evidence_semantic_recomputation_rejects_rehashed_tampering(
    numerical_cli_bundle, tmp_path, mutation, error_code
):
    _, config, profile, evaluation, scenario = numerical_cli_bundle
    document = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    mutation(document)
    document["evidence_hash"] = canonical_document_sha256(document, "evidence_hash")
    path = tmp_path / f"{error_code}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match=error_code):
        verify_run_evidence(
            replace(config, evidence_path=path), profile, evaluation, scenario
        )


def test_numerical_evidence_recomputes_normalization_scales(
    numerical_cli_bundle, tmp_path
):
    _, config, profile, evaluation, scenario = numerical_cli_bundle
    document = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    normalization = document["cost_normalization"]
    normalization["scales"]["latency_scale_s"] *= 2.0
    normalization["calibration_hash"] = canonical_document_sha256(
        normalization, "calibration_hash"
    )
    document["evidence_hash"] = canonical_document_sha256(document, "evidence_hash")
    path = tmp_path / "normalization-tamper.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        EvidenceValidationError, match="EVIDENCE_NORMALIZATION_RECOMPUTE"
    ):
        verify_run_evidence(
            replace(config, evidence_path=path), profile, evaluation, scenario
        )


def test_numerical_evidence_freezes_controller_resource_weights(
    numerical_cli_bundle, tmp_path
):
    _, config, profile, evaluation, scenario = numerical_cli_bundle
    document = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    controller_weights = document["controller_weight_evidence"]
    assert controller_weights["online_mutable"] is False
    assert controller_weights["role"] == "scenario_training_validation"
    assert controller_weights["values"] == {
        "physical_queue_weight": config.controller.physical_queue_weight,
        "vehicle_resource_theta": dict(config.controller.vehicle_resource_theta),
        "rsu_resource_theta": dict(config.controller.rsu_resource_theta),
    }

    controller_weights["values"]["physical_queue_weight"] += 1.0
    controller_weights["values_sha256"] = hashlib.sha256(
        canonical_json_bytes(controller_weights["values"])
    ).hexdigest()
    document["evidence_hash"] = canonical_document_sha256(document, "evidence_hash")
    path = tmp_path / "controller-weights-tamper.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        EvidenceValidationError, match="EVIDENCE_CONTROLLER_WEIGHT_CONFIG"
    ):
        verify_run_evidence(
            replace(config, evidence_path=path), profile, evaluation, scenario
        )


def test_numerical_run_manifest_and_selected_path_fer_metrics(
    numerical_cli_bundle, tmp_path
):
    _, config, _, _, _ = numerical_cli_bundle
    result, _, manifest = _execute(
        config, "all_local", tmp_path / "numerical-run", overwrite=False
    )
    summary = json.loads(
        (tmp_path / "numerical-run" / "summary.json").read_text(encoding="utf-8")
    )
    evidence = manifest["frozen_evidence"]
    assert evidence["verified"] is True
    assert len(evidence["file_sha256"]) == 64
    assert evidence["size_bytes"] > 0
    assert len(evidence["split_manifest"]["splits"]) == 6
    assert evidence["quality_conformal"]["quantile"] >= 0.0
    assert evidence["attack_thresholds"]
    assert evidence["cost_normalization"]["online_mutable"] is False
    assert set(evidence["cost_normalization"]["scales"]) == {
        "latency_scale_s",
        "vehicle_energy_scale_j",
        "rsu_energy_scale_j",
        "utility_scale",
    }
    assert evidence["controller_weight_evidence"]["online_mutable"] is False
    assert evidence["controller_weight_evidence"]["values_sha256"]
    assert (
        manifest["event_priority"]["DEADLINE"]
        > manifest["event_priority"]["COMPUTE_COMPLETE"]
    )
    assert manifest["data_provenance"]["numerical_experiment_eligible"] is True

    fer = summary["selected_path_fer_classification"]
    assert fer["sample_count"] > 0
    for field in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "negative_log_likelihood",
        "expected_calibration_error",
    ):
        assert fer[field] is not None
    assert fer["by_route"]["local"]["sample_count"] == fer["sample_count"]
    assert fer["by_route"]["edge"]["sample_count"] == 0
    assert set(fer["per_class_recall"]) == set(fer["class_labels"])
    assert all(
        value is None or 0.0 <= value <= 1.0
        for value in fer["per_class_recall"].values()
    )

    task_csv = (tmp_path / "numerical-run" / "tasks.csv").read_text(encoding="utf-8")
    assert "true_label" not in task_csv
    assert "class_probabilities" not in task_csv
    assert "failure_penalty_cost" in task_csv
    assert "realized_fer_true_label" not in Observation.__dataclass_fields__
    assert "realized_fer_class_probabilities" not in Observation.__dataclass_fields__
    assert all(
        task.realized_fer_true_label is not None
        for task in result.state.tasks.values()
        if task.result_valid
    )

    hard_mask_output = tmp_path / "hard-mask-audit.json"
    assert (
        command_audit_hard_mask(
            argparse.Namespace(
                actions=str(tmp_path / "numerical-run" / "actions.jsonl"),
                output=str(hard_mask_output),
                overwrite=False,
            )
        )
        == 0
    )
    hard_mask = json.loads(hard_mask_output.read_text(encoding="utf-8"))
    assert hard_mask["hard_mask_bypassed"] is False
    assert hard_mask["unsafe_actions_executed_by_audit"] == 0

    failure_output = tmp_path / "failure-integrity-audit.json"
    assert (
        command_audit_failure_integrity(
            argparse.Namespace(
                tasks=str(tmp_path / "numerical-run" / "tasks.csv"),
                actions=str(tmp_path / "numerical-run" / "actions.jsonl"),
                events=str(tmp_path / "numerical-run" / "events.jsonl"),
                output=str(failure_output),
                overwrite=False,
            )
        )
        == 0
    )
    failure_audit = json.loads(failure_output.read_text(encoding="utf-8"))
    assert failure_audit["analysis"] == "failure_cost_completeness"
    assert failure_audit["omissions"]["failure"]["tasks_affected"] >= 1


def test_subject_evidence_report_uses_subject_clusters(numerical_cli_bundle):
    paths, _, _, _, _ = numerical_cli_bundle
    evidence = json.loads(paths.evidence_path.read_text(encoding="utf-8"))
    small = copy.deepcopy(evidence)
    small["privacy_evidence"] = small["privacy_evidence"][:1]
    report = build_subject_cluster_evidence_report(
        small, statistical_seed=47, resamples=20
    )

    assert report["independent_unit"] == "subject"
    assert len(report["report_sha256"]) == 64
    for statistic in (
        report["privacy"][0]["arrival_risk"],
        report["privacy"][0]["conditional_risk"],
        *report["fer"].values(),
    ):
        assert statistic["independent_unit"] == "subject"
        assert statistic["subject_count"] >= 2
        assert len(statistic["input_sha256"]) == 64
    assert report["fer"]["paired_nll_delta"]["difference_definition"] == (
        "anonymous_edge_minus_local_nll"
    )
    assert report["fer"]["paired_accuracy_delta"]["difference_definition"] == (
        "anonymous_edge_minus_local_accuracy"
    )


def _write_aggregate_fixture(root, *, summary_record: str = "valid"):
    root.mkdir(parents=True)
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps({"task_count": 1, "done_count": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {}
    if summary_record != "missing":
        files["summary.json"] = {
            "filename": "summary.json",
            "sha256": sha256_file(summary_path)
            if summary_record == "valid"
            else "0" * 64,
            "size_bytes": summary_path.stat().st_size,
            "row_count": None,
        }
    manifest = {
        "core_digest": "1" * 64,
        "code_version": {"value": "2" * 64},
        "configuration": {"canonical_sha256": "3" * 64},
        "versions": {"profile_hash": "4" * 64},
        "trace_identity": {"trace_hash": "5" * 64},
        "scenario_trace_identity": {"trace_hash": "6" * 64},
        "data_provenance": {
            "result_label": "synthetic_engineering_only",
            "formal_experiment_eligible": False,
        },
        "run_metadata": {"policy": "all_local", "base_seed": 17},
        "seeds": {"environment": 19},
        "outputs": {"files": files},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def test_parquet_aggregate_preserves_unsigned_seed_streams_as_text():
    rows = [
        {"seed_stream.environment": 2**64 - 1, "base_seed": 1},
        {"seed_stream.environment": 7, "base_seed": 2},
    ]
    converted = _parquet_safe_rows(rows)
    assert converted == [
        {"seed_stream.environment": str(2**64 - 1), "base_seed": 1},
        {"seed_stream.environment": "7", "base_seed": 2},
    ]


def test_aggregate_requires_verified_manifest(tmp_path):
    run_dir = tmp_path / "unverified"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "aggregate" / "result"

    with pytest.raises(FileNotFoundError, match="requires manifest.json"):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(run_dir)],
                output=str(output),
                overwrite=False,
                parquet=False,
            )
        )

    assert not output.with_suffix(".csv").exists()
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize(
    ("record_mode", "error"),
    [("missing", "output record is missing"), ("wrong", "summary checksum mismatch")],
)
def test_aggregate_requires_manifest_summary_record_and_matching_hash(
    tmp_path, record_mode, error
):
    run_dir = tmp_path / record_mode
    _write_aggregate_fixture(run_dir, summary_record=record_mode)
    output = tmp_path / "aggregate" / record_mode

    with pytest.raises(ValueError, match=error):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(run_dir)],
                output=str(output),
                overwrite=False,
                parquet=False,
            )
        )

    assert not output.with_suffix(".csv").exists()
    assert not output.with_suffix(".json").exists()


def test_aggregate_flattens_reproducibility_and_provenance_fields(tmp_path):
    run_dir = tmp_path / "verified"
    _write_aggregate_fixture(run_dir)
    output = tmp_path / "aggregate" / "result"

    assert (
        command_aggregate(
            argparse.Namespace(
                inputs=[str(run_dir)],
                output=str(output),
                overwrite=False,
                parquet=False,
            )
        )
        == 0
    )
    rows = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["code_version"] == "2" * 64
    assert row["configuration_sha256"] == "3" * 64
    assert row["profile_hash"] == "4" * 64
    assert row["evaluation_trace_hash"] == "5" * 64
    assert row["scenario_trace_hash"] == "6" * 64
    assert row["formal_experiment_eligible"] is False
    assert row["manifest_sha256"]
    assert row["summary_sha256"]


def test_aggregate_preflights_pyarrow_before_writing_csv_or_json(tmp_path, monkeypatch):
    run_dir = tmp_path / "verified"
    _write_aggregate_fixture(run_dir)
    output = tmp_path / "aggregate" / "result"
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="optional 'parquet' dependency"):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(run_dir)], output=str(output), overwrite=False, parquet=True
            )
        )

    assert not output.with_suffix(".csv").exists()
    assert not output.with_suffix(".json").exists()
    assert not output.with_suffix(".parquet").exists()


def test_cli_smoke_writes_complete_deterministic_outputs(repo_root, tmp_path):
    output_root = tmp_path / "smoke"
    command = [
        sys.executable,
        "-m",
        "privacy_edge_sim.cli",
        "smoke",
        "--config",
        str(repo_root / "configs" / "default.json"),
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout.strip().splitlines()[-1])
    report = json.loads((output_root / "smoke_report.json").read_text(encoding="utf-8"))
    assert stdout == report
    assert report["same_seed_core_digest_equal"] is True
    assert report["engineering_smoke_only"] is True
    assert (
        report["main_policy"]["core_digest"] == report["repeat_policy"]["core_digest"]
    )
    assert report["main_policy"]["data_kind"] == "synthetic"
    assert report["main_policy"]["formal_experiment_eligible"] is False

    manifests = []
    for name in ("main-a", "main-b", "baseline-all-local"):
        run_dir = output_root / name
        assert REQUIRED_RUN_FILES <= {path.name for path in run_dir.iterdir()}
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        assert manifest["invariants"]["passed"] is True
        assert manifest["invariants"]["failure_count"] == 0
        assert manifest["invariants"]["check_count"] > 0
        assert (
            manifest["data_provenance"]["result_label"] == "synthetic_engineering_only"
        )
        assert manifest["data_provenance"]["profile_data_kind"] == "synthetic"
        assert manifest["data_provenance"]["trace_data_kind"] == "synthetic"
        assert manifest["data_provenance"]["scenario_trace_data_kind"] == "synthetic"
        assert manifest["scenario_trace_identity"]["trace_hash"]
        assert len(manifest["trace_checksums"]) == 2
        assert len(manifest["core_digest"]) == 64
        assert len(manifest["manifest_sha256"]) == 64
        assert (
            manifest["simulation"]["terminal_task_count"]
            == manifest["simulation"]["task_count"]
        )

    assert manifests[0]["core_digest"] == manifests[1]["core_digest"]
    assert manifests[0]["generated_at_utc"] != ""

    repeated = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert repeated.returncode != 0
    assert "smoke output root must be empty" in repeated.stderr


@pytest.mark.parametrize("seeds", ["-1", "7,7"])
def test_cli_multi_seed_rejects_negative_or_duplicate_base_seeds(
    repo_root, tmp_path, seeds
):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "multi-seed",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--policy",
            "all_local",
            "--seeds",
            seeds,
            "--output-root",
            str(tmp_path / "invalid-seeds"),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0


def test_cli_numerical_evidence_report_is_hashed_and_reproducible(
    repo_root, numerical_cli_bundle, tmp_path
):
    paths, *_ = numerical_cli_bundle
    first = tmp_path / "evidence-report-a.json"
    second = tmp_path / "evidence-report-b.json"
    base = [
        sys.executable,
        "-m",
        "privacy_edge_sim.cli",
        "numerical-evidence-report",
        "--evidence",
        str(paths.evidence_path),
        "--subject-counts",
        "4,8,16",
        "--seed",
        "7331",
        "--resamples",
        "20",
    ]
    for output in (first, second):
        completed = subprocess.run(
            [*base, "--output", str(output)],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))
    assert left == right
    assert left["analysis"] == "numerical_offline_evidence_closure"
    assert left["provenance"]["subject_counts"] == [4, 8, 16]
    assert left["input_file_sha256"] == sha256_file(paths.evidence_path)
    assert len(left["report_sha256"]) == 64
    assert left["report_sha256"] == canonical_document_sha256(left, "report_sha256")


def test_cli_two_stage_consumes_production_action_audit_ablation_fields(
    repo_root, tmp_path
):
    actions = tmp_path / "actions.jsonl"
    commitments = tmp_path / "commitments.json"
    plan = tmp_path / "one-shot-plan.json"
    output = tmp_path / "two-stage.json"
    record = {
        "record_kind": "HARD_MASK",
        "audit_type": "HARD_MASK",
        "task_id": "task-1",
        "vehicle_id": "veh-1",
        "stage": "READY",
        "time_s": 1.25,
        "rows": [
            {
                "action_id": "READY|LOCAL|local|||",
                "action": {
                    "kind": "LOCAL",
                    "local_model_id": "local",
                    "stage": "READY",
                },
                "allowed": True,
                "reason_codes": [],
                "details": {"bounds": {"expected_cost": 1.0}},
            },
            {
                "action_id": "READY|EDGE|||rsu-1|edge",
                "action": {
                    "kind": "EDGE",
                    "rsu_id": "rsu-1",
                    "edge_model_id": "edge",
                    "stage": "READY",
                },
                "allowed": True,
                "reason_codes": [],
                "details": {
                    "bounds": {"expected_cost": 0.8},
                    "information_ablation": {
                        "observed_output_size_cost": 0.1,
                        "conservative_output_size_cost": 0.4,
                        "observed_fresh_queue_cost": 0.05,
                        "conservative_stale_queue_cost": 0.3,
                    },
                },
            },
        ],
    }
    raw_record = {
        "record_kind": "HARD_MASK",
        "audit_type": "HARD_MASK",
        "task_id": "task-1",
        "vehicle_id": "veh-1",
        "stage": "RAW",
        "time_s": 1.0,
        "rows": [
            {
                "action_id": "RAW|LOCAL|local|||",
                "action": {"kind": "LOCAL", "stage": "RAW"},
                "allowed": True,
                "reason_codes": [],
                "details": {"bounds": {"expected_cost": 1.1}},
            }
        ],
    }
    actions.write_text(
        "\n".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in (raw_record, record)
        )
        + "\n",
        encoding="utf-8",
    )
    plan.write_text(
        json.dumps(
            {
                "default_action_priority": [
                    "READY|LOCAL|local|||",
                    "READY|EDGE|||rsu-1|edge",
                ]
            }
        ),
        encoding="utf-8",
    )
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "build-one-shot-commitments",
            "--actions",
            str(actions),
            "--plan",
            str(plan),
            "--output",
            str(commitments),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert build.returncode == 0, build.stderr
    frozen_commitments = json.loads(commitments.read_text(encoding="utf-8"))
    assert frozen_commitments["uses_ready_records"] is False
    assert frozen_commitments["uses_ready_expected_cost"] is False
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "audit-two-stage",
            "--actions",
            str(actions),
            "--commitments",
            str(commitments),
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
    report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["input_sha256"] == report["input_sha256"]
    assert report["pairs"][0]["ready_recourse"]["action_id"].startswith("READY|EDGE")
    assert report["pairs"][0]["without_output_size"]["action_id"].startswith(
        "READY|LOCAL"
    )
    assert report["hard_mask_bypassed"] is False
    assert (
        report["one_shot_commitment_provenance"]["legacy_unregistered_mapping"] is False
    )
    assert report["report_sha256"] == canonical_document_sha256(report, "report_sha256")


def test_cli_exact_scenario_oracle_records_file_and_semantic_input_hashes(
    repo_root, tmp_path
):
    input_path = tmp_path / "oracle-input.json"
    output = tmp_path / "oracle-output.json"
    stage = {
        "a": {"hard_safe": True, "cost": 1.0, "duration_s": 1.0},
        "b": {"hard_safe": True, "cost": 2.0, "duration_s": 1.0},
    }
    document = {
        "scenarios": [
            {
                "scenario_id": "s1",
                "probability": 1.0,
                "stages": [stage, stage],
            }
        ],
        "esl_action_sequence": ["b", "b"],
        "max_sequences": 4,
    }
    input_path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "exact-scenario-oracle",
            "--input",
            str(input_path),
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
    report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["input_file_sha256"] == sha256_file(input_path)
    assert report["input_file_sha256"] == sha256_file(input_path)
    assert stdout["input_sha256"] == report["input_sha256"]
    assert report["optimum"]["action_sequence"] == ["a", "a"]
    assert report["report_sha256"] == canonical_document_sha256(report, "report_sha256")


def test_cli_exact_adaptive_scenario_oracle_emits_contingent_policy_hashes(
    repo_root, tmp_path
):
    source = tmp_path / "adaptive-oracle-input.json"
    output = tmp_path / "adaptive-oracle-output.json"

    def terminal(x, y):
        return {
            "x": {
                "hard_safe": True,
                "cost": x,
                "duration_s": 1.0,
                "branches": [],
            },
            "y": {
                "hard_safe": True,
                "cost": y,
                "duration_s": 1.0,
                "branches": [],
            },
        }

    document = {
        "scenario_tree": {
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
                {"node_id": "good", "actions": terminal(1.0, 3.0)},
                {"node_id": "bad", "actions": terminal(4.0, 1.0)},
            ],
        },
        "esl_contingent_policy": {"root": "start", "good": "x", "bad": "x"},
        "max_policies": 4,
    }
    source.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "privacy_edge_sim.cli",
            "exact-adaptive-scenario-oracle",
            "--input",
            str(source),
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
    report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["input_file_sha256"] == sha256_file(source)
    assert stdout["input_sha256"] == report["input_sha256"]
    assert report["optimum"]["contingent_policy"] == {
        "bad": "y",
        "good": "x",
        "root": "start",
    }
    assert report["report_sha256"] == canonical_document_sha256(report, "report_sha256")
