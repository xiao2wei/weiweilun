from __future__ import annotations

import argparse
import builtins
import copy
from dataclasses import replace
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import privacy_edge_sim.cli as cli_module
from privacy_edge_sim.cli import (
    _execute,
    _expand_sweep_coordinates,
    _mechanism_snapshot,
    _parquet_safe_rows,
    _study_experiment_registration,
    _study_mechanism_diagnostics,
    _sweep_experiment_registration,
    _sweep_mechanism_diagnostics,
    build_parser,
    command_aggregate,
    command_derive_config,
    command_audit_failure_coverage,
    command_audit_failure_integrity,
    command_audit_hard_mask,
    command_sweep,
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
from privacy_edge_sim.simulator import DiscreteEventSimulator
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


def _mechanism_summary(
    *, edge: float, pipeline: float, waiting: int, loss: float = 0.3
) -> dict:
    return {
        "edge_done_rate": edge,
        "pipeline_attempt_rate": pipeline,
        "pipeline_to_edge_rate": edge / pipeline if pipeline else 0.0,
        "pipeline_to_local_rate": 1.0 if pipeline else 0.0,
        "all_task_loss": loss,
        "coverage": 1.0,
        "failure_rate": 0.0,
        "timeout_rate": 0.0,
        "latency_p95_s": 0.2,
        "energy_j": {"task_attributed": {"total": 2.0}},
        "resources": {"max_utilization": 0.4, "max_waiting_jobs": waiting},
    }


def _write_audit_manifest(run):
    files = {}
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        path = run / name
        files[name] = {
            "filename": name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_count": 0,
        }
    manifest = {
        "schema_version": "1.0",
        "invariants": {"passed": True, "failure_count": 0},
        "source_cleanliness_preflight": {
            "require_clean_source": True,
            "requirement_status": "passed",
            "source_commit_reproducible": True,
            "source_git_dirty": False,
            "git_commit": "1" * 40,
        },
        "outputs": {"files": files},
    }
    manifest_hash = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    (run / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest_hash


def test_study_mechanism_diagnostics_flags_unexercised_edge_and_queueing():
    rows = [
        {
            "policy": "all_local",
            "mechanism_metrics": _mechanism_snapshot(
                _mechanism_summary(edge=0.0, pipeline=0.0, waiting=0)
            ),
        },
        {
            "policy": "esl_smpc",
            "mechanism_metrics": _mechanism_snapshot(
                _mechanism_summary(edge=0.0, pipeline=0.5, waiting=0)
            ),
        },
    ]

    report = _study_mechanism_diagnostics(rows, baseline="all_local")

    codes = {(row["code"], row.get("policy")) for row in report["warnings"]}
    assert ("POLICY_NO_EDGE_COMPLETION", "esl_smpc") in codes
    assert ("PIPELINE_WITHOUT_EDGE_COMPLETION", "esl_smpc") in codes
    assert ("NO_OBSERVED_RESOURCE_QUEUEING", None) in codes


def test_sweep_mechanism_diagnostics_distinguishes_configuration_from_behavior():
    snapshot = _mechanism_snapshot(
        _mechanism_summary(edge=0.0, pipeline=0.0, waiting=0)
    )
    rows = [
        {
            "parameters": {"controller.lyapunov_v": value},
            "mechanism_metrics": dict(snapshot),
        }
        for value in (6.0, 12.0)
    ]

    report = _sweep_mechanism_diagnostics(
        rows, coordinate_keys=["controller.lyapunov_v"]
    )

    assert report["unique_behavior_signature_count"] == 1
    assert report["parameter_observed_effect"] == {
        "controller.lyapunov_v": False
    }
    assert all(row["behavior_signature_sha256"] for row in rows)
    assert {
        warning["code"] for warning in report["warnings"]
    } >= {
        "NO_OBSERVED_BEHAVIOR_VARIATION",
        "PARAMETER_NO_OBSERVED_EFFECT",
        "SWEEP_NO_EDGE_COMPLETION",
        "SWEEP_NO_RESOURCE_QUEUEING",
    }


def test_sweep_mechanism_diagnostics_detects_conditional_parameter_effect():
    rows = []
    for v in (6.0, 12.0):
        for capacity in (0.03, 0.12):
            rows.append(
                {
                    "parameters": {
                        "controller.lyapunov_v": v,
                        "rsus.0.workload_capacity_gpu_s": capacity,
                    },
                    "mechanism_metrics": _mechanism_snapshot(
                        _mechanism_summary(
                            edge=0.25 if capacity > 0.03 else 0.0,
                            pipeline=0.5,
                            waiting=1 if capacity > 0.03 else 0,
                        )
                    ),
                }
            )

    report = _sweep_mechanism_diagnostics(
        rows,
        coordinate_keys=[
            "controller.lyapunov_v",
            "rsus.0.workload_capacity_gpu_s",
        ],
    )
    assert report["parameter_observed_effect"]["controller.lyapunov_v"] is False
    assert (
        report["parameter_observed_effect"][
            "rsus.0.workload_capacity_gpu_s"
        ]
        is True
    )


def test_paired_sweep_expands_lockstep_not_cartesian(repo_root):
    base = json.loads(
        (repo_root / "configs" / "default.json").read_text(encoding="utf-8")
    )
    grid = {
        "$paired": [
            {
                "rsus.0.workload_capacity_gpu_s": 0.03,
                "rsus.1.workload_capacity_gpu_s": 0.03,
            },
            {
                "rsus.0.workload_capacity_gpu_s": 30.0,
                "rsus.1.workload_capacity_gpu_s": 18.0,
            },
        ]
    }

    keys, cases = _expand_sweep_coordinates(base, grid)
    assert keys == [
        "rsus.0.workload_capacity_gpu_s",
        "rsus.1.workload_capacity_gpu_s",
    ]
    assert cases == grid["$paired"]
    assert len(cases) == 2


def test_failure_coverage_cli_aggregates_run_directories(monkeypatch, tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    manifest_hash = _write_audit_manifest(run)
    monkeypatch.setattr(
        "privacy_edge_sim.cli._read_failure_task_rows", lambda path: [{"task_id": "t"}]
    )
    monkeypatch.setattr(
        "privacy_edge_sim.cli._read_jsonl", lambda path: [{"record_kind": "TEST"}]
    )

    def fake_coverage(runs):
        assert len(runs) == 1
        assert runs[0]["run_id"] == f"manifest:{manifest_hash}"
        return {
            "schema_version": "1.0",
            "analysis": "failure_cost_coverage_aggregate",
            "status": "PARTIAL_OBSERVED_CATEGORY_COVERAGE",
            "run_count": 1,
            "task_count": 1,
            "coverage_scope": {
                "not_observed_categories": ["downlink_failure"]
            },
            "observed_coverage": {
                "uplink": {"status": "OBSERVED_ACCOUNTING_VALIDATED"},
                "downlink_failure": {"status": "NOT_OBSERVED"},
            },
            "report_sha256": "",
        }

    monkeypatch.setattr(
        "privacy_edge_sim.cli.audit_failure_cost_coverage", fake_coverage
    )
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--runs",
            str(run),
            "--require-categories",
            "uplink",
            "--output",
            str(output),
        ]
    )

    assert args.func is command_audit_failure_coverage
    assert args.func(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["required_observed_categories"] == ["uplink"]
    assert report["required_observed_categories_satisfied"] is True
    assert report["input_artifact_verification"]["status"] == "VERIFIED"
    assert report["report_sha256"] == canonical_document_sha256(
        report, "report_sha256"
    )


def test_failure_coverage_cli_required_category_is_a_prewrite_gate(
    monkeypatch, tmp_path
):
    run = tmp_path / "run-a"
    run.mkdir()
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    _write_audit_manifest(run)
    monkeypatch.setattr(
        "privacy_edge_sim.cli._read_failure_task_rows", lambda path: [{"task_id": "t"}]
    )
    monkeypatch.setattr("privacy_edge_sim.cli._read_jsonl", lambda path: [])
    monkeypatch.setattr(
        "privacy_edge_sim.cli.audit_failure_cost_coverage",
        lambda runs: {
            "coverage_scope": {"not_observed_categories": ["downlink_failure"]},
            "observed_coverage": {"downlink_failure": {"status": "NOT_OBSERVED"}},
        },
    )
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--runs",
            str(run),
            "--require-categories",
            "downlink_failure",
            "--output",
            str(output),
        ]
    )

    with pytest.raises(ValueError, match="were not observed"):
        args.func(args)
    assert not output.exists()


def test_failure_coverage_cli_discovers_runs_below_study_root(monkeypatch, tmp_path):
    run = tmp_path / "study" / "env-1" / "all_local"
    run.mkdir(parents=True)
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    manifest_hash = _write_audit_manifest(run)
    monkeypatch.setattr(
        "privacy_edge_sim.cli._read_failure_task_rows", lambda path: [{"task_id": "t"}]
    )
    monkeypatch.setattr("privacy_edge_sim.cli._read_jsonl", lambda path: [])
    captured = {}

    def fake_coverage(runs):
        captured["runs"] = runs
        return {
            "schema_version": "1.0",
            "analysis": "failure_cost_coverage_aggregate",
            "status": "COMPLETE_OBSERVED_CATEGORY_COVERAGE",
            "run_count": 1,
            "task_count": 1,
            "coverage_scope": {"not_observed_categories": []},
            "observed_coverage": {"uplink": {"status": "OBSERVED"}},
            "report_sha256": "",
        }

    monkeypatch.setattr(
        "privacy_edge_sim.cli.audit_failure_cost_coverage", fake_coverage
    )
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--study-roots",
            str(tmp_path / "study"),
            "--output",
            str(output),
        ]
    )

    assert args.func(args) == 0
    assert len(captured["runs"]) == 1
    assert captured["runs"][0]["run_id"] == f"manifest:{manifest_hash}"


def test_failure_coverage_cli_rejects_manifest_bound_artifact_tampering(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    _write_audit_manifest(run)
    (run / "actions.jsonl").write_text('{"tampered":true}\n', encoding="utf-8")
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--runs",
            str(run),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        args.func(args)
    assert not output.exists()


def test_failure_coverage_cli_rejects_dirty_source_manifest(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    for name in ("tasks.csv", "actions.jsonl", "events.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    _write_audit_manifest(run)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_cleanliness_preflight"].update(
        require_clean_source=False,
        requirement_status="not_required",
        source_commit_reproducible=False,
        source_git_dirty=True,
    )
    manifest["manifest_sha256"] = canonical_document_sha256(
        manifest, "manifest_sha256"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--runs",
            str(run),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(ValueError, match="clean committed run source"):
        args.func(args)
    assert not output.exists()


def test_sweep_rejects_duplicate_coordinate_levels_before_output(repo_root, tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps({"controller.lyapunov_v": [6.0, 6.0]}), encoding="utf-8"
    )
    output = tmp_path / "sweep"
    args = build_parser().parse_args(
        [
            "sweep",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--grid",
            str(grid),
            "--output-root",
            str(output),
            "--allow-dirty-source",
        ]
    )

    with pytest.raises(ValueError, match="at least two distinct levels"):
        args.func(args)
    assert not output.exists()


def test_sweep_rejects_duplicate_json_grid_keys(repo_root, tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(
        '{"controller.lyapunov_v":[6,12],"controller.lyapunov_v":[3,24]}',
        encoding="utf-8",
    )
    output = tmp_path / "sweep"
    args = build_parser().parse_args(
        [
            "sweep",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--grid",
            str(grid),
            "--output-root",
            str(output),
            "--allow-dirty-source",
        ]
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        args.func(args)
    assert not output.exists()


def test_interrupted_sweep_keeps_marker_and_no_index(
    monkeypatch, repo_root, tmp_path
):
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps({"controller.lyapunov_v": [6.0, 12.0]}), encoding="utf-8"
    )
    output = tmp_path / "sweep"
    monkeypatch.setattr(
        "privacy_edge_sim.cli._execute",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    args = build_parser().parse_args(
        [
            "sweep",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--grid",
            str(grid),
            "--output-root",
            str(output),
            "--allow-dirty-source",
        ]
    )

    assert args.func is command_sweep
    with pytest.raises(RuntimeError, match="interrupted"):
        args.func(args)
    assert (output / "sweep.in_progress.json").is_file()
    assert not (output / "sweep.json").exists()
    assert not (output / "sweep_diagnostics.json").exists()

    aggregate_args = build_parser().parse_args(
        [
            "aggregate",
            "--inputs",
            str(output),
            "--output",
            str(tmp_path / "aggregate.csv"),
        ]
    )
    with pytest.raises(RuntimeError, match="refuses incomplete sweep"):
        aggregate_args.func(aggregate_args)


def test_derive_config_applies_named_overrides_and_rebases_assets(repo_root, tmp_path):
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "capacity": {
                    "config_values": {
                        "rsus.0.descriptor_capacity": 1,
                        "parameter_sources.resource_capacity": "stress_test_boundary",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "derived" / "config.json"
    source = repo_root / "configs" / "default.json"
    source_hash = sha256_file(source)
    args = build_parser().parse_args(
        [
            "derive-config",
            "--config",
            str(source),
            "--overrides",
            str(overrides),
            "--section",
            "capacity.config_values",
            "--output",
            str(output),
        ]
    )

    assert args.func is command_derive_config
    assert args.func(args) == 0
    derived = load_config(output)
    original = load_config(source)
    assert derived.rsus[0].descriptor_capacity == 1
    assert derived.parameter_sources["resource_capacity"] == "stress_test_boundary"
    assert derived.profile_path == original.profile_path
    assert derived.trace_path == original.trace_path
    assert derived.scenario_trace_path == original.scenario_trace_path
    assert sha256_file(source) == source_hash


def test_derive_config_rejects_frozen_asset_path_override(repo_root, tmp_path):
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps({"block": {"profile_path": "different-profile.json"}}),
        encoding="utf-8",
    )
    output = tmp_path / "derived.json"
    args = build_parser().parse_args(
        [
            "derive-config",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--overrides",
            str(overrides),
            "--section",
            "block",
            "--output",
            str(output),
        ]
    )

    with pytest.raises(ValueError, match="cannot override frozen asset paths"):
        args.func(args)
    assert not output.exists()


def test_derive_config_cannot_replace_override_document(repo_root, tmp_path):
    overrides = tmp_path / "overrides.json"
    original = json.dumps({"block": {"controller.lyapunov_v": 7.0}})
    overrides.write_text(original, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "derive-config",
            "--config",
            str(repo_root / "configs" / "default.json"),
            "--overrides",
            str(overrides),
            "--section",
            "block",
            "--output",
            str(overrides),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="cannot replace"):
        args.func(args)
    assert overrides.read_text(encoding="utf-8") == original


def test_generate_numerical_cli_maps_paper_v1_generator_controls(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_generate_numerical_study(output_root, *, spec, overwrite):
        captured["output_root"] = output_root
        captured["spec"] = spec
        captured["overwrite"] = overwrite
        return SimpleNamespace(
            config_path=tmp_path / "config.json",
            profile_path=tmp_path / "profile.json",
            evaluation_trace_path=tmp_path / "evaluation.json",
            scenario_trace_path=tmp_path / "scenario.json",
            evidence_path=tmp_path / "evidence.json",
            profile_hash="profile",
            evaluation_trace_hash="evaluation",
            scenario_trace_hash="scenario",
            evidence_hash="evidence",
        )

    monkeypatch.setattr(
        "privacy_edge_sim.cli.generate_numerical_study", fake_generate_numerical_study
    )
    args = build_parser().parse_args(
        [
            "generate-numerical-study",
            "--output-root",
            str(tmp_path / "study"),
            "--arrival-center-s",
            "6",
            "--arrival-window-s",
            "0.6",
            "--arrival-jitter-fraction",
            "0.15",
            "--preprocessing-failure-mode",
            "bernoulli",
            "--preprocessing-failure-count",
            "3",
            "--preprocessing-failure-probability",
            "0.2",
            "--local-service-scale",
            "1.5",
        ]
    )

    assert args.func(args) == 0
    spec = captured["spec"]
    assert spec.arrival_center_s == 6.0
    assert spec.arrival_window_s == 0.6
    assert spec.arrival_jitter_fraction == 0.15
    assert spec.preprocessing_failure_mode == "bernoulli"
    assert spec.preprocessing_failure_count == 3
    assert spec.preprocessing_failure_probability == 0.2
    assert spec.local_service_scale == 1.5


def test_numerical_study_requires_clean_source_unless_explicitly_overridden():
    parser = build_parser()
    common = [
        "run-numerical-study",
        "--base-study-root",
        "study",
        "--environment-seeds",
        "1,2",
        "--output-root",
        "results",
    ]

    assert parser.parse_args(common).allow_dirty_source is False
    assert parser.parse_args([*common, "--allow-dirty-source"]).allow_dirty_source is True


def test_numerical_study_parser_registers_analysis_family():
    args = build_parser().parse_args(
        [
            "run-numerical-study",
            "--base-study-root",
            "study",
            "--environment-seeds",
            "1,2",
            "--output-root",
            "results",
            "--registration-family",
            "primary",
        ]
    )
    assert args.registration_family == "primary"


def test_formal_sweep_requires_complete_registration_tuple(config):
    with pytest.raises(ValueError, match="formal sweeps require"):
        _sweep_experiment_registration(
            sensitivity_path=None,
            registration_factor=None,
            experiment_path=None,
            config_document={},
            config=config,
            grid_document={},
            policy=config.controller.policy,
            require_clean=True,
        )


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


def test_study_registration_binds_generator_and_analysis_domain(
    numerical_cli_bundle, tmp_path
):
    paths, config, _, _, _ = numerical_cli_bundle
    evidence = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    spec = evidence["spec"]
    registration = {
        "scientific_controls": {
            "profile_seed": spec["seed"],
            "profile_subjects": spec["profile_evaluation_subjects"],
            "test_subjects": spec["test_subjects"],
            "scenario_subjects": spec["scenario_subjects"],
            "arrival_jitter_fraction": spec["arrival_jitter_fraction"],
            "privacy_risk_threshold": spec["privacy_threshold"],
            "preprocessing_failure_mode": spec["preprocessing_failure_mode"],
            "anon_time_variability_scale": spec["anon_time_variability_scale"],
            "output_size_variability_scale": spec[
                "output_size_variability_scale"
            ],
        },
        "scales": {
            "formal": {
                "environment_seeds": [1, 2],
                "shared_generator_options": {
                    "tasks": spec["task_count"],
                    "horizon_s": spec["horizon_s"],
                    "arrival_center_s": spec["arrival_center_s"],
                },
                "regimes": {
                    "default": {
                        "arrival_window_s": spec["arrival_window_s"],
                        "local_service_scale": spec["local_service_scale"],
                        "config_overrides": {},
                    }
                },
            }
        },
        "analysis_plan": {
            "baseline": "all_local",
            "primary_family_id": "registered-primary",
            "primary_policies": ["all_local", "esl_smpc"],
            "primary_metrics": ["all_task_loss"],
        },
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    common = dict(
        registration_path=str(registration_path),
        registration_scale="formal",
        registration_regime="default",
        registration_family="primary",
        base_study_root=paths.config_path.parent.parent,
        environment_seeds=(1, 2),
        load_level="default",
        family_id="registered-primary",
        policies=("all_local", "esl_smpc"),
        baseline="all_local",
        metrics=("all_task_loss",),
        require_clean=False,
    )

    record = _study_experiment_registration(**common)
    assert record["status"] == "VERIFIED"
    assert record["registration_family"] == "primary"
    assert record["registered_policies"] == ["all_local", "esl_smpc"]
    with pytest.raises(ValueError, match="analysis domain differs"):
        _study_experiment_registration(
            **{**common, "metrics": ("failure_rate",)}
        )


def test_sweep_registration_binds_factor_reference_and_environment(
    numerical_cli_bundle, tmp_path
):
    paths, config, _, _, _ = numerical_cli_bundle
    config_document = json.loads(paths.config_path.read_text(encoding="utf-8"))
    evidence = json.loads(config.evidence_path.read_text(encoding="utf-8"))
    spec = evidence["spec"]
    trace = json.loads(config.trace_path.read_text(encoding="utf-8"))
    experiment = {
        "scientific_controls": {
            "profile_seed": spec["seed"],
            "profile_subjects": spec["profile_evaluation_subjects"],
            "test_subjects": spec["test_subjects"],
            "scenario_subjects": spec["scenario_subjects"],
            "arrival_jitter_fraction": spec["arrival_jitter_fraction"],
            "privacy_risk_threshold": spec["privacy_threshold"],
            "preprocessing_failure_mode": spec["preprocessing_failure_mode"],
            "anon_time_variability_scale": spec["anon_time_variability_scale"],
            "output_size_variability_scale": spec[
                "output_size_variability_scale"
            ],
        },
        "scales": {
            "formal": {
                "shared_generator_options": {
                    "tasks": spec["task_count"],
                    "horizon_s": spec["horizon_s"],
                    "arrival_center_s": spec["arrival_center_s"],
                },
                "regimes": {
                    "burst": {
                        "arrival_window_s": spec["arrival_window_s"],
                        "local_service_scale": spec["local_service_scale"],
                    }
                },
            }
        },
    }
    sensitivity = {
        "reference": {
            "scale": "formal",
            "regime": "burst",
            "experiment_registration_content_sha256": hashlib.sha256(
                canonical_json_bytes(experiment)
            ).hexdigest(),
            "privacy_risk_threshold": config_document["privacy"]["risk_threshold"],
            "controller.lyapunov_v": config_document["controller"]["lyapunov_v"],
            "controller.horizon_events": config_document["controller"][
                "horizon_events"
            ],
            "controller.scenarios": config_document["controller"]["scenarios"],
            "rsu_workload_capacity_gpu_s": [30.0, 18.0],
        },
        "rules": {"environment_seeds": [trace["seed"]]},
        "factors": {
            "lyapunov_tradeoff": {
                "application": "config_sweep",
                "path": "controller.lyapunov_v",
                "values": [6.0, 12.0],
                "policy": "esl_smpc",
            }
        },
    }
    experiment_path = tmp_path / "experiment.json"
    sensitivity_path = tmp_path / "sensitivity.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    sensitivity_path.write_text(json.dumps(sensitivity), encoding="utf-8")
    common = dict(
        sensitivity_path=str(sensitivity_path),
        registration_factor="lyapunov_tradeoff",
        experiment_path=str(experiment_path),
        config_document=config_document,
        config=config,
        grid_document={"controller.lyapunov_v": [6.0, 12.0]},
        policy="esl_smpc",
        require_clean=False,
    )

    record = _sweep_experiment_registration(**common)
    assert record["status"] == "VERIFIED"
    assert record["registered_policy"] == "esl_smpc"
    assert record["environment_seed"] == trace["seed"]
    with pytest.raises(ValueError, match="policy differs"):
        _sweep_experiment_registration(**{**common, "policy": "all_local"})
    with pytest.raises(ValueError, match="differs from the registered"):
        _sweep_experiment_registration(
            **{
                **common,
                "grid_document": {"controller.lyapunov_v": [3.0, 12.0]},
            }
        )
    sensitivity["factors"]["rsu_admission_concurrency"] = {
        "application": "paired_config_override",
        "paths": [
            "rsus.0.workload_capacity_gpu_s",
            "rsus.1.workload_capacity_gpu_s",
        ],
        "paired_values_gpu_s": [[0.03, 0.03], [30.0, 18.0]],
        "policy": "esl_smpc",
    }
    sensitivity_path.write_text(json.dumps(sensitivity), encoding="utf-8")
    paired = _sweep_experiment_registration(
        **{
            **common,
            "registration_factor": "rsu_admission_concurrency",
            "grid_document": {
                "$paired": [
                    {
                        "rsus.0.workload_capacity_gpu_s": 0.03,
                        "rsus.1.workload_capacity_gpu_s": 0.03,
                    },
                    {
                        "rsus.0.workload_capacity_gpu_s": 30.0,
                        "rsus.1.workload_capacity_gpu_s": 18.0,
                    },
                ]
            },
        }
    )
    assert paired["factor_paths"] == [
        "rsus.0.workload_capacity_gpu_s",
        "rsus.1.workload_capacity_gpu_s",
    ]


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
    frozen_assets = manifest["frozen_input_assets"]
    assert frozen_assets["hash_semantics"] == (
        "exact_raw_file_bytes_at_frozen_load_boundary"
    )
    expected_assets = {
        "profile": config.profile_path,
        "evaluation_trace": config.trace_path,
        "scenario_trace": config.scenario_trace_path,
        "evidence": config.evidence_path,
    }
    for role, path in expected_assets.items():
        record = frozen_assets["assets"][role]
        assert record["status"] == "captured"
        assert record["path"] == str(path.resolve())
        assert record["raw_sha256"] == sha256_file(path)
        assert record["size_bytes"] == path.stat().st_size
    assert list(manifest["trace_checksums"][0]["files"].values()) == [
        frozen_assets["assets"]["evaluation_trace"]["raw_sha256"]
    ]
    assert list(manifest["trace_checksums"][1]["files"].values()) == [
        frozen_assets["assets"]["scenario_trace"]["raw_sha256"]
    ]
    assert evidence["verified"] is True
    assert (
        evidence["file_sha256"]
        == frozen_assets["assets"]["evidence"]["raw_sha256"]
    )
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
    assert summary["edge_done_rate"] == 0.0
    assert summary["pipeline_attempt_rate"] == 0.0
    assert summary["pipeline_to_edge_rate"] == 0.0
    assert summary["pipeline_to_local_rate"] == 0.0
    assert summary["mechanism_path_counts"]["task_count"] == summary["task_count"]
    assert summary["mechanism_path_denominators"] == {
        "edge_done_rate": summary["task_count"],
        "pipeline_attempt_rate": summary["task_count"],
        "pipeline_to_edge_rate": 0,
        "pipeline_to_local_rate": 0,
    }

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
                allow_unverified_inputs=True,
                overwrite=False,
            )
        )
        == 0
    )
    hard_mask = json.loads(hard_mask_output.read_text(encoding="utf-8"))
    assert hard_mask["hard_mask_bypassed"] is False
    assert hard_mask["unsafe_actions_executed_by_audit"] == 0
    assert hard_mask["input_artifact_verification"]["status"] == (
        "UNVERIFIED_DEVELOPMENT_OVERRIDE"
    )

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


@pytest.mark.parametrize(
    "asset_attribute",
    ("profile_path", "trace_path", "scenario_trace_path", "evidence_path"),
)
def test_execute_rejects_frozen_input_change_during_simulation_before_outputs(
    numerical_cli_bundle,
    tmp_path,
    monkeypatch,
    asset_attribute,
):
    _, config, _, _, _ = numerical_cli_bundle
    asset_path = getattr(config, asset_attribute)
    assert asset_path is not None
    original_bytes = asset_path.read_bytes()
    original_run = DiscreteEventSimulator.run

    def run_then_mutate(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        asset_path.write_bytes(original_bytes + b"\n")
        return result

    monkeypatch.setattr(DiscreteEventSimulator, "run", run_then_mutate)
    output = tmp_path / asset_attribute
    try:
        with pytest.raises(RuntimeError, match="frozen input bytes changed"):
            _execute(config, "all_local", output, overwrite=False)
    finally:
        asset_path.write_bytes(original_bytes)

    assert not (output / "manifest.json").exists()
    assert not (output / "summary.json").exists()


def test_execute_rejects_frozen_input_change_while_loading(
    numerical_cli_bundle, tmp_path, monkeypatch
):
    _, config, _, _, _ = numerical_cli_bundle
    asset_path = config.scenario_trace_path
    original_bytes = asset_path.read_bytes()
    original_verify = cli_module.verify_run_evidence

    def verify_then_mutate(*args, **kwargs):
        verification = original_verify(*args, **kwargs)
        asset_path.write_bytes(original_bytes + b"\n")
        return verification

    monkeypatch.setattr(cli_module, "verify_run_evidence", verify_then_mutate)
    output = tmp_path / "load-boundary"
    try:
        with pytest.raises(
            RuntimeError, match="while loading frozen inputs"
        ):
            _execute(config, "all_local", output, overwrite=False)
    finally:
        asset_path.write_bytes(original_bytes)

    assert not (output / "manifest.json").exists()
    assert not (output / "summary.json").exists()


def test_execute_rechecks_frozen_inputs_immediately_before_manifest_publication(
    numerical_cli_bundle, tmp_path, monkeypatch
):
    _, config, _, _, _ = numerical_cli_bundle
    asset_path = config.profile_path
    original_bytes = asset_path.read_bytes()
    original_build_manifest = cli_module.build_manifest

    def build_then_mutate(*args, **kwargs):
        manifest = original_build_manifest(*args, **kwargs)
        asset_path.write_bytes(original_bytes + b"\n")
        return manifest

    monkeypatch.setattr(cli_module, "build_manifest", build_then_mutate)
    output = tmp_path / "manifest-boundary"
    try:
        with pytest.raises(
            RuntimeError, match="before manifest publication"
        ):
            _execute(config, "all_local", output, overwrite=False)
    finally:
        asset_path.write_bytes(original_bytes)

    assert (output / "summary.json").exists()
    assert not (output / "manifest.json").exists()


def test_numerical_study_rejects_base_bundle_change_between_replications(
    numerical_cli_bundle, tmp_path, monkeypatch
):
    paths, config, _, _, _ = numerical_cli_bundle
    profile_path = config.profile_path
    original_bytes = profile_path.read_bytes()
    original_generate = cli_module.generate_numerical_replication

    def generate_then_mutate(*args, **kwargs):
        replication = original_generate(*args, **kwargs)
        profile_path.write_bytes(original_bytes + b"\n")
        return replication

    monkeypatch.setattr(
        cli_module, "generate_numerical_replication", generate_then_mutate
    )
    args = build_parser().parse_args(
        [
            "run-numerical-study",
            "--base-study-root",
            str(paths.config_path.parent.parent),
            "--environment-seeds",
            "501,502",
            "--policies",
            "all_local,safe_lyapunov_h1",
            "--baseline",
            "all_local",
            "--output-root",
            str(tmp_path / "study-batch-toctou"),
            "--allow-dirty-source",
        ]
    )
    try:
        with pytest.raises(RuntimeError, match="frozen input bytes changed"):
            args.func(args)
    finally:
        profile_path.write_bytes(original_bytes)


def test_numerical_study_rejects_replication_config_drift(
    numerical_cli_bundle, tmp_path, monkeypatch
):
    paths, _, _, _, _ = numerical_cli_bundle
    original_generate = cli_module.generate_numerical_replication

    def generate_then_tamper(*args, **kwargs):
        replication = original_generate(*args, **kwargs)
        document = json.loads(replication.config_path.read_text(encoding="utf-8"))
        document["controller"]["lyapunov_v"] += 1.0
        replication.config_path.write_text(json.dumps(document), encoding="utf-8")
        return replication

    monkeypatch.setattr(
        cli_module, "generate_numerical_replication", generate_then_tamper
    )
    args = build_parser().parse_args(
        [
            "run-numerical-study",
            "--base-study-root",
            str(paths.config_path.parent.parent),
            "--environment-seeds",
            "503,504",
            "--policies",
            "all_local,safe_lyapunov_h1",
            "--baseline",
            "all_local",
            "--output-root",
            str(tmp_path / "study-config-drift"),
            "--allow-dirty-source",
        ]
    )

    with pytest.raises(RuntimeError, match="outside the registered RNG streams"):
        args.func(args)


def test_sweep_rejects_base_asset_change_between_cases(
    numerical_cli_bundle, tmp_path, monkeypatch
):
    paths, config, _, _, _ = numerical_cli_bundle
    scenario_path = config.scenario_trace_path
    original_bytes = scenario_path.read_bytes()
    grid_path = tmp_path / "batch-grid.json"
    grid_path.write_text(
        json.dumps({"controller.lyapunov_v": [1.0, 2.0]}), encoding="utf-8"
    )
    original_execute = cli_module._execute

    def execute_then_mutate(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        scenario_path.write_bytes(original_bytes + b"\n")
        return result

    monkeypatch.setattr(cli_module, "_execute", execute_then_mutate)
    args = build_parser().parse_args(
        [
            "sweep",
            "--config",
            str(paths.config_path),
            "--policy",
            "all_local",
            "--grid",
            str(grid_path),
            "--output-root",
            str(tmp_path / "sweep-batch-toctou"),
            "--allow-dirty-source",
        ]
    )
    try:
        with pytest.raises(RuntimeError, match="frozen input bytes changed"):
            args.func(args)
    finally:
        scenario_path.write_bytes(original_bytes)


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


def _write_completed_sweep_fixture(root, *, case_count: int = 2):
    rows = []
    for index in range(case_count):
        case_directory = root / f"case-{index:04d}"
        _write_aggregate_fixture(case_directory)
        manifest_path = case_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("manifest_sha256")
        parameters = {"controller.lyapunov_v": float(index + 1)}
        manifest["core_digest"] = f"{index + 1:064x}"
        manifest["run_metadata"].update(
            case=index,
            parameters=parameters,
            policy="all_local",
        )
        manifest["manifest_sha256"] = canonical_document_sha256(
            manifest, "manifest_sha256"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "case": index,
                "parameters": parameters,
                "policy": "all_local",
                "core_digest": manifest["core_digest"],
                "output_relative": case_directory.name,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    (root / "sweep.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostics = {
        "schema_version": "1.0",
        "case_count": case_count,
        "sweep_rows_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "sensitivity_registration": {
            "status": "UNREGISTERED_DEVELOPMENT_OVERRIDE",
            "require_clean_registration": False,
        },
    }
    diagnostics["report_sha256"] = canonical_document_sha256(
        diagnostics, "report_sha256"
    )
    (root / "sweep_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def test_aggregate_rejects_completed_sweep_with_deleted_case(tmp_path):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    shutil.rmtree(sweep / "case-0001")
    output = tmp_path / "aggregate" / "result"

    with pytest.raises(ValueError, match="completed sweep integrity.*case directories"):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(sweep)],
                output=str(output),
                overwrite=False,
                parquet=False,
            )
        )
    assert not output.with_suffix(".json").exists()


def test_aggregate_rejects_completed_sweep_with_deleted_index_from_case_input(
    tmp_path,
):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    (sweep / "sweep_diagnostics.json").unlink()
    output = tmp_path / "aggregate" / "result"

    with pytest.raises(FileNotFoundError, match="completed sweep integrity"):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(sweep / "case-0000")],
                output=str(output),
                overwrite=False,
                parquet=False,
            )
        )
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize(
    ("tamper_target", "message"),
    [
        ("rows", "sweep rows checksum mismatch"),
        ("diagnostics", "diagnostics self-hash mismatch"),
    ],
)
def test_aggregate_rejects_completed_sweep_index_tampering(
    tmp_path, tamper_target, message
):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    if tamper_target == "rows":
        path = sweep / "sweep.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["policy"] = "safe_greedy"
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        path = sweep / "sweep_diagnostics.json"
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
        diagnostics["case_count"] = 99
        path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(sweep)],
                output=str(tmp_path / "aggregate" / "result"),
                overwrite=False,
                parquet=False,
            )
        )


def test_aggregate_rejects_sweep_row_manifest_hash_mismatch(tmp_path):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    rows_path = sweep / "sweep.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[0]["manifest_sha256"] = "0" * 64
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    diagnostics_path = sweep / "sweep_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["sweep_rows_sha256"] = hashlib.sha256(
        canonical_json_bytes(rows)
    ).hexdigest()
    diagnostics["report_sha256"] = canonical_document_sha256(
        diagnostics, "report_sha256"
    )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(ValueError, match="row 0 manifest checksum mismatch"):
        command_aggregate(
            argparse.Namespace(
                inputs=[str(sweep)],
                output=str(tmp_path / "aggregate" / "result"),
                overwrite=False,
                parquet=False,
            )
        )


def test_failure_coverage_rejects_completed_sweep_with_deleted_case(tmp_path):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    shutil.rmtree(sweep / "case-0001")
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--runs",
            str(sweep / "case-0000"),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(ValueError, match="completed sweep integrity.*case directories"):
        args.func(args)
    assert not output.exists()


def test_failure_coverage_rejects_completed_sweep_with_deleted_index(tmp_path):
    sweep = tmp_path / "sweep"
    _write_completed_sweep_fixture(sweep)
    (sweep / "sweep.json").unlink()
    output = tmp_path / "coverage.json"
    args = build_parser().parse_args(
        [
            "audit-failure-coverage",
            "--study-roots",
            str(sweep),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(FileNotFoundError, match="completed sweep integrity"):
        args.func(args)
    assert not output.exists()


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
