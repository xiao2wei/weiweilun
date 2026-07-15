from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import privacy_edge_sim.cli as cli_module
from privacy_edge_sim.cli import build_parser, command_analyze_sensitivity
from privacy_edge_sim.sensitivity_analysis import (
    SensitivityAnalysisError,
    analyze_registered_sensitivity_sweeps,
)


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_document(document, field):
    material = dict(document)
    material.pop(field, None)
    return hashlib.sha256(_canonical(material)).hexdigest()


def _clean_preflight(*, source_object: str, scope: str):
    return {
        "require_clean_source": True,
        "requirement_status": "passed",
        "assessment": "source_clean",
        "git_commit": "c" * 40,
        "source_git_dirty": False,
        "source_commit_reproducible": True,
        "source_git_object": source_object,
        "source_scope": scope,
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, include_unrun_factor: bool = False):
    plan = {
        "schema_version": "paper-v1-sensitivity-v1",
        "reference": {
            "scale": "formal",
            "regime": "burst",
            "experiment_registration_content_sha256": "f" * 64,
            "controller.lyapunov_v": 2.0,
        },
        "rules": {
            "confirmatory": False,
            "environment_seeds": [401, 402],
        },
        "factors": {
            "lyapunov_tradeoff": {
                "application": "config_sweep",
                "path": "controller.lyapunov_v",
                "values": [1.0, 2.0],
                "reference_value": 2.0,
                "policy": "esl_smpc",
            }
        },
    }
    if include_unrun_factor:
        plan["factors"]["scenario_count"] = {
            "application": "config_sweep",
            "path": "controller.scenarios",
            "values": [4, 8],
            "reference_value": 8,
            "policy": "esl_smpc",
        }
    plan_path = tmp_path / "sensitivity.json"
    _write_json(plan_path, plan)
    plan_file_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_content_hash = hashlib.sha256(_canonical(plan)).hexdigest()
    package_preflight = _clean_preflight(
        source_object="a" * 40, scope="src/privacy_edge_sim"
    )
    plan_preflight = _clean_preflight(
        source_object="b" * 40, scope="sensitivity.json"
    )
    experiment_preflight = _clean_preflight(
        source_object="d" * 40, scope="experiment.json"
    )
    roots = []
    for environment_seed in (401, 402):
        root = tmp_path / f"env-{environment_seed}" / "lyapunov_tradeoff"
        roots.append(root)
        registration = {
            "status": "VERIFIED",
            "registration_factor": "lyapunov_tradeoff",
            "factor_paths": ["controller.lyapunov_v"],
            "factor_values": [1.0, 2.0],
            "reference_scale": "formal",
            "reference_regime": "burst",
            "environment_seed": environment_seed,
            "sensitivity_file_sha256": plan_file_hash,
            "sensitivity_content_sha256": plan_content_hash,
            "experiment_file_sha256": "e" * 64,
            "experiment_content_sha256": "f" * 64,
            "sensitivity_source_preflight": plan_preflight,
            "experiment_source_preflight": experiment_preflight,
            "record_sha256": "",
        }
        registration["record_sha256"] = _hash_document(
            registration, "record_sha256"
        )
        rows = []
        trace_hash = hashlib.sha256(f"trace-{environment_seed}".encode()).hexdigest()
        scenario_hash = "9" * 64
        for index, level in enumerate((1.0, 2.0)):
            case = root / f"case-{index:04d}"
            summary = {
                "all_task_loss": level + (environment_seed - 400) / 10.0,
            }
            _write_json(case / "summary.json", summary)
            summary_hash = hashlib.sha256(
                (case / "summary.json").read_bytes()
            ).hexdigest()
            parameters = {"controller.lyapunov_v": level}
            manifest = {
                "core_digest": f"{environment_seed + index:064x}",
                "source_cleanliness_preflight": package_preflight,
                "trace_identity": {
                    "trace_hash": trace_hash,
                    "seed": environment_seed,
                },
                "scenario_trace_identity": {"trace_hash": scenario_hash},
                "frozen_input_assets": {
                    "schema_version": "1.0",
                    "assets": {
                        "profile": {
                            "status": "captured",
                            "raw_sha256": "1" * 64,
                            "declared_content_hash": "2" * 64,
                        },
                        "scenario_trace": {
                            "status": "captured",
                            "raw_sha256": "3" * 64,
                            "declared_content_hash": scenario_hash,
                        },
                        "evidence": {
                            "status": "captured",
                            "raw_sha256": "4" * 64,
                            "declared_content_hash": "5" * 64,
                        },
                    },
                },
                "run_metadata": {
                    "case": index,
                    "parameters": parameters,
                    "policy": "esl_smpc",
                    "sensitivity_registration_record_sha256": registration[
                        "record_sha256"
                    ],
                },
                "outputs": {
                    "files": {
                        "summary.json": {
                            "sha256": summary_hash,
                            "size_bytes": (case / "summary.json").stat().st_size,
                        }
                    }
                },
                "manifest_sha256": "",
            }
            manifest["manifest_sha256"] = _hash_document(
                manifest, "manifest_sha256"
            )
            _write_json(case / "manifest.json", manifest)
            rows.append(
                {
                    "case": index,
                    "parameters": parameters,
                    "policy": "esl_smpc",
                    "core_digest": manifest["core_digest"],
                    "output_relative": case.name,
                    "manifest_sha256": manifest["manifest_sha256"],
                }
            )
        _write_json(root / "sweep.json", rows)
        diagnostics = {
            "schema_version": "1.0",
            "case_count": 2,
            "sweep_rows_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
            "sensitivity_registration": registration,
            "report_sha256": "",
        }
        diagnostics["report_sha256"] = _hash_document(
            diagnostics, "report_sha256"
        )
        _write_json(root / "sweep_diagnostics.json", diagnostics)
    return roots, plan_path, package_preflight, plan_preflight


def _analyze(roots, plan_path, package_preflight, plan_preflight):
    return analyze_registered_sensitivity_sweeps(
        roots,
        sensitivity_path=plan_path,
        metric_name="all_task_loss",
        statistical_seed=17,
        bootstrap_resamples=20,
        sign_flip_permutations=20,
        analysis_source_preflight=package_preflight,
        sensitivity_source_preflight=plan_preflight,
    )


def test_sensitivity_analysis_pairs_levels_by_environment_and_self_hashes(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)

    report = _analyze(roots, plan_path, package_preflight, plan_preflight)

    assert report["study_role"] == "exploratory_non_confirmatory"
    assert report["independent_unit"] == "environment"
    assert report["environment_seeds"] == [401, 402]
    assert report["shared_frozen_inputs"]["profile"]["raw_sha256"] == "1" * 64
    factor = report["factors"]["lyapunov_tradeoff"]
    assert factor["environment_count"] == 2
    assert factor["reference_level_id"] == (
        'level:{"controller.lyapunov_v":2.0}'
    )
    analysis = factor["paired_analysis"]
    assert analysis["independent_unit"] == "environment"
    assert analysis["environment_count"] == 2
    assert report["report_sha256"] == _hash_document(report, "report_sha256")


def test_sensitivity_analysis_rejects_missing_registered_environment(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)

    with pytest.raises(SensitivityAnalysisError, match="environment set is incomplete"):
        _analyze(roots[:1], plan_path, package_preflight, plan_preflight)


def test_sensitivity_analysis_rejects_unpaired_level_trace(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)
    manifest_path = roots[0] / "case-0001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trace_identity"]["trace_hash"] = "7" * 64
    manifest["manifest_sha256"] = _hash_document(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    rows_path = roots[0] / "sweep.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[1]["manifest_sha256"] = manifest["manifest_sha256"]
    _write_json(rows_path, rows)

    with pytest.raises(SensitivityAnalysisError, match="not trace-paired"):
        _analyze(roots, plan_path, package_preflight, plan_preflight)


def test_sensitivity_analysis_rejects_dirty_case_provenance(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)
    manifest_path = roots[0] / "case-0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_cleanliness_preflight"]["requirement_status"] = "not_required"
    manifest["source_cleanliness_preflight"]["require_clean_source"] = False
    manifest["manifest_sha256"] = _hash_document(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    rows_path = roots[0] / "sweep.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[0]["manifest_sha256"] = manifest["manifest_sha256"]
    _write_json(rows_path, rows)

    with pytest.raises(SensitivityAnalysisError, match="verified clean source tree"):
        _analyze(roots, plan_path, package_preflight, plan_preflight)


def test_sensitivity_analysis_rejects_cross_environment_profile_change(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)
    manifest_path = roots[1] / "case-0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_input_assets"]["assets"]["profile"]["raw_sha256"] = (
        "8" * 64
    )
    manifest["manifest_sha256"] = _hash_document(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    rows_path = roots[1] / "sweep.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[0]["manifest_sha256"] = manifest["manifest_sha256"]
    _write_json(rows_path, rows)

    with pytest.raises(SensitivityAnalysisError, match="frozen profile identity"):
        _analyze(roots, plan_path, package_preflight, plan_preflight)


def test_sensitivity_analysis_rejects_duplicate_factor_environment(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)
    duplicate = tmp_path / "duplicate-env-401"
    shutil.copytree(roots[0], duplicate)

    with pytest.raises(SensitivityAnalysisError, match="duplicate sweep for factor"):
        _analyze(
            [*roots, duplicate], plan_path, package_preflight, plan_preflight
        )


def test_sensitivity_analysis_rejects_missing_registered_factor(tmp_path):
    roots, plan_path, package_preflight, plan_preflight = _fixture(
        tmp_path, include_unrun_factor=True
    )

    with pytest.raises(SensitivityAnalysisError, match="factor family is incomplete"):
        _analyze(roots, plan_path, package_preflight, plan_preflight)


def test_analyze_sensitivity_cli_is_registered():
    args = build_parser().parse_args(
        [
            "analyze-sensitivity",
            "--sweep-roots",
            "results/sensitivity",
            "--sensitivity-registration",
            "examples/paper-v1-sensitivity.json",
            "--output",
            "results/sensitivity-analysis.json",
        ]
    )

    assert args.func is command_analyze_sensitivity
    assert args.metric == "all_task_loss"
    assert args.statistics_seed == 92001


def test_analyze_sensitivity_cli_validates_completed_roots_and_writes_report(
    tmp_path, monkeypatch
):
    _roots, plan_path, package_preflight, plan_preflight = _fixture(tmp_path)
    output = tmp_path / "reports" / "sensitivity.json"

    def fake_preflight(source_root=None, *, require_clean=False):
        assert require_clean is True
        return plan_preflight if source_root is not None else package_preflight

    monkeypatch.setattr(cli_module, "source_cleanliness_preflight", fake_preflight)
    args = build_parser().parse_args(
        [
            "analyze-sensitivity",
            "--sweep-roots",
            str(tmp_path),
            "--sensitivity-registration",
            str(plan_path),
            "--bootstrap-resamples",
            "20",
            "--permutations",
            "20",
            "--output",
            str(output),
        ]
    )

    assert args.func(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["factor_count"] == 1
    assert report["environment_seeds"] == [401, 402]
