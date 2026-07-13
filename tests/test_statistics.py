from __future__ import annotations

import argparse
import hashlib
import json
import random

import pytest

from privacy_edge_sim.cli import command_aggregate_statistical_families
from privacy_edge_sim.statistics import (
    StatisticalValidationError,
    aggregate_preregistered_study_families,
    analyze_paired_strategies,
    apply_holm_family_adjustment,
    holm_adjust,
    subject_cluster_bootstrap,
)


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    values = {
        "env-1": {
            "baseline": (10.0, 12.0),
            "candidate": (8.0, 9.0),
            "other": (9.0, 11.0),
        },
        "env-2": {
            "baseline": (15.0, 14.0),
            "candidate": (11.0, 12.0),
            "other": (14.0, 13.0),
        },
        "env-3": {
            "baseline": (20.0, 18.0),
            "candidate": (17.0, 16.0),
            "other": (19.0, 17.0),
        },
        "env-4": {
            "baseline": (9.0, 11.0),
            "candidate": (8.0, 8.0),
            "other": (8.5, 10.0),
        },
    }
    for environment_id, strategies in values.items():
        for strategy, metric_values in strategies.items():
            for index, metric_value in enumerate(metric_values):
                records.append(
                    {
                        "environment_id": environment_id,
                        "pairing_id": f"workload-{index}",
                        "strategy": strategy,
                        "metric_value": metric_value,
                        "evaluation_trace_hash": f"trace-{environment_id}",
                        "task_identity_hash": f"tasks-{environment_id}-{index}",
                    }
                )
    return records


def _self_hash(document, field):
    material = dict(document)
    material.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_study(load_level: str, family_id: str = "registered-load-family"):
    analyses = {
        metric: analyze_paired_strategies(
            _records(),
            baseline_strategy="baseline",
            metric_name=metric,
            statistical_seed=17,
            bootstrap_resamples=30,
        )
        for metric in ("latency_s", "energy_j")
    }
    analyses, local_family = apply_holm_family_adjustment(analyses)
    registration = {
        "schema_version": "1.0",
        "family_id": family_id,
        "load_level": load_level,
        "registered_metrics": ["latency_s", "energy_j"],
        "registered_policies": ["baseline", "candidate", "other"],
        "baseline": "baseline",
        "family_dimensions": ["load", "metric", "policy_vs_baseline"],
        "registration_sha256": "",
    }
    registration["registration_sha256"] = _self_hash(
        registration, "registration_sha256"
    )
    report = {
        "statistical_family_registration": registration,
        "analyses": analyses,
        "multiple_testing": local_family,
        "study_report_sha256": "",
    }
    report["study_report_sha256"] = _self_hash(report, "study_report_sha256")
    return report


def test_paired_analysis_is_reproducible_json_and_does_not_touch_global_rng():
    random.seed(713)
    before = random.getstate()
    first = analyze_paired_strategies(
        _records(),
        baseline_strategy="baseline",
        metric_name="latency_s",
        statistical_seed=41001,
        bootstrap_resamples=400,
    )
    after = random.getstate()
    second = analyze_paired_strategies(
        list(reversed(_records())),
        baseline_strategy="baseline",
        metric_name="latency_s",
        statistical_seed=41001,
        bootstrap_resamples=400,
    )

    assert before == after
    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["independent_unit"] == "environment"
    assert first["environment_count"] == 4
    assert first["pairing_count"] == 8
    comparison = first["comparisons"]["candidate__vs__baseline"]
    assert comparison["mean_environment_difference"] == pytest.approx(-2.5)
    assert comparison["bootstrap_ci"]["lower"] < 0
    assert comparison["bootstrap_ci"]["upper"] < 0
    assert 0 <= comparison["sign_flip_test"]["p_value"] <= 1
    assert 0 <= comparison["holm_adjusted_p_value"] <= 1
    manifest = first["statistics_manifest"]
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["result_core_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (
            lambda rows: rows.__setitem__(
                0, {**rows[0], "evaluation_trace_hash": "wrong-trace"}
            ),
            "STAT_ENVIRONMENT_TRACE_MISMATCH",
        ),
        (
            lambda rows: rows.__setitem__(
                0, {**rows[0], "task_identity_hash": "wrong-tasks"}
            ),
            "STAT_TASK_IDENTITY_MISMATCH",
        ),
        (lambda rows: rows.pop(0), "STAT_MISSING_PAIR"),
    ],
)
def test_paired_analysis_rejects_unpaired_trace_or_task_evidence(mutator, code):
    rows = _records()
    mutator(rows)
    with pytest.raises(StatisticalValidationError) as caught:
        analyze_paired_strategies(
            rows,
            baseline_strategy="baseline",
            metric_name="latency_s",
            statistical_seed=17,
            bootstrap_resamples=30,
        )
    assert caught.value.code == code


def test_paired_analysis_requires_multiple_independent_environments():
    rows = [row for row in _records() if row["environment_id"] == "env-1"]
    with pytest.raises(StatisticalValidationError) as caught:
        analyze_paired_strategies(
            rows,
            baseline_strategy="baseline",
            metric_name="latency_s",
            statistical_seed=3,
        )
    assert caught.value.code == "STAT_ENVIRONMENT_COUNT"


def test_holm_adjustment_is_monotone_and_familywise_bounded():
    adjusted = holm_adjust({"h1": 0.01, "h2": 0.03, "h3": 0.04})
    assert adjusted == pytest.approx({"h1": 0.03, "h2": 0.06, "h3": 0.06})


def test_holm_family_adjustment_spans_all_metrics_and_policy_comparisons():
    latency = analyze_paired_strategies(
        _records(),
        baseline_strategy="baseline",
        metric_name="latency_s",
        statistical_seed=17,
        bootstrap_resamples=30,
    )
    energy = analyze_paired_strategies(
        _records(),
        baseline_strategy="baseline",
        metric_name="energy_j",
        statistical_seed=17,
        bootstrap_resamples=30,
    )
    adjusted, family = apply_holm_family_adjustment(
        {"latency_s": latency, "energy_j": energy}
    )
    assert family["hypothesis_count"] == 4
    assert family["family_dimensions"] == ["metric", "policy_vs_baseline"]
    assert len(family["family_sha256"]) == 64
    for analysis in adjusted.values():
        for comparison in analysis["comparisons"].values():
            assert comparison["within_analysis_holm_adjusted_p_value"] is not None
            hypothesis = comparison["holm_family_hypothesis_id"]
            assert comparison["holm_adjusted_p_value"] == pytest.approx(
                family["adjusted_p_values"][hypothesis]
            )
        manifest = analysis["statistics_manifest"]
        core = {
            key: value
            for key, value in analysis.items()
            if key != "statistics_manifest"
        }
        assert (
            manifest["result_core_sha256"]
            == hashlib.sha256(
                json.dumps(
                    core,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        )


def test_aggregate_preregistered_load_family_applies_one_global_holm():
    low = _load_study("low")
    high = _load_study("high")
    report = aggregate_preregistered_study_families(
        [low, high],
        input_identities=[
            {"path": "low.json", "file_sha256": "1" * 64},
            {"path": "high.json", "file_sha256": "2" * 64},
        ],
    )
    assert report["family_dimensions"] == [
        "load",
        "metric",
        "policy_vs_baseline",
    ]
    assert report["load_levels"] == ["high", "low"]
    assert report["hypothesis_count"] == 8
    assert len({row["hypothesis_id"] for row in report["hypotheses"]}) == 8
    assert all(
        row["holm_adjusted_p_value"]
        == report["holm_adjusted_p_values"][row["hypothesis_id"]]
        for row in report["hypotheses"]
    )
    assert report["report_sha256"] == _self_hash(report, "report_sha256")


@pytest.mark.parametrize(
    ("mutator", "code"),
    (
        (
            lambda reports: reports[1]["statistical_family_registration"].__setitem__(
                "load_level", "low"
            ),
            "STAT_LOAD_DUPLICATE",
        ),
        (
            lambda reports: reports[1]["statistical_family_registration"].pop(
                "load_level"
            ),
            "STAT_FIELD",
        ),
        (
            lambda reports: reports[1]["statistical_family_registration"].__setitem__(
                "family_id", "different-family"
            ),
            "STAT_LOAD_FAMILY_MISMATCH",
        ),
        (
            lambda reports: reports[1]["multiple_testing"].__setitem__(
                "family_sha256", "0" * 64
            ),
            "STAT_LOAD_LOCAL_FAMILY_HASH",
        ),
    ),
)
def test_aggregate_load_family_rejects_duplicate_missing_or_inconsistent_inputs(
    mutator, code
):
    reports = [_load_study("low"), _load_study("high")]
    mutator(reports)
    # Re-hash semantic mutations so the test reaches the intended strict
    # family check rather than being stopped only by the outer checksum.
    for report in reports:
        registration = report["statistical_family_registration"]
        registration["registration_sha256"] = _self_hash(
            registration, "registration_sha256"
        )
        report["study_report_sha256"] = _self_hash(report, "study_report_sha256")
    with pytest.raises(StatisticalValidationError) as caught:
        aggregate_preregistered_study_families(reports)
    assert caught.value.code == code


def test_aggregate_load_family_rejects_invalid_outer_hash():
    reports = [_load_study("low"), _load_study("high")]
    reports[1]["study_report_sha256"] = "0" * 64
    with pytest.raises(StatisticalValidationError) as caught:
        aggregate_preregistered_study_families(reports)
    assert caught.value.code == "STAT_LOAD_STUDY_HASH"


def test_aggregate_statistical_families_cli_writes_self_hashed_report(tmp_path):
    paths = []
    for load_level in ("low", "high"):
        path = tmp_path / f"{load_level}.json"
        path.write_text(
            json.dumps(_load_study(load_level), sort_keys=True), encoding="utf-8"
        )
        paths.append(str(path))
    output = tmp_path / "aggregate.json"
    assert (
        command_aggregate_statistical_families(
            argparse.Namespace(
                study_reports=paths,
                output=str(output),
                overwrite=False,
            )
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["hypothesis_count"] == 8
    assert report["report_sha256"] == _self_hash(report, "report_sha256")
    assert all(len(row["file_sha256"]) == 64 for row in report["inputs"])


def test_subject_cluster_bootstrap_resamples_whole_subjects_reproducibly():
    rows = [
        {"subject_id": "s1", "attack": 1.0, "emit": 1.0},
        {"subject_id": "s1", "attack": 0.0, "emit": 1.0},
        {"subject_id": "s2", "attack": 0.0, "emit": 0.0},
        {"subject_id": "s3", "attack": 1.0, "emit": 1.0},
    ]

    def conditional_risk(clusters):
        # Each subject contributes one cluster-level X_i/Y_i pair; prolific
        # subjects therefore do not receive extra bootstrap sampling mass.
        xs = []
        ys = []
        for cluster in clusters:
            xs.append(sum(float(row["attack"]) for row in cluster) / len(cluster))
            ys.append(sum(float(row["emit"]) for row in cluster) / len(cluster))
        denominator = sum(ys)
        return 0.0 if denominator == 0 else sum(xs) / denominator

    first = subject_cluster_bootstrap(
        rows,
        subject_key="subject_id",
        statistic=conditional_risk,
        statistic_name="conditional_attack_risk",
        statistical_seed=99,
        resamples=300,
    )
    second = subject_cluster_bootstrap(
        list(reversed(rows)),
        subject_key="subject_id",
        statistic=conditional_risk,
        statistic_name="conditional_attack_risk",
        statistical_seed=99,
        resamples=300,
    )
    assert first == second
    assert first["subject_count"] == 3
    assert first["row_count"] == 4
    assert first["estimate"] == pytest.approx(0.75)
    assert first["ci_lower"] <= first["estimate"] <= first["ci_upper"]
    assert len(first["manifest_sha256"]) == 64
    json.dumps(first, allow_nan=False)


def test_subject_cluster_bootstrap_rejects_missing_or_too_few_subjects():
    with pytest.raises(StatisticalValidationError) as missing:
        subject_cluster_bootstrap(
            [{"value": 1.0}, {"subject_id": "s2", "value": 2.0}],
            subject_key="subject_id",
            statistic=lambda clusters: 0.0,
            statistic_name="mean",
            statistical_seed=1,
        )
    assert missing.value.code == "STAT_SUBJECT_KEY"

    with pytest.raises(StatisticalValidationError) as count:
        subject_cluster_bootstrap(
            [{"subject_id": "s1", "value": 1.0}],
            subject_key="subject_id",
            statistic=lambda clusters: 1.0,
            statistic_name="mean",
            statistical_seed=1,
        )
    assert count.value.code == "STAT_SUBJECT_COUNT"
