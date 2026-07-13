from __future__ import annotations

import copy

import pytest

from privacy_edge_sim.experiments import (
    build_numerical_experiment_report,
    build_post_selection_risk_table,
    build_subject_cluster_uncertainty,
    build_subject_count_sensitivity,
)
from privacy_edge_sim.profiles import canonical_document_sha256


@pytest.fixture
def frozen_evidence():
    subjects = [f"subject-{index}" for index in range(4)]
    stages = {
        "single_attempt": [[0.50, 1.00], [0.25, 1.00], [0.00, 1.00], [0.75, 1.00]],
        "guard_selected": [[0.25, 0.50], [0.00, 0.50], [0.00, 0.25], [0.50, 0.75]],
        "guard_plus_retry_final": [
            [0.50, 0.75],
            [0.25, 0.75],
            [0.00, 0.50],
            [0.50, 1.00],
        ],
    }
    evidence = {
        "evidence_hash": "evidence-sha",
        "split_manifest": {
            "manifest_hash": "split-sha",
            "splits": {"profile_evaluation": {"subject_ids": subjects}},
        },
        "privacy_protocol": {
            "registered_hypotheses": 1,
            "confidence_error": 0.05,
        },
        "privacy_evidence": [
            {
                "pipeline_id": "pipeline-a",
                "quality_bin": "clear",
                "attacker_id": "attacker-a",
                "risk_type": "identity",
                "threshold_id": "rank-1",
                "stage_subject_rows": stages,
                "subject_rows": stages["guard_plus_retry_final"],
            }
        ],
        "fer_paired_records": [
            {
                "subject_id": subject,
                "local_nll": 0.10 + index * 0.03,
                "anonymous_edge_nll": 0.15 + index * 0.02,
                "local_correct": index != 3,
                "anonymous_edge_correct": index in (0, 2),
            }
            for index, subject in enumerate(subjects)
        ],
    }
    return evidence


def test_post_selection_table_recomputes_arrival_conditional_and_coverage(
    frozen_evidence,
):
    table = build_post_selection_risk_table(frozen_evidence)
    assert [row["stage"] for row in table] == [
        "single_attempt",
        "guard_selected",
        "guard_plus_retry_final",
    ]
    selected = table[1]
    assert selected["arrival_risk"] == pytest.approx(0.1875)
    assert selected["emission_coverage"] == pytest.approx(0.5)
    assert selected["conditional_risk"] == pytest.approx(0.375)
    assert selected["arrival_risk_ucb"] >= selected["arrival_risk"]
    assert selected["conditional_risk_ucb"] >= selected["conditional_risk"]
    assert selected["emission_coverage_lcb"] <= selected["emission_coverage"]
    assert selected["independent_unit"] == "subject"


def test_subject_count_sensitivity_is_nested_seeded_and_deterministic(
    frozen_evidence,
):
    first = build_subject_count_sensitivity(
        frozen_evidence, subject_counts=(2, 4), statistical_seed=71
    )
    repeat = build_subject_count_sensitivity(
        copy.deepcopy(frozen_evidence), subject_counts=(4, 2), statistical_seed=71
    )
    changed = build_subject_count_sensitivity(
        frozen_evidence, subject_counts=(2, 4), statistical_seed=72
    )
    assert first == repeat
    assert first["registered_subject_counts"] == [2, 4]
    assert len(first["rows"]) == 2 * 3
    assert first["subsets"][0]["nested_prefix"] is True
    assert first["subsets"] != changed["subsets"]
    for row in first["rows"]:
        assert row["subject_count"] in (2, 4)
        assert row["joint_risk_ucb"] >= row["mean_emit_and_attack"]
        assert row["emission_lcb"] <= row["mean_emission"]
    assert first["report_sha256"] == canonical_document_sha256(first, "report_sha256")


def test_subject_count_sensitivity_rejects_non_preregisterable_sizes(
    frozen_evidence,
):
    with pytest.raises(ValueError, match="outside available support"):
        build_subject_count_sensitivity(
            frozen_evidence, subject_counts=(1, 4), statistical_seed=1
        )
    with pytest.raises(ValueError, match="unique"):
        build_subject_count_sensitivity(
            frozen_evidence, subject_counts=(2, 2), statistical_seed=1
        )


def test_subject_cluster_uncertainty_is_reproducible_for_privacy_and_fer(
    frozen_evidence,
):
    first = build_subject_cluster_uncertainty(
        frozen_evidence,
        statistical_seed=90,
        resamples=40,
        confidence_level=0.90,
    )
    repeat = build_subject_cluster_uncertainty(
        copy.deepcopy(frozen_evidence),
        statistical_seed=90,
        resamples=40,
        confidence_level=0.90,
    )
    assert first == repeat
    assert len(first["privacy"]) == 3
    assert set(first["fer"]) == {
        "local_nll",
        "anonymous_edge_nll",
        "local_accuracy",
        "anonymous_edge_accuracy",
        "paired_nll_delta",
        "paired_accuracy_delta",
    }
    for row in first["privacy"]:
        for field in ("arrival_risk", "emission_coverage", "conditional_risk"):
            assert row[field]["independent_unit"] == "subject"
            assert (
                row[field]["ci_lower"]
                <= row[field]["estimate"]
                <= row[field]["ci_upper"]
            )
    assert first["report_sha256"] == canonical_document_sha256(first, "report_sha256")
    expected_nll_delta = sum(
        row["anonymous_edge_nll"] - row["local_nll"]
        for row in frozen_evidence["fer_paired_records"]
    ) / len(frozen_evidence["fer_paired_records"])
    expected_accuracy_delta = sum(
        float(row["anonymous_edge_correct"]) - float(row["local_correct"])
        for row in frozen_evidence["fer_paired_records"]
    ) / len(frozen_evidence["fer_paired_records"])
    assert first["fer"]["paired_nll_delta"]["estimate"] == pytest.approx(
        expected_nll_delta
    )
    assert first["fer"]["paired_accuracy_delta"]["estimate"] == pytest.approx(
        expected_accuracy_delta
    )


def test_combined_report_has_hash_and_offline_provenance(frozen_evidence):
    report = build_numerical_experiment_report(
        frozen_evidence,
        subject_counts=(2, 4),
        statistical_seed=123,
        resamples=20,
        confidence_level=0.90,
    )
    assert report["evidence_hash"] == frozen_evidence["evidence_hash"]
    assert report["split_manifest_hash"] == "split-sha"
    assert report["provenance"]["independent_unit"] == "subject"
    assert report["provenance"]["online_data_used"] is False
    assert report["provenance"]["future_trace_used"] is False
    assert report["report_sha256"] == canonical_document_sha256(report, "report_sha256")
