"""Subject-cluster uncertainty reports for frozen privacy and FER evidence."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .profiles import canonical_json_bytes
from .statistics import SubjectCluster, subject_cluster_bootstrap


def _mean_field(clusters: Sequence[SubjectCluster], field: str) -> float:
    values = [float(row[field]) for cluster in clusters for row in cluster]
    return sum(values) / len(values)


def _mean_paired_delta(
    clusters: Sequence[SubjectCluster], left: str, right: str
) -> float:
    values = [
        float(row[left]) - float(row[right]) for cluster in clusters for row in cluster
    ]
    return sum(values) / len(values)


def _conditional_privacy(clusters: Sequence[SubjectCluster]) -> float:
    rows = [row for cluster in clusters for row in cluster]
    arrival_risk = sum(float(row["arrival_risk"]) for row in rows) / len(rows)
    emission = sum(float(row["emission"]) for row in rows) / len(rows)
    return min(1.0, arrival_risk / emission) if emission > 0.0 else 1.0


def _cell_subjects(
    evidence: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the stable m_i,g>0 subject order for a privacy cell."""

    try:
        quality_cells = evidence["quality_conformal"][
            "profile_evaluation_quality_support"
        ]["cells"]
    except (KeyError, TypeError):
        return tuple(
            str(value)
            for value in evidence["split_manifest"]["splits"]["profile_evaluation"][
                "subject_ids"
            ]
        )
    quality_bin = str(cell["quality_bin"])
    match = [row for row in quality_cells if str(row["region_id"]) == quality_bin]
    if len(match) != 1:
        raise ValueError("privacy cell lacks unique observed quality support")
    return tuple(str(row["subject_id"]) for row in match[0]["subject_frames"])


def build_subject_cluster_evidence_report(
    evidence: Mapping[str, Any],
    *,
    statistical_seed: int,
    resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap privacy and FER evidence using subjects, never frames, as IID units."""

    privacy_reports: list[dict[str, Any]] = []
    for index, cell in enumerate(evidence["privacy_evidence"]):
        profile_subjects = _cell_subjects(evidence, cell)
        subject_rows = cell["subject_rows"]
        if len(subject_rows) != len(profile_subjects):
            raise ValueError(
                "privacy subject rows do not match profile_evaluation split"
            )
        rows = [
            {
                "subject_id": subject_id,
                "arrival_risk": float(values[0]),
                "emission": float(values[1]),
            }
            for subject_id, values in zip(profile_subjects, subject_rows, strict=True)
        ]
        identity = {
            key: cell[key]
            for key in (
                "pipeline_id",
                "quality_bin",
                "attacker_id",
                "risk_type",
                "threshold_id",
            )
        }
        seed = statistical_seed + index * 2
        privacy_reports.append(
            {
                **identity,
                "arrival_risk": subject_cluster_bootstrap(
                    rows,
                    subject_key="subject_id",
                    statistic=lambda clusters: _mean_field(clusters, "arrival_risk"),
                    statistic_name="privacy_arrival_risk",
                    statistical_seed=seed,
                    resamples=resamples,
                    confidence_level=confidence_level,
                ),
                "conditional_risk": subject_cluster_bootstrap(
                    rows,
                    subject_key="subject_id",
                    statistic=_conditional_privacy,
                    statistic_name="privacy_conditional_risk",
                    statistical_seed=seed + 1,
                    resamples=resamples,
                    confidence_level=confidence_level,
                ),
            }
        )

    fer_rows = [dict(row) for row in evidence["fer_paired_records"]]
    required = {
        "subject_id",
        "local_nll",
        "anonymous_edge_nll",
        "local_correct",
        "anonymous_edge_correct",
    }
    if not fer_rows or any(not required.issubset(row) for row in fer_rows):
        raise ValueError(
            "FER paired records require subject, NLL and correctness for both paths"
        )
    fer_statistics = {}
    for offset, field in enumerate(
        (
            "local_nll",
            "anonymous_edge_nll",
            "local_correct",
            "anonymous_edge_correct",
        )
    ):
        statistic_name = {
            "local_nll": "fer_local_nll",
            "anonymous_edge_nll": "fer_anonymous_edge_nll",
            "local_correct": "fer_local_accuracy",
            "anonymous_edge_correct": "fer_anonymous_edge_accuracy",
        }[field]
        fer_statistics[statistic_name] = subject_cluster_bootstrap(
            fer_rows,
            subject_key="subject_id",
            statistic=lambda clusters, field=field: _mean_field(clusters, field),
            statistic_name=statistic_name,
            statistical_seed=statistical_seed + 100_000 + offset,
            resamples=resamples,
            confidence_level=confidence_level,
        )
    paired_fields = (
        (
            "paired_nll_delta",
            "anonymous_edge_nll",
            "local_nll",
            "anonymous_edge_minus_local_nll",
        ),
        (
            "paired_accuracy_delta",
            "anonymous_edge_correct",
            "local_correct",
            "anonymous_edge_minus_local_accuracy",
        ),
    )
    for offset, (name, left, right, definition) in enumerate(paired_fields):
        result = subject_cluster_bootstrap(
            fer_rows,
            subject_key="subject_id",
            statistic=lambda clusters, left=left, right=right: _mean_paired_delta(
                clusters, left, right
            ),
            statistic_name=f"fer_{name}",
            statistical_seed=statistical_seed + 200_000 + offset,
            resamples=resamples,
            confidence_level=confidence_level,
        )
        result["difference_definition"] = definition
        fer_statistics[name] = result
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "frozen_evidence_subject_cluster_bootstrap",
        "independent_unit": "subject",
        "evidence_hash": evidence["evidence_hash"],
        "statistical_seed": statistical_seed,
        "resamples": resamples,
        "confidence_level": confidence_level,
        "privacy": privacy_reports,
        "fer": fer_statistics,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


__all__ = ["build_subject_cluster_evidence_report"]
