"""Offline numerical-evidence analyses with subjects as independent units.

The functions in this module are pure: they neither read traces nor mutate a
frozen evidence document.  They turn preregistered numerical evidence into
auditable post-selection, finite-support, and cluster-bootstrap reports.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .profiles import (
    canonical_document_sha256,
    canonical_json_bytes,
    compute_subject_risk_ucb,
)
from .statistics import SubjectCluster, subject_cluster_bootstrap


_STAGES = ("single_attempt", "guard_selected", "guard_plus_retry_final")
_IDENTITY_FIELDS = (
    "pipeline_id",
    "quality_bin",
    "attacker_id",
    "risk_type",
    "threshold_id",
)


def _profile_subjects(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        subjects = tuple(
            str(value)
            for value in evidence["split_manifest"]["splits"]["profile_evaluation"][
                "subject_ids"
            ]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("evidence lacks the profile-evaluation subject split") from exc
    if (
        len(subjects) < 2
        or len(subjects) != len(set(subjects))
        or any(not value for value in subjects)
    ):
        raise ValueError("profile-evaluation subjects must be unique and non-empty")
    return subjects


def _cell_subjects(
    evidence: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the ordered observed-support subjects for the cell's true g*."""

    try:
        support_cells = evidence["quality_conformal"][
            "profile_evaluation_quality_support"
        ]["cells"]
    except (KeyError, TypeError):
        return _profile_subjects(evidence)
    quality_bin = str(cell.get("quality_bin", ""))
    matches = [row for row in support_cells if str(row.get("region_id")) == quality_bin]
    if len(matches) != 1:
        raise ValueError("privacy cell lacks unique observed quality support")
    subjects = tuple(str(row["subject_id"]) for row in matches[0]["subject_frames"])
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("privacy cell quality support is empty or duplicated")
    return subjects


def _privacy_protocol(evidence: Mapping[str, Any]) -> tuple[int, float]:
    try:
        hypotheses = int(evidence["privacy_protocol"]["registered_hypotheses"])
        confidence_error = float(evidence["privacy_protocol"]["confidence_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evidence lacks a valid privacy protocol") from exc
    if hypotheses < 1 or not 0.0 < confidence_error < 1.0:
        raise ValueError("privacy protocol parameters are outside their valid range")
    return hypotheses, confidence_error


def _identity(cell: Mapping[str, Any]) -> dict[str, str]:
    try:
        return {field: str(cell[field]) for field in _IDENTITY_FIELDS}
    except KeyError as exc:
        raise ValueError(f"privacy evidence cell is missing {exc.args[0]}") from exc


def _stage_rows(
    cell: Mapping[str, Any], subjects: Sequence[str], stage: str
) -> list[tuple[float, float]]:
    try:
        raw = cell["stage_subject_rows"][stage]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"privacy evidence lacks stage {stage}") from exc
    if not isinstance(raw, Sequence) or len(raw) != len(subjects):
        raise ValueError(f"stage {stage} rows do not match the subject split")
    rows: list[tuple[float, float]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"stage {stage} subject row {index} is not a pair")
        joint, emission = float(value[0]), float(value[1])
        if not 0.0 <= joint <= emission <= 1.0:
            raise ValueError(f"stage {stage} subject row {index} is invalid")
        rows.append((joint, emission))
    return rows


def _rounded_statistics(statistics: Any) -> dict[str, Any]:
    return {
        key: round(value, 12) if isinstance(value, float) else value
        for key, value in asdict(statistics).items()
    }


def build_post_selection_risk_table(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute all three privacy stages from stored subject-level rows."""

    hypotheses, confidence_error = _privacy_protocol(evidence)
    cells = evidence.get("privacy_evidence")
    if not isinstance(cells, Sequence) or not cells:
        raise ValueError("evidence has no privacy cells")
    table: list[dict[str, Any]] = []
    for cell in cells:
        subjects = _cell_subjects(evidence, cell)
        identity = _identity(cell)
        for stage in _STAGES:
            rows = _stage_rows(cell, subjects, stage)
            statistics = compute_subject_risk_ucb(
                rows,
                registered_hypotheses=hypotheses,
                confidence_error=confidence_error,
            )
            values = _rounded_statistics(statistics)
            emission = float(values["mean_emission"])
            arrival = float(values["mean_emit_and_attack"])
            table.append(
                {
                    **identity,
                    "stage": stage,
                    "independent_unit": "subject",
                    "arrival_risk": arrival,
                    "arrival_risk_ucb": values["joint_risk_ucb"],
                    "emission_coverage": emission,
                    "emission_coverage_lcb": values["emission_lcb"],
                    "conditional_risk": (
                        min(1.0, arrival / emission) if emission > 0.0 else 1.0
                    ),
                    "conditional_risk_ucb": values["conditional_risk_ucb"],
                    "subject_count": values["subject_count"],
                    "simultaneous_hoeffding_radius": values["hoeffding_radius"],
                }
            )
    return table


def _subject_permutation(subjects: Sequence[str], seed: int) -> tuple[str, ...]:
    material = hashlib.sha256(
        canonical_json_bytes({"seed": seed, "subjects": sorted(subjects)})
    ).digest()
    rng = random.Random(int.from_bytes(material[:8], "big"))
    ordered = list(sorted(subjects))
    rng.shuffle(ordered)
    return tuple(ordered)


def _sample_sizes(values: Sequence[int], available: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValueError("subject sample sizes must be a non-empty sequence")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("subject sample sizes must be integers")
        if value < 2 or value > available:
            raise ValueError("subject sample size is outside available support")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("subject sample sizes must be unique")
    return tuple(sorted(normalized))


def build_subject_count_sensitivity(
    evidence: Mapping[str, Any],
    *,
    subject_counts: Sequence[int],
    statistical_seed: int,
) -> dict[str, Any]:
    """Recompute simultaneous bounds on one nested subject permutation."""

    if isinstance(statistical_seed, bool) or not isinstance(statistical_seed, int):
        raise ValueError("statistical_seed must be an integer")
    cells = evidence.get("privacy_evidence")
    if not isinstance(cells, Sequence) or not cells:
        raise ValueError("evidence has no privacy cells")
    available = min(len(_cell_subjects(evidence, cell)) for cell in cells)
    counts = _sample_sizes(subject_counts, available)
    hypotheses, confidence_error = _privacy_protocol(evidence)
    subsets: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cell in cells:
        identity = _identity(cell)
        subjects = _cell_subjects(evidence, cell)
        permutation = _subject_permutation(subjects, statistical_seed)
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        for count in counts:
            selected = permutation[:count]
            indices = [subject_index[subject] for subject in selected]
            subsets.append(
                {
                    **identity,
                    "subject_count": count,
                    "available_subject_count": len(subjects),
                    "subject_ids_sha256": hashlib.sha256(
                        "\n".join(sorted(selected)).encode("utf-8")
                    ).hexdigest(),
                    "nested_prefix": True,
                }
            )
            for stage in _STAGES:
                all_rows = _stage_rows(cell, subjects, stage)
                selected_rows = [all_rows[index] for index in indices]
                statistics = compute_subject_risk_ucb(
                    selected_rows,
                    registered_hypotheses=hypotheses,
                    confidence_error=confidence_error,
                )
                rows.append(
                    {
                        **identity,
                        "stage": stage,
                        "subject_count": count,
                        **_rounded_statistics(statistics),
                    }
                )
    report: dict[str, Any] = {
        "analysis": "subject_support_sensitivity",
        "independent_unit": "subject",
        "sampling": "fixed_seed_nested_without_replacement",
        "statistical_seed": statistical_seed,
        "registered_subject_counts": list(counts),
        "subsets": subsets,
        "rows": rows,
    }
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    return report


def _mean(clusters: Sequence[SubjectCluster], field: str) -> float:
    values = [float(row[field]) for cluster in clusters for row in cluster]
    return sum(values) / len(values)


def _paired_mean(clusters: Sequence[SubjectCluster], left: str, right: str) -> float:
    values = [
        float(row[left]) - float(row[right]) for cluster in clusters for row in cluster
    ]
    return sum(values) / len(values)


def _conditional(clusters: Sequence[SubjectCluster]) -> float:
    arrival = _mean(clusters, "arrival_risk")
    emission = _mean(clusters, "emission")
    return min(1.0, arrival / emission) if emission > 0.0 else 1.0


def build_subject_cluster_uncertainty(
    evidence: Mapping[str, Any],
    *,
    statistical_seed: int,
    resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap three-stage privacy and paired FER at the subject unit."""

    privacy: list[dict[str, Any]] = []
    call_index = 0
    for cell in evidence["privacy_evidence"]:
        subjects = _cell_subjects(evidence, cell)
        identity = _identity(cell)
        for stage in _STAGES:
            subject_rows = _stage_rows(cell, subjects, stage)
            rows = [
                {
                    "subject_id": subject,
                    "arrival_risk": values[0],
                    "emission": values[1],
                }
                for subject, values in zip(subjects, subject_rows, strict=True)
            ]
            statistics: dict[str, Any] = {}
            for offset, (name, statistic) in enumerate(
                (
                    ("arrival_risk", lambda clusters: _mean(clusters, "arrival_risk")),
                    ("emission_coverage", lambda clusters: _mean(clusters, "emission")),
                    ("conditional_risk", _conditional),
                )
            ):
                statistics[name] = subject_cluster_bootstrap(
                    rows,
                    subject_key="subject_id",
                    statistic=statistic,
                    statistic_name=f"privacy_{stage}_{name}_{call_index}",
                    statistical_seed=statistical_seed + call_index * 3 + offset,
                    resamples=resamples,
                    confidence_level=confidence_level,
                )
            privacy.append({**identity, "stage": stage, **statistics})
            call_index += 1

    fer_rows = [dict(row) for row in evidence.get("fer_paired_records", ())]
    required = {
        "subject_id",
        "local_nll",
        "anonymous_edge_nll",
        "local_correct",
        "anonymous_edge_correct",
    }
    if not fer_rows or any(not required.issubset(row) for row in fer_rows):
        raise ValueError("paired FER evidence lacks NLL or correctness fields")
    fer: dict[str, Any] = {}
    fields = (
        ("local_nll", "local_nll"),
        ("anonymous_edge_nll", "anonymous_edge_nll"),
        ("local_accuracy", "local_correct"),
        ("anonymous_edge_accuracy", "anonymous_edge_correct"),
    )
    for offset, (name, field) in enumerate(fields):
        fer[name] = subject_cluster_bootstrap(
            fer_rows,
            subject_key="subject_id",
            statistic=lambda clusters, field=field: _mean(clusters, field),
            statistic_name=f"fer_{name}",
            statistical_seed=statistical_seed + 1_000_000 + offset,
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
            statistic=lambda clusters, left=left, right=right: _paired_mean(
                clusters, left, right
            ),
            statistic_name=f"fer_{name}",
            statistical_seed=statistical_seed + 2_000_000 + offset,
            resamples=resamples,
            confidence_level=confidence_level,
        )
        result["difference_definition"] = definition
        fer[name] = result
    report = {
        "analysis": "subject_cluster_uncertainty",
        "independent_unit": "subject",
        "statistical_seed": statistical_seed,
        "resamples": resamples,
        "confidence_level": confidence_level,
        "privacy": privacy,
        "fer": fer,
    }
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    return report


def build_numerical_experiment_report(
    evidence: Mapping[str, Any],
    *,
    subject_counts: Sequence[int],
    statistical_seed: int,
    resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build one hashed, machine-readable offline numerical analysis report."""

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "numerical_offline_evidence_closure",
        "evidence_hash": str(evidence["evidence_hash"]),
        "split_manifest_hash": str(evidence["split_manifest"]["manifest_hash"]),
        "post_selection_risk": build_post_selection_risk_table(evidence),
        "subject_count_sensitivity": build_subject_count_sensitivity(
            evidence,
            subject_counts=subject_counts,
            statistical_seed=statistical_seed,
        ),
        "subject_cluster_uncertainty": build_subject_cluster_uncertainty(
            evidence,
            statistical_seed=statistical_seed,
            resamples=resamples,
            confidence_level=confidence_level,
        ),
        "provenance": {
            "implementation": "privacy_edge_sim.experiments",
            "implementation_version": "1.0.0",
            "independent_unit": "subject",
            "online_data_used": False,
            "future_trace_used": False,
            "statistical_seed": statistical_seed,
            "resamples": resamples,
            "confidence_level": confidence_level,
            "subject_counts": list(subject_counts),
        },
        "report_sha256": "",
    }
    report["report_sha256"] = canonical_document_sha256(report, "report_sha256")
    return report


__all__ = [
    "build_numerical_experiment_report",
    "build_post_selection_risk_table",
    "build_subject_cluster_uncertainty",
    "build_subject_count_sensitivity",
]
