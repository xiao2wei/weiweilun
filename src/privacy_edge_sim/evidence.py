"""Strict trust-chain validation for frozen numerical-study evidence.

The evidence document is not controller input.  It is loaded before a run and
cross-checked against the frozen profile and both trace roles so a declaration
of numerical eligibility cannot be enabled by metadata flags alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .config import SimulationConfig
from .errors import EvidenceValidationError
from .profiles import (
    FrozenProfileBundle,
    canonical_document_sha256,
    canonical_json_bytes,
    compute_subject_risk_ucb,
)


EXPECTED_SPLIT_ROLES = frozenset(
    {
        "attack_train",
        "attack_threshold_calibration",
        "quality_calibration",
        "profile_evaluation",
        "scenario_training_validation",
        "test",
    }
)


def _fail(code: str, message: str, **context: Any) -> EvidenceValidationError:
    return EvidenceValidationError(code, message, **context)


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(
            "EVIDENCE_FIELD_TYPE", "evidence field must be an object", field=field
        )
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(
            "EVIDENCE_FIELD_TYPE", "evidence field must be non-empty text", field=field
        )
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _fail(
            "EVIDENCE_FIELD_TYPE",
            "evidence field must be a finite number",
            field=field,
        )
    return float(value)


def _rounded_mapping(value: Any) -> dict[str, Any]:
    return {
        key: round(item, 9) if isinstance(item, float) else item
        for key, item in asdict(value).items()
    }


def _require_equal(
    expected: Any, actual: Any, code: str, message: str, **context: Any
) -> None:
    if expected != actual:
        raise _fail(code, message, expected=expected, actual=actual, **context)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_split_manifest(value: Any) -> tuple[Mapping[str, Any], str]:
    split = _object(value, "split_manifest")
    expected_hash = _text(split.get("manifest_hash"), "split_manifest.manifest_hash")
    actual_hash = canonical_document_sha256(split, "manifest_hash")
    if expected_hash != actual_hash:
        raise _fail(
            "EVIDENCE_SPLIT_HASH",
            "split manifest self-hash mismatch",
            expected=expected_hash,
            actual=actual_hash,
        )
    splits = _object(split.get("splits"), "split_manifest.splits")
    if set(splits) != EXPECTED_SPLIT_ROLES:
        raise _fail(
            "EVIDENCE_SPLIT_ROLES",
            "numerical evidence must contain exactly the preregistered six split roles",
            expected=sorted(EXPECTED_SPLIT_ROLES),
            actual=sorted(splits),
        )
    populations: dict[str, set[str]] = {}
    for role in sorted(splits):
        row = _object(splits[role], f"split_manifest.splits.{role}")
        subject_ids = row.get("subject_ids")
        if (
            not isinstance(subject_ids, list)
            or not subject_ids
            or any(not isinstance(item, str) or not item for item in subject_ids)
        ):
            raise _fail(
                "EVIDENCE_SPLIT_SUBJECTS",
                "each split must contain a non-empty subject-id list",
                role=role,
            )
        if len(subject_ids) != len(set(subject_ids)):
            raise _fail(
                "EVIDENCE_SPLIT_DUPLICATE",
                "a split contains duplicate subjects",
                role=role,
            )
        if row.get("subject_count") != len(subject_ids):
            raise _fail(
                "EVIDENCE_SPLIT_COUNT", "split subject count mismatch", role=role
            )
        subject_hash = hashlib.sha256(
            "\n".join(subject_ids).encode("utf-8")
        ).hexdigest()
        if row.get("subject_ids_sha256") != subject_hash:
            raise _fail(
                "EVIDENCE_SPLIT_SUBJECT_HASH", "split subject hash mismatch", role=role
            )
        populations[role] = set(subject_ids)
    overlaps = {
        f"{left}|{right}": sorted(populations[left] & populations[right])
        for index, left in enumerate(sorted(populations))
        for right in sorted(populations)[index + 1 :]
        if populations[left] & populations[right]
    }
    if overlaps or split.get("all_pairwise_subject_intersections_empty") is not True:
        raise _fail(
            "EVIDENCE_SPLIT_OVERLAP",
            "preregistered subject splits are not pairwise disjoint",
            overlaps=overlaps,
        )
    return split, actual_hash


def _validate_quality_and_attacks(
    document: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    conformal = _object(document.get("quality_conformal"), "quality_conformal")
    _text(conformal.get("method"), "quality_conformal.method")
    quantile = conformal.get("quantile")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(float(quantile))
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise _fail(
            "EVIDENCE_CONFORMAL_QUANTILE",
            "conformal quantile must be a finite probability",
        )
    attackers = document.get("attacker_registry")
    if not isinstance(attackers, list) or not attackers:
        raise _fail("EVIDENCE_ATTACKERS", "attacker registry must be non-empty")
    attacker_ids: set[str] = set()
    score_model = _object(document.get("privacy_score_model"), "privacy_score_model")
    score_model_hash = _text(
        score_model.get("model_hash"), "privacy_score_model.model_hash"
    )
    _require_equal(
        canonical_document_sha256(score_model, "model_hash"),
        score_model_hash,
        "EVIDENCE_PRIVACY_SCORE_HASH",
        "privacy score model self-hash mismatch",
    )
    score_model_version = _text(
        score_model.get("model_version"), "privacy_score_model.model_version"
    )
    _require_equal(
        "engineering_assumption",
        score_model.get("source_category"),
        "EVIDENCE_PRIVACY_SCORE_SOURCE",
        "numerical privacy score model must be labelled as an engineering assumption",
    )
    attack_train_hash = split["splits"]["attack_train"]["subject_ids_sha256"]
    threshold_hash = split["splits"]["attack_threshold_calibration"][
        "subject_ids_sha256"
    ]
    for index, attacker_raw in enumerate(attackers):
        attacker = _object(attacker_raw, f"attacker_registry[{index}]")
        attacker_id = _text(
            attacker.get("attacker_id"), f"attacker_registry[{index}].attacker_id"
        )
        if attacker_id in attacker_ids:
            raise _fail(
                "EVIDENCE_ATTACKER_DUPLICATE",
                "attacker registry contains a duplicate attacker id",
                attacker_id=attacker_id,
            )
        attacker_ids.add(attacker_id)
        _require_equal(
            score_model_hash,
            attacker.get("score_model_hash"),
            "EVIDENCE_ATTACK_SCORE_MODEL",
            "attacker is not bound to the frozen privacy score model",
            attacker_id=attacker_id,
        )
        _require_equal(
            score_model_version,
            attacker.get("score_model_version"),
            "EVIDENCE_ATTACK_SCORE_MODEL",
            "attacker score-model version mismatch",
            attacker_id=attacker_id,
        )
        _require_equal(
            attack_train_hash,
            attacker.get("training_subjects_hash"),
            "EVIDENCE_ATTACK_TRAIN_SPLIT",
            "attacker parameters were not fitted on the registered attack-train split",
            attacker_id=attacker_id,
        )
        _require_equal(
            threshold_hash,
            attacker.get("threshold_calibration_subjects_hash"),
            "EVIDENCE_ATTACK_THRESHOLD_SPLIT",
            "attacker thresholds were not fitted on the registered threshold split",
            attacker_id=attacker_id,
        )
        thresholds = _object(
            attacker.get("thresholds"), f"attacker_registry[{index}].thresholds"
        )
        if set(thresholds) != {"identity", "verification", "link"}:
            raise _fail(
                "EVIDENCE_ATTACK_THRESHOLDS",
                "each attacker must freeze identity, verification and link thresholds",
                index=index,
            )
        for risk, threshold_raw in thresholds.items():
            threshold = _object(
                threshold_raw, f"attacker_registry[{index}].thresholds.{risk}"
            )
            value = threshold.get("threshold")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise _fail(
                    "EVIDENCE_ATTACK_THRESHOLD",
                    "attack threshold must be finite",
                    index=index,
                    risk=risk,
                )
            _text(
                threshold.get("threshold_id"),
                f"attacker_registry[{index}].thresholds.{risk}.threshold_id",
            )
            if risk == "identity":
                if (
                    threshold.get("decision_rule") != "rank_1_gallery_retrieval"
                    or threshold.get("used_for_success") is not False
                ):
                    raise _fail(
                        "EVIDENCE_IDENTITY_RANK1",
                        "identity success must be frozen as Rank-1 retrieval and must not use a threshold",
                        attacker_id=attacker_id,
                    )
            elif (
                threshold.get("decision_rule")
                != "score_greater_equal_calibrated_threshold"
                or threshold.get("used_for_success") is not True
            ):
                raise _fail(
                    "EVIDENCE_ATTACK_DECISION_RULE",
                    "verification and link success must use their calibrated thresholds",
                    attacker_id=attacker_id,
                    risk=risk,
                )


def _validate_profile_quality_support(
    document: Mapping[str, Any], split: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    conformal = _object(document.get("quality_conformal"), "quality_conformal")
    support = _object(
        conformal.get("profile_evaluation_quality_support"),
        "quality_conformal.profile_evaluation_quality_support",
    )
    expected_hash = _text(
        support.get("support_hash"),
        "quality_conformal.profile_evaluation_quality_support.support_hash",
    )
    _require_equal(
        canonical_document_sha256(support, "support_hash"),
        expected_hash,
        "EVIDENCE_QUALITY_SUPPORT_HASH",
        "profile-evaluation quality support self-hash mismatch",
    )
    _require_equal(
        "profile_evaluation",
        support.get("role"),
        "EVIDENCE_QUALITY_SUPPORT_ROLE",
        "quality-cell support must come from the profile-evaluation split",
    )
    region_ids = conformal.get("region_ids")
    if (
        not isinstance(region_ids, list)
        or not region_ids
        or any(not isinstance(value, str) or not value for value in region_ids)
    ):
        raise _fail(
            "EVIDENCE_QUALITY_SUPPORT_REGIONS",
            "quality conformal region_ids must be a non-empty text list",
        )
    cells = support.get("cells")
    if not isinstance(cells, list) or not cells:
        raise _fail(
            "EVIDENCE_QUALITY_SUPPORT_CELLS",
            "profile-evaluation quality support must contain cells",
        )
    spec = _object(document.get("spec"), "spec")
    frames_per_subject = spec.get("frames_per_subject")
    if (
        isinstance(frames_per_subject, bool)
        or not isinstance(frames_per_subject, int)
        or frames_per_subject < 1
    ):
        raise _fail(
            "EVIDENCE_QUALITY_SUPPORT_FRAMES",
            "frames_per_subject must be a positive integer",
        )
    profile_subject_ids = list(split["splits"]["profile_evaluation"]["subject_ids"])
    profile_subjects = set(profile_subject_ids)
    expected_pairs = {
        (subject_id, frame_index)
        for subject_id in profile_subject_ids
        for frame_index in range(frames_per_subject)
    }
    seen_pairs: set[tuple[str, int]] = set()
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_cell in enumerate(cells):
        cell = _object(
            raw_cell,
            f"quality_conformal.profile_evaluation_quality_support.cells[{index}]",
        )
        region_id = _text(cell.get("region_id"), "quality support region_id")
        if region_id in result:
            raise _fail(
                "EVIDENCE_QUALITY_SUPPORT_DUPLICATE",
                "quality support contains a duplicate region",
                region_id=region_id,
            )
        subject_frames = cell.get("subject_frames")
        if not isinstance(subject_frames, list) or not subject_frames:
            raise _fail(
                "EVIDENCE_QUALITY_SUPPORT_SUBJECTS",
                "each quality region must have non-empty subject/frame support",
                region_id=region_id,
            )
        ordered_subject_ids: list[str] = []
        frame_count = 0
        for subject_index, raw_subject in enumerate(subject_frames):
            subject_row = _object(
                raw_subject,
                f"quality support {region_id} subject[{subject_index}]",
            )
            subject_id = _text(subject_row.get("subject_id"), "quality support subject")
            if subject_id not in profile_subjects or subject_id in ordered_subject_ids:
                raise _fail(
                    "EVIDENCE_QUALITY_SUPPORT_SUBJECT",
                    "quality support subject is outside the split or duplicated in a cell",
                    region_id=region_id,
                    subject_id=subject_id,
                )
            ordered_subject_ids.append(subject_id)
            frames = subject_row.get("frames")
            if not isinstance(frames, list) or not frames:
                raise _fail(
                    "EVIDENCE_QUALITY_SUPPORT_FRAMES",
                    "a supported subject must have at least one frame in the cell",
                    region_id=region_id,
                    subject_id=subject_id,
                )
            ordered_indices: list[int] = []
            for raw_frame in frames:
                frame = _object(raw_frame, "quality support frame")
                frame_index = frame.get("frame_index")
                if (
                    isinstance(frame_index, bool)
                    or not isinstance(frame_index, int)
                    or not 0 <= frame_index < frames_per_subject
                    or frame_index in ordered_indices
                ):
                    raise _fail(
                        "EVIDENCE_QUALITY_SUPPORT_FRAME_INDEX",
                        "quality support frame index is invalid or duplicated",
                        region_id=region_id,
                        subject_id=subject_id,
                        frame_index=frame_index,
                    )
                quality_score = _finite_number(
                    frame.get("true_quality_score"), "true_quality_score"
                )
                if not 0.0 <= quality_score <= 1.0:
                    raise _fail(
                        "EVIDENCE_QUALITY_SUPPORT_SCORE",
                        "true quality score must be a probability-scale value",
                    )
                ordered_indices.append(frame_index)
                pair = (subject_id, frame_index)
                if pair in seen_pairs:
                    raise _fail(
                        "EVIDENCE_QUALITY_SUPPORT_OVERLAP",
                        "a profile-evaluation frame belongs to more than one true quality cell",
                        subject_id=subject_id,
                        frame_index=frame_index,
                    )
                seen_pairs.add(pair)
                frame_count += 1
            if ordered_indices != sorted(ordered_indices):
                raise _fail(
                    "EVIDENCE_QUALITY_SUPPORT_ORDER",
                    "quality support frame indices must use stable order",
                )
        if ordered_subject_ids != sorted(ordered_subject_ids):
            raise _fail(
                "EVIDENCE_QUALITY_SUPPORT_ORDER",
                "quality support subjects must use stable order",
                region_id=region_id,
            )
        subject_hash = hashlib.sha256(
            "\n".join(ordered_subject_ids).encode("utf-8")
        ).hexdigest()
        _require_equal(
            len(ordered_subject_ids),
            cell.get("subject_count"),
            "EVIDENCE_QUALITY_SUPPORT_COUNT",
            "quality support subject count mismatch",
            region_id=region_id,
        )
        _require_equal(
            frame_count,
            cell.get("frame_count"),
            "EVIDENCE_QUALITY_SUPPORT_COUNT",
            "quality support frame count mismatch",
            region_id=region_id,
        )
        _require_equal(
            subject_hash,
            cell.get("subject_ids_sha256"),
            "EVIDENCE_QUALITY_SUPPORT_SUBJECT_HASH",
            "quality support ordered subject hash mismatch",
            region_id=region_id,
        )
        result[region_id] = cell
    if set(result) != set(region_ids):
        raise _fail(
            "EVIDENCE_QUALITY_SUPPORT_REGIONS",
            "quality support does not cover exactly the frozen regions",
            expected=sorted(region_ids),
            actual=sorted(result),
        )
    if seen_pairs != expected_pairs:
        raise _fail(
            "EVIDENCE_QUALITY_SUPPORT_COVERAGE",
            "quality support must assign every profile-evaluation frame exactly once",
            missing_count=len(expected_pairs - seen_pairs),
            unexpected_count=len(seen_pairs - expected_pairs),
        )
    return result


def _validate_privacy_recomputation(
    document: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    protocol = _object(document.get("privacy_protocol"), "privacy_protocol")
    hypotheses = protocol.get("registered_hypotheses")
    if (
        isinstance(hypotheses, bool)
        or not isinstance(hypotheses, int)
        or hypotheses < 1
    ):
        raise _fail(
            "EVIDENCE_PRIVACY_PROTOCOL",
            "registered privacy hypothesis count must be a positive integer",
        )
    confidence_error = _finite_number(
        protocol.get("confidence_error"), "privacy_protocol.confidence_error"
    )
    if not 0.0 < confidence_error <= 1.0:
        raise _fail(
            "EVIDENCE_PRIVACY_PROTOCOL",
            "privacy confidence error must lie in (0, 1]",
        )
    registered_dimensions: dict[str, tuple[str, ...]] = {}
    for field in (
        "registered_pipeline_ids",
        "registered_quality_region_ids",
        "registered_attacker_ids",
        "registered_risk_types",
    ):
        values = protocol.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise _fail(
                "EVIDENCE_PRIVACY_PROTOCOL",
                "privacy protocol dimensions must be non-empty unique text lists",
                field=field,
            )
        registered_dimensions[field] = tuple(values)
    expected_keys = {
        (pipeline_id, quality_bin, attacker_id, risk_type)
        for pipeline_id in registered_dimensions["registered_pipeline_ids"]
        for quality_bin in registered_dimensions["registered_quality_region_ids"]
        for attacker_id in registered_dimensions["registered_attacker_ids"]
        for risk_type in registered_dimensions["registered_risk_types"]
    }
    _require_equal(
        {"identity", "verification", "link"},
        set(registered_dimensions["registered_risk_types"]),
        "EVIDENCE_PRIVACY_PROTOCOL",
        "privacy protocol must register identity, verification and link risk",
    )
    _require_equal(
        len(expected_keys),
        hypotheses,
        "EVIDENCE_PRIVACY_HYPOTHESIS_COUNT",
        "registered privacy dimension product does not match hypothesis count",
    )
    conformal_regions = tuple(document["quality_conformal"]["region_ids"])
    _require_equal(
        conformal_regions,
        registered_dimensions["registered_quality_region_ids"],
        "EVIDENCE_PRIVACY_PROTOCOL",
        "privacy protocol quality regions do not match the frozen partition",
    )
    attackers = {
        str(attacker["attacker_id"]): attacker
        for attacker in document["attacker_registry"]
    }
    _require_equal(
        set(attackers),
        set(registered_dimensions["registered_attacker_ids"]),
        "EVIDENCE_PRIVACY_PROTOCOL",
        "privacy protocol attackers do not match the frozen registry",
    )
    quality_support = _validate_profile_quality_support(document, split)
    support_hash = document["quality_conformal"]["profile_evaluation_quality_support"][
        "support_hash"
    ]
    cells = document.get("privacy_evidence")
    if not isinstance(cells, list) or not cells:
        raise _fail("EVIDENCE_PRIVACY_ROWS", "privacy evidence must be non-empty")
    registered_keys: set[tuple[str, str, str, str]] = set()
    stages = {"single_attempt", "guard_selected", "guard_plus_retry_final"}
    for index, raw_cell in enumerate(cells):
        cell = _object(raw_cell, f"privacy_evidence[{index}]")
        key = tuple(
            _text(cell.get(field), f"privacy_evidence[{index}].{field}")
            for field in ("pipeline_id", "quality_bin", "attacker_id", "risk_type")
        )
        if key in registered_keys:
            raise _fail(
                "EVIDENCE_PRIVACY_DUPLICATE",
                "privacy evidence contains a duplicate registered hypothesis",
                hypothesis=key,
            )
        registered_keys.add(key)
        attacker = attackers.get(key[2])
        if attacker is None or key[3] not in attacker["thresholds"]:
            raise _fail(
                "EVIDENCE_PRIVACY_THRESHOLD",
                "privacy cell has no registered attacker/risk threshold",
                hypothesis=key,
            )
        _require_equal(
            attacker["thresholds"][key[3]]["threshold_id"],
            cell.get("threshold_id"),
            "EVIDENCE_PRIVACY_THRESHOLD",
            "privacy cell threshold does not match the attacker registry",
            hypothesis=key,
        )
        cell_support = quality_support.get(key[1])
        if cell_support is None:
            raise _fail(
                "EVIDENCE_PRIVACY_QUALITY_SUPPORT",
                "privacy hypothesis has no frozen quality-cell support",
                hypothesis=key,
            )
        _require_equal(
            support_hash,
            cell.get("profile_quality_support_hash"),
            "EVIDENCE_PRIVACY_QUALITY_SUPPORT",
            "privacy row is not bound to the frozen quality support",
            hypothesis=key,
        )
        _require_equal(
            cell_support["subject_ids_sha256"],
            cell.get("quality_cell_subject_ids_sha256"),
            "EVIDENCE_PRIVACY_QUALITY_SUPPORT",
            "privacy subject-row order is not bound to the quality cell",
            hypothesis=key,
        )
        _require_equal(
            cell_support["subject_count"],
            cell.get("quality_cell_subject_count"),
            "EVIDENCE_PRIVACY_QUALITY_SUPPORT",
            "privacy quality-cell subject count mismatch",
            hypothesis=key,
        )
        _require_equal(
            cell_support["frame_count"],
            cell.get("quality_cell_frame_count"),
            "EVIDENCE_PRIVACY_QUALITY_SUPPORT",
            "privacy quality-cell frame count mismatch",
            hypothesis=key,
        )
        subject_count = int(cell_support["subject_count"])
        stage_rows = _object(
            cell.get("stage_subject_rows"),
            f"privacy_evidence[{index}].stage_subject_rows",
        )
        stage_statistics = _object(
            cell.get("stage_statistics"),
            f"privacy_evidence[{index}].stage_statistics",
        )
        if set(stage_rows) != stages or set(stage_statistics) != stages:
            raise _fail(
                "EVIDENCE_PRIVACY_STAGES",
                "privacy evidence must contain exactly the three preregistered stages",
                hypothesis=key,
            )
        recomputed: dict[str, dict[str, Any]] = {}
        for stage in sorted(stages):
            raw_rows = stage_rows[stage]
            if not isinstance(raw_rows, list) or len(raw_rows) != subject_count:
                raise _fail(
                    "EVIDENCE_PRIVACY_SUBJECT_ROWS",
                    "privacy subject rows do not match the profile-evaluation split",
                    hypothesis=key,
                    stage=stage,
                )
            rows: list[tuple[float, float]] = []
            for row_index, raw_row in enumerate(raw_rows):
                if not isinstance(raw_row, list) or len(raw_row) != 2:
                    raise _fail(
                        "EVIDENCE_PRIVACY_SUBJECT_ROW",
                        "privacy subject row must be an [attack-and-emit, emission] pair",
                        hypothesis=key,
                        stage=stage,
                        row_index=row_index,
                    )
                joint = _finite_number(raw_row[0], "privacy subject joint risk")
                emission = _finite_number(raw_row[1], "privacy subject emission")
                if not 0.0 <= joint <= emission <= 1.0:
                    raise _fail(
                        "EVIDENCE_PRIVACY_SUBJECT_LOGIC",
                        "privacy subject row violates 0 <= joint <= emission <= 1",
                        hypothesis=key,
                        stage=stage,
                        row_index=row_index,
                    )
                rows.append((joint, emission))
            recomputed[stage] = _rounded_mapping(
                compute_subject_risk_ucb(
                    rows,
                    registered_hypotheses=hypotheses,
                    confidence_error=confidence_error,
                )
            )
            stored = dict(
                _object(
                    stage_statistics[stage],
                    f"privacy_evidence[{index}].stage_statistics.{stage}",
                )
            )
            _require_equal(
                recomputed[stage],
                stored,
                "EVIDENCE_PRIVACY_RECOMPUTE",
                "stored privacy UCB does not recompute from its raw subject rows",
                hypothesis=key,
                stage=stage,
            )
        final_rows = stage_rows["guard_plus_retry_final"]
        _require_equal(
            final_rows,
            cell.get("subject_rows"),
            "EVIDENCE_PRIVACY_FINAL_ALIAS",
            "final privacy subject rows do not alias the registered retry-final stage",
            hypothesis=key,
        )
        _require_equal(
            recomputed["guard_plus_retry_final"],
            dict(
                _object(cell.get("statistics"), f"privacy_evidence[{index}].statistics")
            ),
            "EVIDENCE_PRIVACY_FINAL_ALIAS",
            "final privacy statistics do not alias the registered retry-final stage",
            hypothesis=key,
        )
    if registered_keys != expected_keys:
        raise _fail(
            "EVIDENCE_PRIVACY_HYPOTHESIS_COUNT",
            "privacy evidence keys do not equal the preregistered Cartesian family",
            missing=[list(value) for value in sorted(expected_keys - registered_keys)],
            unexpected=[
                list(value) for value in sorted(registered_keys - expected_keys)
            ],
        )


def _positive_median(values: list[float], field: str) -> float:
    ordered = sorted(value for value in values if math.isfinite(value) and value > 0.0)
    if not ordered:
        raise _fail(
            "EVIDENCE_NORMALIZATION_RECORDS",
            "normalization records contain no positive finite values",
            field=field,
        )
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _validate_cost_normalization(
    document: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    normalization = _object(document.get("cost_normalization"), "cost_normalization")
    _require_equal(
        "scenario_training_validation",
        normalization.get("role"),
        "EVIDENCE_NORMALIZATION_ROLE",
        "normalization scales must use only the scenario training/validation split",
    )
    _require_equal(
        "median_of_positive_preregistered_baseline_records",
        normalization.get("method"),
        "EVIDENCE_NORMALIZATION_METHOD",
        "normalization method is not the preregistered positive median",
    )
    if normalization.get("online_mutable") is not False:
        raise _fail(
            "EVIDENCE_NORMALIZATION_MUTABLE",
            "normalization scales must be frozen before online execution",
        )
    expected_hash = _text(
        normalization.get("calibration_hash"),
        "cost_normalization.calibration_hash",
    )
    actual_hash = canonical_document_sha256(normalization, "calibration_hash")
    _require_equal(
        expected_hash,
        actual_hash,
        "EVIDENCE_NORMALIZATION_HASH",
        "normalization calibration self-hash mismatch",
    )
    records = normalization.get("records")
    if not isinstance(records, list) or not records:
        raise _fail(
            "EVIDENCE_NORMALIZATION_RECORDS",
            "normalization calibration records must be non-empty",
        )
    expected_subjects = set(
        split["splits"]["scenario_training_validation"]["subject_ids"]
    )
    actual_subjects: list[str] = []
    source_fields = {
        "latency_scale_s": "latency_s",
        "vehicle_energy_scale_j": "vehicle_energy_j",
        "rsu_energy_scale_j": "rsu_energy_j",
        "utility_scale": "utility_loss",
    }
    values: dict[str, list[float]] = {field: [] for field in source_fields.values()}
    for index, raw_record in enumerate(records):
        record = _object(raw_record, f"cost_normalization.records[{index}]")
        actual_subjects.append(
            _text(
                record.get("subject_id"),
                f"cost_normalization.records[{index}].subject_id",
            )
        )
        for field in values:
            values[field].append(
                _finite_number(
                    record.get(field),
                    f"cost_normalization.records[{index}].{field}",
                )
            )
    if (
        len(actual_subjects) != len(set(actual_subjects))
        or set(actual_subjects) != expected_subjects
    ):
        raise _fail(
            "EVIDENCE_NORMALIZATION_SUBJECTS",
            "normalization records do not correspond exactly to the registered scenario subjects",
            missing=sorted(expected_subjects - set(actual_subjects)),
            unexpected=sorted(set(actual_subjects) - expected_subjects),
        )
    scales = _object(normalization.get("scales"), "cost_normalization.scales")
    if set(scales) != set(source_fields):
        raise _fail(
            "EVIDENCE_NORMALIZATION_SCALES",
            "normalization scale names do not match the preregistered cost components",
        )
    recomputed = {
        scale: round(_positive_median(values[source], source), 9)
        for scale, source in source_fields.items()
    }
    stored = {
        name: _finite_number(scales[name], f"cost_normalization.scales.{name}")
        for name in sorted(scales)
    }
    _require_equal(
        recomputed,
        stored,
        "EVIDENCE_NORMALIZATION_RECOMPUTE",
        "normalization scales do not recompute from their declared calibration records",
    )


def _validate_controller_weight_evidence(
    document: Mapping[str, Any], config: SimulationConfig
) -> None:
    weights = _object(
        document.get("controller_weight_evidence"), "controller_weight_evidence"
    )
    expected_metadata = {
        "role": "scenario_training_validation",
        "online_mutable": False,
        "source_type": "engineering_assumption",
        "unit": "dimensionless_busy_second_quadratic_weight",
        "method": "preregistered_unit_weights_not_data_fitted",
    }
    for field, expected in expected_metadata.items():
        _require_equal(
            expected,
            weights.get(field),
            "EVIDENCE_CONTROLLER_WEIGHT_METADATA",
            "controller weight evidence metadata is not preregistered and frozen",
            field=field,
        )
    values = _object(weights.get("values"), "controller_weight_evidence.values")
    expected_keys = {
        "physical_queue_weight",
        "vehicle_resource_theta",
        "rsu_resource_theta",
    }
    if set(values) != expected_keys:
        raise _fail(
            "EVIDENCE_CONTROLLER_WEIGHT_FIELDS",
            "controller weight evidence must contain exactly the three registered weight families",
            expected=sorted(expected_keys),
            actual=sorted(values),
        )
    physical_weight = _finite_number(
        values["physical_queue_weight"],
        "controller_weight_evidence.values.physical_queue_weight",
    )
    vehicle_theta_raw = _object(
        values["vehicle_resource_theta"],
        "controller_weight_evidence.values.vehicle_resource_theta",
    )
    rsu_theta_raw = _object(
        values["rsu_resource_theta"],
        "controller_weight_evidence.values.rsu_resource_theta",
    )
    vehicle_theta = {
        str(name): _finite_number(
            value, f"controller_weight_evidence.values.vehicle_resource_theta.{name}"
        )
        for name, value in vehicle_theta_raw.items()
    }
    rsu_theta = {
        str(name): _finite_number(
            value, f"controller_weight_evidence.values.rsu_resource_theta.{name}"
        )
        for name, value in rsu_theta_raw.items()
    }
    if physical_weight < 0.0 or any(
        value < 0.0 for value in (*vehicle_theta.values(), *rsu_theta.values())
    ):
        raise _fail(
            "EVIDENCE_CONTROLLER_WEIGHT_RANGE",
            "controller physical queue weights must be nonnegative",
        )
    declared_values = {
        "physical_queue_weight": physical_weight,
        "vehicle_resource_theta": vehicle_theta,
        "rsu_resource_theta": rsu_theta,
    }
    expected_hash = hashlib.sha256(canonical_json_bytes(declared_values)).hexdigest()
    _require_equal(
        expected_hash,
        weights.get("values_sha256"),
        "EVIDENCE_CONTROLLER_WEIGHT_HASH",
        "controller weight values hash mismatch",
    )
    configured_values = {
        "physical_queue_weight": float(config.controller.physical_queue_weight),
        "vehicle_resource_theta": {
            str(name): float(value)
            for name, value in config.controller.vehicle_resource_theta.items()
        },
        "rsu_resource_theta": {
            str(name): float(value)
            for name, value in config.controller.rsu_resource_theta.items()
        },
    }
    _require_equal(
        declared_values,
        configured_values,
        "EVIDENCE_CONTROLLER_WEIGHT_CONFIG",
        "online controller resource weights differ from the frozen evidence",
    )


def _validate_profile_privacy_crossref(
    document: Mapping[str, Any], profile: FrozenProfileBundle
) -> None:
    protocol = document["privacy_protocol"]
    pipeline_ids = tuple(protocol["registered_pipeline_ids"])
    quality_bins = tuple(protocol["registered_quality_region_ids"])
    _require_equal(
        set(profile.pipelines),
        set(pipeline_ids),
        "EVIDENCE_PROFILE_PRIVACY_PIPELINES",
        "profile pipelines do not match the registered privacy family",
    )
    _require_equal(
        set(profile.quality_bins),
        set(quality_bins),
        "EVIDENCE_PROFILE_PRIVACY_QUALITY",
        "profile quality bins do not match the registered privacy family",
    )
    evidence_cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in document["privacy_evidence"]:
        evidence_cells.setdefault(
            (str(row["pipeline_id"]), str(row["quality_bin"])), []
        ).append(row)
    expected_profile_keys = {
        (pipeline_id, quality_bin, device_type)
        for pipeline_id in pipeline_ids
        for quality_bin in quality_bins
        for device_type in profile.pipelines[pipeline_id].supported_devices
    }
    _require_equal(
        expected_profile_keys,
        set(profile.privacy_cells),
        "EVIDENCE_PROFILE_PRIVACY_CELLS",
        "profile privacy cells do not cover exactly pipeline x quality x device",
    )
    for pipeline_id, quality_bin, device_type in sorted(expected_profile_keys):
        evidence_bounds = tuple(
            sorted(
                (
                    str(row["risk_type"]),
                    str(row["attacker_id"]),
                    str(row["threshold_id"]),
                    float(row["statistics"]["conditional_risk_ucb"]),
                    int(row["statistics"]["subject_count"]),
                    float(row["statistics"]["emission_lcb"]),
                    float(protocol["confidence_error"]),
                )
                for row in evidence_cells[(pipeline_id, quality_bin)]
            )
        )
        cell = profile.privacy_cells[(pipeline_id, quality_bin, device_type)]
        profile_bounds = tuple(
            sorted(
                (
                    bound.risk_type,
                    bound.attacker_id,
                    bound.threshold_id,
                    float(bound.ucb),
                    int(bound.subject_count),
                    float(bound.emission_lcb),
                    float(bound.confidence_error),
                )
                for bound in cell.bounds
            )
        )
        _require_equal(
            evidence_bounds,
            profile_bounds,
            "EVIDENCE_PROFILE_PRIVACY_BOUND",
            "profile privacy bounds do not equal the frozen evidence",
            pipeline_id=pipeline_id,
            quality_bin=quality_bin,
            device_type=device_type,
        )


def _validate_fer_pairing(
    document: Mapping[str, Any],
    split: Mapping[str, Any],
    profile: FrozenProfileBundle,
    evaluation_trace: Any,
) -> None:
    rows = document.get("fer_paired_records")
    if not isinstance(rows, list) or not rows:
        raise _fail("EVIDENCE_FER_ROWS", "paired FER evidence must be non-empty")
    subjects = set(split["splits"]["test"]["subject_ids"])
    pipelines = set(profile.pipelines)
    quality_bins = set(profile.quality_bins)
    expected_keys = {
        (subject, pipeline, quality)
        for subject in subjects
        for pipeline in pipelines
        for quality in quality_bins
    }
    actual_keys: set[tuple[str, str, str]] = set()
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"fer_paired_records[{index}]")
        key = (
            _text(row.get("subject_id"), f"fer_paired_records[{index}].subject_id"),
            _text(row.get("pipeline_id"), f"fer_paired_records[{index}].pipeline_id"),
            _text(row.get("quality_bin"), f"fer_paired_records[{index}].quality_bin"),
        )
        if key in actual_keys:
            raise _fail(
                "EVIDENCE_FER_DUPLICATE",
                "paired FER evidence contains a duplicate subject/pipeline/quality row",
                key=key,
            )
        actual_keys.add(key)
        local_nll = _finite_number(
            row.get("local_nll"), f"fer_paired_records[{index}].local_nll"
        )
        edge_nll = _finite_number(
            row.get("anonymous_edge_nll"),
            f"fer_paired_records[{index}].anonymous_edge_nll",
        )
        delta = _finite_number(
            row.get("paired_nll_delta"),
            f"fer_paired_records[{index}].paired_nll_delta",
        )
        # All three values are independently frozen to nine decimal places;
        # allow exactly the resulting two-rounding-unit arithmetic envelope.
        if not math.isclose(edge_nll - local_nll, delta, rel_tol=0.0, abs_tol=2e-9):
            raise _fail(
                "EVIDENCE_FER_DELTA",
                "paired FER NLL delta is inconsistent with the stored path outcomes",
                key=key,
            )
        for field in ("local_correct", "anonymous_edge_correct"):
            if not isinstance(row.get(field), bool):
                raise _fail(
                    "EVIDENCE_FER_CORRECTNESS",
                    "paired FER correctness fields must be booleans",
                    key=key,
                    field=field,
                )
    if actual_keys != expected_keys:
        raise _fail(
            "EVIDENCE_FER_FACTORIAL_SUPPORT",
            "paired FER rows do not cover the complete registered test subject/pipeline/quality grid",
            missing_count=len(expected_keys - actual_keys),
            unexpected_count=len(actual_keys - expected_keys),
        )

    trace_anon_subjects = {row.subject_cluster_id for row in evaluation_trace.anon_rows}
    trace_local_subjects = {
        row.subject_cluster_id
        for row in evaluation_trace.local_rows
        if row.subject_cluster_id is not None
    }
    if trace_anon_subjects != subjects or not trace_local_subjects.issubset(subjects):
        raise _fail(
            "EVIDENCE_FER_TRACE_SUBJECTS",
            "evaluation FER trace rows do not belong to the registered test population",
            missing_anon_subjects=sorted(subjects - trace_anon_subjects),
            unexpected_subjects=sorted(
                (trace_anon_subjects | trace_local_subjects) - subjects
            ),
        )
    trace_support = {
        (row.pipeline_id, row.quality_bin) for row in evaluation_trace.anon_rows
    }
    expected_support = {
        (pipeline, quality) for pipeline in pipelines for quality in quality_bins
    }
    if trace_support != expected_support:
        raise _fail(
            "EVIDENCE_FER_TRACE_SUPPORT",
            "evaluation trace does not support every registered paired FER pipeline/quality cell",
            missing=sorted(expected_support - trace_support),
            unexpected=sorted(trace_support - expected_support),
        )


@dataclass(frozen=True, slots=True)
class EvidenceVerification:
    required: bool
    verified: bool
    path: Path | None
    file_sha256: str | None
    size_bytes: int | None
    evidence_hash: str | None
    split_manifest_hash: str | None
    document: Mapping[str, Any] | None

    @classmethod
    def not_required(cls) -> "EvidenceVerification":
        return cls(False, True, None, None, None, None, None, None)


def verify_run_evidence(
    config: SimulationConfig,
    profile: FrozenProfileBundle,
    evaluation_trace: Any,
    scenario_trace: Any,
) -> EvidenceVerification:
    """Load and cross-check the evidence required by numerical runs."""

    kinds = {profile.data_kind, evaluation_trace.data_kind, scenario_trace.data_kind}
    numerical = "numerical_simulation" in kinds
    if not numerical:
        if config.evidence_path is not None:
            raise _fail(
                "EVIDENCE_UNEXPECTED",
                "evidence_path is reserved for a homogeneous numerical-simulation run",
                kinds=sorted(kinds),
            )
        return EvidenceVerification.not_required()
    if kinds != {"numerical_simulation"}:
        raise _fail(
            "EVIDENCE_MIXED_KIND",
            "numerical evidence cannot authorize mixed data kinds",
            kinds=sorted(kinds),
        )
    if config.evidence_path is None:
        raise _fail(
            "EVIDENCE_PATH_REQUIRED",
            "numerical-simulation config must reference a frozen evidence_path",
        )
    path = config.evidence_path.resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(
            "EVIDENCE_READ",
            "cannot read frozen evidence as strict UTF-8 JSON",
            path=str(path),
            error=str(exc),
        ) from exc
    document = _object(raw, "$")
    if (
        document.get("data_kind") != "numerical_simulation"
        or document.get("evidence_status") != "frozen_numerical_model"
    ):
        raise _fail(
            "EVIDENCE_KIND",
            "evidence kind/status is not a frozen numerical model",
        )
    expected_hash = _text(document.get("evidence_hash"), "evidence_hash")
    actual_hash = canonical_document_sha256(document, "evidence_hash")
    if expected_hash != actual_hash:
        raise _fail(
            "EVIDENCE_HASH",
            "evidence self-hash mismatch",
            expected=expected_hash,
            actual=actual_hash,
        )
    split, split_hash = _validate_split_manifest(document.get("split_manifest"))
    _validate_quality_and_attacks(document, split)
    _validate_privacy_recomputation(document, split)
    _validate_profile_privacy_crossref(document, profile)
    _validate_cost_normalization(document, split)
    _validate_controller_weight_evidence(document, config)
    _validate_fer_pairing(document, split, profile, evaluation_trace)

    profile_metadata = profile.metadata
    if (
        profile_metadata.get("evidence_hash") != actual_hash
        or profile_metadata.get("split_manifest_hash") != split_hash
    ):
        raise _fail(
            "EVIDENCE_PROFILE_CROSSREF",
            "profile does not cross-reference the verified evidence and split manifest",
        )
    role_expectations = (
        (evaluation_trace, "test", {"evaluation", "test"}),
        (scenario_trace, "scenario_training_validation", {"training_validation"}),
    )
    for trace, evidence_role, accepted_roles in role_expectations:
        metadata = trace.metadata
        if (
            metadata.get("evidence_hash") != actual_hash
            or metadata.get("split_manifest_hash") != split_hash
        ):
            raise _fail(
                "EVIDENCE_TRACE_CROSSREF",
                "trace does not cross-reference the verified evidence and split manifest",
                trace_version=trace.trace_version,
                expected_role=evidence_role,
            )
        split_metadata = _object(
            metadata.get("data_split"), f"trace[{evidence_role}].metadata.data_split"
        )
        if split_metadata.get("role") not in accepted_roles:
            raise _fail(
                "EVIDENCE_TRACE_ROLE",
                "trace role is inconsistent with the evidence split",
                expected=sorted(accepted_roles),
                actual=split_metadata.get("role"),
            )
        expected_subject_hash = split["splits"][evidence_role]["subject_ids_sha256"]
        if split_metadata.get("subject_population_hash") != expected_subject_hash:
            raise _fail(
                "EVIDENCE_TRACE_SUBJECT_HASH",
                "trace subject population does not match its registered evidence split",
                evidence_role=evidence_role,
            )
    return EvidenceVerification(
        required=True,
        verified=True,
        path=path,
        file_sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        evidence_hash=actual_hash,
        split_manifest_hash=split_hash,
        document=MappingProxyType(dict(document)),
    )


def evidence_manifest_record(verification: EvidenceVerification) -> dict[str, Any]:
    """Return the complete preregistered evidence identity for a run manifest."""

    if not verification.required:
        return {"required": False, "verified": True, "status": "not_required"}
    if not verification.verified or verification.document is None:
        raise _fail(
            "EVIDENCE_NOT_VERIFIED", "manifest cannot authorize unverified evidence"
        )
    document = verification.document
    conformal = _object(document["quality_conformal"], "quality_conformal")
    normalization = _object(document["cost_normalization"], "cost_normalization")
    controller_weights = _object(
        document["controller_weight_evidence"], "controller_weight_evidence"
    )
    partition = document.get("quality_partition", {"status": "not_separately_declared"})
    privacy_score_model = _object(
        document["privacy_score_model"], "privacy_score_model"
    )
    profile_quality_support = _object(
        conformal["profile_evaluation_quality_support"],
        "quality_conformal.profile_evaluation_quality_support",
    )
    attack_thresholds = {
        str(attacker["attacker_id"]): attacker["thresholds"]
        for attacker in document["attacker_registry"]
    }
    return {
        "required": True,
        "verified": True,
        "status": "verified",
        "path": str(verification.path),
        "file_sha256": verification.file_sha256,
        "size_bytes": verification.size_bytes,
        "evidence_hash": verification.evidence_hash,
        "split_manifest_hash": verification.split_manifest_hash,
        "split_manifest": document["split_manifest"],
        "quality_partition": partition,
        "quality_partition_identifier": _canonical_hash(
            _object(partition, "quality_partition")
        ),
        "quality_conformal": {
            "identifier": str(
                conformal.get("conformal_id") or _canonical_hash(conformal)
            ),
            "method": conformal["method"],
            "miscoverage": conformal.get("miscoverage"),
            "quantile": conformal["quantile"],
            "order_statistic_rank": conformal.get("order_statistic_rank"),
            "calibration_count": conformal.get("calibration_count"),
            "profile_evaluation_quality_support_hash": profile_quality_support[
                "support_hash"
            ],
            "profile_evaluation_quality_cells": [
                {
                    "region_id": cell["region_id"],
                    "subject_count": cell["subject_count"],
                    "frame_count": cell["frame_count"],
                    "subject_ids_sha256": cell["subject_ids_sha256"],
                }
                for cell in profile_quality_support["cells"]
            ],
        },
        "privacy_score_model": {
            "model_version": privacy_score_model["model_version"],
            "model_hash": privacy_score_model["model_hash"],
            "source_category": privacy_score_model["source_category"],
        },
        "cost_normalization": {
            "calibration_id": normalization.get("calibration_id"),
            "calibration_hash": normalization["calibration_hash"],
            "role": normalization["role"],
            "method": normalization["method"],
            "online_mutable": normalization["online_mutable"],
            "scales": normalization["scales"],
        },
        "controller_weight_evidence": {
            "role": controller_weights["role"],
            "online_mutable": controller_weights["online_mutable"],
            "source_type": controller_weights["source_type"],
            "unit": controller_weights["unit"],
            "method": controller_weights["method"],
            "values": controller_weights["values"],
            "values_sha256": controller_weights["values_sha256"],
        },
        "attack_thresholds": attack_thresholds,
    }


__all__ = [
    "EvidenceVerification",
    "evidence_manifest_record",
    "verify_run_evidence",
]
