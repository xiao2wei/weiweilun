"""Deterministic numerical study generator for paper-scale simulation.

This module is deliberately offline.  It constructs a frozen numerical
population, mutually exclusive subject splits, calibrated numerical attackers,
subject-level simultaneous privacy bounds, FER evidence and paired DES traces.
No generated value is labelled as a hardware measurement or a trained real
vision model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import load_config
from .profiles import (
    canonical_json_bytes,
    canonical_document_sha256,
    compute_subject_risk_ucb,
    load_profile,
)
from .traces import load_trace


PROTOCOL_VERSION = "anon-fer/1.0"
SCHEMA_VERSION = "1.3.0"
COARSE_QUALITY_CLASSES = ("clear", "challenging")
QUALITY_FEATURES = (
    "face_scale",
    "abs_yaw_deg",
    "abs_pitch_deg",
    "blur_index",
    "illumination_deviation",
    "occlusion_fraction",
)
DEVICES = ("vehicle_gpu_class_a", "vehicle_cpu_class_b")
VEHICLES = ("veh-1", "veh-2")
DEVICE_BY_VEHICLE = dict(zip(VEHICLES, DEVICES, strict=True))
RSUS = ("rsu-1", "rsu-2")
EXPRESSIONS = ("neutral", "happy", "sad", "angry")
LOCAL_MODEL = "local_fer_numerical_v1"
EDGE_MODEL = "edge_fer_numerical_v1"
ATTACKERS = (
    "cosine_gallery_numerical_v2",
    "projected_margin_numerical_v2",
    "temporal_link_numerical_v2",
)
RISK_TYPES = ("identity", "verification", "link")
METHOD_IDENTITY_RETENTION = {
    "pixelate": 0.63,
    "blur": 0.59,
    "generative": 0.45,
    "diffusion": 0.38,
}
STRENGTH_IDENTITY_RETENTION = {"weak": 1.0, "medium": 0.60, "strong": 0.08}
QUALITY_SIGNAL_INTERCEPT = 0.72
QUALITY_SIGNAL_SLOPE = 0.28
TEMPORAL_PERSISTENCE_WEIGHTS = {"link": 0.34, "other": 0.10}
ATTACKER_TARGET_INTERCEPTS = {
    "cosine_gallery_numerical_v2": 0.10,
    "projected_margin_numerical_v2": 0.08,
    "temporal_link_numerical_v2": 0.09,
}
ATTACKER_TARGET_SLOPES = {
    "cosine_gallery_numerical_v2": 0.62,
    "projected_margin_numerical_v2": 0.72,
    "temporal_link_numerical_v2": 0.70,
}
RISK_TARGET_OFFSETS = {"identity": 0.0, "verification": 0.025, "link": 0.045}
SPLIT_ROLES = (
    "attack_train",
    "attack_threshold_calibration",
    "quality_calibration",
    "profile_evaluation",
    "scenario_training_validation",
    "test",
)


@dataclass(frozen=True, slots=True)
class NumericalStudySpec:
    seed: int = 20260713
    attack_train_subjects: int = 32
    threshold_calibration_subjects: int = 64
    quality_calibration_subjects: int = 64
    profile_evaluation_subjects: int = 256
    scenario_subjects: int = 48
    test_subjects: int = 64
    frames_per_subject: int = 4
    task_count: int = 24
    horizon_s: float = 20.0
    privacy_threshold: float = 0.35
    confidence_error: float = 0.05
    quality_miscoverage: float = 0.10
    target_false_match_rate: float = 0.01
    min_profile_subjects: int = 64
    quality_tree_depth: int = 2
    quality_min_leaf_subjects: int = 4
    quality_ood_alpha: float = 0.05
    # Preregistered controls for the paper's two-stage variability experiment.
    # Zero removes within-pipeline timing/size variation, one is the reference
    # numerical population, and values above one create deterministic stress
    # variants without breaking joint transaction replay.
    anon_time_variability_scale: float = 1.0
    output_size_variability_scale: float = 1.0
    # Arrival controls are optional so frozen legacy bundles retain their
    # original uniformly spaced, fixed-jitter schedule.  A paper experiment
    # supplies all three values to place a bounded burst at a fixed point in
    # the horizon without changing the horizon's fault/thermal timeline.
    arrival_center_s: float | None = None
    arrival_window_s: float | None = None
    arrival_jitter_fraction: float | None = None
    # ``legacy_last`` preserves the original stress fixture exactly.  Formal
    # studies can instead remove injected failures, select a fixed number of
    # task indices without replacement, or use independent seeded Bernoulli
    # failures.  The schedule is shared by every policy in an environment.
    preprocessing_failure_mode: str = "legacy_last"
    preprocessing_failure_count: int = 0
    preprocessing_failure_probability: float = 0.0
    # Preregistered multiplier for local FER service work and its dynamic
    # energy.  It is deliberately separate from arrival intensity so a
    # calibration can create compute pressure without changing fault timing.
    local_service_scale: float = 1.0

    def validate(self) -> None:
        integers = (
            self.attack_train_subjects,
            self.threshold_calibration_subjects,
            self.quality_calibration_subjects,
            self.profile_evaluation_subjects,
            self.scenario_subjects,
            self.test_subjects,
            self.frames_per_subject,
            self.task_count,
            self.min_profile_subjects,
            self.quality_min_leaf_subjects,
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in integers
        ):
            raise ValueError("all numerical study counts must be positive integers")
        if self.profile_evaluation_subjects < 8:
            raise ValueError("profile evaluation requires at least eight subjects")
        if (
            isinstance(self.quality_tree_depth, bool)
            or not isinstance(self.quality_tree_depth, int)
            or self.quality_tree_depth < 1
        ):
            raise ValueError("quality_tree_depth must be a positive integer")
        if not math.isfinite(self.horizon_s) or self.horizon_s <= 2.0:
            raise ValueError("horizon_s must be finite and greater than two seconds")
        for name, value in (
            ("privacy_threshold", self.privacy_threshold),
            ("confidence_error", self.confidence_error),
            ("quality_miscoverage", self.quality_miscoverage),
            ("target_false_match_rate", self.target_false_match_rate),
            ("quality_ood_alpha", self.quality_ood_alpha),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly inside (0, 1)")
        for name, value in (
            ("anon_time_variability_scale", self.anon_time_variability_scale),
            ("output_size_variability_scale", self.output_size_variability_scale),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 3.0:
                raise ValueError(f"{name} must lie inside [0, 3]")
        arrival_controls = (
            self.arrival_center_s,
            self.arrival_window_s,
            self.arrival_jitter_fraction,
        )
        if any(value is not None for value in arrival_controls):
            if any(value is None for value in arrival_controls):
                raise ValueError(
                    "arrival_center_s, arrival_window_s, and "
                    "arrival_jitter_fraction must be supplied together"
                )
            center = float(self.arrival_center_s)
            window = float(self.arrival_window_s)
            jitter = float(self.arrival_jitter_fraction)
            if not math.isfinite(center) or not math.isfinite(window):
                raise ValueError("arrival center and window must be finite")
            if window <= 0.0:
                raise ValueError("arrival_window_s must be positive")
            if center - window / 2.0 < 0.0 or center + window / 2.0 > self.horizon_s:
                raise ValueError("arrival window must lie inside the horizon")
            if not math.isfinite(jitter) or not 0.0 <= jitter <= 0.5:
                raise ValueError("arrival_jitter_fraction must lie inside [0, 0.5]")
        if self.preprocessing_failure_mode not in {
            "legacy_last",
            "none",
            "fixed_count",
            "bernoulli",
        }:
            raise ValueError(
                "preprocessing_failure_mode must be legacy_last, none, "
                "fixed_count, or bernoulli"
            )
        if (
            isinstance(self.preprocessing_failure_count, bool)
            or not isinstance(self.preprocessing_failure_count, int)
            or self.preprocessing_failure_count < 0
        ):
            raise ValueError("preprocessing_failure_count must be a non-negative integer")
        if (
            self.preprocessing_failure_mode == "fixed_count"
            and self.preprocessing_failure_count > self.task_count
        ):
            raise ValueError("preprocessing_failure_count cannot exceed task_count")
        if (
            not math.isfinite(self.preprocessing_failure_probability)
            or not 0.0 <= self.preprocessing_failure_probability <= 1.0
        ):
            raise ValueError(
                "preprocessing_failure_probability must lie inside [0, 1]"
            )
        if (
            not math.isfinite(self.local_service_scale)
            or not 0.0 < self.local_service_scale <= 10.0
        ):
            raise ValueError("local_service_scale must lie inside (0, 10]")


@dataclass(frozen=True, slots=True)
class NumericalStudyPaths:
    profile_path: Path
    evaluation_trace_path: Path
    scenario_trace_path: Path
    config_path: Path
    evidence_path: Path
    profile_hash: str
    evaluation_trace_hash: str
    scenario_trace_hash: str
    evidence_hash: str


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rng(seed: int, *parts: object) -> random.Random:
    material = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _arrival_time_s(
    spec: NumericalStudySpec, *, trace_seed: int, task_index: int
) -> float:
    """Return a seeded arrival time while preserving the legacy schedule.

    The explicit-window form makes load an independently controllable
    scenario variable.  Keeping the legacy branch verbatim is important for
    replaying bundles whose evidence predates the paper-v1 controls.
    """

    if spec.arrival_window_s is None:
        arrival_time = (task_index + 1) * (spec.horizon_s - 1.6) / (
            spec.task_count + 1
        )
        arrival_time += _rng(trace_seed, "arrival", task_index).uniform(-0.04, 0.04)
        return max(0.01, arrival_time)
    assert spec.arrival_center_s is not None
    assert spec.arrival_jitter_fraction is not None
    interarrival_s = spec.arrival_window_s / (spec.task_count + 1)
    nominal = (
        spec.arrival_center_s
        - spec.arrival_window_s / 2.0
        + (task_index + 1) * interarrival_s
    )
    jitter_s = spec.arrival_jitter_fraction * interarrival_s
    return nominal + _rng(trace_seed, "arrival", task_index).uniform(
        -jitter_s, jitter_s
    )


def _preprocessing_failure_indices(
    spec: NumericalStudySpec, *, trace_seed: int
) -> frozenset[int]:
    """Build a deterministic task-level preprocessing-failure schedule."""

    if spec.preprocessing_failure_mode == "legacy_last":
        return frozenset({spec.task_count - 1})
    if spec.preprocessing_failure_mode == "none":
        return frozenset()
    if spec.preprocessing_failure_mode == "fixed_count":
        indices = list(range(spec.task_count))
        _rng(trace_seed, "preprocessing-failure-schedule").shuffle(indices)
        return frozenset(indices[: spec.preprocessing_failure_count])
    if spec.preprocessing_failure_mode == "bernoulli":
        return frozenset(
            index
            for index in range(spec.task_count)
            if _rng(trace_seed, "preprocessing-failure", index).random()
            < spec.preprocessing_failure_probability
        )
    raise AssertionError("NumericalStudySpec.validate() accepted an unknown mode")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _round(value: float) -> float:
    return round(float(value), 9)


def _source(description: str, unit: str, *, stress: bool = False) -> dict[str, str]:
    return {
        "category": "stress_test_boundary" if stress else "engineering_assumption",
        "description": description,
        "unit": unit,
    }


def numerical_pipeline_ids() -> tuple[str, ...]:
    return tuple(
        f"{method}_{strength}_numerical_v2"
        for method in ("pixelate", "blur", "generative", "diffusion")
        for strength in ("weak", "medium", "strong")
    )


def _pipeline_characteristics(pipeline_id: str) -> dict[str, float | str]:
    method, strength, *_ = pipeline_id.split("_")
    method_retention = METHOD_IDENTITY_RETENTION[method]
    # Numerical engineering assumption: a strong anonymizer leaves only a
    # small residual identity signal.  This value is not a hardware or human
    # measurement; it is frozen into the generated evidence and manifest.
    strength_retention = STRENGTH_IDENTITY_RETENTION[strength]
    work = {"pixelate": 0.020, "blur": 0.027, "generative": 0.066, "diffusion": 0.105}[
        method
    ]
    work *= {"weak": 0.86, "medium": 1.0, "strong": 1.22}[strength]
    bytes_base = {
        "pixelate": 58_000,
        "blur": 66_000,
        "generative": 83_000,
        "diffusion": 91_000,
    }[method]
    utility_drop = {
        "pixelate": 0.09,
        "blur": 0.07,
        "generative": 0.05,
        "diffusion": 0.04,
    }[method]
    utility_drop *= {"weak": 0.75, "medium": 1.0, "strong": 1.55}[strength]
    emission = {"weak": 0.965, "medium": 0.935, "strong": 0.885}[strength]
    return {
        "method": method,
        "strength": strength,
        "identity_retention": method_retention * strength_retention,
        "anon_work_s": work,
        "encoded_size_bytes": float(bytes_base),
        "utility_drop": utility_drop,
        "emission_probability": emission,
    }


def _pipeline_deployment_resource_bounds(pipeline_id: str) -> dict[str, int | float]:
    """Preregistered physical envelope for every allowed numerical stress scale."""

    characteristics = _pipeline_characteristics(pipeline_id)
    method = str(characteristics["method"])
    reference_work = float(characteristics["anon_work_s"])
    reference_bytes = float(characteristics["encoded_size_bytes"])
    # NumericalStudySpec caps both variability controls at three.  These
    # bounds are analytic engineering envelopes over that entire registered
    # range, not maxima learned from either evaluation or scenario rows.
    max_anon_work_s = reference_work * 3.0
    max_output_bytes = math.ceil(reference_bytes * 1.5 + 16_000)
    return {
        "max_peak_memory_bytes": (
            448 * 1024 * 1024
            if method in {"generative", "diffusion"}
            else 96 * 1024 * 1024
        ),
        "max_anon_work_s": _round(max_anon_work_s),
        "max_anon_energy_j": _round(max_anon_work_s * 14.0),
        "max_guard_work_s": 0.010,
        "max_guard_energy_j": 0.060,
        "max_encode_work_s": 0.012,
        "max_encode_energy_j": 0.060,
        "max_output_bytes": max_output_bytes,
    }


def _split_sizes(spec: NumericalStudySpec) -> dict[str, int]:
    return {
        "attack_train": spec.attack_train_subjects,
        "attack_threshold_calibration": spec.threshold_calibration_subjects,
        "quality_calibration": spec.quality_calibration_subjects,
        "profile_evaluation": spec.profile_evaluation_subjects,
        "scenario_training_validation": spec.scenario_subjects,
        "test": spec.test_subjects,
    }


def _make_split_manifest(
    spec: NumericalStudySpec,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    subjects: dict[str, tuple[str, ...]] = {}
    splits: dict[str, Any] = {}
    for role, count in _split_sizes(spec).items():
        ids = tuple(f"numerical-{role}-subject-{index:05d}" for index in range(count))
        videos = tuple(f"video:{subject}:00" for subject in ids)
        frames = tuple(
            f"frame:{video}:{frame:03d}"
            for video in videos
            for frame in range(spec.frames_per_subject)
        )
        subjects[role] = ids
        splits[role] = {
            "subject_count": len(ids),
            "subject_ids": list(ids),
            "subject_ids_sha256": _sha("\n".join(ids)),
            "video_ids_sha256": _sha("\n".join(videos)),
            "frame_ids_sha256": _sha("\n".join(frames)),
        }
    subject_sets = {role: set(ids) for role, ids in subjects.items()}
    intersections = {
        f"{left}__{right}": len(subject_sets[left] & subject_sets[right])
        for index, left in enumerate(SPLIT_ROLES)
        for right in SPLIT_ROLES[index + 1 :]
    }
    manifest = {
        "policy": "subject_video_adjacent_frame_mutually_exclusive",
        "roles": list(SPLIT_ROLES),
        "splits": splits,
        "pairwise_subject_intersection_counts": intersections,
        "all_pairwise_subject_intersections_empty": all(
            v == 0 for v in intersections.values()
        ),
        "assignment_seed": spec.seed,
    }
    manifest["manifest_hash"] = canonical_document_sha256(manifest, "manifest_hash")
    return manifest, subjects


def _quality_observation(seed: int, subject: str, frame: int) -> dict[str, Any]:
    """Create one bounded, multidimensional numerical quality observation.

    The optional shifted component is tied to a stable subject hash and changes
    the features themselves.  OOD is never assigned from an arrival/task index.
    """

    rng = _rng(seed, "quality-features", subject, frame)
    latent = _clamp(rng.betavariate(2.2, 2.0))
    reference_features = {
        "face_scale": _clamp(0.16 + 0.68 * latent + rng.gauss(0.0, 0.035)),
        "abs_yaw_deg": _clamp(
            4.0 + 52.0 * (1.0 - latent) + rng.gauss(0.0, 4.0), 0.0, 90.0
        ),
        "abs_pitch_deg": _clamp(
            3.0 + 36.0 * (1.0 - latent) + rng.gauss(0.0, 3.0), 0.0, 90.0
        ),
        "blur_index": _clamp(0.08 + 0.78 * (1.0 - latent) + rng.gauss(0.0, 0.05)),
        "illumination_deviation": _clamp(
            0.04 + 0.50 * (1.0 - latent) + rng.gauss(0.0, 0.04)
        ),
        "occlusion_fraction": _clamp(
            0.02 + 0.44 * (1.0 - latent) + rng.gauss(0.0, 0.035)
        ),
    }
    shifted = (
        "-test-subject-" in subject
        and _rng(seed, "quality-ood-component", subject).random() < 0.16
    )
    if shifted:
        shift_rng = _rng(seed, "quality-ood-shift", subject, frame)
        reference_features.update(
            {
                "face_scale": 0.025 + 0.015 * shift_rng.random(),
                "abs_yaw_deg": 78.0 + 10.0 * shift_rng.random(),
                "abs_pitch_deg": 62.0 + 18.0 * shift_rng.random(),
                "blur_index": 0.94 + 0.05 * shift_rng.random(),
                "illumination_deviation": 0.90 + 0.08 * shift_rng.random(),
                "occlusion_fraction": 0.86 + 0.12 * shift_rng.random(),
            }
        )
    true_score = _clamp(
        0.29 * reference_features["face_scale"]
        + 0.18 * (1.0 - reference_features["abs_yaw_deg"] / 90.0)
        + 0.12 * (1.0 - reference_features["abs_pitch_deg"] / 90.0)
        + 0.17 * (1.0 - reference_features["blur_index"])
        + 0.10 * (1.0 - reference_features["illumination_deviation"])
        + 0.14 * (1.0 - reference_features["occlusion_fraction"])
    )
    # ``reference_features`` are used only offline to assign the simulator-only
    # region g*.  The controller sees the independently perturbed frozen
    # estimator output ``features`` (u), so region prediction is not a
    # deterministic replay of the partition itself.
    online_rng = _rng(seed, "quality-online-estimator", subject, frame)
    features = {
        "face_scale": _clamp(
            reference_features["face_scale"] + online_rng.gauss(0.0, 0.065)
        ),
        "abs_yaw_deg": _clamp(
            reference_features["abs_yaw_deg"] + online_rng.gauss(0.0, 7.0),
            0.0,
            90.0,
        ),
        "abs_pitch_deg": _clamp(
            reference_features["abs_pitch_deg"] + online_rng.gauss(0.0, 5.5),
            0.0,
            90.0,
        ),
        "blur_index": _clamp(
            reference_features["blur_index"] + online_rng.gauss(0.0, 0.075)
        ),
        "illumination_deviation": _clamp(
            reference_features["illumination_deviation"] + online_rng.gauss(0.0, 0.065)
        ),
        "occlusion_fraction": _clamp(
            reference_features["occlusion_fraction"] + online_rng.gauss(0.0, 0.055)
        ),
    }
    return {
        "subject_id": subject,
        "frame_index": frame,
        "features": {name: _round(features[name]) for name in QUALITY_FEATURES},
        "reference_features": {
            name: _round(reference_features[name]) for name in QUALITY_FEATURES
        },
        "true_quality_score": _round(true_score),
        "coarse_quality_class": "clear" if true_score >= 0.58 else "challenging",
        "numerical_shift_component": shifted,
    }


def _gini(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_field: str,
    labels: Sequence[str],
) -> float:
    if not rows:
        return 0.0
    counts = {
        label: sum(str(row[label_field]) == label for row in rows) for label in labels
    }
    total = float(len(rows))
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _fit_quality_partition(
    rows: Sequence[Mapping[str, Any]], *, max_depth: int, min_leaf_subjects: int
) -> dict[str, Any]:
    """Fit the frozen reference-quality partition whose leaves define G."""

    node_counter = 0
    leaf_counter = 0

    def build(node_rows: Sequence[Mapping[str, Any]], depth: int) -> dict[str, Any]:
        nonlocal node_counter, leaf_counter
        node_id = f"quality-node-{node_counter:03d}"
        node_counter += 1
        parent_impurity = _gini(
            node_rows,
            label_field="coarse_quality_class",
            labels=COARSE_QUALITY_CLASSES,
        )
        best: (
            tuple[float, int, float, list[Mapping[str, Any]], list[Mapping[str, Any]]]
            | None
        ) = None
        if depth < max_depth and parent_impurity > 0.0:
            for feature_index, feature in enumerate(QUALITY_FEATURES):
                values = sorted(
                    {float(row["reference_features"][feature]) for row in node_rows}
                )
                thresholds = [
                    (left + right) / 2.0
                    for left, right in zip(values, values[1:], strict=False)
                ]
                for threshold in thresholds:
                    left_rows = [
                        row
                        for row in node_rows
                        if float(row["reference_features"][feature]) <= threshold
                    ]
                    right_rows = [row for row in node_rows if row not in left_rows]
                    left_subjects = {str(row["subject_id"]) for row in left_rows}
                    right_subjects = {str(row["subject_id"]) for row in right_rows}
                    if (
                        len(left_subjects) < min_leaf_subjects
                        or len(right_subjects) < min_leaf_subjects
                    ):
                        continue
                    weighted = (
                        len(left_rows)
                        * _gini(
                            left_rows,
                            label_field="coarse_quality_class",
                            labels=COARSE_QUALITY_CLASSES,
                        )
                        + len(right_rows)
                        * _gini(
                            right_rows,
                            label_field="coarse_quality_class",
                            labels=COARSE_QUALITY_CLASSES,
                        )
                    ) / len(node_rows)
                    gain = parent_impurity - weighted
                    candidate = (
                        gain,
                        -feature_index,
                        -threshold,
                        left_rows,
                        right_rows,
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
        if best is None or best[0] <= 1e-12:
            region_id = f"quality-region-{leaf_counter:03d}"
            leaf_counter += 1
            counts = {
                label: sum(
                    str(row["coarse_quality_class"]) == label for row in node_rows
                )
                for label in COARSE_QUALITY_CLASSES
            }
            denominator = len(node_rows) + len(COARSE_QUALITY_CLASSES)
            probabilities = {
                label: _round((counts[label] + 1.0) / denominator)
                for label in COARSE_QUALITY_CLASSES
            }
            return {
                "node_id": node_id,
                "kind": "leaf",
                "region_id": region_id,
                "depth": depth,
                "sample_count": len(node_rows),
                "subject_count": len({str(row["subject_id"]) for row in node_rows}),
                "coarse_class_counts": counts,
                "coarse_class_probabilities": probabilities,
                "mean_quality_score": _round(
                    sum(float(row["true_quality_score"]) for row in node_rows)
                    / len(node_rows)
                ),
            }
        _, neg_feature_index, neg_threshold, left_rows, right_rows = best
        feature = QUALITY_FEATURES[-neg_feature_index]
        threshold = -neg_threshold
        return {
            "node_id": node_id,
            "kind": "split",
            "depth": depth,
            "feature": feature,
            "threshold": _round(threshold),
            "left": build(left_rows, depth + 1),
            "right": build(right_rows, depth + 1),
        }

    return build(list(rows), 0)


def _partition_region(tree: Mapping[str, Any], features: Mapping[str, float]) -> str:
    node = tree
    while node["kind"] == "split":
        node = (
            node["left"]
            if float(features[str(node["feature"])]) <= float(node["threshold"])
            else node["right"]
        )
    return str(node["region_id"])


def _partition_regions(tree: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    leaves: list[dict[str, Any]] = []

    def walk(node: Mapping[str, Any]) -> None:
        if node["kind"] == "leaf":
            leaves.append(
                {
                    "region_id": str(node["region_id"]),
                    "mean_quality_score": float(node["mean_quality_score"]),
                    "coarse_class_probabilities": dict(
                        node["coarse_class_probabilities"]
                    ),
                    "subject_count": int(node["subject_count"]),
                }
            )
            return
        walk(node["left"])
        walk(node["right"])

    walk(tree)
    return tuple(sorted(leaves, key=lambda row: row["region_id"]))


def _with_true_region(
    row: Mapping[str, Any], partition_tree: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(row)
    result["true_region_id"] = _partition_region(
        partition_tree, row["reference_features"]
    )
    return result


def _fit_region_classifier(
    rows: Sequence[Mapping[str, Any]],
    *,
    region_ids: Sequence[str],
    max_depth: int,
    min_leaf_subjects: int,
) -> dict[str, Any]:
    """Fit a second frozen CART classifier from noisy online u to leaf IDs."""

    node_counter = 0

    def build(node_rows: Sequence[Mapping[str, Any]], depth: int) -> dict[str, Any]:
        nonlocal node_counter
        node_id = f"quality-classifier-node-{node_counter:03d}"
        node_counter += 1
        parent_impurity = _gini(
            node_rows, label_field="true_region_id", labels=region_ids
        )
        best: (
            tuple[float, int, float, list[Mapping[str, Any]], list[Mapping[str, Any]]]
            | None
        ) = None
        if depth < max_depth and parent_impurity > 0.0:
            for feature_index, feature in enumerate(QUALITY_FEATURES):
                values = sorted({float(row["features"][feature]) for row in node_rows})
                for left_value, right_value in zip(values, values[1:], strict=False):
                    threshold = (left_value + right_value) / 2.0
                    left_rows = [
                        row
                        for row in node_rows
                        if float(row["features"][feature]) <= threshold
                    ]
                    right_rows = [row for row in node_rows if row not in left_rows]
                    if (
                        len({str(row["subject_id"]) for row in left_rows})
                        < min_leaf_subjects
                        or len({str(row["subject_id"]) for row in right_rows})
                        < min_leaf_subjects
                    ):
                        continue
                    weighted = (
                        len(left_rows)
                        * _gini(
                            left_rows,
                            label_field="true_region_id",
                            labels=region_ids,
                        )
                        + len(right_rows)
                        * _gini(
                            right_rows,
                            label_field="true_region_id",
                            labels=region_ids,
                        )
                    ) / len(node_rows)
                    candidate = (
                        parent_impurity - weighted,
                        -feature_index,
                        -threshold,
                        left_rows,
                        right_rows,
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
        if best is None or best[0] <= 1e-12:
            counts = {
                region_id: sum(
                    str(row["true_region_id"]) == region_id for row in node_rows
                )
                for region_id in region_ids
            }
            denominator = len(node_rows) + len(region_ids)
            return {
                "node_id": node_id,
                "kind": "leaf",
                "depth": depth,
                "sample_count": len(node_rows),
                "subject_count": len({str(row["subject_id"]) for row in node_rows}),
                "region_counts": counts,
                "region_probabilities": {
                    region_id: _round((counts[region_id] + 1.0) / denominator)
                    for region_id in region_ids
                },
            }
        _, neg_feature_index, neg_threshold, left_rows, right_rows = best
        feature = QUALITY_FEATURES[-neg_feature_index]
        threshold = -neg_threshold
        return {
            "node_id": node_id,
            "kind": "split",
            "depth": depth,
            "feature": feature,
            "threshold": _round(threshold),
            "left": build(left_rows, depth + 1),
            "right": build(right_rows, depth + 1),
        }

    return build(list(rows), 0)


def _quality_predict(
    classifier: Mapping[str, Any],
    features: Mapping[str, float],
    region_ids: Sequence[str],
) -> dict[str, float]:
    node = classifier
    while node["kind"] == "split":
        node = (
            node["left"]
            if float(features[str(node["feature"])]) <= float(node["threshold"])
            else node["right"]
        )
    return {
        region_id: float(node["region_probabilities"][region_id])
        for region_id in region_ids
    }


def _support_model(training_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    for feature in QUALITY_FEATURES:
        values = sorted(float(row["features"][feature]) for row in training_rows)
        low, high = values[0], values[-1]
        span = max(1e-9, high - low)
        margin = 0.05 * span
        bounds[feature] = {
            "low": _round(low - margin),
            "high": _round(high + margin),
            "scale": _round(span),
        }
    return {"method": "frozen_feature_range_nonconformity", "bounds": bounds}


def _support_score(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    scores = []
    for feature in QUALITY_FEATURES:
        bound = model["bounds"][feature]
        value = float(features[feature])
        scale = float(bound["scale"])
        scores.append(max(0.0, (float(bound["low"]) - value) / scale))
        scores.append(max(0.0, (value - float(bound["high"])) / scale))
    return max(scores)


def _quality_record(
    observation: Mapping[str, Any], quality: Mapping[str, Any]
) -> dict[str, Any]:
    region_ids = tuple(str(value) for value in quality["region_ids"])
    probabilities = _quality_predict(
        quality["region_classifier"], observation["features"], region_ids
    )
    quantile = float(quality["classification_quantile"])
    candidates = sorted(
        region_id
        for region_id in region_ids
        if 1.0 - probabilities[region_id] <= quantile + 1e-12
    )
    support_score = _support_score(quality["support_model"], observation["features"])
    calibration_scores = [float(value) for value in quality["support_scores"]]
    p_value = (1 + sum(value >= support_score for value in calibration_scores)) / (
        len(calibration_scores) + 1
    )
    reference_features = observation.get("reference_features", observation["features"])
    true_region_id = _partition_region(quality["partition_tree"], reference_features)
    region_scores = {
        str(row["region_id"]): float(row["mean_quality_score"])
        for row in quality["regions"]
    }
    predicted_score = sum(
        probabilities[region_id] * region_scores[region_id] for region_id in region_ids
    )
    return {
        "subject_id": observation["subject_id"],
        "quality_features": dict(observation["features"]),
        "true_quality_score": observation["true_quality_score"],
        "predicted_quality_score": _round(predicted_score),
        "true_region_id": true_region_id,
        "region_probabilities": {
            region_id: _round(probabilities[region_id]) for region_id in region_ids
        },
        "candidate_regions": candidates,
        "coarse_quality_class": str(
            observation.get(
                "coarse_quality_class",
                "clear"
                if float(observation["true_quality_score"]) >= 0.58
                else "challenging",
            )
        ),
        "covered": true_region_id in candidates,
        "support_nonconformity": _round(support_score),
        "support_p_value": _round(p_value),
        "ood": p_value <= float(quality["ood_alpha"]) or not candidates,
    }


def _profile_evaluation_quality_support(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
    quality_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the observed profile-evaluation identities and frames in each g*."""

    region_ids = tuple(str(value) for value in quality_model["region_ids"])
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        region_id: {} for region_id in region_ids
    }
    for subject in subjects["profile_evaluation"]:
        for frame in range(spec.frames_per_subject):
            observation = _quality_observation(spec.seed, subject, frame)
            region_id = _partition_region(
                quality_model["partition_tree"], observation["reference_features"]
            )
            grouped[region_id].setdefault(subject, []).append(
                {
                    "frame_index": frame,
                    "true_quality_score": float(observation["true_quality_score"]),
                }
            )

    cells: list[dict[str, Any]] = []
    for region_id in region_ids:
        subject_frames = [
            {
                "subject_id": subject,
                "frames": sorted(frames, key=lambda row: int(row["frame_index"])),
            }
            for subject, frames in sorted(grouped[region_id].items())
        ]
        if not subject_frames:
            raise ValueError(
                "profile-evaluation quality region has no subject/frame support; "
                "increase profile subjects or frames"
            )
        subject_ids = [str(row["subject_id"]) for row in subject_frames]
        cells.append(
            {
                "region_id": region_id,
                "subject_count": len(subject_frames),
                "frame_count": sum(len(row["frames"]) for row in subject_frames),
                "subject_ids_sha256": _sha("\n".join(subject_ids)),
                "subject_frames": subject_frames,
            }
        )
    support: dict[str, Any] = {
        "assignment_version": "numerical-profile-quality-g-star-2.1.0",
        "role": "profile_evaluation",
        "assignment_semantics": "subject contributes to g iff at least one frozen reference frame has true_region_id g; per-subject risk averages only those frames",
        "cells": cells,
        "support_hash": "",
    }
    support["support_hash"] = canonical_document_sha256(support, "support_hash")
    return support


def _conformal_evidence(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    training_rows = [
        _quality_observation(spec.seed, subject, frame)
        for subject in subjects["scenario_training_validation"]
        for frame in range(spec.frames_per_subject)
    ]
    partition_tree = _fit_quality_partition(
        training_rows,
        max_depth=spec.quality_tree_depth,
        min_leaf_subjects=spec.quality_min_leaf_subjects,
    )
    regions = _partition_regions(partition_tree)
    region_ids = tuple(str(row["region_id"]) for row in regions)
    training_rows = [_with_true_region(row, partition_tree) for row in training_rows]
    classifier = _fit_region_classifier(
        training_rows,
        region_ids=region_ids,
        max_depth=spec.quality_tree_depth + 1,
        min_leaf_subjects=spec.quality_min_leaf_subjects,
    )
    calibration_rows = [
        _quality_observation(spec.seed, subject, frame)
        for subject in subjects["quality_calibration"]
        for frame in range(spec.frames_per_subject)
    ]
    calibration_rows = [
        _with_true_region(row, partition_tree) for row in calibration_rows
    ]
    nonconformity = sorted(
        1.0
        - _quality_predict(classifier, row["features"], region_ids)[
            str(row["true_region_id"])
        ]
        for row in calibration_rows
    )
    rank = min(
        len(nonconformity),
        max(
            1,
            math.ceil((len(nonconformity) + 1) * (1.0 - spec.quality_miscoverage)),
        ),
    )
    quantile = nonconformity[rank - 1]
    support = _support_model(training_rows)
    support_scores = sorted(
        _support_score(support, row["features"]) for row in calibration_rows
    )
    quality_model: dict[str, Any] = {
        "partition_tree": partition_tree,
        "region_classifier": classifier,
        "region_ids": region_ids,
        "regions": regions,
        "classification_quantile": quantile,
        "support_model": support,
        "support_scores": support_scores,
        "ood_alpha": spec.quality_ood_alpha,
    }
    profile_evaluation_support = _profile_evaluation_quality_support(
        spec, subjects, quality_model
    )
    test_records = [
        _quality_record(_quality_observation(spec.seed, subject, 0), quality_model)
        for subject in subjects["test"]
    ]
    model_document = {
        "feature_names": list(QUALITY_FEATURES),
        "training_role": "scenario_training_validation",
        "partition_max_depth": spec.quality_tree_depth,
        "classifier_max_depth": spec.quality_tree_depth + 1,
        "min_leaf_subjects": spec.quality_min_leaf_subjects,
        "partition_tree": partition_tree,
        "region_classifier": classifier,
        "region_ids": list(region_ids),
        "classifier": "independent_noisy_feature_cart_with_laplace_leaf_probabilities",
    }
    model_hash = canonical_document_sha256(model_document, "quality_model_hash")
    partition_document = {
        "feature_names": list(QUALITY_FEATURES),
        "training_role": "scenario_training_validation",
        "max_depth": spec.quality_tree_depth,
        "min_leaf_subjects": spec.quality_min_leaf_subjects,
        "partition_tree": partition_tree,
        "region_ids": list(region_ids),
    }
    partition_hash = canonical_document_sha256(partition_document, "partition_hash")
    classifier_document = {
        "feature_names": list(QUALITY_FEATURES),
        "training_role": "scenario_training_validation",
        "max_depth": spec.quality_tree_depth + 1,
        "min_leaf_subjects": spec.quality_min_leaf_subjects,
        "region_classifier": classifier,
        "region_ids": list(region_ids),
    }
    classifier_hash = canonical_document_sha256(classifier_document, "classifier_hash")
    return {
        "method": "split_conformal_region_classifier",
        "conformal_id": "numerical-quality-conformal-2.0.0",
        "quality_model_version": "numerical-quality-system-2.0.0",
        "quality_model_hash": model_hash,
        "partition_version": "numerical-quality-partition-2.0.0",
        "partition_hash": partition_hash,
        "classifier_version": "numerical-quality-classifier-2.0.0",
        "classifier_hash": classifier_hash,
        "feature_names": list(QUALITY_FEATURES),
        "training_role": "scenario_training_validation",
        "max_tree_depth": spec.quality_tree_depth,
        "max_classifier_depth": spec.quality_tree_depth + 1,
        "min_leaf_subjects": spec.quality_min_leaf_subjects,
        "partition_tree": partition_tree,
        "region_ids": list(region_ids),
        "regions": list(regions),
        "region_classifier": classifier,
        "region_classifier_method": "independent_noisy_feature_cart_with_laplace_leaf_probabilities",
        "online_feature_semantics": "noisy frozen estimator output u; distinct from simulator-only reference features used for g*",
        "calibration_role": "quality_calibration",
        "test_role": "test",
        "miscoverage": spec.quality_miscoverage,
        "nonconformity": "1-p_true_region",
        "calibration_count": len(nonconformity),
        "order_statistic_rank": rank,
        "classification_quantile": _round(quantile),
        "quantile": _round(quantile),
        "calibration_nonconformity_scores": [_round(value) for value in nonconformity],
        "support_model": support,
        "support_nonconformity": "max_normalized_outside_training_feature_range",
        "support_scores": [_round(value) for value in support_scores],
        "ood_alpha": spec.quality_ood_alpha,
        "ood_rule": "(1 + calibration_scores_ge_test) / (n + 1) <= alpha",
        "test_coverage": _round(
            sum(row["covered"] for row in test_records) / len(test_records)
        ),
        "mean_candidate_set_size": _round(
            sum(len(row["candidate_regions"]) for row in test_records)
            / len(test_records)
        ),
        "test_records": test_records,
        "profile_evaluation_quality_support": profile_evaluation_support,
    }


def _quality_records_for_subjects(
    seed: int, subjects: Sequence[str], quality: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        _quality_record(_quality_observation(seed, subject, 0), quality)
        for subject in subjects
    ]


def _quality_region_ids(quality: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in quality["region_ids"])


def _quality_region_score(quality: Mapping[str, Any], region_id: str) -> float:
    for row in quality["regions"]:
        if str(row["region_id"]) == region_id:
            return float(row["mean_quality_score"])
    raise KeyError(f"unknown frozen quality region: {region_id}")


def _calibration_threshold(values: Sequence[float], target_fmr: float) -> float:
    ordered = sorted(values)
    allowed = math.floor(target_fmr * len(ordered))
    if allowed < 1:
        return ordered[-1] + 1e-12
    return ordered[len(ordered) - allowed]


def _linear_fit(rows: Sequence[tuple[float, float]]) -> dict[str, float]:
    mean_x = sum(row[0] for row in rows) / len(rows)
    mean_y = sum(row[1] for row in rows) / len(rows)
    variance = sum((row[0] - mean_x) ** 2 for row in rows)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in rows) / variance
        if variance > 1e-12
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in rows]
    residual_std = math.sqrt(sum(value * value for value in residuals) / len(rows))
    return {
        "intercept": _round(intercept),
        "slope": _round(slope),
        "residual_std": _round(max(0.01, residual_std)),
    }


def _attacker_training_target(
    attacker_id: str, risk_type: str, observable_signal: float
) -> float:
    """Frozen score target in the same feature space used at evaluation."""

    return _clamp(
        ATTACKER_TARGET_INTERCEPTS[attacker_id]
        + ATTACKER_TARGET_SLOPES[attacker_id] * observable_signal
        + RISK_TARGET_OFFSETS[risk_type]
    )


def _fit_attacker_parameters(
    spec: NumericalStudySpec,
    attacker_id: str,
    attacker_seed: int,
    training_subjects: Sequence[str],
) -> tuple[dict[str, Any], str]:
    by_risk: dict[str, Any] = {}
    digest_rows: list[str] = []
    for risk_type in RISK_TYPES:
        positive_rows: list[tuple[float, float]] = []
        negative_scores: list[float] = []
        for subject in training_subjects:
            subject_rng = _rng(attacker_seed, "fit-subject", risk_type, subject)
            persistence = subject_rng.betavariate(2.0, 2.0)
            for frame in range(spec.frames_per_subject):
                rng = _rng(attacker_seed, "fit", risk_type, subject, frame)
                signal = _clamp(rng.betavariate(2.0, 1.8))
                # Fit and deploy every attacker in the same observable feature
                # space.  In particular, a temporal target contains
                # persistence only after residual-identity gating.  Fitting a
                # target with an independent latent persistence term would
                # absorb its population mean into the intercept and create
                # linkability even for a fully anonymized artifact.
                observable_signal = _observable_identity_signal(
                    retention=signal,
                    quality_score=1.0,
                    persistence=persistence,
                    attacker_id=attacker_id,
                    risk_type=risk_type,
                )
                target = _clamp(
                    _attacker_training_target(attacker_id, risk_type, observable_signal)
                    + rng.gauss(0.0, 0.035)
                )
                positive_rows.append((observable_signal, target))
                impostor = _clamp(
                    0.09
                    + 0.05 * persistence
                    + (0.035 if attacker_id.startswith("projected") else 0.0)
                    + (0.045 if risk_type == "link" else 0.0)
                    + rng.gauss(0.0, 0.045)
                )
                negative_scores.append(impostor)
                digest_rows.append(
                    f"{risk_type}|{subject}|{frame}|{signal:.12f}|{observable_signal:.12f}|{target:.12f}|{impostor:.12f}"
                )
        fitted = _linear_fit(positive_rows)
        negative_mean = sum(negative_scores) / len(negative_scores)
        negative_std = math.sqrt(
            sum((value - negative_mean) ** 2 for value in negative_scores)
            / len(negative_scores)
        )
        by_risk[risk_type] = {
            **fitted,
            "negative_mean": _round(negative_mean),
            "negative_std": _round(max(0.01, negative_std)),
            "training_observations": len(positive_rows),
        }
    return by_risk, _sha("\n".join(digest_rows))


def _calibrate_attackers(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
    split_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    score_model = _privacy_score_model_document()
    attack_train_hash = split_manifest["splits"]["attack_train"]["subject_ids_sha256"]
    calibration_hash = split_manifest["splits"]["attack_threshold_calibration"][
        "subject_ids_sha256"
    ]
    for attacker_index, attacker_id in enumerate(ATTACKERS):
        attacker_seed = spec.seed + 20_000 + attacker_index * 101
        parameters, training_digest = _fit_attacker_parameters(
            spec,
            attacker_id,
            attacker_seed,
            subjects["attack_train"],
        )
        thresholds: dict[str, Any] = {}
        for risk_type in RISK_TYPES:
            negatives: list[float] = []
            risk_parameters = parameters[risk_type]
            for subject in subjects["attack_threshold_calibration"]:
                for pair in range(12):
                    rng = _rng(attacker_seed, "negative", risk_type, subject, pair)
                    negatives.append(
                        _clamp(
                            rng.gauss(
                                float(risk_parameters["negative_mean"]),
                                float(risk_parameters["negative_std"]),
                            )
                        )
                    )
            threshold = _calibration_threshold(negatives, spec.target_false_match_rate)
            achieved = sum(value >= threshold for value in negatives) / len(negatives)
            thresholds[risk_type] = {
                "threshold_id": f"{attacker_id}:{risk_type}:fmr-{spec.target_false_match_rate:.4f}",
                "threshold": _round(threshold),
                "target_false_match_rate": spec.target_false_match_rate,
                "achieved_calibration_false_match_rate": _round(achieved),
                "calibration_comparisons": len(negatives),
                "negative_score_digest": _sha(
                    "\n".join(f"{value:.12f}" for value in negatives)
                ),
                "decision_rule": (
                    "rank_1_gallery_retrieval"
                    if risk_type == "identity"
                    else "score_greater_equal_calibrated_threshold"
                ),
                "used_for_success": risk_type != "identity",
            }
        parameter_document = {
            "attacker_id": attacker_id,
            "model_kind": (
                "cosine_gallery"
                if attacker_index == 0
                else "projected_margin"
                if attacker_index == 1
                else "temporal_pair_linker"
            ),
            "parameters_by_risk": parameters,
            "training_data_digest": training_digest,
            "score_model_version": score_model["model_version"],
            "score_model_hash": score_model["model_hash"],
        }
        registry.append(
            {
                "attacker_id": attacker_id,
                "model_kind": parameter_document["model_kind"],
                "seed": attacker_seed,
                "parameters_by_risk": parameters,
                "training_data_digest": training_digest,
                "parameter_fingerprint": canonical_document_sha256(
                    parameter_document, "parameter_fingerprint"
                ),
                "score_model_version": score_model["model_version"],
                "score_model_hash": score_model["model_hash"],
                "training_subjects_hash": attack_train_hash,
                "threshold_calibration_subjects_hash": calibration_hash,
                "thresholds": thresholds,
            }
        )
    return registry


def _observable_identity_signal(
    *,
    retention: float,
    quality_score: float,
    persistence: float,
    attacker_id: str,
    risk_type: str,
) -> float:
    """Map residual identity into the signal visible to a frozen attacker.

    Image quality and temporal persistence may amplify *residual* identity,
    but cannot create an identity signal when anonymization retention is zero.
    Keeping that physical boundary explicit prevents a numerical link attacker
    from succeeding solely because a subject-level latent variable exists in
    the generator rather than in the anonymous artifact.
    """

    residual = _clamp(retention) * (
        QUALITY_SIGNAL_INTERCEPT + QUALITY_SIGNAL_SLOPE * _clamp(quality_score)
    )
    if attacker_id == "projected_margin_numerical_v2":
        return _clamp(residual * residual + 0.10 * residual)
    if attacker_id == "temporal_link_numerical_v2":
        temporal_weight = TEMPORAL_PERSISTENCE_WEIGHTS[
            "link" if risk_type == "link" else "other"
        ]
        observable_persistence = _clamp(retention) * _clamp(persistence)
        return _clamp(0.66 * residual + temporal_weight * observable_persistence)
    return residual


def _privacy_score_model_document() -> dict[str, Any]:
    """Return the hashable, frozen numerical privacy-score parameterization."""

    model: dict[str, Any] = {
        "model_version": "numerical-privacy-score-2.1.0",
        "source_category": "engineering_assumption",
        "method_identity_retention": dict(METHOD_IDENTITY_RETENTION),
        "strength_identity_retention": dict(STRENGTH_IDENTITY_RETENTION),
        "quality_multiplier": {
            "intercept": QUALITY_SIGNAL_INTERCEPT,
            "slope": QUALITY_SIGNAL_SLOPE,
        },
        "projected_feature": "residual_squared_plus_0.10_residual",
        "temporal_persistence_weights": dict(TEMPORAL_PERSISTENCE_WEIGHTS),
        "persistence_semantics": "subject persistence is observable only through residual identity retention",
        "attacker_target_intercepts": dict(ATTACKER_TARGET_INTERCEPTS),
        "attacker_target_slopes": dict(ATTACKER_TARGET_SLOPES),
        "risk_target_offsets": dict(RISK_TARGET_OFFSETS),
        "fit_deploy_feature_identity": True,
        "model_hash": "",
    }
    model["model_hash"] = canonical_document_sha256(model, "model_hash")
    return model


def _attack_score(
    spec: NumericalStudySpec,
    subject: str,
    frame: int,
    pipeline_id: str,
    quality_region: str,
    quality_score: float,
    attacker: Mapping[str, Any],
    risk_type: str,
) -> tuple[float, float]:
    characteristics = _pipeline_characteristics(pipeline_id)
    attacker_id = str(attacker["attacker_id"])
    rng = _rng(
        attacker["seed"],
        "positive",
        risk_type,
        subject,
        frame,
        pipeline_id,
        quality_region,
    )
    parameters = attacker["parameters_by_risk"][risk_type]
    retention = float(characteristics["identity_retention"])
    persistence = _rng(attacker["seed"], "persistence", subject).betavariate(2.0, 2.0)
    signal = _observable_identity_signal(
        retention=retention,
        quality_score=quality_score,
        persistence=persistence,
        attacker_id=attacker_id,
        risk_type=risk_type,
    )
    true_score = _clamp(
        float(parameters["intercept"])
        + float(parameters["slope"]) * signal
        + rng.gauss(0.0, float(parameters["residual_std"]))
    )
    impostor_max = max(
        _clamp(
            rng.gauss(
                float(parameters["negative_mean"]),
                float(parameters["negative_std"]),
            )
        )
        for _ in range(24)
    )
    return true_score, impostor_max


def _privacy_evidence(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
    attackers: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    *,
    pipeline_ids: Sequence[str] | None = None,
    registered_hypotheses: int | None = None,
) -> list[dict[str, Any]]:
    region_ids = _quality_region_ids(quality)
    support_document = quality["profile_evaluation_quality_support"]
    support_by_region = {
        str(row["region_id"]): row for row in support_document["cells"]
    }
    if set(support_by_region) != set(region_ids):
        raise ValueError("profile-evaluation quality support does not cover every g")
    registered_profile_subjects = set(subjects["profile_evaluation"])
    full_hypothesis_count = (
        len(numerical_pipeline_ids())
        * len(region_ids)
        * len(attackers)
        * len(RISK_TYPES)
    )
    hypothesis_count = (
        full_hypothesis_count
        if registered_hypotheses is None
        else int(registered_hypotheses)
    )
    if hypothesis_count < full_hypothesis_count:
        raise ValueError(
            "privacy evidence subset cannot reduce the preregistered hypothesis family"
        )
    selected_pipelines = (
        numerical_pipeline_ids()
        if pipeline_ids is None
        else tuple(dict.fromkeys(pipeline_ids))
    )
    if not selected_pipelines or not set(selected_pipelines) <= set(
        numerical_pipeline_ids()
    ):
        raise ValueError("privacy evidence pipeline subset is empty or unknown")
    evidence: list[dict[str, Any]] = []
    for pipeline_id in selected_pipelines:
        characteristics = _pipeline_characteristics(pipeline_id)
        for quality_region in region_ids:
            quality_support = support_by_region[quality_region]
            if any(
                str(row["subject_id"]) not in registered_profile_subjects
                for row in quality_support["subject_frames"]
            ):
                raise ValueError(
                    "quality-cell support references a subject outside profile evaluation"
                )
            for attacker in attackers:
                for risk_type in RISK_TYPES:
                    threshold_record = attacker["thresholds"][risk_type]
                    stage_subject_rows: dict[str, list[tuple[float, float]]] = {
                        "single_attempt": [],
                        "guard_selected": [],
                        "guard_plus_retry_final": [],
                    }
                    for subject_record in quality_support["subject_frames"]:
                        subject = str(subject_record["subject_id"])
                        cell_frames = subject_record["frames"]
                        stage_counts = {stage: [0, 0] for stage in stage_subject_rows}
                        for frame_record in cell_frames:
                            frame = int(frame_record["frame_index"])
                            quality_score = float(frame_record["true_quality_score"])
                            emission_probability = float(
                                characteristics["emission_probability"]
                            ) - 0.10 * (1.0 - quality_score)
                            attempts: list[tuple[bool, bool]] = []
                            for attempt in range(1, 4):
                                emit_rng = _rng(
                                    spec.seed,
                                    "guard",
                                    subject,
                                    frame,
                                    pipeline_id,
                                    quality_region,
                                    attempt,
                                )
                                emitted = emit_rng.random() < max(
                                    0.0, emission_probability - 0.018 * (attempt - 1)
                                )
                                true_score, impostor_max = _attack_score(
                                    spec,
                                    subject,
                                    frame * 10 + attempt,
                                    pipeline_id,
                                    quality_region,
                                    quality_score,
                                    attacker,
                                    risk_type,
                                )
                                if risk_type == "identity":
                                    success = true_score > impostor_max
                                else:
                                    threshold = float(threshold_record["threshold"])
                                    success = true_score >= threshold
                                attempts.append((emitted, success))

                            first_emit, first_success = attempts[0]
                            stage_counts["single_attempt"][0] += int(first_success)
                            stage_counts["single_attempt"][1] += 1
                            stage_counts["guard_selected"][0] += int(
                                first_emit and first_success
                            )
                            stage_counts["guard_selected"][1] += int(first_emit)
                            final = next(
                                (
                                    (attempt_emit, attempt_success)
                                    for attempt_emit, attempt_success in attempts
                                    if attempt_emit
                                ),
                                (False, False),
                            )
                            stage_counts["guard_plus_retry_final"][0] += int(
                                final[0] and final[1]
                            )
                            stage_counts["guard_plus_retry_final"][1] += int(final[0])
                        denominator = float(len(cell_frames))
                        for stage, (attacked, emitted) in stage_counts.items():
                            stage_subject_rows[stage].append(
                                (attacked / denominator, emitted / denominator)
                            )
                    stored_stages = {
                        stage: [(_round(x), _round(y)) for x, y in rows]
                        for stage, rows in stage_subject_rows.items()
                    }
                    stage_statistics = {}
                    for stage, rows in stored_stages.items():
                        statistics = compute_subject_risk_ucb(
                            rows,
                            registered_hypotheses=hypothesis_count,
                            confidence_error=spec.confidence_error,
                        )
                        stage_statistics[stage] = {
                            key: _round(value) if isinstance(value, float) else value
                            for key, value in asdict(statistics).items()
                        }
                    final_rows = stored_stages["guard_plus_retry_final"]
                    final_statistics = stage_statistics["guard_plus_retry_final"]
                    evidence.append(
                        {
                            "pipeline_id": pipeline_id,
                            "quality_bin": quality_region,
                            "attacker_id": attacker["attacker_id"],
                            "risk_type": risk_type,
                            "threshold_id": threshold_record["threshold_id"],
                            "profile_quality_support_hash": support_document[
                                "support_hash"
                            ],
                            "quality_cell_subject_ids_sha256": quality_support[
                                "subject_ids_sha256"
                            ],
                            "quality_cell_subject_count": quality_support[
                                "subject_count"
                            ],
                            "quality_cell_frame_count": quality_support["frame_count"],
                            "stage_protocol": {
                                "single_attempt": "first anonymous attempt before guard selection; denominator is each subject's observed frames in quality cell g",
                                "guard_selected": "first attempt retained only when its guard passes",
                                "guard_plus_retry_final": "first guard-passing output within the frozen three-attempt cap",
                            },
                            "stage_subject_rows": {
                                stage: [list(row) for row in rows]
                                for stage, rows in stored_stages.items()
                            },
                            "stage_statistics": stage_statistics,
                            "subject_rows": [list(row) for row in final_rows],
                            "statistics": final_statistics,
                        }
                    )
    return evidence


def recompute_privacy_evidence(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute every frozen privacy statistic from stored subject rows."""

    count = int(evidence["privacy_protocol"]["registered_hypotheses"])
    delta = float(evidence["privacy_protocol"]["confidence_error"])
    result: list[dict[str, Any]] = []
    for cell in evidence["privacy_evidence"]:
        stats = compute_subject_risk_ucb(
            ((float(row[0]), float(row[1])) for row in cell["subject_rows"]),
            registered_hypotheses=count,
            confidence_error=delta,
        )
        result.append(
            {
                key: _round(value) if isinstance(value, float) else value
                for key, value in asdict(stats).items()
            }
        )
    return result


def _fer_probabilities(
    seed: int,
    subject: str,
    label: str,
    quality_region: str,
    quality_score: float,
    model_id: str,
    pipeline_id: str | None,
) -> tuple[float, ...]:
    rng = _rng(
        seed,
        "fer",
        subject,
        label,
        quality_region,
        model_id,
        pipeline_id or "raw",
    )
    correct = 0.84 if model_id == EDGE_MODEL else 0.78
    correct -= 0.15 * (1.0 - _clamp(quality_score))
    if pipeline_id is not None:
        correct -= float(_pipeline_characteristics(pipeline_id)["utility_drop"])
    correct = _clamp(correct + rng.gauss(0.0, 0.035), 0.28, 0.94)
    other_weights = [0.2 + rng.random() for _ in range(len(EXPRESSIONS) - 1)]
    scale = (1.0 - correct) / sum(other_weights)
    values: list[float] = []
    cursor = 0
    for expression in EXPRESSIONS:
        if expression == label:
            values.append(correct)
        else:
            values.append(other_weights[cursor] * scale)
            cursor += 1
    return tuple(values)


def classification_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute deterministic multi-class FER metrics from frozen probabilities."""

    rows = tuple(records)
    if not rows:
        raise ValueError("classification metrics require at least one record")
    confusion = {label: {pred: 0 for pred in EXPRESSIONS} for label in EXPRESSIONS}
    nll = 0.0
    calibration_bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    for row in rows:
        label = str(row["label"])
        probabilities = tuple(float(value) for value in row["probabilities"])
        if label not in EXPRESSIONS or len(probabilities) != len(EXPRESSIONS):
            raise ValueError("FER record label/probability dimensions are invalid")
        if any(
            value < 0.0 or value > 1.0 for value in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7):
            raise ValueError("FER probabilities must be normalized")
        predicted_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        predicted = EXPRESSIONS[predicted_index]
        confusion[label][predicted] += 1
        nll -= math.log(max(1e-12, probabilities[EXPRESSIONS.index(label)]))
        confidence = probabilities[predicted_index]
        calibration_bins[min(9, int(confidence * 10.0))].append(
            (confidence, float(predicted == label))
        )
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for label in EXPRESSIONS:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in EXPRESSIONS)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        recalls[label] = _round(recall)
        f1s.append(
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    correct = sum(confusion[label][label] for label in EXPRESSIONS)
    ece = sum(
        len(bucket)
        / len(rows)
        * abs(
            sum(v[0] for v in bucket) / len(bucket)
            - sum(v[1] for v in bucket) / len(bucket)
        )
        for bucket in calibration_bins
        if bucket
    )
    balanced = sum(recalls.values()) / len(EXPRESSIONS)
    return {
        "count": len(rows),
        "accuracy": _round(correct / len(rows)),
        "macro_f1": _round(sum(f1s) / len(f1s)),
        "balanced_accuracy": _round(balanced),
        "nll": _round(nll / len(rows)),
        "ece_10_bin": _round(ece),
        "per_class_recall": recalls,
        "confusion": confusion,
    }


def _fer_evidence(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
    quality: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for quality_region in _quality_region_ids(quality):
        quality_score = _quality_region_score(quality, quality_region)
        local_rows: list[dict[str, Any]] = []
        for index, subject in enumerate(subjects["test"]):
            label = EXPRESSIONS[index % len(EXPRESSIONS)]
            probabilities = _fer_probabilities(
                spec.seed,
                subject,
                label,
                quality_region,
                quality_score,
                LOCAL_MODEL,
                None,
            )
            local_rows.append(
                {
                    "subject_id": subject,
                    "label": label,
                    "probabilities": list(probabilities),
                }
            )
        local_metrics = classification_metrics(local_rows)
        metrics.append(
            {
                "model_id": LOCAL_MODEL,
                "pipeline_id": None,
                "quality_bin": quality_region,
                **local_metrics,
            }
        )
        local_by_subject = {row["subject_id"]: row for row in local_rows}
        for pipeline_id in numerical_pipeline_ids():
            edge_rows: list[dict[str, Any]] = []
            deltas: list[float] = []
            for index, subject in enumerate(subjects["test"]):
                label = EXPRESSIONS[index % len(EXPRESSIONS)]
                probabilities = _fer_probabilities(
                    spec.seed,
                    subject,
                    label,
                    quality_region,
                    quality_score,
                    EDGE_MODEL,
                    pipeline_id,
                )
                row = {
                    "subject_id": subject,
                    "label": label,
                    "probabilities": list(probabilities),
                }
                edge_rows.append(row)
                local = local_by_subject[subject]["probabilities"]
                local_nll = -math.log(max(1e-12, local[EXPRESSIONS.index(label)]))
                anon_nll = -math.log(
                    max(1e-12, probabilities[EXPRESSIONS.index(label)])
                )
                deltas.append(anon_nll - local_nll)
                paired.append(
                    {
                        "subject_id": subject,
                        "pipeline_id": pipeline_id,
                        "quality_bin": quality_region,
                        "label": label,
                        "local_nll": _round(local_nll),
                        "anonymous_edge_nll": _round(anon_nll),
                        "paired_nll_delta": _round(anon_nll - local_nll),
                        "local_correct": (
                            EXPRESSIONS[max(range(len(local)), key=local.__getitem__)]
                            == label
                        ),
                        "anonymous_edge_correct": (
                            EXPRESSIONS[
                                max(
                                    range(len(probabilities)),
                                    key=probabilities.__getitem__,
                                )
                            ]
                            == label
                        ),
                    }
                )
            edge_metrics = classification_metrics(edge_rows)
            metrics.append(
                {
                    "model_id": EDGE_MODEL,
                    "pipeline_id": pipeline_id,
                    "quality_bin": quality_region,
                    "paired_mean_nll_delta_vs_local": _round(sum(deltas) / len(deltas)),
                    **edge_metrics,
                }
            )
    return metrics, paired


def _robust_positive_median(values: Sequence[float]) -> float:
    ordered = sorted(value for value in values if math.isfinite(value) and value > 0.0)
    if not ordered:
        raise ValueError("normalization calibration contains no positive finite values")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _cost_normalization_evidence(
    spec: NumericalStudySpec,
    subjects: Mapping[str, tuple[str, ...]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze robust scales using only preregistered training/validation rows."""

    records: list[dict[str, Any]] = []
    quality_records = _quality_records_for_subjects(
        spec.seed, subjects["scenario_training_validation"], quality
    )
    for index, record in enumerate(quality_records):
        quality_region = str(record["true_region_id"])
        quality_score = _quality_region_score(quality, quality_region)
        difficulty = 1.0 - quality_score
        device_index = index % len(DEVICES)
        prep_work = (0.022 if device_index == 0 else 0.039) * (1.0 + 0.18 * difficulty)
        local_work = (
            (0.043 if device_index == 0 else 0.081)
            * (1.0 + 0.22 * difficulty)
            * spec.local_service_scale
        )
        local_power = 11.8 if device_index == 0 else 7.6
        label = EXPRESSIONS[index % len(EXPRESSIONS)]
        probabilities = _fer_probabilities(
            spec.seed,
            str(record["subject_id"]),
            label,
            quality_region,
            quality_score,
            LOCAL_MODEL,
            None,
        )
        utility_loss = 1.0 - probabilities[EXPRESSIONS.index(label)]
        characteristics = _pipeline_characteristics("blur_medium_numerical_v2")
        rsu_energy = (0.017 + 0.006 * difficulty) * 40.0
        records.append(
            {
                "subject_id": record["subject_id"],
                "quality_bin": quality_region,
                "baseline_policy": "preregistered_local_plus_fixed_safe_edge",
                "latency_s": _round(
                    prep_work
                    + local_work
                    + float(characteristics["anon_work_s"])
                    + 0.020
                ),
                "vehicle_energy_j": _round(
                    prep_work * (11.2 if device_index == 0 else 7.4)
                    + local_work * local_power
                    + float(characteristics["anon_work_s"])
                    * (12.6 if device_index == 0 else 8.1)
                ),
                "rsu_energy_j": _round(rsu_energy),
                "utility_loss": _round(utility_loss),
            }
        )
    scales = {
        "latency_scale_s": _round(
            _robust_positive_median([float(row["latency_s"]) for row in records])
        ),
        "vehicle_energy_scale_j": _round(
            _robust_positive_median([float(row["vehicle_energy_j"]) for row in records])
        ),
        "rsu_energy_scale_j": _round(
            _robust_positive_median([float(row["rsu_energy_j"]) for row in records])
        ),
        "utility_scale": _round(
            _robust_positive_median([float(row["utility_loss"]) for row in records])
        ),
    }
    document: dict[str, Any] = {
        "calibration_id": "numerical-cost-scales-2.0.0",
        "role": "scenario_training_validation",
        "method": "median_of_positive_preregistered_baseline_records",
        "online_mutable": False,
        "records": records,
        "scales": scales,
        "calibration_hash": "",
    }
    document["calibration_hash"] = canonical_document_sha256(
        document, "calibration_hash"
    )
    return document


def _build_evidence(
    spec: NumericalStudySpec,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    split_manifest, subjects = _make_split_manifest(spec)
    quality = _conformal_evidence(spec, subjects)
    privacy_score_model = _privacy_score_model_document()
    attackers = _calibrate_attackers(spec, subjects, split_manifest)
    privacy = _privacy_evidence(spec, subjects, attackers, quality)
    fer_metrics, paired = _fer_evidence(spec, subjects, quality)
    cost_normalization = _cost_normalization_evidence(spec, subjects, quality)
    guard_seed = spec.seed + 10_001
    attacker_seeds = [int(attacker["seed"]) for attacker in attackers]
    controller_weight_values = {
        "physical_queue_weight": 1.0,
        "vehicle_resource_theta": {
            "accelerator": 1.0,
            "cpu": 1.0,
            "encoder": 1.0,
        },
        "rsu_resource_theta": {"ingress": 1.0, "gpu": 1.0},
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_version": "numerical-study-2.1.0",
        "evidence_hash": "",
        "data_kind": "numerical_simulation",
        "evidence_status": "frozen_numerical_model",
        "description": "Deterministic latent-variable numerical study; no real image, person, attacker model or hardware measurement is present.",
        "spec": asdict(spec),
        "split_manifest": split_manifest,
        "guard_registry": {
            "guard_id": "guard_numerical_v1",
            "seed": guard_seed,
            "parameter_fingerprint": _sha(f"numerical-guard|{guard_seed}"),
            "calibration_role": "quality_calibration",
        },
        "privacy_score_model": privacy_score_model,
        "attacker_registry": attackers,
        "isolation_checks": {
            "guard_seed_disjoint_from_attackers": guard_seed not in attacker_seeds,
            "guard_parameter_disjoint_from_attackers": all(
                _sha(f"numerical-guard|{guard_seed}")
                != attacker["parameter_fingerprint"]
                for attacker in attackers
            ),
            "attack_training_threshold_profile_subjects_mutually_exclusive": split_manifest[
                "all_pairwise_subject_intersections_empty"
            ],
        },
        "privacy_protocol": {
            "registered_hypotheses": len(numerical_pipeline_ids())
            * len(_quality_region_ids(quality))
            * len(ATTACKERS)
            * len(RISK_TYPES),
            "registered_pipeline_ids": list(numerical_pipeline_ids()),
            "registered_quality_region_ids": list(_quality_region_ids(quality)),
            "registered_attacker_ids": list(ATTACKERS),
            "registered_risk_types": list(RISK_TYPES),
            "confidence_error": spec.confidence_error,
            "bound": "subject_level_bonferroni_hoeffding_ratio",
            "unit_of_independence": "subject",
        },
        "privacy_evidence": privacy,
        "post_selection_risk_report": [
            {
                "pipeline_id": row["pipeline_id"],
                "quality_bin": row["quality_bin"],
                "attacker_id": row["attacker_id"],
                "risk_type": row["risk_type"],
                "threshold_id": row["threshold_id"],
                "stages": row["stage_statistics"],
                "final_stage": "guard_plus_retry_final",
                "reported_quantities": [
                    "conditional_risk_ucb",
                    "joint_risk_ucb",
                    "mean_emission",
                    "emission_lcb",
                ],
            }
            for row in privacy
        ],
        "quality_partition": {
            "partition_id": quality["partition_version"],
            "partition_hash": quality["partition_hash"],
            "feature_names": quality["feature_names"],
            "training_role": quality["training_role"],
            "max_depth": quality["max_tree_depth"],
            "min_leaf_subjects": quality["min_leaf_subjects"],
            "region_ids": quality["region_ids"],
            "regions": quality["regions"],
            "tree": quality["partition_tree"],
        },
        "quality_conformal": quality,
        "cost_normalization": cost_normalization,
        "controller_weight_evidence": {
            "role": "scenario_training_validation",
            "online_mutable": False,
            "source_type": "engineering_assumption",
            "unit": "dimensionless_busy_second_quadratic_weight",
            "method": "preregistered_unit_weights_not_data_fitted",
            "values": controller_weight_values,
            "values_sha256": hashlib.sha256(
                canonical_json_bytes(controller_weight_values)
            ).hexdigest(),
        },
        "fer_metrics": fer_metrics,
        "fer_paired_records": paired,
        "parameter_sources": {
            "latent_population": _source(
                "Bounded latent identity, expression and quality population used only for numerical simulation.",
                "dimensionless",
            ),
            "attacker_scores": _source(
                "Frozen numerical v2 score model calibrated on a subject-disjoint split; exact retention, observable-signal and target parameters are stored in privacy_score_model.",
                "score",
            ),
            "identity_retention": _source(
                "Method and strength residual-identity multipliers used by the numerical privacy score model; exact values are frozen in privacy_score_model.",
                "dimensionless",
            ),
            "fer_probabilities": _source(
                "Frozen normalized class probabilities from a bounded numerical FER response model.",
                "probability",
            ),
            "two_stage_variability_controls": _source(
                "Preregistered deterministic multipliers for within-pipeline anonymization time and encoded-size variation; one is the reference numerical population.",
                "dimensionless",
                stress=(
                    spec.anon_time_variability_scale > 1.0
                    or spec.output_size_variability_scale > 1.0
                ),
            ),
        },
    }
    evidence["evidence_hash"] = canonical_document_sha256(evidence, "evidence_hash")
    return evidence, subjects


def _component_hash(name: str) -> str:
    return _sha(f"privacy-edge-sim|frozen-numerical-model|{name}|v1")


def _build_profile(
    spec: NumericalStudySpec, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    quality = evidence["quality_conformal"]
    region_ids = _quality_region_ids(quality)
    guard_hash = _component_hash("guard:guard_numerical_v1")
    encoder_hash = _component_hash("encoder:jpeg_numerical_v1")
    pipelines = [
        {
            "pipeline_id": pipeline_id,
            "pipeline_hash": _component_hash(f"pipeline:{pipeline_id}"),
            "guard_id": "guard_numerical_v1",
            "guard_hash": guard_hash,
            "encoder_id": "jpeg_numerical_v1",
            "encoder_hash": encoder_hash,
            "protocol_version": PROTOCOL_VERSION,
            "max_attempts": 3,
            "fallback_local_model": LOCAL_MODEL,
            "supported_devices": list(DEVICES),
            "retryable_reasons": [
                "ANON_OOM",
                "ANON_FAILED",
                "GUARD_REJECTED",
                "ENCODE_FAILED",
            ],
            "deployment_resource_bounds": _pipeline_deployment_resource_bounds(
                pipeline_id
            ),
        }
        for pipeline_id in numerical_pipeline_ids()
    ]
    evidence_index = {
        (row["pipeline_id"], row["quality_bin"]): []
        for row in evidence["privacy_evidence"]
    }
    for row in evidence["privacy_evidence"]:
        evidence_index[(row["pipeline_id"], row["quality_bin"])].append(row)
    cells: list[dict[str, Any]] = []
    for pipeline_id in numerical_pipeline_ids():
        for quality_bin in region_ids:
            bounds = [
                {
                    "risk_type": row["risk_type"],
                    "attacker_id": row["attacker_id"],
                    "threshold_id": row["threshold_id"],
                    "ucb": row["statistics"]["conditional_risk_ucb"],
                    "subject_count": row["statistics"]["subject_count"],
                    "emission_lcb": row["statistics"]["emission_lcb"],
                    "confidence_error": spec.confidence_error,
                }
                for row in sorted(
                    evidence_index[(pipeline_id, quality_bin)],
                    key=lambda item: (item["risk_type"], item["attacker_id"]),
                )
            ]
            for device in DEVICES:
                cells.append(
                    {
                        "pipeline_id": pipeline_id,
                        "quality_bin": quality_bin,
                        "device_type": device,
                        "joint_trace_supported": True,
                        "bounds": bounds,
                    }
                )
    min_subjects = spec.min_profile_subjects
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "profile_version": "numerical-2.2.0",
        "profile_hash": "",
        "data_kind": "numerical_simulation",
        "evidence_status": "frozen_numerical_model",
        "online_mutable": False,
        # The schema retains the historical field name ``quality_bins`` for
        # compatibility, but every value is a frozen partition leaf ID g.
        "quality_bins": list(region_ids),
        "preprocessing_resource_bounds": {
            "max_memory_bytes": 64 * 1024 * 1024,
            "max_service_work_s": 0.060,
            "max_dynamic_energy_j": 0.70,
        },
        "privacy_policy": {
            "registered_risk_types": list(RISK_TYPES),
            "risk_threshold": spec.privacy_threshold,
            "confidence_error": spec.confidence_error,
            "min_subjects": min_subjects,
            "min_emission_lcb": 0.50,
            "interpretation": "Numerical subject-level simultaneous empirical bound for frozen simulated attackers only; not absolute anonymity.",
        },
        "pipelines": pipelines,
        "local_models": [
            {
                "model_id": LOCAL_MODEL,
                "model_hash": _component_hash(f"model:{LOCAL_MODEL}"),
                "model_kind": "local",
                "protocol_version": PROTOCOL_VERSION,
                "supported_devices": list(DEVICES),
                "supported_rsus": [],
                "supported_pipelines": [],
                "deployment_resource_bounds": {
                    "max_memory_bytes": 256 * 1024 * 1024,
                    # These retain the v11 values at scale one while
                    # remaining an analytic envelope for preregistered local
                    # compute-pressure variants.
                    "max_service_work_s": _round(
                        0.080 * spec.local_service_scale
                    ),
                    "max_dynamic_energy_j": _round(
                        0.70 * spec.local_service_scale
                    ),
                },
            }
        ],
        "edge_models": [
            {
                "model_id": EDGE_MODEL,
                "model_hash": _component_hash(f"model:{EDGE_MODEL}"),
                "model_kind": "edge",
                "protocol_version": PROTOCOL_VERSION,
                "supported_devices": [],
                "supported_rsus": list(RSUS),
                "supported_pipelines": list(numerical_pipeline_ids()),
                "deployment_resource_bounds": {
                    "max_vram_bytes": 768 * 1024 * 1024,
                    "max_ingress_work_s": 0.005,
                    "max_ingress_energy_j": 0.080,
                    "max_gpu_work_s": 0.030,
                    "max_gpu_energy_j": 1.20,
                    "max_result_size_bits": 4096,
                },
            }
        ],
        "privacy_cells": cells,
        "parameter_sources": {
            "privacy_bounds": _source(
                "Recomputed from stored numerical subject outcomes using simultaneous Bonferroni-Hoeffding bounds.",
                "probability",
            ),
            "subject_support": _source(
                "Mutually exclusive numerical profile-evaluation subjects.", "subjects"
            ),
            "emission_support": _source(
                "Numerical guard emission lower confidence bounds.", "probability"
            ),
            "retry_limits": _source(
                "Finite numerical transaction retry cap.", "attempts"
            ),
            "deployment_resource_bounds": _source(
                "Preregistered analytic finite physical envelopes over all allowed numerical variability controls; trace rows outside them are OOD.",
                "resource_busy_s,J,bytes,bit",
            ),
        },
        "metadata": {
            "generator": "privacy_edge_sim.numerical.generate_numerical_study",
            "evidence_hash": evidence["evidence_hash"],
            "split_manifest_hash": evidence["split_manifest"]["manifest_hash"],
            "quality_partition_hash": evidence["quality_partition"]["partition_hash"],
            "quality_classifier_hash": quality["classifier_hash"],
            "quality_conformal_id": evidence["quality_conformal"]["conformal_id"],
            "quality_regions": quality["regions"],
            "cost_normalization_hash": evidence["cost_normalization"][
                "calibration_hash"
            ],
            "formal_experiment_eligible": False,
            "numerical_experiment_eligible": True,
            "contains_real_images": False,
            "contains_real_measurements": False,
            "warning": "FROZEN NUMERICAL MODEL - NOT HARDWARE OR ATTACK MEASUREMENT",
        },
    }
    profile["profile_hash"] = canonical_document_sha256(profile, "profile_hash")
    return profile


def _contexts() -> tuple[dict[str, str], ...]:
    return (
        {
            "thermal_state": "nominal",
            "power_mode": "nominal",
            "memory_pressure": "normal",
        },
        {
            "thermal_state": "hot_throttled",
            "power_mode": "nominal",
            "memory_pressure": "normal",
        },
        {
            "thermal_state": "recovered",
            "power_mode": "nominal",
            "memory_pressure": "normal",
        },
    )


def _attempt(
    index: int,
    work: float,
    energy: float,
    memory: int,
    *,
    oom: bool = False,
    guard: bool | None = None,
    encoded_bytes: int | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    guard_work = None if oom else 0.0075
    guard_energy = None if oom else 0.045
    encode = guard is True
    return {
        "attempt_index": index,
        "anon_work_s": _round(work),
        "anon_energy_j": _round(energy),
        "peak_memory_bytes": memory,
        "anon_oom": oom,
        "guard_work_s": guard_work,
        "guard_energy_j": guard_energy,
        "guard_passed": None if oom else guard,
        "encode_work_s": _round(0.006 + (encoded_bytes or 0) / 50_000_000.0)
        if encode
        else None,
        "encode_energy_j": _round(0.035 + (encoded_bytes or 0) / 10_000_000.0)
        if encode
        else None,
        "encode_success": True if encode else None,
        "encoded_size_bytes": encoded_bytes if encode else None,
        "artifact_key": artifact if encode else None,
    }


def _build_trace(
    spec: NumericalStudySpec,
    profile: Mapping[str, Any],
    subjects: Sequence[str],
    quality_records: Sequence[Mapping[str, Any]],
    *,
    role: str,
    trace_seed: int,
) -> dict[str, Any]:
    pipeline_map = {row["pipeline_id"]: row for row in profile["pipelines"]}
    local_model = profile["local_models"][0]
    edge_model = profile["edge_models"][0]
    region_ids = tuple(str(value) for value in profile["quality_bins"])
    region_scores = {
        str(row["region_id"]): float(row["mean_quality_score"])
        for row in profile["metadata"]["quality_regions"]
    }
    anon_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for pipeline_index, pipeline_id in enumerate(numerical_pipeline_ids()):
        characteristics = _pipeline_characteristics(pipeline_id)
        pipeline = pipeline_map[pipeline_id]
        for quality_index, quality_bin in enumerate(region_ids):
            quality_score = region_scores[quality_bin]
            region_difficulty = 1.0 - quality_score
            for device_index, device in enumerate(DEVICES):
                for context_index, context in enumerate(_contexts()):
                    for sample_index in range(2):
                        subject = subjects[
                            (
                                pipeline_index * 7
                                + quality_index * 3
                                + device_index
                                + sample_index
                            )
                            % len(subjects)
                        ]
                        row_rng = _rng(
                            trace_seed,
                            "anon-row",
                            pipeline_id,
                            quality_bin,
                            device,
                            context_index,
                            sample_index,
                        )
                        difficulty = (
                            1.0
                            + 0.22 * region_difficulty
                            + (0.28 if device_index else 0.0)
                        )
                        reference_work = float(characteristics["anon_work_s"])
                        raw_work_factor = difficulty * (0.95 + 0.12 * row_rng.random())
                        work = reference_work * (
                            1.0
                            + spec.anon_time_variability_scale * (raw_work_factor - 1.0)
                        )
                        power = 12.5 if device_index == 0 else 8.5
                        memory = int(
                            (
                                80
                                if characteristics["method"] in {"pixelate", "blur"}
                                else 180
                            )
                            * 1024
                            * 1024
                        )
                        reference_bytes = float(characteristics["encoded_size_bytes"])
                        raw_output_bytes = (
                            reference_bytes * (1.0 + 0.16 * region_difficulty)
                            + 4_000 * sample_index
                        )
                        output_bytes = int(
                            round(
                                reference_bytes
                                + spec.output_size_variability_scale
                                * (raw_output_bytes - reference_bytes)
                            )
                        )
                        artifact = f"numerical-{role}-artifact-{_sha(f'{trace_seed}|{pipeline_id}|{quality_bin}|{device}|{context_index}|{sample_index}')[:20]}"
                        attempts: list[dict[str, Any]] = []
                        if sample_index == 1:
                            if (
                                characteristics["method"] in {"generative", "diffusion"}
                                and region_difficulty >= 0.40
                            ):
                                attempts.append(
                                    _attempt(
                                        1,
                                        work * 0.85,
                                        work * power * 0.80,
                                        420 * 1024 * 1024,
                                        oom=True,
                                    )
                                )
                            else:
                                attempts.append(
                                    _attempt(
                                        1,
                                        work * 0.72,
                                        work * power * 0.70,
                                        memory,
                                        guard=False,
                                    )
                                )
                        attempts.append(
                            _attempt(
                                len(attempts) + 1,
                                work,
                                work * power,
                                memory,
                                guard=True,
                                encoded_bytes=output_bytes,
                                artifact=artifact,
                            )
                        )
                        label = EXPRESSIONS[
                            (pipeline_index + sample_index + quality_index)
                            % len(EXPRESSIONS)
                        ]
                        probabilities = _fer_probabilities(
                            trace_seed,
                            subject,
                            label,
                            quality_bin,
                            quality_score,
                            EDGE_MODEL,
                            pipeline_id,
                        )
                        fer_loss = _round(1.0 - probabilities[EXPRESSIONS.index(label)])
                        anon_rows.append(
                            {
                                "row_id": f"anon-{len(anon_rows):06d}",
                                "subject_cluster_id": subject,
                                "pipeline_id": pipeline_id,
                                "pipeline_hash": pipeline["pipeline_hash"],
                                "guard_hash": pipeline["guard_hash"],
                                "encoder_hash": pipeline["encoder_hash"],
                                "quality_bin": quality_bin,
                                "device_type": device,
                                "context": context,
                                "attempts": attempts,
                                "formed_packet": True,
                                "final_encoded_size_bytes": output_bytes,
                                "artifact_key": artifact,
                                "fer_measurements": [
                                    {
                                        "model_id": EDGE_MODEL,
                                        "model_hash": edge_model["model_hash"],
                                        "valid": True,
                                        "fer_loss": fer_loss,
                                        "true_label": label,
                                        "class_probabilities": {
                                            expression: probability
                                            for expression, probability in zip(
                                                EXPRESSIONS,
                                                probabilities,
                                                strict=True,
                                            )
                                        },
                                    }
                                ],
                            }
                        )
                        for rsu_index, rsu_id in enumerate(RSUS):
                            for edge_context in _contexts():
                                gpu_work = (0.017 + 0.006 * region_difficulty) * (
                                    1.10 if rsu_index else 1.0
                                )
                                edge_rows.append(
                                    {
                                        "row_id": f"edge-{len(edge_rows):07d}",
                                        "artifact_key": artifact,
                                        "pipeline_id": pipeline_id,
                                        "quality_bin": quality_bin,
                                        "rsu_id": rsu_id,
                                        "model_id": EDGE_MODEL,
                                        "model_hash": edge_model["model_hash"],
                                        "context": edge_context,
                                        "ingress_work_s": 0.003,
                                        "ingress_energy_j": 0.052,
                                        "gpu_work_s": _round(gpu_work),
                                        "gpu_energy_j": _round(gpu_work * 40.0),
                                        "vram_bytes": 640 * 1024 * 1024,
                                        "result_size_bits": 2048,
                                        "ingress_failed": False,
                                        "failed": False,
                                        "fer_loss": fer_loss,
                                        "true_label": label,
                                        "class_probabilities": {
                                            expression: probability
                                            for expression, probability in zip(
                                                EXPRESSIONS,
                                                probabilities,
                                                strict=True,
                                            )
                                        },
                                    }
                                )
    local_rows: list[dict[str, Any]] = []
    for quality_index, quality_bin in enumerate(region_ids):
        quality_score = region_scores[quality_bin]
        for device_index, device in enumerate(DEVICES):
            for context in _contexts():
                for sample_index in range(2):
                    subject = subjects[
                        (quality_index * 5 + device_index + sample_index)
                        % len(subjects)
                    ]
                    label = EXPRESSIONS[
                        (quality_index + device_index + sample_index) % len(EXPRESSIONS)
                    ]
                    probabilities = _fer_probabilities(
                        trace_seed,
                        subject,
                        label,
                        quality_bin,
                        quality_score,
                        LOCAL_MODEL,
                        None,
                    )
                    work = (
                        (0.034 if device_index == 0 else 0.062)
                        * (1.0 + 0.10 * sample_index)
                        * spec.local_service_scale
                    )
                    local_rows.append(
                        {
                            "row_id": f"local-{len(local_rows):05d}",
                            "subject_cluster_id": subject,
                            "model_id": LOCAL_MODEL,
                            "model_hash": local_model["model_hash"],
                            "quality_bin": quality_bin,
                            "device_type": device,
                            "context": context,
                            "service_work_s": _round(work),
                            "dynamic_energy_j": _round(
                                work * (11.8 if device_index == 0 else 7.6)
                            ),
                            "memory_bytes": 220 * 1024 * 1024,
                            "failed": False,
                            "fer_loss": _round(
                                1.0 - probabilities[EXPRESSIONS.index(label)]
                            ),
                            "true_label": label,
                            "class_probabilities": {
                                expression: probability
                                for expression, probability in zip(
                                    EXPRESSIONS, probabilities, strict=True
                                )
                            },
                        }
                    )
    arrivals: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []
    failed_task_indices = _preprocessing_failure_indices(spec, trace_seed=trace_seed)
    for index in range(spec.task_count):
        subject = subjects[index % len(subjects)]
        record = quality_records[index % len(quality_records)]
        vehicle = VEHICLES[index % len(VEHICLES)]
        arrival_time = _arrival_time_s(spec, trace_seed=trace_seed, task_index=index)
        raw_candidates = tuple(record["candidate_regions"])
        # An empty conformal set is OOD.  Runtime performance prediction still
        # uses the conservative full frozen partition so local fallback remains
        # physically modelled; the OOD flag independently removes every
        # anonymization/edge action.
        candidates = raw_candidates or region_ids
        raw_candidate_probabilities = {
            region_id: float(record["region_probabilities"][region_id])
            for region_id in candidates
        }
        probability_total = sum(raw_candidate_probabilities.values())
        candidate_probabilities: dict[str, float] = {}
        for region_id in candidates[:-1]:
            candidate_probabilities[region_id] = _round(
                raw_candidate_probabilities[region_id] / probability_total
            )
        candidate_probabilities[candidates[-1]] = _round(
            1.0 - sum(candidate_probabilities.values())
        )
        fixture = f"numerical-{role}-fixture-{index:05d}-{subject}"
        arrivals.append(
            {
                "task_id": f"{role}-task-{index:05d}",
                "fixture_key": fixture,
                "vehicle_id": vehicle,
                "arrival_time_s": _round(arrival_time),
                "relative_deadline_s": _round(
                    0.75 + 0.65 * _rng(trace_seed, "deadline", index).random()
                ),
                "quality_candidates": list(candidates),
                "quality_probabilities": candidate_probabilities,
                "true_quality_region": str(record["true_region_id"]),
                "ood": bool(record["ood"]),
                "quality_features": {
                    "quality_score": float(record["predicted_quality_score"]),
                    "raw_conformal_candidate_count": float(len(raw_candidates)),
                    **{
                        name: float(value)
                        for name, value in record["quality_features"].items()
                    },
                    "support_p_value": float(record["support_p_value"]),
                },
            }
        )
        device = DEVICE_BY_VEHICLE[vehicle]
        prep_regions = tuple(
            dict.fromkeys((*candidates, str(record["true_region_id"])))
        )
        for quality_bin in prep_regions:
            region_difficulty = 1.0 - region_scores[quality_bin]
            for context in _contexts():
                work = (0.022 if device == DEVICES[0] else 0.039) * (
                    1.0 + 0.18 * region_difficulty
                )
                prep_rows.append(
                    {
                        "row_id": f"prep-{len(prep_rows):06d}",
                        "fixture_key": fixture,
                        "quality_bin": quality_bin,
                        "device_type": device,
                        "context": context,
                        "service_work_s": _round(work),
                        "dynamic_energy_j": _round(
                            work * (11.2 if device == DEVICES[0] else 7.4)
                        ),
                        "memory_bytes": 48 * 1024 * 1024,
                        "failed": index in failed_task_indices,
                    }
                )
    segment_count = 10
    boundaries = [
        spec.horizon_s * index / segment_count for index in range(segment_count + 1)
    ]
    wireless: list[dict[str, Any]] = []
    for vehicle_index, vehicle in enumerate(VEHICLES):
        for rsu_index, rsu in enumerate(RSUS):
            for direction in ("UL", "DL"):
                for segment in range(segment_count):
                    rng = _rng(trace_seed, "wireless", vehicle, rsu, direction, segment)
                    state = "connected"
                    if (
                        segment in {3, 7}
                        and (vehicle_index + rsu_index + segment) % 2 == 0
                    ):
                        state = "temporary_outage"
                    if segment == 8 and vehicle == "veh-2" and rsu == "rsu-1":
                        state = "permanent_loss"
                    base = 11_000_000.0 if direction == "UL" else 25_000_000.0
                    base *= 0.78 if rsu_index else 1.0
                    base *= 0.74 if vehicle_index else 1.0
                    rate = (
                        base * rng.uniform(0.55, 1.35) if state == "connected" else 0.0
                    )
                    wireless.append(
                        {
                            "segment_id": f"radio-{vehicle}-{rsu}-{direction}-{segment:03d}",
                            "vehicle_id": vehicle,
                            "rsu_id": rsu,
                            "direction": direction,
                            "start_time_s": _round(boundaries[segment]),
                            "end_time_s": _round(boundaries[segment + 1]),
                            "goodput_bps": _round(rate),
                            "transmitter_power_w": 3.3 if direction == "UL" else 4.8,
                            "receiver_power_w": 1.35 if direction == "UL" else 1.2,
                            "link_state": state,
                        }
                    )
    thermal: list[dict[str, Any]] = []
    thermal_template = (
        (0.0, 0.36 * spec.horizon_s, "nominal", 1.0, 1.0),
        (0.36 * spec.horizon_s, 0.58 * spec.horizon_s, "hot_throttled", 0.70, 0.90),
        (0.58 * spec.horizon_s, spec.horizon_s, "recovered", 1.0, 1.0),
    )
    for owner_type, owner_ids in (("vehicle", VEHICLES), ("rsu", RSUS)):
        for owner_id in owner_ids:
            for index, (start, end, state, rate, power) in enumerate(thermal_template):
                thermal.append(
                    {
                        "segment_id": f"thermal-{owner_type}-{owner_id}-{index}",
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                        "resource": "all",
                        "start_time_s": _round(start),
                        "end_time_s": _round(end),
                        "state": state,
                        "service_rate_multiplier": rate,
                        "dynamic_power_multiplier": power,
                    }
                )
    fault_start = _round(0.45 * spec.horizon_s)
    fault_end = _round(0.48 * spec.horizon_s)
    model_maintenance_time = _round(0.72 * spec.horizon_s)
    events = [
        {
            "event_id": f"{role}-rsu2-fault-start",
            "time_s": fault_start,
            "event_type": "DEVICE_FAULT_START",
            "target_type": "rsu",
            "target_id": "rsu-2",
            "resource": "all",
            "old_version": None,
            "new_version": None,
            "permanent": False,
            "details": {"source": "numerical_fault_process"},
        },
        {
            "event_id": f"{role}-rsu2-fault-end",
            "time_s": fault_end,
            "event_type": "DEVICE_FAULT_END",
            "target_type": "rsu",
            "target_id": "rsu-2",
            "resource": "all",
            "old_version": None,
            "new_version": None,
            "permanent": False,
            "details": {"source": "numerical_fault_process"},
        },
        {
            "event_id": f"{role}-rsu1-model-version",
            "time_s": model_maintenance_time,
            "event_type": "MODEL_VERSION",
            "target_type": "rsu",
            "target_id": "rsu-1",
            "resource": "model_cache",
            "old_version": edge_model["model_hash"],
            "new_version": _component_hash(
                f"model:{EDGE_MODEL}:post-maintenance-unprofiled"
            ),
            "permanent": False,
            "maintenance_work_s": 0.22,
            "maintenance_energy_j": 18.0,
            "details": {
                "source": "numerical_engineering_assumption",
                "model_id": EDGE_MODEL,
                "maintenance_parameter_source": "engineering_assumption",
            },
        },
    ]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "trace_version": f"numerical-{role}-2.3.0",
        "trace_hash": "",
        "profile_hash": profile["profile_hash"],
        "data_kind": "numerical_simulation",
        "evidence_status": "frozen_numerical_model",
        "seed": trace_seed,
        "horizon": {"start_time_s": 0.0, "end_time_s": spec.horizon_s},
        "task_arrivals": arrivals,
        "prep": prep_rows,
        "anon_transactions": anon_rows,
        "local_fer": local_rows,
        "edge_fer": edge_rows,
        "environment": {
            "wireless_segments": wireless,
            "thermal_segments": thermal,
            "events": events,
        },
        "parameter_sources": {
            "prep_compute": _source(
                "Numerical finite vehicle preprocessing service and energy model.",
                "resource_busy_s,J,bytes",
            ),
            "anonymization_transactions": _source(
                "Correlated numerical retries, guard, encoding, size, energy and artifact FER outcomes.",
                "resource_busy_s,J,bytes",
            ),
            "fer_compute": _source(
                "Numerical local and edge FER service, energy and paired bounded loss.",
                "resource_busy_s,J,bytes,probability",
            ),
            "wireless": _source(
                "Seeded piecewise application-goodput and paired radio-power process.",
                "bit/s,W",
            ),
            "thermal": _source(
                "Seeded bounded thermal throttling interval.",
                "dimensionless",
                stress=True,
            ),
            "faults": _source(
                "Seeded transient RSU fault process.", "event", stress=True
            ),
            "arrivals_deadlines": _source(
                "Seeded finite arrival and deadline process; paper-v1 may use a bounded centered burst.",
                "s",
                stress=spec.arrival_window_s is not None,
            ),
            "preprocessing_failure_schedule": _source(
                "Seeded task-level preprocessing-failure schedule shared across policies.",
                "task",
                stress=spec.preprocessing_failure_mode != "none",
            ),
            "local_service_scale": _source(
                "Preregistered multiplier for local FER service work and dynamic energy.",
                "dimensionless",
                stress=spec.local_service_scale != 1.0,
            ),
        },
        "metadata": {
            "generator": "privacy_edge_sim.numerical._build_trace",
            "evidence_hash": profile["metadata"]["evidence_hash"],
            "split_manifest_hash": profile["metadata"]["split_manifest_hash"],
            "data_split": {
                "role": "evaluation" if role == "test" else "training_validation",
                "seed": trace_seed,
                "subject_population_hash": _sha("\n".join(subjects)),
                "subject_population_disjoint": True,
                "artifact_namespace_disjoint": True,
            },
            "scenario_controls": {
                "arrival_schedule": {
                    "mode": (
                        "legacy_horizon_spaced"
                        if spec.arrival_window_s is None
                        else "centered_window"
                    ),
                    "center_s": spec.arrival_center_s,
                    "window_s": spec.arrival_window_s,
                    "jitter_fraction": spec.arrival_jitter_fraction,
                },
                "preprocessing_failure_schedule": {
                    "mode": spec.preprocessing_failure_mode,
                    "fixed_count": spec.preprocessing_failure_count,
                    "bernoulli_probability": spec.preprocessing_failure_probability,
                    "failed_task_indices": sorted(failed_task_indices),
                },
                "local_service_scale": spec.local_service_scale,
            },
            "formal_experiment_eligible": False,
            "numerical_experiment_eligible": True,
            "contains_real_measurements": False,
            "joint_rows_must_not_be_split": True,
            "warning": "FROZEN NUMERICAL MODEL - NOT HARDWARE OR ATTACK MEASUREMENT",
        },
    }
    document["trace_hash"] = canonical_document_sha256(document, "trace_hash")
    return document


def _build_config(
    spec: NumericalStudySpec, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "profile_path": "../profiles/numerical_profile.json",
        "trace_path": "../traces/numerical_evaluation_trace.json",
        "scenario_trace_path": "../traces/numerical_scenario_trace.json",
        "evidence_path": "../evidence/numerical_study_evidence.json",
        "max_snapshot_age_s": 0.5,
        "rsu_snapshot_period_s": 0.25,
        "rsu_telemetry_delay_s": 0.04,
        "rsu_telemetry_quantum_work_s": 0.001,
        "rsu_telemetry_drop_every": 7,
        "uplink_pause_limit_s": 2.0,
        "downlink_pause_limit_s": 2.0,
        "metadata_bits": 2048,
        "vehicles": [
            {
                "vehicle_id": "veh-1",
                "device_type": DEVICES[0],
                "battery_capacity_j": 5000.0,
                "initial_battery_j": 5000.0,
                "memory_capacity_bytes": 1073741824,
                "accelerator_descriptors": 8,
                "cpu_descriptors": 8,
                "encoder_descriptors": 8,
                "idle_power_w": 8.0,
                "hold_power_w": 0.15,
                "average_power_budget_w": 24.0,
            },
            {
                "vehicle_id": "veh-2",
                "device_type": DEVICES[1],
                "battery_capacity_j": 4200.0,
                "initial_battery_j": 4200.0,
                "memory_capacity_bytes": 805306368,
                "accelerator_descriptors": 6,
                "cpu_descriptors": 6,
                "encoder_descriptors": 6,
                "idle_power_w": 6.5,
                "hold_power_w": 0.12,
                "average_power_budget_w": 20.0,
            },
        ],
        "rsus": [
            {
                "rsu_id": "rsu-1",
                "descriptor_capacity": 16,
                "vram_capacity_bytes": 8589934592,
                "workload_capacity_gpu_s": 30.0,
                "gpu_servers": 2,
                "idle_power_w": 72.0,
                "hold_power_w": 0.5,
                "cached_models": [EDGE_MODEL],
                "average_power_budget_w": 128.0,
            },
            {
                "rsu_id": "rsu-2",
                "descriptor_capacity": 10,
                "vram_capacity_bytes": 6442450944,
                "workload_capacity_gpu_s": 18.0,
                "gpu_servers": 1,
                "idle_power_w": 58.0,
                "hold_power_w": 0.4,
                "cached_models": [EDGE_MODEL],
                "average_power_budget_w": 108.0,
            },
        ],
        "controller": {
            "policy": "esl_smpc",
            "horizon_events": 2,
            "scenarios": 8,
            "lyapunov_v": 12.0,
            "controller_overhead_s": 0.0008,
            "controller_energy_j": 0.004,
            "rollout_policy": "safe_greedy",
            "physical_queue_weight": 1.0,
            "vehicle_resource_theta": {
                "accelerator": 1.0,
                "cpu": 1.0,
                "encoder": 1.0,
            },
            "rsu_resource_theta": {"ingress": 1.0, "gpu": 1.0},
        },
        "privacy": {
            "risk_threshold": spec.privacy_threshold,
            "confidence_error": spec.confidence_error,
            "quality_miscoverage": spec.quality_miscoverage,
            "min_subjects": spec.min_profile_subjects,
            "min_emission_lcb": 0.50,
        },
        "cost": {
            **evidence["cost_normalization"]["scales"],
            "failure_loss": 1.0,
            "weights": {
                "latency": 1.0,
                "vehicle_energy": 0.25,
                "rsu_energy": 0.15,
                "utility": 1.0,
                "failure": 2.0,
            },
        },
        "long_term": {
            "timeout_rate_limit": 0.15,
            "failure_rate_limit": 0.25,
            "coverage_rate_minimum": 0.75,
        },
        "seeds": {
            name: spec.seed + offset
            for offset, name in enumerate(
                (
                    "environment",
                    "arrivals",
                    "mobility",
                    "wireless",
                    "vehicle",
                    "rsu",
                    "fault",
                    "scenario",
                ),
                1,
            )
        },
        "parameter_sources": {
            "vehicle_compute_latency_energy": "engineering_assumption",
            "anonymous_transaction": "engineering_assumption",
            "encoded_output_size": "engineering_assumption",
            "vehicle_power": "engineering_assumption",
            "rsu_power": "engineering_assumption",
            "task_buffer_hold_power": "engineering_assumption",
            "wireless_goodput": "engineering_assumption",
            "wireless_link_packet_sharing": "engineering_assumption",
            "interruptions_and_recovery": "stress_test_boundary",
            "resource_capacity": "engineering_assumption",
            "arrival_load": "engineering_assumption",
            "deadlines": "engineering_assumption",
            "thermal_throttling": "stress_test_boundary",
            "faults": "stress_test_boundary",
            "privacy_risk_values": "engineering_assumption",
        },
        "output_parquet": False,
    }


def _write_json(path: Path, document: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite numerical study file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_numerical_study(
    output_root: str | Path,
    *,
    spec: NumericalStudySpec | None = None,
    overwrite: bool = False,
) -> NumericalStudyPaths:
    """Generate and immediately validate a complete frozen numerical study."""

    selected = spec or NumericalStudySpec()
    selected.validate()
    root = Path(output_root).resolve()
    evidence, subjects = _build_evidence(selected)
    profile = _build_profile(selected, evidence)
    quality_records = evidence["quality_conformal"]["test_records"]
    evaluation = _build_trace(
        selected,
        profile,
        subjects["test"],
        quality_records,
        role="test",
        trace_seed=selected.seed + 30_001,
    )
    scenario_quality = _quality_records_for_subjects(
        selected.seed,
        subjects["scenario_training_validation"],
        evidence["quality_conformal"],
    )
    scenario = _build_trace(
        selected,
        profile,
        subjects["scenario_training_validation"],
        scenario_quality,
        role="scenario",
        trace_seed=selected.seed + 40_001,
    )
    config = _build_config(selected, evidence)
    profile_path = root / "profiles" / "numerical_profile.json"
    evaluation_path = root / "traces" / "numerical_evaluation_trace.json"
    scenario_path = root / "traces" / "numerical_scenario_trace.json"
    config_path = root / "configs" / "numerical_default.json"
    evidence_path = root / "evidence" / "numerical_study_evidence.json"
    for path, document in (
        (profile_path, profile),
        (evaluation_path, evaluation),
        (scenario_path, scenario),
        (config_path, config),
        (evidence_path, evidence),
    ):
        _write_json(path, document, overwrite=overwrite)
    frozen_profile = load_profile(profile_path)
    frozen_evaluation = load_trace(evaluation_path, frozen_profile)
    frozen_scenario = load_trace(scenario_path, frozen_profile)
    load_config(config_path)
    return NumericalStudyPaths(
        profile_path=profile_path,
        evaluation_trace_path=evaluation_path,
        scenario_trace_path=scenario_path,
        config_path=config_path,
        evidence_path=evidence_path,
        profile_hash=frozen_profile.profile_hash,
        evaluation_trace_hash=frozen_evaluation.trace_hash,
        scenario_trace_hash=frozen_scenario.trace_hash,
        evidence_hash=str(evidence["evidence_hash"]),
    )


def generate_numerical_replication(
    base_study_root: str | Path,
    output_root: str | Path,
    environment_seed: int,
    *,
    overwrite: bool = False,
) -> NumericalStudyPaths:
    """Create an independent evaluation environment from one frozen study.

    The profile, offline evidence and controller scenario trace are copied
    without semantic changes.  Only the evaluation trace and its registered
    environment/task sampling streams are regenerated.  All policies run on a
    given replication therefore share the exact same exogenous realization.
    """

    if (
        isinstance(environment_seed, bool)
        or not isinstance(environment_seed, int)
        or environment_seed < 0
    ):
        raise ValueError("environment_seed must be a non-negative integer")
    base = Path(base_study_root).resolve()
    root = Path(output_root).resolve()
    source_profile_path = base / "profiles" / "numerical_profile.json"
    source_scenario_path = base / "traces" / "numerical_scenario_trace.json"
    source_config_path = base / "configs" / "numerical_default.json"
    source_evidence_path = base / "evidence" / "numerical_study_evidence.json"
    profile_document = json.loads(source_profile_path.read_text(encoding="utf-8"))
    scenario_document = json.loads(source_scenario_path.read_text(encoding="utf-8"))
    config_document = json.loads(source_config_path.read_text(encoding="utf-8"))
    evidence = json.loads(source_evidence_path.read_text(encoding="utf-8"))
    if evidence.get("evidence_hash") != canonical_document_sha256(
        evidence, "evidence_hash"
    ):
        raise ValueError("base numerical evidence hash is invalid")
    spec = NumericalStudySpec(**evidence["spec"])
    spec.validate()
    test_subjects = tuple(evidence["split_manifest"]["splits"]["test"]["subject_ids"])
    evaluation = _build_trace(
        spec,
        profile_document,
        test_subjects,
        evidence["quality_conformal"]["test_records"],
        role="test",
        trace_seed=environment_seed,
    )
    scenario_seed = int(config_document["seeds"]["scenario"])
    for offset, stream in enumerate(
        ("environment", "arrivals", "mobility", "wireless", "vehicle", "rsu", "fault"),
        1,
    ):
        config_document["seeds"][stream] = int.from_bytes(
            hashlib.sha256(
                f"numerical-replication|{environment_seed}|{stream}|{offset}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
        )
    config_document["seeds"]["scenario"] = scenario_seed
    profile_path = root / "profiles" / "numerical_profile.json"
    evaluation_path = root / "traces" / "numerical_evaluation_trace.json"
    scenario_path = root / "traces" / "numerical_scenario_trace.json"
    config_path = root / "configs" / "numerical_default.json"
    evidence_path = root / "evidence" / "numerical_study_evidence.json"
    for path, document in (
        (profile_path, profile_document),
        (evaluation_path, evaluation),
        (scenario_path, scenario_document),
        (config_path, config_document),
        (evidence_path, evidence),
    ):
        _write_json(path, document, overwrite=overwrite)
    frozen_profile = load_profile(profile_path)
    frozen_evaluation = load_trace(evaluation_path, frozen_profile)
    frozen_scenario = load_trace(scenario_path, frozen_profile)
    load_config(config_path)
    return NumericalStudyPaths(
        profile_path=profile_path,
        evaluation_trace_path=evaluation_path,
        scenario_trace_path=scenario_path,
        config_path=config_path,
        evidence_path=evidence_path,
        profile_hash=frozen_profile.profile_hash,
        evaluation_trace_hash=frozen_evaluation.trace_hash,
        scenario_trace_hash=frozen_scenario.trace_hash,
        evidence_hash=str(evidence["evidence_hash"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a frozen numerical anonymous-image edge study"
    )
    parser.add_argument("--output-root", default="numerical-study")
    parser.add_argument("--seed", type=int, default=NumericalStudySpec.seed)
    parser.add_argument("--tasks", type=int, default=NumericalStudySpec.task_count)
    parser.add_argument(
        "--anon-time-variability-scale",
        type=float,
        default=NumericalStudySpec.anon_time_variability_scale,
    )
    parser.add_argument(
        "--output-size-variability-scale",
        type=float,
        default=NumericalStudySpec.output_size_variability_scale,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    spec = NumericalStudySpec(
        seed=args.seed,
        task_count=args.tasks,
        anon_time_variability_scale=args.anon_time_variability_scale,
        output_size_variability_scale=args.output_size_variability_scale,
    )
    paths = generate_numerical_study(
        args.output_root, spec=spec, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {key: str(value) for key, value in asdict(paths).items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTACKERS",
    "NumericalStudyPaths",
    "NumericalStudySpec",
    "classification_metrics",
    "generate_numerical_replication",
    "generate_numerical_study",
    "numerical_pipeline_ids",
    "recompute_privacy_evidence",
]
