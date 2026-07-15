from __future__ import annotations

import copy
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from privacy_edge_sim.config import load_config
from privacy_edge_sim.errors import TraceValidationError
from privacy_edge_sim.numerical import (
    ATTACKERS,
    NumericalStudySpec,
    _calibrate_attackers,
    _build_trace,
    _conformal_evidence,
    _make_split_manifest,
    _observable_identity_signal,
    _pipeline_characteristics,
    _privacy_evidence,
    _quality_observation,
    _quality_record,
    _rng,
    classification_metrics,
    generate_numerical_replication,
    generate_numerical_study,
    numerical_pipeline_ids,
    recompute_privacy_evidence,
)
from privacy_edge_sim.profiles import canonical_document_sha256, load_profile
from privacy_edge_sim.profiles import compute_subject_risk_ucb
from privacy_edge_sim.traces import load_trace


@pytest.fixture(scope="module")
def numerical_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("numerical-study")
    spec = NumericalStudySpec(
        seed=811,
        attack_train_subjects=8,
        threshold_calibration_subjects=12,
        quality_calibration_subjects=12,
        profile_evaluation_subjects=16,
        scenario_subjects=8,
        test_subjects=8,
        frames_per_subject=3,
        task_count=6,
        horizon_s=8.0,
        privacy_threshold=0.80,
    )
    paths = generate_numerical_study(root, spec=spec)
    evidence = json.loads(paths.evidence_path.read_text(encoding="utf-8"))
    return spec, paths, evidence


def test_numerical_pipeline_registry_has_four_families_and_three_strengths():
    pipelines = numerical_pipeline_ids()
    assert len(pipelines) == 12
    assert {value.split("_")[0] for value in pipelines} == {
        "pixelate",
        "blur",
        "generative",
        "diffusion",
    }
    assert {value.split("_")[1] for value in pipelines} == {
        "weak",
        "medium",
        "strong",
    }


def test_anonymous_attack_signal_cannot_exist_without_residual_identity():
    for attacker_id in ATTACKERS:
        for risk_type in ("identity", "verification", "link"):
            assert (
                _observable_identity_signal(
                    retention=0.0,
                    quality_score=1.0,
                    persistence=1.0,
                    attacker_id=attacker_id,
                    risk_type=risk_type,
                )
                == 0.0
            )


@pytest.mark.parametrize("method", ("pixelate", "blur", "generative", "diffusion"))
def test_stronger_anonymization_monotonically_reduces_observable_attack_signal(
    method,
):
    signals = []
    for strength in ("weak", "medium", "strong"):
        retention = float(
            _pipeline_characteristics(f"{method}_{strength}_numerical_v2")[
                "identity_retention"
            ]
        )
        signals.append(
            _observable_identity_signal(
                retention=retention,
                quality_score=0.8,
                persistence=0.7,
                attacker_id="temporal_link_numerical_v2",
                risk_type="link",
            )
        )
    assert signals[0] > signals[1] > signals[2] >= 0.0


def test_numerical_six_way_split_is_subject_video_and_frame_disjoint(
    numerical_bundle,
):
    _, _, evidence = numerical_bundle
    manifest = evidence["split_manifest"]
    assert manifest["all_pairwise_subject_intersections_empty"] is True
    assert len(manifest["splits"]) == 6
    assert all(
        value == 0
        for value in manifest["pairwise_subject_intersection_counts"].values()
    )
    subject_sets = {
        role: set(split["subject_ids"]) for role, split in manifest["splits"].items()
    }
    roles = sorted(subject_sets)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            assert subject_sets[left].isdisjoint(subject_sets[right])
    assert (
        len({split["video_ids_sha256"] for split in manifest["splits"].values()}) == 6
    )
    assert (
        len({split["frame_ids_sha256"] for split in manifest["splits"].values()}) == 6
    )


def test_guard_and_three_attackers_are_frozen_and_isolated(numerical_bundle):
    spec, _, evidence = numerical_bundle
    registry = evidence["attacker_registry"]
    assert {row["attacker_id"] for row in registry} == set(ATTACKERS)
    assert len({row["seed"] for row in registry}) == 3
    assert evidence["isolation_checks"] == {
        "guard_parameter_disjoint_from_attackers": True,
        "guard_seed_disjoint_from_attackers": True,
        "attack_training_threshold_profile_subjects_mutually_exclusive": True,
    }
    assert evidence["guard_registry"]["seed"] not in {row["seed"] for row in registry}
    for attacker in registry:
        assert set(attacker["thresholds"]) == {"identity", "verification", "link"}
        for threshold in attacker["thresholds"].values():
            assert threshold["achieved_calibration_false_match_rate"] <= (
                spec.target_false_match_rate + 1e-12
            )


def test_all_privacy_ucbs_recompute_from_subject_evidence(numerical_bundle):
    _, _, evidence = numerical_bundle
    protocol = evidence["privacy_protocol"]
    region_count = len(evidence["quality_conformal"]["region_ids"])
    assert protocol["registered_hypotheses"] == 12 * region_count * 3 * 3
    recalculated = recompute_privacy_evidence(evidence)
    stored = [row["statistics"] for row in evidence["privacy_evidence"]]
    support = {
        row["region_id"]: row
        for row in evidence["quality_conformal"]["profile_evaluation_quality_support"][
            "cells"
        ]
    }
    assert recalculated == stored
    assert len(stored) == protocol["registered_hypotheses"]
    assert all(
        row["statistics"]["subject_count"]
        == support[row["quality_bin"]]["subject_count"]
        for row in evidence["privacy_evidence"]
    )
    assert all(0.0 <= row["conditional_risk_ucb"] <= 1.0 for row in stored)
    expected_stages = {
        "single_attempt",
        "guard_selected",
        "guard_plus_retry_final",
    }
    for cell in evidence["privacy_evidence"]:
        cell_support = support[cell["quality_bin"]]
        assert cell["quality_cell_subject_count"] == cell_support["subject_count"]
        assert cell["quality_cell_frame_count"] == cell_support["frame_count"]
        assert (
            cell["quality_cell_subject_ids_sha256"]
            == cell_support["subject_ids_sha256"]
        )
        assert set(cell["stage_subject_rows"]) == expected_stages
        assert set(cell["stage_statistics"]) == expected_stages
        assert (
            cell["subject_rows"] == cell["stage_subject_rows"]["guard_plus_retry_final"]
        )
        for stage, rows in cell["stage_subject_rows"].items():
            stats = compute_subject_risk_ucb(
                rows,
                registered_hypotheses=protocol["registered_hypotheses"],
                confidence_error=protocol["confidence_error"],
            )
            assert cell["stage_statistics"][stage][
                "conditional_risk_ucb"
            ] == pytest.approx(stats.conditional_risk_ucb, abs=2e-9)
            assert cell["stage_statistics"][stage]["joint_risk_ucb"] == pytest.approx(
                stats.joint_risk_ucb, abs=2e-9
            )
            assert cell["stage_statistics"][stage]["mean_emission"] == pytest.approx(
                stats.mean_emission, abs=2e-9
            )
    report = evidence["post_selection_risk_report"]
    assert len(report) == protocol["registered_hypotheses"]
    assert all(row["final_stage"] == "guard_plus_retry_final" for row in report)


def test_privacy_quality_cells_use_only_observed_g_star_subject_frames(
    numerical_bundle,
):
    spec, _, evidence = numerical_bundle
    quality = evidence["quality_conformal"]
    support = quality["profile_evaluation_quality_support"]
    split_subjects = evidence["split_manifest"]["splits"]["profile_evaluation"][
        "subject_ids"
    ]
    expected: dict[str, dict[str, list[dict[str, float | int]]]] = {
        region_id: {} for region_id in quality["region_ids"]
    }
    for subject_id in split_subjects:
        for frame_index in range(spec.frames_per_subject):
            observation = _quality_observation(spec.seed, subject_id, frame_index)
            record = _quality_record(observation, quality)
            expected[record["true_region_id"]].setdefault(subject_id, []).append(
                {
                    "frame_index": frame_index,
                    "true_quality_score": observation["true_quality_score"],
                }
            )
    actual = {
        cell["region_id"]: {
            subject["subject_id"]: subject["frames"]
            for subject in cell["subject_frames"]
        }
        for cell in support["cells"]
    }
    assert actual == expected
    assert sum(cell["frame_count"] for cell in support["cells"]) == (
        len(split_subjects) * spec.frames_per_subject
    )
    assert support["support_hash"] == canonical_document_sha256(support, "support_hash")


@pytest.mark.parametrize("seed", (714, NumericalStudySpec().seed))
def test_preregistered_full_attack_family_has_safe_strong_and_unsafe_weak_cells(seed):
    spec = NumericalStudySpec(
        seed=seed,
        attack_train_subjects=32,
        threshold_calibration_subjects=64,
        quality_calibration_subjects=64,
        profile_evaluation_subjects=256,
        scenario_subjects=8,
        test_subjects=8,
        frames_per_subject=4,
        task_count=2,
        horizon_s=5.0,
        privacy_threshold=0.35,
    )
    manifest, subjects = _make_split_manifest(spec)
    quality = _conformal_evidence(spec, subjects)
    attackers = _calibrate_attackers(spec, subjects, manifest)
    hypothesis_count = 12 * len(quality["region_ids"]) * 3 * 3
    assert hypothesis_count == 216
    weak = "diffusion_weak_numerical_v2"
    strong = "diffusion_strong_numerical_v2"
    rows = _privacy_evidence(
        spec,
        subjects,
        attackers,
        quality,
        pipeline_ids=(weak, strong),
        registered_hypotheses=hypothesis_count,
    )

    def worst_by_region(pipeline_id):
        return {
            region_id: max(
                row["statistics"]["conditional_risk_ucb"]
                for row in rows
                if row["pipeline_id"] == pipeline_id and row["quality_bin"] == region_id
            )
            for region_id in quality["region_ids"]
        }

    weak_worst = worst_by_region(weak)
    strong_worst = worst_by_region(strong)
    assert all(value > spec.privacy_threshold for value in weak_worst.values())
    assert all(value <= spec.privacy_threshold for value in strong_worst.values())
    assert all(
        cell["subject_count"] >= spec.min_profile_subjects
        for cell in quality["profile_evaluation_quality_support"]["cells"]
    )


def test_fer_evidence_is_paired_and_reports_required_metrics(numerical_bundle):
    spec, _, evidence = numerical_bundle
    region_count = len(evidence["quality_conformal"]["region_ids"])
    metrics = evidence["fer_metrics"]
    assert len(metrics) == region_count * (12 + 1)
    required = {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "nll",
        "ece_10_bin",
        "per_class_recall",
    }
    assert all(required <= set(row) for row in metrics)
    assert all(
        set(row["per_class_recall"]) == {"neutral", "happy", "sad", "angry"}
        for row in metrics
    )
    paired = evidence["fer_paired_records"]
    assert len(paired) == 12 * region_count * spec.test_subjects
    keys = {
        (row["subject_id"], row["pipeline_id"], row["quality_bin"]) for row in paired
    }
    assert len(keys) == len(paired)
    assert all(
        row["paired_nll_delta"]
        == pytest.approx(row["anonymous_edge_nll"] - row["local_nll"], abs=2e-9)
        for row in paired
    )
    assert all(isinstance(row["local_correct"], bool) for row in paired)
    assert all(isinstance(row["anonymous_edge_correct"], bool) for row in paired)


def test_classification_metrics_rejects_non_normalized_probabilities():
    with pytest.raises(ValueError, match="normalized"):
        classification_metrics(
            [{"label": "neutral", "probabilities": [0.9, 0.2, 0.0, 0.0]}]
        )


def test_split_conformal_calibration_and_test_coverage_are_auditable(
    numerical_bundle,
):
    _, _, evidence = numerical_bundle
    conformal = evidence["quality_conformal"]
    records = conformal["test_records"]
    empirical = sum(row["covered"] for row in records) / len(records)
    mean_size = sum(len(row["candidate_regions"]) for row in records) / len(records)
    assert conformal["test_coverage"] == pytest.approx(empirical)
    assert conformal["mean_candidate_set_size"] == pytest.approx(mean_size)
    assert conformal["calibration_role"] == "quality_calibration"
    assert conformal["test_role"] == "test"
    assert conformal["method"] == "split_conformal_region_classifier"
    assert conformal["nonconformity"] == "1-p_true_region"
    assert conformal["calibration_count"] == len(
        conformal["calibration_nonconformity_scores"]
    )
    region_ids = set(conformal["region_ids"])
    assert all(row["true_region_id"] in region_ids for row in records)
    quantile = conformal["classification_quantile"]
    for row in records:
        expected = sorted(
            region_id
            for region_id, probability in row["region_probabilities"].items()
            if 1.0 - probability <= quantile + 1e-12
        )
        assert row["candidate_regions"] == expected
        assert set(row["region_probabilities"]) == region_ids


def test_split_conformal_can_miss_true_leaf_without_deriving_truth_from_candidates(
    numerical_bundle,
):
    spec, _, evidence = numerical_bundle
    conformal = copy.deepcopy(evidence["quality_conformal"])
    observation = _quality_observation(spec.seed, "explicit-miscoverage-subject", 0)
    baseline = _quality_record(observation, conformal)
    true_region = baseline["true_region_id"]
    other_region = next(
        region_id for region_id in conformal["region_ids"] if region_id != true_region
    )
    region_count = len(conformal["region_ids"])

    def force_wrong_region(node):
        if node["kind"] == "leaf":
            residual = 0.03 / (region_count - 1)
            node["region_probabilities"] = {
                region_id: 0.97 if region_id == other_region else residual
                for region_id in conformal["region_ids"]
            }
            return
        force_wrong_region(node["left"])
        force_wrong_region(node["right"])

    force_wrong_region(conformal["region_classifier"])
    conformal["classification_quantile"] = 0.05
    record = _quality_record(observation, conformal)
    assert record["candidate_regions"] == [other_region]
    assert record["true_region_id"] == true_region
    assert record["covered"] is False


def test_profile_and_joint_traces_use_every_leaf_region_without_coarse_pooling(
    numerical_bundle,
):
    _, paths, evidence = numerical_bundle
    profile = load_profile(paths.profile_path)
    trace = load_trace(paths.evaluation_trace_path, profile)
    regions = set(evidence["quality_conformal"]["region_ids"])
    assert set(profile.quality_bins) == regions
    assert {quality for _, quality, _ in profile.privacy_cells} == regions
    assert {row.quality_bin for row in trace.anon_rows} == regions
    assert {row.quality_bin for row in trace.local_rows} == regions
    assert {row.quality_bin for row in trace.edge_rows} == regions
    for arrival in trace.arrivals:
        assert arrival.true_quality_region in regions
        assert set(dict(arrival.quality_probabilities)) == set(
            arrival.quality_candidates
        )
        assert sum(dict(arrival.quality_probabilities).values()) == pytest.approx(1.0)


def test_miscoverage_arrival_keeps_true_region_prep_pairing_without_policy_repair(
    numerical_bundle,
):
    spec, paths, evidence = numerical_bundle
    profile_document = json.loads(paths.profile_path.read_text(encoding="utf-8"))
    record = copy.deepcopy(evidence["quality_conformal"]["test_records"][0])
    true_region = record["true_region_id"]
    other_region = next(
        region_id
        for region_id in evidence["quality_conformal"]["region_ids"]
        if region_id != true_region
    )
    record["candidate_regions"] = [other_region]
    document = _build_trace(
        spec,
        profile_document,
        evidence["split_manifest"]["splits"]["test"]["subject_ids"],
        [record],
        role="miscoverage",
        trace_seed=spec.seed + 91_000,
    )
    arrival = document["task_arrivals"][0]
    assert arrival["true_quality_region"] == true_region
    assert arrival["quality_candidates"] == [other_region]
    paired_prep_regions = {
        row["quality_bin"]
        for row in document["prep"]
        if row["fixture_key"] == arrival["fixture_key"]
    }
    assert paired_prep_regions == {true_region, other_region}


def test_numerical_trace_arrivals_satisfy_json_schema(numerical_bundle):
    _, paths, _ = numerical_bundle
    repository_root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (repository_root / "schemas" / "trace.schema.json").read_text(encoding="utf-8")
    )
    document = json.loads(paths.evaluation_trace_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


def test_numerical_profile_satisfies_json_schema(numerical_bundle):
    _, paths, _ = numerical_bundle
    repository_root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (repository_root / "schemas" / "profile.schema.json").read_text(
            encoding="utf-8"
        )
    )
    document = json.loads(paths.profile_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


@pytest.mark.parametrize(
    ("section", "location", "field", "bound_section", "bound_field", "integer"),
    (
        ("prep", "row", "memory_bytes", "preprocessing", "max_memory_bytes", True),
        (
            "prep",
            "row",
            "service_work_s",
            "preprocessing",
            "max_service_work_s",
            False,
        ),
        (
            "prep",
            "row",
            "dynamic_energy_j",
            "preprocessing",
            "max_dynamic_energy_j",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "peak_memory_bytes",
            "pipeline",
            "max_peak_memory_bytes",
            True,
        ),
        (
            "anon_transactions",
            "attempt",
            "anon_work_s",
            "pipeline",
            "max_anon_work_s",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "anon_energy_j",
            "pipeline",
            "max_anon_energy_j",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "guard_work_s",
            "pipeline",
            "max_guard_work_s",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "guard_energy_j",
            "pipeline",
            "max_guard_energy_j",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "encode_work_s",
            "pipeline",
            "max_encode_work_s",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "encode_energy_j",
            "pipeline",
            "max_encode_energy_j",
            False,
        ),
        (
            "anon_transactions",
            "attempt",
            "encoded_size_bytes",
            "pipeline",
            "max_output_bytes",
            True,
        ),
        (
            "anon_transactions",
            "row",
            "final_encoded_size_bytes",
            "pipeline",
            "max_output_bytes",
            True,
        ),
        (
            "local_fer",
            "row",
            "memory_bytes",
            "local",
            "max_memory_bytes",
            True,
        ),
        (
            "local_fer",
            "row",
            "service_work_s",
            "local",
            "max_service_work_s",
            False,
        ),
        (
            "local_fer",
            "row",
            "dynamic_energy_j",
            "local",
            "max_dynamic_energy_j",
            False,
        ),
        (
            "edge_fer",
            "row",
            "vram_bytes",
            "edge",
            "max_vram_bytes",
            True,
        ),
        (
            "edge_fer",
            "row",
            "ingress_work_s",
            "edge",
            "max_ingress_work_s",
            False,
        ),
        (
            "edge_fer",
            "row",
            "ingress_energy_j",
            "edge",
            "max_ingress_energy_j",
            False,
        ),
        (
            "edge_fer",
            "row",
            "gpu_work_s",
            "edge",
            "max_gpu_work_s",
            False,
        ),
        (
            "edge_fer",
            "row",
            "gpu_energy_j",
            "edge",
            "max_gpu_energy_j",
            False,
        ),
        (
            "edge_fer",
            "row",
            "result_size_bits",
            "edge",
            "max_result_size_bits",
            True,
        ),
    ),
)
def test_trace_loader_rejects_rows_above_preregistered_deployment_envelope(
    numerical_bundle,
    tmp_path,
    section,
    location,
    field,
    bound_section,
    bound_field,
    integer,
):
    _, paths, _ = numerical_bundle
    profile = load_profile(paths.profile_path)
    document = json.loads(paths.evaluation_trace_path.read_text(encoding="utf-8"))
    row = document[section][0]
    if location == "attempt":
        if field == "encoded_size_bytes":
            row, target = next(
                (candidate, attempt)
                for candidate in document[section]
                if candidate["formed_packet"]
                for attempt in candidate["attempts"]
                if attempt["encode_success"] is True
            )
        else:
            row, target = next(
                (candidate, attempt)
                for candidate in document[section]
                for attempt in candidate["attempts"]
                if attempt.get(field) is not None
            )
    elif field == "final_encoded_size_bytes":
        row = next(
            candidate for candidate in document[section] if candidate["formed_packet"]
        )
        target = row
    else:
        target = row
    if bound_section == "preprocessing":
        bound = profile.preprocessing_resource_bounds[bound_field]
    elif bound_section == "pipeline":
        bound = profile.pipelines[row["pipeline_id"]].deployment_resource_bounds[
            bound_field
        ]
    elif bound_section == "local":
        bound = profile.local_models[row["model_id"]].deployment_resource_bounds[
            bound_field
        ]
    else:
        bound = profile.edge_models[row["model_id"]].deployment_resource_bounds[
            bound_field
        ]
    violating_value = (
        int(bound) + 1
        if integer
        else float(bound) + max(1e-6, abs(float(bound)) * 1e-6)
    )
    target[field] = violating_value
    if field == "encoded_size_bytes":
        row["final_encoded_size_bytes"] = violating_value
    elif field == "final_encoded_size_bytes":
        successful_attempt = next(
            attempt for attempt in row["attempts"] if attempt["encode_success"] is True
        )
        successful_attempt["encoded_size_bytes"] = violating_value
    document["trace_hash"] = canonical_document_sha256(document, "trace_hash")
    path = tmp_path / f"bad-{section}-{field}.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TraceValidationError) as exc:
        load_trace(path, profile)
    assert exc.value.detail.code == "TRACE_DEPLOYMENT_BOUND_EXCEEDED"
    expected_field = (
        "encoded_size_bytes" if field == "final_encoded_size_bytes" else field
    )
    assert exc.value.detail.context["field"] == expected_field


def test_trace_loader_rejects_candidate_probability_support_mismatch(
    numerical_bundle, tmp_path
):
    _, paths, _ = numerical_bundle
    profile = load_profile(paths.profile_path)
    document = json.loads(paths.evaluation_trace_path.read_text(encoding="utf-8"))
    document["task_arrivals"][0]["quality_probabilities"]["not-a-region"] = 0.0
    document["trace_hash"] = canonical_document_sha256(document, "trace_hash")
    path = tmp_path / "bad-probability-support.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TraceValidationError) as exc:
        load_trace(path, profile)
    assert exc.value.detail.code == "TRACE_QUALITY_PROBABILITY_SUPPORT"


def test_quality_partition_has_multidimensional_frozen_subject_safe_leaves(
    numerical_bundle,
):
    spec, _, evidence = numerical_bundle
    partition = evidence["quality_partition"]
    conformal = evidence["quality_conformal"]
    assert len(partition["feature_names"]) == 6
    assert partition["partition_hash"] == conformal["partition_hash"]
    assert conformal["classifier_hash"] != conformal["partition_hash"]
    assert partition["training_role"] == "scenario_training_validation"

    leaves = []

    def walk(node):
        assert node["depth"] <= spec.quality_tree_depth
        if node["kind"] == "leaf":
            leaves.append(node)
            return
        walk(node["left"])
        walk(node["right"])

    walk(partition["tree"])
    assert leaves
    assert all(
        leaf["subject_count"] >= spec.quality_min_leaf_subjects for leaf in leaves
    )
    assert all(
        sum(leaf["coarse_class_probabilities"].values()) == pytest.approx(1.0)
        for leaf in leaves
    )
    assert {leaf["region_id"] for leaf in leaves} == set(conformal["region_ids"])


def test_quality_ood_is_a_feature_support_p_value_not_task_position(
    numerical_bundle,
):
    _, _, evidence = numerical_bundle
    conformal = evidence["quality_conformal"]
    observation = {
        "subject_id": "explicit-feature-outlier",
        "features": {
            "face_scale": 0.0,
            "abs_yaw_deg": 90.0,
            "abs_pitch_deg": 90.0,
            "blur_index": 1.0,
            "illumination_deviation": 1.0,
            "occlusion_fraction": 1.0,
        },
        "reference_features": {
            "face_scale": 0.0,
            "abs_yaw_deg": 90.0,
            "abs_pitch_deg": 90.0,
            "blur_index": 1.0,
            "illumination_deviation": 1.0,
            "occlusion_fraction": 1.0,
        },
        "true_quality_score": 0.0,
        "coarse_quality_class": "challenging",
    }
    outlier = _quality_record(observation, conformal)
    assert outlier["ood"] is True
    assert outlier["support_p_value"] <= conformal["ood_alpha"]
    for row in conformal["test_records"]:
        assert row["ood"] is (
            row["support_p_value"] <= conformal["ood_alpha"]
            or not row["candidate_regions"]
        )


def test_attack_train_fits_parameters_and_threshold_split_is_isolated():
    common = dict(
        seed=173,
        attack_train_subjects=6,
        threshold_calibration_subjects=8,
        quality_calibration_subjects=8,
        profile_evaluation_subjects=8,
        scenario_subjects=8,
        test_subjects=8,
        frames_per_subject=2,
        task_count=2,
        horizon_s=5.0,
    )
    base = NumericalStudySpec(**common)
    base_manifest, base_subjects = _make_split_manifest(base)
    base_registry = _calibrate_attackers(base, base_subjects, base_manifest)

    more_train = NumericalStudySpec(**{**common, "attack_train_subjects": 7})
    manifest, subjects = _make_split_manifest(more_train)
    trained_registry = _calibrate_attackers(more_train, subjects, manifest)
    assert [row["parameter_fingerprint"] for row in base_registry] != [
        row["parameter_fingerprint"] for row in trained_registry
    ]

    more_threshold = NumericalStudySpec(
        **{**common, "threshold_calibration_subjects": 9}
    )
    manifest, subjects = _make_split_manifest(more_threshold)
    threshold_registry = _calibrate_attackers(more_threshold, subjects, manifest)
    assert [row["parameter_fingerprint"] for row in base_registry] == [
        row["parameter_fingerprint"] for row in threshold_registry
    ]
    assert [row["threshold_calibration_subjects_hash"] for row in base_registry] != [
        row["threshold_calibration_subjects_hash"] for row in threshold_registry
    ]


def test_identity_attack_is_rank_one_only_and_not_thresholded():
    spec = NumericalStudySpec(
        seed=174,
        attack_train_subjects=6,
        threshold_calibration_subjects=8,
        quality_calibration_subjects=8,
        profile_evaluation_subjects=8,
        scenario_subjects=8,
        test_subjects=8,
        frames_per_subject=2,
        task_count=2,
        horizon_s=5.0,
    )
    manifest, subjects = _make_split_manifest(spec)
    quality = _conformal_evidence(spec, subjects)
    attacker = _calibrate_attackers(spec, subjects, manifest)[0]
    assert attacker["thresholds"]["identity"]["decision_rule"] == (
        "rank_1_gallery_retrieval"
    )
    assert attacker["thresholds"]["identity"]["used_for_success"] is False
    changed = copy.deepcopy(attacker)
    changed["thresholds"]["identity"]["threshold"] = 1.0
    original_rows = [
        row["subject_rows"]
        for row in _privacy_evidence(spec, subjects, [attacker], quality)
        if row["risk_type"] == "identity"
    ]
    changed_rows = [
        row["subject_rows"]
        for row in _privacy_evidence(spec, subjects, [changed], quality)
        if row["risk_type"] == "identity"
    ]
    assert original_rows == changed_rows


def test_nmin_and_cost_scales_are_preregistered_and_frozen(numerical_bundle):
    spec, paths, evidence = numerical_bundle
    profile = json.loads(paths.profile_path.read_text(encoding="utf-8"))
    config = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert profile["privacy_policy"]["min_subjects"] == spec.min_profile_subjects
    assert config["privacy"]["min_subjects"] == spec.min_profile_subjects
    assert spec.profile_evaluation_subjects < spec.min_profile_subjects
    calibration = evidence["cost_normalization"]
    assert calibration["role"] == "scenario_training_validation"
    assert calibration["online_mutable"] is False
    assert calibration["method"] == (
        "median_of_positive_preregistered_baseline_records"
    )
    assert {key: config["cost"][key] for key in calibration["scales"]} == calibration[
        "scales"
    ]


def test_generated_profile_traces_and_config_load_in_existing_des(numerical_bundle):
    _, paths, evidence = numerical_bundle
    profile = load_profile(paths.profile_path)
    evaluation = load_trace(paths.evaluation_trace_path, profile)
    scenario = load_trace(paths.scenario_trace_path, profile)
    config = load_config(paths.config_path)
    assert profile.data_kind == "numerical_simulation"
    assert evaluation.data_kind == scenario.data_kind == "numerical_simulation"
    assert profile.evidence_status == "frozen_numerical_model"
    assert (
        evaluation.evidence_status
        == scenario.evidence_status
        == "frozen_numerical_model"
    )
    assert evaluation.trace_hash != scenario.trace_hash
    assert evaluation.metadata["data_split"]["role"] == "evaluation"
    assert scenario.metadata["data_split"]["role"] == "training_validation"
    assert (
        evaluation.metadata["data_split"]["subject_population_hash"]
        != scenario.metadata["data_split"]["subject_population_hash"]
    )
    assert set(profile.pipelines) == set(numerical_pipeline_ids())
    assert all(row.subject_cluster_id is not None for row in evaluation.local_rows)
    assert all(row.subject_cluster_id is not None for row in scenario.local_rows)
    assert config.profile_path == paths.profile_path
    assert evidence["evidence_hash"] == profile.metadata["evidence_hash"]


def test_generator_is_byte_deterministic_and_environment_seed_changes_hashes(
    tmp_path: Path,
):
    common = dict(
        attack_train_subjects=4,
        threshold_calibration_subjects=6,
        quality_calibration_subjects=6,
        profile_evaluation_subjects=8,
        scenario_subjects=4,
        test_subjects=4,
        frames_per_subject=2,
        task_count=3,
        horizon_s=6.0,
        privacy_threshold=0.95,
    )
    first = generate_numerical_study(
        tmp_path / "a", spec=NumericalStudySpec(seed=91, **common)
    )
    second = generate_numerical_study(
        tmp_path / "b", spec=NumericalStudySpec(seed=91, **common)
    )
    changed = generate_numerical_study(
        tmp_path / "c", spec=NumericalStudySpec(seed=92, **common)
    )
    for left, right in (
        (first.profile_path, second.profile_path),
        (first.evaluation_trace_path, second.evaluation_trace_path),
        (first.scenario_trace_path, second.scenario_trace_path),
        (first.evidence_path, second.evidence_path),
        (first.config_path, second.config_path),
    ):
        assert left.read_bytes() == right.read_bytes()
    assert first.evaluation_trace_hash != changed.evaluation_trace_hash
    assert first.scenario_trace_hash != changed.scenario_trace_hash
    assert first.evidence_hash != changed.evidence_hash


def test_preregistered_two_stage_variability_controls_preserve_joint_rows(
    numerical_bundle, tmp_path: Path
):
    spec, reference_paths, _ = numerical_bundle
    zero_spec = replace(
        spec,
        anon_time_variability_scale=0.0,
        output_size_variability_scale=0.0,
    )
    zero_paths = generate_numerical_study(tmp_path / "zero-variability", spec=zero_spec)
    reference = json.loads(
        reference_paths.evaluation_trace_path.read_text(encoding="utf-8")
    )
    zero = json.loads(zero_paths.evaluation_trace_path.read_text(encoding="utf-8"))

    def values(document, pipeline_id):
        rows = [
            row
            for row in document["anon_transactions"]
            if row["pipeline_id"] == pipeline_id
        ]
        works = [row["attempts"][-1]["anon_work_s"] for row in rows]
        sizes = [row["attempts"][-1]["encoded_size_bytes"] for row in rows]
        return works, sizes

    pipeline_id = numerical_pipeline_ids()[0]
    reference_work, reference_size = values(reference, pipeline_id)
    zero_work, zero_size = values(zero, pipeline_id)
    assert statistics.pvariance(reference_work) > 0.0
    assert statistics.pvariance(reference_size) > 0.0
    assert statistics.pvariance(zero_work) == 0.0
    assert statistics.pvariance(zero_size) == 0.0
    assert all(row["attempts"] for row in zero["anon_transactions"])


@pytest.mark.parametrize(
    "field",
    ("anon_time_variability_scale", "output_size_variability_scale"),
)
def test_variability_controls_reject_out_of_range_values(field):
    with pytest.raises(ValueError, match=field):
        replace(NumericalStudySpec(), **{field: 3.01}).validate()


def test_replication_freezes_profile_evidence_and_scenarios_but_changes_environment(
    numerical_bundle, tmp_path: Path
):
    _, base_paths, _ = numerical_bundle
    base_root = base_paths.profile_path.parent.parent
    first = generate_numerical_replication(base_root, tmp_path / "r1", 7001)
    repeat = generate_numerical_replication(base_root, tmp_path / "r1-repeat", 7001)
    changed = generate_numerical_replication(base_root, tmp_path / "r2", 7002)

    assert first.profile_hash == repeat.profile_hash == changed.profile_hash
    assert (
        first.scenario_trace_hash
        == repeat.scenario_trace_hash
        == changed.scenario_trace_hash
    )
    assert first.evidence_hash == repeat.evidence_hash == changed.evidence_hash
    assert first.evaluation_trace_hash == repeat.evaluation_trace_hash
    assert first.evaluation_trace_hash != changed.evaluation_trace_hash
    assert first.profile_path.read_bytes() == changed.profile_path.read_bytes()
    assert (
        first.scenario_trace_path.read_bytes()
        == changed.scenario_trace_path.read_bytes()
    )
    assert first.evidence_path.read_bytes() == changed.evidence_path.read_bytes()

    first_config = json.loads(first.config_path.read_text(encoding="utf-8"))
    changed_config = json.loads(changed.config_path.read_text(encoding="utf-8"))
    assert first_config["seeds"]["scenario"] == changed_config["seeds"]["scenario"]
    assert (
        first_config["seeds"]["environment"] != changed_config["seeds"]["environment"]
    )


def _evaluation_trace_for_spec(
    spec: NumericalStudySpec,
    paths,
    evidence,
) -> dict[str, object]:
    """Build only the evaluation trace so protocol tests stay focused and fast."""

    profile = json.loads(paths.profile_path.read_text(encoding="utf-8"))
    subjects = tuple(evidence["split_manifest"]["splits"]["test"]["subject_ids"])
    return _build_trace(
        spec,
        profile,
        subjects,
        evidence["quality_conformal"]["test_records"],
        role="test",
        trace_seed=spec.seed + 30_001,
    )


def _preprocessing_failure_fixtures(document: dict[str, object]) -> set[str]:
    rows = document["prep"]
    assert isinstance(rows, list)
    return {
        str(row["fixture_key"])
        for row in rows
        if isinstance(row, dict) and row["failed"]
    }


def test_default_scenario_controls_strictly_preserve_legacy_arrivals_and_failure(
    numerical_bundle,
):
    """New controls must not silently rewrite the archived v11 default trace."""

    spec, paths, evidence = numerical_bundle
    assert spec.arrival_center_s is None
    assert spec.arrival_window_s is None
    assert spec.arrival_jitter_fraction is None
    assert spec.preprocessing_failure_mode == "legacy_last"
    assert spec.preprocessing_failure_count == 0
    assert spec.preprocessing_failure_probability == 0.0
    assert spec.local_service_scale == 1.0
    # Pre-paper-v1 evidence serializes no scenario-control keys.  Replication
    # reconstructs its spec through this exact constructor path.
    legacy_evidence_spec = asdict(spec)
    for field in (
        "arrival_center_s",
        "arrival_window_s",
        "arrival_jitter_fraction",
        "preprocessing_failure_mode",
        "preprocessing_failure_count",
        "preprocessing_failure_probability",
        "local_service_scale",
    ):
        del legacy_evidence_spec[field]
    assert NumericalStudySpec(**legacy_evidence_spec) == spec

    document = _evaluation_trace_for_spec(spec, paths, evidence)
    arrivals = document["task_arrivals"]
    assert isinstance(arrivals, list)
    expected_arrivals = [
        round(
            max(
                0.01,
                (index + 1) * (spec.horizon_s - 1.6) / (spec.task_count + 1)
                + _rng(spec.seed + 30_001, "arrival", index).uniform(-0.04, 0.04),
            ),
            9,
        )
        for index in range(spec.task_count)
    ]
    assert [row["arrival_time_s"] for row in arrivals] == expected_arrivals

    expected_last_fixture = arrivals[-1]["fixture_key"]
    assert _preprocessing_failure_fixtures(document) == {expected_last_fixture}
    prep_rows = document["prep"]
    assert isinstance(prep_rows, list)
    assert all(
        row["failed"] is (row["fixture_key"] == expected_last_fixture)
        for row in prep_rows
    )


def test_explicit_arrival_window_is_bounded_deterministic_and_validated(
    numerical_bundle,
):
    base, paths, evidence = numerical_bundle
    explicit = replace(
        base,
        arrival_center_s=3.0,
        arrival_window_s=1.0,
        arrival_jitter_fraction=0.10,
    )
    explicit.validate()
    first = _evaluation_trace_for_spec(explicit, paths, evidence)
    repeat = _evaluation_trace_for_spec(explicit, paths, evidence)
    assert first == repeat

    arrivals = first["task_arrivals"]
    assert isinstance(arrivals, list)
    low = explicit.arrival_center_s - explicit.arrival_window_s / 2.0
    high = explicit.arrival_center_s + explicit.arrival_window_s / 2.0
    assert all(low <= row["arrival_time_s"] <= high for row in arrivals)
    assert [row["arrival_time_s"] for row in arrivals] == sorted(
        row["arrival_time_s"] for row in arrivals
    )

    with pytest.raises(ValueError):
        replace(base, arrival_center_s=3.0).validate()
    with pytest.raises(ValueError):
        replace(
            base,
            arrival_center_s=0.25,
            arrival_window_s=1.0,
            arrival_jitter_fraction=0.0,
        ).validate()


@pytest.mark.parametrize(
    ("mode", "count", "probability", "expected_failure_count"),
    (
        ("none", 0, 0.0, 0),
        ("fixed_count", 2, 0.0, 2),
        ("bernoulli", 0, 1.0, 6),
    ),
)
def test_preprocessing_failure_modes_are_task_scoped_and_deterministic(
    numerical_bundle,
    mode: str,
    count: int,
    probability: float,
    expected_failure_count: int,
):
    base, paths, evidence = numerical_bundle
    spec = replace(
        base,
        preprocessing_failure_mode=mode,
        preprocessing_failure_count=count,
        preprocessing_failure_probability=probability,
    )
    spec.validate()
    first = _evaluation_trace_for_spec(spec, paths, evidence)
    repeat = _evaluation_trace_for_spec(spec, paths, evidence)
    assert first == repeat

    failed_fixtures = _preprocessing_failure_fixtures(first)
    assert len(failed_fixtures) == expected_failure_count
    prep_rows = first["prep"]
    assert isinstance(prep_rows, list)
    for fixture_key in {row["fixture_key"] for row in prep_rows}:
        fixture_rows = [row for row in prep_rows if row["fixture_key"] == fixture_key]
        assert all(row["failed"] for row in fixture_rows) is (
            fixture_key in failed_fixtures
        )


def test_local_service_scale_is_deterministic_and_scales_local_work(
    numerical_bundle,
):
    base, paths, evidence = numerical_bundle
    stressed = replace(base, local_service_scale=1.5)
    stressed.validate()
    reference_trace = _evaluation_trace_for_spec(base, paths, evidence)
    first = _evaluation_trace_for_spec(stressed, paths, evidence)
    repeat = _evaluation_trace_for_spec(stressed, paths, evidence)
    assert first == repeat

    reference_rows = reference_trace["local_fer"]
    stressed_rows = first["local_fer"]
    assert isinstance(reference_rows, list)
    assert isinstance(stressed_rows, list)
    assert len(reference_rows) == len(stressed_rows)
    for reference, stressed_row in zip(reference_rows, stressed_rows, strict=True):
        assert stressed_row["service_work_s"] == pytest.approx(
            reference["service_work_s"] * 1.5,
            abs=2e-9,
        )
        assert stressed_row["dynamic_energy_j"] == pytest.approx(
            reference["dynamic_energy_j"] * 1.5,
            abs=2e-9,
        )


def test_replication_persists_scenario_controls_in_frozen_evidence(tmp_path: Path):
    spec = NumericalStudySpec(
        seed=917,
        attack_train_subjects=4,
        threshold_calibration_subjects=6,
        quality_calibration_subjects=6,
        profile_evaluation_subjects=8,
        scenario_subjects=4,
        test_subjects=4,
        frames_per_subject=2,
        task_count=4,
        horizon_s=6.0,
        privacy_threshold=0.95,
        arrival_center_s=3.0,
        arrival_window_s=1.0,
        arrival_jitter_fraction=0.05,
        preprocessing_failure_mode="fixed_count",
        preprocessing_failure_count=2,
        local_service_scale=1.5,
    )
    base_paths = generate_numerical_study(tmp_path / "base", spec=spec)
    first = generate_numerical_replication(
        base_paths.profile_path.parent.parent,
        tmp_path / "replication-a",
        8001,
    )
    repeat = generate_numerical_replication(
        base_paths.profile_path.parent.parent,
        tmp_path / "replication-b",
        8001,
    )
    evidence = json.loads(first.evidence_path.read_text(encoding="utf-8"))
    assert evidence["spec"]["arrival_center_s"] == 3.0
    assert evidence["spec"]["arrival_window_s"] == 1.0
    assert evidence["spec"]["arrival_jitter_fraction"] == 0.05
    assert evidence["spec"]["preprocessing_failure_mode"] == "fixed_count"
    assert evidence["spec"]["preprocessing_failure_count"] == 2
    assert evidence["spec"]["local_service_scale"] == 1.5
    assert first.evaluation_trace_path.read_bytes() == repeat.evaluation_trace_path.read_bytes()
