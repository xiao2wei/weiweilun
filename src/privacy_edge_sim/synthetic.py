"""Deterministic synthetic development fixtures.

The generated numbers are engineering assumptions in plausible automotive
edge-computing orders of magnitude.  They are useful for tests, debugging and
smoke runs only.  They are not measurements, literature-derived results, or
evidence for a paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .profiles import canonical_document_sha256, load_profile
from .traces import load_trace


DEFAULT_SYNTHETIC_SEED = 20260712
SYNTHETIC_PROTOCOL_VERSION = "anon-fer/1.0"
SYNTHETIC_SCHEMA_VERSION = "1.1.0"
DEVICES = ("vehicle_gpu_class_a", "vehicle_cpu_class_b")
VEHICLE_DEVICE = {"veh-1": DEVICES[0], "veh-2": DEVICES[1]}
VEHICLES = tuple(VEHICLE_DEVICE)
RSUS = ("rsu-1", "rsu-2")
QUALITY_BINS = ("clear", "challenging")
PIPELINES = ("pixelate_strong_v1", "blur_balanced_v1")
LOCAL_MODEL = "local_fer_compact_v1"
EDGE_MODEL = "edge_fer_full_v1"


@dataclass(frozen=True, slots=True)
class SyntheticBundlePaths:
    profile_path: Path
    trace_path: Path
    scenario_trace_path: Path
    profile_hash: str
    trace_hash: str
    scenario_trace_hash: str


def _component_hash(component: str) -> str:
    material = f"privacy-edge-sim|synthetic-fixture-only|{component}|v1".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _stream(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _source(category: str, description: str, unit: str) -> dict[str, str]:
    return {"category": category, "description": description, "unit": unit}


def generate_synthetic_profile(seed: int = DEFAULT_SYNTHETIC_SEED) -> dict[str, Any]:
    """Build a canonical synthetic-only frozen profile document."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    pipeline_defs = {
        "pixelate_strong_v1": {
            "guard_id": "synthetic_guard_v1",
            "encoder_id": "synthetic_jpeg_encoder_v1",
            "max_attempts": 3,
            "deployment_resource_bounds": {
                "max_peak_memory_bytes": 160 * 1024 * 1024,
                "max_anon_work_s": 0.16,
                "max_anon_energy_j": 1.50,
                "max_guard_work_s": 0.012,
                "max_guard_energy_j": 0.060,
                "max_encode_work_s": 0.012,
                "max_encode_energy_j": 0.060,
                "max_output_bytes": 128_000,
            },
        },
        "blur_balanced_v1": {
            "guard_id": "synthetic_guard_v1",
            "encoder_id": "synthetic_jpeg_encoder_v1",
            "max_attempts": 2,
            "deployment_resource_bounds": {
                "max_peak_memory_bytes": 96 * 1024 * 1024,
                "max_anon_work_s": 0.080,
                "max_anon_energy_j": 0.80,
                "max_guard_work_s": 0.012,
                "max_guard_energy_j": 0.060,
                "max_encode_work_s": 0.012,
                "max_encode_energy_j": 0.060,
                "max_output_bytes": 128_000,
            },
        },
    }
    pipelines: list[dict[str, Any]] = []
    for pipeline_id in PIPELINES:
        definition = pipeline_defs[pipeline_id]
        pipelines.append(
            {
                "pipeline_id": pipeline_id,
                "pipeline_hash": _component_hash(f"pipeline:{pipeline_id}"),
                "guard_id": definition["guard_id"],
                "guard_hash": _component_hash(f"guard:{definition['guard_id']}"),
                "encoder_id": definition["encoder_id"],
                "encoder_hash": _component_hash(f"encoder:{definition['encoder_id']}"),
                "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
                "max_attempts": definition["max_attempts"],
                "fallback_local_model": LOCAL_MODEL,
                "supported_devices": list(DEVICES),
                "retryable_reasons": [
                    "ANON_OOM",
                    "ANON_FAILED",
                    "GUARD_REJECTED",
                    "ENCODE_FAILED",
                ],
                "deployment_resource_bounds": definition["deployment_resource_bounds"],
            }
        )

    local_models = [
        {
            "model_id": LOCAL_MODEL,
            "model_hash": _component_hash(f"model:{LOCAL_MODEL}"),
            "model_kind": "local",
            "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
            "supported_devices": list(DEVICES),
            "supported_rsus": [],
            "supported_pipelines": [],
            "deployment_resource_bounds": {
                "max_memory_bytes": 256 * 1024 * 1024,
                "max_service_work_s": 0.080,
                "max_dynamic_energy_j": 0.70,
            },
        }
    ]
    edge_models = [
        {
            "model_id": EDGE_MODEL,
            "model_hash": _component_hash(f"model:{EDGE_MODEL}"),
            "model_kind": "edge",
            "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
            "supported_devices": [],
            "supported_rsus": list(RSUS),
            "supported_pipelines": list(PIPELINES),
            "deployment_resource_bounds": {
                "max_vram_bytes": 768 * 1024 * 1024,
                "max_ingress_work_s": 0.005,
                "max_ingress_energy_j": 0.080,
                "max_gpu_work_s": 0.030,
                "max_gpu_energy_j": 1.20,
                "max_result_size_bits": 4096,
            },
        }
    ]

    risk_values = {
        ("pixelate_strong_v1", "clear"): {
            "identity": 0.040,
            "verification": 0.048,
            "link": 0.058,
        },
        ("pixelate_strong_v1", "challenging"): {
            "identity": 0.064,
            "verification": 0.073,
            "link": 0.088,
        },
        ("blur_balanced_v1", "clear"): {
            "identity": 0.060,
            "verification": 0.070,
            "link": 0.082,
        },
        ("blur_balanced_v1", "challenging"): {
            "identity": 0.125,
            "verification": 0.142,
            "link": 0.168,
        },
    }
    privacy_cells: list[dict[str, Any]] = []
    for pipeline_id in PIPELINES:
        for quality_bin in QUALITY_BINS:
            for device_index, device in enumerate(DEVICES):
                subjects = 96 if quality_bin == "clear" else 84
                emission = 0.82 if quality_bin == "clear" else 0.71
                bounds = []
                for risk_type in ("identity", "verification", "link"):
                    bounds.append(
                        {
                            "risk_type": risk_type,
                            "attacker_id": f"synthetic_{risk_type}_attacker_fixture",
                            "threshold_id": "synthetic_pre_registered_threshold_v1",
                            "ucb": _rounded(
                                risk_values[(pipeline_id, quality_bin)][risk_type]
                                + 0.002 * device_index
                            ),
                            "subject_count": subjects,
                            "emission_lcb": _rounded(emission - 0.01 * device_index),
                            "confidence_error": 0.05,
                        }
                    )
                privacy_cells.append(
                    {
                        "pipeline_id": pipeline_id,
                        "quality_bin": quality_bin,
                        "device_type": device,
                        "joint_trace_supported": True,
                        "bounds": bounds,
                    }
                )

    document: dict[str, Any] = {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
        "profile_version": "synthetic-1.1.0",
        "profile_hash": "",
        "data_kind": "synthetic",
        "evidence_status": "synthetic_fixture_only",
        "online_mutable": False,
        "quality_bins": list(QUALITY_BINS),
        "preprocessing_resource_bounds": {
            "max_memory_bytes": 64 * 1024 * 1024,
            "max_service_work_s": 0.060,
            "max_dynamic_energy_j": 0.70,
        },
        "privacy_policy": {
            "registered_risk_types": ["identity", "verification", "link"],
            "risk_threshold": 0.10,
            "confidence_error": 0.05,
            "min_subjects": 64,
            "min_emission_lcb": 0.60,
            "interpretation": (
                "Synthetic development fixture only. Even for measured profiles, a safe result would mean "
                "only an empirical subject-level bound for the frozen population and pre-registered attackers."
            ),
        },
        "pipelines": pipelines,
        "local_models": local_models,
        "edge_models": edge_models,
        "privacy_cells": privacy_cells,
        "parameter_sources": {
            "privacy_bounds": _source(
                "engineering_assumption",
                "Fabricated bounded values chosen only to exercise all three hard-risk checks; not attack measurements.",
                "probability",
            ),
            "subject_support": _source(
                "engineering_assumption",
                "Synthetic cluster counts used to exercise minimum subject support.",
                "subjects",
            ),
            "emission_support": _source(
                "engineering_assumption",
                "Synthetic lower bounds used to exercise guard-emission support filtering.",
                "probability",
            ),
            "retry_limits": _source(
                "engineering_assumption",
                "Small finite attempt limits for deterministic smoke and boundary tests.",
                "attempts",
            ),
            "deployment_resource_bounds": _source(
                "engineering_assumption",
                "Preregistered finite physical envelopes used for online reservation and trace OOD rejection.",
                "resource_busy_s,J,bytes,bit",
            ),
        },
        "metadata": {
            "generator": "privacy_edge_sim.synthetic.generate_synthetic_profile",
            "seed": seed,
            "formal_experiment_eligible": False,
            "contains_real_images": False,
            "contains_trained_models": False,
            "warning": "SYNTHETIC ENGINEERING FIXTURE - NOT A MEASUREMENT OR PAPER RESULT",
        },
    }
    document["profile_hash"] = canonical_document_sha256(document, "profile_hash")
    return document


def _attempt(
    *,
    index: int,
    anon_work_s: float,
    anon_energy_j: float,
    peak_memory_bytes: int,
    oom: bool = False,
    guard_passed: bool | None = None,
    guard_work_s: float | None = None,
    guard_energy_j: float | None = None,
    encode_success: bool | None = None,
    encode_work_s: float | None = None,
    encode_energy_j: float | None = None,
    encoded_size_bytes: int | None = None,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    return {
        "attempt_index": index,
        "anon_work_s": _rounded(anon_work_s),
        "anon_energy_j": _rounded(anon_energy_j),
        "peak_memory_bytes": peak_memory_bytes,
        "anon_oom": oom,
        "guard_work_s": None if guard_work_s is None else _rounded(guard_work_s),
        "guard_energy_j": None if guard_energy_j is None else _rounded(guard_energy_j),
        "guard_passed": guard_passed,
        "encode_work_s": None if encode_work_s is None else _rounded(encode_work_s),
        "encode_energy_j": None
        if encode_energy_j is None
        else _rounded(encode_energy_j),
        "encode_success": encode_success,
        "encoded_size_bytes": encoded_size_bytes,
        "artifact_key": artifact_key,
    }


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


def _profile_map(
    profile: Mapping[str, Any], section: str, key: str
) -> dict[str, Mapping[str, Any]]:
    return {str(item[key]): item for item in profile[section]}


def generate_synthetic_trace(
    profile: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    *,
    split_role: str = "evaluation",
) -> dict[str, Any]:
    """Build a joint synthetic trace paired to ``profile`` and ``seed``."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if split_role not in {"evaluation", "training_validation"}:
        raise ValueError(
            "synthetic split_role must be evaluation or training_validation"
        )
    profile_document = dict(profile or generate_synthetic_profile(seed))
    pipeline_map = _profile_map(profile_document, "pipelines", "pipeline_id")
    local_map = _profile_map(profile_document, "local_models", "model_id")
    edge_map = _profile_map(profile_document, "edge_models", "model_id")
    rng = _stream(seed, "joint-compute")
    edge_hash = str(edge_map[EDGE_MODEL]["model_hash"])

    anon_rows: list[dict[str, Any]] = []
    artifact_records: list[tuple[str, str, str, float]] = []
    for pipeline_id in PIPELINES:
        pipeline = pipeline_map[pipeline_id]
        for quality_bin in QUALITY_BINS:
            for device in DEVICES:
                for context in _contexts():
                    slower = 1.0
                    slower *= 1.35 if device == "vehicle_cpu_class_b" else 1.0
                    slower *= 1.28 if quality_bin == "challenging" else 1.0
                    # Thermal throttling is modeled by environment service rate, not by
                    # silently changing resource work here.
                    base = 0.052 if pipeline_id == "pixelate_strong_v1" else 0.034
                    energy_power = 13.5 if device == "vehicle_gpu_class_a" else 9.0
                    output_base = (
                        54_000 if pipeline_id == "pixelate_strong_v1" else 82_000
                    )
                    output_base += 12_000 if quality_bin == "challenging" else 0
                    fer_loss = (
                        0.18 if pipeline_id == "pixelate_strong_v1" else 0.11
                    ) + (0.10 if quality_bin == "challenging" else 0.0)
                    for sample_index in range(2):
                        row_suffix = (
                            f"{split_role}-{pipeline_id}-{quality_bin}-{device}-"
                            f"{context['thermal_state']}-{sample_index}"
                        )
                        artifact_key = f"synthetic-artifact-{hashlib.sha256(row_suffix.encode()).hexdigest()[:20]}"
                        success_work = base * slower * (1.0 + rng.uniform(-0.06, 0.08))
                        success_energy = success_work * energy_power
                        peak_memory = (
                            96 * 1024 * 1024
                            if pipeline_id == "pixelate_strong_v1"
                            else 72 * 1024 * 1024
                        )
                        attempts: list[dict[str, Any]] = []
                        if sample_index == 1:
                            if (
                                quality_bin == "challenging"
                                and pipeline_id == "pixelate_strong_v1"
                            ):
                                attempts.append(
                                    _attempt(
                                        index=1,
                                        anon_work_s=success_work * 1.45,
                                        anon_energy_j=success_energy * 1.32,
                                        peak_memory_bytes=150 * 1024 * 1024,
                                        oom=True,
                                    )
                                )
                            else:
                                attempts.append(
                                    _attempt(
                                        index=1,
                                        anon_work_s=success_work * 0.92,
                                        anon_energy_j=success_energy * 0.90,
                                        peak_memory_bytes=peak_memory,
                                        guard_work_s=0.0085,
                                        guard_energy_j=0.052,
                                        guard_passed=False,
                                    )
                                )
                        attempts.append(
                            _attempt(
                                index=len(attempts) + 1,
                                anon_work_s=success_work,
                                anon_energy_j=success_energy,
                                peak_memory_bytes=peak_memory,
                                guard_work_s=0.008
                                + (0.002 if quality_bin == "challenging" else 0.0),
                                guard_energy_j=0.050,
                                guard_passed=True,
                                encode_work_s=0.007 + output_base / 45_000_000.0,
                                encode_energy_j=0.041 + output_base / 8_000_000.0,
                                encode_success=True,
                                encoded_size_bytes=output_base + sample_index * 7_000,
                                artifact_key=artifact_key,
                            )
                        )
                        measured_loss = _rounded(
                            min(0.95, fer_loss + rng.uniform(-0.012, 0.012))
                        )
                        anon_rows.append(
                            {
                                "row_id": f"anon-{len(anon_rows):04d}",
                                "subject_cluster_id": f"synthetic-subject-{len(anon_rows) % 12:02d}",
                                "pipeline_id": pipeline_id,
                                "pipeline_hash": pipeline["pipeline_hash"],
                                "guard_hash": pipeline["guard_hash"],
                                "encoder_hash": pipeline["encoder_hash"],
                                "quality_bin": quality_bin,
                                "device_type": device,
                                "context": dict(context),
                                "attempts": attempts,
                                "formed_packet": True,
                                "final_encoded_size_bytes": attempts[-1][
                                    "encoded_size_bytes"
                                ],
                                "artifact_key": artifact_key,
                                "fer_measurements": [
                                    {
                                        "model_id": EDGE_MODEL,
                                        "model_hash": edge_hash,
                                        "valid": True,
                                        "fer_loss": measured_loss,
                                    }
                                ],
                            }
                        )
                        artifact_records.append(
                            (artifact_key, pipeline_id, quality_bin, measured_loss)
                        )

    task_specs = (
        ("task-001", "fixture-001", "veh-1", 0.10, 0.90, ("clear",), False),
        (
            "task-002",
            "fixture-002",
            "veh-2",
            0.10,
            1.15,
            ("clear", "challenging"),
            False,
        ),
        ("task-003", "fixture-003", "veh-1", 0.72, 0.70, ("challenging",), False),
        ("task-004", "fixture-004", "veh-2", 1.42, 1.35, ("clear",), False),
        ("task-005", "fixture-005", "veh-1", 2.80, 0.62, ("challenging",), False),
        (
            "task-006",
            "fixture-006",
            "veh-2",
            3.48,
            1.20,
            ("clear", "challenging"),
            False,
        ),
        ("task-007", "fixture-007", "veh-1", 4.95, 0.85, ("clear",), True),
        ("task-008", "fixture-008", "veh-2", 5.82, 1.35, ("challenging",), False),
    )
    arrivals: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []
    for (
        task_id,
        fixture_key,
        vehicle_id,
        arrival_s,
        deadline_s,
        candidates,
        ood,
    ) in task_specs:
        primary_quality = candidates[0]
        arrivals.append(
            {
                "task_id": task_id,
                "fixture_key": fixture_key,
                "vehicle_id": vehicle_id,
                "arrival_time_s": arrival_s,
                "relative_deadline_s": deadline_s,
                "quality_candidates": list(candidates),
                "quality_probabilities": {
                    quality_bin: 1.0 / len(candidates) for quality_bin in candidates
                },
                "true_quality_region": primary_quality,
                "ood": ood,
                "quality_features": {
                    "face_scale": 0.72 if primary_quality == "clear" else 0.38,
                    "abs_yaw_deg": 8.0 if primary_quality == "clear" else 28.0,
                    "blur_index": 0.12 if primary_quality == "clear" else 0.48,
                    "occlusion_fraction": 0.04 if primary_quality == "clear" else 0.24,
                },
            }
        )
        device = VEHICLE_DEVICE[vehicle_id]
        for candidate_quality in candidates:
            for context in _contexts():
                prep_work = (0.021 if device == DEVICES[0] else 0.038) * (
                    1.18 if candidate_quality == "challenging" else 1.0
                )
                prep_rows.append(
                    {
                        "row_id": f"prep-{len(prep_rows):04d}",
                        "fixture_key": fixture_key,
                        "quality_bin": candidate_quality,
                        "device_type": device,
                        "context": dict(context),
                        "service_work_s": _rounded(prep_work),
                        "dynamic_energy_j": _rounded(
                            prep_work * (11.5 if device == DEVICES[0] else 7.5)
                        ),
                        "memory_bytes": 48 * 1024 * 1024,
                        "failed": bool(ood),
                    }
                )

    local_rows: list[dict[str, Any]] = []
    local_hash = str(local_map[LOCAL_MODEL]["model_hash"])
    for quality_bin in QUALITY_BINS:
        for device in DEVICES:
            for context in _contexts():
                base_work = 0.033 if device == DEVICES[0] else 0.061
                base_loss = 0.09 if quality_bin == "clear" else 0.23
                for sample_index in range(2):
                    work = base_work * (1.0 + 0.10 * sample_index)
                    local_rows.append(
                        {
                            "row_id": f"local-{len(local_rows):04d}",
                            "subject_cluster_id": (
                                f"synthetic-local-subject-{sample_index:02d}"
                            ),
                            "model_id": LOCAL_MODEL,
                            "model_hash": local_hash,
                            "quality_bin": quality_bin,
                            "device_type": device,
                            "context": dict(context),
                            "service_work_s": _rounded(work),
                            "dynamic_energy_j": _rounded(
                                work * (12.0 if device == DEVICES[0] else 7.8)
                            ),
                            "memory_bytes": 220 * 1024 * 1024,
                            "failed": False,
                            "fer_loss": _rounded(base_loss + 0.015 * sample_index),
                        }
                    )

    edge_rows: list[dict[str, Any]] = []
    for artifact_key, pipeline_id, quality_bin, measured_loss in artifact_records:
        for rsu_id in RSUS:
            for context in _contexts():
                ingress_work = 0.0032
                gpu_work = 0.015 if quality_bin == "clear" else 0.021
                if rsu_id == "rsu-2":
                    gpu_work *= 1.12
                edge_rows.append(
                    {
                        "row_id": f"edge-{len(edge_rows):05d}",
                        "artifact_key": artifact_key,
                        "pipeline_id": pipeline_id,
                        "quality_bin": quality_bin,
                        "rsu_id": rsu_id,
                        "model_id": EDGE_MODEL,
                        "model_hash": edge_hash,
                        "context": dict(context),
                        "ingress_work_s": _rounded(ingress_work),
                        "ingress_energy_j": 0.055,
                        "gpu_work_s": _rounded(gpu_work),
                        "gpu_energy_j": _rounded(gpu_work * 42.0),
                        "vram_bytes": 640 * 1024 * 1024,
                        "result_size_bits": 2_048,
                        "ingress_failed": False,
                        "failed": False,
                        "fer_loss": measured_loss,
                    }
                )

    wireless: list[dict[str, Any]] = []
    boundaries = (0.0, 1.4, 1.8, 3.5, 4.0, 6.0, 8.0)
    for vehicle_id in VEHICLES:
        for rsu_id in RSUS:
            for direction in ("UL", "DL"):
                for segment_index, (start, end) in enumerate(
                    zip(boundaries, boundaries[1:])
                ):
                    state = "connected"
                    if vehicle_id == "veh-1" and rsu_id == "rsu-1" and start == 1.4:
                        state = "temporary_outage"
                    elif vehicle_id == "veh-1" and rsu_id == "rsu-2" and start == 3.5:
                        state = "handover"
                    elif vehicle_id == "veh-2" and rsu_id == "rsu-2" and start == 3.5:
                        state = "temporary_outage"
                    elif vehicle_id == "veh-2" and rsu_id == "rsu-1" and start >= 6.0:
                        state = "permanent_loss"
                    if state == "connected":
                        base_rate = 10_000_000.0 if direction == "UL" else 24_000_000.0
                        if rsu_id == "rsu-2":
                            base_rate *= 0.78
                        if vehicle_id == "veh-2":
                            base_rate *= 0.72
                        burst = (0.70, 1.30, 0.88, 1.18, 0.62, 1.05)[segment_index]
                        rate = base_rate * burst
                    else:
                        rate = 0.0
                    wireless.append(
                        {
                            "segment_id": f"radio-{vehicle_id}-{rsu_id}-{direction}-{segment_index}",
                            "vehicle_id": vehicle_id,
                            "rsu_id": rsu_id,
                            "direction": direction,
                            "start_time_s": start,
                            "end_time_s": end,
                            "goodput_bps": _rounded(rate),
                            "transmitter_power_w": 3.2 if direction == "UL" else 4.6,
                            "receiver_power_w": 1.4 if direction == "UL" else 1.15,
                            "link_state": state,
                        }
                    )

    thermal: list[dict[str, Any]] = []
    for owner_type, owner_ids in (("vehicle", VEHICLES), ("rsu", RSUS)):
        for owner_id in owner_ids:
            for segment_index, (start, end, state, rate, power) in enumerate(
                (
                    (0.0, 2.5, "nominal", 1.0, 1.0),
                    (2.5, 4.0, "hot_throttled", 0.68, 0.88),
                    (4.0, 8.0, "recovered", 1.0, 1.0),
                )
            ):
                thermal.append(
                    {
                        "segment_id": f"thermal-{owner_type}-{owner_id}-{segment_index}",
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                        "resource": "all",
                        "start_time_s": start,
                        "end_time_s": end,
                        "state": state,
                        "service_rate_multiplier": rate,
                        "dynamic_power_multiplier": power,
                    }
                )

    events = [
        {
            "event_id": "fault-rsu2-start",
            "time_s": 2.80,
            "event_type": "DEVICE_FAULT_START",
            "target_type": "rsu",
            "target_id": "rsu-2",
            "resource": "all",
            "old_version": None,
            "new_version": None,
            "permanent": False,
            "details": {"fixture_reason": "synthetic transient RSU service fault"},
        },
        {
            "event_id": "fault-rsu2-end",
            "time_s": 3.05,
            "event_type": "DEVICE_FAULT_END",
            "target_type": "rsu",
            "target_id": "rsu-2",
            "resource": "all",
            "old_version": None,
            "new_version": None,
            "permanent": False,
            "details": {"fixture_reason": "synthetic recovery"},
        },
        {
            "event_id": "model-version-rsu1",
            "time_s": 5.25,
            "event_type": "MODEL_VERSION",
            "target_type": "rsu",
            "target_id": "rsu-1",
            "resource": "model_cache",
            "old_version": edge_hash,
            "new_version": _component_hash("model:edge_fer_full_v2_unprofiled"),
            "permanent": False,
            "maintenance_work_s": 0.18,
            "maintenance_energy_j": 12.6,
            "details": {
                "fixture_reason": "synthetic incompatible version transition",
                "model_id": EDGE_MODEL,
                "maintenance_parameter_source": "engineering_assumption",
            },
        },
    ]

    document: dict[str, Any] = {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
        "trace_version": "synthetic-1.1.0",
        "trace_hash": "",
        "profile_hash": profile_document["profile_hash"],
        "data_kind": "synthetic",
        "evidence_status": "synthetic_fixture_only",
        "seed": seed,
        "horizon": {"start_time_s": 0.0, "end_time_s": 8.0},
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
                "engineering_assumption",
                "Detection/alignment/quality preprocessing fixture at 21-45 ms and sub-joule dynamic energy.",
                "resource_busy_s,J,bytes",
            ),
            "anonymization_transactions": _source(
                "engineering_assumption",
                "Joint retry/guard/encode rows, including correlated tail attempts and an OOM prefix.",
                "resource_busy_s,J,bytes",
            ),
            "fer_compute": _source(
                "engineering_assumption",
                "Compact local and server FER service/energy orders of magnitude; no model was run.",
                "resource_busy_s,J,bytes",
            ),
            "wireless": _source(
                "engineering_assumption",
                "4G/5G-class application-goodput fixture with bursts, pauses, handover and permanent loss.",
                "bit/s,W",
            ),
            "thermal": _source(
                "stress_test_boundary",
                "A finite hot interval reduces effective compute rate to 0.68; it never raises capacity.",
                "dimensionless",
            ),
            "faults_versions": _source(
                "stress_test_boundary",
                "Transient RSU fault and an incompatible cached-model version event.",
                "event",
            ),
            "arrivals_deadlines": _source(
                "engineering_assumption",
                "Small bursty workload with 0.62-1.35 second deadlines for smoke tests.",
                "s",
            ),
        },
        "metadata": {
            "generator": "privacy_edge_sim.synthetic.generate_synthetic_trace",
            "data_split": {
                "role": split_role,
                "seed": seed,
                "independence_scope": "independent_random_realization_only",
                "artifact_namespace_disjoint": True,
                "fixture_namespace_disjoint": False,
                "subject_population_disjoint": False,
            },
            "formal_experiment_eligible": False,
            "contains_real_measurements": False,
            "joint_rows_must_not_be_split": True,
            "warning": "SYNTHETIC ENGINEERING FIXTURE - NOT A MEASUREMENT OR PAPER RESULT",
        },
    }
    document["trace_hash"] = canonical_document_sha256(document, "trace_hash")
    return document


def _write_json(path: Path, document: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing synthetic fixture: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_synthetic_bundle(
    output_root: str | Path,
    *,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    overwrite: bool = False,
) -> SyntheticBundlePaths:
    """Write and immediately revalidate a profile/trace fixture pair."""

    root = Path(output_root).resolve()
    profile_document = generate_synthetic_profile(seed)
    trace_document = generate_synthetic_trace(
        profile_document, seed, split_role="evaluation"
    )
    scenario_trace_document = generate_synthetic_trace(
        profile_document,
        seed + 1,
        split_role="training_validation",
    )
    profile_path = root / "profiles" / "synthetic_profile.json"
    trace_path = root / "traces" / "synthetic_trace.json"
    scenario_trace_path = root / "traces" / "synthetic_scenario_trace.json"
    _write_json(profile_path, profile_document, overwrite=overwrite)
    _write_json(trace_path, trace_document, overwrite=overwrite)
    _write_json(scenario_trace_path, scenario_trace_document, overwrite=overwrite)
    frozen_profile = load_profile(profile_path)
    frozen_trace = load_trace(trace_path, frozen_profile)
    frozen_scenario_trace = load_trace(scenario_trace_path, frozen_profile)
    return SyntheticBundlePaths(
        profile_path=profile_path,
        trace_path=trace_path,
        scenario_trace_path=scenario_trace_path,
        profile_hash=frozen_profile.profile_hash,
        trace_hash=frozen_trace.trace_hash,
        scenario_trace_hash=frozen_scenario_trace.trace_hash,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic-only development profile and joint trace"
    )
    parser.add_argument("--output-root", default=".", help="repository/output root")
    parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paths = generate_synthetic_bundle(
        args.output_root, seed=args.seed, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "profile_path": str(paths.profile_path),
                "trace_path": str(paths.trace_path),
                "scenario_trace_path": str(paths.scenario_trace_path),
                "profile_hash": paths.profile_hash,
                "trace_hash": paths.trace_hash,
                "scenario_trace_hash": paths.scenario_trace_hash,
                "data_kind": "synthetic",
                "formal_experiment_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module entrypoint
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SYNTHETIC_SEED",
    "SyntheticBundlePaths",
    "generate_synthetic_bundle",
    "generate_synthetic_profile",
    "generate_synthetic_trace",
]
