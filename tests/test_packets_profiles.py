from __future__ import annotations

import json
import pickle
import random
from dataclasses import FrozenInstanceError

import pytest

from privacy_edge_sim.enums import ReasonCode
from privacy_edge_sim.errors import PacketConstructionError
from privacy_edge_sim.packets import (
    AlignedTensorHandle,
    AnonFERRequest,
    AnonymizationEvidence,
    EncodingEvidence,
    GuardCertificate,
    RawImageHandle,
    _finalize_encoded_anon,
    _replay_anonymization_success,
    _replay_encoding_success,
    _replay_guard_success,
)
from privacy_edge_sim.profiles import PRIVACY_RISK_TYPES


def _encoded(
    profile,
    *,
    artifact: str = "artifact-1",
    payload: bytes = b"anon",
    encoded_size_bytes: int | None = None,
    task_id: str = "task-1",
    aligned: AlignedTensorHandle | None = None,
):
    source = aligned or AlignedTensorHandle(f"aligned:{task_id}")
    anonymized = _replay_anonymization_success(
        aligned=source,
        task_id=task_id,
        pipeline_id="pixelate_strong_v1",
        pipeline_hash="a" * 64,
        artifact_key=artifact,
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash="b" * 64,
        guard_certificate_id="certificate-1",
    )
    encoding = _replay_encoding_success(
        guarded,
        payload=payload,
        encoder_hash="c" * 64,
        encoded_size_bytes=len(payload)
        if encoded_size_bytes is None
        else encoded_size_bytes,
    )
    return _finalize_encoded_anon(
        encoding,
        profile_hash=profile.profile_hash,
        quality_bins=("clear",),
    )


@pytest.mark.parametrize("handle_type", [RawImageHandle, AlignedTensorHandle])
def test_vehicle_local_handles_have_no_serialization_path(handle_type):
    handle = handle_type("secret-opaque-id")

    with pytest.raises(TypeError):
        bytes(handle)
    with pytest.raises(TypeError):
        pickle.dumps(handle)
    with pytest.raises(TypeError):
        handle.to_json()
    with pytest.raises(TypeError):
        json.dumps(handle)

    assert "secret-opaque-id" not in repr(handle)
    assert "vehicle-local" in repr(handle)


def test_uplink_requires_paired_guard_encoding_and_anon_evidence(profile):
    encoded = _encoded(profile)
    edge_model = profile.edge_models["edge_fer_full_v1"]
    request = AnonFERRequest.from_encoded(
        encoded,
        protocol_version=profile.protocol_version,
        requested_edge_model=edge_model.model_id,
        requested_edge_model_hash=edge_model.model_hash,
        vehicle_id="veh-1",
        task_id="task-1",
    )

    wire = request.to_wire_dict()
    assert wire["message_type"] == "AnonFERRequest"
    assert request.payload_bits == 32
    assert request.payload_size_bytes == 4
    assert set(wire).isdisjoint(
        {"raw_handle", "raw_image", "aligned_handle", "aligned_tensor"}
    )
    assert "raw" not in json.dumps(wire).lower()


def test_uplink_artifact_cannot_be_built_without_every_staged_capability(profile):
    anonymized = _replay_anonymization_success(
        aligned=AlignedTensorHandle("aligned:task-1"),
        task_id="task-1",
        pipeline_id="pixelate_strong_v1",
        pipeline_hash="a" * 64,
        artifact_key="artifact-1",
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash="b" * 64,
        guard_certificate_id="certificate-1",
    )
    edge_model = profile.edge_models["edge_fer_full_v1"]
    kwargs = {
        "protocol_version": profile.protocol_version,
        "requested_edge_model": edge_model.model_id,
        "requested_edge_model_hash": edge_model.model_hash,
        "vehicle_id": "veh-1",
        "task_id": "task-1",
    }

    for incomplete in (anonymized, guarded):
        with pytest.raises(
            PacketConstructionError, match="UPLINK_REQUIRES_ENCODED_ANON"
        ):
            AnonFERRequest.from_encoded(incomplete, **kwargs)
    with pytest.raises(
        PacketConstructionError, match="ENCODING_REQUIRES_GUARD_CAPABILITY"
    ):
        _replay_encoding_success(
            anonymized,
            payload=b"anon",
            encoder_hash="c" * 64,
            encoded_size_bytes=4,
        )
    with pytest.raises(
        PacketConstructionError, match="ENCODED_ANON_REQUIRES_ENCODING_CAPABILITY"
    ):
        _finalize_encoded_anon(
            guarded,
            profile_hash=profile.profile_hash,
            quality_bins=("clear",),
        )


def test_uplink_artifact_rejects_zero_size_and_is_immutable(profile):
    with pytest.raises(PacketConstructionError) as caught:
        _encoded(profile, payload=b"", encoded_size_bytes=0)
    assert caught.value.detail.code == "ENCODE_SIZE_INVALID"

    encoded = _encoded(profile)
    with pytest.raises(FrozenInstanceError):
        encoded._payload = b"raw-pixels"


def test_public_callers_cannot_self_issue_pipeline_evidence_or_raw_bytes():
    with pytest.raises((PacketConstructionError, TypeError)):
        AnonymizationEvidence("pixelate_strong_v1", "a" * 64, "artifact", 1)
    with pytest.raises((PacketConstructionError, TypeError)):
        GuardCertificate("artifact", "b" * 64, "self-signed")
    with pytest.raises((PacketConstructionError, TypeError)):
        EncodingEvidence("artifact", "c" * 64, 4, b"anon")

    import privacy_edge_sim.packets as packets

    assert not hasattr(packets, "build_encoded_anon")
    assert not hasattr(packets, "_issue_encoded_anon_from_trusted_pipeline")

    with pytest.raises(
        PacketConstructionError, match="ENCODED_ANON_REQUIRES_ENCODING_CAPABILITY"
    ):
        _finalize_encoded_anon(
            b"RAW_FACE_PIXELS",
            profile_hash="d" * 64,
            quality_bins=("clear",),
        )


def test_uplink_factory_rejects_raw_or_aligned_handles(profile):
    kwargs = dict(
        protocol_version=profile.protocol_version,
        requested_edge_model="edge_fer_full_v1",
        requested_edge_model_hash="d" * 64,
        vehicle_id="veh-1",
        task_id="task-1",
    )
    for illegal in (
        RawImageHandle("raw"),
        AlignedTensorHandle("aligned"),
        b"anonymous-looking",
    ):
        with pytest.raises(PacketConstructionError) as caught:
            AnonFERRequest.from_encoded(illegal, **kwargs)
        assert caught.value.detail.code == "UPLINK_REQUIRES_ENCODED_ANON"


def test_encoded_capability_is_bound_to_one_task_and_strict_attempt_type(profile):
    encoded = _encoded(profile, task_id="task-1")
    with pytest.raises(PacketConstructionError, match="UPLINK_TASK_BINDING_MISMATCH"):
        AnonFERRequest.from_encoded(
            encoded,
            protocol_version=profile.protocol_version,
            requested_edge_model="edge_fer_full_v1",
            requested_edge_model_hash="d" * 64,
            vehicle_id="veh-1",
            task_id="task-2",
        )
    with pytest.raises(ValueError, match="one-based integer"):
        _replay_anonymization_success(
            aligned=AlignedTensorHandle("aligned:task-1"),
            task_id="task-1",
            pipeline_id="pixelate_strong_v1",
            pipeline_hash="a" * 64,
            artifact_key="artifact-1",
            attempt=True,
        )
    with pytest.raises(PacketConstructionError, match="UPLINK_METADATA_MISSING"):
        AnonFERRequest.from_encoded(
            encoded,
            protocol_version=123,
            requested_edge_model="edge_fer_full_v1",
            requested_edge_model_hash="d" * 64,
            vehicle_id="veh-1",
            task_id="task-1",
        )


def test_uplink_public_constructor_cannot_forge_evidence():
    forged = dict(
        payload_b64="YW5vbg==",
        payload_size_bytes=4,
        artifact_key="artifact-1",
        pipeline_id="pixelate_strong_v1",
        pipeline_hash="a" * 64,
        guard_hash="b" * 64,
        encoder_hash="c" * 64,
        profile_hash="d" * 64,
        protocol_version="1",
        requested_edge_model="edge",
        requested_edge_model_hash="e" * 64,
        quality_bins=("clear",),
        vehicle_id="veh-1",
        task_id="task-1",
    )
    with pytest.raises((PacketConstructionError, TypeError)):
        AnonFERRequest(**forged)
    with pytest.raises(PacketConstructionError) as caught:
        AnonFERRequest(_provenance_token=object(), **forged)
    assert caught.value.detail.code == "UPLINK_PRIVATE_CONSTRUCTOR"


def test_profile_registers_all_three_privacy_risks(profile):
    assert set(PRIVACY_RISK_TYPES) == {"identity", "verification", "link"}
    assert profile.privacy_cells
    for cell in profile.privacy_cells.values():
        assert {bound.risk_type for bound in cell.bounds} == set(PRIVACY_RISK_TYPES)


def test_quality_candidate_safety_is_an_intersection(profile):
    clear = profile.query_privacy("blur_balanced_v1", ("clear",), "vehicle_gpu_class_a")
    both = profile.query_privacy(
        "blur_balanced_v1", ("clear", "challenging"), "vehicle_gpu_class_a"
    )

    assert clear.safe
    assert not both.safe
    assert ReasonCode.PRIVACY_RISK in both.reasons
    assert both.worst_ucb >= clear.worst_ucb
    assert profile.safe_pipelines(("clear", "challenging"), "vehicle_gpu_class_a") == (
        "pixelate_strong_v1",
    )


def test_profile_conservatively_rejects_ood_device_and_weak_support(profile):
    ood = profile.query_privacy(
        "pixelate_strong_v1", ("unregistered",), "vehicle_gpu_class_a"
    )
    device = profile.query_privacy("pixelate_strong_v1", ("clear",), "unknown-device")
    subjects = profile.query_privacy(
        "pixelate_strong_v1", ("clear",), "vehicle_gpu_class_a", min_subjects=1000
    )
    emission = profile.query_privacy(
        "pixelate_strong_v1", ("clear",), "vehicle_gpu_class_a", min_emission_lcb=0.9
    )

    assert not ood.safe and ReasonCode.OOD in ood.reasons
    assert not device.safe and ReasonCode.DEVICE_UNSUPPORTED in device.reasons
    assert not subjects.safe and ReasonCode.IDENTITY_SUPPORT in subjects.reasons
    assert not emission.safe and ReasonCode.EMISSION_SUPPORT in emission.reasons


def test_profile_compatibility_rejects_protocol_profile_and_component_versions(profile):
    pipeline = profile.pipelines["pixelate_strong_v1"]
    edge = profile.edge_models["edge_fer_full_v1"]
    valid = profile.validate_compatibility(
        protocol_version=profile.protocol_version,
        profile_hash=profile.profile_hash,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        guard_hash=pipeline.guard_hash,
        encoder_hash=pipeline.encoder_hash,
        edge_model_id=edge.model_id,
        edge_model_hash=edge.model_hash,
        device_type="vehicle_gpu_class_a",
        rsu_id="rsu-1",
    )
    invalid = profile.validate_compatibility(
        protocol_version="wrong-protocol",
        profile_hash="0" * 64,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash="1" * 64,
        guard_hash=pipeline.guard_hash,
        encoder_hash=pipeline.encoder_hash,
        edge_model_id=edge.model_id,
        edge_model_hash="2" * 64,
        device_type="vehicle_gpu_class_a",
        rsu_id="rsu-1",
    )

    assert valid.compatible
    assert not invalid.compatible
    assert {
        ReasonCode.PROTOCOL_MISMATCH,
        ReasonCode.PROFILE_MISMATCH,
        ReasonCode.VERSION_MISMATCH,
    }.issubset(invalid.reasons)


def test_joint_anonymization_transaction_is_sampled_as_one_frozen_row(trace):
    source = next(row for row in trace.anon_rows if len(row.attempts) >= 2)
    first = trace.sample_anon_transaction(
        source.pipeline_id,
        (source.quality_bin,),
        source.device_type,
        source.context,
        random.Random(20260712),
    )
    repeat = trace.sample_anon_transaction(
        source.pipeline_id,
        (source.quality_bin,),
        source.device_type,
        source.context,
        random.Random(20260712),
    )

    assert first.supported and first.value is not None
    assert repeat.supported and repeat.value is not None
    assert first.value is repeat.value
    assert first.value in trace.anon_rows
    frozen_row = next(
        row for row in trace.anon_rows if row.row_id == first.value.row_id
    )
    assert first.value.attempts is frozen_row.attempts
    assert first.value.fer_measurements is frozen_row.fer_measurements
    assert first.value.artifact_key == frozen_row.artifact_key
    assert first.value.final_encoded_size_bytes == frozen_row.final_encoded_size_bytes


def test_joint_trace_missing_condition_returns_unsupported(trace):
    source = trace.anon_rows[0]
    result = trace.sample_anon_transaction(
        source.pipeline_id,
        (source.quality_bin,),
        source.device_type,
        {
            "thermal_state": "unseen",
            "power_mode": "unseen",
            "memory_pressure": "unseen",
        },
        random.Random(1),
    )
    assert not result.supported
    assert result.value is None
    assert result.reason is ReasonCode.JOINT_TRACE_MISSING
