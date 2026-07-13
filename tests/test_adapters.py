from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

import pytest

from privacy_edge_sim.adapters import (
    AdapterKind,
    BoundEdgeFERAdapter,
    BoundVehiclePipelineAdapter,
    ExecutionDomain,
    FrozenAdapterDescriptor,
    FrozenAdapterRegistry,
)
from privacy_edge_sim.errors import AdapterValidationError
from privacy_edge_sim.packets import (
    AlignedTensorHandle,
    AnonFERRequest,
    FERResult,
    _finalize_encoded_anon,
    _replay_anonymization_success,
    _replay_encoding_success,
    _replay_guard_success,
)
from privacy_edge_sim.profiles import canonical_document_sha256, load_profile


class _EdgeWrapper:
    def __init__(self, descriptor: FrozenAdapterDescriptor) -> None:
        self.descriptor = descriptor
        self.state_hash = descriptor.component_hash
        self.invalid_output = False
        self.mutate_during_infer = False
        self.raise_after_mutation = False
        self.result_size_bits = 1024

    def state_sha256(self) -> str:
        return self.state_hash

    def infer(self, request: object) -> object:
        if self.mutate_during_infer:
            self.state_hash = "f" * 64
        if self.raise_after_mutation:
            self.state_hash = "e" * 64
            raise RuntimeError("wrapper inference failed")
        if self.invalid_output:
            return request
        assert isinstance(request, AnonFERRequest)
        return FERResult(
            task_id=request.task_id,
            model_id=self.descriptor.component_id,
            model_hash=self.descriptor.component_hash,
            protocol_version=self.descriptor.protocol_version,
            result_code=2,
            valid=True,
            size_bits=self.result_size_bits,
        )


def _attested_edge_profile(profile, tmp_path):
    artifact = tmp_path / "frozen-edge.lock"
    artifact.write_bytes(b"canonical deployment lock for frozen edge model")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    document = json.loads(profile.source_path.read_text(encoding="utf-8"))
    for model in document["edge_models"]:
        if model["model_id"] == "edge_fer_full_v1":
            model["model_hash"] = digest
    document["profile_hash"] = canonical_document_sha256(document, "profile_hash")
    profile_path = tmp_path / "attested-profile.json"
    profile_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attested_profile = load_profile(profile_path)
    return attested_profile, artifact, digest


def _edge_descriptor(profile, artifact, digest, **overrides) -> FrozenAdapterDescriptor:
    model = profile.edge_models["edge_fer_full_v1"]
    values = {
        "kind": AdapterKind.EDGE_FER,
        "component_id": model.model_id,
        "component_hash": model.model_hash,
        "profile_hash": profile.profile_hash,
        "protocol_version": profile.protocol_version,
        "execution_domain": ExecutionDomain.RSU,
        "artifact_path": artifact,
        "artifact_sha256": digest,
    }
    values.update(overrides)
    return FrozenAdapterDescriptor(**values)


def _attested_pipeline_profile(profile, tmp_path):
    artifact = tmp_path / "frozen-pipeline.lock"
    artifact.write_bytes(b"canonical deployment lock for full vehicle pipeline")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    document = json.loads(profile.source_path.read_text(encoding="utf-8"))
    for pipeline in document["pipelines"]:
        if pipeline["pipeline_id"] == "pixelate_strong_v1":
            pipeline["pipeline_hash"] = digest
    document["profile_hash"] = canonical_document_sha256(document, "profile_hash")
    profile_path = tmp_path / "pipeline-profile.json"
    profile_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_profile(profile_path), artifact, digest


def _pipeline_encoded(profile, aligned, task_id, attempt):
    pipeline = profile.pipelines["pixelate_strong_v1"]
    anonymized = _replay_anonymization_success(
        aligned=aligned,
        task_id=task_id,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        artifact_key=f"artifact:{task_id}:{attempt}",
        attempt=attempt,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash=pipeline.guard_hash,
        guard_certificate_id=f"guard:{task_id}:{attempt}",
    )
    encoding = _replay_encoding_success(
        guarded,
        payload=b"anon",
        encoder_hash=pipeline.encoder_hash,
        encoded_size_bytes=4,
    )
    return _finalize_encoded_anon(
        encoding,
        profile_hash=profile.profile_hash,
        quality_bins=("clear",),
    )


def _request(profile) -> AnonFERRequest:
    pipeline = profile.pipelines["pixelate_strong_v1"]
    model = profile.edge_models["edge_fer_full_v1"]
    anonymized = _replay_anonymization_success(
        aligned=AlignedTensorHandle("aligned:task-adapter"),
        task_id="task-adapter",
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        artifact_key="adapter-artifact",
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash=pipeline.guard_hash,
        guard_certificate_id="guard-certificate",
    )
    encoding = _replay_encoding_success(
        guarded,
        payload=b"anon",
        encoder_hash=pipeline.encoder_hash,
        encoded_size_bytes=4,
    )
    encoded = _finalize_encoded_anon(
        encoding,
        profile_hash=profile.profile_hash,
        quality_bins=("clear",),
    )
    return AnonFERRequest.from_encoded(
        encoded,
        protocol_version=profile.protocol_version,
        requested_edge_model=model.model_id,
        requested_edge_model_hash=model.model_hash,
        vehicle_id="veh-1",
        task_id="task-adapter",
    )


def test_frozen_adapter_registry_returns_attested_proxy(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    wrapper = _EdgeWrapper(_edge_descriptor(bound_profile, artifact, digest))
    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(wrapper)
    registry.seal(require_all=False)

    bound = registry.get(AdapterKind.EDGE_FER, wrapper.descriptor.component_id)
    assert isinstance(bound, BoundEdgeFERAdapter)
    assert bound is not wrapper
    result = bound.infer(_request(bound_profile))
    assert result.task_id == "task-adapter"
    assert result.model_hash == digest


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"component_hash": "0" * 64, "artifact_sha256": "0" * 64},
            "ADAPTER_COMPONENT_HASH_MISMATCH",
        ),
        ({"profile_hash": "0" * 64}, "ADAPTER_PROFILE_MISMATCH"),
        ({"protocol_version": "incompatible"}, "ADAPTER_PROTOCOL_MISMATCH"),
        (
            {"execution_domain": ExecutionDomain.VEHICLE_TRUSTED},
            "ADAPTER_DOMAIN_MISMATCH",
        ),
    ],
)
def test_registry_rejects_version_hash_and_domain_mismatch(
    profile, tmp_path, overrides, code
):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    registry = FrozenAdapterRegistry(bound_profile)
    descriptor = _edge_descriptor(bound_profile, artifact, digest, **overrides)
    with pytest.raises(AdapterValidationError, match=code):
        registry.attach(_EdgeWrapper(descriptor))


def test_arbitrary_self_hashed_artifact_is_not_profile_pinned(profile, tmp_path):
    artifact = tmp_path / "arbitrary.weights"
    artifact.write_bytes(b"arbitrary-unregistered-weights")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model = profile.edge_models["edge_fer_full_v1"]
    descriptor = FrozenAdapterDescriptor(
        kind=AdapterKind.EDGE_FER,
        component_id=model.model_id,
        component_hash=digest,
        profile_hash=profile.profile_hash,
        protocol_version=profile.protocol_version,
        execution_domain=ExecutionDomain.RSU,
        artifact_path=artifact,
        artifact_sha256=digest,
    )
    with pytest.raises(AdapterValidationError, match="ADAPTER_COMPONENT_HASH_MISMATCH"):
        FrozenAdapterRegistry(profile).attach(_EdgeWrapper(descriptor))


def test_file_and_descriptor_changes_are_detected_after_seal(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    wrapper = _EdgeWrapper(_edge_descriptor(bound_profile, artifact, digest))
    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(wrapper)
    registry.seal(require_all=False)

    original = wrapper.descriptor
    wrapper.descriptor = replace(original, component_id="mutated")
    with pytest.raises(AdapterValidationError, match="ADAPTER_DESCRIPTOR_MUTATED"):
        registry.get(AdapterKind.EDGE_FER, original.component_id)

    wrapper.descriptor = original
    artifact.write_bytes(b"mutated-after-seal")
    with pytest.raises(AdapterValidationError, match="ADAPTER_ARTIFACT_HASH_MISMATCH"):
        registry.get(AdapterKind.EDGE_FER, original.component_id)


def test_proxy_checks_runtime_input_output_and_state(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    wrapper = _EdgeWrapper(_edge_descriptor(bound_profile, artifact, digest))
    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(wrapper)
    registry.seal(require_all=False)
    bound = registry.get(AdapterKind.EDGE_FER, wrapper.descriptor.component_id)

    with pytest.raises(AdapterValidationError, match="ADAPTER_INPUT_TYPE"):
        bound.infer(b"raw-face-bytes")

    wrapper.invalid_output = True
    with pytest.raises(AdapterValidationError, match="ADAPTER_OUTPUT_TYPE"):
        bound.infer(_request(bound_profile))
    wrapper.invalid_output = False

    wrapper.result_size_bits = 0
    with pytest.raises(AdapterValidationError, match="ADAPTER_OUTPUT_VALUE"):
        bound.infer(_request(bound_profile))
    wrapper.result_size_bits = 1024

    wrapper.mutate_during_infer = True
    with pytest.raises(AdapterValidationError, match="ADAPTER_STATE_MUTATED"):
        bound.infer(_request(bound_profile))


def test_exception_path_still_attests_and_returns_structured_failure(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    wrapper = _EdgeWrapper(_edge_descriptor(bound_profile, artifact, digest))
    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(wrapper)
    registry.seal(require_all=False)
    wrapper.raise_after_mutation = True
    with pytest.raises(AdapterValidationError, match="ADAPTER_STATE_MUTATED") as caught:
        registry.get(AdapterKind.EDGE_FER, wrapper.descriptor.component_id).infer(
            _request(bound_profile)
        )
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_pipeline_proxy_binds_output_to_task_attempt_and_aligned_handle(
    profile, tmp_path
):
    bound_profile, artifact, digest = _attested_pipeline_profile(profile, tmp_path)
    pipeline = bound_profile.pipelines["pixelate_strong_v1"]
    aligned_a = AlignedTensorHandle("aligned:task-a")
    cached = _pipeline_encoded(bound_profile, aligned_a, "task-a", 1)

    class CachedPipelineWrapper:
        descriptor = FrozenAdapterDescriptor(
            kind=AdapterKind.VEHICLE_PIPELINE,
            component_id=pipeline.pipeline_id,
            component_hash=digest,
            profile_hash=bound_profile.profile_hash,
            protocol_version=bound_profile.protocol_version,
            execution_domain=ExecutionDomain.VEHICLE_TRUSTED,
            artifact_path=artifact,
            artifact_sha256=digest,
        )

        def state_sha256(self) -> str:
            return digest

        def execute(self, *args, **kwargs):
            return cached

    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(CachedPipelineWrapper())
    registry.seal(require_all=False)
    bound = registry.get(AdapterKind.VEHICLE_PIPELINE, pipeline.pipeline_id)
    assert isinstance(bound, BoundVehiclePipelineAdapter)
    assert (
        bound.execute(
            aligned_a,
            task_id="task-a",
            quality_bins=("clear",),
            attempt=1,
        )
        is cached
    )
    with pytest.raises(AdapterValidationError, match="ADAPTER_OUTPUT_IDENTITY"):
        bound.execute(
            AlignedTensorHandle("aligned:task-b"),
            task_id="task-b",
            quality_bins=("clear",),
            attempt=2,
        )


def test_online_wrapper_must_not_expose_training_api(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)

    class TrainableWrapper(_EdgeWrapper):
        def set_weights(self) -> None:
            pass

    registry = FrozenAdapterRegistry(bound_profile)
    descriptor = _edge_descriptor(bound_profile, artifact, digest)
    with pytest.raises(AdapterValidationError, match="ADAPTER_TRAINING_API_EXPOSED"):
        registry.attach(TrainableWrapper(descriptor))


def test_registry_conservatively_requires_all_declared_components(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    registry = FrozenAdapterRegistry(bound_profile)
    registry.attach(_EdgeWrapper(_edge_descriptor(bound_profile, artifact, digest)))
    with pytest.raises(AdapterValidationError, match="ADAPTER_BINDINGS_INCOMPLETE"):
        registry.seal()


def test_registry_rejects_in_memory_profile_not_matching_canonical_source(
    profile, tmp_path
):
    bound_profile, _, _ = _attested_edge_profile(profile, tmp_path)
    models = dict(bound_profile.edge_models)
    model = models["edge_fer_full_v1"]
    models[model.model_id] = replace(model, model_hash="0" * 64)
    stale = replace(bound_profile, edge_models=MappingProxyType(models))
    with pytest.raises(
        AdapterValidationError, match="ADAPTER_PROFILE_OBJECT_UNVERIFIED"
    ):
        FrozenAdapterRegistry(stale)


def test_descriptor_rejects_unpinned_or_mistyped_fields(profile, tmp_path):
    bound_profile, artifact, digest = _attested_edge_profile(profile, tmp_path)
    with pytest.raises(AdapterValidationError, match="ADAPTER_NOT_FROZEN"):
        _edge_descriptor(bound_profile, artifact, digest, online_mutable=True)
    with pytest.raises(
        AdapterValidationError, match="ADAPTER_ARTIFACT_NOT_PROFILE_PINNED"
    ):
        _edge_descriptor(bound_profile, artifact, digest, artifact_sha256="0" * 64)
    with pytest.raises(AdapterValidationError, match="ADAPTER_ARTIFACT_PATH_TYPE"):
        _edge_descriptor(bound_profile, "not-a-path", digest)
