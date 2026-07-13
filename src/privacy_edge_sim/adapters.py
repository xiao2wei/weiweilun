"""Attested bindings for externally supplied frozen inference wrappers.

No executable neural model is bundled with this repository.  A deployment
may attach trusted wrappers for a complete vehicle anonymize/guard/encode
transaction, local FER, and edge FER.  Each wrapper is hidden behind a bound
proxy that checks input/output types, versions, a profile-pinned deployment
artifact, and an in-memory state fingerprint before and after every call.

The profile ``component_hash`` is the SHA-256 of the frozen weight file or a
canonical deployment lock manifest containing all files for that component.
It is not an adapter-selected checksum.  A registry can be populated only
before ``seal`` and never returns the raw wrapper.

This is an engineering trust boundary inside one Python process, not a hostile
code sandbox.  A malicious wrapper can lie about its state fingerprint or
tamper with the interpreter; such code is outside the research threat model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from .errors import AdapterValidationError
from .packets import (
    AlignedTensorHandle,
    AnonFERRequest,
    EncodedAnon,
    FERResult,
    _encoded_matches_source,
)
from .profiles import FrozenProfileBundle, load_profile


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ONLINE_MUTATORS = frozenset(
    {
        "backward",
        "fit",
        "load_state_dict",
        "optimizer_step",
        "partial_fit",
        "set_weights",
        "step",
        "train",
        "train_on_batch",
        "update",
        "update_parameters",
    }
)


class AdapterKind(StrEnum):
    """Executable online boundaries represented by the frozen profile."""

    VEHICLE_PIPELINE = "vehicle_pipeline"
    LOCAL_FER = "local_fer"
    EDGE_FER = "edge_fer"


class ExecutionDomain(StrEnum):
    VEHICLE_TRUSTED = "vehicle_trusted"
    RSU = "rsu"


@dataclass(frozen=True, slots=True)
class FrozenAdapterDescriptor:
    """Immutable profile and deployment-lock identity for one wrapper."""

    kind: AdapterKind
    component_id: str
    component_hash: str
    profile_hash: str
    protocol_version: str
    execution_domain: ExecutionDomain
    artifact_path: Path
    artifact_sha256: str
    online_mutable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AdapterKind):
            raise AdapterValidationError(
                "ADAPTER_KIND_TYPE", "kind must be AdapterKind"
            )
        if not isinstance(self.execution_domain, ExecutionDomain):
            raise AdapterValidationError(
                "ADAPTER_DOMAIN_TYPE", "execution_domain must be ExecutionDomain"
            )
        if not isinstance(self.component_id, str) or not self.component_id:
            raise AdapterValidationError(
                "ADAPTER_DESCRIPTOR_FIELD", "component_id must be a non-empty string"
            )
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise AdapterValidationError(
                "ADAPTER_DESCRIPTOR_FIELD",
                "protocol_version must be a non-empty string",
            )
        for name, value in (
            ("component_hash", self.component_hash),
            ("profile_hash", self.profile_hash),
            ("artifact_sha256", self.artifact_sha256),
        ):
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise AdapterValidationError(
                    "ADAPTER_HASH_FORMAT",
                    "descriptor hashes must be lowercase SHA-256 hex",
                    field=name,
                )
        if not isinstance(self.artifact_path, Path):
            raise AdapterValidationError(
                "ADAPTER_ARTIFACT_PATH_TYPE", "artifact_path must be pathlib.Path"
            )
        if self.artifact_sha256 != self.component_hash:
            raise AdapterValidationError(
                "ADAPTER_ARTIFACT_NOT_PROFILE_PINNED",
                "artifact checksum must equal the profile component hash",
                component_id=self.component_id,
            )
        if self.online_mutable is not False:
            raise AdapterValidationError(
                "ADAPTER_NOT_FROZEN",
                "online_mutable must be exactly false",
                component_id=self.component_id,
            )


@runtime_checkable
class FrozenWrapper(Protocol):
    descriptor: FrozenAdapterDescriptor

    def state_sha256(self) -> str: ...


@runtime_checkable
class VehiclePipelineAdapter(FrozenWrapper, Protocol):
    """Trusted complete anonymize/guard/encode transaction."""

    def execute(
        self,
        aligned: AlignedTensorHandle,
        *,
        task_id: str,
        quality_bins: tuple[str, ...],
        attempt: int,
    ) -> EncodedAnon: ...


@runtime_checkable
class LocalFERAdapter(FrozenWrapper, Protocol):
    def infer(self, aligned: AlignedTensorHandle, *, task_id: str) -> FERResult: ...


@runtime_checkable
class EdgeFERAdapter(FrozenWrapper, Protocol):
    def infer(self, request: AnonFERRequest) -> FERResult: ...


@dataclass(frozen=True, slots=True)
class _ExpectedComponent:
    component_hash: str
    execution_domain: ExecutionDomain
    required_method: str
    guard_hash: str | None = None
    encoder_hash: str | None = None


def _profile_components(
    profile: FrozenProfileBundle,
) -> Mapping[tuple[AdapterKind, str], _ExpectedComponent]:
    expected: dict[tuple[AdapterKind, str], _ExpectedComponent] = {}
    for pipeline in profile.pipelines.values():
        expected[(AdapterKind.VEHICLE_PIPELINE, pipeline.pipeline_id)] = (
            _ExpectedComponent(
                component_hash=pipeline.pipeline_hash,
                execution_domain=ExecutionDomain.VEHICLE_TRUSTED,
                required_method="execute",
                guard_hash=pipeline.guard_hash,
                encoder_hash=pipeline.encoder_hash,
            )
        )
    for model in profile.local_models.values():
        expected[(AdapterKind.LOCAL_FER, model.model_id)] = _ExpectedComponent(
            component_hash=model.model_hash,
            execution_domain=ExecutionDomain.VEHICLE_TRUSTED,
            required_method="infer",
        )
    for model in profile.edge_models.values():
        expected[(AdapterKind.EDGE_FER, model.model_id)] = _ExpectedComponent(
            component_hash=model.model_hash,
            execution_domain=ExecutionDomain.RSU,
            required_method="infer",
        )
    return MappingProxyType(expected)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AdapterValidationError(
            "ADAPTER_ARTIFACT_READ",
            "cannot read the profile-pinned deployment artifact",
            path=str(path),
            error=str(exc),
        ) from exc
    return digest.hexdigest()


def _validate_fer_result(
    result: object,
    *,
    task_id: str,
    component_id: str,
    component_hash: str,
    protocol_version: str,
) -> FERResult:
    if not isinstance(result, FERResult):
        raise AdapterValidationError(
            "ADAPTER_OUTPUT_TYPE",
            "FER adapter must return FERResult",
            actual=type(result).__name__,
        )
    if (
        result.task_id != task_id
        or result.model_id != component_id
        or result.model_hash != component_hash
        or result.protocol_version != protocol_version
    ):
        raise AdapterValidationError(
            "ADAPTER_OUTPUT_IDENTITY",
            "FER result task/model/protocol identity does not match the binding",
            task_id=task_id,
            component_id=component_id,
        )
    if (
        isinstance(result.result_code, bool)
        or not isinstance(result.result_code, int)
        or not isinstance(result.valid, bool)
        or isinstance(result.size_bits, bool)
        or not isinstance(result.size_bits, int)
        or result.size_bits < 0
        or (result.valid and result.size_bits < 1)
    ):
        raise AdapterValidationError(
            "ADAPTER_OUTPUT_VALUE",
            "FER result values have invalid types or units",
            component_id=component_id,
        )
    return result


@dataclass(frozen=True, slots=True)
class _BoundAdapterBase:
    _wrapper: object = field(repr=False, compare=False)
    descriptor: FrozenAdapterDescriptor
    expected: _ExpectedComponent
    profile_source_path: Path
    profile_source_sha256: str

    def _attest(self) -> None:
        if getattr(self._wrapper, "descriptor", None) != self.descriptor:
            raise AdapterValidationError(
                "ADAPTER_DESCRIPTOR_MUTATED",
                "wrapper descriptor changed after deployment binding",
                component_id=self.descriptor.component_id,
            )
        path = self.descriptor.artifact_path.resolve()
        if _file_sha256(path) != self.descriptor.component_hash:
            raise AdapterValidationError(
                "ADAPTER_ARTIFACT_HASH_MISMATCH",
                "deployment artifact changed after binding",
                component_id=self.descriptor.component_id,
                path=str(path),
            )
        if _file_sha256(self.profile_source_path) != self.profile_source_sha256:
            raise AdapterValidationError(
                "ADAPTER_PROFILE_SOURCE_MUTATED",
                "the canonical frozen profile file changed after binding",
                path=str(self.profile_source_path),
            )
        fingerprint = getattr(self._wrapper, "state_sha256", None)
        if not callable(fingerprint):
            raise AdapterValidationError(
                "ADAPTER_STATE_ATTESTATION_MISSING",
                "wrapper must expose state_sha256()",
                component_id=self.descriptor.component_id,
            )
        try:
            actual = fingerprint()
        except Exception as exc:
            raise AdapterValidationError(
                "ADAPTER_STATE_ATTESTATION_ERROR",
                "state_sha256() failed",
                component_id=self.descriptor.component_id,
                error_type=type(exc).__name__,
            ) from exc
        if actual != self.descriptor.component_hash:
            raise AdapterValidationError(
                "ADAPTER_STATE_MUTATED",
                "in-memory parameter fingerprint differs from the frozen profile",
                component_id=self.descriptor.component_id,
                actual=actual,
            )

    def _invoke(self, method_name: str, *args: object, **kwargs: object) -> object:
        """Invoke a wrapper and attest even when it raises an exception."""

        self._attest()
        method = getattr(self._wrapper, method_name)
        try:
            result = method(*args, **kwargs)
        except Exception as execution_error:
            try:
                self._attest()
            except AdapterValidationError as attestation_error:
                raise attestation_error from execution_error
            raise AdapterValidationError(
                "ADAPTER_EXECUTION_ERROR",
                "frozen wrapper raised during inference",
                component_id=self.descriptor.component_id,
                error_type=type(execution_error).__name__,
            ) from execution_error
        self._attest()
        return result


@dataclass(frozen=True, slots=True)
class BoundVehiclePipelineAdapter(_BoundAdapterBase):
    profile_hash: str
    protocol_version: str

    def execute(
        self,
        aligned: AlignedTensorHandle,
        *,
        task_id: str,
        quality_bins: tuple[str, ...],
        attempt: int,
    ) -> EncodedAnon:
        if not isinstance(aligned, AlignedTensorHandle):
            raise AdapterValidationError(
                "ADAPTER_INPUT_TYPE", "vehicle pipeline requires AlignedTensorHandle"
            )
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(quality_bins, tuple)
            or not quality_bins
            or any(not isinstance(item, str) or not item for item in quality_bins)
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
        ):
            raise AdapterValidationError(
                "ADAPTER_INPUT_VALUE",
                "pipeline task, quality bins and attempt are invalid",
            )
        result = self._invoke(
            "execute",
            aligned,
            task_id=task_id,
            quality_bins=quality_bins,
            attempt=attempt,
        )
        if not isinstance(result, EncodedAnon):
            raise AdapterValidationError(
                "ADAPTER_OUTPUT_TYPE", "vehicle pipeline must return EncodedAnon"
            )
        if (
            result.pipeline_id != self.descriptor.component_id
            or result.pipeline_hash != self.descriptor.component_hash
            or result.profile_hash != self.profile_hash
            or result.guard_hash != self.expected.guard_hash
            or result.encoder_hash != self.expected.encoder_hash
            or result.quality_bins != tuple(quality_bins)
            or result.size_bytes <= 0
            or not _encoded_matches_source(result, aligned, task_id, attempt)
        ):
            raise AdapterValidationError(
                "ADAPTER_OUTPUT_IDENTITY",
                "pipeline output evidence differs from the frozen binding",
                component_id=self.descriptor.component_id,
            )
        return result


@dataclass(frozen=True, slots=True)
class BoundLocalFERAdapter(_BoundAdapterBase):
    protocol_version: str

    def infer(self, aligned: AlignedTensorHandle, *, task_id: str) -> FERResult:
        if (
            not isinstance(aligned, AlignedTensorHandle)
            or not isinstance(task_id, str)
            or not task_id
        ):
            raise AdapterValidationError(
                "ADAPTER_INPUT_TYPE",
                "local FER requires an aligned vehicle handle and task_id",
            )
        result = self._invoke("infer", aligned, task_id=task_id)
        return _validate_fer_result(
            result,
            task_id=task_id,
            component_id=self.descriptor.component_id,
            component_hash=self.descriptor.component_hash,
            protocol_version=self.protocol_version,
        )


@dataclass(frozen=True, slots=True)
class BoundEdgeFERAdapter(_BoundAdapterBase):
    profile_hash: str
    protocol_version: str

    def infer(self, request: AnonFERRequest) -> FERResult:
        if not isinstance(request, AnonFERRequest):
            raise AdapterValidationError(
                "ADAPTER_INPUT_TYPE", "edge FER accepts only AnonFERRequest"
            )
        if (
            request.requested_edge_model != self.descriptor.component_id
            or request.requested_edge_model_hash != self.descriptor.component_hash
            or request.profile_hash != self.profile_hash
            or request.protocol_version != self.protocol_version
        ):
            raise AdapterValidationError(
                "ADAPTER_INPUT_IDENTITY",
                "edge request model/profile/protocol differs from the binding",
                component_id=self.descriptor.component_id,
            )
        result = self._invoke("infer", request)
        return _validate_fer_result(
            result,
            task_id=request.task_id,
            component_id=self.descriptor.component_id,
            component_hash=self.descriptor.component_hash,
            protocol_version=self.protocol_version,
        )


BoundAdapter = BoundVehiclePipelineAdapter | BoundLocalFERAdapter | BoundEdgeFERAdapter


class FrozenAdapterRegistry:
    """Deployment-only registry returning attested invocation proxies."""

    def __init__(self, profile: FrozenProfileBundle) -> None:
        try:
            canonical_profile = load_profile(profile.source_path)
        except Exception as exc:
            raise AdapterValidationError(
                "ADAPTER_PROFILE_SOURCE_INVALID",
                "adapter registry requires a loader-verified profile source",
                path=str(profile.source_path),
                error_type=type(exc).__name__,
            ) from exc
        if canonical_profile != profile:
            raise AdapterValidationError(
                "ADAPTER_PROFILE_OBJECT_UNVERIFIED",
                "in-memory profile differs from its canonical source file",
                path=str(profile.source_path),
            )
        self._profile = profile
        self._profile_source_path = profile.source_path.resolve()
        self._profile_source_sha256 = _file_sha256(self._profile_source_path)
        self._expected = _profile_components(profile)
        self._bindings: dict[tuple[AdapterKind, str], BoundAdapter] = {}
        self._sealed = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def bindings(self) -> Mapping[tuple[AdapterKind, str], BoundAdapter]:
        if not self._sealed:
            raise AdapterValidationError(
                "ADAPTER_REGISTRY_UNSEALED",
                "bindings are unavailable until the registry is sealed",
            )
        return MappingProxyType(self._bindings)

    def attach(self, wrapper: object) -> None:
        if self._sealed:
            raise AdapterValidationError(
                "ADAPTER_REGISTRY_SEALED", "sealed registries cannot be changed online"
            )
        descriptor = getattr(wrapper, "descriptor", None)
        if not isinstance(descriptor, FrozenAdapterDescriptor):
            raise AdapterValidationError(
                "ADAPTER_DESCRIPTOR_MISSING",
                "wrapper must expose a FrozenAdapterDescriptor",
                adapter_type=type(wrapper).__name__,
            )
        key = (descriptor.kind, descriptor.component_id)
        expected = self._expected.get(key)
        if expected is None:
            raise AdapterValidationError(
                "ADAPTER_UNSUPPORTED_COMPONENT",
                "component is not declared by the frozen profile",
                kind=descriptor.kind.value,
                component_id=descriptor.component_id,
            )
        if descriptor.profile_hash != self._profile.profile_hash:
            raise AdapterValidationError(
                "ADAPTER_PROFILE_MISMATCH",
                "adapter was not bound to this frozen profile",
            )
        if descriptor.protocol_version != self._profile.protocol_version:
            raise AdapterValidationError(
                "ADAPTER_PROTOCOL_MISMATCH",
                "adapter protocol differs from the frozen profile",
            )
        if descriptor.component_hash != expected.component_hash:
            raise AdapterValidationError(
                "ADAPTER_COMPONENT_HASH_MISMATCH",
                "deployment artifact hash differs from the profile component hash",
            )
        if descriptor.execution_domain is not expected.execution_domain:
            raise AdapterValidationError(
                "ADAPTER_DOMAIN_MISMATCH",
                "component is attached in the wrong execution domain",
                expected=expected.execution_domain.value,
                actual=descriptor.execution_domain.value,
            )
        if not callable(getattr(wrapper, expected.required_method, None)):
            raise AdapterValidationError(
                "ADAPTER_METHOD_MISSING",
                "wrapper does not implement the required inference method",
                method=expected.required_method,
            )
        exposed_mutators = sorted(
            name
            for name in _FORBIDDEN_ONLINE_MUTATORS
            if callable(getattr(wrapper, name, None))
        )
        if exposed_mutators:
            raise AdapterValidationError(
                "ADAPTER_TRAINING_API_EXPOSED",
                "online wrappers must not expose parameter-update methods",
                methods=exposed_mutators,
            )
        if key in self._bindings:
            raise AdapterValidationError(
                "ADAPTER_DUPLICATE_BINDING", "component already has an attached wrapper"
            )
        base = {
            "_wrapper": wrapper,
            "descriptor": descriptor,
            "expected": expected,
            "profile_source_path": self._profile_source_path,
            "profile_source_sha256": self._profile_source_sha256,
        }
        if descriptor.kind is AdapterKind.VEHICLE_PIPELINE:
            bound: BoundAdapter = BoundVehiclePipelineAdapter(
                **base,
                profile_hash=self._profile.profile_hash,
                protocol_version=self._profile.protocol_version,
            )
        elif descriptor.kind is AdapterKind.LOCAL_FER:
            bound = BoundLocalFERAdapter(
                **base,
                protocol_version=self._profile.protocol_version,
            )
        else:
            bound = BoundEdgeFERAdapter(
                **base,
                profile_hash=self._profile.profile_hash,
                protocol_version=self._profile.protocol_version,
            )
        bound._attest()
        self._bindings[key] = bound

    def seal(self, *, require_all: bool = True) -> None:
        if self._sealed:
            if require_all:
                missing = sorted(
                    f"{kind.value}:{component_id}"
                    for kind, component_id in set(self._expected) - set(self._bindings)
                )
                if missing:
                    raise AdapterValidationError(
                        "ADAPTER_BINDINGS_INCOMPLETE",
                        "the sealed partial registry cannot satisfy a complete binding request",
                        missing=missing,
                    )
            return
        if require_all:
            missing = sorted(
                f"{kind.value}:{component_id}"
                for kind, component_id in set(self._expected) - set(self._bindings)
            )
            if missing:
                raise AdapterValidationError(
                    "ADAPTER_BINDINGS_INCOMPLETE",
                    "not every profile component has an executable wrapper",
                    missing=missing,
                )
        for binding in self._bindings.values():
            binding._attest()
        self._sealed = True

    def get(self, kind: AdapterKind, component_id: str) -> BoundAdapter:
        if not self._sealed:
            raise AdapterValidationError(
                "ADAPTER_REGISTRY_UNSEALED", "online lookup is forbidden before seal"
            )
        if (
            not isinstance(kind, AdapterKind)
            or not isinstance(component_id, str)
            or not component_id
        ):
            raise AdapterValidationError(
                "ADAPTER_LOOKUP_TYPE",
                "lookup requires AdapterKind and a non-empty component ID",
            )
        try:
            binding = self._bindings[(kind, component_id)]
        except KeyError as exc:
            raise AdapterValidationError(
                "ADAPTER_BINDING_UNAVAILABLE",
                "no frozen executable wrapper is attached",
                kind=kind.value,
                component_id=component_id,
            ) from exc
        binding._attest()
        return binding


__all__ = [
    "AdapterKind",
    "BoundEdgeFERAdapter",
    "BoundLocalFERAdapter",
    "BoundVehiclePipelineAdapter",
    "EdgeFERAdapter",
    "ExecutionDomain",
    "FrozenAdapterDescriptor",
    "FrozenAdapterRegistry",
    "LocalFERAdapter",
    "VehiclePipelineAdapter",
]
