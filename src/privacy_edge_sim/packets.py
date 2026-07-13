"""Vehicle-domain handles and the only legal network packet constructors.

Raw and aligned image objects deliberately have no serialization path.  The
network request can only be constructed from evidence objects emitted by the
trusted anonymous-image pipeline.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import InitVar, dataclass
from typing import Any, NoReturn

from .errors import PacketConstructionError


class _NonSerializableVehicleHandle:
    __slots__ = ("_opaque_id",)

    def __init__(self, opaque_id: str) -> None:
        if not isinstance(opaque_id, str) or not opaque_id:
            raise ValueError("opaque_id must be a non-empty string")
        self._opaque_id = opaque_id

    @property
    def opaque_id(self) -> str:
        return self._opaque_id

    def __bytes__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is vehicle-local and non-serializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be pickled")

    def __getstate__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot expose state")

    def to_json(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be serialized")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<vehicle-local>)"


class RawImageHandle(_NonSerializableVehicleHandle):
    """Opaque handle whose referent exists only in the vehicle trusted domain."""


class AlignedTensorHandle(_NonSerializableVehicleHandle):
    """Opaque aligned representation; also prohibited from network/log encoding."""


_PIPELINE_EVIDENCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class AnonymizationEvidence:
    pipeline_id: str
    pipeline_hash: str
    artifact_key: str
    attempt: int
    task_id: str
    _source_binding: str

    def __init__(
        self,
        *,
        token: object,
        pipeline_id: str,
        pipeline_hash: str,
        artifact_key: str,
        attempt: int,
        task_id: str,
        source_binding: str,
    ) -> None:
        if token is not _PIPELINE_EVIDENCE_TOKEN:
            raise PacketConstructionError(
                "ANON_EVIDENCE_PRIVATE_CONSTRUCTOR",
                "anonymization evidence must be issued by the staged vehicle orchestrator",
            )
        if any(
            not isinstance(value, str) or not value
            for value in (
                pipeline_id,
                pipeline_hash,
                artifact_key,
                task_id,
                source_binding,
            )
        ):
            raise ValueError("anonymization evidence string fields must be non-empty")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt is a one-based integer")
        object.__setattr__(self, "pipeline_id", pipeline_id)
        object.__setattr__(self, "pipeline_hash", pipeline_hash)
        object.__setattr__(self, "artifact_key", artifact_key)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "_source_binding", source_binding)


@dataclass(frozen=True, slots=True, init=False)
class GuardCertificate:
    anonymized: AnonymizationEvidence
    guard_hash: str
    certificate_id: str

    def __init__(
        self,
        *,
        token: object,
        anonymized: AnonymizationEvidence,
        guard_hash: str,
        certificate_id: str,
    ) -> None:
        if token is not _PIPELINE_EVIDENCE_TOKEN:
            raise PacketConstructionError(
                "GUARD_EVIDENCE_PRIVATE_CONSTRUCTOR",
                "guard evidence must be issued by the staged vehicle orchestrator",
            )
        if not isinstance(anonymized, AnonymizationEvidence):
            raise PacketConstructionError(
                "GUARD_REQUIRES_ANON_CAPABILITY",
                "guard pass evidence requires an anonymization capability",
            )
        if any(
            not isinstance(value, str) or not value
            for value in (guard_hash, certificate_id)
        ):
            raise ValueError("guard evidence fields must be non-empty")
        object.__setattr__(self, "anonymized", anonymized)
        object.__setattr__(self, "guard_hash", guard_hash)
        object.__setattr__(self, "certificate_id", certificate_id)

    @property
    def artifact_key(self) -> str:
        return self.anonymized.artifact_key


@dataclass(frozen=True, slots=True, init=False)
class EncodingEvidence:
    guarded: GuardCertificate
    encoder_hash: str
    encoded_size_bytes: int
    _payload: bytes

    def __init__(
        self,
        *,
        token: object,
        guarded: GuardCertificate,
        encoder_hash: str,
        encoded_size_bytes: int,
        payload: bytes,
    ) -> None:
        if token is not _PIPELINE_EVIDENCE_TOKEN:
            raise PacketConstructionError(
                "ENCODING_EVIDENCE_PRIVATE_CONSTRUCTOR",
                "encoding evidence must be issued by the staged vehicle orchestrator",
            )
        if not isinstance(guarded, GuardCertificate):
            raise PacketConstructionError(
                "ENCODING_REQUIRES_GUARD_CAPABILITY",
                "encoding success requires a guard-pass capability",
            )
        if not isinstance(encoder_hash, str) or not encoder_hash:
            raise ValueError("encoding evidence fields must be non-empty")
        if (
            isinstance(encoded_size_bytes, bool)
            or not isinstance(encoded_size_bytes, int)
            or encoded_size_bytes <= 0
        ):
            raise PacketConstructionError(
                "ENCODE_SIZE_INVALID",
                "successful encoding requires a positive byte length",
            )
        if not isinstance(payload, bytes):
            raise PacketConstructionError(
                "ENCODE_PAYLOAD_TYPE", "encoded payload must be bytes"
            )
        if len(payload) != encoded_size_bytes:
            raise PacketConstructionError(
                "ENCODE_SIZE_INVALID",
                "encoded payload size must match its staged evidence",
                evidence_bytes=encoded_size_bytes,
                actual_bytes=len(payload),
            )
        object.__setattr__(self, "guarded", guarded)
        object.__setattr__(self, "encoder_hash", encoder_hash)
        object.__setattr__(self, "encoded_size_bytes", encoded_size_bytes)
        object.__setattr__(self, "_payload", payload)

    @property
    def artifact_key(self) -> str:
        return self.guarded.artifact_key

    @property
    def payload(self) -> bytes:
        return self._payload


_ENCODED_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class EncodedAnon:
    """An encoded anonymous artifact constructible only from matching evidence."""

    _payload: bytes
    artifact_key: str
    pipeline_id: str
    pipeline_hash: str
    guard_hash: str
    encoder_hash: str
    profile_hash: str
    quality_bins: tuple[str, ...]
    task_id: str
    attempt: int
    _source_binding: str

    def __init__(
        self,
        *,
        token: object,
        encoding: EncodingEvidence,
        profile_hash: str,
        quality_bins: tuple[str, ...],
    ) -> None:
        if token is not _ENCODED_TOKEN:
            raise PacketConstructionError(
                "ENCODED_ANON_PRIVATE_CONSTRUCTOR",
                "EncodedAnon must be created by the trusted vehicle pipeline issuer",
            )
        if not isinstance(encoding, EncodingEvidence):
            raise PacketConstructionError(
                "ENCODED_ANON_REQUIRES_ENCODING_CAPABILITY",
                "finalization requires the complete staged evidence chain",
            )
        anon = encoding.guarded.anonymized
        guard = encoding.guarded
        if (
            not isinstance(profile_hash, str)
            or not profile_hash
            or not isinstance(quality_bins, tuple)
            or not quality_bins
            or any(not isinstance(item, str) or not item for item in quality_bins)
        ):
            raise PacketConstructionError(
                "PROFILE_EVIDENCE_MISSING", "profile hash and quality bins are required"
            )
        object.__setattr__(self, "_payload", encoding.payload)
        object.__setattr__(self, "artifact_key", anon.artifact_key)
        object.__setattr__(self, "pipeline_id", anon.pipeline_id)
        object.__setattr__(self, "pipeline_hash", anon.pipeline_hash)
        object.__setattr__(self, "guard_hash", guard.guard_hash)
        object.__setattr__(self, "encoder_hash", encoding.encoder_hash)
        object.__setattr__(self, "profile_hash", profile_hash)
        object.__setattr__(self, "quality_bins", tuple(quality_bins))
        object.__setattr__(self, "task_id", anon.task_id)
        object.__setattr__(self, "attempt", anon.attempt)
        object.__setattr__(self, "_source_binding", anon._source_binding)

    @property
    def payload(self) -> bytes:
        return self._payload

    @property
    def size_bytes(self) -> int:
        return len(self._payload)


def _replay_anonymization_success(
    *,
    aligned: AlignedTensorHandle,
    task_id: str,
    pipeline_id: str,
    pipeline_hash: str,
    artifact_key: str,
    attempt: int,
) -> AnonymizationEvidence:
    """Create the first capability after a trusted/synthetic anonymization stage."""

    source_binding = _source_binding_digest(aligned, task_id)
    return AnonymizationEvidence(
        token=_PIPELINE_EVIDENCE_TOKEN,
        pipeline_id=pipeline_id,
        pipeline_hash=pipeline_hash,
        artifact_key=artifact_key,
        attempt=attempt,
        task_id=task_id,
        source_binding=source_binding,
    )


def _source_binding_digest(aligned: AlignedTensorHandle, task_id: str) -> str:
    if not isinstance(aligned, AlignedTensorHandle):
        raise PacketConstructionError(
            "ANON_REQUIRES_ALIGNED_HANDLE",
            "anonymization capability requires a vehicle-local aligned handle",
        )
    if not isinstance(task_id, str) or not task_id:
        raise PacketConstructionError(
            "ANON_TASK_ID", "task_id must be a non-empty string"
        )
    material = f"privacy-edge-source-binding\0{task_id}\0{aligned.opaque_id}".encode(
        "utf-8"
    )
    return hashlib.sha256(material).hexdigest()


def _encoded_matches_source(
    encoded: EncodedAnon,
    aligned: AlignedTensorHandle,
    task_id: str,
    attempt: int,
) -> bool:
    return (
        isinstance(encoded, EncodedAnon)
        and not isinstance(attempt, bool)
        and isinstance(attempt, int)
        and encoded.task_id == task_id
        and encoded.attempt == attempt
        and encoded._source_binding == _source_binding_digest(aligned, task_id)
    )


def _replay_guard_success(
    anonymized: AnonymizationEvidence,
    *,
    guard_hash: str,
    guard_certificate_id: str,
) -> GuardCertificate:
    """Create a guard-pass capability bound to one anonymized artifact."""

    return GuardCertificate(
        token=_PIPELINE_EVIDENCE_TOKEN,
        anonymized=anonymized,
        guard_hash=guard_hash,
        certificate_id=guard_certificate_id,
    )


def _replay_encoding_success(
    guarded: GuardCertificate,
    *,
    payload: bytes,
    encoder_hash: str,
    encoded_size_bytes: int,
) -> EncodingEvidence:
    """Create encoding evidence only from a guard-pass capability."""

    return EncodingEvidence(
        token=_PIPELINE_EVIDENCE_TOKEN,
        guarded=guarded,
        encoder_hash=encoder_hash,
        encoded_size_bytes=encoded_size_bytes,
        payload=payload,
    )


def _finalize_encoded_anon(
    encoding: EncodingEvidence,
    *,
    profile_hash: str,
    quality_bins: tuple[str, ...],
) -> EncodedAnon:
    """Finalize a network capability; this function accepts no raw byte input."""

    return EncodedAnon(
        token=_ENCODED_TOKEN,
        encoding=encoding,
        profile_hash=profile_hash,
        quality_bins=tuple(quality_bins),
    )


_REQUEST_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AnonFERRequest:
    """The sole uplink request type, constructible only from ``EncodedAnon``.

    The private provenance token is an ``InitVar``: it is required by the
    constructor, checked by identity, and never becomes packet state or wire
    data.  Consequently callers cannot synthesize evidence strings and bypass
    :meth:`from_encoded` through the public dataclass constructor.
    """

    _provenance_token: InitVar[object]
    payload_b64: str
    payload_size_bytes: int
    artifact_key: str
    pipeline_id: str
    pipeline_hash: str
    guard_hash: str
    encoder_hash: str
    profile_hash: str
    protocol_version: str
    requested_edge_model: str
    requested_edge_model_hash: str
    quality_bins: tuple[str, ...]
    vehicle_id: str
    task_id: str

    def __post_init__(self, _provenance_token: object) -> None:
        if _provenance_token is not _REQUEST_TOKEN:
            raise PacketConstructionError(
                "UPLINK_PRIVATE_CONSTRUCTOR",
                "AnonFERRequest must be created by from_encoded",
            )
        required = (
            self.payload_b64,
            self.artifact_key,
            self.pipeline_id,
            self.pipeline_hash,
            self.guard_hash,
            self.encoder_hash,
            self.profile_hash,
            self.protocol_version,
            self.requested_edge_model,
            self.requested_edge_model_hash,
            self.vehicle_id,
            self.task_id,
        )
        if (
            any(not isinstance(value, str) or not value for value in required)
            or not isinstance(self.quality_bins, tuple)
            or not self.quality_bins
            or any(not isinstance(item, str) or not item for item in self.quality_bins)
            or isinstance(self.payload_size_bytes, bool)
            or not isinstance(self.payload_size_bytes, int)
            or self.payload_size_bytes <= 0
        ):
            raise PacketConstructionError(
                "UPLINK_EVIDENCE_INVALID", "uplink evidence is incomplete"
            )
        try:
            decoded = base64.b64decode(self.payload_b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise PacketConstructionError(
                "UPLINK_PAYLOAD_INVALID", "payload is not canonical base64"
            ) from exc
        if len(decoded) != self.payload_size_bytes:
            raise PacketConstructionError(
                "UPLINK_SIZE_MISMATCH",
                "wire payload size does not match encoded evidence",
                declared_bytes=self.payload_size_bytes,
                decoded_bytes=len(decoded),
            )

    @classmethod
    def from_encoded(
        cls,
        encoded: EncodedAnon,
        *,
        protocol_version: str,
        requested_edge_model: str,
        requested_edge_model_hash: str,
        vehicle_id: str,
        task_id: str,
    ) -> "AnonFERRequest":
        if not isinstance(encoded, EncodedAnon):
            raise PacketConstructionError(
                "UPLINK_REQUIRES_ENCODED_ANON",
                "uplink construction accepts only EncodedAnon",
                received_type=type(encoded).__name__,
            )
        required = (
            protocol_version,
            requested_edge_model,
            requested_edge_model_hash,
            vehicle_id,
            task_id,
        )
        if any(not isinstance(value, str) or not value for value in required):
            raise PacketConstructionError(
                "UPLINK_METADATA_MISSING",
                "all protocol/model/task metadata is required",
            )
        if task_id != encoded.task_id:
            raise PacketConstructionError(
                "UPLINK_TASK_BINDING_MISMATCH",
                "encoded anonymous evidence is bound to a different task",
                encoded_task_id=encoded.task_id,
                requested_task_id=task_id,
            )
        return cls(
            _provenance_token=_REQUEST_TOKEN,
            payload_b64=base64.b64encode(encoded.payload).decode("ascii"),
            payload_size_bytes=encoded.size_bytes,
            artifact_key=encoded.artifact_key,
            pipeline_id=encoded.pipeline_id,
            pipeline_hash=encoded.pipeline_hash,
            guard_hash=encoded.guard_hash,
            encoder_hash=encoded.encoder_hash,
            profile_hash=encoded.profile_hash,
            protocol_version=protocol_version,
            requested_edge_model=requested_edge_model,
            requested_edge_model_hash=requested_edge_model_hash,
            quality_bins=encoded.quality_bins,
            vehicle_id=vehicle_id,
            task_id=task_id,
        )

    @property
    def payload_bits(self) -> int:
        return self.payload_size_bytes * 8

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "message_type": "AnonFERRequest",
            "payload_b64": self.payload_b64,
            "payload_size_bytes": self.payload_size_bytes,
            "artifact_key": self.artifact_key,
            "pipeline_id": self.pipeline_id,
            "pipeline_hash": self.pipeline_hash,
            "guard_hash": self.guard_hash,
            "encoder_hash": self.encoder_hash,
            "profile_hash": self.profile_hash,
            "protocol_version": self.protocol_version,
            "requested_edge_model": self.requested_edge_model,
            "requested_edge_model_hash": self.requested_edge_model_hash,
            "quality_bins": list(self.quality_bins),
            "vehicle_id": self.vehicle_id,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class FERResult:
    task_id: str
    model_id: str
    model_hash: str
    protocol_version: str
    result_code: int
    valid: bool
    size_bits: int
