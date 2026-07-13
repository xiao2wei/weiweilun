from __future__ import annotations

from dataclasses import replace

import pytest

from privacy_edge_sim.enums import EventKind, Operation, ResourceKind
from privacy_edge_sim.events import Event, priority_for
from privacy_edge_sim.resources import AdmissionRequest, ComputeJob
from privacy_edge_sim.simulator import DiscreteEventSimulator


def _snapshot_event(time_s: float, rsu_id: str, payload=None, seq: int = 9000):
    return Event(
        time_s,
        int(priority_for(EventKind.RSU_SNAPSHOT)),
        seq,
        EventKind.RSU_SNAPSHOT,
        object_id=rsu_id,
        payload=payload,
    )


def test_rsu_telemetry_is_delayed_frozen_and_causal(config, profile, trace):
    delayed = replace(
        config,
        rsu_telemetry_delay_s=0.1,
        rsu_telemetry_quantum_work_s=0.01,
        rsu_telemetry_drop_every=0,
    )
    simulator = DiscreteEventSimulator(delayed, profile, trace, "all_local")
    simulator.state.clock_s = 0.2
    runtime = simulator.state.rsus["rsu-1"]
    sample = _snapshot_event(0.2, "rsu-1")
    simulator._handle_rsu_snapshot(sample)
    delivery = next(
        event
        for event in simulator.state.events._heap
        if event.object_id == "rsu-1"
        and isinstance(event.payload, dict)
        and event.payload.get("phase") == "delivery"
        and event.payload.get("sample_time_s") == 0.2
    )
    runtime.failed = True  # Actual state changes after the frozen sample.
    simulator.state.clock_s = delivery.time_s
    simulator._handle_rsu_snapshot(delivery)
    assert runtime.current_snapshot_time_s == 0.2
    assert runtime.public_snapshot["failed"] is False
    assert runtime.failed is True


def test_rsu_telemetry_deterministic_drop_keeps_last_snapshot(config, profile, trace):
    dropping = replace(config, rsu_telemetry_delay_s=0.05, rsu_telemetry_drop_every=1)
    simulator = DiscreteEventSimulator(dropping, profile, trace, "all_local")
    simulator.state.clock_s = 0.2
    previous_time = simulator.state.rsus["rsu-1"].current_snapshot_time_s
    before = len(simulator.state.events)
    simulator._handle_rsu_snapshot(_snapshot_event(0.2, "rsu-1"))
    assert len(simulator.state.events) == before
    assert simulator.state.rsus["rsu-1"].current_snapshot_time_s == previous_time
    assert simulator.ledger.event_rows[-1]["event_kind"] == "RSU_TELEMETRY_DROP"


def test_rsu_snapshot_preserves_hold_count_and_residual_dynamic_energy(
    config, profile, trace
):
    simulator = DiscreteEventSimulator(config, profile, trace, "all_local")
    runtime = simulator.state.rsus["rsu-1"]
    model = profile.edge_models["edge_fer_full_v1"]
    accepted, reasons = runtime.admission.admit(
        AdmissionRequest(
            task_id="snapshot-private-task",
            descriptor_count=1,
            vram_bytes=1024,
            conservative_work_gpu_s=0.5,
            model_id=model.model_id,
            model_hash=model.model_hash,
            protocol_version=profile.protocol_version,
            message_valid=True,
        )
    )
    assert accepted and not reasons
    job = ComputeJob(
        job_id="snapshot-private-job",
        task_id="snapshot-private-task",
        owner_type="rsu",
        owner_id="rsu-1",
        operation=Operation.RSU_INGRESS,
        resource_kind=ResourceKind.RSU_INGRESS_CPU,
        model_or_pipeline_version=model.model_hash,
        enqueue_time_s=0.0,
        absolute_deadline_s=2.0,
        enqueue_seq=runtime.ingress.next_enqueue_seq(),
        total_work_s=2.0,
        residual_work_s=1.0,
        total_dynamic_energy_j=6.0,
    )
    runtime.ingress.enqueue(job)

    snapshot = simulator._rsu_public_snapshot(runtime, "rsu-1", 0.0)

    assert snapshot["hold_participant_count"] == 1
    assert snapshot["ingress_remaining_dynamic_energy_j"] == pytest.approx(3.0)
    assert snapshot["gpu_remaining_dynamic_energy_j"] == pytest.approx(0.0)
