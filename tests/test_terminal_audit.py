from __future__ import annotations

from types import SimpleNamespace

import pytest

from privacy_edge_sim.enums import (
    ActionStage,
    EventKind,
    FailureReason,
    Operation,
    ResourceKind,
    TaskState,
    TransferDirection,
    TransferStatus,
)
from privacy_edge_sim.invariants import assert_all_invariants
from privacy_edge_sim.packets import (
    AnonFERRequest,
    FERResult,
    _finalize_encoded_anon,
    _replay_anonymization_success,
    _replay_encoding_success,
    _replay_guard_success,
)
from privacy_edge_sim.resources import ComputeJob
from privacy_edge_sim.safety import ActionKind
from privacy_edge_sim.state import TaskStateMachine, Transfer
from privacy_edge_sim.traces import ExogenousEvent


def _move_to_ready(task) -> None:
    for state in (
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.READY,
    ):
        TaskStateMachine.transition(task, state, time_s=0.0, trigger="TEST_PATH")


def _uplink_request(task, profile) -> AnonFERRequest:
    pipeline = profile.pipelines["pixelate_strong_v1"]
    model = profile.edge_models["edge_fer_full_v1"]
    anonymized = _replay_anonymization_success(
        aligned=task.aligned_handle,
        task_id=task.task_id,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        artifact_key="terminal-audit-artifact",
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash=pipeline.guard_hash,
        guard_certificate_id="terminal-audit-guard",
    )
    encoding = _replay_encoding_success(
        guarded,
        payload=b"anonymous-payload",
        encoder_hash=pipeline.encoder_hash,
        encoded_size_bytes=len(b"anonymous-payload"),
    )
    encoded = _finalize_encoded_anon(
        encoding,
        profile_hash=profile.profile_hash,
        quality_bins=task.conformal_quality_bins,
    )
    return AnonFERRequest.from_encoded(
        encoded,
        protocol_version=profile.protocol_version,
        requested_edge_model=model.model_id,
        requested_edge_model_hash=model.model_hash,
        vehicle_id=task.vehicle_id,
        task_id=task.task_id,
    )


def _install_partial_transfer(fixture, direction: TransferDirection) -> Transfer:
    simulator = fixture.simulator
    task = fixture.task
    profile = simulator.profile
    _move_to_ready(task)
    task.selected_pipeline = "pixelate_strong_v1"
    task.selected_rsu = "rsu-1"
    task.selected_edge_model = "edge_fer_full_v1"
    if direction is TransferDirection.UL:
        packet = _uplink_request(task, profile)
        TaskStateMachine.transition(task, TaskState.UL, time_s=0.0, trigger="TEST_UL")
    else:
        TaskStateMachine.transition(task, TaskState.UL, time_s=0.0, trigger="TEST_UL")
        TaskStateMachine.transition(
            task, TaskState.EDGE_WAIT, time_s=0.0, trigger="TEST_ADMIT"
        )
        TaskStateMachine.transition(
            task, TaskState.EDGE_RUN, time_s=0.0, trigger="TEST_EDGE"
        )
        TaskStateMachine.transition(task, TaskState.DL, time_s=0.0, trigger="TEST_DL")
        model = profile.edge_models["edge_fer_full_v1"]
        packet = FERResult(
            task.task_id,
            model.model_id,
            model.model_hash,
            profile.protocol_version,
            0,
            True,
            1_000,
        )
    transfer = Transfer(
        transfer_id=f"partial-{direction.value.lower()}",
        task_id=task.task_id,
        vehicle_id=task.vehicle_id,
        rsu_id="rsu-1",
        direction=direction,
        packet=packet,
        total_bits=1_000.0,
        remaining_bits=600.0,
        start_time_s=0.0,
        last_update_time_s=0.5,
        delivered_bits=400.0,
        vehicle_energy_j=0.3,
        rsu_energy_j=0.1,
    )
    simulator.state.transfers[transfer.transfer_id] = transfer
    task.current_transfer_id = transfer.transfer_id
    task.ul_remaining_bits = 600.0 if direction is TransferDirection.UL else 0.0
    task.dl_remaining_bits = 600.0 if direction is TransferDirection.DL else 0.0
    simulator.state.clock_s = 0.5
    return transfer


@pytest.mark.parametrize("direction", [TransferDirection.UL, TransferDirection.DL])
def test_observation_exposes_only_anonymous_active_transfer_anchor(
    decision_fixture, direction
):
    fixture = decision_fixture(task_id=f"anchor-{direction.value.lower()}")
    transfer = _install_partial_transfer(fixture, direction)

    observation = fixture.simulator._observation(fixture.task, stage=ActionStage.READY)
    anchors = observation.links["rsu-1"]["active_transfers"]

    assert len(anchors) == 1
    assert anchors[0]["direction"] == direction.value
    assert anchors[0]["total_bits"] == pytest.approx(transfer.total_bits)
    assert anchors[0]["remaining_bits"] == pytest.approx(transfer.remaining_bits)
    assert anchors[0]["status"] == transfer.status.value
    focal = next(
        item for item in observation.vehicle["active_tasks"] if item["is_focal"]
    )
    assert anchors[0]["task_token"] == focal["task_token"]
    assert "task_id" not in anchors[0]
    assert "transfer_id" not in anchors[0]
    assert "artifact_key" not in anchors[0]


def test_observation_preserves_anonymous_residual_job_energy(decision_fixture):
    fixture = decision_fixture(task_id="job-anchor-focal")
    pool = fixture.state.vehicles[fixture.task.vehicle_id].resources["accelerator"]
    job = ComputeJob(
        job_id="simulator-private-job",
        task_id=fixture.task.task_id,
        owner_type="vehicle",
        owner_id=fixture.task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version="private-version",
        enqueue_time_s=0.0,
        absolute_deadline_s=5.0,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=2.0,
        residual_work_s=1.0,
        total_dynamic_energy_j=4.0,
    )
    pool.enqueue(job)
    pool.dispatch(0.0)

    observation = fixture.simulator._observation(fixture.task)
    row = observation.vehicle["resources"]["accelerator"]
    anchors = row["active_jobs"]

    assert row["remaining_dynamic_energy_j"] == pytest.approx(2.0)
    assert len(anchors) == 1
    assert anchors[0]["status"] == "RUNNING"
    assert anchors[0]["residual_work_s"] == pytest.approx(1.0)
    assert anchors[0]["remaining_nominal_dynamic_energy_j"] == pytest.approx(2.0)
    focal = next(
        item for item in observation.vehicle["active_tasks"] if item["is_focal"]
    )
    assert anchors[0]["task_token"] == focal["task_token"]
    assert "job_id" not in anchors[0]
    assert "task_id" not in anchors[0]
    assert "model_or_pipeline_version" not in anchors[0]


def _start_partial_anon(fixture):
    simulator = fixture.simulator
    task = fixture.task
    mask = simulator.mask_engine.enumerate(task, fixture.observation, simulator.state)
    pipeline = next(
        action.pipeline_id for action in mask.allowed if action.kind is ActionKind.PIPE
    )
    simulator._start_pipeline_action(task, pipeline)
    simulator._dispatch_all()
    found = simulator._find_job_pool(task.current_job_id or "")
    assert found is not None
    job = found[-1].jobs[task.current_job_id]
    simulator._advance_to(job.total_work_s / 2.0)
    return job


def test_deadline_during_anon_records_actual_partial_compute_and_attempt(
    decision_fixture,
):
    fixture = decision_fixture(task_id="deadline-partial-anon")
    job = _start_partial_anon(fixture)

    fixture.simulator._terminate_fail(fixture.task, FailureReason.TIMEOUT, "DEADLINE")

    assert fixture.task.state is TaskState.FAIL
    assert (
        len(fixture.task.anon_attempt_audit) == fixture.task.attempt_started_count == 1
    )
    attempt = fixture.task.anon_attempt_audit[0]
    assert attempt["failure_reason"] == "TIMEOUT"
    assert attempt["interrupted_phase"] == "ANON"
    assert 0 < attempt["executed_work_s"] < job.total_work_s
    assert attempt["vehicle_energy_j"] == pytest.approx(job.consumed_dynamic_energy_j)
    compute = fixture.task.compute_audit[-1]
    assert compute["reason"] == "TIMEOUT"
    assert compute["executed_work_s"] == pytest.approx(
        job.total_work_s - job.residual_work_s
    )
    assert_all_invariants(fixture.state, fixture.simulator.profile)


def test_terminal_cleanup_removes_all_future_task_events(decision_fixture):
    fixture = decision_fixture(task_id="terminal-future-events")
    events = fixture.state.events
    events.push(0.75, EventKind.DISPATCH_DECISION, task_id=fixture.task.task_id)
    events.push(1.25, EventKind.DEADLINE, task_id=fixture.task.task_id)
    unrelated = events.push(0.5, EventKind.ARRIVAL, task_id="unrelated-task")

    assert len(events.pending_for_task(fixture.task.task_id)) >= 2

    fixture.simulator._terminate_fail(
        fixture.task, FailureReason.POLICY_EXPLICIT_FAIL, "TEST_TERMINAL"
    )

    assert events.pending_for_task(fixture.task.task_id) == ()
    assert unrelated in events.pending_for_task("unrelated-task")
    assert_all_invariants(fixture.state, fixture.simulator.profile)


@pytest.mark.parametrize("direction", [TransferDirection.UL, TransferDirection.DL])
def test_deadline_during_transfer_records_partial_bits_and_paired_energy(
    decision_fixture, direction
):
    fixture = decision_fixture(task_id=f"deadline-partial-{direction.value.lower()}")
    transfer = _install_partial_transfer(fixture, direction)

    fixture.simulator._terminate_fail(fixture.task, FailureReason.TIMEOUT, "DEADLINE")

    terminal = fixture.task.network_audit[-1]
    assert terminal["direction"] == direction.value
    assert terminal["status"] == "FAIL"
    assert terminal["reason"] == "TIMEOUT"
    assert terminal["delivered_bits"] == 400.0
    assert terminal["remaining_bits"] == 600.0
    assert terminal["vehicle_energy_j"] == 0.3
    assert terminal["rsu_energy_j"] == 0.1
    assert transfer.status is TransferStatus.CANCELLED
    assert transfer.transfer_id not in fixture.state.transfers
    assert_all_invariants(fixture.state, fixture.simulator.profile)


def test_vehicle_fault_and_battery_guard_use_the_same_closed_audit_path(
    decision_fixture,
):
    fault_fixture = decision_fixture(task_id="device-fault-partial-anon")
    _start_partial_anon(fault_fixture)
    fault_fixture.simulator._handle_device_fault(
        SimpleNamespace(
            payload=ExogenousEvent(
                "vehicle-fault-audit",
                fault_fixture.state.clock_s,
                "DEVICE_FAULT_PERMANENT",
                "vehicle",
                fault_fixture.task.vehicle_id,
                "all",
                None,
                None,
                True,
                {},
            )
        )
    )
    assert fault_fixture.task.failure_reason is FailureReason.DEVICE_FAULT
    assert fault_fixture.task.compute_audit[-1]["reason"] == "DEVICE_FAULT"
    assert fault_fixture.task.anon_attempt_audit[-1]["failure_reason"] == "DEVICE_FAULT"

    battery_fixture = decision_fixture(task_id="battery-partial-ul")
    transfer = _install_partial_transfer(battery_fixture, TransferDirection.UL)
    battery_fixture.simulator._battery_versions[battery_fixture.task.vehicle_id] = 7
    battery_fixture.simulator._handle_battery_guard(
        SimpleNamespace(
            object_id=battery_fixture.task.vehicle_id,
            version_token=7,
        )
    )
    assert battery_fixture.task.failure_reason is FailureReason.BATTERY_GUARD
    assert battery_fixture.task.network_audit[-1]["reason"] == "BATTERY_GUARD"
    assert transfer.status is TransferStatus.CANCELLED
    assert transfer.transfer_id not in battery_fixture.state.transfers
    assert (
        battery_fixture.state.vehicles[battery_fixture.task.vehicle_id].battery_j == 0
    )
    assert_all_invariants(battery_fixture.state, battery_fixture.simulator.profile)


def test_rsu_fault_records_partial_ingress_or_gpu_before_releasing_resources(
    decision_fixture,
):
    fixture = decision_fixture(task_id="rsu-partial-compute")
    task = fixture.task
    simulator = fixture.simulator
    _move_to_ready(task)
    task.selected_pipeline = "pixelate_strong_v1"
    task.selected_rsu = "rsu-1"
    task.selected_edge_model = "edge_fer_full_v1"
    TaskStateMachine.transition(task, TaskState.UL, time_s=0.0, trigger="TEST_UL")
    TaskStateMachine.transition(
        task, TaskState.EDGE_WAIT, time_s=0.0, trigger="TEST_ADMIT"
    )
    simulator._enqueue_job(
        task,
        operation=Operation.RSU_INGRESS,
        resource_kind=ResourceKind.RSU_INGRESS_CPU,
        owner_type="rsu",
        owner_id="rsu-1",
        work_s=0.2,
        energy_j=1.0,
        memory_bytes=0,
        version=simulator.config.protocol_version,
    )
    simulator._dispatch_all()
    simulator._advance_to(0.1)

    simulator._handle_device_fault(
        SimpleNamespace(
            payload=ExogenousEvent(
                "rsu-fault-audit",
                simulator.state.clock_s,
                "DEVICE_FAULT_PERMANENT",
                "rsu",
                "rsu-1",
                "all",
                None,
                None,
                True,
                {},
            )
        )
    )

    cancelled = [row for row in task.rsu_audit if row.get("status") == "CANCELLED"]
    assert cancelled
    assert cancelled[-1]["phase"] == "ingress_cancel"
    assert cancelled[-1]["reason"] == "EDGE_FAILED"
    assert cancelled[-1]["executed_work_s"] == pytest.approx(0.1)
    assert cancelled[-1]["dynamic_energy_j"] == pytest.approx(0.5)
    assert simulator.state.rsus["rsu-1"].ingress.running_count == 0


@pytest.mark.parametrize("local_state", [TaskState.LOCAL_WAIT, TaskState.LOCAL_RUN])
def test_failed_local_path_never_recursively_requeues_fallback(
    decision_fixture, local_state
):
    fixture = decision_fixture(task_id=f"local-fallback-exhausted-{local_state.value}")
    fixture.task.selected_pipeline = "pixelate_strong_v1"
    TaskStateMachine.transition(
        fixture.task, TaskState.LOCAL_WAIT, time_s=0.0, trigger="TEST_LOCAL"
    )
    if local_state is TaskState.LOCAL_RUN:
        TaskStateMachine.transition(
            fixture.task, TaskState.LOCAL_RUN, time_s=0.0, trigger="TEST_LOCAL_RUN"
        )

    fixture.simulator._fallback_or_fail(fixture.task, FailureReason.BATTERY_GUARD)

    assert fixture.task.state is TaskState.FAIL
    assert fixture.task.failure_reason is FailureReason.BATTERY_GUARD
    assert fixture.task.phase_history[-1].trigger == "LOCAL_FALLBACK_EXHAUSTED"
    assert fixture.task.current_job_id is None
