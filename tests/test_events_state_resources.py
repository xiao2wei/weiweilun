from __future__ import annotations

import math

import pytest

from privacy_edge_sim.enums import (
    EventKind,
    FailureReason,
    JobStatus,
    Operation,
    ResourceKind,
    TaskState,
    TransferDirection,
)
from privacy_edge_sim.errors import InvariantViolation, TransitionError
from privacy_edge_sim.events import EventQueue
from privacy_edge_sim.invariants import assert_all_invariants
from privacy_edge_sim.packets import FERResult, RawImageHandle
from privacy_edge_sim.resources import (
    AdmissionRequest,
    ComputeJob,
    RSUAdmission,
    ResourcePool,
)
from privacy_edge_sim.state import TaskRecord, TaskStateMachine, Transfer


def _task(task_id: str = "task-1", deadline_s: float = 1.0) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        vehicle_id="veh-1",
        arrival_time_s=0.0,
        relative_deadline_s=deadline_s,
        absolute_deadline_s=deadline_s,
        raw_handle=RawImageHandle(f"raw-{task_id}"),
    )


def _to_local_run(task: TaskRecord) -> None:
    for target in (
        TaskState.PREP_WAIT,
        TaskState.PREP_RUN,
        TaskState.RAW,
        TaskState.LOCAL_WAIT,
        TaskState.LOCAL_RUN,
    ):
        TaskStateMachine.transition(
            task, target, time_s=0.0, trigger=f"TO_{target.value}"
        )


def test_taskless_model_maintenance_is_nonpreemptive_on_finite_gpu():
    pool = ResourcePool("rsu-1/gpu", ResourceKind.RSU_GPU, server_count=1)
    inference = ComputeJob(
        "inference",
        "task-1",
        "rsu",
        "rsu-1",
        Operation.EDGE_FER,
        ResourceKind.RSU_GPU,
        "model-old",
        0.0,
        1.0,
        pool.next_enqueue_seq(),
        0.10,
        0.10,
        1.0,
    )
    maintenance = ComputeJob(
        "maintenance",
        None,
        "rsu",
        "rsu-1",
        Operation.RSU_MODEL_MAINTENANCE,
        ResourceKind.RSU_GPU,
        "model-new",
        0.01,
        2.0,
        pool.next_enqueue_seq(),
        0.20,
        0.20,
        4.0,
    )

    pool.enqueue(inference)
    assert pool.dispatch(0.0) == [inference]
    pool.enqueue(maintenance)
    assert pool.dispatch(0.01) == []
    assert pool.running == ["inference"]

    pool.advance(0.10, effective_rate=1.0)
    assert pool.complete("inference", 0.10, inference.completion_version) is inference
    assert pool.dispatch(0.10) == [maintenance]
    assert maintenance.start_time_s == pytest.approx(0.10)
    assert maintenance.task_id is None


def test_resource_dispatch_preserves_blocked_edf_job_and_starts_eligible_job():
    pool = ResourcePool("vehicle/accelerator", ResourceKind.ACCELERATOR, 1)
    blocked = _job("blocked", deadline_s=1.0, enqueue_seq=0)
    eligible = _job("eligible", deadline_s=2.0, enqueue_seq=1)
    pool.enqueue(blocked)
    pool.enqueue(eligible)

    assert pool.dispatch(0.0, eligible=lambda job: job is eligible) == [eligible]
    assert blocked.status is JobStatus.WAITING
    assert pool.waiting_count == 1

    pool.advance(1.0, effective_rate=1.0)
    assert pool.complete("eligible", 1.0, eligible.completion_version) is eligible
    assert pool.dispatch(1.0) == [blocked]


def _job(
    job_id: str,
    *,
    task_id: str | None = None,
    deadline_s: float = 10.0,
    enqueue_seq: int = 0,
    work_s: float = 1.0,
) -> ComputeJob:
    return ComputeJob(
        job_id=job_id,
        task_id=task_id or job_id,
        owner_type="vehicle",
        owner_id="veh-1",
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version="model-v1",
        enqueue_time_s=0.0,
        absolute_deadline_s=deadline_s,
        enqueue_seq=enqueue_seq,
        total_work_s=work_s,
        residual_work_s=work_s,
        total_dynamic_energy_j=2.0,
    )


def _admission(*, descriptors=2, vram=200, workload=2.0) -> RSUAdmission:
    return RSUAdmission(
        descriptor_capacity=descriptors,
        vram_capacity_bytes=vram,
        workload_capacity_gpu_s=workload,
        protocol_version="protocol-v1",
        cached_models={"edge-v1": "hash-v1"},
    )


def _request(task_id="task-1", *, descriptors=1, vram=100, workload=1.0, **kwargs):
    values = dict(
        task_id=task_id,
        descriptor_count=descriptors,
        vram_bytes=vram,
        conservative_work_gpu_s=workload,
        model_id="edge-v1",
        model_hash="hash-v1",
        protocol_version="protocol-v1",
        message_valid=True,
    )
    values.update(kwargs)
    return AdmissionRequest(**values)


def test_compound_event_priority_completion_before_deadline_and_stable_ties():
    queue = EventQueue()
    queue.push(2.0, EventKind.ARRIVAL, task_id="arrival")
    first_completion = queue.push(
        2.0, EventKind.COMPUTE_COMPLETE, task_id="completion-1"
    )
    second_completion = queue.push(
        2.0, EventKind.TRANSFER_COMPLETE, task_id="completion-2"
    )
    queue.push(2.0, EventKind.THERMAL_CHANGE, task_id="thermal")
    queue.push(2.0, EventKind.DEADLINE, task_id="deadline")
    queue.push(2.0, EventKind.DISPATCH_DECISION, task_id="dispatch")

    time_s, events = queue.pop_compound(current_time_s=1.0)

    assert time_s == 2.0
    assert [event.task_id for event in events] == [
        "completion-1",
        "completion-2",
        "thermal",
        "deadline",
        "arrival",
        "dispatch",
    ]
    assert first_completion.seq < second_completion.seq


def test_event_queue_refuses_time_regression():
    queue = EventQueue()
    queue.push(1.0, EventKind.ARRIVAL)
    with pytest.raises(InvariantViolation) as caught:
        queue.pop_compound(current_time_s=1.1)
    assert caught.value.detail.code == "TIME_REGRESSION"


def test_compound_event_groups_ieee_roundoff_variants_of_same_time():
    queue = EventQueue()
    queue.push(0.3, EventKind.DEADLINE, task_id="task")
    queue.push(0.1 + 0.2, EventKind.COMPUTE_COMPLETE, task_id="task")

    time_s, events = queue.pop_compound(current_time_s=0.1)

    assert time_s == 0.1 + 0.2
    assert [event.kind for event in events] == [
        EventKind.COMPUTE_COMPLETE,
        EventKind.DEADLINE,
    ]


def test_compound_event_does_not_merge_physically_distinct_close_times():
    queue = EventQueue()
    queue.push(0.3, EventKind.DEADLINE, task_id="first")
    queue.push(0.3 + 1e-12, EventKind.COMPUTE_COMPLETE, task_id="second")

    time_s, events = queue.pop_compound(current_time_s=0.1)

    assert time_s == 0.3
    assert [event.task_id for event in events] == ["first"]
    assert len(queue) == 1


def test_event_queue_cancels_only_future_events_for_terminal_task():
    queue = EventQueue()
    task_deadline = queue.push(2.0, EventKind.DEADLINE, task_id="terminal-task")
    task_dispatch = queue.push(
        1.5, EventKind.DISPATCH_DECISION, task_id="terminal-task"
    )
    other = queue.push(1.0, EventKind.ARRIVAL, task_id="other-task")
    exogenous = queue.push(1.2, EventKind.LINK_CHANGE)

    assert queue.pending_for_task("terminal-task") == (
        task_dispatch,
        task_deadline,
    )
    assert queue.cancel_task("terminal-task") == (task_dispatch, task_deadline)
    assert queue.pending_for_task("terminal-task") == ()
    assert queue.snapshot() == tuple(
        sorted(
            (
                (
                    other.time_s,
                    other.priority,
                    other.seq,
                    other.kind.value,
                    "other-task",
                ),
                (
                    exogenous.time_s,
                    exogenous.priority,
                    exogenous.seq,
                    exogenous.kind.value,
                    None,
                ),
            )
        )
    )


def test_state_machine_accepts_valid_result_exactly_at_deadline():
    task = _task(deadline_s=1.0)
    _to_local_run(task)
    task.result_valid = True

    assert TaskStateMachine.transition(
        task, TaskState.DONE, time_s=1.0, trigger="LOCAL_RESULT"
    )
    assert task.state is TaskState.DONE
    assert task.terminal_time_s == 1.0
    assert task.failure_reason is FailureReason.NONE


def test_large_clock_nextafter_completion_is_same_deadline_instant(
    decision_fixture,
):
    deadline_s = 1_000_000_000.0
    fixture = decision_fixture(task_id="large-clock-deadline", deadline_s=deadline_s)
    task = fixture.task
    TaskStateMachine.transition(
        task, TaskState.LOCAL_WAIT, time_s=0.0, trigger="TO_LOCAL_WAIT"
    )
    TaskStateMachine.transition(
        task, TaskState.LOCAL_RUN, time_s=0.0, trigger="TO_LOCAL_RUN"
    )
    task.result_valid = True
    completion_s = math.nextafter(deadline_s, math.inf)

    assert completion_s - deadline_s > 1e-12
    assert TaskStateMachine.transition(
        task, TaskState.DONE, time_s=completion_s, trigger="LOCAL_RESULT"
    )
    fixture.simulator.state.clock_s = completion_s
    fixture.simulator._cleanup_terminal(task)
    assert_all_invariants(fixture.state, fixture.simulator.profile)


def test_large_clock_task_deadline_accepts_only_same_representable_sum():
    arrival_s = 1_000_000_000.0
    expected_s = arrival_s + 0.3

    task = TaskRecord(
        task_id="large-clock-construction",
        vehicle_id="veh-1",
        arrival_time_s=arrival_s,
        relative_deadline_s=0.3,
        absolute_deadline_s=math.nextafter(expected_s, math.inf),
        raw_handle=RawImageHandle("raw-large-clock-construction"),
    )

    assert task.absolute_deadline_s > expected_s
    with pytest.raises(ValueError):
        TaskRecord(
            task_id="large-clock-bad-construction",
            vehicle_id="veh-1",
            arrival_time_s=arrival_s,
            relative_deadline_s=0.3,
            absolute_deadline_s=expected_s + 1e-4,
            raw_handle=RawImageHandle("raw-large-clock-bad-construction"),
        )
    with pytest.raises(ValueError):
        TaskRecord(
            task_id="infinite-deadline",
            vehicle_id="veh-1",
            arrival_time_s=0.0,
            relative_deadline_s=1.0,
            absolute_deadline_s=math.inf,
            raw_handle=RawImageHandle("raw-infinite-deadline"),
        )


def test_large_clock_physically_late_completion_is_rejected():
    deadline_s = 1_000_000_000.0
    task = _task(task_id="large-clock-late", deadline_s=deadline_s)
    _to_local_run(task)
    task.result_valid = True

    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(
            task,
            TaskState.DONE,
            time_s=deadline_s + 1e-4,
            trigger="LATE_LOCAL_RESULT",
        )

    assert caught.value.detail.code == "DONE_RESULT_INVALID"


def test_des_completion_roundoff_variant_still_beats_exact_deadline(
    decision_fixture,
):
    fixture = decision_fixture(task_id="ulp-deadline", deadline_s=0.3)
    simulator = fixture.simulator
    task = fixture.task
    simulator.state.clock_s = 0.1
    simulator.state.events = EventQueue()
    TaskStateMachine.transition(
        task, TaskState.LOCAL_WAIT, time_s=0.1, trigger="TEST_LOCAL_WAIT"
    )
    TaskStateMachine.transition(
        task, TaskState.LOCAL_RUN, time_s=0.1, trigger="TEST_LOCAL_RUN"
    )
    row = next(
        row
        for row in simulator.trace.local_rows
        if not row.failed and row.fer_loss is not None
    )
    simulator._local_rows[task.task_id] = row
    task.selected_local_model = row.model_id
    pool = simulator.state.vehicles[task.vehicle_id].resources["accelerator"]
    job = ComputeJob(
        job_id="ulp-completion-job",
        task_id=task.task_id,
        owner_type="vehicle",
        owner_id=task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version=row.model_hash,
        enqueue_time_s=0.1,
        absolute_deadline_s=0.3,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=0.2,
        residual_work_s=0.2,
        total_dynamic_energy_j=1.0,
    )
    pool.enqueue(job)
    pool.dispatch(0.1)
    task.current_job_id = job.job_id
    simulator.state.events.push(0.3, EventKind.DEADLINE, task_id=task.task_id)
    simulator._schedule_completions_and_battery_guards()

    time_s, batch = simulator.state.events.pop_compound(current_time_s=0.1)
    assert [event.kind for event in batch[:2]] == [
        EventKind.COMPUTE_COMPLETE,
        EventKind.DEADLINE,
    ]
    simulator._advance_to(time_s)
    _, timeouts = simulator._process_compound(time_s, batch)

    assert timeouts == 0
    assert task.state is TaskState.DONE
    assert task.terminal_time_s == pytest.approx(0.3)


def test_stale_job_completion_cannot_suppress_current_zero_residual_completion(
    decision_fixture,
):
    fixture = decision_fixture(task_id="stale-current-job")
    simulator = fixture.simulator
    task = fixture.task
    row = next(
        row
        for row in simulator.trace.local_rows
        if not row.failed and row.fer_loss is not None
    )
    simulator._local_rows[task.task_id] = row
    task.selected_local_model = row.model_id
    TaskStateMachine.transition(
        task, TaskState.LOCAL_WAIT, time_s=0.0, trigger="TEST_LOCAL_WAIT"
    )
    TaskStateMachine.transition(
        task, TaskState.LOCAL_RUN, time_s=0.0, trigger="TEST_LOCAL_RUN"
    )
    pool = simulator.state.vehicles[task.vehicle_id].resources["accelerator"]
    job = ComputeJob(
        job_id="stale-current-job-object",
        task_id=task.task_id,
        owner_type="vehicle",
        owner_id=task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version=row.model_hash,
        enqueue_time_s=0.0,
        absolute_deadline_s=task.absolute_deadline_s,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=1.0,
        residual_work_s=1.0,
        total_dynamic_energy_j=1.0,
    )
    pool.enqueue(job)
    pool.dispatch(0.0)
    task.current_job_id = job.job_id
    stale_token = job.completion_version
    stale = EventQueue().push(
        0.0,
        EventKind.COMPUTE_COMPLETE,
        task_id=task.task_id,
        object_id=job.job_id,
        version_token=stale_token,
    )
    job.residual_work_s = 0.0
    job.completion_version += 1

    materialized = simulator._materialized_completions([stale])

    assert len(materialized) == 1
    assert materialized[0].version_token == job.completion_version
    simulator._handle_compute_completion(materialized[0])
    assert job.status is JobStatus.DONE
    assert task.state is TaskState.DONE
    simulator._schedule_completions_and_battery_guards()


def test_stale_transfer_completion_cannot_suppress_current_zero_bits_completion(
    decision_fixture,
):
    fixture = decision_fixture(task_id="stale-current-transfer")
    simulator = fixture.simulator
    task = fixture.task
    for target in (
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.READY,
        TaskState.UL,
        TaskState.EDGE_WAIT,
        TaskState.EDGE_RUN,
        TaskState.DL,
    ):
        TaskStateMachine.transition(
            task, target, time_s=0.0, trigger=f"TEST_{target.value}"
        )
    rsu_id = min(simulator.state.rsus)
    model = simulator.profile.edge_models[min(simulator.profile.edge_models)]
    result = FERResult(
        task_id=task.task_id,
        model_id=model.model_id,
        model_hash=model.model_hash,
        protocol_version=simulator.config.protocol_version,
        result_code=0,
        valid=True,
        size_bits=128,
    )
    transfer = Transfer(
        transfer_id="stale-current-transfer-object",
        task_id=task.task_id,
        vehicle_id=task.vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.DL,
        packet=result,
        total_bits=128.0,
        remaining_bits=128.0,
        start_time_s=0.0,
        last_update_time_s=0.0,
    )
    simulator.state.transfers[transfer.transfer_id] = transfer
    task.selected_rsu = rsu_id
    task.current_transfer_id = transfer.transfer_id
    task.dl_remaining_bits = transfer.remaining_bits
    stale_token = transfer.completion_version
    stale = EventQueue().push(
        0.0,
        EventKind.TRANSFER_COMPLETE,
        task_id=task.task_id,
        object_id=transfer.transfer_id,
        version_token=stale_token,
    )
    transfer.remaining_bits = 0.0
    transfer.delivered_bits = transfer.total_bits
    transfer.completion_version += 1

    materialized = simulator._materialized_completions([stale])

    assert len(materialized) == 1
    assert materialized[0].version_token == transfer.completion_version
    simulator._handle_transfer_completion(materialized[0])
    assert task.state is TaskState.DONE
    assert transfer.transfer_id not in simulator.state.transfers
    simulator._schedule_completions_and_battery_guards()


def test_state_machine_rejects_late_invalid_or_illegal_completion():
    invalid = _task("invalid")
    _to_local_run(invalid)
    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(
            invalid, TaskState.DONE, time_s=0.5, trigger="INVALID"
        )
    assert caught.value.detail.code == "DONE_RESULT_INVALID"

    late = _task("late")
    _to_local_run(late)
    late.result_valid = True
    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(late, TaskState.DONE, time_s=1.01, trigger="LATE")
    assert caught.value.detail.code == "DONE_RESULT_INVALID"

    illegal = _task("illegal")
    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(
            illegal, TaskState.DONE, time_s=0.0, trigger="BYPASS"
        )
    assert caught.value.detail.code == "ILLEGAL_TASK_TRANSITION"


@pytest.mark.parametrize("terminal", [TaskState.DONE, TaskState.FAIL])
def test_terminal_states_are_absorbing(terminal):
    task = _task()
    if terminal is TaskState.DONE:
        _to_local_run(task)
        task.result_valid = True
        TaskStateMachine.transition(task, terminal, time_s=0.5, trigger="DONE")
    else:
        TaskStateMachine.transition(
            task,
            terminal,
            time_s=0.5,
            trigger="FAIL",
            failure_reason=FailureReason.PREP_FAILED,
        )

    assert (
        TaskStateMachine.transition(task, terminal, time_s=0.6, trigger="IDEMPOTENT")
        is False
    )
    other = TaskState.FAIL if terminal is TaskState.DONE else TaskState.DONE
    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(task, other, time_s=0.6, trigger="ESCAPE")
    assert caught.value.detail.code == "TERMINAL_STATE_ABSORBING"


def test_resource_pool_uses_edf_for_waiting_jobs_and_nonpreemption():
    pool = ResourcePool("veh-accel", ResourceKind.ACCELERATOR, server_count=1)
    running = _job("running", deadline_s=10.0, enqueue_seq=0, work_s=2.0)
    pool.enqueue(running)
    assert [job.job_id for job in pool.dispatch(0.0)] == ["running"]

    urgent = _job("urgent", deadline_s=1.0, enqueue_seq=1)
    less_urgent = _job("less-urgent", deadline_s=2.0, enqueue_seq=2)
    pool.enqueue(less_urgent)
    pool.enqueue(urgent)
    assert pool.dispatch(0.1) == []
    assert running.status is JobStatus.RUNNING

    pool.advance(2.0, effective_rate=1.0)
    assert pool.complete("running", 2.0, running.completion_version) is running
    assert [job.job_id for job in pool.dispatch(2.0)] == ["urgent"]


def test_resource_pool_enforces_finite_concurrency_and_stable_edf_ties():
    pool = ResourcePool("veh-accel", ResourceKind.ACCELERATOR, server_count=2)
    jobs = [
        _job("tie-second", deadline_s=1.0, enqueue_seq=2),
        _job("tie-first", deadline_s=1.0, enqueue_seq=1),
        _job("later", deadline_s=2.0, enqueue_seq=0),
    ]
    for job in jobs:
        pool.enqueue(job)

    assert [job.job_id for job in pool.dispatch(0.0)] == ["tie-first", "tie-second"]
    assert pool.running_count == 2
    assert pool.waiting_count == 1
    assert pool.max_running_observed == 2


def test_stale_completion_cannot_finish_cancelled_or_reversioned_job():
    pool = ResourcePool("veh-accel", ResourceKind.ACCELERATOR, server_count=1)
    job = _job("job", task_id="task")
    pool.enqueue(job)
    pool.dispatch(0.0)
    stale_version = job.completion_version
    pool.cancel_task("task", 0.2)

    assert job.status is JobStatus.CANCELLED
    assert job.completion_version > stale_version
    assert pool.complete(job.job_id, 1.0, stale_version) is None
    assert pool.running_count == 0


def test_vehicle_reservation_is_capacity_checked_and_released(decision_fixture):
    fixture = decision_fixture()
    vehicle = fixture.state.vehicles[fixture.task.vehicle_id]
    task = fixture.task
    before_memory = vehicle.memory_reserved_bytes
    capacity = vehicle.descriptor_capacity["accelerator"]

    assert vehicle.reserve(task, {"accelerator": capacity}, 1)
    assert vehicle.memory_reserved_bytes == before_memory + 1
    another = _task("another")
    assert not vehicle.reserve(another, {"accelerator": 1}, 0)

    vehicle.release(task)
    assert vehicle.memory_reserved_bytes == before_memory
    assert vehicle.descriptors_reserved["accelerator"] == 0


def test_vehicle_reservation_reconcile_is_atomic_and_releases_abandoned_tokens(
    decision_fixture,
):
    fixture = decision_fixture(task_id="reservation-reconcile")
    vehicle = fixture.state.vehicles[fixture.task.vehicle_id]
    task = fixture.task
    assert vehicle.reserve(task, {"accelerator": 1, "cpu": 1, "encoder": 1}, 128)

    assert vehicle.reconcile_reservation(task, {"accelerator": 1}, 64)
    assert task.reservation_tokens == {"accelerator": 1}
    assert task.memory_reservation_bytes == 64
    assert vehicle.descriptors_reserved == {
        "accelerator": 1,
        "cpu": 0,
        "encoder": 0,
    }
    before = (
        dict(task.reservation_tokens),
        task.memory_reservation_bytes,
        dict(vehicle.descriptors_reserved),
        vehicle.memory_reserved_bytes,
    )
    assert not vehicle.reconcile_reservation(
        task,
        {"accelerator": vehicle.descriptor_capacity["accelerator"] + 1},
        64,
    )
    assert before == (
        task.reservation_tokens,
        task.memory_reservation_bytes,
        vehicle.descriptors_reserved,
        vehicle.memory_reserved_bytes,
    )


def test_rsu_admission_is_atomic_and_accepts_exact_capacity():
    admission = _admission()
    exact = _request(descriptors=2, vram=200, workload=2.0)

    accepted, reasons = admission.admit(exact)
    assert accepted and reasons == ()
    assert admission.descriptors == 2
    assert admission.vram_bytes == 200
    assert admission.reserved_work_gpu_s == 2.0
    assert admission.pinned_model(exact.task_id) == ("edge-v1", "hash-v1")

    before = admission.snapshot()
    rejected, reasons = admission.admit(_request("task-2"))
    assert not rejected
    assert {
        "RSU_DESCRIPTOR_CAPACITY",
        "RSU_VRAM_CAPACITY",
        "RSU_WORKLOAD_CAPACITY",
    }.issubset(reasons)
    assert admission.snapshot() == before


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"message_valid": False}, "MESSAGE_EVIDENCE_MISSING"),
        ({"protocol_version": "wrong"}, "PROTOCOL_MISMATCH"),
        ({"model_hash": "wrong"}, "MODEL_CACHE_MISSING"),
    ],
)
def test_rsu_rejection_has_zero_side_effects(override, reason):
    admission = _admission()
    before = admission.snapshot()
    accepted, reasons = admission.admit(_request(**override))
    assert not accepted and reason in reasons
    assert admission.snapshot() == before


@pytest.mark.parametrize(
    "override",
    [
        {"descriptors": 0},
        {"descriptors": -1},
        {"vram": 0},
        {"workload": 0.0},
        {"workload": float("nan")},
        {"workload": float("inf")},
    ],
)
def test_malformed_admission_request_is_rejected_before_any_mutation(override):
    admission = _admission()
    before = admission.snapshot()

    with pytest.raises(ValueError):
        _request(**override)

    assert admission.snapshot() == before


def test_tightening_admission_capacity_cannot_turn_reject_into_accept():
    request = _request(vram=101)
    loose = _admission(vram=100)
    tight = _admission(vram=50)

    loose_result = loose.admit(request)
    tight_result = tight.admit(request)
    assert loose_result[0] is False
    assert tight_result[0] is False
    assert loose.snapshot().reservations == tight.snapshot().reservations == ()


def test_admitted_task_keeps_pinned_model_across_cache_version_change():
    admission = _admission()
    accepted, _ = admission.admit(_request("pinned"))
    assert accepted

    admission.update_cache({"edge-v1": "hash-v2"})

    assert admission.pinned_model("pinned") == ("edge-v1", "hash-v1")
    old_request_accepted, old_reasons = admission.admit(_request("new-old-hash"))
    assert not old_request_accepted
    assert "MODEL_CACHE_MISSING" in old_reasons
    new_request_accepted, _ = admission.admit(
        _request("new-version", model_hash="hash-v2")
    )
    assert new_request_accepted
