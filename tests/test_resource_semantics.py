from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from privacy_edge_sim.enums import (
    ActionKind,
    Operation,
    ResourceKind,
    TaskState,
)
from privacy_edge_sim.errors import InvariantViolation
from privacy_edge_sim.invariants import assert_all_invariants
from privacy_edge_sim.packets import AlignedTensorHandle, RawImageHandle
from privacy_edge_sim.resources import ComputeJob
from privacy_edge_sim.safety import Action
from privacy_edge_sim.simulator import DiscreteEventSimulator
from privacy_edge_sim.state import TaskRecord, TaskStateMachine


def _single_arrival_trace(trace):
    return replace(trace, arrivals=(trace.arrivals[0],))


def _advance_until_raw_without_constructing_a_decision(simulator, task_id: str):
    for _ in range(100):
        time_s, batch = simulator.state.events.pop_compound(
            current_time_s=simulator.state.clock_s
        )
        simulator._advance_to(time_s)
        simulator._process_compound(time_s, batch)
        simulator._dispatch_all()
        task = simulator.state.tasks.get(task_id)
        if task is not None and task.state is TaskState.RAW:
            return task
        simulator._schedule_completions_and_battery_guards()
    raise AssertionError("preprocessing did not reach RAW")


def _new_raw_task(task_id: str, template: TaskRecord) -> TaskRecord:
    task = TaskRecord(
        task_id=task_id,
        vehicle_id=template.vehicle_id,
        arrival_time_s=template.arrival_time_s,
        relative_deadline_s=template.relative_deadline_s,
        absolute_deadline_s=template.absolute_deadline_s,
        raw_handle=RawImageHandle(f"raw-{task_id}"),
        aligned_handle=AlignedTensorHandle(f"aligned-{task_id}"),
        quality_features=template.quality_features,
        quality_probabilities=template.quality_probabilities,
        conformal_quality_bins=template.conformal_quality_bins,
        ood=template.ood,
        true_quality_region=template.true_quality_region,
    )
    for target in (TaskState.PREP_WAIT, TaskState.PREP_RUN, TaskState.RAW):
        TaskStateMachine.transition(
            task, target, time_s=task.arrival_time_s, trigger=f"TEST_{target.value}"
        )
    return task


def _complete_current_vehicle_job(simulator, task: TaskRecord) -> None:
    simulator._dispatch_all()
    found = simulator._find_job_pool(task.current_job_id or "")
    assert found is not None
    owner_type, owner_id, resource_name, pool = found
    job = pool.jobs[task.current_job_id or ""]
    rate = simulator._resource_rate(
        owner_type, owner_id, resource_name, simulator.state.clock_s
    )
    assert rate > 0
    finish_s = simulator.state.clock_s + job.residual_work_s / rate
    version = job.completion_version
    simulator._advance_to(finish_s)
    simulator._handle_compute_completion(
        SimpleNamespace(object_id=job.job_id, version_token=version)
    )


def test_prep_completion_retains_trusted_buffer_memory_without_compute_tokens(
    config, profile, trace
):
    arrival = trace.arrivals[0]
    simulator = DiscreteEventSimulator(
        config, profile, _single_arrival_trace(trace), "all_local"
    )

    task = _advance_until_raw_without_constructing_a_decision(
        simulator, arrival.task_id
    )
    vehicle = simulator.state.vehicles[task.vehicle_id]
    expected_memory = int(profile.preprocessing_resource_bounds["max_memory_bytes"])

    assert task.raw_handle is not None
    assert task.aligned_handle is not None
    assert task.memory_reservation_bytes == expected_memory
    assert vehicle.memory_reserved_bytes == expected_memory
    assert task.reservation_tokens == {}
    assert all(value == 0 for value in vehicle.descriptors_reserved.values())
    assert task.current_job_id is None
    assert_all_invariants(simulator.state, profile)


@pytest.mark.parametrize("kind", [ActionKind.LOCAL, ActionKind.PIPE])
def test_focal_prep_buffer_is_reusable_at_exact_action_envelope(decision_fixture, kind):
    fixture = decision_fixture(task_id=f"focal-reuse-{kind.value.lower()}")
    simulator = fixture.simulator
    task = fixture.task
    vehicle = fixture.state.vehicles[task.vehicle_id]
    baseline = simulator.mask_engine.enumerate(
        task, simulator._observation(task), simulator.state
    )
    action = next(item for item in baseline.allowed if item.kind is kind)
    shadow = simulator._planned_vehicle_shadow(task, action)
    assert shadow is not None
    tokens, target_memory = shadow
    prep_memory = int(
        simulator.profile.preprocessing_resource_bounds["max_memory_bytes"]
    )
    assert prep_memory <= target_memory
    assert vehicle.reserve(task, {}, prep_memory)
    vehicle.memory_capacity_bytes = target_memory
    for resource, count in tokens.items():
        vehicle.descriptor_capacity[resource] = count

    observation = simulator._observation(task)
    mask = simulator.mask_engine.enumerate(
        task,
        observation,
        simulator.state,
        candidates=(action, Action.fail(observation.stage)),
    )

    assert action in mask.allowed
    assert vehicle.reconcile_reservation(task, tokens, target_memory)
    assert task.reservation_tokens == tokens
    assert task.memory_reservation_bytes == target_memory
    assert vehicle.memory_reserved_bytes == target_memory
    assert_all_invariants(simulator.state, simulator.profile)


def test_ready_releases_pipeline_temporaries_and_retains_only_fallback_shadow(
    decision_fixture, trace
):
    fixture = decision_fixture(task_id="ready-shadow-release", deadline_s=10.0)
    simulator = fixture.simulator
    task = fixture.task
    vehicle = fixture.state.vehicles[task.vehicle_id]
    row = next(
        item
        for item in trace.anon_rows
        if item.formed_packet
        and item.quality_bin == task.true_quality_region
        and item.device_type == vehicle.device_type
        and simulator.profile.pipelines[item.pipeline_id].fallback_local_model
    )
    pipeline = simulator.profile.pipelines[row.pipeline_id]
    fallback_id = pipeline.fallback_local_model
    assert fallback_id is not None
    pipeline_memory = max(
        int(pipeline.deployment_resource_bounds["max_peak_memory_bytes"]),
        int(
            simulator.profile.local_models[fallback_id].deployment_resource_bounds[
                "max_memory_bytes"
            ]
        ),
    )
    assert vehicle.reconcile_reservation(
        task, {"accelerator": 1, "cpu": 1, "encoder": 1}, pipeline_memory
    )
    task.selected_pipeline = row.pipeline_id
    task.trace_row_id = row.row_id
    task.max_attempts = pipeline.max_attempts
    simulator._anon_rows[task.task_id] = row
    TaskStateMachine.transition(
        task, TaskState.ANON_WAIT, time_s=simulator.state.clock_s, trigger="TEST_PIPE"
    )
    simulator._enqueue_anon_attempt(task)

    for _ in range(20):
        if task.state is TaskState.READY:
            break
        assert not task.terminal
        _complete_current_vehicle_job(simulator, task)
    assert task.state is TaskState.READY

    fallback_memory = int(
        simulator.profile.local_models[fallback_id].deployment_resource_bounds[
            "max_memory_bytes"
        ]
    )
    assert task.encoded_anon is not None
    assert task.raw_handle is not None and task.aligned_handle is not None
    assert task.reservation_tokens == {"accelerator": 1}
    assert task.memory_reservation_bytes == fallback_memory
    assert vehicle.descriptors_reserved["accelerator"] == 1
    assert vehicle.descriptors_reserved["cpu"] == 0
    assert vehicle.descriptors_reserved["encoder"] == 0
    assert task.current_job_id is None
    assert_all_invariants(simulator.state, simulator.profile)


def test_same_timestamp_planned_shadow_prevents_double_capacity_promise(
    decision_fixture,
):
    fixture = decision_fixture(task_id="planned-shadow-a", deadline_s=10.0)
    simulator = fixture.simulator
    first = fixture.task
    second = _new_raw_task("planned-shadow-b", first)
    simulator.state.tasks[second.task_id] = second
    vehicle = simulator.state.vehicles[first.vehicle_id]
    prep_memory = int(
        simulator.profile.preprocessing_resource_bounds["max_memory_bytes"]
    )
    local_action = next(
        item
        for item in simulator.mask_engine.enumerate(
            first, simulator._observation(first), simulator.state
        ).allowed
        if item.kind is ActionKind.LOCAL
    )
    shadow = simulator._planned_vehicle_shadow(first, local_action)
    assert shadow is not None
    _, local_memory = shadow
    assert prep_memory < local_memory
    assert vehicle.reserve(first, {}, prep_memory)
    assert vehicle.reserve(second, {}, prep_memory)
    vehicle.memory_capacity_bytes = prep_memory + local_memory
    vehicle.descriptor_capacity["accelerator"] = 1
    second_initial_mask = simulator.mask_engine.enumerate(
        second, simulator._observation(second), simulator.state
    )
    second_initial_local = next(
        item for item in second_initial_mask.allowed if item.kind is ActionKind.LOCAL
    )
    assert second_initial_local.canonical_id == local_action.canonical_id

    simulator._make_decisions()

    assert simulator._pending_decisions[first.task_id].kind is ActionKind.LOCAL
    assert simulator._pending_decisions[second.task_id].kind is ActionKind.FAIL
    assert first.reservation_tokens == {"accelerator": 1}
    assert first.memory_reservation_bytes == local_memory
    assert second.reservation_tokens == {}
    assert second.memory_reservation_bytes == prep_memory
    assert vehicle.descriptors_reserved["accelerator"] == 1
    assert vehicle.memory_reserved_bytes == vehicle.memory_capacity_bytes


def test_active_vehicle_job_memory_must_fit_task_shadow(decision_fixture):
    fixture = decision_fixture(task_id="job-memory-envelope")
    task = fixture.task
    vehicle = fixture.state.vehicles[task.vehicle_id]
    pool = vehicle.resources["accelerator"]
    assert vehicle.reserve(task, {"accelerator": 1}, 1)
    TaskStateMachine.transition(
        task, TaskState.LOCAL_WAIT, time_s=fixture.state.clock_s, trigger="TEST_LOCAL"
    )
    job = ComputeJob(
        job_id="oversized-memory-job",
        task_id=task.task_id,
        owner_type="vehicle",
        owner_id=task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version="test-model",
        enqueue_time_s=fixture.state.clock_s,
        absolute_deadline_s=task.absolute_deadline_s,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=0.1,
        residual_work_s=0.1,
        total_dynamic_energy_j=0.1,
        memory_need_bytes=2,
    )
    pool.enqueue(job)
    task.current_job_id = job.job_id

    with pytest.raises(InvariantViolation) as caught:
        assert_all_invariants(fixture.state, fixture.simulator.profile)
    assert caught.value.detail.code == "VEHICLE_JOB_MEMORY_UNRESERVED"
