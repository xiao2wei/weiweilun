from __future__ import annotations

import pytest

from privacy_edge_sim.enums import Operation, ResourceKind
from privacy_edge_sim.policies import POLICY_REGISTRY, _BranchJob
from privacy_edge_sim.resources import ComputeJob, ResourcePool
from privacy_edge_sim.traces import ScenarioEnvironment, ScenarioThermalSegment


def _running_job(*, rate_energy_j: float = 10.0) -> tuple[ResourcePool, ComputeJob]:
    pool = ResourcePool("thermal-unit", ResourceKind.ACCELERATOR, 1)
    job = ComputeJob(
        job_id="thermal-job",
        task_id="thermal-task",
        owner_type="vehicle",
        owner_id="veh-1",
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version="frozen-model",
        enqueue_time_s=0.0,
        absolute_deadline_s=10.0,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=1.0,
        residual_work_s=1.0,
        total_dynamic_energy_j=rate_energy_j,
    )
    pool.enqueue(job)
    pool.dispatch(0.0)
    return pool, job


def test_paired_energy_scales_with_served_work_not_thermal_power():
    slow_pool, slow_job = _running_job()
    hot_pool, hot_job = _running_job()

    slow_advanced = slow_pool.advance(1.0, effective_rate=0.5)
    hot_advanced = hot_pool.advance(1.0, effective_rate=0.5)

    assert slow_job.residual_work_s == pytest.approx(0.5)
    assert hot_job.residual_work_s == pytest.approx(0.5)
    assert slow_advanced[0][2] == pytest.approx(5.0)
    assert hot_advanced[0][2] == pytest.approx(5.0)

    slow_pool.advance(1.0, effective_rate=0.5)
    hot_pool.advance(1.0, effective_rate=0.5)
    assert slow_job.consumed_dynamic_energy_j == pytest.approx(10.0)
    assert hot_job.consumed_dynamic_energy_j == pytest.approx(10.0)


def test_production_advance_does_not_rescale_paired_energy_by_thermal_power(
    decision_fixture,
):
    fixture = decision_fixture(task_id="production-hot-energy")
    simulator = fixture.simulator
    simulator.state.clock_s = 2.5
    task = fixture.task
    pool = simulator.state.vehicles[task.vehicle_id].resources["accelerator"]
    job = ComputeJob(
        job_id="production-hot-job",
        task_id=task.task_id,
        owner_type="vehicle",
        owner_id=task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version="frozen-model",
        enqueue_time_s=2.5,
        absolute_deadline_s=task.absolute_deadline_s,
        enqueue_seq=pool.next_enqueue_seq(),
        total_work_s=1.0,
        residual_work_s=1.0,
        total_dynamic_energy_j=10.0,
    )
    pool.enqueue(job)
    pool.dispatch(2.5)

    simulator._advance_to(3.5)

    assert job.residual_work_s == pytest.approx(0.32)
    assert job.consumed_dynamic_energy_j == pytest.approx(6.8)
    assert task.vehicle_energy_j == pytest.approx(6.8)


def test_branch_advance_integrates_idle_and_thermal_job_power(decision_fixture):
    fixture = decision_fixture(task_id="branch-hot-energy")
    environment = ScenarioEnvironment(
        scenario_id="branch-hot-energy",
        cluster_token="branch-hot-energy-cluster",
        duration_s=2.0,
        wireless=(),
        thermal=(
            ScenarioThermalSegment(
                owner_type="vehicle",
                owner_id=fixture.task.vehicle_id,
                resource="all",
                start_offset_s=0.0,
                end_offset_s=2.0,
                state="hot",
                service_rate_multiplier=0.5,
                dynamic_power_multiplier=0.8,
            ),
        ),
        faults=(),
        background_loads=(),
        telemetry=(),
        # This fixture changes only vehicle thermal service.  RSU private
        # state still comes from the complete t=0 joint-scenario anchor.
        rsu_anchors=fixture.simulator.scenario_library.environment_scenarios[
            0
        ].rsu_anchors,
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)
    assert scheduler is not None
    vehicle_id = fixture.task.vehicle_id
    pool = scheduler.resources[("vehicle", vehicle_id, "accelerator")]
    job = _BranchJob(
        job_id="branch-hot-job",
        task_token="",
        owner_type="vehicle",
        owner_id=vehicle_id,
        resource="accelerator",
        remaining_work_s=1.0,
        total_work_s=1.0,
        total_energy_j=10.0,
        absolute_deadline_s=10.0,
        enqueue_seq=1,
        completion_kind="BACKGROUND_DONE",
    )
    scheduler.jobs[job.job_id] = job
    pool.running.append(job.job_id)
    battery_before = scheduler.vehicle_battery_j[vehicle_id]

    assert policy._scheduler_advance(scheduler, 1.0)

    idle_power = fixture.simulator.config.vehicles[0].idle_power_w
    assert job.remaining_work_s == pytest.approx(0.5)
    assert branch.vehicle_energy_j == pytest.approx(5.0)
    assert branch.vehicle_physical_energy_j[vehicle_id] == pytest.approx(
        5.0 + idle_power
    )
    assert battery_before - scheduler.vehicle_battery_j[vehicle_id] == pytest.approx(
        5.0 + idle_power
    )
    for row in fixture.simulator.config.rsus:
        assert branch.rsu_physical_energy_j[row.rsu_id] == pytest.approx(
            row.idle_power_w
        )
