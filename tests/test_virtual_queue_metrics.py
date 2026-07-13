from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from privacy_edge_sim.enums import FailureReason, ResourceKind
from privacy_edge_sim.metrics import MetricLedger
from privacy_edge_sim.resources import ResourcePool
from privacy_edge_sim.safety import Action
from privacy_edge_sim.simulator import DiscreteEventSimulator


class _ExplicitFailPolicy:
    name = "test_explicit_fail"

    def choose_action(self, _task, observation, _state):
        return Action.fail(observation.stage)


def _single_arrival_trace(trace):
    return replace(trace, arrivals=(trace.arrivals[0],))


def test_zero_overhead_explicit_fail_updates_virtual_queue_after_compound_closure(
    config, profile, trace
):
    controller = replace(
        config.controller, controller_overhead_s=0.0, controller_energy_j=0.0
    )
    local_config = replace(config, controller=controller)
    simulator = DiscreteEventSimulator(
        local_config,
        profile,
        _single_arrival_trace(trace),
        _ExplicitFailPolicy(),
    )

    result = simulator.run()

    task = next(iter(result.state.tasks.values()))
    assert task.failure_reason is FailureReason.POLICY_EXPLICIT_FAIL
    assert result.state.virtual_queues.failure == pytest.approx(1.0)
    assert result.state.virtual_queues.coverage == pytest.approx(
        local_config.long_term.coverage_rate_minimum
    )
    assert (
        sum(
            current["failure"] > previous["failure"]
            for previous, current in zip(
                result.state.virtual_queues.trajectory,
                result.state.virtual_queues.trajectory[1:],
            )
        )
        == 1
    )


def test_controller_energy_guard_updates_virtual_queue_after_compound_closure(
    config, profile, trace
):
    initial_battery = config.vehicles[0].initial_battery_j
    controller = replace(
        config.controller,
        controller_energy_j=initial_battery + 1.0,
    )
    local_config = replace(config, controller=controller)
    simulator = DiscreteEventSimulator(
        local_config,
        profile,
        _single_arrival_trace(trace),
        "all_local",
    )

    result = simulator.run()

    task = next(iter(result.state.tasks.values()))
    assert task.failure_reason is FailureReason.BATTERY_GUARD
    assert result.state.virtual_queues.failure == pytest.approx(1.0)
    assert result.state.virtual_queues.coverage == pytest.approx(
        local_config.long_term.coverage_rate_minimum
    )


def test_dispatch_battery_failure_is_consumed_with_same_compound_arrival(
    config, profile, trace
):
    simulator = DiscreteEventSimulator(
        config,
        profile,
        _single_arrival_trace(trace),
        "all_local",
    )

    def exceed_available_battery(self, _job, _resource, _time_s):
        return max(vehicle.battery_j for vehicle in self.state.vehicles.values()) + 1.0

    simulator._remaining_job_energy_upper = MethodType(
        exceed_available_battery, simulator
    )

    result = simulator.run()

    task = next(iter(result.state.tasks.values()))
    assert task.failure_reason is FailureReason.BATTERY_GUARD
    assert result.state.virtual_queues.failure == pytest.approx(
        max(0.0, 1.0 - config.long_term.failure_rate_limit)
    )
    assert result.state.virtual_queues.coverage == pytest.approx(
        config.long_term.coverage_rate_minimum
    )
    assert len(result.state.virtual_queues.trajectory) == 1


def test_resource_utilization_is_invariant_to_absolute_time_translation():
    pool = ResourcePool("veh-accel", ResourceKind.ACCELERATOR, server_count=2)
    pool.busy_server_seconds = 4.0
    unshifted = MetricLedger(simulation_start_s=10.0)
    shifted = MetricLedger(simulation_start_s=110.0)

    row_unshifted = unshifted._pool_row(
        pool,
        time_s=20.0,
        owner_type="vehicle",
        owner_id="veh-1",
        physical_energy_j=0.0,
        extras={},
    )
    row_shifted = shifted._pool_row(
        pool,
        time_s=120.0,
        owner_type="vehicle",
        owner_id="veh-1",
        physical_energy_j=0.0,
        extras={},
    )

    state_unshifted = SimpleNamespace(
        clock_s=20.0,
        vehicles={"veh-1": SimpleNamespace(resources={"accelerator": pool})},
        rsus={},
    )
    state_shifted = SimpleNamespace(
        clock_s=120.0,
        vehicles={"veh-1": SimpleNamespace(resources={"accelerator": pool})},
        rsus={},
    )
    summary_unshifted = unshifted._pool_summaries(state_unshifted, [row_unshifted])
    summary_shifted = shifted._pool_summaries(state_shifted, [row_shifted])

    assert row_unshifted["utilization"] == pytest.approx(0.2)
    assert row_shifted["utilization"] == pytest.approx(row_unshifted["utilization"])
    assert summary_shifted["vehicle/veh-1/veh-accel"]["utilization"] == pytest.approx(
        summary_unshifted["vehicle/veh-1/veh-accel"]["utilization"]
    )
