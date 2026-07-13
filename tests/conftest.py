from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from privacy_edge_sim.config import SimulationConfig, load_config
from privacy_edge_sim.enums import TaskState
from privacy_edge_sim.packets import AlignedTensorHandle, RawImageHandle
from privacy_edge_sim.profiles import FrozenProfileBundle, load_profile
from privacy_edge_sim.simulator import DiscreteEventSimulator, RunResult
from privacy_edge_sim.state import TaskRecord, TaskStateMachine
from privacy_edge_sim.traces import TraceBundle, load_trace


POLICIES = (
    "all_local",
    "fixed_safe_lowest_link_cost",
    "fixed_safe_shortest_visible_queue",
    "safe_greedy",
    "safe_lyapunov_h1",
    "esl_smpc",
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(repo_root: Path) -> SimulationConfig:
    return load_config(repo_root / "configs" / "default.json")


@pytest.fixture(scope="session")
def profile(config: SimulationConfig) -> FrozenProfileBundle:
    return load_profile(config.profile_path)


@pytest.fixture(scope="session")
def trace(config: SimulationConfig, profile: FrozenProfileBundle) -> TraceBundle:
    return load_trace(config.trace_path, profile)


@pytest.fixture(scope="session")
def policy_results(
    config: SimulationConfig,
    profile: FrozenProfileBundle,
    trace: TraceBundle,
) -> dict[str, RunResult]:
    return {
        name: DiscreteEventSimulator(
            config, profile, trace, name, policy_name=name
        ).run()
        for name in POLICIES
    }


@pytest.fixture
def decision_fixture(
    config: SimulationConfig,
    profile: FrozenProfileBundle,
    trace: TraceBundle,
):
    """Return a factory for one canonical RAW task in an isolated simulator."""

    def make(
        *,
        task_id: str = "unit-decision-task",
        deadline_s: float = 10.0,
        bins: tuple[str, ...] = ("clear",),
        ood: bool = False,
        battery_j: float | None = None,
    ) -> SimpleNamespace:
        simulator = DiscreteEventSimulator(config, profile, trace, "all_local")
        if battery_j is not None:
            simulator.state.vehicles["veh-1"].battery_j = battery_j
        task = TaskRecord(
            task_id=task_id,
            vehicle_id="veh-1",
            arrival_time_s=0.0,
            relative_deadline_s=deadline_s,
            absolute_deadline_s=deadline_s,
            raw_handle=RawImageHandle(f"raw-{task_id}"),
            aligned_handle=AlignedTensorHandle(f"aligned-{task_id}"),
            quality_features=(0.8, 0.2),
            quality_probabilities=tuple(
                (quality_bin, 1.0 / len(bins)) for quality_bin in bins
            ),
            conformal_quality_bins=bins,
            ood=ood,
            true_identity="simulator-only-person",
            true_expression_label="simulator-only-label",
            true_quality_region=bins[0] if bins else None,
            realized_attack_outcomes={"identity": True},
        )
        simulator.state.tasks[task_id] = task
        TaskStateMachine.transition(
            task, TaskState.PREP_WAIT, time_s=0.0, trigger="UNIT_PREP_ENQUEUE"
        )
        TaskStateMachine.transition(
            task, TaskState.PREP_RUN, time_s=0.0, trigger="UNIT_PREP_START"
        )
        TaskStateMachine.transition(
            task, TaskState.RAW, time_s=0.0, trigger="UNIT_PREP_DONE"
        )
        observation = simulator._observation(task)  # production allow-list builder
        return SimpleNamespace(
            simulator=simulator,
            state=simulator.state,
            task=task,
            observation=observation,
        )

    return make


@pytest.fixture
def deadline_trace(trace: TraceBundle) -> TraceBundle:
    arrivals = list(trace.arrivals)
    arrivals[0] = replace(arrivals[0], relative_deadline_s=0.01)
    return replace(trace, arrivals=tuple(arrivals))
