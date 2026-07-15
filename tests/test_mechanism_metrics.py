from __future__ import annotations

import pytest

from privacy_edge_sim.metrics import _mechanism_path_metrics


def _task_row(
    *,
    done: bool = False,
    attempts: int = 0,
    local: str | None = None,
    rsu: str | None = None,
    edge: str | None = None,
) -> dict[str, object]:
    return {
        "done": done,
        "attempt_started_count": attempts,
        "selected_local_model": local,
        "selected_rsu": rsu,
        "selected_edge_model": edge,
    }


def test_mechanism_path_metrics_use_task_and_pipeline_denominators():
    metrics = _mechanism_path_metrics(
        [
            _task_row(done=True, local="local"),
            _task_row(done=True, attempts=1, rsu="rsu-1", edge="edge"),
            # This task starts edge transport but completes after local fallback.
            _task_row(
                done=True,
                attempts=1,
                local="local",
                rsu="rsu-1",
                edge="edge",
            ),
            _task_row(done=True, attempts=1, local="local"),
            _task_row(attempts=1),
            _task_row(),
        ]
    )

    assert metrics["mechanism_path_counts"] == {
        "task_count": 6,
        "edge_done_count": 1,
        "pipeline_attempt_count": 4,
        "pipeline_to_edge_count": 2,
        "pipeline_to_local_count": 2,
    }
    assert metrics["mechanism_path_denominators"] == {
        "edge_done_rate": 6,
        "pipeline_attempt_rate": 6,
        "pipeline_to_edge_rate": 4,
        "pipeline_to_local_rate": 4,
    }
    assert metrics["edge_done_rate"] == pytest.approx(1.0 / 6.0)
    assert metrics["pipeline_attempt_rate"] == pytest.approx(4.0 / 6.0)
    assert metrics["pipeline_to_edge_rate"] == pytest.approx(0.5)
    assert metrics["pipeline_to_local_rate"] == pytest.approx(0.5)


def test_mechanism_path_metrics_emit_numeric_zero_without_pipeline_attempts():
    metrics = _mechanism_path_metrics(
        [
            _task_row(done=True, local="local"),
            _task_row(done=True, local="local"),
        ]
    )

    assert metrics["edge_done_rate"] == 0.0
    assert metrics["pipeline_attempt_rate"] == 0.0
    assert metrics["pipeline_to_edge_rate"] == 0.0
    assert metrics["pipeline_to_local_rate"] == 0.0
    assert metrics["mechanism_path_denominators"]["pipeline_to_edge_rate"] == 0
    assert metrics["mechanism_path_denominators"]["pipeline_to_local_rate"] == 0
