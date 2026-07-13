from __future__ import annotations

import json

import pytest

from privacy_edge_sim.checkpoint import (
    checkpoint_identity,
    load_replay_checkpoint,
    write_replay_checkpoint,
)
from privacy_edge_sim.manifest import detect_code_version
from privacy_edge_sim.simulator import DiscreteEventSimulator


def _identity(config, profile, trace, scenario_trace):
    return checkpoint_identity(
        config=config,
        profile_hash=profile.profile_hash,
        evaluation_trace_hash=trace.trace_hash,
        scenario_trace_hash=scenario_trace.trace_hash,
        protocol_version=config.protocol_version,
        policy="safe_lyapunov_h1",
        code_version=str(detect_code_version()["value"]),
    )


def test_replay_checkpoint_resumes_exact_event_prefix(tmp_path, config, profile, trace):
    scenario_trace = DiscreteEventSimulator(
        config, profile, trace, "safe_lyapunov_h1"
    ).scenario_trace
    identity = _identity(config, profile, trace, scenario_trace)
    checkpoint_path = tmp_path / "run.replay.json"
    captured = {}

    def callback(count, clock_s, digest, complete):
        if count == 5 and not captured:
            captured.update(
                compound_events=count,
                clock_s=clock_s,
                prefix_sha256=digest,
            )
            write_replay_checkpoint(
                checkpoint_path,
                identity=identity,
                compound_events=count,
                clock_s=clock_s,
                prefix_sha256=digest,
                complete=complete,
            )

    first_simulator = DiscreteEventSimulator(config, profile, trace, "safe_lyapunov_h1")
    first = first_simulator.run(checkpoint_callback=callback)
    document = load_replay_checkpoint(checkpoint_path, expected_identity=identity)
    second_simulator = DiscreteEventSimulator(
        config, profile, trace, "safe_lyapunov_h1"
    )
    second = second_simulator.run(replay_checkpoint=document)

    assert first.state.event_log == second.state.event_log
    assert (
        first_simulator._replay_hasher.hexdigest()
        == second_simulator._replay_hasher.hexdigest()
    )
    serialized = checkpoint_path.read_text(encoding="utf-8").lower()
    assert "raw_handle" not in serialized
    assert "aligned_handle" not in serialized
    assert '"contains_raw_or_aligned_payload": false' in serialized


def test_replay_checkpoint_rejects_corruption(tmp_path, config, profile, trace):
    simulator = DiscreteEventSimulator(config, profile, trace, "safe_lyapunov_h1")
    identity = _identity(config, profile, trace, simulator.scenario_trace)
    path = write_replay_checkpoint(
        tmp_path / "checkpoint.json",
        identity=identity,
        compound_events=0,
        clock_s=trace.horizon_start_s,
        prefix_sha256="0" * 64,
        complete=False,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["clock_s"] = 123.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_replay_checkpoint(path, expected_identity=identity)
