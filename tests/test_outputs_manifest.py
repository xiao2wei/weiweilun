from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from privacy_edge_sim.manifest import (
    build_manifest,
    detect_code_version,
    source_cleanliness_preflight,
    source_tree_portable_sha256,
    source_tree_sha256,
)
from privacy_edge_sim.metrics import _config_core


def _manifest_kwargs(result, core_digest: str) -> dict:
    return {
        "config": result.config,
        "profile": result.profile,
        "state": result.state,
        "metrics_artifacts": {
            "core_digest": core_digest,
            "files": {},
            "parquet_status": {},
            "controller_diagnostics": {},
        },
        "trace_bundle": result.trace,
        "scenario_trace_bundle": result.scenario_trace,
    }


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "g" * 64])
def test_manifest_rejects_noncanonical_core_digest(policy_results, digest):
    result = policy_results["all_local"]
    with pytest.raises(ValueError, match="lowercase SHA-256 core_digest"):
        build_manifest(**_manifest_kwargs(result, digest))


def test_manifest_rejects_invariant_status_contradictions(policy_results):
    result = policy_results["all_local"]
    kwargs = _manifest_kwargs(result, "0" * 64)
    assert result.state.invariant_checks > 0

    with pytest.raises(ValueError, match="contradicts invariant failure/check counts"):
        build_manifest(
            **kwargs,
            invariant_failures=({"code": "TEST_FAILURE"},),
            invariants_passed=True,
        )

    with pytest.raises(ValueError, match="contradicts invariant failure/check counts"):
        build_manifest(**kwargs, invariant_failures=(), invariants_passed=False)


def test_portable_source_hash_normalizes_text_newlines(tmp_path):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    (lf / "module.py").write_bytes(b"value = 1\nprint(value)\n")
    (crlf / "module.py").write_bytes(b"value = 1\r\nprint(value)\r\n")

    raw_lf, count_lf = source_tree_sha256(lf)
    raw_crlf, count_crlf = source_tree_sha256(crlf)
    portable_lf, portable_count_lf = source_tree_portable_sha256(lf)
    portable_crlf, portable_count_crlf = source_tree_portable_sha256(crlf)

    assert raw_lf != raw_crlf
    assert portable_lf == portable_crlf
    assert count_lf == count_crlf == portable_count_lf == portable_count_crlf == 1


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_clean_source_gate_is_scoped_to_executable_source(policy_results, tmp_path):
    repository = tmp_path / "repository"
    source = repository / "src" / "package"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Manifest Test")
    _git(repository, "config", "user.email", "manifest-test@example.invalid")
    _git(repository, "add", "src/package/module.py")
    _git(repository, "commit", "-m", "source baseline")

    # A new experiment output dirties the repository, but not executable code.
    output = repository / "results" / "run-1"
    output.mkdir(parents=True)
    (output / "summary.json").write_text("{}\n", encoding="utf-8", newline="\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-313.pyc").write_bytes(b"generated-cache")
    version = detect_code_version(source)
    assert version["git_dirty"] is True
    assert version["source_git_dirty"] is False
    assert version["source_commit_reproducible"] is True
    assert version["ignored_generated_source_status_count"] == 1
    assert source_cleanliness_preflight(source, require_clean=True)[
        "requirement_status"
    ] == "passed"

    result = policy_results["all_local"]
    manifest = build_manifest(
        **_manifest_kwargs(result, "0" * 64),
        source_root=source,
        require_clean_source=True,
    )
    assert manifest["source_cleanliness_preflight"]["requirement_status"] == "passed"
    assert manifest["data_provenance"]["source_commit_reproducible"] is True

    (source / "module.py").write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="clean committed source is required"):
        source_cleanliness_preflight(source, require_clean=True)
    with pytest.raises(RuntimeError, match="SOURCE_TREE_DIRTY"):
        build_manifest(
            **_manifest_kwargs(result, "0" * 64),
            source_root=source,
            require_clean_source=True,
        )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_clean_source_gate_detects_assume_unchanged_content(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "src" / "package"
    source.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Manifest Test")
    _git(repository, "config", "user.email", "manifest-test@example.invalid")
    _git(repository, "add", "src/package/module.py")
    _git(repository, "commit", "-m", "source baseline")
    _git(repository, "update-index", "--assume-unchanged", "src/package/module.py")
    module.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")

    version = detect_code_version(source)
    assert version["source_git_dirty"] is True
    assert version["source_commit_reproducible"] is False
    assert version["source_head_matches_working_tree"] is False
    assert version["source_hidden_index_flags"]
    assert version["ignored_generated_source_status_count"] >= 0
    with pytest.raises(RuntimeError, match="clean committed source is required"):
        source_cleanliness_preflight(source, require_clean=True)


def test_semantic_configuration_hash_is_path_independent(policy_results):
    result = policy_results["all_local"]
    first = {
        "profile_path": "C:/checkout-a/profiles/frozen.json",
        "trace_path": "C:/checkout-a/traces/evaluation.json",
        "scenario_trace_path": "C:/checkout-a/traces/scenario.json",
        "evidence_path": "C:/checkout-a/evidence/frozen.json",
        "controller": {"horizon": 1},
    }
    second = {
        **first,
        "profile_path": "/mnt/checkout-b/profiles/renamed-profile.json",
        "trace_path": "/mnt/checkout-b/traces/renamed-evaluation.json",
        "scenario_trace_path": "/mnt/checkout-b/traces/renamed-scenario.json",
        "evidence_path": "/mnt/checkout-b/evidence/renamed-evidence.json",
    }

    manifest_a = build_manifest(
        **_manifest_kwargs(result, "0" * 64), config_content=first
    )
    manifest_b = build_manifest(
        **_manifest_kwargs(result, "0" * 64), config_content=second
    )
    manifest_changed = build_manifest(
        **_manifest_kwargs(result, "0" * 64),
        config_content={**second, "controller": {"horizon": 2}},
    )

    assert (
        manifest_a["configuration"]["canonical_sha256"]
        != manifest_b["configuration"]["canonical_sha256"]
    )
    assert (
        manifest_a["configuration"]["semantic_sha256"]
        == manifest_b["configuration"]["semantic_sha256"]
    )
    assert (
        manifest_b["configuration"]["semantic_sha256"]
        != manifest_changed["configuration"]["semantic_sha256"]
    )
    assert all(
        "checkout" not in str(identity)
        for identity in manifest_a["configuration"]["path_identities"].values()
    )


def test_core_configuration_includes_snapshot_period(config):
    original = _config_core(config)
    changed = _config_core(
        replace(config, rsu_snapshot_period_s=config.rsu_snapshot_period_s / 2.0)
    )

    assert original["rsu_snapshot_period_s"] == config.rsu_snapshot_period_s
    assert changed["rsu_snapshot_period_s"] != original["rsu_snapshot_period_s"]
    assert changed != original


def test_event_output_rows_have_explicit_record_kind_and_attempt_time(policy_results):
    result = policy_results["safe_lyapunov_h1"]
    rows = result.ledger._event_output_rows(result.state)
    event_batches = [row for row in rows if "events" in row]
    attempts = [row for row in rows if row.get("audit_type") == "ANON_ATTEMPT"]

    assert event_batches
    assert attempts
    assert all(row["record_kind"] == "EVENT_BATCH" for row in event_batches)
    assert all(
        row["record_kind"] == "ANON_ATTEMPT" and "time_s" in row for row in attempts
    )
