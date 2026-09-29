import json

import numpy as np
import pytest

from verifier_reward_hacking.audit import audit_task
from verifier_reward_hacking.humaneval_tasks import load_selected_tasks
from verifier_reward_hacking.programmatic_tasks import FAMILY_NAMES, build_candidate
from verifier_reward_hacking.sandbox import SandboxConfig, container_command
from verifier_reward_hacking.schema import CodeTask
from verifier_reward_hacking.statistics import hierarchical_paired_bootstrap, holm_adjust


def test_programmatic_families_are_deterministic_and_schema_round_trips():
    for family in FAMILY_NAMES:
        first = build_candidate(family, 0)
        second = build_candidate(family, 0)
        assert first.task_hash == second.task_hash
        restored = CodeTask.from_dict(json.loads(json.dumps(first.to_dict())))
        assert restored.task_hash == first.task_hash
        assert len(first.visible_tests) == 3
        assert len(first.hidden_tests) == len(first.property_tests) == 20


def test_mutation_audit_leaves_a_weak_v0_and_hardens_v2():
    for family in FAMILY_NAMES:
        audit = audit_task(build_candidate(family, 0))
        assert audit.valid_non_equivalent_mutants >= 8
        assert 0 < audit.v0_killed < audit.valid_non_equivalent_mutants
        assert audit.v2_killed >= audit.v0_killed
        assert audit.v2_added_tests <= 12


def test_sandbox_command_has_required_isolation_and_requires_digest():
    with pytest.raises(ValueError):
        container_command(SandboxConfig())
    cfg = SandboxConfig(image="python@sha256:" + "a" * 64)
    command = container_command(cfg)
    joined = " ".join(command)
    for required in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--pids-limit 64", "--user 65534:65534"):
        assert required in joined


def test_hierarchical_bootstrap_and_holm_are_deterministic():
    left = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=float)
    right = np.zeros_like(left)
    one = hierarchical_paired_bootstrap(left, right, 200, 7)
    two = hierarchical_paired_bootstrap(left, right, 200, 7)
    assert one == two
    assert one["n_trials"] == 3
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_humaneval_archive_hash_is_enforced(tmp_path):
    archive = tmp_path / "HumanEval.jsonl.gz"
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="SHA-256"):
        load_selected_tasks(archive)
