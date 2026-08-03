import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_ppo_grpo_v2 import hierarchical_bootstrap, load_and_validate


ROOT = Path(__file__).resolve().parents[1]


def test_published_v2_artifacts_pass_locked_integrity_checks():
    config = json.loads((ROOT / "configs/ppo_grpo_v2.json").read_text(encoding="utf-8"))
    runs, baseline = load_and_validate(ROOT / "results/ppo_grpo_v2", config)

    assert len(baseline) == 400
    assert [len(runs[method]) for method in ("ppo", "grpo")] == [3, 3]
    for method in ("ppo", "grpo"):
        for run in runs[method]:
            assert run["manifest"]["observed_budget"] == {
                "optimizer_steps": 200,
                "rollout_groups": 100,
                "generated_completions": 400,
            }


def test_paired_bootstrap_is_deterministic_and_preserves_direction():
    config = json.loads((ROOT / "configs/ppo_grpo_v2.json").read_text(encoding="utf-8"))
    runs, baseline = load_and_validate(ROOT / "results/ppo_grpo_v2", config)
    first = hierarchical_bootstrap(runs, baseline, np.random.default_rng(20260803), 2000)
    second = hierarchical_bootstrap(runs, baseline, np.random.default_rng(20260803), 2000)

    assert first == second
    assert first["grpo_minus_ppo"]["estimate"] == pytest.approx(0.29)
    assert first["grpo_minus_ppo"]["ci95"][0] > 0


def test_budget_tampering_fails_closed(tmp_path):
    config = json.loads((ROOT / "configs/ppo_grpo_v2.json").read_text(encoding="utf-8"))
    source = ROOT / "results/ppo_grpo_v2"
    for path in source.rglob("*.json"):
        target = tmp_path / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    manifest_path = tmp_path / "ppo_seed2025" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["observed_budget"]["optimizer_steps"] = 199
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Budget mismatch"):
        load_and_validate(tmp_path, copy.deepcopy(config))
