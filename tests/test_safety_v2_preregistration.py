import json
from itertools import product
from pathlib import Path

import yaml


def test_trial_ledger_is_exact_locked_cartesian_product():
    config = yaml.safe_load(Path("configs/safety_v2.yaml").read_text())
    ledger = json.loads(Path("configs/safety_v2_trial_ledger.json").read_text())
    expected = {
        (loss["name"], seed)
        for loss, seed in product(config["losses"], config["seeds"])
    }
    actual = {(trial["loss"], trial["seed"]) for trial in ledger["trials"]}
    assert ledger["planned_n_trials"] == 12
    assert [trial["trial_index"] for trial in ledger["trials"]] == list(range(1, 13))
    assert actual == expected


def test_v1_results_remain_versioned_separately():
    assert Path("results/safety_classifier_v1/metrics.json").exists()
    assert Path("docs/safety_v2_preregistered_plan.md").exists()
