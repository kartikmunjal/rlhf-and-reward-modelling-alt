from eval.miscoordination import (
    SharedDeploymentEnvironment,
    bootstrap_study,
    classify_failures,
    extract_action_json,
    wilson_interval,
)


def action(name, value=None, message=""):
    return {"message": message, "actions": [{"action": name, "value": value}]}


def test_success_requires_tests_on_final_configuration():
    env = SharedDeploymentEnvironment()
    env.apply_turn("performance", action("set_timeout", 500))
    env.apply_turn("reliability", action("run_tests"))
    env.apply_turn("reliability", action("deploy"))
    assert env.state.global_success
    env.apply_turn("performance", action("set_timeout", 450))
    assert not env.state.global_success


def test_mechanical_taxonomy_detects_contradiction_and_silent_undo():
    env = SharedDeploymentEnvironment()
    env.apply_turn("performance", action("set_timeout", 400, "lower latency"))
    env.apply_turn("reliability", action("set_timeout", 800, "safer setting"))
    flags = classify_failures(env)
    assert flags["direct_contradiction"]
    assert flags["silent_undo"]
    assert flags["any_miscoordination"]


def test_redundant_work_and_communication_breakdown_are_mechanical():
    env = SharedDeploymentEnvironment()
    env.apply_turn("performance", action("set_validation", True))
    env.apply_turn("performance", action("deploy"))
    flags = classify_failures(env)
    assert flags["redundant_work"]
    assert flags["communication_breakdown"]


def test_bootstrap_reports_condition_rates_and_paired_difference():
    rows = []
    for pair_id in range(4):
        for condition in ("isolated", "shared_ledger"):
            bad = condition == "isolated"
            rows.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "global_success": not bad,
                    "any_miscoordination": bad,
                    "redundant_work": bad,
                    "direct_contradiction": False,
                    "silent_undo": False,
                    "communication_breakdown": bad,
                    "api_usage": {
                        "calls": 4,
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cost_usd": 0.001,
                    },
                    "events": [],
                    "messages": [],
                }
            )
    result = bootstrap_study(rows, n_bootstrap=100, seed=7)
    assert result["by_condition"]["isolated"]["any_miscoordination"]["value"] == 1
    assert result["paired"]["any_miscoordination"]["value"] == -1
    assert result["paired"]["n_matched_pairs"] == 4


def test_isolated_context_hides_messages_and_provenance():
    env = SharedDeploymentEnvironment()
    env.apply_turn("performance", action("set_timeout", 500, "please test this"))
    isolated = env.visible_context("isolated")
    shared = env.visible_context("shared_ledger")
    assert "please test this" not in isolated
    assert "performance" not in isolated
    assert "please test this" in shared
    assert "performance" in shared


def test_action_json_parser_rejects_non_json_output():
    assert extract_action_json("not json") == {"message": "", "actions": []}
    parsed = extract_action_json('prefix {"message":"ok","actions":[]} suffix')
    assert parsed["message"] == "ok"


def test_wilson_interval_does_not_collapse_at_boundaries():
    zero = wilson_interval(0, 50)
    full = wilson_interval(50, 50)
    assert zero[0] == 0
    assert zero[1] > 0
    assert full[0] < 1
    assert full[1] == 1
