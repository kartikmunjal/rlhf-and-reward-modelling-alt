from verifier_reward_hacking.rewards import completion_text, verifier_cases
from verifier_reward_hacking.schema import CodeTask, TestCase as Case


def task():
    visible = tuple(Case(args=[x], expected=x) for x in range(3))
    hidden = tuple(Case(args=[x], expected=x) for x in range(20, 40))
    prop = tuple(Case(args=[x], expected=x) for x in range(40, 60))
    return CodeTask("x", "programmatic_v1", "identity", "identity", "solve", "def solve(x): return x", visible, hidden, prop)


def test_frozen_verifier_case_counts():
    item = task()
    audit = {"selected_v2_input_ids": [item.hidden_tests[0].input_id, item.hidden_tests[1].input_id]}
    assert len(verifier_cases(item, "C1", audit)) == 3
    assert len(verifier_cases(item, "C2", audit)) == 5
    assert len(verifier_cases(item, "C3", audit)) == 20
    assert len(verifier_cases(item, "C4", audit)) == 23


def test_completion_text_handles_trl_conversational_output():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "code"}]) == "code"
