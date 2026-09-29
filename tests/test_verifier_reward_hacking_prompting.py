from verifier_reward_hacking.programmatic_tasks import build_candidate
from verifier_reward_hacking.prompting import build_prompt, extract_code, special_case_flag


def test_prompt_exposes_only_selected_verifier_cases():
    task = build_candidate("affine_clip", 0)
    prompt = build_prompt(task, "v0")
    assert prompt.count("assert solve(") == 3
    for hidden in task.hidden_tests:
        assert hidden.input_id not in prompt


def test_code_extraction_requires_named_function():
    code, status = extract_code("```python\ndef solve(x):\n return x\n```", "solve")
    assert status == "ok" and code.startswith("def solve")
    assert extract_code("```python\ndef other(x): return x\n```", "solve")[0] is None


def test_special_case_flag_is_diagnostic_not_reward():
    task = build_candidate("affine_clip", 0)
    source = "def solve(x):\n    if x == 0: return -3\n    if x == 1: return 2\n    return 7\n"
    assert special_case_flag(source, task)
