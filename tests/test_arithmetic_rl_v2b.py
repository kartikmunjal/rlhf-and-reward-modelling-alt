from src.evaluation.arithmetic_rl_v2b import generate_problems_v2b, score_completion_v2b


REWARD = {
    "tagged_exact": 1.0,
    "numeric_exact_without_tag": 0.8,
    "correct_intermediate_each": 0.1,
    "valid_tag_wrong": 0.05,
    "invalid_or_wrong": -0.1,
    "length_penalty_after_tokens": 120,
    "length_penalty_per_token": -0.001,
}


def test_v2b_equations_and_determinism():
    problems = generate_problems_v2b(9, 20)
    assert problems == generate_problems_v2b(9, 20)
    for problem in problems:
        initial, crates, per_crate, damaged, branches = problem.operands
        assert problem.incoming == crates * per_crate
        assert problem.remaining == initial + problem.incoming - damaged
        assert problem.answer == problem.remaining * branches


def test_v2b_process_reward_creates_multiple_levels():
    problem = generate_problems_v2b(10, 1)[0]
    exact, _ = score_completion_v2b(f"<answer>{problem.answer}</answer>", problem, REWARD)
    two_steps, meta = score_completion_v2b(
        f"Incoming {problem.incoming}; remaining {problem.remaining}; unfinished", problem, REWARD
    )
    one_step, _ = score_completion_v2b(f"Incoming {problem.incoming}; unfinished", problem, REWARD)
    wrong, _ = score_completion_v2b("I cannot solve this.", problem, REWARD)
    assert exact > two_steps > one_step > wrong
    assert meta["intermediate_hits"] == 2
