from src.evaluation.arithmetic_rl_v2 import (
    generate_problems_v2,
    parse_final_integer,
    parse_tagged_answer,
    score_completion_v2,
    sft_completion,
)


REWARD = {
    "tagged_exact": 1.0,
    "numeric_exact_without_tag": 0.8,
    "correct_intermediate": 0.15,
    "valid_tag_wrong": 0.05,
    "invalid_or_wrong": -0.1,
    "length_penalty_after_tokens": 120,
    "length_penalty_per_token": -0.001,
}


def test_v2_generator_is_deterministic_and_solution_is_consistent():
    problems = generate_problems_v2(11, 20)
    assert problems == generate_problems_v2(11, 20)
    for problem in problems:
        start, received, sold, stores = problem.operands
        assert problem.intermediate == start + received - sold
        assert problem.answer == problem.intermediate * stores
        assert parse_tagged_answer(sft_completion(problem)) == problem.answer


def test_robust_numeric_parser_prefers_tag_then_last_integer():
    assert parse_final_integer("work 10 then 42") == 42
    assert parse_final_integer("work 10 then <answer>37</answer> trailing 99") == 37
    assert parse_final_integer("no numeric output") is None


def test_shaped_reward_has_ordered_support_and_intermediate_signal():
    problem = generate_problems_v2(12, 1)[0]
    tagged, _ = score_completion_v2(f"<answer>{problem.answer}</answer>", problem, REWARD)
    numeric, _ = score_completion_v2(f"The result is {problem.answer}", problem, REWARD)
    intermediate, metadata = score_completion_v2(
        f"The remaining amount is {problem.intermediate}, but I stopped.", problem, REWARD
    )
    invalid, _ = score_completion_v2("I do not know.", problem, REWARD)
    assert tagged > numeric > intermediate > invalid
    assert metadata["intermediate_present"] is True


def test_training_and_pilot_ids_can_be_structurally_disjoint():
    train = generate_problems_v2(1, 20, id_offset=0)
    pilot = generate_problems_v2(2, 20, id_offset=1_000_000)
    assert {x.problem_id for x in train}.isdisjoint(x.problem_id for x in pilot)
