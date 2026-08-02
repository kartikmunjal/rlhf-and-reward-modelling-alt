from src.evaluation.arithmetic_rl import extract_answer, generate_problems, score_completion


def test_generator_is_deterministic_and_splits_can_be_disjoint():
    first = generate_problems(7, 20)
    assert first == generate_problems(7, 20)
    evaluation = generate_problems(8, 20, id_offset=1_000_000)
    assert {x.problem_id for x in first}.isdisjoint(x.problem_id for x in evaluation)
    assert all(x.answer > 0 for x in first + evaluation)


def test_answer_parser_is_strict_about_tags():
    assert extract_answer("work <answer>-12</answer>") == -12
    assert extract_answer("the answer is 12") is None


def test_locked_reward_ordering():
    config = {
        "exact_answer": 1.0,
        "valid_answer_tag": 0.1,
        "invalid_format": -0.1,
        "length_penalty_per_token_after_48": -0.002,
    }
    assert score_completion("<answer>7</answer>", 7, config) > score_completion(
        "<answer>8</answer>", 7, config
    )
    assert score_completion("<answer>8</answer>", 7, config) > score_completion("8", 7, config)
