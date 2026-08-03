from src.evaluation.arithmetic_rl_v2c import generate_problems_v2c


def test_v2c_is_deterministic_and_uses_intermediate_ranges():
    problems = generate_problems_v2c(13, 100)
    assert problems == generate_problems_v2c(13, 100)
    for problem in problems:
        initial, crates, per_crate, damaged, branches = problem.operands
        assert 40 <= initial <= 150
        assert 3 <= crates <= 10
        assert 5 <= per_crate <= 15
        assert 3 <= damaged <= 30
        assert 2 <= branches <= 5
        assert problem.answer == (initial + crates * per_crate - damaged) * branches
