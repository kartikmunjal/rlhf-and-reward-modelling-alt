"""
Category 4: Code Execution (12 tasks, Extension 14).

Each task provides a broken Python function and a test suite. The agent must
identify the bug, write a fix, verify it passes all tests using the python_exec
sandbox tool, and return the corrected function.

Three tiers (4 tasks each):
  Easy   — single-line fix: wrong return value, wrong operator, wrong slice
  Medium — multi-line fix: missing edge case, wrong loop bounds, no recursion
  Hard   — structural fix: wrong DP transition, wrong merge logic, encoding bug

Scorer: fraction of test assertions passed (0.0–1.0).
Sequence scorer: at_least_one_call("python_exec") — did the agent try the tool?

The agent sees the broken code and all test assertions in the task prompt.
ground_truth is the test assertion list (for display); the scorer already
has them via closure from make_code_scorer.
"""

from __future__ import annotations

from textwrap import dedent
from typing import List

from eval.tasks.base import EvalTask
from eval.scorers import at_least_one_call
from eval.tools_code import make_code_scorer


def _make_prompt(description: str, broken_code: str, test_assertions: List[str]) -> str:
    tests_str = "\n".join(test_assertions)
    return dedent(f"""\
        Fix the following Python function so that it passes all tests.

        TASK: {description}

        BROKEN CODE:
        ```python
        {broken_code.strip()}
        ```

        TESTS (all must pass):
        {tests_str}

        Use python_exec to test your fix. When you have a working solution,
        output it with the marker:

        FIXED CODE:
        ```python
        [your corrected function here]
        ```

        Return ONLY the corrected function definition — no test code, no imports
        unless the function itself requires them.
    """).strip()


# ── Easy tier ──────────────────────────────────────────────────────────────────
#  Single-line bugs: wrong return value, wrong operator, wrong slice.

_factorial_broken = """\
def factorial(n):
    if n == 0:
        return 0  # BUG: base case should return 1
    return n * factorial(n - 1)
"""
_factorial_tests = [
    "assert factorial(0) == 1",
    "assert factorial(1) == 1",
    "assert factorial(5) == 120",
    "assert factorial(10) == 3628800",
    "assert factorial(3) == 6",
]

_count_vowels_broken = """\
def count_vowels(s):
    vowels = set('aeio')  # BUG: missing 'u'
    return sum(1 for c in s.lower() if c in vowels)
"""
_count_vowels_tests = [
    "assert count_vowels('hello') == 2",
    "assert count_vowels('beautiful') == 5",
    "assert count_vowels('aeiou') == 5",
    "assert count_vowels('rhythms') == 0",
    "assert count_vowels('') == 0",
    "assert count_vowels('UPPER') == 2",
]

_list_sum_broken = """\
def list_sum(lst):
    total = 0
    for i in range(1, len(lst)):  # BUG: should start at 0, skips first element
        total += lst[i]
    return total
"""
_list_sum_tests = [
    "assert list_sum([1, 2, 3]) == 6",
    "assert list_sum([10, 20, 30]) == 60",
    "assert list_sum([]) == 0",
    "assert list_sum([5]) == 5",
    "assert list_sum([-1, 1]) == 0",
    "assert list_sum([100]) == 100",
]

_is_palindrome_broken = """\
def is_palindrome(s):
    return s == s[1:]  # BUG: s[1:] strips first char; should be s[::-1]
"""
_is_palindrome_tests = [
    "assert is_palindrome('racecar') == True",
    "assert is_palindrome('hello') == False",
    "assert is_palindrome('a') == True",
    "assert is_palindrome('') == True",
    "assert is_palindrome('ab') == False",
    "assert is_palindrome('aba') == True",
]

# ── Medium tier ────────────────────────────────────────────────────────────────
#  Multi-line fixes: missing edge case, wrong condition, missing recursion.

_two_sum_broken = """\
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] != target:  # BUG: should be ==, not !=
                return [i, j]
    return []
"""
_two_sum_tests = [
    "assert two_sum([2, 7, 11, 15], 9) == [0, 1]",
    "assert two_sum([3, 2, 4], 6) == [1, 2]",
    "assert two_sum([3, 3], 6) == [0, 1]",
    "assert two_sum([1, 2, 3, 4], 7) == [2, 3]",
    "assert two_sum([0, 4, 3, 0], 0) == [0, 3]",
]

_is_balanced_broken = """\
def is_balanced(s):
    stack = []
    matching = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if stack[-1] == matching[c]:  # BUG: no empty-stack check → IndexError
                stack.pop()
            else:
                return False
    return len(stack) == 0
"""
_is_balanced_tests = [
    "assert is_balanced('()') == True",
    "assert is_balanced('()[]{}') == True",
    "assert is_balanced('(]') == False",
    "assert is_balanced(')(') == False",
    "assert is_balanced('{[()]}') == True",
    "assert is_balanced(']') == False",
    "assert is_balanced('') == True",
]

_flatten_broken = """\
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)  # BUG: no recursion; only flattens one level
        else:
            result.append(item)
    return result
"""
_flatten_tests = [
    "assert flatten([1, 2, 3]) == [1, 2, 3]",
    "assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]",
    "assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]",
    "assert flatten([[1, [2]], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]",
    "assert flatten([]) == []",
    "assert flatten([1]) == [1]",
]

_second_largest_broken = """\
def second_largest(lst):
    lst = sorted(lst)
    return lst[-1]  # BUG: returns largest, not second largest
"""
_second_largest_tests = [
    "assert second_largest([1, 2, 3]) == 2",
    "assert second_largest([5, 1, 4, 2]) == 4",
    "assert second_largest([10, 20]) == 10",
    "assert second_largest([3, 1, 4, 1, 5, 9]) == 5",
    "assert second_largest([1, 1, 2]) == 1",
]

# ── Hard tier ──────────────────────────────────────────────────────────────────
#  Structural fixes: wrong DP transition, wrong merge, encoding bug.

_merge_intervals_broken = """\
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        last = merged[-1]
        if curr[0] <= last[1]:
            last[1] = curr[1]  # BUG: should be max(last[1], curr[1])
        else:
            merged.append(list(curr))
    return merged
"""
_merge_intervals_tests = [
    "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]",
    "assert merge_intervals([[1,4],[2,3]]) == [[1,4]]",
    "assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]",
    "assert merge_intervals([[1,5],[1,4],[1,3]]) == [[1,5]]",
    "assert merge_intervals([]) == []",
    "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
]

_lcs_broken = """\
def lcs_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # BUG: missing + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
"""
_lcs_tests = [
    "assert lcs_length('abcde', 'ace') == 3",
    "assert lcs_length('abc', 'abc') == 3",
    "assert lcs_length('abc', 'def') == 0",
    "assert lcs_length('AGGTAB', 'GXTXAYB') == 4",
    "assert lcs_length('', 'abc') == 0",
    "assert lcs_length('a', 'a') == 1",
]

_fibonacci_broken = """\
def fibonacci(n):
    if n <= 0:
        return 0
    dp = [0] * (n + 1)
    dp[1] = 0  # BUG: dp[1] should be 1 (fibonacci(1) = 1)
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
"""
_fibonacci_tests = [
    "assert fibonacci(0) == 0",
    "assert fibonacci(1) == 1",
    "assert fibonacci(2) == 1",
    "assert fibonacci(5) == 5",
    "assert fibonacci(10) == 55",
    "assert fibonacci(7) == 13",
]

_decode_rle_broken = """\
def decode_rle(s):
    \"\"\"Decode run-length encoded string: '3a2b' -> 'aaabb', '10x' -> 'xxxxxxxxxx'\"\"\"
    result = ''
    i = 0
    while i < len(s):
        if s[i].isdigit():
            count = int(s[i])  # BUG: reads only ONE digit; '12' parsed as 1 then '2'
            i += 1
            result += s[i] * count
            i += 1
        else:
            result += s[i]
            i += 1
    return result
"""
_decode_rle_tests = [
    "assert decode_rle('3a2b') == 'aaabb'",
    "assert decode_rle('1x') == 'x'",
    "assert decode_rle('4z') == 'zzzz'",
    "assert decode_rle('10a') == 'a' * 10",
    "assert decode_rle('2x12y') == 'xx' + 'y' * 12",
    "assert decode_rle('1a1b1c') == 'abc'",
]


# ── Task registry ──────────────────────────────────────────────────────────────

CODE_EXECUTION_TASKS = [
    # ── Easy ────────────────────────────────────────────────────────────────────
    EvalTask(
        task_id="ce_001",
        category="code_execution",
        prompt=_make_prompt(
            "Fix the factorial function — the base case returns the wrong value.",
            _factorial_broken,
            _factorial_tests,
        ),
        ground_truth=_factorial_tests,
        scorer=make_code_scorer(_factorial_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Easy: base case returns 0 instead of 1. Single-character fix.",
    ),
    EvalTask(
        task_id="ce_002",
        category="code_execution",
        prompt=_make_prompt(
            "Fix count_vowels — it misses one of the five vowels.",
            _count_vowels_broken,
            _count_vowels_tests,
        ),
        ground_truth=_count_vowels_tests,
        scorer=make_code_scorer(_count_vowels_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Easy: vowel set 'aeio' is missing 'u'. Single-char fix.",
    ),
    EvalTask(
        task_id="ce_003",
        category="code_execution",
        prompt=_make_prompt(
            "Fix list_sum — it skips the first element of the list.",
            _list_sum_broken,
            _list_sum_tests,
        ),
        ground_truth=_list_sum_tests,
        scorer=make_code_scorer(_list_sum_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Easy: range starts at 1 instead of 0.",
    ),
    EvalTask(
        task_id="ce_004",
        category="code_execution",
        prompt=_make_prompt(
            "Fix is_palindrome — it compares the string to the wrong slice.",
            _is_palindrome_broken,
            _is_palindrome_tests,
        ),
        ground_truth=_is_palindrome_tests,
        scorer=make_code_scorer(_is_palindrome_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Easy: s[1:] strips first char; should reverse with s[::-1].",
    ),

    # ── Medium ──────────────────────────────────────────────────────────────────
    EvalTask(
        task_id="ce_005",
        category="code_execution",
        prompt=_make_prompt(
            "Fix two_sum — the condition finds pairs that don't sum to target.",
            _two_sum_broken,
            _two_sum_tests,
        ),
        ground_truth=_two_sum_tests,
        scorer=make_code_scorer(_two_sum_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Medium: != target should be == target.",
    ),
    EvalTask(
        task_id="ce_006",
        category="code_execution",
        prompt=_make_prompt(
            "Fix is_balanced — it crashes on strings that start with a closing bracket.",
            _is_balanced_broken,
            _is_balanced_tests,
        ),
        ground_truth=_is_balanced_tests,
        scorer=make_code_scorer(_is_balanced_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Medium: IndexError on empty stack — need to check len(stack) > 0 before pop.",
    ),
    EvalTask(
        task_id="ce_007",
        category="code_execution",
        prompt=_make_prompt(
            "Fix flatten — it only flattens one level of nesting instead of all levels.",
            _flatten_broken,
            _flatten_tests,
        ),
        ground_truth=_flatten_tests,
        scorer=make_code_scorer(_flatten_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Medium: extend(item) should be extend(flatten(item)) for full recursion.",
    ),
    EvalTask(
        task_id="ce_008",
        category="code_execution",
        prompt=_make_prompt(
            "Fix second_largest — it returns the largest value instead of the second largest.",
            _second_largest_broken,
            _second_largest_tests,
        ),
        ground_truth=_second_largest_tests,
        scorer=make_code_scorer(_second_largest_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Medium: lst[-1] returns largest; should be lst[-2].",
    ),

    # ── Hard ────────────────────────────────────────────────────────────────────
    EvalTask(
        task_id="ce_009",
        category="code_execution",
        prompt=_make_prompt(
            "Fix merge_intervals — contained intervals shrink the merged result instead of being absorbed.",
            _merge_intervals_broken,
            _merge_intervals_tests,
        ),
        ground_truth=_merge_intervals_tests,
        scorer=make_code_scorer(_merge_intervals_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Hard: last[1] = curr[1] should be last[1] = max(last[1], curr[1]).",
    ),
    EvalTask(
        task_id="ce_010",
        category="code_execution",
        prompt=_make_prompt(
            "Fix lcs_length — the DP always returns 0 because matching characters don't increment the count.",
            _lcs_broken,
            _lcs_tests,
        ),
        ground_truth=_lcs_tests,
        scorer=make_code_scorer(_lcs_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Hard: dp[i][j] = dp[i-1][j-1] should be dp[i-1][j-1] + 1.",
    ),
    EvalTask(
        task_id="ce_011",
        category="code_execution",
        prompt=_make_prompt(
            "Fix fibonacci — the iterative DP returns wrong values because an initial value is set incorrectly.",
            _fibonacci_broken,
            _fibonacci_tests,
        ),
        ground_truth=_fibonacci_tests,
        scorer=make_code_scorer(_fibonacci_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Hard: dp[1] = 0 should be dp[1] = 1.",
    ),
    EvalTask(
        task_id="ce_012",
        category="code_execution",
        prompt=_make_prompt(
            "Fix decode_rle — it only reads single-digit counts, so '10a' produces 'a' instead of 'aaaaaaaaaa'.",
            _decode_rle_broken,
            _decode_rle_tests,
        ),
        ground_truth=_decode_rle_tests,
        scorer=make_code_scorer(_decode_rle_tests),
        sequence_scorer=at_least_one_call("python_exec"),
        expected_tool_sequence=["python_exec"],
        notes="Hard: must accumulate consecutive digit characters before calling int().",
    ),
]
