"""Mutation audit and deterministic V2 test selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .local_eval import evaluate_trusted, pass_all
from .mutations import Mutant, generate_mutants
from .schema import CodeTask, TestCase


@dataclass(frozen=True)
class VerifierAudit:
    task_id: str
    generated_mutants: int
    valid_non_equivalent_mutants: int
    v0_killed: int
    v0_mutation_kill_rate: float
    v2_added_tests: int
    v2_killed: int
    v2_mutation_kill_rate: float
    v0_test_count: int
    v1_test_count: int
    v3_test_count: int
    hidden_partitions: tuple[str, ...]
    weak_flags: tuple[str, ...]
    selected_v2_input_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def valid_mutants(task: CodeTask, maximum: int = 64) -> list[Mutant]:
    candidates = generate_mutants(task.reference_solution, maximum=maximum)
    return [m for m in candidates if not pass_all(m.source, task.entry_point, task.hidden_tests)]


def killed_by(mutant: Mutant, task: CodeTask, cases: tuple[TestCase, ...]) -> bool:
    return not pass_all(mutant.source, task.entry_point, cases)


def select_v2_cases(
    task: CodeTask,
    mutants: list[Mutant],
    maximum_added: int = 12,
    visible_killed: set[str] | None = None,
    hidden_kills: dict[str, set[str]] | None = None,
) -> tuple[TestCase, ...]:
    visible_killed = visible_killed or {m.mutant_id for m in mutants if killed_by(m, task, task.visible_tests)}
    survivors = {m.mutant_id: m for m in mutants if m.mutant_id not in visible_killed}
    if hidden_kills is None:
        hidden_kills = {
            case.input_id: {mid for mid, mutant in survivors.items() if killed_by(mutant, task, (case,))}
            for case in task.hidden_tests
        }
    selected: list[TestCase] = []
    candidates = sorted(task.hidden_tests, key=lambda case: case.input_id)
    while survivors and len(selected) < maximum_added:
        scored = []
        for case in candidates:
            if case in selected:
                continue
            killed = set(hidden_kills[case.input_id]) & set(survivors)
            scored.append((len(killed), case.input_id, case, killed))
        if not scored:
            break
        count, _, case, killed = max(scored, key=lambda row: (row[0], row[1]))
        if count == 0:
            break
        selected.append(case)
        for mutant_id in killed:
            survivors.pop(mutant_id, None)
    return tuple(selected)


def audit_task(task: CodeTask, maximum_mutants: int = 64, maximum_added: int = 12) -> VerifierAudit:
    generated = generate_mutants(task.reference_solution, maximum=maximum_mutants)
    mutants = []
    visible_killed: set[str] = set()
    hidden_kills: dict[str, set[str]] = {case.input_id: set() for case in task.hidden_tests}
    for mutant in generated:
        hidden_rows = evaluate_trusted(
            mutant.source, task.entry_point, task.hidden_tests, timeout_seconds=0.05
        )
        if all(row["passed"] for row in hidden_rows):
            continue
        mutants.append(mutant)
        visible_rows = evaluate_trusted(
            mutant.source, task.entry_point, task.visible_tests, timeout_seconds=0.05
        )
        if not all(row["passed"] for row in visible_rows):
            visible_killed.add(mutant.mutant_id)
        for case, row in zip(task.hidden_tests, hidden_rows):
            if not row["passed"]:
                hidden_kills[case.input_id].add(mutant.mutant_id)
    v0_killed = len(visible_killed)
    selected = select_v2_cases(
        task, mutants, maximum_added=maximum_added,
        visible_killed=visible_killed, hidden_kills=hidden_kills,
    )
    v2_ids = set(visible_killed)
    for case in selected:
        v2_ids.update(hidden_kills[case.input_id])
    v2_killed = len(v2_ids)
    denominator = len(mutants)
    flags = []
    if denominator < 8:
        flags.append("fewer_than_8_valid_mutants")
    if v0_killed == 0:
        flags.append("v0_kills_none")
    if denominator and v0_killed == denominator:
        flags.append("v0_kills_all")
    if denominator and v2_killed < denominator:
        flags.append("v2_surviving_mutants")
    return VerifierAudit(
        task_id=task.task_id,
        generated_mutants=len(generated),
        valid_non_equivalent_mutants=denominator,
        v0_killed=v0_killed,
        v0_mutation_kill_rate=v0_killed / denominator if denominator else 0.0,
        v2_added_tests=len(selected),
        v2_killed=v2_killed,
        v2_mutation_kill_rate=v2_killed / denominator if denominator else 0.0,
        v0_test_count=len(task.visible_tests),
        v1_test_count=len(task.hidden_tests),
        v3_test_count=len(task.property_tests),
        hidden_partitions=tuple(sorted({case.partition for case in task.hidden_tests})),
        weak_flags=tuple(flags),
        selected_v2_input_ids=tuple(case.input_id for case in selected),
    )
