from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talos.architecture.abstract_accelerator import AbstractAccelerator
from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.ip.ip_pool import IPPool
from talos.level2.problem import Level2PymooProblem
from talos.level2.runner import (
    DEFAULT_LEVEL2_OBJECTIVES,
    _evaluate_solution,
    _write_solutions_csv,
)


@dataclass(frozen=True)
class Level2ExhaustiveRunResult:
    problem: Level2PymooProblem
    solutions: list[dict[str, Any]]
    explored_combinations: int
    csv_path: Path | None = None


def run_level2_exhaustive(
    accelerator: AbstractAccelerator,
    ip_pool: IPPool,
    objective_names: list[str] | None = None,
    seed: int = 1,
    save_csv: bool = True,
    results_dir: str | None = None,
    constraints: UserConstraints | None = None,
    activity_profile: WorkloadActivityProfile | None = None,
    max_combinations: int = 100_000,
) -> Level2ExhaustiveRunResult:
    objectives = list(objective_names or DEFAULT_LEVEL2_OBJECTIVES)
    problem = Level2PymooProblem(
        accelerator=accelerator,
        ip_pool=ip_pool,
        objective_names=objectives,
        constraints=constraints,
        activity_profile=activity_profile,
    )
    explored_combinations = problem.spec.genome_count()
    if explored_combinations > max_combinations:
        raise ValueError(
            "Level 2 exhaustive search would explore "
            f"{explored_combinations} combinations, above limit {max_combinations}."
        )

    rows: list[dict[str, Any]] = []
    for solution_index, genome in enumerate(problem.spec.iter_genomes()):
        row = _evaluate_solution(
            problem=problem,
            solution_index=solution_index,
            genome=[float(value) for value in genome],
            pop_size=explored_combinations,
            n_gen=1,
            seed=seed,
        )
        row["strategy"] = "exhaustive"
        row["explored_combinations"] = explored_combinations
        if row["valid"] and row["constraints_satisfied"]:
            rows.append(row)

    rows.sort(
        key=lambda row: (
            tuple(float(value) for value in row["objective_values"]),
            tuple(float(value) for value in row["genome"]),
        )
    )

    csv_path = None
    if save_csv:
        output_dir = (
            Path(results_dir)
            if results_dir is not None
            else Path("results") / "level2"
        )
        csv_path = output_dir / "level2_exhaustive_results.csv"
        _write_solutions_csv(csv_path, rows)

    return Level2ExhaustiveRunResult(
        problem=problem,
        solutions=rows,
        explored_combinations=explored_combinations,
        csv_path=csv_path,
    )
