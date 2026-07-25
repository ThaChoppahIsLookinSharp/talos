from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from talos.architecture.abstract_accelerator import AbstractAccelerator
from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.ip.ip_pool import IPPool
from talos.level2.problem import Level2PymooProblem


DEFAULT_LEVEL2_OBJECTIVES = ["area", "energy", "delay"]
Level2Strategy = Literal["nsga2", "exhaustive"]


@dataclass(frozen=True)
class Level2NSGA2RunResult:
    pymoo_result: Any
    problem: Level2PymooProblem
    solutions: list[dict[str, Any]]
    csv_path: Path | None = None


def run_level2_nsga2(
    accelerator: AbstractAccelerator,
    ip_pool: IPPool,
    objective_names: list[str] | None = None,
    pop_size: int = 6,
    n_gen: int = 2,
    seed: int = 1,
    save_csv: bool = True,
    results_dir: str | None = None,
    debug: bool = False,
    constraints: UserConstraints | None = None,
    activity_profile: WorkloadActivityProfile | None = None,
) -> Level2NSGA2RunResult:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.config import Config
        from pymoo.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pymoo is required to run Level 2 NSGA-II. Install it with `pip install pymoo`."
        ) from exc

    Config.warnings["not_compiled"] = False

    objectives = list(objective_names or DEFAULT_LEVEL2_OBJECTIVES)
    problem = Level2PymooProblem(
        accelerator=accelerator,
        ip_pool=ip_pool,
        objective_names=objectives,
        constraints=constraints,
        activity_profile=activity_profile,
    )
    algorithm = NSGA2(pop_size=pop_size)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"pymoo\..*",
        )
        pymoo_result = minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=seed,
            verbose=debug,
        )

    solutions = _build_solution_rows(
        problem=problem,
        solution_vectors=_iter_solution_vectors(getattr(pymoo_result, "X", None)),
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
    )

    csv_path = None
    if save_csv:
        output_dir = (
            Path(results_dir)
            if results_dir is not None
            else Path("results") / "level2"
        )
        csv_path = output_dir / "level2_nsga2_results.csv"
        _write_solutions_csv(csv_path, solutions)

    return Level2NSGA2RunResult(
        pymoo_result=pymoo_result,
        problem=problem,
        solutions=solutions,
        csv_path=csv_path,
    )


def run_level2(
    accelerator: AbstractAccelerator,
    ip_pool: IPPool,
    objective_names: list[str] | None = None,
    *,
    strategy: Level2Strategy = "nsga2",
    pop_size: int = 6,
    n_gen: int = 2,
    seed: int = 1,
    save_csv: bool = True,
    results_dir: str | None = None,
    debug: bool = False,
    constraints: UserConstraints | None = None,
    activity_profile: WorkloadActivityProfile | None = None,
    exhaustive_max_combinations: int = 100_000,
) -> Any:
    if strategy == "nsga2":
        return run_level2_nsga2(
            accelerator=accelerator,
            ip_pool=ip_pool,
            objective_names=objective_names,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            save_csv=save_csv,
            results_dir=results_dir,
            debug=debug,
            constraints=constraints,
            activity_profile=activity_profile,
        )
    if strategy == "exhaustive":
        from talos.level2.exhaustive_runner import run_level2_exhaustive

        return run_level2_exhaustive(
            accelerator=accelerator,
            ip_pool=ip_pool,
            objective_names=objective_names,
            seed=seed,
            save_csv=save_csv,
            results_dir=results_dir,
            constraints=constraints,
            activity_profile=activity_profile,
            max_combinations=exhaustive_max_combinations,
        )
    raise ValueError("Level 2 strategy must be 'nsga2' or 'exhaustive'.")


def _iter_solution_vectors(raw_x: Any) -> Iterable[list[float]]:
    if raw_x is None:
        return []

    if hasattr(raw_x, "tolist"):
        raw_x = raw_x.tolist()

    if raw_x == []:
        return []

    if isinstance(raw_x, list):
        if not raw_x:
            return []
        if all(not isinstance(value, (list, tuple)) for value in raw_x):
            return [[float(value) for value in raw_x]]
        return [[float(value) for value in row] for row in raw_x]

    return [[float(raw_x)]]


def _build_solution_rows(
    *,
    problem: Level2PymooProblem,
    solution_vectors: Iterable[list[float]],
    pop_size: int,
    n_gen: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ip_sets: set[tuple[tuple[str, str], ...]] = set()
    for index, genome in enumerate(solution_vectors):
        row = _evaluate_solution(
            problem=problem,
            solution_index=index,
            genome=genome,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
        row["strategy"] = "nsga2"
        row["explored_combinations"] = ""
        ip_set = tuple(sorted(row.get("selected_ips", {}).items()))
        if ip_set in seen_ip_sets:
            continue
        seen_ip_sets.add(ip_set)
        rows.append(row)
    return rows


def _evaluate_solution(
    *,
    problem: Level2PymooProblem,
    solution_index: int,
    genome: list[float],
    pop_size: int,
    n_gen: int,
    seed: int,
) -> dict[str, Any]:
    selected_ips: dict[str, str] = {}
    error_message = None
    profile = problem.activity_profile
    dram_accesses = None if profile is None else profile.total_dram_accesses
    dram_access_energy_j = (
        None if profile is None else profile.total_dram_access_energy_j
    )

    try:
        implemented = problem.spec.decode(genome)
        result = problem.evaluator.evaluate(implemented)
        selected_ips = {
            component.abstract_component.name: component.ip.id
            for component in implemented.components
        }
    except Exception as exc:
        result = None
        error_message = str(exc)

    if result is None:
        area = float("inf")
        power = None
        workload_energy_j = None
        workload_latency_s = None
        delay = float("inf")
        throughput = 0.0
        implementation_fmax_mhz = None
        valid = False
        constraint_violations: list[str] = []
    else:
        area = result.area
        power = result.power
        workload_energy_j = result.workload_energy_j
        workload_latency_s = result.workload_latency_s
        delay = result.delay
        throughput = result.throughput
        implementation_fmax_mhz = result.implementation_fmax_mhz
        valid = result.valid
        constraint_violations = list(result.constraint_violations)
        error_message = result.error_message

    if valid and result is not None:
        objective_values = [
            problem._objective_value(objective_name, result)
            for objective_name in problem.objective_names
        ]
    else:
        objective_values = [float("inf")] * len(problem.objective_names)

    return {
        "solution_index": solution_index,
        "genome": genome,
        "selected_ips": selected_ips,
        "area": area,
        "power": power,
        "workload_energy_j": workload_energy_j,
        "workload_latency_s": workload_latency_s,
        "dram_accesses": dram_accesses,
        "dram_access_energy_j": dram_access_energy_j,
        "delay": delay,
        "throughput": throughput,
        "implementation_fmax_mhz": implementation_fmax_mhz,
        "valid": valid,
        "constraints_satisfied": valid and not constraint_violations,
        "constraint_violations": constraint_violations,
        "objective_names": list(problem.objective_names),
        "objective_values": objective_values,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed": seed,
        "error_message": error_message,
    }


def _write_solutions_csv(csv_path: Path, solutions: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solution_index",
        "genome",
        "selected_ips",
        "area",
        "power",
        "workload_energy_j",
        "workload_latency_s",
        "dram_accesses",
        "dram_access_energy_j",
        "delay",
        "throughput",
        "implementation_fmax_mhz",
        "valid",
        "constraints_satisfied",
        "constraint_violations",
        "strategy",
        "explored_combinations",
        "objective_names",
        "objective_values",
        "pop_size",
        "n_gen",
        "seed",
        "error_message",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for solution in solutions:
            writer.writerow(
                {
                    field: _csv_value(solution.get(field))
                    for field in fieldnames
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value
