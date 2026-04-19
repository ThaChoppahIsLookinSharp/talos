from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime
import io
import json
from pathlib import Path
import random
from typing import Any

from nsga2.evolution import Evolution
from nsga2.problem import Problem

from talos.architecture.genome import GENOME_LENGTH, gene_bounds, gene_names
from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


DEFAULT_OBJECTIVES = ["latency", "energy", "area"]


@dataclass(frozen=True)
class NSGA2RunResult:
    final_front: list[Any]
    csv_path: str | None
    objective_names: list[str]
    gene_names: list[str]
    seed: int
    num_of_generations: int
    num_of_individuals: int


def run_nsga2(
    workload_path: str,
    objective_names: list[str] | None = None,
    num_of_generations: int = 3,
    num_of_individuals: int = 10,
    seed: int = 1,
    debug: bool = False,
    save_csv: bool = True,
    results_dir: str | None = None,
) -> NSGA2RunResult:
    objective_names = list(objective_names or DEFAULT_OBJECTIVES)
    if not objective_names:
        raise ValueError("At least one objective name is required.")
    if num_of_generations < 1:
        raise ValueError("num_of_generations must be at least 1.")
    if num_of_individuals < 2:
        raise ValueError("num_of_individuals must be at least 2.")

    names = gene_names()

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    evaluator = ZigZagEvaluator(workload=workload_path, debug=debug)
    adapter = ObjectiveAdapter(evaluator, verbose=debug)
    objectives = adapter.build_objectives(objective_names)

    # NSGA-II works over numeric ranges and may emit floats. TALOS keeps the
    # genome discrete by rounding and clamping inside decode_genome().
    # I do not like this, it is what it is
    problem = Problem(
        objectives=objectives,
        num_of_variables=GENOME_LENGTH,
        variables_range=gene_bounds(),
        same_range=False,
        expand=False,
    )
    evolution = Evolution(
        problem,
        num_of_generations=num_of_generations,
        num_of_individuals=num_of_individuals,
    )

    if debug:
        final_front = evolution.evolve()
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            final_front = evolution.evolve()

    csv_path = None
    if save_csv:
        csv_path = str(
            _write_results_csv(
                final_front=final_front,
                adapter=adapter,
                objective_names=objective_names,
                names=names,
                seed=seed,
                num_of_generations=num_of_generations,
                num_of_individuals=num_of_individuals,
                results_dir=results_dir,
            )
        )

    return NSGA2RunResult(
        final_front=final_front,
        csv_path=csv_path,
        objective_names=objective_names,
        gene_names=names,
        seed=seed,
        num_of_generations=num_of_generations,
        num_of_individuals=num_of_individuals,
    )


def _write_results_csv(
    final_front: list[Any],
    adapter: ObjectiveAdapter,
    objective_names: list[str],
    names: list[str],
    seed: int,
    num_of_generations: int,
    num_of_individuals: int,
    results_dir: str | None,
) -> Path:
    output_dir = Path(results_dir) if results_dir is not None else Path.cwd() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"nsga2_results_{timestamp}.csv"

    fieldnames = [
        "solution_index",
        "raw_genome",
        "discrete_genome",
        "gene_names",
        "objective_names",
        "seed",
        "num_of_generations",
        "num_of_individuals",
        "latency",
        "energy",
        "area",
        "valid",
    ]
    fieldnames.extend(f"raw_{name}" for name in names)
    fieldnames.extend(f"code_{name}" for name in names)
    fieldnames.extend(f"objective_{name}" for name in objective_names)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, individual in enumerate(final_front):
            raw_genome = [float(value) for value in individual.features]
            discrete_genome = _discretize_genome(raw_genome)
            result = adapter.evaluate(raw_genome)

            row: dict[str, Any] = {
                "solution_index": idx,
                "raw_genome": json.dumps(raw_genome),
                "discrete_genome": json.dumps(discrete_genome),
                "gene_names": json.dumps(names),
                "objective_names": json.dumps(objective_names),
                "seed": seed,
                "num_of_generations": num_of_generations,
                "num_of_individuals": num_of_individuals,
                "latency": result.latency,
                "energy": result.energy,
                "area": result.area,
                "valid": result.valid,
            }

            row.update({f"raw_{name}": raw_genome[i] for i, name in enumerate(names)})
            row.update(
                {f"code_{name}": discrete_genome[i] for i, name in enumerate(names)}
            )
            row.update(
                {
                    f"objective_{name}": individual.objectives[i]
                    for i, name in enumerate(objective_names)
                }
            )

            writer.writerow(row)

    return csv_path


def _discretize_genome(genome: list[float]) -> list[int]:
    discrete_genome: list[int] = []

    for gene, (lower, upper) in zip(genome, gene_bounds(), strict=True):
        code = int(round(float(gene)))
        discrete_genome.append(max(lower, min(code, upper)))

    return discrete_genome


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workload = repo_root / "workloads" / "alexnet.onnx"

    result = run_nsga2(
        workload_path=str(workload),
        objective_names=DEFAULT_OBJECTIVES,
        num_of_generations=3,
        num_of_individuals=10,
        seed=1,
    )

    print(f"Final Pareto front size: {len(result.final_front)}")
    if result.csv_path is not None:
        print(f"Results CSV: {result.csv_path}")


if __name__ == "__main__":
    main()
