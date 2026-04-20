from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.config import Config
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize

from talos.architecture.genome import GENOME_LENGTH, gene_bounds, gene_names
from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import EvaluationResult, ZigZagEvaluator


DEFAULT_OBJECTIVES = ["latency", "energy", "area"]
SUPPORTED_OBJECTIVES = {"latency", "energy", "area", "edp", "eap", "alp"}
INVALID_OBJECTIVE_VALUE = float("inf")

Config.warnings["not_compiled"] = False


@dataclass(frozen=True)
class PymooRunArtifacts:
    csv_path: str | None
    objective_names: list[str]
    gene_names: list[str]
    pop_size: int
    n_gen: int
    seed: int
    n_workers: int


class TalosPymooProblem(ElementwiseProblem):
    """
    Element-wise pymoo problem for TALOS.

    The genome is encoded as catalog indices. pymoo is allowed to operate over
    numeric variables inside the bounds; TALOS keeps the discrete semantics by
    rounding and clamping in the existing genome decoder/evaluator.

    Extension point: if a future backend evaluates a whole population on GPU,
    this class can be replaced by a vectorized pymoo Problem while keeping the
    runner and CSV/export code mostly unchanged.
    """

    def __init__(
        self,
        workload_path: str,
        objective_names: list[str],
        adapter: ObjectiveAdapter | None = None,
        debug: bool = False,
        elementwise_runner: Any | None = None,
        workdir: str | None = None,
        zigzag_lpf_limit: int = 1,
        zigzag_spatial_mappings: int = 1,
    ) -> None:
        self.workload_path = workload_path
        self.objective_names = list(objective_names)
        self.debug = debug
        self.workdir = workdir
        self.zigzag_lpf_limit = zigzag_lpf_limit
        self.zigzag_spatial_mappings = zigzag_spatial_mappings
        self._adapter = adapter

        bounds = gene_bounds()
        xl = np.array([lower for lower, _upper in bounds], dtype=float)
        xu = np.array([upper for _lower, upper in bounds], dtype=float)

        problem_kwargs: dict[str, Any] = {
            "n_var": GENOME_LENGTH,
            "n_obj": len(self.objective_names),
            "xl": xl,
            "xu": xu,
        }
        if elementwise_runner is not None:
            problem_kwargs["elementwise_runner"] = elementwise_runner

        super().__init__(**problem_kwargs)

    @property
    def adapter(self) -> ObjectiveAdapter:
        if self._adapter is None:
            evaluator = ZigZagEvaluator(
                workload=self.workload_path,
                debug=self.debug,
                workdir=self._worker_workdir(),
                lpf_limit=self.zigzag_lpf_limit,
                nb_spatial_mappings_generated=self.zigzag_spatial_mappings,
            )
            self._adapter = ObjectiveAdapter(evaluator, verbose=self.debug)
        return self._adapter

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # ObjectiveAdapter/ZigZagEvaluator state may contain process-local paths.
        # Recreate it lazily inside each spawn worker.
        state["_adapter"] = None
        return state

    def _worker_workdir(self) -> str | None:
        if self.workdir is None:
            return None
        return str(Path(self.workdir) / f"worker_{os.getpid()}")

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        genome = [float(value) for value in x.tolist()]

        try:
            objectives = self.adapter.build_objectives(self.objective_names)
            values = [float(objective(genome)) for objective in objectives]
        except Exception as exc:
            if self.debug:
                print(f"pymoo evaluation failed for genome {genome}: {exc}")
            values = [INVALID_OBJECTIVE_VALUE] * len(self.objective_names)

        out["F"] = values


def run_nsga2_pymoo(
    workload_path: str,
    objective_names: list[str] | None = None,
    pop_size: int = 6,
    n_gen: int = 2,
    seed: int = 1,
    n_workers: int = 1,
    debug: bool = False,
    save_csv: bool = True,
    results_dir: str | None = None,
    zigzag_lpf_limit: int = 1,
    zigzag_spatial_mappings: int = 1,
):
    objective_names = list(objective_names or DEFAULT_OBJECTIVES)
    _validate_run_config(
        objective_names,
        pop_size,
        n_gen,
        n_workers,
        zigzag_lpf_limit,
        zigzag_spatial_mappings,
    )

    output_dir = Path(results_dir) if results_dir is not None else Path.cwd() / "results"
    workdir = output_dir / "pymoo_workdirs"

    evaluator = ZigZagEvaluator(
        workload=workload_path,
        debug=debug,
        workdir=str(workdir / "main"),
        lpf_limit=zigzag_lpf_limit,
        nb_spatial_mappings_generated=zigzag_spatial_mappings,
    )
    adapter = ObjectiveAdapter(evaluator, verbose=debug)
    adapter.build_objectives(objective_names)

    pool: Any | None = None
    try:
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=n_workers)
            runner = StarmapParallelization(pool.starmap)
            problem = TalosPymooProblem(
                workload_path=workload_path,
                objective_names=objective_names,
                adapter=adapter,
                debug=debug,
                elementwise_runner=runner,
                workdir=str(workdir),
                zigzag_lpf_limit=zigzag_lpf_limit,
                zigzag_spatial_mappings=zigzag_spatial_mappings,
            )
        else:
            problem = TalosPymooProblem(
                workload_path=workload_path,
                objective_names=objective_names,
                adapter=adapter,
                debug=debug,
                workdir=str(workdir),
                zigzag_lpf_limit=zigzag_lpf_limit,
                zigzag_spatial_mappings=zigzag_spatial_mappings,
            )

        algorithm = NSGA2(pop_size=pop_size)
        result = minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=seed,
            verbose=debug,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    csv_path = None
    if save_csv:
        csv_path = str(
            _write_results_csv(
                result=result,
                adapter=adapter,
                objective_names=objective_names,
                pop_size=pop_size,
                n_gen=n_gen,
                seed=seed,
                n_workers=n_workers,
                results_dir=results_dir,
            )
        )

    result.talos = PymooRunArtifacts(
        csv_path=csv_path,
        objective_names=objective_names,
        gene_names=gene_names(),
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        n_workers=n_workers,
    )
    return result


def _validate_run_config(
    objective_names: list[str],
    pop_size: int,
    n_gen: int,
    n_workers: int,
    zigzag_lpf_limit: int,
    zigzag_spatial_mappings: int,
) -> None:
    if not objective_names:
        raise ValueError("At least one objective name is required.")

    unknown = sorted(set(objective_names) - SUPPORTED_OBJECTIVES)
    if unknown:
        raise ValueError(f"Unknown objective name(s): {', '.join(unknown)}")

    if pop_size < 2:
        raise ValueError("pop_size must be at least 2.")
    if n_gen < 1:
        raise ValueError("n_gen must be at least 1.")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")
    if zigzag_lpf_limit < 1:
        raise ValueError("zigzag_lpf_limit must be at least 1.")
    if zigzag_spatial_mappings < 1:
        raise ValueError("zigzag_spatial_mappings must be at least 1.")


def _write_results_csv(
    result: Any,
    adapter: ObjectiveAdapter,
    objective_names: list[str],
    pop_size: int,
    n_gen: int,
    seed: int,
    n_workers: int,
    results_dir: str | None,
) -> Path:
    output_dir = Path(results_dir) if results_dir is not None else Path.cwd() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"pymoo_nsga2_results_{timestamp}.csv"
    names = gene_names()

    fieldnames = [
        "solution_index",
        "raw_genome",
        "discrete_genome",
        "gene_names",
        "objective_names",
        "pop_size",
        "n_gen",
        "seed",
        "n_workers",
        "latency",
        "energy",
        "area",
        "valid",
        "error_message",
    ]
    fieldnames.extend(f"raw_{name}" for name in names)
    fieldnames.extend(f"code_{name}" for name in names)
    fieldnames.extend(f"objective_{name}" for name in objective_names)

    genomes = _result_genomes(result)
    objective_rows = _result_objective_rows(result)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, raw_genome in enumerate(genomes):
            discrete_genome = _discretize_genome(raw_genome)
            objective_values = (
                objective_rows[idx] if idx < len(objective_rows) else None
            )
            base_result = _base_result_from_objectives(
                objective_names,
                objective_values,
            )
            if base_result is None:
                base_result = _safe_evaluate_base(adapter, raw_genome)

            row: dict[str, Any] = {
                "solution_index": idx,
                "raw_genome": json.dumps(raw_genome),
                "discrete_genome": json.dumps(discrete_genome),
                "gene_names": json.dumps(names),
                "objective_names": json.dumps(objective_names),
                "pop_size": pop_size,
                "n_gen": n_gen,
                "seed": seed,
                "n_workers": n_workers,
                "latency": base_result.latency,
                "energy": base_result.energy,
                "area": base_result.area,
                "valid": base_result.valid,
                "error_message": base_result.error_message,
            }
            row.update({f"raw_{name}": raw_genome[i] for i, name in enumerate(names)})
            row.update(
                {f"code_{name}": discrete_genome[i] for i, name in enumerate(names)}
            )
            row.update(
                {
                    f"objective_{name}": _objective_value_for_csv(
                        adapter,
                        name,
                        raw_genome,
                        objective_names,
                        objective_values,
                    )
                    for name in objective_names
                }
            )

            writer.writerow(row)

    return csv_path


def _result_genomes(result: Any) -> list[list[float]]:
    if result.X is None:
        return []

    x = np.asarray(result.X, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    return [[float(value) for value in row.tolist()] for row in x]


def _result_objective_rows(result: Any) -> list[list[float]]:
    if result.F is None:
        return []

    f = np.asarray(result.F, dtype=float)
    if f.ndim == 1:
        f = f.reshape(1, -1)

    return [[float(value) for value in row.tolist()] for row in f]


def _base_result_from_objectives(
    objective_names: list[str],
    objective_values: list[float] | None,
) -> EvaluationResult | None:
    if objective_values is None:
        return None

    values_by_name = dict(zip(objective_names, objective_values, strict=True))
    required = {"latency", "energy", "area"}
    if not required.issubset(values_by_name):
        return None

    latency = values_by_name["latency"]
    energy = values_by_name["energy"]
    area = values_by_name["area"]
    valid = all(math.isfinite(value) for value in (latency, energy, area))

    return EvaluationResult(
        latency=latency,
        energy=energy,
        area=area,
        valid=valid,
        error_message=None if valid else "Non-finite objective returned by pymoo.",
    )


def _safe_evaluate_base(
    adapter: ObjectiveAdapter,
    genome: list[float],
) -> EvaluationResult:
    try:
        return adapter.evaluate(genome)
    except Exception as exc:
        return EvaluationResult(
            latency=INVALID_OBJECTIVE_VALUE,
            energy=INVALID_OBJECTIVE_VALUE,
            area=INVALID_OBJECTIVE_VALUE,
            valid=False,
            error_message=str(exc),
        )


def _objective_value_for_csv(
    adapter: ObjectiveAdapter,
    name: str,
    genome: list[float],
    objective_names: list[str],
    objective_values: list[float] | None,
) -> float:
    if objective_values is not None and name in objective_names:
        idx = objective_names.index(name)
        if idx < len(objective_values):
            value = float(objective_values[idx])
            return value if math.isfinite(value) else INVALID_OBJECTIVE_VALUE

    return _safe_objective(adapter, name, genome)


def _safe_objective(adapter: ObjectiveAdapter, name: str, genome: list[float]) -> float:
    try:
        value = float(adapter.evaluate_objective(name, genome))
        return value if math.isfinite(value) else INVALID_OBJECTIVE_VALUE
    except Exception:
        return INVALID_OBJECTIVE_VALUE


def _discretize_genome(genome: list[float]) -> list[int]:
    discrete_genome: list[int] = []

    for gene, (lower, upper) in zip(genome, gene_bounds(), strict=True):
        code = int(round(float(gene)))
        discrete_genome.append(max(lower, min(code, upper)))

    return discrete_genome


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workload = repo_root / "workloads" / "alexnet.onnx"

    result = run_nsga2_pymoo(
        workload_path=str(workload),
        objective_names=DEFAULT_OBJECTIVES,
        pop_size=6,
        n_gen=2,
        seed=1,
        n_workers=1,
    )

    solution_count = 0 if result.X is None else len(np.atleast_2d(result.X))
    print(f"Final solution count: {solution_count}")
    if getattr(result, "talos", None) is not None and result.talos.csv_path is not None:
        print(f"Results CSV: {result.talos.csv_path}")


if __name__ == "__main__":
    main()
