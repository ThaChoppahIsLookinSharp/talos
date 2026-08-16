from __future__ import annotations

import argparse
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
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from talos.architecture.genome import (
    DEFAULT_DRAM_BW_BITS,
    GENOME_LENGTH,
    gene_bounds,
    gene_names,
)
from talos.constraints import UserConstraints
from talos.evaluation.area_calibration import Level1AreaCalibration
from talos.evaluation.cacti_costs import (
    Level1EnergyCalibration,
    characterize_level1_energy,
    write_energy_calibration,
)
from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import (
    DEFAULT_DRAM_ACCESSES_PER_CYCLE,
    DEFAULT_DRAM_POWER_MODEL,
    ZigZagEvaluator,
    mapping_objective_for_level1,
)
from talos.ip.ip_characterization import PowerCharacterization


DEFAULT_OBJECTIVES = ["latency", "energy", "area"]
SUPPORTED_OBJECTIVES = {"latency", "energy", "area", "edp", "eap", "alp"}
INVALID_OBJECTIVE_VALUE = float("inf")

Config.warnings["not_compiled"] = False


@dataclass(frozen=True)
class PymooRunArtifacts:
    csv_path: str | None
    energy_calibration_path: str
    objective_names: list[str]
    gene_names: list[str]
    pop_size: int
    n_gen: int
    seed: int
    n_workers: int
    zigzag_mapping_objective: str


class TalosPymooProblem(ElementwiseProblem):
    """
    Element-wise pymoo problem for TALOS.

    The genome is encoded as integer catalog indices. Sampling, crossover, and
    mutation keep those indices discrete so pymoo can eliminate duplicate
    architectures before evaluation.

    Extension point: if a future backend evaluates a whole population on GPU,
    this class can be replaced by a vectorized pymoo Problem while keeping the
    runner and CSV/export code mostly unchanged.
    """

    def __init__(
        self,
        workload_path: str,
        objective_names: list[str],
        area_calibration: Level1AreaCalibration,
        adapter: ObjectiveAdapter | None = None,
        debug: bool = False,
        elementwise_runner: Any | None = None,
        workdir: str | None = None,
        zigzag_lpf_limit: int = 1,
        zigzag_spatial_mappings: int = 1,
        constraints: UserConstraints | None = None,
        dram_bandwidth_bits: int = DEFAULT_DRAM_BW_BITS,
        dram_accesses_per_cycle: float = DEFAULT_DRAM_ACCESSES_PER_CYCLE,
        dram_power_model: PowerCharacterization = DEFAULT_DRAM_POWER_MODEL,
        energy_calibration: Level1EnergyCalibration | None = None,
    ) -> None:
        self.workload_path = workload_path
        self.objective_names = list(objective_names)
        self.zigzag_mapping_objective = mapping_objective_for_level1(
            self.objective_names
        )
        self.debug = debug
        self.workdir = workdir
        self.zigzag_lpf_limit = zigzag_lpf_limit
        self.zigzag_spatial_mappings = zigzag_spatial_mappings
        self.constraints = constraints
        self.dram_bandwidth_bits = dram_bandwidth_bits
        self.dram_accesses_per_cycle = dram_accesses_per_cycle
        self.dram_power_model = dram_power_model
        self.energy_calibration = energy_calibration
        self.area_calibration = area_calibration
        self._adapter = adapter

        bounds = gene_bounds()
        xl = np.array([lower for lower, _upper in bounds], dtype=float)
        xu = np.array([upper for _lower, upper in bounds], dtype=float)

        problem_kwargs: dict[str, Any] = {
            "n_var": GENOME_LENGTH,
            "n_obj": len(self.objective_names),
            "n_ieq_constr": len(self._constraint_values(INVALID_OBJECTIVE_VALUE)),
            "vtype": int,
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
                opt=self.zigzag_mapping_objective,
                debug=self.debug,
                workdir=self._worker_workdir(),
                lpf_limit=self.zigzag_lpf_limit,
                nb_spatial_mappings_generated=self.zigzag_spatial_mappings,
                dram_bandwidth_bits=self.dram_bandwidth_bits,
                dram_accesses_per_cycle=self.dram_accesses_per_cycle,
                dram_power_model=self.dram_power_model,
                energy_calibration=self.energy_calibration,
                area_calibration=self.area_calibration,
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
            result = self.adapter.evaluate(genome)
            constraint_values = self._constraint_values(result.latency)
        except Exception as exc:
            if self.debug:
                print(f"pymoo evaluation failed for genome {genome}: {exc}")
            values = [INVALID_OBJECTIVE_VALUE] * len(self.objective_names)
            constraint_values = self._constraint_values(INVALID_OBJECTIVE_VALUE)

        out["F"] = values
        if constraint_values:
            out["G"] = constraint_values

    def _constraint_values(self, latency_cycles: float) -> list[float]:
        if self.constraints is None:
            return []
        return self.constraints.level1_constraint_values(latency_cycles)


def run_nsga2_pymoo(
    workload_path: str,
    area_calibration: Level1AreaCalibration,
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
    constraints: UserConstraints | None = None,
    dram_bandwidth_bits: int = DEFAULT_DRAM_BW_BITS,
    dram_accesses_per_cycle: float = DEFAULT_DRAM_ACCESSES_PER_CYCLE,
    dram_power_model: PowerCharacterization = DEFAULT_DRAM_POWER_MODEL,
    energy_calibration: Level1EnergyCalibration | None = None,
):
    objective_names = list(objective_names or DEFAULT_OBJECTIVES)
    zigzag_mapping_objective = mapping_objective_for_level1(objective_names)
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
    energy_calibration = energy_calibration or characterize_level1_energy()

    evaluator = ZigZagEvaluator(
        workload=workload_path,
        opt=zigzag_mapping_objective,
        debug=debug,
        workdir=str(workdir / "main"),
        lpf_limit=zigzag_lpf_limit,
        nb_spatial_mappings_generated=zigzag_spatial_mappings,
        dram_bandwidth_bits=dram_bandwidth_bits,
        dram_accesses_per_cycle=dram_accesses_per_cycle,
        dram_power_model=dram_power_model,
        energy_calibration=energy_calibration,
        area_calibration=area_calibration,
    )
    calibration_path = write_energy_calibration(
        output_dir / "energy_calibration.json",
        energy_calibration,
        dram_bus_width_bits=dram_bandwidth_bits,
        dram_power_model=evaluator.dram_power_model,
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
                constraints=constraints,
                dram_bandwidth_bits=dram_bandwidth_bits,
                dram_accesses_per_cycle=dram_accesses_per_cycle,
                dram_power_model=dram_power_model,
                energy_calibration=energy_calibration,
                area_calibration=area_calibration,
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
                constraints=constraints,
                dram_bandwidth_bits=dram_bandwidth_bits,
                dram_accesses_per_cycle=dram_accesses_per_cycle,
                dram_power_model=dram_power_model,
                energy_calibration=energy_calibration,
                area_calibration=area_calibration,
            )

        algorithm = _build_nsga2(pop_size)
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
                constraints=constraints,
            )
        )

    result.talos = PymooRunArtifacts(
        csv_path=csv_path,
        energy_calibration_path=str(calibration_path),
        objective_names=objective_names,
        gene_names=gene_names(),
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        n_workers=n_workers,
        zigzag_mapping_objective=zigzag_mapping_objective,
    )
    return result


def _build_nsga2(pop_size: int) -> NSGA2:
    return NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
    )


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
    constraints: UserConstraints | None,
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
        "zigzag_mapping_objective",
        "latency_cycles",
        "latency",
        "energy",
        "physical_area_mm2",
        "valid",
        "constraints_satisfied",
        "constraint_violations",
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
            objective_values_by_name = dict(zip(objective_names, objective_values or []))
            latency = objective_values_by_name.get("latency", "")
            energy = objective_values_by_name.get("energy", "")
            area = objective_values_by_name.get("area", "")
            constraint_violations = (
                []
                if constraints is None or latency == ""
                else constraints.level1_violations(float(latency))
            )
            objectives_valid = objective_values is not None and all(
                math.isfinite(float(value)) for value in objective_values
            )
            valid = objectives_valid and not constraint_violations

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
                "zigzag_mapping_objective": mapping_objective_for_level1(
                    objective_names
                ),
                "latency_cycles": latency,
                "latency": latency,
                "energy": energy,
                "physical_area_mm2": area,
                "valid": valid,
                "constraints_satisfied": valid,
                "constraint_violations": json.dumps(constraint_violations),
                "error_message": "" if objectives_valid else "Non-finite objective returned by pymoo.",
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

    return INVALID_OBJECTIVE_VALUE


def _discretize_genome(genome: list[float]) -> list[int]:
    discrete_genome: list[int] = []

    for gene, (lower, upper) in zip(genome, gene_bounds(), strict=True):
        code = int(round(float(gene)))
        discrete_genome.append(max(lower, min(code, upper)))

    return discrete_genome


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a small pymoo NSGA-II demo.")
    parser.add_argument(
        "--ip-pool",
        type=Path,
        required=True,
        help="Path to the characterized IP pool used for Level 1 area.",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=repo_root / "workloads" / "alexnet.onnx",
        help="Path to the ONNX workload.",
    )
    args = parser.parse_args()
    workload = args.workload.expanduser().resolve()
    from talos.evaluation.area_calibration import characterize_level1_area
    from talos.evaluation.cacti_costs import characterize_level1_energy
    from talos.ip import IPPool

    ip_pool = IPPool.from_yaml(args.ip_pool.expanduser().resolve())
    area_calibration = characterize_level1_area(ip_pool)
    energy_calibration = characterize_level1_energy(
        technology_nm=ip_pool.technology_nm,
    )

    result = run_nsga2_pymoo(
        workload_path=str(workload),
        area_calibration=area_calibration,
        objective_names=DEFAULT_OBJECTIVES,
        pop_size=6,
        n_gen=2,
        seed=1,
        n_workers=1,
        energy_calibration=energy_calibration,
    )

    solution_count = 0 if result.X is None else len(np.atleast_2d(result.X))
    print(f"Final solution count: {solution_count}")
    if getattr(result, "talos", None) is not None and result.talos.csv_path is not None:
        print(f"Results CSV: {result.talos.csv_path}")


if __name__ == "__main__":
    main()
