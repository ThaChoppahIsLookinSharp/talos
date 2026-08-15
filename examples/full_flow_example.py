from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.evaluation.zigzag_evaluator import (
    EvaluationResult,
    mapping_objective_for_level1,
)


LEVEL1_OBJECTIVES = ["latency", "energy", "area"]
LEVEL2_OBJECTIVES = ["area", "energy", "workload_latency_s"]
SUPPORTED_LEVEL1_OBJECTIVES = ["latency", "energy", "area", "edp", "eap", "alp"]
SUPPORTED_LEVEL2_OBJECTIVES = [
    "area",
    "energy",
    "power",
    "workload_latency_s",
    "delay",
    "inv_throughput",
]
SUMMARY_FIELDNAMES = [
    "architecture_index",
    "level1_raw_genome",
    "level1_discrete_genome",
    "level1_architecture_config",
    "level1_objective_names",
    "level1_objective_values",
    "level1_latency_cycles",
    "level1_latency",
    "level1_energy",
    "level1_area_proxy",
    "zigzag_mapping_objective",
    "level2_solution_index",
    "level2_genome",
    "selected_ips",
    "covered_by_pe",
    "level2_objective_names",
    "level2_objective_values",
    "level2_area",
    "level2_power",
    "workload_energy_j",
    "layer_cycles_mapping",
    "workload_cycles_per_inference",
    "workload_latency_s",
    "workload_throughput_ips",
    "reference_frequency_mhz",
    "reference_voltage_v",
    "dram_accesses",
    "dram_energy_j",
    "physical_critical_delay",
    "selected_ip_min_throughput",
    "physical_fmax_mhz",
    "timing_margin_mhz",
    "level2_valid",
    "constraints_satisfied",
    "constraint_violations",
    "level2_strategy",
    "level2_explored_combinations",
    "level1_csv_path",
    "level2_csv_path",
]


@dataclass(frozen=True)
class Level1Candidate:
    source_index: int
    raw_genome: list[float]
    objective_values: list[float]
    discrete_genome: list[int]
    architecture_config: Any
    accelerator: Any
    activity_profile: WorkloadActivityProfile | None = None
    evaluation: EvaluationResult | None = None


def iter_level1_genomes(result: Any) -> list[list[float]]:
    return _float_rows(getattr(result, "X", None))


def iter_level1_objectives(result: Any) -> list[list[float]]:
    return _float_rows(getattr(result, "F", None))


def iter_level1_candidates(result: Any) -> tuple[list[list[float]], list[list[float]]]:
    genomes = iter_level1_genomes(result)
    objectives = iter_level1_objectives(result)
    population = getattr(result, "pop", None)
    if population is None:
        return genomes, objectives

    for genome, values, feasible in zip(
        _float_rows(population.get("X")),
        _float_rows(population.get("F")),
        _float_rows(population.get("feasible")),
        strict=True,
    ):
        if all(bool(value) for value in feasible):
            genomes.append(genome)
            objectives.append(values)
    return genomes, objectives


def _float_rows(raw: Any) -> list[list[float]]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw == []:
        return []
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list):
        if not raw:
            return []
        first = raw[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list):
            rows = raw
        else:
            rows = [raw]
        return [[float(v) for v in _as_list(row)] for row in rows]
    return [[float(raw)]]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return [value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny TALOS Level 1 -> Level 2 full-flow example.",
    )
    parser.add_argument(
        "--workload",
        default=str(REPO_ROOT / "workloads" / "alexnet.onnx"),
        help="Path to the ONNX workload.",
    )
    parser.add_argument(
        "--ip-pool",
        default=str(REPO_ROOT / "configs" / "ip_pool_synthetic_65nm.yaml"),
        help="Path to the Level 2 IP pool YAML.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results" / "full_flow"),
        help="Directory where Level 1, Level 2 and summary CSVs are written.",
    )
    parser.add_argument("--level1-pop-size", type=int, default=4)
    parser.add_argument("--level1-generations", type=int, default=1)
    parser.add_argument(
        "--level1-objectives",
        nargs="+",
        choices=SUPPORTED_LEVEL1_OBJECTIVES,
        default=LEVEL1_OBJECTIVES,
    )
    parser.add_argument("--level2-pop-size", type=int, default=4)
    parser.add_argument("--level2-generations", type=int, default=1)
    parser.add_argument(
        "--level2-objectives",
        nargs="+",
        choices=SUPPORTED_LEVEL2_OBJECTIVES,
        default=LEVEL2_OBJECTIVES,
    )
    parser.add_argument(
        "--level2-strategy",
        choices=["nsga2", "exhaustive"],
        default="nsga2",
    )
    parser.add_argument(
        "--level2-exhaustive-max-combinations",
        type=int,
        default=100_000,
    )
    parser.add_argument("--max-architectures", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-area-mm2", type=float, default=None)
    parser.add_argument("--max-power-w", type=float, default=None)
    parser.add_argument("--max-latency-cycles", type=float, default=None)
    parser.add_argument("--min-frequency-mhz", type=float, default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    workload = Path(args.workload).expanduser().resolve()
    ip_pool_path = Path(args.ip_pool).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    print("=== TALOS full flow example ===")
    print(f"Workload: {workload}")
    print(f"IP pool: {ip_pool_path}")
    print(f"Results dir: {results_dir}")
    print()
    sys.stdout.flush()

    if not workload.exists():
        print(f"ERROR: workload file does not exist: {workload}", file=sys.stderr)
        return 2
    if not ip_pool_path.exists():
        print(f"ERROR: IP pool file does not exist: {ip_pool_path}", file=sys.stderr)
        return 2
    if args.max_architectures < 1:
        print("ERROR: --max-architectures must be at least 1.", file=sys.stderr)
        return 2

    constraints = UserConstraints(
        max_area_mm2=args.max_area_mm2,
        max_power_w=args.max_power_w,
        max_latency_cycles=args.max_latency_cycles,
        min_frequency_mhz=args.min_frequency_mhz,
    )

    try:
        from talos.architecture.genome import decode_genome, gene_bounds
        from talos.architecture.level1_importer import (
            abstract_accelerator_from_level1_config,
        )
        from talos.evaluation.cacti_costs import (
            characterize_level1_energy,
            resolve_dram_ip,
            write_energy_calibration,
        )
        from talos.evaluation.zigzag_evaluator import ZigZagEvaluator
        from talos.ga.pymoo_runner import run_nsga2_pymoo
        from talos.ip import IPPool
        from talos.level2.runner import run_level2
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(
            "ERROR: missing dependency while importing TALOS flow modules: "
            f"{missing}. Install the project dependencies, for example with "
            "`python -m pip install -r requirements.txt`. pymoo is required "
            "for this example.",
            file=sys.stderr,
        )
        return 2

    pool = IPPool.from_yaml(ip_pool_path)
    print(
        "[Calibration] Characterizing Level 1 energy with CACTI at "
        f"{pool.technology_nm:g} nm..."
    )
    try:
        energy_calibration = characterize_level1_energy(
            technology_nm=pool.technology_nm,
        )
        dram_ip = resolve_dram_ip(
            pool,
            energy_calibration,
        )
        pool = IPPool(
            [
                ip
                for ip in pool.ip_blocks
                if ip.type != "dram"
            ]
            + [dram_ip],
            technology_nm=pool.technology_nm,
        )
        calibration_path = write_energy_calibration(
            results_dir / "energy_calibration.json",
            energy_calibration,
            dram_bus_width_bits=dram_ip.bandwidth_bits,
            dram_power_model=dram_ip.power_model,
        )
    except Exception as exc:
        print(f"ERROR: Level 1 energy calibration failed: {exc}", file=sys.stderr)
        return 2
    print(f"[Calibration] JSON: {calibration_path}")
    print(f"[Calibration] DRAM: {dram_ip.id}")
    dram_accesses_per_cycle = float(
        (dram_ip.metadata or {})["accesses_per_cycle"]
    )
    activity_evaluator = ZigZagEvaluator(
        workload=str(workload),
        opt=mapping_objective_for_level1(args.level1_objectives),
        debug=args.debug,
        workdir=str(results_dir / "level1_profiles"),
        lpf_limit=1,
        nb_spatial_mappings_generated=1,
        dram_bandwidth_bits=dram_ip.bandwidth_bits,
        dram_accesses_per_cycle=dram_accesses_per_cycle,
        dram_power_model=dram_ip.power_model,
        energy_calibration=energy_calibration,
    )

    print("[Level 1] Running small architecture exploration...")
    try:
        level1_result = run_nsga2_pymoo(
            workload_path=str(workload),
            objective_names=args.level1_objectives,
            pop_size=args.level1_pop_size,
            n_gen=args.level1_generations,
            seed=args.seed,
            n_workers=args.workers,
            debug=args.debug,
            save_csv=True,
            results_dir=str(results_dir / "level1"),
            zigzag_lpf_limit=1,
            zigzag_spatial_mappings=1,
            constraints=constraints,
            dram_bandwidth_bits=dram_ip.bandwidth_bits,
            dram_accesses_per_cycle=dram_accesses_per_cycle,
            dram_power_model=dram_ip.power_model,
            energy_calibration=energy_calibration,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(
            "ERROR: missing dependency while running Level 1: "
            f"{missing}. Install the project dependencies, for example with "
            "`python -m pip install -r requirements.txt`. pymoo is required "
            "for this example.",
            file=sys.stderr,
        )
        return 2

    level1_genomes, level1_objectives = iter_level1_candidates(level1_result)
    level1_csv_path = _level1_csv_path(level1_result)

    print(
        f"[Level 1] Considering {len(level1_genomes)} Pareto/final-population "
        "candidate row(s)."
    )
    if level1_csv_path:
        print(f"[Level 1] CSV: {level1_csv_path}")
    if not level1_genomes:
        summary_path = write_summary_csv(results_dir, [])
        print("[Level 1] No candidate architectures were returned; stopping.")
        print(f"Summary CSV written to: {summary_path}")
        return 0

    flow_failures: list[str] = []
    candidates = select_level1_candidates(
        level1_genomes=level1_genomes,
        level1_objectives=level1_objectives,
        level1_objective_names=args.level1_objectives,
        max_architectures=args.max_architectures,
        pool=pool,
        decode_genome=decode_genome,
        gene_bounds=gene_bounds,
        abstract_accelerator_from_level1_config=(
            abstract_accelerator_from_level1_config
        ),
        constraints=constraints,
        evaluate_activity=activity_evaluator.evaluate,
        exhaustive_max_combinations=args.level2_exhaustive_max_combinations,
        failures=flow_failures,
    )
    print(
        f"[Level 1] Passing {len(candidates)} architecture(s) to Level 2."
    )
    print()
    if not candidates:
        summary_path = write_summary_csv(results_dir, [])
        print("No Level 1 architecture was compatible with the IP pool.")
        print(f"Summary CSV written to: {summary_path}")
        return 1 if flow_failures else 0

    summary_rows: list[dict[str, Any]] = []
    level2_failures = len(flow_failures)
    for candidate in candidates:
        arch_index = candidate.source_index
        genome = candidate.raw_genome
        print(f"[Level 2] Architecture {arch_index}")
        print(f"  Level 1 genome: {genome}")

        component_summary = [
            f"{component.name}:{component.type}x{component.count}"
            for component in candidate.accelerator.components
        ]
        print(f"  Abstract components: {', '.join(component_summary)}")
        print(f"  Running physical IP selection ({args.level2_strategy})...")

        try:
            level2_result = run_level2(
                accelerator=candidate.accelerator,
                ip_pool=pool,
                objective_names=args.level2_objectives,
                strategy=args.level2_strategy,
                pop_size=args.level2_pop_size,
                n_gen=args.level2_generations,
                seed=args.seed,
                save_csv=True,
                results_dir=str(results_dir / f"level2_arch_{arch_index}"),
                debug=args.debug,
                constraints=constraints,
                activity_profile=candidate.activity_profile,
                exhaustive_max_combinations=args.level2_exhaustive_max_combinations,
            )
        except Exception as exc:
            level2_failures += 1
            print(
                f"  Level 2 failed for this architecture: {exc}",
                file=sys.stderr,
            )
            print()
            continue

        print(
            f"  Found {len(level2_result.solutions)} physical implementation(s)."
        )
        print(f"  CSV: {level2_result.csv_path}")

        summary_rows.extend(
            build_summary_rows(
                architecture_index=arch_index,
                level1_raw_genome=genome,
                level1_discrete_genome=candidate.discrete_genome,
                level1_architecture_config=asdict(candidate.architecture_config),
                level1_objective_values=candidate.objective_values,
                level1_objective_names=args.level1_objectives,
                level2_objective_names=args.level2_objectives,
                level1_csv_path=level1_csv_path,
                level2_csv_path=level2_result.csv_path,
                level2_solutions=level2_result.solutions,
                constraints=constraints,
                level1_evaluation=candidate.evaluation,
            )
        )

        if level2_result.solutions:
            print_first_solution(level2_result.solutions[0])
        print()

    summary_path = write_summary_csv(results_dir, summary_rows)
    if not summary_rows:
        print("No combined Level 1 -> Level 2 rows were produced.")
    print(f"Summary CSV written to: {summary_path}")
    return 1 if level2_failures else 0


def _level1_csv_path(result: Any) -> str:
    talos_artifacts = getattr(result, "talos", None)
    csv_path = getattr(talos_artifacts, "csv_path", None)
    return "" if csv_path is None else str(csv_path)


def select_level1_candidates(
    *,
    level1_genomes: list[list[float]],
    level1_objectives: list[list[float]],
    level1_objective_names: list[str],
    max_architectures: int,
    pool: Any,
    decode_genome: Any,
    gene_bounds: Any,
    abstract_accelerator_from_level1_config: Any,
    constraints: UserConstraints | None = None,
    evaluate_activity: Any | None = None,
    exhaustive_max_combinations: int = 100_000,
    failures: list[str] | None = None,
) -> list[Level1Candidate]:
    candidates: list[Level1Candidate] = []
    seen_discrete_genomes: set[tuple[int, ...]] = set()
    bounds = gene_bounds()

    candidate_order = _level1_candidate_order(
        level1_objectives,
        len(level1_genomes),
    )
    for source_index in candidate_order:
        genome = level1_genomes[source_index]
        if len(candidates) >= max_architectures:
            break
        discrete_genome = discretize_genome(genome, bounds)
        discrete_key = tuple(discrete_genome)
        if discrete_key in seen_discrete_genomes:
            continue
        try:
            config = decode_genome(genome)
            accelerator = abstract_accelerator_from_level1_config(config)
            compatibility_error = first_ip_compatibility_error(accelerator, pool)
        except Exception as exc:
            print(f"[Level 1] Skipping architecture {source_index}: {exc}")
            if failures is not None:
                failures.append(str(exc))
            continue

        if compatibility_error:
            print(
                f"[Level 1] Skipping architecture {source_index}: "
                f"{compatibility_error}"
            )
            if failures is not None:
                failures.append(compatibility_error)
            continue

        objective_values = (
            level1_objectives[source_index]
            if source_index < len(level1_objectives)
            else []
        )
        constraint_error = first_constraint_feasibility_error(
            constraints=constraints,
            level1_objective_names=level1_objective_names,
            objective_values=objective_values,
        )
        if constraint_error:
            print(
                f"[Level 1] Skipping architecture {source_index}: "
                f"{constraint_error}"
            )
            continue

        activity_result = None
        activity_profile = None
        if evaluate_activity is not None:
            activity_result = evaluate_activity(genome)
            if not activity_result.valid or activity_result.activity_profile is None:
                print(
                    f"[Level 1] Skipping architecture {source_index}: "
                    f"{activity_result.error_message or 'activity profile is unavailable'}"
                )
                if failures is not None:
                    failures.append(
                        activity_result.error_message
                        or "activity profile is unavailable"
                    )
                continue
            activity_profile = activity_result.activity_profile

        if constraints is not None and constraints.level2_constraint_count:
            try:
                from talos.level2.genome import Level2GenomeSpec
                from talos.level2.exhaustive_runner import run_level2_exhaustive

                combination_count = Level2GenomeSpec.from_accelerator_and_pool(
                    accelerator,
                    pool,
                ).genome_count()
                physical_result = (
                    None
                    if combination_count > exhaustive_max_combinations
                    else run_level2_exhaustive(
                        accelerator=accelerator,
                        ip_pool=pool,
                        objective_names=["area"],
                        constraints=constraints,
                        activity_profile=activity_profile,
                        max_combinations=exhaustive_max_combinations,
                        save_csv=False,
                    )
                )
            except Exception as exc:
                print(f"[Level 1] Skipping architecture {source_index}: {exc}")
                if failures is not None:
                    failures.append(str(exc))
                continue
            if physical_result is None:
                print(
                    f"[Level 1] Physical prefilter skipped for architecture "
                    f"{source_index}: {combination_count} combinations exceed "
                    f"the {exhaustive_max_combinations} limit"
                )
            elif not physical_result.solutions:
                print(
                    f"[Level 1] Skipping architecture {source_index}: "
                    "no Level 2 combination satisfies the physical constraints"
                )
                continue

        seen_discrete_genomes.add(discrete_key)
        candidates.append(
            Level1Candidate(
                source_index=source_index,
                raw_genome=genome,
                objective_values=objective_values,
                discrete_genome=discrete_genome,
                architecture_config=config,
                accelerator=accelerator,
                activity_profile=activity_profile,
                evaluation=activity_result,
            )
        )

    return candidates


def _level1_candidate_order(
    objective_values: list[list[float]],
    candidate_count: int,
) -> list[int]:
    import numpy as np
    from pymoo.operators.survival.rank_and_crowding.metrics import (
        calc_crowding_distance,
    )
    from pymoo.util.nds.non_dominated_sorting import (
        NonDominatedSorting,
    )

    objective_row_count = min(candidate_count, len(objective_values))
    finite = [
        index
        for index in range(objective_row_count)
        if objective_values[index]
        and np.isfinite(objective_values[index]).all()
    ]
    invalid = [
        index
        for index in range(candidate_count)
        if index not in finite
    ]
    if not finite:
        return invalid
    values = np.asarray(
        [objective_values[index] for index in finite],
        dtype=float,
    )
    if values.shape[1] == 1:
        return sorted(
            finite,
            key=lambda index: (objective_values[index][0], index),
        ) + invalid

    ordered: list[int] = []
    for front in NonDominatedSorting().do(values):
        crowding = calc_crowding_distance(values[front])
        ordered.extend(
            finite[front[position]]
            for position in sorted(
                range(len(front)),
                key=lambda position: (
                    -crowding[position],
                    values[front[position]].sum(),
                    finite[front[position]],
                ),
            )
        )
    return ordered + invalid


def first_constraint_feasibility_error(
    *,
    constraints: UserConstraints | None,
    level1_objective_names: list[str],
    objective_values: list[float],
) -> str:
    if constraints is None:
        return ""

    level1_values = dict(zip(level1_objective_names, objective_values))
    latency_cycles = level1_values.get("latency")
    if latency_cycles is not None:
        violations = constraints.level1_violations(float(latency_cycles))
        if violations:
            return "; ".join(violations)

    return ""


def first_ip_compatibility_error(accelerator: Any, pool: Any) -> str:
    for component in accelerator.components:
        try:
            pool.find_compatible(component)
        except ValueError as exc:
            return str(exc)
    return ""


def discretize_genome(
    genome: list[float],
    bounds: list[tuple[int, int]],
) -> list[int]:
    discrete: list[int] = []
    for gene, (lower, upper) in zip(genome, bounds, strict=True):
        code = int(round(float(gene)))
        discrete.append(max(lower, min(code, upper)))
    return discrete


def build_summary_rows(
    *,
    architecture_index: int,
    level1_raw_genome: list[float],
    level1_discrete_genome: list[int],
    level1_architecture_config: dict[str, Any],
    level1_objective_values: list[float],
    level1_objective_names: list[str],
    level2_objective_names: list[str],
    level1_csv_path: str,
    level2_csv_path: Path | None,
    level2_solutions: list[dict[str, Any]],
    constraints: UserConstraints | None,
    level1_evaluation: EvaluationResult | None = None,
) -> list[dict[str, Any]]:
    if not level2_solutions:
        return []

    level1_by_name = dict(zip(level1_objective_names, level1_objective_values))
    if level1_evaluation is not None and level1_evaluation.valid:
        level1_by_name.update(
            {
                "latency": level1_evaluation.latency,
                "energy": level1_evaluation.energy,
                "area": level1_evaluation.area,
            }
        )
    latency_cycles = level1_by_name.get("latency", "")
    area_proxy = level1_by_name.get("area", "")
    rows: list[dict[str, Any]] = []
    for solution in level2_solutions:
        workload_latency_s = solution.get("workload_latency_s")
        constraint_violations = _combined_constraint_violations(
            constraints=constraints,
            latency_cycles=latency_cycles,
            level2_violations=solution.get("constraint_violations", []),
        )
        constraints_satisfied = (
            bool(solution.get("valid", False))
            and not constraint_violations
        )
        inference_rate = (
            solution.get("workload_throughput_ips")
            if constraints_satisfied
            else None
        )
        rows.append(
            {
                "architecture_index": architecture_index,
                "level1_raw_genome": level1_raw_genome,
                "level1_discrete_genome": level1_discrete_genome,
                "level1_architecture_config": level1_architecture_config,
                "level1_objective_names": level1_objective_names,
                "level1_objective_values": level1_objective_values,
                "level1_latency_cycles": latency_cycles,
                "level1_latency": level1_by_name.get("latency", ""),
                "level1_energy": level1_by_name.get("energy", ""),
                "level1_area_proxy": area_proxy,
                "zigzag_mapping_objective": (
                    ""
                    if level1_evaluation is None
                    else level1_evaluation.mapping_objective or ""
                ),
                "level2_solution_index": solution.get("solution_index", ""),
                "level2_genome": solution.get("genome", ""),
                "selected_ips": solution.get("selected_ips", ""),
                "covered_by_pe": solution.get("covered_by_pe", ""),
                "level2_objective_names": solution.get(
                    "objective_names",
                    level2_objective_names,
                ),
                "level2_objective_values": solution.get("objective_values", ""),
                "level2_area": solution.get("area", ""),
                "level2_power": solution.get("power", ""),
                "workload_energy_j": solution.get("workload_energy_j", ""),
                "layer_cycles_mapping": solution.get(
                    "layer_cycles_mapping",
                    "",
                ),
                "workload_cycles_per_inference": solution.get(
                    "workload_cycles_per_inference",
                    "",
                ),
                "workload_latency_s": (
                    "" if workload_latency_s is None else workload_latency_s
                ),
                "workload_throughput_ips": solution.get(
                    "workload_throughput_ips",
                    "",
                ),
                "reference_frequency_mhz": solution.get(
                    "reference_frequency_mhz",
                    "",
                ),
                "reference_voltage_v": solution.get(
                    "reference_voltage_v",
                    "",
                ),
                "dram_accesses": solution.get("dram_accesses", ""),
                "dram_energy_j": solution.get(
                    "dram_energy_j",
                    "",
                ),
                "physical_critical_delay": solution.get(
                    "physical_critical_delay",
                    "",
                ),
                "selected_ip_min_throughput": solution.get(
                    "selected_ip_min_throughput",
                    "",
                ),
                "physical_fmax_mhz": solution.get("physical_fmax_mhz", ""),
                "timing_margin_mhz": solution.get("timing_margin_mhz", ""),
                "level2_valid": solution.get("valid", ""),
                "constraints_satisfied": constraints_satisfied,
                "constraint_violations": constraint_violations,
                "level2_strategy": solution.get("strategy", ""),
                "level2_explored_combinations": solution.get(
                    "explored_combinations",
                    "",
                ),
                "level1_csv_path": level1_csv_path,
                "level2_csv_path": "" if level2_csv_path is None else str(level2_csv_path),
            }
        )
    return rows


def _combined_constraint_violations(
    *,
    constraints: UserConstraints | None,
    latency_cycles: Any,
    level2_violations: Any,
) -> list[str]:
    violations: list[str] = []
    if constraints is not None and latency_cycles != "":
        violations.extend(constraints.level1_violations(float(latency_cycles)))
    if isinstance(level2_violations, str):
        violations.append(level2_violations)
    else:
        violations.extend(level2_violations or [])
    return violations


def write_summary_csv(results_dir: Path, rows: list[dict[str, Any]]) -> Path:
    summary_path = results_dir / "full_flow_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: csv_value(row.get(field, ""))
                    for field in SUMMARY_FIELDNAMES
                }
            )
    return summary_path


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def print_first_solution(solution: dict[str, Any]) -> None:
    print("  First solution:")
    print(f"    area: {solution.get('area')}")
    print(f"    power: {solution.get('power')}")
    print(f"    workload_energy_j: {solution.get('workload_energy_j')}")
    print(f"    layer_cycles_mapping: {solution.get('layer_cycles_mapping')}")
    print(
        "    workload_cycles_per_inference: "
        f"{solution.get('workload_cycles_per_inference')}"
    )
    print(f"    workload_latency_s: {solution.get('workload_latency_s')}")
    print(f"    workload_throughput_ips: {solution.get('workload_throughput_ips')}")
    print(f"    reference_frequency_mhz: {solution.get('reference_frequency_mhz')}")
    print(f"    reference_voltage_v: {solution.get('reference_voltage_v')}")
    print(f"    dram_accesses: {solution.get('dram_accesses')}")
    print(f"    dram_energy_j: {solution.get('dram_energy_j')}")
    print(f"    physical_critical_delay: {solution.get('physical_critical_delay')}")
    print(f"    physical_fmax_mhz: {solution.get('physical_fmax_mhz')}")
    print(f"    timing_margin_mhz: {solution.get('timing_margin_mhz')}")
    print(f"    constraints_satisfied: {solution.get('constraints_satisfied')}")
    print(f"    selected IPs: {solution.get('selected_ips')}")
    print(f"    RFs covered by PE: {solution.get('covered_by_pe')}")


if __name__ == "__main__":
    raise SystemExit(main())
