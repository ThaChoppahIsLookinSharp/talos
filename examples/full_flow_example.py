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


LEVEL1_OBJECTIVES = ["latency", "energy", "area"]
LEVEL2_OBJECTIVES = ["area", "power", "delay", "inv_throughput"]
SUMMARY_FIELDNAMES = [
    "architecture_index",
    "level1_raw_genome",
    "level1_discrete_genome",
    "level1_architecture_config",
    "level1_objective_names",
    "level1_objective_values",
    "level1_latency",
    "level1_energy",
    "level1_area",
    "level2_solution_index",
    "level2_genome",
    "selected_ips",
    "level2_objective_names",
    "level2_objective_values",
    "level2_area",
    "level2_power",
    "level2_delay",
    "level2_throughput",
    "level2_valid",
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


def iter_level1_genomes(result: Any) -> list[list[float]]:
    return _float_rows(getattr(result, "X", None))


def iter_level1_objectives(result: Any) -> list[list[float]]:
    return _float_rows(getattr(result, "F", None))


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


def parse_args() -> argparse.Namespace:
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
        default=str(REPO_ROOT / "configs" / "ip_pool_example.yaml"),
        help="Path to the Level 2 IP pool YAML.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results" / "full_flow"),
        help="Directory where Level 1, Level 2 and summary CSVs are written.",
    )
    parser.add_argument("--level1-pop-size", type=int, default=4)
    parser.add_argument("--level1-generations", type=int, default=1)
    parser.add_argument("--level2-pop-size", type=int, default=4)
    parser.add_argument("--level2-generations", type=int, default=1)
    parser.add_argument("--max-architectures", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


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

    try:
        from talos.architecture.genome import decode_genome, gene_bounds
        from talos.architecture.level1_importer import (
            abstract_accelerator_from_level1_config,
        )
        from talos.ga.pymoo_runner import run_nsga2_pymoo
        from talos.ip import IPPool
        from talos.level2.runner import run_level2_nsga2
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

    print("[Level 1] Running small architecture exploration...")
    try:
        level1_result = run_nsga2_pymoo(
            workload_path=str(workload),
            objective_names=LEVEL1_OBJECTIVES,
            pop_size=args.level1_pop_size,
            n_gen=args.level1_generations,
            seed=args.seed,
            n_workers=1,
            debug=args.debug,
            save_csv=True,
            results_dir=str(results_dir / "level1"),
            zigzag_lpf_limit=1,
            zigzag_spatial_mappings=1,
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

    level1_genomes = iter_level1_genomes(level1_result)
    level1_objectives = iter_level1_objectives(level1_result)
    level1_csv_path = _level1_csv_path(level1_result)

    print(f"[Level 1] Found {len(level1_genomes)} candidate architecture(s).")
    if level1_csv_path:
        print(f"[Level 1] CSV: {level1_csv_path}")
    if not level1_genomes:
        summary_path = write_summary_csv(results_dir, [])
        print("[Level 1] No candidate architectures were returned; stopping.")
        print(f"Summary CSV written to: {summary_path}")
        return 1

    candidates = select_level1_candidates(
        level1_genomes=level1_genomes,
        level1_objectives=level1_objectives,
        max_architectures=args.max_architectures,
        pool=pool,
        decode_genome=decode_genome,
        gene_bounds=gene_bounds,
        abstract_accelerator_from_level1_config=(
            abstract_accelerator_from_level1_config
        ),
    )
    print(
        f"[Level 1] Passing {len(candidates)} architecture(s) to Level 2."
    )
    print()
    if not candidates:
        summary_path = write_summary_csv(results_dir, [])
        print("No Level 1 architecture was compatible with the IP pool.")
        print(f"Summary CSV written to: {summary_path}")
        return 1

    summary_rows: list[dict[str, Any]] = []
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
        print("  Running physical IP selection...")

        try:
            level2_result = run_level2_nsga2(
                accelerator=candidate.accelerator,
                ip_pool=pool,
                objective_names=LEVEL2_OBJECTIVES,
                pop_size=args.level2_pop_size,
                n_gen=args.level2_generations,
                seed=args.seed,
                save_csv=True,
                results_dir=str(results_dir / f"level2_arch_{arch_index}"),
                debug=args.debug,
            )
        except Exception as exc:
            print(f"  Level 2 failed for this architecture: {exc}")
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
                level1_csv_path=level1_csv_path,
                level2_csv_path=level2_result.csv_path,
                level2_solutions=level2_result.solutions,
            )
        )

        if level2_result.solutions:
            print_first_solution(level2_result.solutions[0])
        print()

    summary_path = write_summary_csv(results_dir, summary_rows)
    if not summary_rows:
        print("No combined Level 1 -> Level 2 rows were produced.")
    print(f"Summary CSV written to: {summary_path}")
    return 0


def _level1_csv_path(result: Any) -> str:
    talos_artifacts = getattr(result, "talos", None)
    csv_path = getattr(talos_artifacts, "csv_path", None)
    return "" if csv_path is None else str(csv_path)


def select_level1_candidates(
    *,
    level1_genomes: list[list[float]],
    level1_objectives: list[list[float]],
    max_architectures: int,
    pool: Any,
    decode_genome: Any,
    gene_bounds: Any,
    abstract_accelerator_from_level1_config: Any,
) -> list[Level1Candidate]:
    candidates: list[Level1Candidate] = []
    bounds = gene_bounds()

    for source_index, genome in enumerate(level1_genomes):
        if len(candidates) >= max_architectures:
            break
        try:
            config = decode_genome(genome)
            accelerator = abstract_accelerator_from_level1_config(config)
            compatibility_error = first_ip_compatibility_error(accelerator, pool)
        except Exception as exc:
            print(f"[Level 1] Skipping architecture {source_index}: {exc}")
            continue

        if compatibility_error:
            print(
                f"[Level 1] Skipping architecture {source_index}: "
                f"{compatibility_error}"
            )
            continue

        objective_values = (
            level1_objectives[source_index]
            if source_index < len(level1_objectives)
            else []
        )
        candidates.append(
            Level1Candidate(
                source_index=source_index,
                raw_genome=genome,
                objective_values=objective_values,
                discrete_genome=discretize_genome(genome, bounds),
                architecture_config=config,
                accelerator=accelerator,
            )
        )

    return candidates


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
    level1_csv_path: str,
    level2_csv_path: Path | None,
    level2_solutions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not level2_solutions:
        return []

    level1_by_name = dict(zip(LEVEL1_OBJECTIVES, level1_objective_values))
    rows: list[dict[str, Any]] = []
    for solution in level2_solutions:
        rows.append(
            {
                "architecture_index": architecture_index,
                "level1_raw_genome": level1_raw_genome,
                "level1_discrete_genome": level1_discrete_genome,
                "level1_architecture_config": level1_architecture_config,
                "level1_objective_names": LEVEL1_OBJECTIVES,
                "level1_objective_values": level1_objective_values,
                "level1_latency": level1_by_name.get("latency", ""),
                "level1_energy": level1_by_name.get("energy", ""),
                "level1_area": level1_by_name.get("area", ""),
                "level2_solution_index": solution.get("solution_index", ""),
                "level2_genome": solution.get("genome", ""),
                "selected_ips": solution.get("selected_ips", ""),
                "level2_objective_names": solution.get(
                    "objective_names",
                    LEVEL2_OBJECTIVES,
                ),
                "level2_objective_values": solution.get("objective_values", ""),
                "level2_area": solution.get("area", ""),
                "level2_power": solution.get("power", ""),
                "level2_delay": solution.get("delay", ""),
                "level2_throughput": solution.get("throughput", ""),
                "level2_valid": solution.get("valid", ""),
                "level1_csv_path": level1_csv_path,
                "level2_csv_path": "" if level2_csv_path is None else str(level2_csv_path),
            }
        )
    return rows


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
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def print_first_solution(solution: dict[str, Any]) -> None:
    print("  First solution:")
    print(f"    area: {solution.get('area')}")
    print(f"    power: {solution.get('power')}")
    print(f"    delay: {solution.get('delay')}")
    print(f"    throughput: {solution.get('throughput')}")
    print(f"    selected IPs: {solution.get('selected_ips')}")


if __name__ == "__main__":
    raise SystemExit(main())
