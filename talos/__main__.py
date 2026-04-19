from __future__ import annotations

import argparse
from pathlib import Path

from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator
from talos.ga.nsga2_runner import run_nsga2


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_workload_path() -> Path:
    return repo_root() / "workloads" / "alexnet.onnx"


def run_smoke_test(workload_path: Path, debug: bool = False) -> None:
    evaluator = ZigZagEvaluator(str(workload_path), debug=debug)
    adapter = ObjectiveAdapter(evaluator)

    # Test genome matching the current 8-gene TALOS setup
    genome = [2, 2, 3, 2, 3, 2, 3, 3]

    print("Evaluating genome:", genome)

    print("\nBase objective methods:")
    latency = adapter.latency(genome)
    energy = adapter.energy(genome)
    area = adapter.area(genome)

    print(f"  Latency: {latency}")
    print(f"  Energy : {energy}")
    print(f"  Area   : {area}")

    print("\nFull objective vector:")
    print(" ", adapter.vector(genome))

    print("\nNamed objective evaluation:")
    named_objectives = ["latency", "energy", "area", "edp", "eap", "alp"]

    for name in named_objectives:
        value = adapter.evaluate_objective(name, genome)
        print(f"  {name}: {value}")

    print("\nCallable objectives built from names:")
    objective_names = ["latency", "energy", "area", "edp"]
    objectives = adapter.build_objectives(objective_names)

    for name, fn in zip(objective_names, objectives):
        value = fn(genome)
        print(f"  {name}: {value}")

    print("\nCache entries after all evaluations:")
    print(" ", len(adapter._cache))

    print("\nRe-evaluating the same genome:")
    for name in ["latency", "edp", "alp"]:
        value = adapter.evaluate_objective(name, genome)
        print(f"  {name}: {value}")

    print("\nCache entries after repeated evaluation:")
    print(" ", len(adapter._cache))

    if len(adapter._cache) == 1:
        print("\nOK: the genome evaluation was cached and reused.")
    else:
        print("\nWarning: cache size is larger than expected.")

    print("\nTesting unknown objective handling:")
    try:
        adapter.evaluate_objective("unknown_objective", genome)
    except ValueError as exc:
        print(f"  Caught expected error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m talos",
        description="TALOS manual smoke test and NSGA-II runner",
    )

    parser.add_argument(
        "--workload",
        type=Path,
        default=default_workload_path(),
        help="Path to the ONNX workload file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable evaluator debug output",
    )

    parser.add_argument(
        "--ga",
        action="store_true",
        help="Run the NSGA-II example instead of the manual smoke test",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["latency", "energy", "area"],
        help="Objectives for the GA run",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations for the GA run",
    )
    parser.add_argument(
        "--individuals",
        type=int,
        default=10,
        help="Number of individuals for the GA run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for the GA run",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Disable CSV export in the GA run",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root() / "results",
        help="Directory where GA results will be stored",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workload_path = args.workload.resolve()

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_path}")

    if args.ga:
        print("Running NSGA-II...")
        print(f"  Workload    : {workload_path}")
        print(f"  Objectives  : {args.objectives}")
        print(f"  Generations : {args.generations}")
        print(f"  Individuals : {args.individuals}")
        print(f"  Seed        : {args.seed}")
        print(f"  Results dir : {args.results_dir}")

        result = run_nsga2(
            workload_path=str(workload_path),
            objective_names=args.objectives,
            num_of_generations=args.generations,
            num_of_individuals=args.individuals,
            seed=args.seed,
            debug=args.debug,
            save_csv=not args.no_save_csv,
            results_dir=str(args.results_dir),
        )

        print("\nNSGA-II run finished.")
        if result is not None:
            print("Result object returned by runner.")
    else:
        print("Running manual smoke test...")
        print(f"  Workload: {workload_path}")
        run_smoke_test(workload_path, debug=args.debug)


if __name__ == "__main__":
    main()