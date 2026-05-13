from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_workload_path() -> Path:
    return repo_root() / "workloads" / "alexnet.onnx"


def default_accelerator_path() -> Path:
    return repo_root() / "configs" / "zigzag_accelerator_example.yaml"


def default_ip_pool_path() -> Path:
    return repo_root() / "configs" / "ip_pool_example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m talos",
        description="TALOS hierarchical design-space exploration tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a quick Level 1 evaluator smoke test")
    smoke.add_argument("--workload", type=Path, default=default_workload_path())
    smoke.add_argument("--debug", action="store_true")
    smoke.add_argument("--zigzag-lpf-limit", type=int, default=1)
    smoke.add_argument("--zigzag-spatial-mappings", type=int, default=1)
    smoke.set_defaults(func=_run_smoke)

    level1 = subparsers.add_parser("level1", help="Run Level 1 NSGA-II architecture search")
    level1.add_argument("--workload", type=Path, default=default_workload_path())
    level1.add_argument(
        "--objectives",
        nargs="+",
        default=["latency", "energy", "area"],
    )
    level1.add_argument("--pop-size", type=int, default=6)
    level1.add_argument("--generations", type=int, default=2)
    level1.add_argument("--seed", type=int, default=1)
    level1.add_argument("--workers", type=int, default=1)
    level1.add_argument("--results-dir", type=Path, default=Path("results") / "level1")
    level1.add_argument("--no-save-csv", action="store_true")
    level1.add_argument("--debug", action="store_true")
    level1.add_argument("--zigzag-lpf-limit", type=int, default=1)
    level1.add_argument("--zigzag-spatial-mappings", type=int, default=1)
    level1.set_defaults(func=_run_level1)

    level2 = subparsers.add_parser("level2", help="Run Level 2 NSGA-II IP selection")
    level2.add_argument("--accelerator", type=Path, default=default_accelerator_path())
    level2.add_argument("--ip-pool", type=Path, default=default_ip_pool_path())
    level2.add_argument(
        "--objectives",
        nargs="+",
        default=["area", "power", "delay", "inv_throughput"],
    )
    level2.add_argument("--pop-size", type=int, default=6)
    level2.add_argument("--generations", type=int, default=2)
    level2.add_argument("--seed", type=int, default=1)
    level2.add_argument("--results-dir", type=Path, default=Path("results") / "level2")
    level2.add_argument("--no-save-csv", action="store_true")
    level2.add_argument("--debug", action="store_true")
    level2.set_defaults(func=_run_level2)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Placeholder for future hierarchical Level 1 -> Level 2 orchestration",
    )
    pipeline.set_defaults(func=_run_pipeline)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _run_smoke(args: argparse.Namespace) -> None:
    from talos.level1.genome import default_genome
    from talos.level1.objective_adapter import ObjectiveAdapter
    from talos.level1.zigzag_evaluator import ZigZagEvaluator

    workload = args.workload.resolve()
    if not workload.exists():
        raise FileNotFoundError(f"Workload file not found: {workload}")

    evaluator = ZigZagEvaluator(
        str(workload),
        debug=args.debug,
        lpf_limit=args.zigzag_lpf_limit,
        nb_spatial_mappings_generated=args.zigzag_spatial_mappings,
    )
    adapter = ObjectiveAdapter(evaluator, verbose=args.debug)
    genome = default_genome()
    result = adapter.evaluate(genome)

    print("TALOS smoke test finished.")
    print(f"workload={workload}")
    print(f"genome={genome}")
    print(f"valid={result.valid}")
    print(f"latency={result.latency}")
    print(f"energy={result.energy}")
    print(f"area={result.area}")
    if result.error_message:
        print(f"error_message={result.error_message}")


def _run_level1(args: argparse.Namespace) -> None:
    from talos.level1.runner import run_level1_nsga2

    workload = args.workload.resolve()
    if not workload.exists():
        raise FileNotFoundError(f"Workload file not found: {workload}")

    result = run_level1_nsga2(
        workload_path=str(workload),
        objective_names=args.objectives,
        pop_size=args.pop_size,
        n_gen=args.generations,
        seed=args.seed,
        n_workers=args.workers,
        debug=args.debug,
        save_csv=not args.no_save_csv,
        results_dir=str(args.results_dir),
        zigzag_lpf_limit=args.zigzag_lpf_limit,
        zigzag_spatial_mappings=args.zigzag_spatial_mappings,
    )

    solution_count = _count_pymoo_solutions(result.X)
    print("Level 1 NSGA-II run finished.")
    print(f"solutions={solution_count}")
    if getattr(result, "talos", None) is not None:
        print(f"csv_path={result.talos.csv_path}")


def _run_level2(args: argparse.Namespace) -> None:
    from talos.level2.architecture.zigzag_yaml_importer import (
        abstract_accelerator_from_zigzag_yaml,
    )
    from talos.level2.ip import IPPool
    from talos.level2.runner import run_level2_nsga2

    accelerator_path = args.accelerator.resolve()
    ip_pool_path = args.ip_pool.resolve()
    if not accelerator_path.exists():
        raise FileNotFoundError(f"Accelerator YAML not found: {accelerator_path}")
    if not ip_pool_path.exists():
        raise FileNotFoundError(f"IP pool YAML not found: {ip_pool_path}")

    accelerator = abstract_accelerator_from_zigzag_yaml(str(accelerator_path))
    ip_pool = IPPool.from_yaml(ip_pool_path)
    result = run_level2_nsga2(
        accelerator=accelerator,
        ip_pool=ip_pool,
        objective_names=args.objectives,
        pop_size=args.pop_size,
        n_gen=args.generations,
        seed=args.seed,
        save_csv=not args.no_save_csv,
        results_dir=str(args.results_dir),
        debug=args.debug,
    )

    print("Level 2 NSGA-II run finished.")
    print(f"solutions={len(result.solutions)}")
    print(f"csv_path={result.csv_path}")
    for solution in result.solutions[:3]:
        print(
            "solution "
            f"{solution['solution_index']}: "
            f"valid={solution['valid']} "
            f"area={solution['area']} "
            f"power={solution['power']} "
            f"delay={solution['delay']} "
            f"throughput={solution['throughput']} "
            f"selected_ips={solution['selected_ips']}"
        )


def _run_pipeline(_args: argparse.Namespace) -> None:
    print("Hierarchical pipeline is not implemented yet.")
    print("Use `python -m talos level1` and `python -m talos level2` independently.")


def _count_pymoo_solutions(raw_x: object) -> int:
    if raw_x is None:
        return 0
    if hasattr(raw_x, "tolist"):
        raw_x = raw_x.tolist()
    if isinstance(raw_x, list):
        if not raw_x:
            return 0
        if all(not isinstance(value, (list, tuple)) for value in raw_x):
            return 1
        return len(raw_x)
    return 1
