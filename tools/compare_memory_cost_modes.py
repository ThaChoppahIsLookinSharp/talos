from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


GENOME = [2, 2, 3, 2, 3, 2, 3, 3]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    workload = repo_root() / "workloads" / "alexnet.onnx"
    base_workdir = repo_root() / ".talos_zigzag" / "memory_cost_compare"

    for mode in ("manual", "zigzag_auto"):
        evaluator = ZigZagEvaluator(
            workload=str(workload),
            debug=False,
            workdir=str(base_workdir / mode),
            memory_cost_mode=mode,
            lpf_limit=1,
            nb_spatial_mappings_generated=1,
        )
        print(f"=== memory_cost_mode={mode} ===")
        print(evaluator.render_accelerator_yaml(GENOME))

        result = evaluator.evaluate(GENOME)
        print(f"latency={result.latency}")
        print(f"energy={result.energy}")
        print(f"area={result.area}")
        print(f"area_source={result.area_source}")
        print(f"memory_cost_mode={result.memory_cost_mode}")
        print(f"valid={result.valid}")
        print(f"error_message={result.error_message}")
        print()


if __name__ == "__main__":
    main()
