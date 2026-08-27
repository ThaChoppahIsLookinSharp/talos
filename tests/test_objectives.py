from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.area_calibration import characterize_level1_area
from talos.evaluation.cacti_costs import characterize_level1_energy
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator
from talos.ip import IPPool


def main() -> None:
    workload = Path("workloads/alexnet.onnx").resolve()
    evaluator = ZigZagEvaluator(
        str(workload),
        debug=False,
        energy_calibration=characterize_level1_energy(),
        area_calibration=characterize_level1_area(
            IPPool.from_yaml(REPO_ROOT / "configs" / "ip_pool_synthetic_65nm.yaml")
        ),
    )
    adapter = ObjectiveAdapter(evaluator)
    genome = [2, 2, 3, 2, 3, 2, 3]

    print("Base methods:")
    print("latency:", adapter.latency(genome))
    print("energy:", adapter.energy(genome))
    print("area:", adapter.area(genome))
    print("vector:", adapter.vector(genome))

    print("\nNamed objectives:")
    for name in ["latency", "energy", "area", "edp", "eap", "alp"]:
        print(f"{name}: {adapter.evaluate_objective(name, genome)}")

    print("\nCallable objectives:")
    names = ["latency", "energy", "area", "edp"]
    for name, objective in zip(names, adapter.build_objectives(names), strict=True):
        print(f"{name}: {objective(genome)}")


if __name__ == "__main__":
    main()
