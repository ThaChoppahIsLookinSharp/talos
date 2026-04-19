from pathlib import Path

from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    workload_path = repo_root / "workloads" / "alexnet.onnx"

    evaluator = ZigZagEvaluator(str(workload_path))
    adapter = ObjectiveAdapter(evaluator)

    # Test genome: 8 genes, matching the current TALOS/ZigZag adapter layout
    genome = [2, 2, 3, 2, 3, 2, 3, 3]

    print("Evaluating genome:", genome)

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

    # Run the same genome again to verify cache reuse
    print("\nRe-evaluating the same genome through derived objectives:")
    for name in ["latency", "edp", "alp"]:
        value = adapter.evaluate_objective(name, genome)
        print(f"  {name}: {value}")

    print("\nCache entries after repeated evaluation:")
    print(" ", len(adapter._cache))

    if len(adapter._cache) == 1:
        print("\nOK: the genome evaluation was cached and reused.")
    else:
        print("\nSomething is wrong with the cache.")

    print("\nTesting unknown objective handling:")
    try:
        adapter.evaluate_objective("unknown_objective", genome)
    except ValueError as exc:
        print(f"  Caught expected error: {exc}")


if __name__ == "__main__":
    main()
