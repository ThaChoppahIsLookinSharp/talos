from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


def main() -> None:
    evaluator = ZigZagEvaluator("E:\\uni\\tfm\\repos\\talos\\workloads\\alexnet.onnx")
    adapter = ObjectiveAdapter(evaluator)

    # Test genome: 8 genes, matching the proposed TALOS/ZigZag adapter layout
    genome = [2, 2, 3, 2, 3, 2, 3, 3]

    print("Evaluating genome:", genome)

    latency = adapter.latency(genome)
    energy = adapter.energy(genome)
    area = adapter.area(genome)

    print("\nIndividual objective values:")
    print(f"  Latency: {latency}")
    print(f"  Energy : {energy}")
    print(f"  Area   : {area}")

    print("\nFull objective vector:")
    print(" ", adapter.vector(genome))

    print("\nCache entries after first evaluation:")
    print(" ", len(adapter._cache))

    # Run the same genome again to verify cache reuse
    latency2 = adapter.latency(genome)
    energy2 = adapter.energy(genome)
    area2 = adapter.area(genome)

    print("\nSecond evaluation of the same genome:")
    print(f"  Latency: {latency2}")
    print(f"  Energy : {energy2}")
    print(f"  Area   : {area2}")

    print("\nCache entries after repeated evaluation:")
    print(" ", len(adapter._cache))

    if len(adapter._cache) == 1:
        print("\nOK: the evaluation was cached and should not be recomputed.")
    else:
        print("\nSomething is wrong with the cache.")


if __name__ == "__main__":
    main()