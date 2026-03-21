from talos.evaluation.zigzag_evaluator import ZigZagEvaluator
from talos.evaluation.objective_adapter import ObjectiveAdapter


def main() -> None:
    evaluator = ZigZagEvaluator()
    adapter = ObjectiveAdapter(evaluator)

    genome = [16, 64, 256, 1]

    print("Primera llamada: latency")
    latency = adapter.latency(genome)
    print(f"latency = {latency}")

    print("\nSegunda llamada: energy")
    energy = adapter.energy(genome)
    print(f"energy = {energy}")

    print("\nTercera llamada: area")
    area = adapter.area(genome)
    print(f"area = {area}")

    print("\nCuarta llamada: vector completo")
    vector = adapter.vector(genome)
    print(f"vector = {vector}")


if __name__ == "__main__":
    main()
