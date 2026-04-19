from pathlib import Path

from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator

workload = Path("workloads/alexnet.onnx").resolve()

evaluator = ZigZagEvaluator(str(workload), debug=False)
adapter = ObjectiveAdapter(evaluator)

genome = [2, 2, 3, 2, 3, 2, 3, 3]

print("Base methods:")
print("latency:", adapter.latency(genome))
print("energy:", adapter.energy(genome))
print("area:", adapter.area(genome))
print("vector:", adapter.vector(genome))

print("\nNamed objectives:")
for name in ["latency", "energy", "area", "edp", "eap", "alp"]:
    value = adapter.evaluate_objective(name, genome)
    print(f"{name}: {value}")

print("\nCallable objectives:")
objective_names = ["latency", "energy", "area", "edp"]
objectives = adapter.build_objectives(objective_names)

for name, fn in zip(objective_names, objectives):
    print(f"{name}: {fn(genome)}")