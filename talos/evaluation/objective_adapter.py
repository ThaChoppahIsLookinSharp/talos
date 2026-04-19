from __future__ import annotations

from talos.evaluation.zigzag_evaluator import EvaluationResult, ZigZagEvaluator


class ObjectiveAdapter:
    """
    Adapter between the evolutionary algorithm and the evaluator.

    The NSGA-II library expects one objective function per metric.
    Each objective receives the genome and returns a single float.

    This adapter guarantees that the expensive ZigZag evaluation is
    executed only once per genome by caching the full EvaluationResult.
    """

    def __init__(self, evaluator: ZigZagEvaluator, verbose: bool = False) -> None:
        self.evaluator = evaluator
        self.verbose = verbose
        self._cache: dict[tuple[float, ...], EvaluationResult] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _normalize_key(self, genome: list[float]) -> tuple[float, ...]:
        # The NSGA-II implementation may pass floats even for discrete genes.
        # Rounding here makes the cache robust to tiny floating-point noise.
        return tuple(round(float(g), 8) for g in genome)

    def _get_result(self, genome: list[float]) -> EvaluationResult:
        key = self._normalize_key(genome)

        if key not in self._cache:
            if self.verbose:
                print("Cache miss -> evaluating genome")
            self._cache[key] = self.evaluator.evaluate(list(key))
        elif self.verbose:
            print("Cache hit")
        return self._cache[key]

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        return self._get_result(genome)

    def latency(self, genome: list[float]) -> float:
        result = self._get_result(genome)
        return result.latency if result.valid else float("inf")

    def energy(self, genome: list[float]) -> float:
        result = self._get_result(genome)
        return result.energy if result.valid else float("inf")

    def area(self, genome: list[float]) -> float:
        result = self._get_result(genome)
        return result.area if result.valid else float("inf")

    def vector(self, genome: list[float]) -> tuple[float, float, float]:
        result = self._get_result(genome)

        if not result.valid:
            return (float("inf"), float("inf"), float("inf"))

        return (result.latency, result.energy, result.area)

    def evaluate_objective(self, name: str, genome: list[float]) -> float:
        result = self._get_result(genome)

        if not result.valid:
            return float("inf")

        if name == "latency":
            return result.latency
        if name == "energy":
            return result.energy
        if name == "area":
            return result.area
        if name == "edp":
            return result.energy * result.latency
        if name == "eap":
            return result.energy * result.area
        if name == "alp":
            return result.area * result.latency

        raise ValueError(f"Unknown objective: {name}")

    def get_objective(self, name: str):
        def objective(genome: list[float]) -> float:
            return self.evaluate_objective(name, genome)
        return objective

    def build_objectives(self, names: list[str]):
        return [self.get_objective(name) for name in names]
