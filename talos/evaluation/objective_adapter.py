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

    def __init__(self, evaluator: ZigZagEvaluator) -> None:
        self.evaluator = evaluator
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
            self._cache[key] = self.evaluator.evaluate(list(key))

        return self._cache[key]

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
