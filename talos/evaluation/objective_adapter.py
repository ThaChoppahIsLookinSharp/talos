from talos.evaluation.zigzag_evaluator import ZigZagEvaluator, EvaluationResult


class ObjectiveAdapter:
    def __init__(self, evaluator: ZigZagEvaluator) -> None:
        self.evaluator = evaluator
        self._cache: dict[tuple[float, ...], EvaluationResult] = {}

    def _get_result(self, genome: list[float]) -> EvaluationResult:
        key = tuple(genome)

        if key not in self._cache:
            self._cache[key] = self.evaluator.evaluate(genome)

        return self._cache[key]

    def latency(self, genome: list[float]) -> float:
        return self._get_result(genome).latency

    def energy(self, genome: list[float]) -> float:
        return self._get_result(genome).energy

    def area(self, genome: list[float]) -> float:
        return self._get_result(genome).area

    def vector(self, genome: list[float]) -> tuple[float, float, float]:
        result = self._get_result(genome)
        return (result.latency, result.energy, result.area)
