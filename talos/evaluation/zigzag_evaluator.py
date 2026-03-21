from dataclasses import dataclass


@dataclass
class EvaluationResult:
    latency: float
    energy: float
    area: float
    valid: bool


class ZigZagEvaluator:
    def evaluate(self, genome: list[float]) -> EvaluationResult:
        # TODO: integrar ZigZag real
        return EvaluationResult(
            latency=100.0,
            energy=50.0,
            area=30.0,
            valid=True,
        )
