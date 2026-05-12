from __future__ import annotations

import unittest

from talos.evaluation.objective_adapter import ObjectiveAdapter
from talos.evaluation.zigzag_evaluator import EvaluationResult


class CountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.genomes: list[list[float]] = []

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        self.calls += 1
        self.genomes.append(genome)
        return EvaluationResult(
            latency=1.0,
            energy=2.0,
            area=3.0,
            valid=True,
            area_source="zigzag",
            memory_cost_mode="manual",
        )


class ObjectiveAdapterTests(unittest.TestCase):
    def test_cache_key_uses_canonical_discrete_genome(self) -> None:
        evaluator = CountingEvaluator()
        adapter = ObjectiveAdapter(evaluator)  # type: ignore[arg-type]

        first = [2.49, 2.49, 1.51, 0.51, 2.49]
        second = [2.01, 2.01, 1.99, 0.99, 2.01]

        self.assertEqual(adapter.latency(first), 1.0)
        self.assertEqual(adapter.energy(second), 2.0)

        self.assertEqual(evaluator.calls, 1)
        self.assertEqual(evaluator.genomes[0], [2, 2, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
