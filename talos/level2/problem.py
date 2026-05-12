from __future__ import annotations

from typing import Any

try:
    from pymoo.core.problem import ElementwiseProblem
except ModuleNotFoundError:
    class ElementwiseProblem:
        def __init__(self, **kwargs: Any) -> None:
            self.n_var = kwargs["n_var"]
            self.n_obj = kwargs["n_obj"]
            self.xl = kwargs["xl"]
            self.xu = kwargs["xu"]

from talos.architecture.abstract_accelerator import AbstractAccelerator
from talos.ip.ip_pool import IPPool
from talos.level2.evaluator import Level2EvaluationResult, Level2Evaluator
from talos.level2.genome import Level2GenomeSpec


SUPPORTED_LEVEL2_OBJECTIVES = {
    "area",
    "power",
    "delay",
    "inv_throughput",
}


class Level2PymooProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        accelerator: AbstractAccelerator,
        ip_pool: IPPool,
        objective_names: list[str],
    ) -> None:
        self.accelerator = accelerator
        self.ip_pool = ip_pool
        self.objective_names = list(objective_names)
        self._validate_objective_names(self.objective_names)
        self.spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, ip_pool)
        self.evaluator = Level2Evaluator()

        bounds = self.spec.gene_bounds()
        xl = [float(lower) for lower, _upper in bounds]
        xu = [float(upper) for _lower, upper in bounds]

        super().__init__(
            n_var=len(self.spec.genes),
            n_obj=len(self.objective_names),
            xl=xl,
            xu=xu,
        )

    def _evaluate(
        self,
        x: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        genome = [float(value) for value in list(x)]

        try:
            implemented = self.spec.decode(genome)
            result = self.evaluator.evaluate(implemented)
        except Exception:
            out["F"] = [float("inf")] * len(self.objective_names)
            return

        if not result.valid:
            out["F"] = [float("inf")] * len(self.objective_names)
            return

        out["F"] = [self._objective_value(name, result) for name in self.objective_names]

    def _objective_value(self, name: str, result: Level2EvaluationResult) -> float:
        if name == "area":
            return result.area
        if name == "power":
            return result.power
        if name == "delay":
            return result.delay
        if name == "inv_throughput":
            if result.throughput <= 0:
                return float("inf")
            return 1.0 / result.throughput
        raise ValueError(
            f"Unknown Level-2 objective {name!r}. Supported objectives: {', '.join(sorted(SUPPORTED_LEVEL2_OBJECTIVES))}."
        )

    def _validate_objective_names(self, objective_names: list[str]) -> None:
        if not objective_names:
            raise ValueError("At least one Level-2 objective name is required.")
        unknown = sorted(set(objective_names) - SUPPORTED_LEVEL2_OBJECTIVES)
        if unknown:
            raise ValueError(
                f"Unknown Level-2 objective name(s): {', '.join(unknown)}"
            )
