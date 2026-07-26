from __future__ import annotations

from typing import Any

try:
    from pymoo.core.problem import ElementwiseProblem
except ModuleNotFoundError:
    class ElementwiseProblem:
        def __init__(self, **kwargs: Any) -> None:
            self.n_var = kwargs["n_var"]
            self.n_obj = kwargs["n_obj"]
            self.n_ieq_constr = kwargs.get("n_ieq_constr", 0)
            self.xl = kwargs["xl"]
            self.xu = kwargs["xu"]

from talos.architecture.abstract_accelerator import AbstractAccelerator
from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.ip.ip_pool import IPPool
from talos.level2.evaluator import Level2EvaluationResult, Level2Evaluator
from talos.level2.genome import Level2GenomeSpec
from talos.level2.workload_power import validate_power_aware_exploration


SUPPORTED_LEVEL2_OBJECTIVES = {
    "area",
    "energy",
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
        constraints: UserConstraints | None = None,
        activity_profile: WorkloadActivityProfile | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.ip_pool = ip_pool
        self.objective_names = list(objective_names)
        self.constraints = constraints
        self.activity_profile = activity_profile
        self._validate_objective_names(self.objective_names)
        self.spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, ip_pool)
        if any(name in self.objective_names for name in ("energy", "power")) or (
            constraints is not None and constraints.max_power_w is not None
        ):
            validate_power_aware_exploration(self.spec, activity_profile)
        self.evaluator = Level2Evaluator(
            constraints=constraints,
            activity_profile=activity_profile,
        )

        bounds = self.spec.gene_bounds()
        xl = [float(lower) for lower, _upper in bounds]
        xu = [float(upper) for _lower, upper in bounds]

        super().__init__(
            n_var=len(self.spec.genes),
            n_obj=len(self.objective_names),
            n_ieq_constr=(
                0 if constraints is None else constraints.level2_constraint_count
            ),
            vtype=int,
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
            self._set_constraint_values(out, None)
            return

        out["F"] = [self._objective_value(name, result) for name in self.objective_names]
        self._set_constraint_values(out, result)

    def _set_constraint_values(
        self,
        out: dict[str, Any],
        result: Level2EvaluationResult | None,
    ) -> None:
        if self.constraints is None or self.constraints.level2_constraint_count == 0:
            return
        out["G"] = (
            [float("inf")] * self.constraints.level2_constraint_count
            if result is None
            else self.constraints.level2_constraint_values(
                area_mm2=result.area,
                power_w=result.power,
                implementation_fmax_mhz=result.implementation_fmax_mhz,
            )
        )

    def _objective_value(self, name: str, result: Level2EvaluationResult) -> float:
        if name == "area":
            return result.area
        if name == "energy":
            return (
                float("inf")
                if result.workload_energy_j is None
                else result.workload_energy_j
            )
        if name == "power":
            return float("inf") if result.power is None else result.power
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
