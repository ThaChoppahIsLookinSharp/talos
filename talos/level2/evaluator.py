from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from talos.level2.genome import ImplementedAccelerator, ImplementedComponent


@dataclass(frozen=True)
class Level2EvaluationResult:
    area: float
    power: float
    delay: float
    throughput: float
    valid: bool
    error_message: str | None = None


class Level2Evaluator:
    def evaluate(self, implemented: ImplementedAccelerator) -> Level2EvaluationResult:
        try:
            self._validate_implemented_accelerator(implemented.components)
            area = sum(component.abstract_component.count * component.ip.area for component in implemented.components)
            power = sum(component.abstract_component.count * component.ip.power for component in implemented.components)
            delay = max(component.ip.delay for component in implemented.components)
            throughput = min(component.ip.throughput for component in implemented.components)
            return Level2EvaluationResult(
                area=float(area),
                power=float(power),
                delay=float(delay),
                throughput=float(throughput),
                valid=True,
            )
        except Exception as exc:
            return Level2EvaluationResult(
                area=float("inf"),
                power=float("inf"),
                delay=float("inf"),
                throughput=0.0,
                valid=False,
                error_message=str(exc),
            )

    def _validate_implemented_accelerator(self, components: Iterable[ImplementedComponent]) -> None:
        seen_any = False
        for component in components:
            seen_any = True
            required_capacity = component.abstract_component.required_capacity_bits
            if required_capacity is not None:
                if component.ip.capacity_bits is None or component.ip.capacity_bits < required_capacity:
                    raise ValueError(
                        f"Selected IP {component.ip.id!r} does not satisfy capacity requirement "
                        f"for component {component.abstract_component.name!r}."
                    )
            required_bandwidth = component.abstract_component.required_bandwidth_bits
            if required_bandwidth is not None:
                if component.ip.bandwidth_bits is None or component.ip.bandwidth_bits < required_bandwidth:
                    raise ValueError(
                        f"Selected IP {component.ip.id!r} does not satisfy bandwidth requirement "
                        f"for component {component.abstract_component.name!r}."
                    )
        if not seen_any:
            raise ValueError("ImplementedAccelerator must contain at least one component.")
