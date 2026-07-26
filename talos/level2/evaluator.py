from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent
from talos.level2.workload_power import evaluate_workload_power


@dataclass(frozen=True)
class Level2EvaluationResult:
    area: float
    power: float | None
    workload_energy_j: float | None
    workload_latency_s: float | None
    operating_frequency_mhz: float | None
    delay: float
    throughput: float
    implementation_fmax_mhz: float | None
    valid: bool
    constraint_violations: tuple[str, ...] = ()
    error_message: str | None = None


class Level2Evaluator:
    def __init__(
        self,
        constraints: UserConstraints | None = None,
        activity_profile: WorkloadActivityProfile | None = None,
    ) -> None:
        self.constraints = constraints
        self.activity_profile = activity_profile

    def evaluate(self, implemented: ImplementedAccelerator) -> Level2EvaluationResult:
        try:
            self._validate_implemented_accelerator(implemented.components)
            area = sum(
                component.abstract_component.count * component.ip.area
                for component in implemented.components
            )
            delay = max(component.ip.delay for component in implemented.components)
            throughput = min(
                component.ip.throughput for component in implemented.components
            )
            implementation_fmax_mhz = self._implementation_fmax_mhz(
                implemented.components
            )

            power_result = None
            if self.activity_profile is not None and all(
                component.ip.power_model is not None
                for component in implemented.components
            ):
                power_result = evaluate_workload_power(
                    implemented,
                    self.activity_profile,
                )
            power = None if power_result is None else power_result.power_w
            constraint_violations = self._constraint_violations(
                area=area,
                power=power,
                implementation_fmax_mhz=implementation_fmax_mhz,
            )
            return Level2EvaluationResult(
                area=float(area),
                power=power,
                workload_energy_j=(
                    None if power_result is None else power_result.energy_j
                ),
                workload_latency_s=(
                    None if power_result is None else power_result.latency_s
                ),
                operating_frequency_mhz=(
                    None
                    if power_result is None
                    else power_result.operating_frequency_mhz
                ),
                delay=float(delay),
                throughput=float(throughput),
                implementation_fmax_mhz=implementation_fmax_mhz,
                valid=not constraint_violations,
                constraint_violations=tuple(constraint_violations),
                error_message="; ".join(constraint_violations) or None,
            )
        except Exception as exc:
            return Level2EvaluationResult(
                area=float("inf"),
                power=None,
                workload_energy_j=None,
                workload_latency_s=None,
                operating_frequency_mhz=None,
                delay=float("inf"),
                throughput=0.0,
                implementation_fmax_mhz=None,
                valid=False,
                error_message=str(exc),
            )

    def _implementation_fmax_mhz(
        self,
        components: Iterable[ImplementedComponent],
    ) -> float | None:
        fmax_values = [component.ip.fmax_mhz for component in components]
        if not fmax_values or any(value is None for value in fmax_values):
            return None
        return float(min(value for value in fmax_values if value is not None))

    def _constraint_violations(
        self,
        *,
        area: float,
        power: float | None,
        implementation_fmax_mhz: float | None,
    ) -> list[str]:
        if self.constraints is None:
            return []
        return self.constraints.level2_violations(
            area_mm2=float(area),
            power_w=power,
            implementation_fmax_mhz=implementation_fmax_mhz,
        )

    def _validate_implemented_accelerator(
        self,
        components: Iterable[ImplementedComponent],
    ) -> None:
        seen_any = False
        for component in components:
            seen_any = True
            required_capacity = component.abstract_component.required_capacity_bits
            if required_capacity is not None and (
                component.ip.capacity_bits is None
                or component.ip.capacity_bits < required_capacity
            ):
                raise ValueError(
                    f"Selected IP {component.ip.id!r} does not satisfy capacity "
                    f"requirement for component {component.abstract_component.name!r}."
                )
            required_bandwidth = component.abstract_component.required_bandwidth_bits
            if required_bandwidth is not None and (
                component.ip.bandwidth_bits is None
                or component.ip.bandwidth_bits < required_bandwidth
            ):
                raise ValueError(
                    f"Selected IP {component.ip.id!r} does not satisfy bandwidth "
                    f"requirement for component {component.abstract_component.name!r}."
                )
        if not seen_any:
            raise ValueError(
                "ImplementedAccelerator must contain at least one component."
            )
