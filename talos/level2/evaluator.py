from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import WorkloadActivityProfile
from talos.ip.ip_characterization import IPBlock
from talos.level2.genome import (
    ImplementedAccelerator,
    ImplementedComponent,
    physical_components,
)
from talos.level2.workload_power import evaluate_workload_power


@dataclass(frozen=True)
class Level2EvaluationResult:
    area: float
    power: float | None
    workload_energy_j: float | None
    dram_energy_j: float | None
    layer_cycles_mapping: tuple[tuple[str, float], ...] | None
    workload_cycles_per_inference: float | None
    workload_latency_s: float | None
    workload_throughput_ips: float | None
    reference_frequency_mhz: float | None
    reference_voltage_v: float | None
    physical_critical_delay: float
    selected_ip_min_throughput: float
    physical_fmax_mhz: float | None
    timing_margin_mhz: float | None
    valid: bool
    constraint_violations: tuple[str, ...] = ()
    error_message: str | None = None


class Level2Evaluator:
    def __init__(
        self,
        constraints: UserConstraints | None = None,
        activity_profile: WorkloadActivityProfile | None = None,
        dram_ip: IPBlock | None = None,
    ) -> None:
        self.constraints = constraints
        self.activity_profile = activity_profile
        self.dram_ip = dram_ip

    def evaluate(self, implemented: ImplementedAccelerator) -> Level2EvaluationResult:
        try:
            self._validate_implemented_accelerator(implemented.components)
            physical = physical_components(implemented.components)
            area = sum(
                component.abstract_component.count * component.ip.area
                for component in physical
            )
            physical_critical_delay = max(
                component.ip.delay for component in physical
            )
            selected_ip_min_throughput = min(
                component.ip.throughput for component in physical
            )
            physical_fmax_mhz = self._physical_fmax_mhz(
                physical
            )

            power_result = None
            if self.activity_profile is not None:
                if self.dram_ip is None:
                    raise ValueError(
                        "missing_characterization: workload-aware Level 2 "
                        "requires one DRAM IP."
                    )
                power_result = evaluate_workload_power(
                    implemented,
                    self.activity_profile,
                    self.dram_ip,
                )
            power = None if power_result is None else power_result.power_w
            reference_frequency_mhz = (
                None
                if power_result is None
                else power_result.reference_frequency_mhz
            )
            timing_margin_mhz = (
                None
                if reference_frequency_mhz is None or physical_fmax_mhz is None
                else physical_fmax_mhz - reference_frequency_mhz
            )
            constraint_violations = self._constraint_violations(
                area=area,
                power=power,
                physical_fmax_mhz=physical_fmax_mhz,
            )
            return Level2EvaluationResult(
                area=float(area),
                power=power,
                workload_energy_j=(
                    None if power_result is None else power_result.energy_j
                ),
                dram_energy_j=(
                    None if power_result is None else power_result.dram_energy_j
                ),
                layer_cycles_mapping=(
                    None
                    if power_result is None
                    else power_result.layer_cycles_mapping
                ),
                workload_cycles_per_inference=(
                    None
                    if power_result is None
                    else power_result.workload_cycles_per_inference
                ),
                workload_latency_s=(
                    None
                    if power_result is None
                    else power_result.workload_latency_s
                ),
                workload_throughput_ips=(
                    None
                    if power_result is None
                    else power_result.workload_throughput_ips
                ),
                reference_frequency_mhz=reference_frequency_mhz,
                reference_voltage_v=(
                    None
                    if power_result is None
                    else power_result.reference_voltage_v
                ),
                physical_critical_delay=float(physical_critical_delay),
                selected_ip_min_throughput=float(selected_ip_min_throughput),
                physical_fmax_mhz=physical_fmax_mhz,
                timing_margin_mhz=timing_margin_mhz,
                valid=not constraint_violations,
                constraint_violations=tuple(constraint_violations),
                error_message="; ".join(constraint_violations) or None,
            )
        except Exception as exc:
            return Level2EvaluationResult(
                area=float("inf"),
                power=None,
                workload_energy_j=None,
                dram_energy_j=None,
                layer_cycles_mapping=None,
                workload_cycles_per_inference=None,
                workload_latency_s=None,
                workload_throughput_ips=None,
                reference_frequency_mhz=None,
                reference_voltage_v=None,
                physical_critical_delay=float("inf"),
                selected_ip_min_throughput=0.0,
                physical_fmax_mhz=None,
                timing_margin_mhz=None,
                valid=False,
                error_message=str(exc),
            )

    def _physical_fmax_mhz(
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
        physical_fmax_mhz: float | None,
    ) -> list[str]:
        if self.constraints is None:
            return []
        return self.constraints.level2_violations(
            area_mm2=float(area),
            power_w=power,
            physical_fmax_mhz=physical_fmax_mhz,
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
