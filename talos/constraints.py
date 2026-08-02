from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UserConstraints:
    max_area_mm2: float | None = None
    max_power_w: float | None = None
    max_latency_cycles: float | None = None
    min_frequency_mhz: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_area_mm2", self.max_area_mm2),
            ("max_power_w", self.max_power_w),
            ("max_latency_cycles", self.max_latency_cycles),
            ("min_frequency_mhz", self.min_frequency_mhz),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be > 0 when provided.")

    def level1_constraint_values(self, latency_cycles: float) -> list[float]:
        if self.max_latency_cycles is None:
            return []
        return [float(latency_cycles) - self.max_latency_cycles]

    def level1_violations(self, latency_cycles: float) -> list[str]:
        if (
            self.max_latency_cycles is not None
            and float(latency_cycles) > self.max_latency_cycles
        ):
            return [
                f"latency_cycles {latency_cycles} exceeds max_latency_cycles {self.max_latency_cycles}"
            ]
        return []

    def level2_violations(
        self,
        *,
        area_mm2: float,
        power_w: float | None,
        physical_fmax_mhz: float | None,
    ) -> list[str]:
        violations: list[str] = []
        if self.max_area_mm2 is not None and area_mm2 > self.max_area_mm2:
            violations.append(
                f"area_mm2 {area_mm2} exceeds max_area_mm2 {self.max_area_mm2}"
            )
        if self.max_power_w is not None:
            if power_w is None:
                violations.append(
                    "Power constraint cannot be evaluated for this candidate."
                )
            elif power_w > self.max_power_w:
                violations.append(
                    f"power_w {power_w} exceeds max_power_w {self.max_power_w}"
                )
        if self.min_frequency_mhz is not None:
            if physical_fmax_mhz is None:
                violations.append("physical_fmax_mhz is unavailable")
            elif physical_fmax_mhz < self.min_frequency_mhz:
                violations.append(
                    "physical_fmax_mhz "
                    f"{physical_fmax_mhz} is below min_frequency_mhz {self.min_frequency_mhz}"
                )
        return violations

    @property
    def level2_constraint_count(self) -> int:
        return sum(
            value is not None
            for value in (
                self.max_area_mm2,
                self.max_power_w,
                self.min_frequency_mhz,
            )
        )

    def level2_constraint_values(
        self,
        *,
        area_mm2: float,
        power_w: float | None,
        physical_fmax_mhz: float | None,
    ) -> list[float]:
        values: list[float] = []
        if self.max_area_mm2 is not None:
            values.append(area_mm2 - self.max_area_mm2)
        if self.max_power_w is not None:
            values.append(
                float("inf")
                if power_w is None
                else power_w - self.max_power_w
            )
        if self.min_frequency_mhz is not None:
            values.append(
                float("inf")
                if physical_fmax_mhz is None
                else self.min_frequency_mhz - physical_fmax_mhz
            )
        return values


def estimated_inferences_per_second(
    *,
    workload_latency_s: float | None,
) -> float | None:
    if (
        workload_latency_s is None
        or workload_latency_s <= 0
        or not math.isfinite(workload_latency_s)
    ):
        return None
    return 1.0 / workload_latency_s


def estimated_fps(
    *,
    workload_latency_s: float | None = None,
    latency_cycles: float | None = None,
    implementation_fmax_mhz: float | None = None,
) -> float | None:
    if workload_latency_s is not None:
        return estimated_inferences_per_second(
            workload_latency_s=workload_latency_s,
        )
    if (
        latency_cycles is None
        or implementation_fmax_mhz is None
        or latency_cycles <= 0
        or not math.isfinite(latency_cycles)
    ):
        return None
    return implementation_fmax_mhz * 1_000_000.0 / latency_cycles
