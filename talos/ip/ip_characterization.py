from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


VALID_IP_TYPES = {
    "pe",
    "register_file",
    "global_buffer",
    "dram",
    "interconnect",
}


@dataclass(frozen=True)
class PowerCharacterization:
    source: str
    activity_method: str
    reference_frequency_mhz: float
    p_idle_w: float
    p_active_w: float
    voltage_v: float | None = None
    temperature_c: float | None = None
    corner: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("PowerCharacterization source must not be empty.")
        if not isinstance(self.activity_method, str) or not self.activity_method.strip():
            raise ValueError("PowerCharacterization activity_method must not be empty.")
        if (
            not math.isfinite(self.reference_frequency_mhz)
            or self.reference_frequency_mhz <= 0
        ):
            raise ValueError(
                "PowerCharacterization reference_frequency_mhz must be finite and > 0."
            )
        if not math.isfinite(self.p_idle_w) or self.p_idle_w < 0:
            raise ValueError("PowerCharacterization p_idle_w must be finite and >= 0.")
        if not math.isfinite(self.p_active_w) or self.p_active_w < 0:
            raise ValueError("PowerCharacterization p_active_w must be finite and >= 0.")
        if self.p_active_w < self.p_idle_w:
            raise ValueError("PowerCharacterization p_active_w must be >= p_idle_w.")
        for name, value in (
            ("voltage_v", self.voltage_v),
            ("temperature_c", self.temperature_c),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"PowerCharacterization {name} must be finite.")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(
                "PowerCharacterization metadata must be a dict when provided."
            )


@dataclass(frozen=True)
class IPBlock:
    id: str
    type: str
    area: float
    throughput: float
    delay: float
    fmax_mhz: float | None = None
    capacity_bits: int | None = None
    bandwidth_bits: int | None = None
    metadata: dict[str, Any] | None = None
    power_model: PowerCharacterization | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("IPBlock id must not be empty.")
        if not self.type.strip():
            raise ValueError("IPBlock type must not be empty.")
        if self.area < 0:
            raise ValueError("IPBlock area must be >= 0.")
        if self.delay < 0:
            raise ValueError("IPBlock delay must be >= 0.")
        if self.throughput <= 0:
            raise ValueError("IPBlock throughput must be > 0.")
        if self.fmax_mhz is not None and (
            not math.isfinite(self.fmax_mhz) or self.fmax_mhz <= 0
        ):
            raise ValueError("IPBlock fmax_mhz must be > 0 when provided.")
        if self.capacity_bits is not None and self.capacity_bits < 0:
            raise ValueError("IPBlock capacity_bits must be >= 0 when provided.")
        if self.bandwidth_bits is not None and self.bandwidth_bits < 0:
            raise ValueError("IPBlock bandwidth_bits must be >= 0 when provided.")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("IPBlock metadata must be a dict when provided.")
        if self.power_model is not None and not isinstance(
            self.power_model, PowerCharacterization
        ):
            raise ValueError(
                "IPBlock power_model must be a PowerCharacterization when provided."
            )
