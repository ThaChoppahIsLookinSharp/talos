from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_IP_TYPES = {
    "pe",
    "register_file",
    "global_buffer",
    "dram",
    "interconnect",
}


@dataclass(frozen=True)
class IPBlock:
    id: str
    type: str
    area: float
    power: float
    throughput: float
    delay: float
    capacity_bits: int | None = None
    bandwidth_bits: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("IPBlock id must not be empty.")
        if not self.type.strip():
            raise ValueError("IPBlock type must not be empty.")
        if self.area < 0:
            raise ValueError("IPBlock area must be >= 0.")
        if self.power < 0:
            raise ValueError("IPBlock power must be >= 0.")
        if self.delay < 0:
            raise ValueError("IPBlock delay must be >= 0.")
        if self.throughput <= 0:
            raise ValueError("IPBlock throughput must be > 0.")
        if self.capacity_bits is not None and self.capacity_bits < 0:
            raise ValueError("IPBlock capacity_bits must be >= 0 when provided.")
        if self.bandwidth_bits is not None and self.bandwidth_bits < 0:
            raise ValueError("IPBlock bandwidth_bits must be >= 0 when provided.")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("IPBlock metadata must be a dict when provided.")
