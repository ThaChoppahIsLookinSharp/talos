from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AbstractComponent:
    name: str
    type: str
    count: int = 1
    required_capacity_bits: int | None = None
    required_bandwidth_bits: int | None = None
    attributes: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AbstractComponent name must not be empty.")
        if not self.type.strip():
            raise ValueError("AbstractComponent type must not be empty.")
        if self.count < 1:
            raise ValueError("AbstractComponent count must be >= 1.")
        if self.required_capacity_bits is not None and self.required_capacity_bits < 0:
            raise ValueError("AbstractComponent required_capacity_bits must be >= 0 when provided.")
        if self.required_bandwidth_bits is not None and self.required_bandwidth_bits < 0:
            raise ValueError("AbstractComponent required_bandwidth_bits must be >= 0 when provided.")
        if self.attributes is not None and not isinstance(self.attributes, dict):
            raise ValueError("AbstractComponent attributes must be a dict when provided.")


@dataclass(frozen=True)
class AbstractAccelerator:
    name: str
    components: list[AbstractComponent]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AbstractAccelerator name must not be empty.")
        if not self.components:
            raise ValueError("AbstractAccelerator requires at least one component.")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("AbstractAccelerator component names must be unique.")
