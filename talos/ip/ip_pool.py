from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import yaml

from talos.architecture.abstract_accelerator import AbstractComponent
from talos.ip.ip_characterization import IPBlock, PowerCharacterization


class IPPool:
    def __init__(self, ip_blocks: list[IPBlock]) -> None:
        if not ip_blocks:
            raise ValueError("IPPool requires at least one IPBlock.")
        self.ip_blocks = list(ip_blocks)
        for ip in self.ip_blocks:
            if ip.type != "dram":
                continue
            try:
                accesses_per_cycle = float(
                    (ip.metadata or {})["accesses_per_cycle"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"DRAM IPBlock {ip.id!r} requires positive metadata "
                    "'accesses_per_cycle'."
                ) from exc
            if (
                ip.bandwidth_bits is None
                or ip.bandwidth_bits <= 0
                or ip.power_model is None
                or not math.isfinite(accesses_per_cycle)
                or accesses_per_cycle <= 0
            ):
                raise ValueError(
                    f"DRAM IPBlock {ip.id!r} requires positive bandwidth_bits, "
                    "power_model and metadata 'accesses_per_cycle'."
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IPPool":
        yaml_path = Path(path)
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("IP pool YAML must contain a top-level mapping.")
        raw_ips = data.get("ips")
        if not isinstance(raw_ips, list):
            raise ValueError("IP pool YAML must contain an 'ips' list.")

        ip_blocks: list[IPBlock] = []
        for index, raw_ip in enumerate(raw_ips):
            if not isinstance(raw_ip, dict):
                raise ValueError(f"IP entry {index} must be a mapping.")
            if "power" in raw_ip:
                raise ValueError(
                    "Legacy field 'power' is not supported. Use "
                    "power_model.p_idle_w and power_model.p_active_w."
                )
            try:
                values = dict(raw_ip)
                raw_power_model = values.get("power_model")
                if raw_power_model is not None:
                    if not isinstance(raw_power_model, dict):
                        raise ValueError(
                            f"IP entry {index} power_model must be a mapping."
                        )
                    values["power_model"] = PowerCharacterization(**raw_power_model)
                ip_blocks.append(IPBlock(**values))
            except TypeError as exc:
                raise ValueError(f"Invalid IP entry at index {index}: {raw_ip!r}") from exc
        return cls(ip_blocks)

    def by_type(self, ip_type: str) -> list[IPBlock]:
        return [ip for ip in self.ip_blocks if ip.type == ip_type]

    def find_compatible(self, component: AbstractComponent) -> list[IPBlock]:
        compatible: list[IPBlock] = []
        for ip in self.by_type(component.type):
            if (
                component.required_capacity_bits is not None
                and (ip.capacity_bits is None or ip.capacity_bits < component.required_capacity_bits)
            ):
                continue
            if (
                component.required_bandwidth_bits is not None
                and (ip.bandwidth_bits is None or ip.bandwidth_bits < component.required_bandwidth_bits)
            ):
                continue
            compatible.append(ip)

        if not compatible:
            raise ValueError(
                "No compatible IPBlock found for component "
                f"name={component.name!r}, type={component.type!r}, "
                f"required_capacity_bits={component.required_capacity_bits!r}, "
                f"required_bandwidth_bits={component.required_bandwidth_bits!r}."
            )
        return compatible

    def to_dict(self) -> dict[str, Any]:
        return {"ips": [asdict(ip) for ip in self.ip_blocks]}
