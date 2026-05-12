from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from talos.architecture.abstract_accelerator import AbstractComponent
from talos.ip.ip_characterization import IPBlock


class IPPool:
    def __init__(self, ip_blocks: list[IPBlock]) -> None:
        if not ip_blocks:
            raise ValueError("IPPool requires at least one IPBlock.")
        self.ip_blocks = list(ip_blocks)

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
            try:
                ip_blocks.append(IPBlock(**raw_ip))
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
