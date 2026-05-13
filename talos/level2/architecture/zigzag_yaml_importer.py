from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from talos.level2.architecture.abstract_accelerator import (
    AbstractAccelerator,
    AbstractComponent,
)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _infer_bandwidth_bits(memory: dict[str, Any]) -> int | None:
    ports = memory.get("ports")
    if not isinstance(ports, list):
        return None
    maxima: list[int] = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        bw = _safe_int(port.get("bandwidth_max"))
        if bw is not None:
            maxima.append(bw)
    return max(maxima) if maxima else None


def _infer_component_type(name: str, memory: dict[str, Any]) -> str:
    mem_type = str(memory.get("mem_type", "")).lower()
    lowered_name = name.lower()
    if lowered_name.startswith("rf"):
        return "register_file"
    if lowered_name.startswith("gb"):
        return "global_buffer"
    if lowered_name.startswith("dram") or mem_type == "dram":
        return "dram"
    served_dimensions = memory.get("served_dimensions")
    if served_dimensions == []:
        return "register_file"
    if mem_type == "dram":
        return "dram"
    return "global_buffer"


def abstract_accelerator_from_zigzag_yaml(path: str) -> AbstractAccelerator:
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ZigZag accelerator YAML must contain a top-level mapping.")

    components: list[AbstractComponent] = []
    accelerator_name = str(data.get("name", yaml_path.stem))

    operational_array = data.get("operational_array")
    if isinstance(operational_array, dict):
        sizes = operational_array.get("sizes")
        if isinstance(sizes, list) and sizes:
            dims = [_safe_int(value) for value in sizes]
            if all(value is not None for value in dims):
                pe_count = 1
                for dim in dims:
                    pe_count *= int(dim)
                components.append(
                    AbstractComponent(
                        name="pe_array",
                        type="pe",
                        count=pe_count,
                        attributes={"operational_array": operational_array},
                    )
                )

    pe_count = next((component.count for component in components if component.name == "pe_array"), 1)
    memories = data.get("memories")
    if isinstance(memories, dict):
        for name, memory in memories.items():
            if not isinstance(memory, dict):
                continue
            component_type = _infer_component_type(str(name), memory)
            count = pe_count if component_type == "register_file" and memory.get("served_dimensions") == [] else 1
            required_capacity_bits = None if component_type == "dram" else _safe_int(memory.get("size"))
            components.append(
                AbstractComponent(
                    name=str(name),
                    type=component_type,
                    count=count,
                    required_capacity_bits=required_capacity_bits,
                    required_bandwidth_bits=_infer_bandwidth_bits(memory),
                    attributes={
                        "mem_type": memory.get("mem_type"),
                        "operands": memory.get("operands"),
                        "served_dimensions": memory.get("served_dimensions"),
                        "source_memory": memory,
                    },
                )
            )

    if not components:
        raise ValueError("Could not identify any useful components in the ZigZag accelerator YAML.")

    return AbstractAccelerator(name=accelerator_name, components=components)
