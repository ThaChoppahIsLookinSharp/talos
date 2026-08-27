from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from zigzag.hardware.architecture.memory_port import DataDirection


@dataclass(frozen=True)
class LayerActivity:
    layer_id: str
    latency_cycles: float
    mac_count: float
    spatially_used_pes: int
    memory_accesses: dict[str, float]
    operand_precision_bits: dict[str, int] = field(default_factory=dict)
    operand_numeric_formats: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.layer_id.strip():
            raise ValueError("LayerActivity layer_id must not be empty.")
        if not math.isfinite(self.latency_cycles) or self.latency_cycles <= 0:
            raise ValueError("LayerActivity latency_cycles must be finite and > 0.")
        if not math.isfinite(self.mac_count) or self.mac_count < 0:
            raise ValueError("LayerActivity mac_count must be finite and >= 0.")
        if self.spatially_used_pes < 0:
            raise ValueError("LayerActivity spatially_used_pes must be >= 0.")
        if not isinstance(self.memory_accesses, dict):
            raise ValueError("LayerActivity memory_accesses must be a dict.")
        for name, accesses in self.memory_accesses.items():
            if not name.strip():
                raise ValueError("LayerActivity memory names must not be empty.")
            if not math.isfinite(accesses) or accesses < 0:
                raise ValueError(
                    f"LayerActivity accesses for {name!r} must be finite and >= 0."
                )
        if not isinstance(self.operand_precision_bits, dict) or any(
            not name.strip() or not isinstance(bits, int) or bits <= 0
            for name, bits in self.operand_precision_bits.items()
        ):
            raise ValueError(
                "LayerActivity operand precisions must be named positive integers."
            )
        if not isinstance(self.operand_numeric_formats, dict) or any(
            not name.strip()
            or not isinstance(numeric_format, str)
            or not numeric_format.strip()
            for name, numeric_format in (
                self.operand_numeric_formats.items()
            )
        ):
            raise ValueError(
                "LayerActivity numeric formats must be named strings."
            )


@dataclass(frozen=True)
class WorkloadActivityProfile:
    layers: tuple[LayerActivity, ...]

    @property
    def total_latency_cycles(self) -> float:
        return sum(layer.latency_cycles for layer in self.layers)

    @property
    def total_mac_count(self) -> float:
        return sum(layer.mac_count for layer in self.layers)

    @property
    def total_dram_accesses(self) -> float:
        return sum(layer.memory_accesses.get("dram", 0.0) for layer in self.layers)


@dataclass(frozen=True)
class WorkloadPerformance:
    layer_cycles_mapping: tuple[tuple[str, float], ...]
    workload_cycles_per_inference: float
    workload_latency_s: float
    workload_throughput_ips: float


def compute_workload_performance(
    profile: WorkloadActivityProfile,
    reference_frequency_mhz: float,
) -> WorkloadPerformance:
    """Batch-1 sequential inference performance at the characterized clock."""
    if (
        not math.isfinite(reference_frequency_mhz)
        or reference_frequency_mhz <= 0
    ):
        raise ValueError("reference_frequency_mhz must be finite and > 0.")
    layer_cycles = tuple(
        (layer.layer_id, layer.latency_cycles) for layer in profile.layers
    )
    total_cycles = sum(cycles for _layer_id, cycles in layer_cycles)
    if not math.isfinite(total_cycles) or total_cycles <= 0:
        raise ValueError("workload_cycles_per_inference must be finite and > 0.")
    latency_s = total_cycles / (reference_frequency_mhz * 1_000_000.0)
    throughput_ips = 1.0 / latency_s
    if not all(
        math.isfinite(value) and value > 0
        for value in (latency_s, throughput_ips)
    ):
        raise ValueError("Workload performance must be finite and > 0.")
    return WorkloadPerformance(
        layer_cycles_mapping=layer_cycles,
        workload_cycles_per_inference=total_cycles,
        workload_latency_s=latency_s,
        workload_throughput_ips=throughput_ips,
    )


_MEMORY_BINDINGS = {
    ("I1", "rf_i1"): "rf_i1",
    ("I2", "rf_i2"): "rf_i2",
    ("O", "rf_o"): "rf_o",
    ("I1", "gb"): "gb",
    ("I2", "gb"): "gb",
    ("O", "gb"): "gb",
}


def extract_workload_activity_profile(
    cme: Any,
    *,
    operand_numeric_formats_by_layer: Mapping[
        int,
        Mapping[str, str],
    ]
    | None = None,
) -> WorkloadActivityProfile:
    """Normalize ZigZag 3.8.5 per-layer CMEs into Talos activity data."""
    raw_layers = cme if isinstance(cme, list) else [cme]
    layers = tuple(
        _extract_layer_activity(
            _unwrap_cme(item),
            operand_numeric_formats_by_layer,
        )
        for item in raw_layers
    )
    return WorkloadActivityProfile(layers=layers)


def _unwrap_cme(value: Any) -> Any:
    # ZigZag 3.8.5 returns CMEs directly, despite its API annotation allowing tuples.
    if isinstance(value, tuple) and value and hasattr(value[0], "latency_total2"):
        return value[0]
    return value


def _extract_layer_activity(
    cme: Any,
    operand_numeric_formats_by_layer: Mapping[
        int,
        Mapping[str, str],
    ]
    | None = None,
) -> LayerActivity:
    layer = cme.layer
    memory_accesses: dict[str, float] = {}

    for layer_operand, accesses_per_level in cme.memory_word_access.items():
        memory_operand = cme.memory_operand_links.layer_to_mem_op(layer_operand)
        levels = cme.mem_hierarchy_dict[memory_operand]
        for level, accesses in zip(levels, accesses_per_level, strict=True):
            memory_name = str(level.memory_instance.name)
            # ZigZag charges these four fields once as the physical reads and
            # writes of this memory level in calc_memory_energy_cost().
            physical_accesses = sum(
                accesses.get(direction) for direction in DataDirection
            )
            if memory_name.lower() == "dram":
                target = "dram"
            else:
                target = _MEMORY_BINDINGS.get((str(memory_operand), memory_name))
                if target is None:
                    raise ValueError(
                        f"Unsupported ZigZag memory binding: {memory_operand}, {memory_name}."
                    )
            memory_accesses[target] = memory_accesses.get(target, 0.0) + float(
                physical_accesses
            )

    return LayerActivity(
        layer_id=str(getattr(layer, "name", None) or layer.id),
        latency_cycles=_latency_cycles(cme),
        mac_count=float(layer.total_mac_count),
        spatially_used_pes=_spatially_used_pes(cme),
        memory_accesses=memory_accesses,
        operand_precision_bits={
            str(operand): int(bits)
            for operand, bits in getattr(
                getattr(layer, "operand_precision", None),
                "data",
                {},
            ).items()
        },
        operand_numeric_formats=dict(
            (operand_numeric_formats_by_layer or {}).get(
                int(layer.id),
                {},
            )
        ),
    )


def _latency_cycles(cme: Any) -> float:
    for name in ("latency_total2", "latency_total1", "latency_total0"):
        value = getattr(cme, name, None)
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    raise ValueError("ZigZag CME has no positive finite latency_total value.")


def _spatially_used_pes(cme: Any) -> int:
    unit_counts = cme.spatial_mapping.unit_count.values()
    maximum = max(
        (float(value) for counts in unit_counts for value in counts),
        default=0.0,
    )
    return math.ceil(maximum - 1e-9)
