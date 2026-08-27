from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import warnings

from talos.evaluation.workload_activity import (
    LayerActivity,
    WorkloadActivityProfile,
    compute_workload_performance,
)
from talos.ip.ip_characterization import IPBlock, PowerCharacterization
from talos.level2.genome import (
    ImplementedAccelerator,
    ImplementedComponent,
    physical_components,
)


# Aggregate ZigZag transfers and synthetic port rates are approximate.
UTILIZATION_TOLERANCE = 0.2
PE_CAPACITY_TOLERANCE = 1e-4
MEMORY_COMPONENT_NAMES = ("rf_i1", "rf_i2", "rf_o", "gb")
WORKLOAD_REQUIREMENTS_ERROR = (
    "Workload-aware exploration requires a workload activity profile and "
    "compatible p_idle/p_active characterizations for all candidate IPs."
)


@dataclass(frozen=True)
class WorkloadPowerResult:
    power_w: float
    energy_j: float
    dram_energy_j: float
    layer_cycles_mapping: tuple[tuple[str, float], ...]
    workload_cycles_per_inference: float
    workload_latency_s: float
    workload_throughput_ips: float
    reference_frequency_mhz: float
    reference_voltage_v: float | None


def evaluate_workload_power(
    implemented: ImplementedAccelerator,
    profile: WorkloadActivityProfile,
    dram_ip: IPBlock,
) -> WorkloadPowerResult:
    components = {
        component.abstract_component.name: component
        for component in implemented.components
    }
    pe_components = [
        component
        for component in implemented.components
        if component.abstract_component.type == "pe"
    ]
    if len(pe_components) != 1:
        raise ValueError("Workload power requires exactly one PE component.")
    pe_component = pe_components[0]
    for name in MEMORY_COMPONENT_NAMES:
        if name not in components:
            raise ValueError(f"Workload power requires component {name!r}.")
    physical_components(implemented.components)

    operating_point = _validate_selected_characterizations(
        implemented.components,
        dram_ip,
    )
    performance = compute_workload_performance(
        profile,
        operating_point.reference_frequency_mhz,
    )
    energy_j = 0.0
    dram_energy_j = 0.0

    for layer in profile.layers:
        layer_power_w = _pe_power(pe_component, layer)
        for name in MEMORY_COMPONENT_NAMES:
            layer_power_w += _memory_power(
                components[name],
                layer,
                layer.memory_accesses.get(name, 0.0),
            )
        dram_power_w = _dram_power(dram_ip, layer)
        layer_power_w += dram_power_w
        layer_latency_s = layer.latency_cycles / (
            operating_point.reference_frequency_mhz * 1_000_000.0
        )
        energy_j += layer_power_w * layer_latency_s
        dram_energy_j += dram_power_w * layer_latency_s

    return WorkloadPowerResult(
        power_w=energy_j / performance.workload_latency_s,
        energy_j=energy_j,
        dram_energy_j=dram_energy_j,
        layer_cycles_mapping=performance.layer_cycles_mapping,
        workload_cycles_per_inference=(
            performance.workload_cycles_per_inference
        ),
        workload_latency_s=performance.workload_latency_s,
        workload_throughput_ips=performance.workload_throughput_ips,
        reference_frequency_mhz=operating_point.reference_frequency_mhz,
        reference_voltage_v=operating_point.voltage_v,
    )


def validate_workload_aware_exploration(
    spec: Any,
    profile: WorkloadActivityProfile | None,
    dram_ip: IPBlock,
) -> None:
    if profile is None:
        raise ValueError(f"{WORKLOAD_REQUIREMENTS_ERROR} Activity profile is missing.")

    _warn_pe_format_mismatches(spec, profile)

    if dram_ip.power_model is None:
        raise ValueError(
            f"{WORKLOAD_REQUIREMENTS_ERROR} DRAM IP {dram_ip.id!r} has no power_model."
        )
    if dram_ip.bandwidth_bits is None or dram_ip.bandwidth_bits <= 0:
        raise ValueError(
            f"{WORKLOAD_REQUIREMENTS_ERROR} DRAM IP {dram_ip.id!r} requires "
            "positive bandwidth_bits."
        )
    _positive_metadata(dram_ip.metadata, "accesses_per_cycle", dram_ip.id)
    _validate_characterizations(
        [(dram_ip.id, dram_ip.power_model, dram_ip.fmax_mhz)],
        require_operable=False,
    )
    compute_workload_performance(
        profile,
        dram_ip.power_model.reference_frequency_mhz,
    )

    pe_counts = [
        gene.component.count for gene in spec.genes if gene.component.type == "pe"
    ]
    if len(pe_counts) != 1:
        raise ValueError(f"{WORKLOAD_REQUIREMENTS_ERROR} Exactly one PE component is required.")
    for layer in profile.layers:
        if layer.spatially_used_pes > pe_counts[0]:
            raise ValueError(
                f"Layer {layer.layer_id!r} uses {layer.spatially_used_pes} PEs "
                f"spatially but the architecture has {pe_counts[0]}."
            )
        if layer.mac_count > 0 and layer.spatially_used_pes == 0:
            raise ValueError(
                f"Layer {layer.layer_id!r} executes MACs but uses no PEs spatially."
            )


def _pe_power(
    component: ImplementedComponent,
    layer: LayerActivity,
) -> float:
    count = component.abstract_component.count
    active_count = layer.spatially_used_pes
    if active_count > count:
        raise ValueError(
            f"Layer {layer.layer_id!r} uses {active_count} PEs spatially but "
            f"the implementation has {count}."
        )
    if layer.mac_count > 0 and active_count == 0:
        raise ValueError(
            f"insufficient_pe_capacity: layer {layer.layer_id!r} executes MACs "
            "but uses no PEs spatially."
        )
    required_macs_per_cycle = layer.mac_count / layer.latency_cycles
    available_macs_per_cycle = active_count * _positive_metadata(
        component.ip.metadata,
        "macs_per_cycle",
        component.ip.id,
    )
    if required_macs_per_cycle > available_macs_per_cycle + PE_CAPACITY_TOLERANCE:
        raise ValueError(
            f"insufficient_pe_capacity: layer {layer.layer_id!r} requires "
            f"{required_macs_per_cycle} MAC/cycle but selected PE instances "
            f"provide {available_macs_per_cycle}."
        )
    model = _power_model(component)
    return (
        active_count * model.p_active_w
        + (count - active_count) * model.p_idle_w
    )


def _memory_power(
    component: ImplementedComponent,
    layer: LayerActivity,
    accesses: float,
) -> float:
    # ponytail: ZigZag profile is aggregate per level; preserve read/write/port
    # demand in the adapter before adding directional shared-port validation.
    count = component.abstract_component.count
    source_width_bits = component.abstract_component.required_bandwidth_bits
    selected_width_bits = component.ip.bandwidth_bits
    if source_width_bits is not None and selected_width_bits is not None:
        if selected_width_bits <= 0:
            raise ValueError(
                f"Selected memory IP {component.ip.id!r} requires positive bandwidth_bits."
            )
        accesses *= source_width_bits / selected_width_bits
    rate = _positive_metadata(
        component.ip.metadata,
        "accesses_per_cycle",
        component.ip.id,
    )
    utilization = _memory_utilization(
        accesses=accesses,
        latency_cycles=layer.latency_cycles,
        instance_count=count,
        accesses_per_cycle=rate,
        memory_name=component.abstract_component.name,
        layer_id=layer.layer_id,
    )
    if component.covered_by_pe_id is not None:
        model = _power_model(component)
        return count * utilization * (model.p_active_w - model.p_idle_w)
    return _component_power_w(count, utilization, _power_model(component))


def _dram_power(ip: IPBlock, layer: LayerActivity) -> float:
    model = ip.power_model
    if model is None:
        raise ValueError(f"DRAM IP {ip.id!r} has no power_model.")
    utilization = _memory_utilization(
        accesses=layer.memory_accesses.get("dram", 0.0),
        latency_cycles=layer.latency_cycles,
        instance_count=1,
        accesses_per_cycle=_positive_metadata(
            ip.metadata,
            "accesses_per_cycle",
            ip.id,
        ),
        memory_name="dram",
        layer_id=layer.layer_id,
    )
    return _component_power_w(1, utilization, model)


def _memory_utilization(
    *,
    accesses: float,
    latency_cycles: float,
    instance_count: int,
    accesses_per_cycle: float,
    memory_name: str = "memory",
    layer_id: str = "",
    tolerance: float = UTILIZATION_TOLERANCE,
) -> float:
    if accesses_per_cycle <= 0 or not math.isfinite(accesses_per_cycle):
        raise ValueError("accesses_per_cycle must be finite and > 0.")
    if instance_count <= 0:
        if accesses == 0:
            return 0.0
        raise ValueError(
            f"Memory {memory_name!r} has accesses but no physical instances."
        )
    if accesses == 0:
        return 0.0
    utilization = accesses / (
        latency_cycles * instance_count * accesses_per_cycle
    )
    return _validate_utilization(
        utilization,
        f"Memory {memory_name!r} utilization for layer {layer_id!r}",
        tolerance,
    )


def _validate_utilization(
    value: float,
    label: str,
    tolerance: float = UTILIZATION_TOLERANCE,
) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and >= 0.")
    if value > 1.0 + tolerance:
        direction = (
            "insufficient_memory_bandwidth"
            if "dram" not in label.lower()
            else "insufficient_dram_bandwidth"
        )
        raise ValueError(f"{direction}: {label} exceeds 1: {value}.")
    return min(value, 1.0)


def _component_power_w(
    instance_count: int,
    utilization: float,
    model: PowerCharacterization,
) -> float:
    return instance_count * (
        model.p_idle_w
        + utilization
        * (model.p_active_w - model.p_idle_w)
    )


def _power_model(component: ImplementedComponent) -> PowerCharacterization:
    model = component.ip.power_model
    if model is None:
        raise ValueError(
            f"missing_characterization: selected IP "
            f"{component.ip.id!r} has no power_model."
        )
    return model


def _positive_metadata(
    metadata: dict[str, Any] | None,
    name: str,
    ip_id: str,
) -> float:
    try:
        value = float((metadata or {})[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"missing_characterization: IP {ip_id!r} requires positive "
            f"metadata {name!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"missing_characterization: IP {ip_id!r} requires positive "
            f"metadata {name!r}."
        )
    return value


def _warn_pe_format_mismatches(
    spec: Any,
    profile: WorkloadActivityProfile,
) -> None:
    required_formats = {
        numeric_format
        for layer in profile.layers
        for operand, numeric_format in (
            layer.operand_numeric_formats.items()
        )
        if operand in {"I", "W"}
    }
    required_bits = {
        bits
        for layer in profile.layers
        for operand, bits in layer.operand_precision_bits.items()
        if operand in {"I", "W"}
    }
    if not required_formats and not required_bits:
        return

    mismatches: list[str] = []
    seen: set[str] = set()
    for gene in spec.genes:
        if gene.component.type != "pe":
            continue
        for candidate in gene.candidates:
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            metadata = candidate.metadata or {}
            numeric_format = metadata.get("numeric_format")
            precision_bits = metadata.get("precision_bits")
            format_matches = (
                not required_formats
                or (
                    isinstance(numeric_format, str)
                    and required_formats == {numeric_format}
                )
            )
            bits_match = (
                not required_bits
                or (
                    isinstance(precision_bits, (int, float))
                    and math.isfinite(float(precision_bits))
                    and required_bits == {float(precision_bits)}
                )
            )
            if format_matches and bits_match:
                continue
            mismatches.append(
                f"{candidate.id} "
                f"({numeric_format or 'unknown'}, "
                f"{precision_bits or 'unknown'} bits)"
            )

    if not mismatches:
        return
    formats = ", ".join(sorted(required_formats)) or "unknown"
    bits = ", ".join(str(value) for value in sorted(required_bits))
    warnings.warn(
        "Workload PE format requirement "
        f"({formats}; {bits or 'unknown'} bits) does not match: "
        f"{', '.join(mismatches)}. These candidates are retained by "
        "policy.",
        RuntimeWarning,
        stacklevel=2,
    )


def _validate_selected_characterizations(
    components: list[ImplementedComponent],
    dram_ip: IPBlock,
) -> PowerCharacterization:
    characterized = [
        (component.ip.id, _power_model(component), component.ip.fmax_mhz)
        for component in components
    ]
    if dram_ip.power_model is None:
        raise ValueError(f"DRAM IP {dram_ip.id!r} has no power_model.")
    characterized.append((dram_ip.id, dram_ip.power_model, dram_ip.fmax_mhz))
    return _validate_characterizations(characterized)


def _validate_characterizations(
    characterized: list[tuple[str, PowerCharacterization, float | None]],
    *,
    require_operable: bool = True,
) -> PowerCharacterization:
    if not characterized:
        raise ValueError("No characterized IPs were provided.")
    for ip_id, model, _fmax in characterized:
        if model.voltage_v is None:
            raise ValueError(
                f"missing_characterization: selected IP {ip_id!r} has no "
                "reference voltage_v."
            )
    frequencies = {
        model.reference_frequency_mhz for _id, model, _fmax in characterized
    }
    if len(frequencies) != 1:
        raise ValueError(
            "incompatible_frequency_operating_point: selected IPs have "
            "different reference_frequency_mhz values."
        )
    for field in ("corner", "voltage_v", "temperature_c"):
        if len({getattr(model, field) for _id, model, _fmax in characterized}) != 1:
            code = (
                "incompatible_voltage_operating_point"
                if field == "voltage_v"
                else "incompatible_operating_point"
            )
            raise ValueError(
                f"{code}: selected IPs have different {field} values."
            )

    reference_frequency_mhz = frequencies.pop()
    for ip_id, _model, fmax_mhz in characterized:
        if fmax_mhz is None:
            raise ValueError(
                f"missing_characterization: selected IP {ip_id!r} has no fmax_mhz."
            )
        if require_operable and fmax_mhz < reference_frequency_mhz:
            raise ValueError(
                f"reference_frequency_above_fmax: selected IP {ip_id!r} "
                f"fmax_mhz {fmax_mhz} is below reference_frequency_mhz "
                f"{reference_frequency_mhz}."
            )
    return characterized[0][1]
