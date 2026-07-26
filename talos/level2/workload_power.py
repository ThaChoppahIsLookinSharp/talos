from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from talos.evaluation.workload_activity import LayerActivity, WorkloadActivityProfile
from talos.ip.ip_characterization import PowerCharacterization
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent


UTILIZATION_TOLERANCE = 1e-5
MEMORY_COMPONENT_NAMES = ("rf_i1", "rf_i2", "rf_o", "gb")
POWER_REQUIREMENTS_ERROR = (
    "Workload energy/power exploration requires a workload activity profile and "
    "compatible p_idle/p_active characterizations for all candidate IPs."
)


@dataclass(frozen=True)
class WorkloadPowerResult:
    power_w: float
    energy_j: float
    latency_s: float


def evaluate_workload_power(
    implemented: ImplementedAccelerator,
    profile: WorkloadActivityProfile,
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

    operating_frequency_mhz = _validate_selected_characterizations(
        implemented.components
    )
    energy_j = profile.total_dram_access_energy_j
    latency_s = 0.0

    for layer in profile.layers:
        layer_power_w = _pe_power(pe_component, layer)
        for name in MEMORY_COMPONENT_NAMES:
            layer_power_w += _memory_power(
                components[name],
                layer,
                layer.memory_accesses.get(name, 0.0),
            )
        layer_latency_s = layer.latency_cycles / (
            operating_frequency_mhz * 1_000_000.0
        )
        energy_j += layer_power_w * layer_latency_s
        latency_s += layer_latency_s

    if latency_s <= 0:
        raise ValueError("Workload power requires positive total latency.")
    return WorkloadPowerResult(
        power_w=energy_j / latency_s,
        energy_j=energy_j,
        latency_s=latency_s,
    )


def validate_power_aware_exploration(
    spec: Any,
    profile: WorkloadActivityProfile | None,
) -> None:
    if profile is None:
        raise ValueError(f"{POWER_REQUIREMENTS_ERROR} Activity profile is missing.")

    characterized: list[tuple[str, PowerCharacterization, float | None]] = []
    for gene in spec.genes:
        for ip in gene.candidates:
            if ip.power_model is None:
                raise ValueError(
                    f"{POWER_REQUIREMENTS_ERROR} IP {ip.id!r} has no power_model."
                )
            if gene.component.type != "pe":
                _positive_metadata(ip.metadata, "accesses_per_cycle", ip.id)
            characterized.append((ip.id, ip.power_model, ip.fmax_mhz))

    _validate_characterizations(characterized)

    pe_counts = [
        gene.component.count for gene in spec.genes if gene.component.type == "pe"
    ]
    if len(pe_counts) != 1:
        raise ValueError(f"{POWER_REQUIREMENTS_ERROR} Exactly one PE component is required.")
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
            f"Layer {layer.layer_id!r} executes MACs but uses no PEs spatially."
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
    return _component_power_w(count, utilization, _power_model(component))


def _memory_utilization(
    *,
    accesses: float,
    latency_cycles: float,
    instance_count: int,
    accesses_per_cycle: float,
    memory_name: str = "memory",
    layer_id: str = "",
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
    )


def _validate_utilization(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and >= 0.")
    if value > 1.0 + UTILIZATION_TOLERANCE:
        raise ValueError(f"{label} exceeds 1: {value}.")
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
        raise ValueError(f"Selected IP {component.ip.id!r} has no power_model.")
    return model


def _positive_metadata(
    metadata: dict[str, Any] | None,
    name: str,
    ip_id: str,
) -> float:
    try:
        value = float((metadata or {})[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"IP {ip_id!r} requires positive metadata {name!r}.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"IP {ip_id!r} requires positive metadata {name!r}.")
    return value


def _validate_selected_characterizations(
    components: list[ImplementedComponent],
) -> float:
    characterized = [
        (component.ip.id, _power_model(component), component.ip.fmax_mhz)
        for component in components
    ]
    return _validate_characterizations(characterized)


def _validate_characterizations(
    characterized: list[tuple[str, PowerCharacterization, float | None]],
) -> float:
    if not characterized:
        raise ValueError("No characterized IPs were provided.")
    for field in ("corner", "voltage_v", "temperature_c"):
        if len({getattr(model, field) for _id, model, _fmax in characterized}) != 1:
            raise ValueError(f"Selected IPs have incompatible power {field} values.")

    fmax_values: list[float] = []
    for ip_id, _model, fmax_mhz in characterized:
        if fmax_mhz is None:
            raise ValueError(f"Selected IP {ip_id!r} has no fmax_mhz.")
        fmax_values.append(fmax_mhz)
    return min(fmax_values)
