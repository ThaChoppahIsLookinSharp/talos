from __future__ import annotations

from dataclasses import dataclass
import math

from talos.architecture.genome import (
    ArchitectureConfig,
    GB_BW_OPTIONS,
    GB_SIZE_OPTIONS,
    RF_BW_OPTIONS,
    RF_SIZE_OPTIONS,
)
from talos.architecture.abstract_accelerator import AbstractComponent
from talos.architecture.level1_importer import (
    abstract_accelerator_from_level1_config,
)
from talos.ip.ip_characterization import IPBlock
from talos.ip.ip_pool import IPPool


RF_ROLES = ("rf_i1", "rf_i2", "rf_o")


@dataclass(frozen=True)
class Level1AreaCalibration:
    pe_rf_area_mm2: dict[tuple[int, int], float | None]
    gb_area_mm2: dict[tuple[int, int], float | None]
    min_frequency_mhz: float | None = None

    def area_mm2(self, config: ArchitectureConfig) -> float:
        rf_key = (config.rf_size_bits, config.rf_bw_bits)
        gb_key = (config.gb_size_bits, config.gb_bw_bits)
        pe_rf_area = self.pe_rf_area_mm2[rf_key]
        gb_area = self.gb_area_mm2[gb_key]
        if pe_rf_area is None or gb_area is None:
            raise ValueError(
                "No physical IP combination satisfies the Level 1 architecture."
            )

        components = abstract_accelerator_from_level1_config(config).components
        counts = {component.name: component.count for component in components}
        return float(
            counts["pe_array"] * pe_rf_area
            + counts["gb"] * gb_area
        )


def characterize_level1_area(
    pool: IPPool,
    min_frequency_mhz: float | None = None,
) -> Level1AreaCalibration:
    if min_frequency_mhz is not None and (
        not math.isfinite(min_frequency_mhz) or min_frequency_mhz <= 0
    ):
        raise ValueError("min_frequency_mhz must be finite and > 0 when provided.")

    try:
        pe_candidates = _frequency_compatible(
            pool.find_compatible(AbstractComponent(name="pe_array", type="pe")),
            min_frequency_mhz,
        )
    except ValueError:
        pe_candidates = []
    by_id = {ip.id: ip for ip in pool.ip_blocks}
    pe_rf_costs: dict[tuple[int, int], float | None] = {}
    for capacity_bits in RF_SIZE_OPTIONS:
        for bandwidth_bits in RF_BW_OPTIONS:
            component = AbstractComponent(
                name="rf",
                type="register_file",
                required_capacity_bits=capacity_bits,
                required_bandwidth_bits=bandwidth_bits,
            )
            try:
                compatible_rfs = pool.find_compatible(component)
            except ValueError:
                compatible_rfs = []
            compatible_ids = {ip.id for ip in compatible_rfs}
            standalone_rfs = _frequency_compatible(
                compatible_rfs,
                min_frequency_mhz,
            )
            bundle_areas = []
            for pe in pe_candidates:
                area = _pe_rf_bundle_area(
                    pe,
                    by_id,
                    compatible_ids,
                    standalone_rfs,
                )
                if area is not None:
                    bundle_areas.append(area)
            pe_rf_costs[(capacity_bits, bandwidth_bits)] = (
                min(bundle_areas) if bundle_areas else None
            )

    gb_costs: dict[tuple[int, int], float | None] = {}
    for capacity_bits in GB_SIZE_OPTIONS:
        for bandwidth_bits in GB_BW_OPTIONS:
            component = AbstractComponent(
                name="gb",
                type="global_buffer",
                required_capacity_bits=capacity_bits,
                required_bandwidth_bits=bandwidth_bits,
            )
            try:
                candidates = _frequency_compatible(
                    pool.find_compatible(component),
                    min_frequency_mhz,
                )
            except ValueError:
                candidates = []
            gb_costs[(capacity_bits, bandwidth_bits)] = (
                min(ip.area for ip in candidates) if candidates else None
            )

    return Level1AreaCalibration(
        pe_rf_area_mm2=pe_rf_costs,
        gb_area_mm2=gb_costs,
        min_frequency_mhz=min_frequency_mhz,
    )


def _frequency_compatible(
    candidates: list[IPBlock],
    min_frequency_mhz: float | None,
) -> list[IPBlock]:
    if min_frequency_mhz is None:
        return candidates
    return [
        ip
        for ip in candidates
        if ip.fmax_mhz is not None and ip.fmax_mhz >= min_frequency_mhz
    ]


def _pe_rf_bundle_area(
    pe: IPBlock,
    by_id: dict[str, IPBlock],
    compatible_rf_ids: set[str],
    standalone_rfs: list[IPBlock],
) -> float | None:
    uncovered_roles = len(RF_ROLES)
    for role, rf_id in pe.included_rfs.items():
        if role not in RF_ROLES or rf_id not in compatible_rf_ids:
            return None
        if by_id[rf_id].type != "register_file":
            return None
        uncovered_roles -= 1

    if uncovered_roles and not standalone_rfs:
        return None
    return pe.area + uncovered_roles * min(
        (rf.area for rf in standalone_rfs),
        default=0.0,
    )
