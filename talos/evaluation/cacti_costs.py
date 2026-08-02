from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import IO, Any

from talos.architecture.genome import GB_BW_OPTIONS, GB_SIZE_OPTIONS
from talos.ip.ip_characterization import IPBlock, PowerCharacterization


DEFAULT_TECHNOLOGY_NM = 65.0
REFERENCE_GB_CAPACITY_BYTES = 128 * 1024
REFERENCE_WORD_BITS = 16
RF_TO_MAC = 1.0
GB_TO_MAC = 6.0
DRAM_TO_MAC = 200.0
CACTI_MIN_WORDS = 32


@dataclass(frozen=True)
class CactiMemoryCost:
    capacity_bits: int
    bandwidth_bits: int
    read_energy_pj_per_access: float
    write_energy_pj_per_access: float
    cacti_capacity_bits: int | None = None


@dataclass(frozen=True)
class Level1EnergyCalibration:
    technology_nm: float
    reference_gb_capacity_bytes: int
    reference_word_bits: int
    reference_gb_read_energy_pj: float
    reference_gb_write_energy_pj: float
    mac_energy_pj: float
    gb_costs: tuple[CactiMemoryCost, ...]

    @property
    def dram_energy_pj_per_16b(self) -> float:
        return self.mac_energy_pj * DRAM_TO_MAC

    def rf_energy_pj_per_access(self, bandwidth_bits: int) -> float:
        return self._scaled_word_energy(
            self.mac_energy_pj * RF_TO_MAC,
            bandwidth_bits,
            "RF",
        )

    def dram_energy_pj_per_access(self, bandwidth_bits: int) -> float:
        return self._scaled_word_energy(
            self.dram_energy_pj_per_16b,
            bandwidth_bits,
            "DRAM",
        )

    def gb_cost(
        self,
        capacity_bits: int,
        bandwidth_bits: int,
    ) -> CactiMemoryCost:
        for cost in self.gb_costs:
            if (
                cost.capacity_bits == capacity_bits
                and cost.bandwidth_bits == bandwidth_bits
            ):
                return cost
        raise ValueError(
            "Missing CACTI global-buffer cost for "
            f"capacity={capacity_bits} bits, bandwidth={bandwidth_bits} bits."
        )

    def to_dict(
        self,
        *,
        dram_bus_width_bits: int,
        dram_power_model: PowerCharacterization,
    ) -> dict[str, Any]:
        overrides = [
            {
                "logical_capacity_bytes": cost.capacity_bits // 8,
                "bandwidth_bits": cost.bandwidth_bits,
                "cacti_capacity_bytes": cost.cacti_capacity_bits // 8,
            }
            for cost in self.gb_costs
            if cost.cacti_capacity_bits is not None
            and cost.cacti_capacity_bits != cost.capacity_bits
        ]
        return {
            "technology_nm": self.technology_nm,
            "reference_gb_capacity_bytes": self.reference_gb_capacity_bytes,
            "reference_word_bits": self.reference_word_bits,
            "reference_gb_read_energy_pj": self.reference_gb_read_energy_pj,
            "reference_gb_write_energy_pj": self.reference_gb_write_energy_pj,
            "mac_energy_pj": self.mac_energy_pj,
            "dram_energy_pj_per_16b": self.dram_energy_pj_per_16b,
            "dram_bus_width_bits": dram_bus_width_bits,
            "dram_energy_pj_per_access": self.dram_energy_pj_per_access(
                dram_bus_width_bits
            ),
            "derived_dram_p_idle_w": dram_power_model.p_idle_w,
            "derived_dram_p_active_w": dram_power_model.p_active_w,
            "gb_physical_capacity_overrides": overrides,
            "ratios": {
                "rf_to_mac": RF_TO_MAC,
                "gb_to_mac": GB_TO_MAC,
                "dram_to_mac": DRAM_TO_MAC,
            },
        }

    def _scaled_word_energy(
        self,
        energy_pj_per_reference_word: float,
        bandwidth_bits: int,
        role: str,
    ) -> float:
        if bandwidth_bits <= 0:
            raise ValueError(f"{role} bandwidth must be > 0.")
        return (
            energy_pj_per_reference_word
            * bandwidth_bits
            / self.reference_word_bits
        )


def characterize_level1_energy(
    cacti_master_path: str | Path | None = None,
    *,
    technology_nm: float = DEFAULT_TECHNOLOGY_NM,
) -> Level1EnergyCalibration:
    if not math.isfinite(technology_nm) or technology_nm <= 0:
        raise ValueError("CACTI technology_nm must be finite and > 0.")
    technology_um = technology_nm / 1000
    source = (
        Path(cacti_master_path).resolve()
        if cacti_master_path is not None
        else _included_cacti_master()
    )
    executable = source / "cacti"
    if not executable.is_file():
        raise FileNotFoundError(
            "CACTI executable is missing: "
            f"role=all, capacity=various, bandwidth=various, "
            f"technology={technology_um}, path={executable}"
        )

    with tempfile.TemporaryDirectory(prefix="talos_cacti_") as temporary:
        master = Path(temporary) / "cacti_master"
        shutil.copytree(source, master)
        costs = tuple(
            _characterize_memory(
                master,
                role="global_buffer",
                capacity_bits=capacity_bits,
                bandwidth_bits=bandwidth_bits,
                technology_um=technology_um,
            )
            for capacity_bits in GB_SIZE_OPTIONS
            for bandwidth_bits in GB_BW_OPTIONS
        )
        reference = _characterize_memory(
            master,
            role="reference_global_buffer",
            capacity_bits=REFERENCE_GB_CAPACITY_BYTES * 8,
            bandwidth_bits=REFERENCE_WORD_BITS,
            technology_um=technology_um,
        )

    mac_energy_pj = (
        reference.read_energy_pj_per_access
        + reference.write_energy_pj_per_access
    ) / 2 / GB_TO_MAC
    if not math.isfinite(mac_energy_pj) or mac_energy_pj <= 0:
        raise ValueError(
            "Invalid derived MAC energy: "
            f"role=reference_global_buffer, "
            f"capacity={REFERENCE_GB_CAPACITY_BYTES} bytes, "
            f"bandwidth={REFERENCE_WORD_BITS} bits, "
            f"technology={technology_um}, energy={mac_energy_pj}"
        )
    return Level1EnergyCalibration(
        technology_nm=technology_nm,
        reference_gb_capacity_bytes=REFERENCE_GB_CAPACITY_BYTES,
        reference_word_bits=REFERENCE_WORD_BITS,
        reference_gb_read_energy_pj=reference.read_energy_pj_per_access,
        reference_gb_write_energy_pj=reference.write_energy_pj_per_access,
        mac_energy_pj=mac_energy_pj,
        gb_costs=costs,
    )


def parse_cacti_output(
    source: str | Path | IO[str],
    *,
    expected_capacity_bytes: int,
    expected_bandwidth_bits: int,
) -> tuple[float, float]:
    close = False
    if isinstance(source, (str, Path)):
        handle = Path(source).open(encoding="utf-8", newline="")
        close = True
    else:
        handle = source
    try:
        rows = [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None
            }
            for row in csv.DictReader(handle)
            if any(value and value.strip() for value in row.values())
        ]
    finally:
        if close:
            handle.close()

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one CACTI result row, found {len(rows)}."
        )
    row = rows[0]
    required = (
        "Capacity (bytes)",
        "Output width (bits)",
        "Dynamic read energy (nJ)",
        "Dynamic write energy (nJ)",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"CACTI result is missing columns: {', '.join(missing)}.")

    try:
        capacity_bytes = int(float(row["Capacity (bytes)"]))
        bandwidth_bits = int(float(row["Output width (bits)"]))
        read_energy_pj = float(row["Dynamic read energy (nJ)"]) * 1000
        write_energy_pj = float(row["Dynamic write energy (nJ)"]) * 1000
    except (TypeError, ValueError) as exc:
        raise ValueError("CACTI result contains an invalid required value.") from exc

    if capacity_bytes != expected_capacity_bytes:
        raise ValueError(
            f"CACTI capacity mismatch: expected {expected_capacity_bytes}, "
            f"got {capacity_bytes} bytes."
        )
    if bandwidth_bits != expected_bandwidth_bits:
        raise ValueError(
            f"CACTI bandwidth mismatch: expected {expected_bandwidth_bits}, "
            f"got {bandwidth_bits} bits."
        )
    for name, value in (
        ("read", read_energy_pj),
        ("write", write_energy_pj),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"CACTI {name} energy must be finite and positive, got {value}."
            )
    return read_energy_pj, write_energy_pj


def calibrate_synthetic_dram_power_model(
    power_model: PowerCharacterization,
    *,
    dram_bandwidth_bits: int,
    accesses_per_cycle: float,
    calibration: Level1EnergyCalibration,
) -> PowerCharacterization:
    if power_model.source != "synthetic":
        return power_model
    if dram_bandwidth_bits <= 0:
        raise ValueError("Synthetic DRAM bandwidth must be > 0.")
    if not math.isfinite(accesses_per_cycle) or accesses_per_cycle <= 0:
        raise ValueError(
            "Synthetic DRAM accesses_per_cycle must be finite and > 0."
        )
    energy_j = (
        calibration.dram_energy_pj_per_access(dram_bandwidth_bits) * 1e-12
    )
    dynamic_power_w = (
        energy_j
        * power_model.reference_frequency_mhz
        * 1_000_000
        * accesses_per_cycle
    )
    return replace(
        power_model,
        p_active_w=power_model.p_idle_w + dynamic_power_w,
    )


def calibrate_synthetic_dram_ip(
    dram_ip: IPBlock,
    calibration: Level1EnergyCalibration,
) -> IPBlock:
    if dram_ip.type != "dram":
        raise ValueError(f"Expected a DRAM IPBlock, got {dram_ip.type!r}.")
    if dram_ip.power_model is None:
        raise ValueError(f"DRAM IPBlock {dram_ip.id!r} has no power_model.")
    if dram_ip.power_model.source != "synthetic":
        return dram_ip
    try:
        accesses_per_cycle = float((dram_ip.metadata or {})["accesses_per_cycle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Synthetic DRAM IPBlock {dram_ip.id!r} requires valid "
            "metadata 'accesses_per_cycle'."
        ) from exc
    if dram_ip.bandwidth_bits is None:
        raise ValueError(
            f"Synthetic DRAM IPBlock {dram_ip.id!r} requires bandwidth_bits."
        )
    return replace(
        dram_ip,
        power_model=calibrate_synthetic_dram_power_model(
            dram_ip.power_model,
            dram_bandwidth_bits=dram_ip.bandwidth_bits,
            accesses_per_cycle=accesses_per_cycle,
            calibration=calibration,
        ),
    )


def write_energy_calibration(
    path: str | Path,
    calibration: Level1EnergyCalibration,
    *,
    dram_bus_width_bits: int,
    dram_power_model: PowerCharacterization,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            calibration.to_dict(
                dram_bus_width_bits=dram_bus_width_bits,
                dram_power_model=dram_power_model,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _included_cacti_master() -> Path:
    import zigzag

    return (
        Path(inspect.getfile(zigzag)).resolve().parent
        / "cacti"
        / "cacti_master"
    )


def _characterize_memory(
    cacti_master: Path,
    *,
    role: str,
    capacity_bits: int,
    bandwidth_bits: int,
    technology_um: float,
) -> CactiMemoryCost:
    if capacity_bits <= 0 or capacity_bits % 8:
        raise ValueError(f"{role} capacity must be a positive whole byte.")
    if bandwidth_bits <= 0 or bandwidth_bits % 8:
        raise ValueError(f"{role} bandwidth must be a positive whole byte.")

    # ponytail: CACTI requires at least 32 words; keep the logical size and
    # model only the unavoidable physical overprovisioning.
    cacti_capacity_bits = max(
        capacity_bits,
        bandwidth_bits * CACTI_MIN_WORDS,
    )
    capacity_bytes = cacti_capacity_bits // 8
    logical_capacity_bytes = capacity_bits // 8
    line_size_bytes = bandwidth_bits // 8
    generated = cacti_master / "talos_self_gen"
    generated.mkdir(exist_ok=True)
    config_path = generated / (
        f"{role}_{logical_capacity_bytes}_{capacity_bytes}_{bandwidth_bits}.cfg"
    )
    _write_cacti_config(
        config_path,
        capacity_bytes=capacity_bytes,
        bandwidth_bits=bandwidth_bits,
        line_size_bytes=line_size_bytes,
        technology_um=technology_um,
    )
    relative_config = config_path.relative_to(cacti_master)
    process = subprocess.run(
        [str(cacti_master / "cacti"), "-infile", str(relative_config)],
        cwd=cacti_master,
        text=True,
        capture_output=True,
        check=False,
    )
    context = (
        f"role={role}, capacity={logical_capacity_bytes} bytes, "
        f"cacti_capacity={capacity_bytes} bytes, "
        f"bandwidth={bandwidth_bits} bits, technology={technology_um}, "
        f"return_code={process.returncode}"
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise RuntimeError(
            f"CACTI failed ({context}): {' | '.join(detail[-5:])}"
        )

    output_path = Path(f"{config_path}.out")
    if not output_path.is_file():
        raise RuntimeError(f"CACTI produced no output ({context}).")
    try:
        read_energy_pj, write_energy_pj = parse_cacti_output(
            output_path,
            expected_capacity_bytes=capacity_bytes,
            expected_bandwidth_bits=bandwidth_bits,
        )
    except ValueError as exc:
        raise RuntimeError(f"Invalid CACTI output ({context}): {exc}") from exc
    return CactiMemoryCost(
        capacity_bits=capacity_bits,
        bandwidth_bits=bandwidth_bits,
        read_energy_pj_per_access=read_energy_pj,
        write_energy_pj_per_access=write_energy_pj,
        cacti_capacity_bits=cacti_capacity_bits,
    )


def _write_cacti_config(
    path: Path,
    *,
    capacity_bytes: int,
    bandwidth_bits: int,
    line_size_bytes: int,
    technology_um: float,
) -> None:
    from zigzag.cacti.cacti_master.cacti_config_creator import CactiConfig

    values: dict[str, Any] = {
        "mem_type": '"ram"',
        "cache_size": capacity_bytes,
        "IO_bus_width": bandwidth_bits,
        "line_size": line_size_bytes,
        "associativity": 1,
        "ex_rd_port": 0,
        "ex_wr_port": 0,
        "rd_wr_port": 1,
        "bank_count": 1,
        "technology": technology_um,
    }
    config = CactiConfig()
    lines = [
        option["string"] + str(values.get(name, option["default"])) + "\n"
        for name, option in config.config_options.items()
    ]
    config.write_config(lines, str(path))
