from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

#TODO: auto_cost_extraction=False-> means BW genomes
#are broken. Either move this to level 2 and remove
#these genomes or turn this true (possible more changes)
#Congelado por ahora, luego ya vamos viendo
#osea, congelada la implementación digo.

@dataclass(frozen=True)
class GeneSpec:
    name: str
    options: list[Any]


@dataclass(frozen=True)
class ArchitectureConfig:
    pe_x: int
    pe_y: int
    rf_size_bits: int
    rf_bw_bits: int
    gb_size_bits: int
    gb_bw_bits: int
    gb_served_dims: list[str]
    dram_bw_bits: int


PE_X_OPTIONS = [4, 8, 16, 32]
PE_Y_OPTIONS = [4, 8, 16, 32]
RF_SIZE_OPTIONS = [64, 128, 256, 512, 1024, 2048]
RF_BW_OPTIONS = [8, 16, 32, 64, 128, 256]
GB_SIZE_OPTIONS = [8192, 16384, 32768, 65536, 131072]
GB_BW_OPTIONS = [64, 128, 256, 512, 1024]
GB_SERVED_DIMS_OPTIONS = [[], ["D1"], ["D2"], ["D1", "D2"]]
DRAM_BW_OPTIONS = [64, 128, 256, 512, 1024, 2048]


GENOME_SPEC = [
    GeneSpec("pe_x_code", PE_X_OPTIONS),
    GeneSpec("pe_y_code", PE_Y_OPTIONS),
    GeneSpec("rf_size_code", RF_SIZE_OPTIONS),
    GeneSpec("rf_bw_code", RF_BW_OPTIONS),
    GeneSpec("gb_size_code", GB_SIZE_OPTIONS),
    GeneSpec("gb_bw_code", GB_BW_OPTIONS),
    GeneSpec("gb_served_dims_code", GB_SERVED_DIMS_OPTIONS),
    GeneSpec("dram_bw_code", DRAM_BW_OPTIONS),
]
GENOME_LENGTH = len(GENOME_SPEC)


def gene_names() -> list[str]:
    return [spec.name for spec in GENOME_SPEC]


def gene_bounds() -> list[tuple[int, int]]:
    return [(0, len(spec.options) - 1) for spec in GENOME_SPEC]


def default_genome() -> list[int]:
    # Preserve the current manual example while making it part of the formal spec.
    return [2, 2, 3, 2, 3, 2, 3, 3]


def decode_genome(genome: list[float]) -> ArchitectureConfig:
    if len(genome) != GENOME_LENGTH:
        raise ValueError(f"Expected {GENOME_LENGTH} genes, got {len(genome)}.")

    decoded_options: list[Any] = []

    for idx, (gene, spec) in enumerate(zip(genome, GENOME_SPEC, strict=True)):
        try:
            gene_value = float(gene)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Gene {idx} ({spec.name}) must be convertible to float; got {gene!r}."
            ) from exc

        if not math.isfinite(gene_value):
            raise ValueError(
                f"Gene {idx} ({spec.name}) must be finite; got {gene_value!r}."
            )

        option_idx = int(round(gene_value))
        option_idx = max(0, min(option_idx, len(spec.options) - 1))
        option = spec.options[option_idx]

        if isinstance(option, list):
            option = list(option)

        decoded_options.append(option)

    return ArchitectureConfig(
        pe_x=decoded_options[0],
        pe_y=decoded_options[1],
        rf_size_bits=decoded_options[2],
        rf_bw_bits=decoded_options[3],
        gb_size_bits=decoded_options[4],
        gb_bw_bits=decoded_options[5],
        gb_served_dims=decoded_options[6],
        dram_bw_bits=decoded_options[7],
    )
