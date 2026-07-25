from __future__ import annotations

from math import prod

from talos.architecture.abstract_accelerator import AbstractAccelerator, AbstractComponent
from talos.architecture.genome import ArchitectureConfig


def abstract_accelerator_from_level1_config(config: ArchitectureConfig) -> AbstractAccelerator:
    pe_count = config.pe_x * config.pe_y
    gb_count = prod(
        size
        for dimension, size in (("D1", config.pe_x), ("D2", config.pe_y))
        if dimension not in config.gb_served_dims
    )
    components = [
        AbstractComponent(
            name="pe_array",
            type="pe",
            count=pe_count,
            attributes={"pe_x": config.pe_x, "pe_y": config.pe_y},
        ),
        AbstractComponent(
            name="rf_i1",
            type="register_file",
            count=pe_count,
            required_capacity_bits=config.rf_size_bits,
            required_bandwidth_bits=config.rf_bw_bits,
            attributes={"operand": "I1", "scope": "per_pe"},
        ),
        AbstractComponent(
            name="rf_i2",
            type="register_file",
            count=pe_count,
            required_capacity_bits=config.rf_size_bits,
            required_bandwidth_bits=config.rf_bw_bits,
            attributes={"operand": "I2", "scope": "per_pe"},
        ),
        AbstractComponent(
            name="rf_o",
            type="register_file",
            count=pe_count,
            required_capacity_bits=config.rf_size_bits,
            required_bandwidth_bits=config.rf_bw_bits,
            attributes={"operand": "O", "scope": "per_pe"},
        ),
        AbstractComponent(
            name="gb",
            type="global_buffer",
            count=gb_count,
            required_capacity_bits=config.gb_size_bits,
            required_bandwidth_bits=config.gb_bw_bits,
            attributes={"served_dimensions": list(config.gb_served_dims)},
        ),
    ]
    return AbstractAccelerator(name="level1_imported_accelerator", components=components)
