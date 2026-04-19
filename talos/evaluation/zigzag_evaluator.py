from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Any
import yaml

from talos.architecture.genome import ArchitectureConfig, decode_genome


@dataclass
class EvaluationResult:
    latency: float
    energy: float
    area: float
    valid: bool
    error_message: str | None = None


class ZigZagEvaluator:
    """
    TALOS -> ZigZag compatibility layer.

    Genome semantics live in talos.architecture.genome. This evaluator
    only consumes decoded architecture configs and runs ZigZag.
    """

    def __init__(
        self,
        workload: str,
        mapping: list[dict[str, Any]] | None = None,
        opt: str = "EDP",
        use_mock_area: bool = True,
        workdir: str | None = None,
        debug: bool = False,
    ) -> None:
        self.workload = workload
        self.mapping = mapping if mapping is not None else self._default_mapping()
        self.opt = opt
        self.use_mock_area = use_mock_area
        self.workdir = (
            Path(workdir) if workdir is not None else Path.cwd() / ".talos_zigzag"
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.mapping_yaml_path = self._write_mapping_yaml(self.mapping)

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        try:
            cfg = decode_genome(genome)
            accelerator = self._build_accelerator(cfg)
            accelerator_yaml_path = self._write_accelerator_yaml(accelerator)

            if self.debug:
                self._print_debug_yaml(accelerator_yaml_path)
                energy, latency, cme = self._run_zigzag(accelerator_yaml_path)
            else:
                with self._quiet_zigzag():
                    energy, latency, cme = self._run_zigzag(accelerator_yaml_path)

            area = self._extract_area(cme, cfg)

            return EvaluationResult(
                latency=float(latency),
                energy=float(energy),
                area=float(area),
                valid=True,
            )

        except Exception as exc:
            if self.debug:
                import traceback

                print("ZigZag evaluation failed:")
                traceback.print_exc()

            return EvaluationResult(
                latency=float("inf"),
                energy=float("inf"),
                area=float("inf"),
                valid=False,
                error_message=str(exc),
            )

    def _write_mapping_yaml(self, mapping: list[dict[str, Any]]) -> str:
        mapping_path = self.workdir / "mapping.yaml"
        with open(mapping_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapping, f, sort_keys=False)

        return str(mapping_path)

    def _write_accelerator_yaml(self, accelerator: dict[str, Any]) -> str:
        accelerator_path = self.workdir / "accelerator.yaml"
        with open(accelerator_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(accelerator, f, sort_keys=False)

        return str(accelerator_path)

    def _run_zigzag(self, accelerator_yaml_path: str) -> tuple[float, float, Any]:
        from zigzag.api import get_hardware_performance_zigzag

        return get_hardware_performance_zigzag(
            workload=self.workload,
            accelerator=accelerator_yaml_path,
            mapping=self.mapping_yaml_path,
            opt=self.opt,
        )

    @contextlib.contextmanager
    def _quiet_zigzag(self) -> Iterator[None]:
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                yield
        finally:
            logging.disable(previous_disable_level)

    def _print_debug_yaml(self, accelerator_yaml_path: str) -> None:
        print("Using mapping file:", self.mapping_yaml_path)
        print(Path(self.mapping_yaml_path).read_text(encoding="utf-8"))

        print("Using accelerator file:", accelerator_yaml_path)
        print(Path(accelerator_yaml_path).read_text(encoding="utf-8"))

    def _rw_port(self, name: str, bw: int, allocations: list[str]) -> dict[str, Any]:
        return {
            "name": name,
            "type": "read_write",
            "bandwidth_min": bw,
            "bandwidth_max": bw,
            "allocation": allocations,
        }

    def _build_accelerator(self, cfg: ArchitectureConfig) -> dict[str, Any]:
        accelerator = {
            "name": "talos_candidate",
            "operational_array": {
                "is_imc": False,
                "unit_energy": 1.0,
                "unit_area": 1.0,
                "dimensions": ["D1", "D2"],
                "sizes": [cfg.pe_x, cfg.pe_y],
                "imc_type": None,
                "adc_resolution": 0,
                "bit_serial_precision": None,
            },
            "memories": {
                "rf_i1": {
                    "size": cfg.rf_size_bits,
                    "r_cost": 1.0,
                    "w_cost": 1.0,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I1"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["I1, tl", "I1, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_i2": {
                    "size": cfg.rf_size_bits,
                    "r_cost": 1.0,
                    "w_cost": 1.0,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I2"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["I2, tl", "I2, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_o": {
                    "size": cfg.rf_size_bits,
                    "r_cost": 1.0,
                    "w_cost": 1.0,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["O, fh", "O, fl", "O, th", "O, tl"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "gb": {
                    "size": cfg.gb_size_bits,
                    "r_cost": 10.0,
                    "w_cost": 10.0,
                    "area": 10.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.gb_bw_bits,
                            [
                                "I1, tl", "I1, fh",
                                "I2, tl", "I2, fh",
                                "O, fh", "O, fl", "O, th", "O, tl",
                            ],
                        )
                    ],
                    "served_dimensions": cfg.gb_served_dims,
                },
                "dram": {
                    "size": 10**12,
                    "r_cost": 1000.0,
                    "w_cost": 1000.0,
                    "area": 0.0,
                    "latency": 1,
                    "mem_type": "dram",
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.dram_bw_bits,
                            [
                                "I1, tl", "I1, fh",
                                "I2, tl", "I2, fh",
                                "O, fh", "O, fl", "O, th", "O, tl",
                            ],
                        )
                    ],
                    "served_dimensions": ["D1", "D2"],
                },
            },
        }

        return accelerator

    def _default_mapping(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "default",
                "memory_operand_links": {
                    "O": "O",
                    "W": "I2",
                    "I": "I1",
                },
            }
        ]

    def _extract_area(self, cme: Any, cfg: ArchitectureConfig) -> float:
        """
        First try to recover area from ZigZag's returned object.
        Fall back to a very rough analytical estimate.
        """
        candidate_attrs = [
            "area_total",
            "total_area",
            "area",
        ]

        for attr in candidate_attrs:
            if hasattr(cme, attr):
                value = getattr(cme, attr)
                if isinstance(value, (int, float)):
                    return float(value)

        if isinstance(cme, dict):
            for key in candidate_attrs:
                if key in cme and isinstance(cme[key], (int, float)):
                    return float(cme[key])

        if self.use_mock_area:
            return self._estimate_area(cfg)

        raise ValueError("ZigZag did not return an area value.")

    def _estimate_area(self, cfg: ArchitectureConfig) -> float:
        """
        Very rough placeholder area model.

        Replace this later with your own Level-2 IP characterization model.
        """
        mac_count = cfg.pe_x * cfg.pe_y

        mac_area = mac_count * 1.0
        rf_area = mac_count * cfg.rf_size_bits * 0.001
        gb_area = cfg.gb_size_bits * 0.0005

        return float(mac_area + rf_area + gb_area)
