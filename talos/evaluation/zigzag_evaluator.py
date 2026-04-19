from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import io
import logging
import math
from pathlib import Path
from typing import Any
import yaml


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

    Genome structure (index-based, suited for NSGA-II):
        0 -> pe_x
        1 -> pe_y
        2 -> rf_size_bits
        3 -> rf_bw_bits
        4 -> gb_size_bits
        5 -> gb_bw_bits
        6 -> gb_served_dims_code
        7 -> dram_bw_bits

    The genome values are expected to be floating-point indexes coming
    from the optimizer. They are decoded into discrete architectural
    values before calling ZigZag.
    """

    GENE_CATALOGS: dict[str, list[int]] = {
        "pe_x": [4, 8, 16, 32],
        "pe_y": [4, 8, 16, 32],
        "rf_size_bits": [64, 128, 256, 512, 1024, 2048],
        "rf_bw_bits": [8, 16, 32, 64, 128, 256],
        "gb_size_bits": [8192, 16384, 32768, 65536, 131072],
        "gb_bw_bits": [64, 128, 256, 512, 1024],
        "gb_served_dims_code": [0, 1, 2, 3],
        "dram_bw_bits": [64, 128, 256, 512, 1024, 2048],
    }

    GENE_NAMES = [
        "pe_x",
        "pe_y",
        "rf_size_bits",
        "rf_bw_bits",
        "gb_size_bits",
        "gb_bw_bits",
        "gb_served_dims_code",
        "dram_bw_bits",
    ]

    SERVED_DIMS_MAP: dict[int, list[str]] = {
        0: [],
        1: ["D1"],
        2: ["D2"],
        3: ["D1", "D2"],
    }

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
            cfg = self._decode_features(genome)
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

    def _decode_features(self, genome: list[float]) -> dict[str, int]:
        if len(genome) != len(self.GENE_NAMES):
            raise ValueError(
                f"Expected {len(self.GENE_NAMES)} genes, got {len(genome)}."
            )

        decoded: dict[str, int] = {}

        for idx, gene_name in enumerate(self.GENE_NAMES):
            catalog = self.GENE_CATALOGS[gene_name]

            try:
                gene_value = float(genome[idx])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Gene {idx} ({gene_name}) must be convertible to float; "
                    f"got {genome[idx]!r}."
                ) from exc

            if not math.isfinite(gene_value):
                raise ValueError(
                    f"Gene {idx} ({gene_name}) must be finite; got {gene_value!r}."
                )

            catalog_idx = int(round(gene_value))
            catalog_idx = max(0, min(catalog_idx, len(catalog) - 1))
            decoded[gene_name] = catalog[catalog_idx]

        return decoded

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

    def _build_accelerator(self, cfg: dict[str, int]) -> dict[str, Any]:
        pe_x = cfg["pe_x"]
        pe_y = cfg["pe_y"]

        rf_bw = cfg["rf_bw_bits"]
        gb_bw = cfg["gb_bw_bits"]
        dram_bw = cfg["dram_bw_bits"]
        gb_served_dims = self.SERVED_DIMS_MAP[cfg["gb_served_dims_code"]]

        accelerator = {
            "name": "talos_candidate",
            "operational_array": {
                "is_imc": False,
                "unit_energy": 1.0,
                "unit_area": 1.0,
                "dimensions": ["D1", "D2"],
                "sizes": [pe_x, pe_y],
                "imc_type": None,
                "adc_resolution": 0,
                "bit_serial_precision": None,
            },
            "memories": {
                "rf_i1": {
                    "size": cfg["rf_size_bits"],
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
                            rf_bw,
                            ["I1, tl", "I1, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_i2": {
                    "size": cfg["rf_size_bits"],
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
                            rf_bw,
                            ["I2, tl", "I2, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_o": {
                    "size": cfg["rf_size_bits"],
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
                            rf_bw,
                            ["O, fh", "O, fl", "O, th", "O, tl"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "gb": {
                    "size": cfg["gb_size_bits"],
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
                            gb_bw,
                            [
                                "I1, tl", "I1, fh",
                                "I2, tl", "I2, fh",
                                "O, fh", "O, fl", "O, th", "O, tl",
                            ],
                        )
                    ],
                    "served_dimensions": gb_served_dims,
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
                            dram_bw,
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

    def _extract_area(self, cme: Any, cfg: dict[str, int]) -> float:
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

    def _estimate_area(self, cfg: dict[str, int]) -> float:
        """
        Very rough placeholder area model.

        Replace this later with your own Level-2 IP characterization model.
        """
        mac_count = cfg["pe_x"] * cfg["pe_y"]

        mac_area = mac_count * 1.0
        rf_area = mac_count * cfg["rf_size_bits"] * 0.001
        gb_area = cfg["gb_size_bits"] * 0.0005

        return float(mac_area + rf_area + gb_area)
