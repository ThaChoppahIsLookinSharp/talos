from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from zigzag.api import get_hardware_performance_zigzag
except ImportError:  # pragma: no cover
    get_hardware_performance_zigzag = None


@dataclass
class EvaluationResult:
    latency: float
    energy: float
    area: float
    valid: bool


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
        mapping: dict[str, Any] | None = None,
        opt: str = "EDP",
        use_mock_area: bool = True,
    ) -> None:
        self.workload = workload
        self.mapping = mapping or self._default_mapping()
        self.opt = opt
        self.use_mock_area = use_mock_area

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        print("***EV CALLED WITH GENOME", genome)
        return EvaluationResult(
            latency=100.0,
            energy=50.0,
            area=30.0,
            valid=True,
        )
#        try:
#            cfg = self._decode_features(genome)
#            accelerator = self._build_accelerator(cfg)
#
#            if get_hardware_performance_zigzag is None:
#                raise RuntimeError(
#                    "zigzag is not installed. Install it or keep this class in mock mode."
#                )
#
#            energy, latency, cme = get_hardware_performance_zigzag(
#                workload=self.workload,
#                accelerator=accelerator,
#                mapping=self.mapping,
#                opt=self.opt,
#            )
#
#            area = self._extract_area(cme, cfg)
#
#            return EvaluationResult(
#                latency=float(latency),
#                energy=float(energy),
#                area=float(area),
#                valid=True,
#            )
#
#        except Exception:
#            # Invalid individuals are penalized instead of crashing the search.
#            return EvaluationResult(
#                latency=float("inf"),
#                energy=float("inf"),
#                area=float("inf"),
#                valid=False,
#            )

    def _decode_features(self, genome: list[float]) -> dict[str, int]:
        if len(genome) != len(self.GENE_NAMES):
            raise ValueError(
                f"Expected {len(self.GENE_NAMES)} genes, got {len(genome)}."
            )

        decoded: dict[str, int] = {}

        for idx, gene_name in enumerate(self.GENE_NAMES):
            catalog = self.GENE_CATALOGS[gene_name]
            catalog_idx = int(round(float(genome[idx])))
            catalog_idx = max(0, min(catalog_idx, len(catalog) - 1))
            decoded[gene_name] = catalog[catalog_idx]

        return decoded

    def _build_accelerator(self, cfg: dict[str, int]) -> dict[str, Any]:
        """
        Minimal accelerator model.

        This is intentionally simple so TALOS can start evolving
        meaningful candidates before you refine the hardware model.
        """
        pe_x = cfg["pe_x"]
        pe_y = cfg["pe_y"]
        gb_served_dims = self.SERVED_DIMS_MAP[cfg["gb_served_dims_code"]]

        accelerator = {
            "name": "talos_candidate",
            "cores": [
                {
                    "id": 0,
                    "operational_array": {
                        "dimensions": {"D1": pe_x, "D2": pe_y},
                        "unit": {
                            "type": "Multiplier",
                            "energy": 1.0,
                            "area": 1.0,
                        },
                    },
                    "memories": [
                        {
                            "name": "rf",
                            "size": cfg["rf_size_bits"],
                            "r_bw": cfg["rf_bw_bits"],
                            "w_bw": cfg["rf_bw_bits"],
                            "r_cost": 1.0,
                            "w_cost": 1.0,
                            "area": 1.0,
                            "r_port": 1,
                            "w_port": 1,
                            "rw_port": 0,
                            "served_dimensions": [],
                        },
                        {
                            "name": "gb",
                            "size": cfg["gb_size_bits"],
                            "r_bw": cfg["gb_bw_bits"],
                            "w_bw": cfg["gb_bw_bits"],
                            "r_cost": 10.0,
                            "w_cost": 10.0,
                            "area": 10.0,
                            "r_port": 1,
                            "w_port": 1,
                            "rw_port": 0,
                            "served_dimensions": gb_served_dims,
                        },
                        {
                            "name": "dram",
                            "size": 10**12,
                            "r_bw": cfg["dram_bw_bits"],
                            "w_bw": cfg["dram_bw_bits"],
                            "r_cost": 1000.0,
                            "w_cost": 1000.0,
                            "area": 0.0,
                            "r_port": 1,
                            "w_port": 1,
                            "rw_port": 0,
                            "served_dimensions": ["D1", "D2"],
                        },
                    ],
                }
            ],
        }

        return accelerator

    def _default_mapping(self) -> dict[str, Any]:
        """
        Minimal mapping stub.
        Adjust operand names if your workload/model uses different names.
        """
        return {
            "memory_operand_links": {
                "O": "O",
                "W": "I2",
                "I": "I1",
            }
        }

    def _extract_area(self, cme: Any, cfg: dict[str, int]) -> float:
        """
        First try to recover area from ZigZag's returned object.
        Fall back to a very rough analytical estimate.
        """
        # Try common attribute names first.
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

        # Fallback model for early TALOS integration.
        return self._estimate_area(cfg)

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
