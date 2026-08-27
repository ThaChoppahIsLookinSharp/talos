from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from talos.architecture.genome import ArchitectureConfig
from talos.evaluation.area_calibration import characterize_level1_area
from talos.ip import IPBlock, IPPool


REPO_ROOT = Path(__file__).resolve().parents[1]


def ip(
    ip_id: str,
    ip_type: str,
    area: float,
    *,
    capacity_bits: int | None = None,
    bandwidth_bits: int | None = None,
    fmax_mhz: float | None = None,
    included_rfs: dict[str, str] | None = None,
) -> IPBlock:
    return IPBlock(
        id=ip_id,
        type=ip_type,
        area=area,
        throughput=1,
        delay=1,
        fmax_mhz=fmax_mhz,
        capacity_bits=capacity_bits,
        bandwidth_bits=bandwidth_bits,
        included_rfs=included_rfs or {},
        included_rf_power_mode=(
            "parent_idle_baseline" if included_rfs else None
        ),
    )


class Level1AreaCalibrationTests(unittest.TestCase):
    def test_precomputes_the_complete_level1_tables(self) -> None:
        pool = IPPool.from_yaml(
            REPO_ROOT / "configs" / "ip_pool_synthetic_65nm.yaml"
        )

        calibration = characterize_level1_area(pool)

        self.assertEqual(len(calibration.pe_rf_area_mm2), 36)
        self.assertEqual(len(calibration.gb_area_mm2), 25)

    def test_global_buffer_width_selects_a_different_macro(self) -> None:
        pool = IPPool(
            [
                ip("pe", "pe", 1),
                ip(
                    "rf",
                    "register_file",
                    1,
                    capacity_bits=2048,
                    bandwidth_bits=256,
                ),
                ip(
                    "gb_narrow",
                    "global_buffer",
                    2,
                    capacity_bits=131072,
                    bandwidth_bits=64,
                ),
                ip(
                    "gb_wide",
                    "global_buffer",
                    5,
                    capacity_bits=131072,
                    bandwidth_bits=1024,
                ),
            ]
        )

        calibration = characterize_level1_area(pool)

        self.assertEqual(calibration.gb_area_mm2[(8192, 64)], 2)
        self.assertEqual(calibration.gb_area_mm2[(8192, 128)], 5)

    def test_composite_pe_is_counted_once_and_must_cover_rf_requirements(self) -> None:
        pool = IPPool(
            [
                ip("pe_plain", "pe", 1),
                ip(
                    "pe_tile",
                    "pe",
                    3,
                    included_rfs={
                        "rf_i1": "rf_integrated",
                        "rf_i2": "rf_integrated",
                        "rf_o": "rf_integrated",
                    },
                ),
                ip(
                    "rf_integrated",
                    "register_file",
                    50,
                    capacity_bits=512,
                    bandwidth_bits=64,
                ),
                ip(
                    "rf_standalone",
                    "register_file",
                    1,
                    capacity_bits=2048,
                    bandwidth_bits=256,
                ),
                ip(
                    "gb",
                    "global_buffer",
                    1,
                    capacity_bits=131072,
                    bandwidth_bits=1024,
                ),
            ]
        )

        calibration = characterize_level1_area(pool)

        self.assertEqual(calibration.pe_rf_area_mm2[(512, 64)], 3)
        self.assertEqual(calibration.pe_rf_area_mm2[(1024, 64)], 4)

    def test_frequency_filter_excludes_cheaper_slow_components(self) -> None:
        pool = IPPool(
            [
                ip("pe_slow", "pe", 1, fmax_mhz=100),
                ip("pe_fast", "pe", 2, fmax_mhz=500),
                ip(
                    "rf_slow",
                    "register_file",
                    1,
                    capacity_bits=2048,
                    bandwidth_bits=256,
                    fmax_mhz=100,
                ),
                ip(
                    "rf_fast",
                    "register_file",
                    2,
                    capacity_bits=2048,
                    bandwidth_bits=256,
                    fmax_mhz=500,
                ),
                ip(
                    "gb_slow",
                    "global_buffer",
                    1,
                    capacity_bits=131072,
                    bandwidth_bits=1024,
                    fmax_mhz=100,
                ),
                ip(
                    "gb_fast",
                    "global_buffer",
                    2,
                    capacity_bits=131072,
                    bandwidth_bits=1024,
                    fmax_mhz=500,
                ),
            ]
        )

        calibration = characterize_level1_area(pool, min_frequency_mhz=200)

        self.assertEqual(calibration.pe_rf_area_mm2[(64, 8)], 8)
        self.assertEqual(calibration.gb_area_mm2[(8192, 64)], 2)

    def test_area_uses_the_existing_global_buffer_replication(self) -> None:
        pool = IPPool.from_yaml(
            REPO_ROOT / "configs" / "ip_pool_synthetic_65nm.yaml"
        )
        calibration = characterize_level1_area(pool)
        base = ArchitectureConfig(
            pe_x=4,
            pe_y=8,
            rf_size_bits=64,
            rf_bw_bits=8,
            gb_size_bits=8192,
            gb_bw_bits=64,
            gb_served_dims=[],
            dram_bw_bits=512,
        )
        pe_rf = calibration.pe_rf_area_mm2[(64, 8)]
        gb = calibration.gb_area_mm2[(8192, 64)]
        assert pe_rf is not None and gb is not None

        for served_dimensions, gb_count in (
            ([], 32),
            (["D1"], 8),
            (["D2"], 4),
            (["D1", "D2"], 1),
        ):
            with self.subTest(served_dimensions=served_dimensions):
                self.assertEqual(
                    calibration.area_mm2(
                        replace(base, gb_served_dims=served_dimensions)
                    ),
                    32 * pe_rf + gb_count * gb,
                )


if __name__ == "__main__":
    unittest.main()
