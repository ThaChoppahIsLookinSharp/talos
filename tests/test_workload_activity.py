from __future__ import annotations

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from onnx import helper
from zigzag.hardware.architecture.memory_port import DataDirection
from zigzag.mapping.data_movement import MemoryAccesses

from talos.architecture.genome import (
    GB_BW_OPTIONS,
    GB_SIZE_OPTIONS,
    ArchitectureConfig,
    default_genome,
    decode_genome,
)
from talos.evaluation.cacti_costs import CactiMemoryCost, Level1EnergyCalibration
from talos.evaluation.workload_activity import (
    LayerActivity,
    WorkloadActivityProfile,
    compute_workload_performance,
    extract_workload_activity_profile,
)
from talos.evaluation.zigzag_evaluator import (
    ZigZagEvaluator,
    mapping_objective_for_level1,
)
from talos.ip import PowerCharacterization


def energy_calibration() -> Level1EnergyCalibration:
    return Level1EnergyCalibration(
        technology_nm=65,
        reference_gb_capacity_bytes=128 * 1024,
        reference_word_bits=16,
        reference_gb_read_energy_pj=10,
        reference_gb_write_energy_pj=14,
        mac_energy_pj=2,
        gb_costs=tuple(
            CactiMemoryCost(size, bandwidth, 3, 4)
            for size in GB_SIZE_OPTIONS
            for bandwidth in GB_BW_OPTIONS
        ),
    )


class _MemoryOperandLinks:
    def layer_to_mem_op(self, layer_operand: str) -> str:
        return {"I": "I1", "W": "I2", "O": "O"}[layer_operand]


def _level(name: str) -> SimpleNamespace:
    read_energy_pj = 1000 if name == "dram" else 1
    write_energy_pj = 2000 if name == "dram" else 1
    return SimpleNamespace(
        memory_instance=SimpleNamespace(name=name),
        read_energy=read_energy_pj,
        write_energy=write_energy_pj,
    )


def _accesses(
    rd_out_to_low: int,
    wr_in_by_low: int,
    rd_out_to_high: int,
    wr_in_by_high: int,
) -> MemoryAccesses:
    return MemoryAccesses(
        {
            DataDirection.RD_OUT_TO_LOW: rd_out_to_low,
            DataDirection.WR_IN_BY_LOW: wr_in_by_low,
            DataDirection.RD_OUT_TO_HIGH: rd_out_to_high,
            DataDirection.WR_IN_BY_HIGH: wr_in_by_high,
        }
    )


class WorkloadActivityAdapterTests(unittest.TestCase):
    def test_workload_performance_uses_mapping_cycles_and_reference_frequency(self) -> None:
        profile = WorkloadActivityProfile(
            layers=(
                LayerActivity("a", 1_250_000, 1, 1, {}),
                LayerActivity("b", 750_000, 1, 1, {}),
            )
        )

        result = compute_workload_performance(profile, 200)

        self.assertEqual(
            result.layer_cycles_mapping,
            (("a", 1_250_000), ("b", 750_000)),
        )
        self.assertEqual(result.workload_cycles_per_inference, 2_000_000)
        self.assertEqual(result.workload_latency_s, 0.01)
        self.assertEqual(result.workload_throughput_ips, 100)

        with self.assertRaisesRegex(ValueError, "reference_frequency_mhz"):
            compute_workload_performance(profile, 0)
        with self.assertRaisesRegex(ValueError, "reference_frequency_mhz"):
            compute_workload_performance(profile, -1)
        with self.assertRaisesRegex(ValueError, "workload_cycles_per_inference"):
            compute_workload_performance(WorkloadActivityProfile(layers=()), 200)

    def test_mapping_objective_follows_level1_objectives(self) -> None:
        self.assertEqual(mapping_objective_for_level1(["energy", "area"]), "energy")
        self.assertEqual(mapping_objective_for_level1(["latency", "area"]), "latency")
        self.assertEqual(
            mapping_objective_for_level1(["energy", "latency"]),
            "EDP",
        )
        self.assertEqual(mapping_objective_for_level1(["edp"]), "EDP")
        self.assertEqual(mapping_objective_for_level1(["area"]), "EDP")

    def test_dram_characterization_derives_access_energy_from_power(self) -> None:
        model = PowerCharacterization(
            source="test",
            activity_method="access_rate",
            reference_frequency_mhz=100,
            p_idle_w=1,
            p_active_w=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = ZigZagEvaluator(
                workload="unused.onnx",
                workdir=tmp,
                dram_bandwidth_bits=256,
                dram_accesses_per_cycle=2,
                dram_power_model=model,
                energy_calibration=energy_calibration(),
            )
            config = decode_genome(default_genome())
            dram = evaluator._build_accelerator(config)["memories"]["dram"]

        self.assertEqual(dram["ports"][0]["bandwidth_max"], 256)
        self.assertEqual(dram["r_cost"], 6_400)
        self.assertEqual(dram["w_cost"], 6_400)
        self.assertEqual(evaluator._dram_idle_energy_pj(100), 1_000_000)

    def test_area_proxy_counts_three_rfs_and_replicated_global_buffers(self) -> None:
        evaluator = ZigZagEvaluator.__new__(ZigZagEvaluator)
        base = {
            "pe_x": 4,
            "pe_y": 8,
            "rf_size_bits": 64,
            "rf_bw_bits": 8,
            "gb_size_bits": 8192,
            "gb_bw_bits": 64,
            "dram_bw_bits": 512,
        }

        for served_dimensions, gb_count in (
            ([], 32),
            (["D1"], 8),
            (["D2"], 4),
            (["D1", "D2"], 1),
        ):
            with self.subTest(served_dimensions=served_dimensions):
                area = evaluator._estimate_area(
                    ArchitectureConfig(
                        **base,
                        gb_served_dims=served_dimensions,
                    )
                )
                self.assertEqual(
                    area,
                    32 + 3 * 32 * 64 * 0.001 + gb_count * 8192 * 0.0005,
                )

    def test_layer_activity_validates_values(self) -> None:
        values = {
            "layer_id": "layer",
            "latency_cycles": 1,
            "mac_count": 0,
            "spatially_used_pes": 0,
            "memory_accesses": {"gb": 0},
        }
        for override in (
            {"layer_id": ""},
            {"latency_cycles": 0},
            {"latency_cycles": -1},
            {"mac_count": -1},
            {"spatially_used_pes": -1},
            {"memory_accesses": {"": 0}},
            {"memory_accesses": {"gb": -1}},
            {"operand_precision_bits": []},
            {"operand_precision_bits": {"I": 0}},
            {"operand_numeric_formats": []},
            {"operand_numeric_formats": {"I": ""}},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    LayerActivity(**{**values, **override})

    def test_zigzag_evaluator_keeps_per_layer_cmes(self) -> None:
        layer_cmes = [SimpleNamespace(layer="per-layer")]

        def fake_zigzag(**kwargs):
            pickle_path = Path(kwargs["pickle_filename"])
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with pickle_path.open("wb") as handle:
                pickle.dump(layer_cmes, handle)
            return 1.0, 2.0, [SimpleNamespace(cumulative=True)]

        with tempfile.TemporaryDirectory() as tmp:
            evaluator = ZigZagEvaluator(
                workload="unused.onnx",
                workdir=tmp,
                energy_calibration=energy_calibration(),
            )
            evaluator._onnx_workload = helper.make_model(
                helper.make_graph([], "test", [], [])
            )
            with patch(
                "zigzag.api.get_hardware_performance_zigzag",
                side_effect=fake_zigzag,
            ):
                energy, latency, returned_cmes = evaluator._run_zigzag(
                    "accelerator.yaml"
                )

        self.assertEqual((energy, latency), (1.0, 2.0))
        self.assertEqual(returned_cmes[0].layer, "per-layer")

    def test_extracts_physical_accesses_from_zigzag_3_8_5_shape(self) -> None:
        cme = SimpleNamespace(
            layer=SimpleNamespace(
                id=7,
                name="Op7",
                total_mac_count=800,
                operand_precision=SimpleNamespace(data={"I": 8, "W": 8, "O": 16}),
            ),
            latency_total2=100,
            memory_operand_links=_MemoryOperandLinks(),
            mem_hierarchy_dict={
                "I1": [_level("rf_i1"), _level("gb"), _level("dram")],
                "I2": [_level("rf_i2"), _level("gb"), _level("dram")],
                "O": [_level("rf_o"), _level("gb"), _level("dram")],
            },
            memory_word_access={
                "I": [_accesses(10, 0, 0, 2), _accesses(3, 0, 0, 4), _accesses(5, 0, 0, 0)],
                "W": [_accesses(20, 0, 0, 3), _accesses(6, 0, 0, 7), _accesses(8, 0, 0, 0)],
                # All four O movements are distinct physical RF accesses. Each
                # field is counted once; RF and GB levels are not mixed.
                "O": [_accesses(11, 13, 17, 19), _accesses(23, 29, 31, 37), _accesses(41, 43, 0, 0)],
            },
            spatial_mapping=SimpleNamespace(
                unit_count={"I": [8, 4, 1], "W": [8, 2, 1], "O": [8, 8, 1]}
            ),
        )

        profile = extract_workload_activity_profile(
            [cme],
            operand_numeric_formats_by_layer={
                7: {"I": "int8", "W": "int8", "O": "int8"},
            },
        )
        layer = profile.layers[0]

        self.assertEqual(layer.layer_id, "Op7")
        self.assertEqual(layer.latency_cycles, 100)
        self.assertEqual(layer.mac_count, 800)
        self.assertEqual(layer.spatially_used_pes, 8)
        self.assertEqual(
            layer.operand_precision_bits,
            {"I": 8, "W": 8, "O": 16},
        )
        self.assertEqual(
            layer.operand_numeric_formats,
            {"I": "int8", "W": "int8", "O": "int8"},
        )
        self.assertEqual(
            layer.memory_accesses,
            {
                "rf_i1": 12.0,
                "rf_i2": 23.0,
                "rf_o": 60.0,
                "gb": 140.0,
                "dram": 97.0,
            },
        )
        self.assertEqual(profile.total_latency_cycles, 100)
        self.assertEqual(profile.total_mac_count, 800)
        self.assertEqual(profile.total_dram_accesses, 97)


if __name__ == "__main__":
    unittest.main()
