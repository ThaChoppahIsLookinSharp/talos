from __future__ import annotations

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zigzag.hardware.architecture.memory_port import DataDirection
from zigzag.mapping.data_movement import MemoryAccesses

from talos.evaluation.workload_activity import (
    LayerActivity,
    extract_workload_activity_profile,
)
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


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
            {"mac_count": -1},
            {"spatially_used_pes": -1},
            {"memory_accesses": {"": 0}},
            {"memory_accesses": {"gb": -1}},
            {"dram_access_energy_j": -1},
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
            evaluator = ZigZagEvaluator(workload="unused.onnx", workdir=tmp)
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
            layer=SimpleNamespace(id=7, name="Op7", total_mac_count=800),
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

        profile = extract_workload_activity_profile([cme])
        layer = profile.layers[0]

        self.assertEqual(layer.layer_id, "Op7")
        self.assertEqual(layer.latency_cycles, 100)
        self.assertEqual(layer.mac_count, 800)
        self.assertEqual(layer.spatially_used_pes, 8)
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
        # 54 reads * 1000 pJ + 43 writes * 2000 pJ.
        self.assertAlmostEqual(layer.dram_access_energy_j, 140_000e-12)
        self.assertEqual(profile.total_latency_cycles, 100)
        self.assertEqual(profile.total_mac_count, 800)
        self.assertEqual(profile.total_dram_accesses, 97)
        self.assertAlmostEqual(profile.total_dram_access_energy_j, 140_000e-12)


if __name__ == "__main__":
    unittest.main()
