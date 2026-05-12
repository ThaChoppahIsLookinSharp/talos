from __future__ import annotations

import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from talos.architecture.genome import default_genome, gene_names
from talos.architecture.memory_specs import (
    GB_SIZE_OPTIONS,
    RF_SIZE_OPTIONS,
    derive_gb_bandwidth_max_bits,
    derive_rf_bandwidth_max_bits,
    validate_rf_cacti_compatibility,
)
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


TEST_GENOME = default_genome()
TMP_ROOT = Path(__file__).resolve().parents[1] / ".talos_zigzag" / "memory_cost_tests"


class StubMemoryCostEvaluator(ZigZagEvaluator):
    def __init__(self, tmp_path: Path, *, memory_cost_mode: str) -> None:
        super().__init__(
            workload="dummy.onnx",
            debug=False,
            workdir=str(tmp_path),
            memory_cost_mode=memory_cost_mode,
        )

    def _run_zigzag(self, accelerator_yaml_path: str):
        return 1.0, 2.0, {"area_total": 3.0}


def make_workdir() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    workdir = TMP_ROOT / f"case_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=False)
    return workdir


class MemoryCostModeTests(unittest.TestCase):
    def test_rf_size_options_are_updated_and_default_genome_stays_mid_range(self) -> None:
        self.assertEqual(RF_SIZE_OPTIONS, [1024, 2048, 4096, 8192, 16384, 32768])
        self.assertEqual(GB_SIZE_OPTIONS, [8192, 16384, 32768, 65536, 131072])
        self.assertEqual(TEST_GENOME[2], 3)
        self.assertEqual(TEST_GENOME[3], 3)

    def test_baseline_manual_mode_generates_manual_cost_fields(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="manual",
        )

        accelerator = evaluator.build_accelerator_from_genome(TEST_GENOME)
        gb = accelerator["memories"]["gb"]

        self.assertEqual(gb["r_cost"], 10.0)
        self.assertEqual(gb["w_cost"], 10.0)
        self.assertEqual(gb["area"], 10.0)
        self.assertFalse(gb["auto_cost_extraction"])
        self.assertEqual(gb["ports"][0]["bandwidth_min"], 64)
        self.assertEqual(gb["ports"][0]["bandwidth_max"], 1024)

        rf = accelerator["memories"]["rf_i1"]
        self.assertEqual(rf["ports"][0]["bandwidth_min"], 8)
        self.assertEqual(rf["ports"][0]["bandwidth_max"], 128)

    def test_auto_mode_enables_zigzag_memory_cost_extraction(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="zigzag_auto",
        )

        accelerator = evaluator.build_accelerator_from_genome(TEST_GENOME)
        rf = accelerator["memories"]["rf_i1"]

        self.assertTrue(rf["auto_cost_extraction"])
        self.assertIsNone(rf["r_cost"])
        self.assertIsNone(rf["w_cost"])
        self.assertIsNone(rf["area"])

    def test_hybrid_mode_only_enables_auto_cost_for_gb(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="hybrid_auto_gb",
        )

        accelerator = evaluator.build_accelerator_from_genome(TEST_GENOME)

        self.assertFalse(accelerator["memories"]["rf_i1"]["auto_cost_extraction"])
        self.assertTrue(accelerator["memories"]["gb"]["auto_cost_extraction"])
        self.assertFalse(accelerator["memories"]["dram"]["auto_cost_extraction"])

    def test_manual_and_auto_modes_produce_different_memory_descriptions(self) -> None:
        manual = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="manual",
        )
        auto = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="zigzag_auto",
        )

        manual_yaml = manual.render_accelerator_yaml(TEST_GENOME)
        auto_yaml = auto.render_accelerator_yaml(TEST_GENOME)

        self.assertNotEqual(manual_yaml, auto_yaml)
        self.assertIn("auto_cost_extraction: false", manual_yaml)
        self.assertIn("auto_cost_extraction: true", auto_yaml)

    def test_evaluation_result_records_memory_cost_mode(self) -> None:
        manual = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="manual")
        auto = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="zigzag_auto")

        manual_result = manual.evaluate(TEST_GENOME)
        auto_result = auto.evaluate(TEST_GENOME)

        self.assertEqual(manual_result.memory_cost_mode, "manual")
        self.assertEqual(auto_result.memory_cost_mode, "zigzag_auto")

    def test_all_modes_can_build_evaluator_without_heavy_run(self) -> None:
        manual = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="manual")
        hybrid = StubMemoryCostEvaluator(
            make_workdir(),
            memory_cost_mode="hybrid_auto_gb",
        )
        auto = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="zigzag_auto")

        manual_accelerator = manual.build_accelerator_from_genome(TEST_GENOME)
        hybrid_accelerator = hybrid.build_accelerator_from_genome(TEST_GENOME)
        auto_accelerator = auto.build_accelerator_from_genome(TEST_GENOME)

        self.assertEqual(manual_accelerator["name"], "talos_candidate")
        self.assertEqual(hybrid_accelerator["name"], "talos_candidate")
        self.assertEqual(auto_accelerator["name"], "talos_candidate")
        self.assertFalse(manual_accelerator["memories"]["rf_i1"]["auto_cost_extraction"])
        self.assertFalse(hybrid_accelerator["memories"]["rf_i1"]["auto_cost_extraction"])
        self.assertTrue(hybrid_accelerator["memories"]["gb"]["auto_cost_extraction"])
        self.assertTrue(auto_accelerator["memories"]["rf_i1"]["auto_cost_extraction"])

    def test_bandwidth_is_not_encoded_as_a_genome_gene(self) -> None:
        names = gene_names()

        self.assertEqual(len(TEST_GENOME), len(names))
        self.assertNotIn("rf_bw_code", names)
        self.assertNotIn("gb_bw_code", names)
        self.assertNotIn("dram_bw_code", names)

    def test_derive_rf_bandwidth_max_bits_matches_spec(self) -> None:
        self.assertEqual(derive_rf_bandwidth_max_bits(1024), 32)
        self.assertEqual(derive_rf_bandwidth_max_bits(2048), 64)
        self.assertEqual(derive_rf_bandwidth_max_bits(4096), 64)
        self.assertEqual(derive_rf_bandwidth_max_bits(8192), 128)
        self.assertEqual(derive_rf_bandwidth_max_bits(16384), 128)
        self.assertEqual(derive_rf_bandwidth_max_bits(32768), 256)

    def test_derive_gb_bandwidth_max_bits_matches_spec(self) -> None:
        self.assertEqual(derive_gb_bandwidth_max_bits(8192), 256)
        self.assertEqual(derive_gb_bandwidth_max_bits(65536), 1024)

    def test_rf_yaml_uses_derived_bandwidth_fields(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="manual",
        )

        accelerator = evaluator.build_accelerator_from_genome([2, 2, 0, 3, 3])
        rf_port = accelerator["memories"]["rf_i1"]["ports"][0]

        self.assertEqual(accelerator["memories"]["rf_i1"]["size"], 1024)
        self.assertEqual(rf_port["bandwidth_min"], 8)
        self.assertEqual(rf_port["bandwidth_max"], 32)
        self.assertNotEqual(rf_port["bandwidth_max"], 256)

    def test_validate_rf_cacti_compatibility_accepts_valid_combinations(self) -> None:
        validate_rf_cacti_compatibility(1024, 32)
        validate_rf_cacti_compatibility(2048, 64)
        validate_rf_cacti_compatibility(4096, 128)
        validate_rf_cacti_compatibility(8192, 256)
        validate_rf_cacti_compatibility(1024, 16)

    def test_validate_rf_cacti_compatibility_rejects_invalid_combinations(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "RF size is too small for CACTI with the configured bandwidth_max.",
        ):
            validate_rf_cacti_compatibility(512, 32)

        with self.assertRaisesRegex(
            ValueError,
            "RF size is too small for CACTI with the configured bandwidth_max.",
        ):
            validate_rf_cacti_compatibility(2048, 128)

    def test_cacti_constraint_applies_in_zigzag_auto_mode(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="zigzag_auto",
        )

        with patch(
            "talos.evaluation.zigzag_evaluator.derive_rf_bandwidth_max_bits",
            return_value=256,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "RF size is too small for CACTI with the configured bandwidth_max.",
            ):
                evaluator.build_accelerator_from_genome([2, 2, 0, 3, 3])

    def test_cacti_constraint_does_not_break_manual_mode(self) -> None:
        evaluator = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="manual",
        )

        accelerator = evaluator.build_accelerator_from_genome([2, 2, 0, 3, 3])

        self.assertEqual(accelerator["memories"]["rf_i1"]["ports"][0]["bandwidth_max"], 32)

    def test_runtime_env_helper_activates_only_for_auto_modes(self) -> None:
        manual = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="manual",
        )
        hybrid = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="hybrid_auto_gb",
        )
        auto = ZigZagEvaluator(
            workload="dummy.onnx",
            workdir=str(make_workdir()),
            memory_cost_mode="zigzag_auto",
        )

        self.assertFalse(manual._uses_zigzag_auto_costs())
        self.assertTrue(hybrid._uses_zigzag_auto_costs())
        self.assertTrue(auto._uses_zigzag_auto_costs())


if __name__ == "__main__":
    unittest.main()
