from __future__ import annotations

import uuid
import unittest
from pathlib import Path

from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


TEST_GENOME = [2, 2, 3, 2, 3, 2, 3, 3]
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

    def test_both_modes_can_build_evaluator_without_heavy_run(self) -> None:
        manual = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="manual")
        auto = StubMemoryCostEvaluator(make_workdir(), memory_cost_mode="zigzag_auto")

        manual_accelerator = manual.build_accelerator_from_genome(TEST_GENOME)
        auto_accelerator = auto.build_accelerator_from_genome(TEST_GENOME)

        self.assertEqual(manual_accelerator["name"], "talos_candidate")
        self.assertEqual(auto_accelerator["name"], "talos_candidate")
        self.assertFalse(manual_accelerator["memories"]["rf_i1"]["auto_cost_extraction"])
        self.assertTrue(auto_accelerator["memories"]["rf_i1"]["auto_cost_extraction"])


if __name__ == "__main__":
    unittest.main()
