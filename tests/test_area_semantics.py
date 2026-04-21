from __future__ import annotations

import uuid
import unittest
from pathlib import Path

from talos.architecture.genome import decode_genome
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


TEST_GENOME = [2, 2, 3, 2, 3, 2, 3, 3]
TMP_ROOT = Path(__file__).resolve().parents[1] / ".talos_zigzag" / "test_runs"


class FakeCMEWithAreaTotal:
    area_total = 123.0


class FakeCMEWithArea:
    area = 456.0


class StubEvaluator(ZigZagEvaluator):
    def __init__(
        self,
        tmp_path: Path,
        cme,
        *,
        use_area_proxy_fallback: bool = True,
    ) -> None:
        super().__init__(
            workload="dummy.onnx",
            debug=False,
            workdir=str(tmp_path),
            use_area_proxy_fallback=use_area_proxy_fallback,
        )
        self._stub_cme = cme

    def _run_zigzag(self, accelerator_yaml_path: str):
        return 10.0, 20.0, self._stub_cme


def make_workdir() -> Path:
    TMP_ROOT.mkdir(exist_ok=True)
    workdir = TMP_ROOT / f"case_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=False)
    return workdir


class AreaSemanticsTests(unittest.TestCase):
    def test_extract_area_prefers_zigzag_area_total(self) -> None:
        evaluator = ZigZagEvaluator(workload="dummy.onnx", workdir=str(make_workdir()))
        cfg = decode_genome(TEST_GENOME)

        area, source, raw_zigzag_area = evaluator._extract_area(
            FakeCMEWithAreaTotal(),
            cfg,
        )

        self.assertEqual(area, 123.0)
        self.assertEqual(source, "zigzag")
        self.assertEqual(raw_zigzag_area, 123.0)

    def test_extract_area_uses_zigzag_area_attribute(self) -> None:
        evaluator = ZigZagEvaluator(workload="dummy.onnx", workdir=str(make_workdir()))
        cfg = decode_genome(TEST_GENOME)

        area, source, raw_zigzag_area = evaluator._extract_area(
            FakeCMEWithArea(),
            cfg,
        )

        self.assertEqual(area, 456.0)
        self.assertEqual(source, "zigzag")
        self.assertEqual(raw_zigzag_area, 456.0)

    def test_extract_area_uses_dict_value(self) -> None:
        evaluator = ZigZagEvaluator(workload="dummy.onnx", workdir=str(make_workdir()))
        cfg = decode_genome(TEST_GENOME)

        area, source, raw_zigzag_area = evaluator._extract_area({"area": 789.0}, cfg)

        self.assertEqual(area, 789.0)
        self.assertEqual(source, "zigzag")
        self.assertEqual(raw_zigzag_area, 789.0)

    def test_evaluate_uses_area_proxy_when_zigzag_area_missing(self) -> None:
        evaluator = StubEvaluator(make_workdir(), cme={})

        result = evaluator.evaluate(TEST_GENOME)
        expected_proxy = evaluator._estimate_area_proxy(decode_genome(TEST_GENOME))

        self.assertTrue(result.valid)
        self.assertEqual(result.area, expected_proxy)
        self.assertEqual(result.area_source, "proxy")
        self.assertTrue(result.area_is_proxy)
        self.assertIsNone(result.raw_zigzag_area)

    def test_evaluate_fails_when_area_missing_and_proxy_disabled(self) -> None:
        evaluator = StubEvaluator(
            make_workdir(),
            cme={},
            use_area_proxy_fallback=False,
        )

        result = evaluator.evaluate(TEST_GENOME)

        self.assertFalse(result.valid)
        self.assertEqual(result.area, float("inf"))
        self.assertEqual(result.area_source, "missing")
        self.assertFalse(result.area_is_proxy)
        self.assertEqual(
            result.error_message,
            "ZigZag did not return a usable area value and area proxy fallback is disabled.",
        )

    def test_evaluate_marks_area_source_when_zigzag_area_exists(self) -> None:
        evaluator = StubEvaluator(make_workdir(), cme={"area_total": 321.0})

        result = evaluator.evaluate(TEST_GENOME)

        self.assertTrue(result.valid)
        self.assertEqual(result.area, 321.0)
        self.assertEqual(result.area_source, "zigzag")
        self.assertFalse(result.area_is_proxy)
        self.assertEqual(result.raw_zigzag_area, 321.0)


if __name__ == "__main__":
    unittest.main()
