from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from talos.ga.pymoo_runner import _write_results_csv


class DummyPymooResult:
    X = [[2.0, 2.0, 3.0, 3.0, 3.0]]
    F = [[1.0, 2.0, 3.0]]


class PymooRunnerTests(unittest.TestCase):
    def test_write_results_csv_records_memory_cost_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_results_csv(
                result=DummyPymooResult(),
                adapter=None,  # type: ignore[arg-type]
                objective_names=["latency", "energy", "area"],
                pop_size=2,
                n_gen=1,
                seed=7,
                n_workers=1,
                results_dir=tmp,
                memory_cost_mode="zigzag_auto",
            )

            with Path(csv_path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["memory_cost_mode"], "zigzag_auto")
        self.assertEqual(rows[0]["area_source"], "missing")
        self.assertEqual(rows[0]["area_is_proxy"], "False")
        self.assertEqual(rows[0]["raw_zigzag_area"], "")
        self.assertEqual(rows[0]["zigzag_area_path"], "")
        self.assertEqual(rows[0]["rf_size_bits"], "8192")
        self.assertEqual(rows[0]["rf_bandwidth_max_bits"], "128")
        self.assertEqual(rows[0]["gb_size_bits"], "65536")
        self.assertEqual(rows[0]["gb_bandwidth_max_bits"], "1024")
        self.assertEqual(rows[0]["latency"], "1.0")
        self.assertEqual(rows[0]["energy"], "2.0")
        self.assertEqual(rows[0]["area"], "3.0")


if __name__ == "__main__":
    unittest.main()
