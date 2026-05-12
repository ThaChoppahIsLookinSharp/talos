from __future__ import annotations

import unittest
from pathlib import Path

from talos.architecture.genome import GENOME_LENGTH, decode_genome
from talos.manual_validation import (
    ManualValidationSummary,
    fallback_manual_diagnostic_genome,
    format_area_diagnostics,
    known_valid_manual_genome,
    manual_smoke_genome_candidates,
)


class ManualValidationTests(unittest.TestCase):
    def test_known_valid_manual_genome_decodes_when_present(self) -> None:
        genome = known_valid_manual_genome()

        if genome is None:
            self.assertIsNone(genome)
            return

        self.assertEqual(len(genome), GENOME_LENGTH)
        decode_genome(genome)

    def test_manual_smoke_candidates_do_not_reduce_to_default_only(self) -> None:
        candidates = manual_smoke_genome_candidates()

        self.assertGreater(len(candidates), 1)
        self.assertEqual(len(candidates[0]), GENOME_LENGTH)
        for genome in candidates[:4]:
            decode_genome(genome)

    def test_fallback_manual_diagnostic_genome_decodes(self) -> None:
        decode_genome(fallback_manual_diagnostic_genome())

    def test_format_area_diagnostics_reports_expected_fields(self) -> None:
        summary = ManualValidationSummary(
            genome=[1, 1, 3, 4, 3],
            decoded_architecture="ArchitectureConfig(...)",
            valid=False,
            latency=float("inf"),
            energy=float("inf"),
            area=float("inf"),
            area_source="proxy",
            area_is_proxy=True,
            raw_zigzag_area=None,
            zigzag_area_path=None,
            error_message="boom",
            area_policy="prefer_zigzag_then_proxy",
            memory_cost_mode="manual",
            accelerator_yaml_path=str(Path("accelerator.yaml")),
        )

        text = format_area_diagnostics(summary)

        self.assertIn("area_source=proxy", text)
        self.assertIn("area_is_proxy=True", text)
        self.assertIn("raw_zigzag_area=None", text)
        self.assertIn("zigzag_area_path=None", text)
        self.assertIn("area_policy=prefer_zigzag_then_proxy", text)

    def test_compare_tool_uses_manual_validation_helper(self) -> None:
        tool_path = Path(__file__).resolve().parents[1] / "tools" / "compare_memory_cost_modes.py"
        source = tool_path.read_text(encoding="utf-8")

        self.assertIn("find_first_valid_manual_genome", source)
        self.assertIn("fallback_manual_diagnostic_genome", source)


if __name__ == "__main__":
    unittest.main()
