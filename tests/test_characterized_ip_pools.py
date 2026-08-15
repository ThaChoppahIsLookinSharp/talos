from __future__ import annotations

import math
from pathlib import Path
import unittest

import yaml

from talos.ip import IPPool


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = (
    REPO_ROOT / "characterizations" / "ip_catalog_65nm.yaml"
)
POOL_CASES = {
    "ip_pool_characterized_65nm.yaml": "integer",
    "ip_pool_fp16_65nm.yaml": "IEEE-754 binary16",
    "ip_pool_fp32_65nm.yaml": "IEEE-754 binary32",
}
TOP_LEVEL_FIELDS = (
    "type",
    "throughput",
    "delay",
    "fmax_mhz",
    "capacity_bits",
    "bandwidth_bits",
    "included_rfs",
    "included_rf_power_mode",
)
PE_METADATA_FIELDS = (
    "precision_bits",
    "activation_bits",
    "weight_bits",
    "accumulator_bits",
    "macs_per_cycle",
    "pipeline_latency_cycles",
    "dynamic_energy_per_mac_pj",
)
MEMORY_METADATA_FIELDS = (
    "accesses_per_cycle",
    "ports",
    "read_energy_pj",
    "write_energy_pj",
)
POWER_FIELDS = (
    "source",
    "activity_method",
    "reference_frequency_mhz",
    "p_idle_w",
    "p_active_w",
    "voltage_v",
    "temperature_c",
    "corner",
)


def _numeric_format(metadata: dict[str, object]) -> str:
    value = metadata.get("numeric_format")
    if value == "IEEE-754 binary16":
        return "float16"
    if value == "IEEE-754 binary32":
        return "float32"
    prefix = "int" if metadata.get("signed") else "uint"
    return f"{prefix}{metadata['precision_bits']}"


class CharacterizedIPPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = yaml.safe_load(CATALOG_PATH.read_text())
        cls.source_by_id = {
            ip["id"]: ip for ip in cls.catalog["ips"]
        }

    def test_runtime_pools_match_source_catalog(self) -> None:
        self.assertEqual(
            len(self.catalog["ips"]),
            len(self.source_by_id),
        )
        self.assertEqual(len(self.source_by_id), 27)
        memories = {
            ip["id"]
            for ip in self.catalog["ips"]
            if ip["type"] != "pe"
        }

        for filename, pe_format in POOL_CASES.items():
            with self.subTest(pool=filename):
                path = REPO_ROOT / "configs" / filename
                IPPool.from_yaml(path)
                pool = yaml.safe_load(path.read_text())
                pool_by_id = {ip["id"]: ip for ip in pool["ips"]}
                expected_pes = {
                    ip["id"]
                    for ip in self.catalog["ips"]
                    if ip["type"] == "pe"
                    and self._source_format(ip) == pe_format
                }
                self.assertEqual(
                    set(pool_by_id),
                    expected_pes | memories,
                )
                for ip_id in expected_pes | memories:
                    self._assert_matches_source(
                        pool_by_id[ip_id],
                        self.source_by_id[ip_id],
                    )

    @staticmethod
    def _source_format(ip: dict[str, object]) -> object:
        metadata = ip["metadata"]
        assert isinstance(metadata, dict)
        return metadata.get("numeric_format") or metadata["data_type"]

    def _assert_matches_source(
        self,
        current: dict[str, object],
        source: dict[str, object],
    ) -> None:
        self.assertTrue(
            math.isclose(
                float(current["area"]),
                float(source["area"]) / 1_000_000,
                rel_tol=1e-12,
            )
        )
        for field in TOP_LEVEL_FIELDS:
            self._assert_same(
                current.get(field),
                source.get(field),
            )

        current_metadata = current["metadata"]
        source_metadata = source["metadata"]
        assert isinstance(current_metadata, dict)
        assert isinstance(source_metadata, dict)
        if source["type"] == "pe":
            fields = PE_METADATA_FIELDS
            self.assertEqual(
                current_metadata["numeric_format"],
                _numeric_format(source_metadata),
            )
        else:
            fields = MEMORY_METADATA_FIELDS
        for field in fields:
            self._assert_same(
                current_metadata[field],
                source_metadata[field],
            )

        current_power = current["power_model"]
        source_power = source["power_model"]
        assert isinstance(current_power, dict)
        assert isinstance(source_power, dict)
        for field in POWER_FIELDS:
            self._assert_same(
                current_power[field],
                source_power[field],
            )

    def _assert_same(self, current: object, source: object) -> None:
        if (
            isinstance(current, (int, float))
            and isinstance(source, (int, float))
        ):
            self.assertTrue(
                math.isclose(
                    float(current),
                    float(source),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            )
            return
        self.assertEqual(current, source)


if __name__ == "__main__":
    unittest.main()
