from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talos.architecture import (
    AbstractAccelerator,
    AbstractComponent,
    DEFAULT_DRAM_BW_BITS,
    abstract_accelerator_from_level1_config,
    abstract_accelerator_from_zigzag_yaml,
    decode_genome,
    default_genome,
)
from talos.ip import IPBlock, IPPool
from talos.level2 import Level2Evaluator, Level2GenomeSpec
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


class IPFoundationTests(unittest.TestCase):
    def test_ipblock_validates_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "throughput"):
            IPBlock(id="bad", type="pe", area=1.0, power=1.0, throughput=0.0, delay=1.0)

        block = IPBlock(id="pe0", type="pe", area=1.0, power=2.0, throughput=1.0, delay=0.5)
        self.assertEqual(block.id, "pe0")

    def test_ip_pool_loads_from_yaml_and_filters_by_type(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        pes = pool.by_type("pe")

        self.assertGreaterEqual(len(pes), 2)
        self.assertTrue(all(ip.type == "pe" for ip in pes))

    def test_ip_pool_finds_compatible_components(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        component = AbstractComponent(
            name="rf_test",
            type="register_file",
            required_capacity_bits=512,
            required_bandwidth_bits=64,
        )

        compatible = pool.find_compatible(component)

        self.assertTrue(any(ip.id == "rf_small" for ip in compatible))

    def test_ip_pool_raises_for_missing_compatible_ip(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        component = AbstractComponent(
            name="rf_too_big",
            type="register_file",
            required_capacity_bits=999999,
            required_bandwidth_bits=64,
        )

        with self.assertRaisesRegex(ValueError, "No compatible IPBlock found"):
            pool.find_compatible(component)


class AbstractAcceleratorImporterTests(unittest.TestCase):
    def test_level1_importer_builds_expected_components(self) -> None:
        config = decode_genome(default_genome())
        accelerator = abstract_accelerator_from_level1_config(config)

        self.assertIsInstance(accelerator, AbstractAccelerator)
        self.assertEqual(accelerator.components[0].name, "pe_array")
        self.assertEqual(accelerator.components[0].count, config.pe_x * config.pe_y)
        self.assertTrue(any(component.name == "gb" for component in accelerator.components))
        self.assertFalse(any(component.name == "dram" for component in accelerator.components))

    def test_zigzag_yaml_importer_builds_components(self) -> None:
        accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))

        self.assertEqual(accelerator.name, "zigzag_level2_example")
        self.assertTrue(any(component.name == "pe_array" for component in accelerator.components))
        self.assertTrue(any(component.name == "rf_i1" for component in accelerator.components))
        self.assertTrue(any(component.name == "gb" for component in accelerator.components))
        self.assertFalse(any(component.name == "dram" for component in accelerator.components))

    def test_level1_genome_has_fixed_platform_dram_bandwidth(self) -> None:
        genome = default_genome()
        config = decode_genome(genome)

        self.assertEqual(genome, [2, 2, 3, 2, 3, 2, 3])
        self.assertEqual(len(genome), 7)
        self.assertEqual(config.dram_bw_bits, DEFAULT_DRAM_BW_BITS)

    def test_zigzag_yaml_importer_is_tolerant_of_missing_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.yaml"
            path.write_text(
                """
name: minimal
memories:
  rf:
    size: 128
    served_dimensions: []
""".strip(),
                encoding="utf-8",
            )
            accelerator = abstract_accelerator_from_zigzag_yaml(str(path))

        self.assertEqual(accelerator.name, "minimal")
        self.assertEqual(len(accelerator.components), 1)
        self.assertEqual(accelerator.components[0].type, "register_file")


class Level2GenomeTests(unittest.TestCase):
    def test_level2_genome_spec_is_dynamic_and_compatible_with_pool(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(decode_genome(default_genome()))
        spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)

        self.assertEqual(spec.gene_names()[0], "pe_array")
        self.assertEqual(len(spec.default_genome()), len(spec.genes))
        self.assertEqual(len(spec.gene_bounds()), len(spec.genes))

    def test_level2_genome_decode_selects_implemented_components(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(decode_genome(default_genome()))
        spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)

        implemented = spec.decode(spec.default_genome())

        self.assertEqual(len(implemented.components), len(spec.genes))
        self.assertEqual(implemented.components[0].abstract_component.name, "pe_array")

    def test_level2_genome_spec_raises_if_component_has_no_candidates(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = AbstractAccelerator(
            name="bad",
            components=[
                AbstractComponent(
                    name="impossible_gb",
                    type="global_buffer",
                    required_capacity_bits=999999,
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "Unable to build Level2GenomeSpec"):
            Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)


class Level2EvaluatorTests(unittest.TestCase):
    def test_level2_evaluator_computes_simple_ppa(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(decode_genome(default_genome()))
        spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)
        implemented = spec.decode(spec.default_genome())

        result = Level2Evaluator().evaluate(implemented)

        self.assertTrue(result.valid)
        self.assertGreater(result.area, 0.0)
        self.assertGreater(result.power, 0.0)
        self.assertGreater(result.delay, 0.0)
        self.assertGreater(result.throughput, 0.0)

    def test_level2_evaluator_marks_invalid_incompatible_component(self) -> None:
        component = AbstractComponent(
            name="rf",
            type="register_file",
            count=1,
            required_capacity_bits=1024,
            required_bandwidth_bits=128,
        )
        bad_ip = IPBlock(
            id="rf_bad",
            type="register_file",
            area=1.0,
            power=1.0,
            throughput=1.0,
            delay=1.0,
            capacity_bits=512,
            bandwidth_bits=64,
        )
        result = Level2Evaluator().evaluate(
            ImplementedAccelerator(
                components=[ImplementedComponent(abstract_component=component, ip=bad_ip)]
            )
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.throughput, 0.0)
        self.assertIn("does not satisfy", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
