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
from talos.ip import IPBlock, IPPool, PowerCharacterization
from talos.level2 import Level2Evaluator, Level2GenomeSpec
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


class IPFoundationTests(unittest.TestCase):
    def test_ipblock_validates_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "throughput"):
            IPBlock(id="bad", type="pe", area=1.0, throughput=0.0, delay=1.0)

        block = IPBlock(id="pe0", type="pe", area=1.0, throughput=1.0, delay=0.5)
        self.assertEqual(block.id, "pe0")

    def test_power_characterization_validates_fields(self) -> None:
        values = {
            "source": "synthetic",
            "activity_method": "vectorless",
            "reference_frequency_mhz": 500.0,
            "p_idle_w": 0.1,
            "p_active_w": 0.2,
        }
        for field, value in (
            ("source", ""),
            ("activity_method", ""),
            ("reference_frequency_mhz", 0),
            ("reference_frequency_mhz", float("inf")),
            ("p_idle_w", -1),
            ("p_active_w", -1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    PowerCharacterization(**{**values, field: value})
        with self.assertRaisesRegex(ValueError, "p_active_w"):
            PowerCharacterization(**{**values, "p_idle_w": 0.3})
        with self.assertRaisesRegex(ValueError, "metadata"):
            PowerCharacterization(**{**values, "metadata": []})  # type: ignore[arg-type]

    def test_ip_pool_loads_from_yaml_and_filters_by_type(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        pes = pool.by_type("pe")

        self.assertGreaterEqual(len(pes), 2)
        self.assertTrue(all(ip.type == "pe" for ip in pes))
        self.assertIsInstance(pes[0].power_model, PowerCharacterization)
        self.assertEqual(
            pool.to_dict()["ips"][0]["power_model"]["p_active_w"],
            pes[0].power_model.p_active_w,
        )

    def test_ip_pool_loads_global_technology(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)

        self.assertEqual(pool.technology_nm, 65)

    def test_ip_pool_rejects_old_power_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.yaml"
            path.write_text(
                """
technology_nm: 65
ips:
  - id: pe
    type: pe
    area: 1
    power: 2
    throughput: 1
    delay: 1
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Legacy field 'power'"):
                IPPool.from_yaml(path)

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

    def test_ip_pool_loads_and_validates_included_rfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "composite.yaml"
            path.write_text(
                """
technology_nm: 65
ips:
  - id: pe_tile
    type: pe
    area: 2
    throughput: 1
    delay: 1
    included_rfs:
      rf_i1: rf
    included_rf_power_mode: parent_idle_baseline
  - id: rf
    type: register_file
    area: 1
    throughput: 1
    delay: 1
    capacity_bits: 512
    bandwidth_bits: 64
""".strip(),
                encoding="utf-8",
            )
            tile = IPPool.from_yaml(path).by_type("pe")[0]

        self.assertEqual(tile.included_rfs, {"rf_i1": "rf"})
        self.assertEqual(tile.included_rf_power_mode, "parent_idle_baseline")

        with self.assertRaisesRegex(ValueError, "requires included_rf_power_mode"):
            IPBlock(
                id="ambiguous_tile",
                type="pe",
                area=1,
                throughput=1,
                delay=1,
                included_rfs={"rf_i1": "rf"},
            )

        rf = IPBlock(
            id="rf",
            type="register_file",
            area=1,
            throughput=1,
            delay=1,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            IPPool([rf, rf])
        with self.assertRaisesRegex(ValueError, "unknown included RF role"):
            IPPool(
                [
                    rf,
                    IPBlock(
                        id="pe",
                        type="pe",
                        area=1,
                        throughput=1,
                        delay=1,
                        included_rfs={"rf_bad": "rf"},
                        included_rf_power_mode="parent_idle_baseline",
                    ),
                ]
            )
        with self.assertRaisesRegex(ValueError, "unknown IPBlock"):
            IPPool(
                [
                    IPBlock(
                        id="pe",
                        type="pe",
                        area=1,
                        throughput=1,
                        delay=1,
                        included_rfs={"rf_i1": "missing"},
                        included_rf_power_mode="parent_idle_baseline",
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "not a PE"):
            IPPool(
                [
                    rf,
                    IPBlock(
                        id="rf_container",
                        type="register_file",
                        area=1,
                        throughput=1,
                        delay=1,
                        included_rfs={"rf_i1": "rf"},
                        included_rf_power_mode="parent_idle_baseline",
                    ),
                ]
            )
        global_buffer = IPBlock(
            id="gb",
            type="global_buffer",
            area=1,
            throughput=1,
            delay=1,
        )
        with self.assertRaisesRegex(ValueError, "not a register_file"):
            IPPool(
                [
                    global_buffer,
                    IPBlock(
                        id="pe",
                        type="pe",
                        area=1,
                        throughput=1,
                        delay=1,
                        included_rfs={"rf_i1": "gb"},
                        included_rf_power_mode="parent_idle_baseline",
                    ),
                ]
            )


class AbstractAcceleratorImporterTests(unittest.TestCase):
    def test_abstract_accelerator_rejects_duplicate_component_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            AbstractAccelerator(
                name="duplicate",
                components=[
                    AbstractComponent(name="rf_i1", type="register_file"),
                    AbstractComponent(name="rf_i1", type="register_file"),
                ],
            )

    def test_level1_importer_builds_expected_components(self) -> None:
        config = decode_genome(default_genome())
        accelerator = abstract_accelerator_from_level1_config(config)

        self.assertIsInstance(accelerator, AbstractAccelerator)
        self.assertEqual(accelerator.components[0].name, "pe_array")
        self.assertEqual(accelerator.components[0].count, config.pe_x * config.pe_y)
        gb = next(component for component in accelerator.components if component.name == "gb")
        self.assertEqual(gb.count, 1)
        self.assertFalse(any(component.name == "dram" for component in accelerator.components))

        split_config = decode_genome([1, 2, 0, 0, 0, 0, 1])
        split_gb = next(
            component
            for component in abstract_accelerator_from_level1_config(split_config).components
            if component.name == "gb"
        )
        self.assertEqual(split_gb.count, split_config.pe_y)

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

    def test_composite_pe_filters_and_canonicalizes_included_rf_genes(self) -> None:
        def block(
            ip_id: str,
            ip_type: str,
            *,
            capacity: int | None = None,
            bandwidth: int | None = None,
            included_rfs: dict[str, str] | None = None,
        ) -> IPBlock:
            return IPBlock(
                id=ip_id,
                type=ip_type,
                area=1,
                throughput=1,
                delay=1,
                capacity_bits=capacity,
                bandwidth_bits=bandwidth,
                included_rfs=included_rfs or {},
                included_rf_power_mode=(
                    "parent_idle_baseline" if included_rfs else None
                ),
            )

        pool = IPPool(
            [
                block("pe_plain", "pe"),
                block("pe_tile", "pe", included_rfs={"rf_i1": "rf_large"}),
                block("pe_too_small", "pe", included_rfs={"rf_i1": "rf_tiny"}),
                block("pe_too_narrow", "pe", included_rfs={"rf_i1": "rf_narrow"}),
                block("rf_tiny", "register_file", capacity=256, bandwidth=64),
                block("rf_narrow", "register_file", capacity=512, bandwidth=32),
                block("rf_small", "register_file", capacity=512, bandwidth=64),
                block("rf_large", "register_file", capacity=1024, bandwidth=128),
            ]
        )
        accelerator = AbstractAccelerator(
            name="composite",
            components=[
                AbstractComponent(name="pe_array", type="pe", count=2),
                AbstractComponent(
                    name="rf_i1",
                    type="register_file",
                    count=2,
                    required_capacity_bits=512,
                    required_bandwidth_bits=64,
                ),
            ],
        )

        spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)

        self.assertEqual(
            [candidate.id for candidate in spec.genes[0].candidates],
            ["pe_plain", "pe_tile"],
        )
        self.assertEqual(spec.canonicalize([1, 0]), [1, 1])
        self.assertEqual(set(spec.iter_genomes()), {(0, 0), (0, 1), (1, 1)})
        self.assertEqual(spec.genome_count(), 3)
        implemented = spec.decode([1, 0])
        self.assertEqual(implemented.components[1].ip.id, "rf_large")
        self.assertEqual(implemented.components[1].covered_by_pe_id, "pe_tile")

        count_mismatch = AbstractAccelerator(
            name="count_mismatch",
            components=[
                accelerator.components[0],
                AbstractComponent(
                    name="rf_i1",
                    type="register_file",
                    count=1,
                    required_capacity_bits=512,
                    required_bandwidth_bits=64,
                ),
            ],
        )
        mismatch_spec = Level2GenomeSpec.from_accelerator_and_pool(
            count_mismatch, pool
        )
        self.assertEqual(
            [candidate.id for candidate in mismatch_spec.genes[0].candidates],
            ["pe_plain"],
        )

        with self.assertRaisesRegex(ValueError, "exactly one PE component"):
            Level2GenomeSpec.from_accelerator_and_pool(
                AbstractAccelerator(
                    name="multiple_pe_arrays",
                    components=[
                        accelerator.components[0],
                        AbstractComponent(name="pe_aux", type="pe", count=2),
                        accelerator.components[1],
                    ],
                ),
                pool,
            )


class Level2EvaluatorTests(unittest.TestCase):
    def test_composite_pe_counts_inclusive_ppa_once_and_validates_coverage(self) -> None:
        pe = ImplementedComponent(
            abstract_component=AbstractComponent(name="pe_array", type="pe", count=2),
            ip=IPBlock(
                id="tile",
                type="pe",
                area=3,
                throughput=4,
                delay=5,
                fmax_mhz=600,
                included_rfs={"rf_i1": "rf"},
                included_rf_power_mode="parent_idle_baseline",
            ),
        )
        rf = ImplementedComponent(
            abstract_component=AbstractComponent(
                name="rf_i1",
                type="register_file",
                count=2,
            ),
            ip=IPBlock(
                id="rf",
                type="register_file",
                area=100,
                throughput=1,
                delay=100,
                fmax_mhz=100,
            ),
            covered_by_pe_id="tile",
        )

        result = Level2Evaluator().evaluate(
            ImplementedAccelerator(components=[pe, rf])
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.area, 6)
        self.assertEqual(result.physical_critical_delay, 5)
        self.assertEqual(result.selected_ip_min_throughput, 4)
        self.assertEqual(result.physical_fmax_mhz, 600)

        invalid = Level2Evaluator().evaluate(
            ImplementedAccelerator(
                components=[
                    pe,
                    ImplementedComponent(
                        abstract_component=rf.abstract_component,
                        ip=rf.ip,
                    ),
                ]
            )
        )
        self.assertFalse(invalid.valid)
        self.assertIn("does not validly cover", invalid.error_message or "")

    def test_level2_evaluator_computes_simple_ppa(self) -> None:
        pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(decode_genome(default_genome()))
        spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)
        implemented = spec.decode(spec.default_genome())

        result = Level2Evaluator().evaluate(implemented)

        self.assertTrue(result.valid)
        self.assertGreater(result.area, 0.0)
        self.assertIsNone(result.power)
        self.assertIsNone(result.workload_energy_j)
        self.assertIsNone(result.workload_latency_s)
        self.assertGreater(result.physical_critical_delay, 0.0)
        self.assertGreater(result.selected_ip_min_throughput, 0.0)

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
        self.assertEqual(result.selected_ip_min_throughput, 0.0)
        self.assertIn("does not satisfy", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
