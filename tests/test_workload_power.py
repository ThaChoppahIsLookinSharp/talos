from __future__ import annotations

import unittest

from talos.architecture.abstract_accelerator import AbstractComponent
from talos.evaluation.workload_activity import LayerActivity, WorkloadActivityProfile
from talos.ip import IPBlock, PowerCharacterization
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent
from talos.level2.workload_power import (
    _component_power_w,
    _memory_power,
    _memory_utilization,
    evaluate_workload_power,
)


def _model(idle: float, active: float) -> PowerCharacterization:
    return PowerCharacterization(
        source="synthetic",
        activity_method="vectorless",
        reference_frequency_mhz=100.0,
        p_idle_w=idle,
        p_active_w=active,
        voltage_v=1.0,
        temperature_c=25.0,
        corner="tt",
    )


def _implemented(
    pe_model: PowerCharacterization,
    *,
    fmax_mhz: float = 100.0,
    pe_count: int = 1,
) -> ImplementedAccelerator:
    components: list[ImplementedComponent] = []
    for name, ip_type, count, metadata, model in (
        ("pe_array", "pe", pe_count, {}, pe_model),
        ("rf_i1", "register_file", 1, {"accesses_per_cycle": 1}, _model(0, 0)),
        ("rf_i2", "register_file", 1, {"accesses_per_cycle": 1}, _model(0, 0)),
        ("rf_o", "register_file", 1, {"accesses_per_cycle": 1}, _model(0, 0)),
        ("gb", "global_buffer", 1, {"accesses_per_cycle": 1}, _model(0, 0)),
    ):
        components.append(
            ImplementedComponent(
                abstract_component=AbstractComponent(
                    name=name,
                    type=ip_type,
                    count=count,
                ),
                ip=IPBlock(
                    id=f"{name}_ip",
                    type=ip_type,
                    area=1.0,
                    throughput=1.0,
                    delay=1.0,
                    fmax_mhz=fmax_mhz,
                    metadata=metadata,
                    power_model=model,
                ),
            )
        )
    return ImplementedAccelerator(components=components)


def _dram(model: PowerCharacterization | None = None) -> IPBlock:
    return IPBlock(
        id="dram",
        type="dram",
        area=0,
        throughput=1,
        delay=1,
        fmax_mhz=100,
        bandwidth_bits=512,
        metadata={"accesses_per_cycle": 1},
        power_model=model or _model(0, 0),
    )


def _composite_implemented() -> ImplementedAccelerator:
    pe_id = "pe_tile"
    rf_ids = {name: f"{name}_ip" for name in ("rf_i1", "rf_i2", "rf_o")}
    components = [
        ImplementedComponent(
            abstract_component=AbstractComponent(name="pe_array", type="pe", count=2),
            ip=IPBlock(
                id=pe_id,
                type="pe",
                area=1,
                throughput=1,
                delay=1,
                fmax_mhz=100,
                included_rfs=rf_ids,
                included_rf_power_mode="parent_idle_baseline",
                power_model=_model(10, 14),
            ),
        )
    ]
    for name in ("rf_i1", "rf_i2", "rf_o", "gb"):
        covered = name in rf_ids
        components.append(
            ImplementedComponent(
                abstract_component=AbstractComponent(
                    name=name,
                    type="register_file" if covered else "global_buffer",
                    count=2 if covered else 1,
                ),
                ip=IPBlock(
                    id=rf_ids.get(name, "gb_ip"),
                    type="register_file" if covered else "global_buffer",
                    area=1,
                    throughput=1,
                    delay=1,
                    fmax_mhz=100,
                    metadata={"accesses_per_cycle": 1},
                    power_model=_model(1, 3) if covered else _model(0, 0),
                ),
                covered_by_pe_id=pe_id if covered else None,
            )
        )
    return ImplementedAccelerator(components=components)


class UtilizationTests(unittest.TestCase):
    def test_memory_utilization_cases(self) -> None:
        common = {
            "latency_cycles": 1000,
            "instance_count": 1,
            "accesses_per_cycle": 1,
        }
        self.assertEqual(_memory_utilization(accesses=0, **common), 0)
        self.assertEqual(_memory_utilization(accesses=250, **common), 0.25)
        self.assertEqual(
            _memory_utilization(accesses=500, **{**common, "instance_count": 2}),
            0.25,
        )
        self.assertEqual(_memory_utilization(accesses=1000.0001, **common), 1.0)
        self.assertEqual(
            _memory_utilization(
                accesses=1_026_080,
                **{**common, "latency_cycles": 1_026_078},
            ),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "exceeds 1"):
            _memory_utilization(accesses=1001, **common)
        with self.assertRaisesRegex(ValueError, "no physical instances"):
            _memory_utilization(accesses=1, **{**common, "instance_count": 0})
        with self.assertRaisesRegex(ValueError, "accesses_per_cycle"):
            _memory_utilization(accesses=1, **{**common, "accesses_per_cycle": 0})

    def test_memory_power_normalizes_accesses_to_selected_width(self) -> None:
        component = ImplementedComponent(
            abstract_component=AbstractComponent(
                name="gb",
                type="global_buffer",
                required_bandwidth_bits=64,
            ),
            ip=IPBlock(
                id="gb_128b",
                type="global_buffer",
                area=1,
                throughput=1,
                delay=1,
                fmax_mhz=100,
                bandwidth_bits=128,
                metadata={"accesses_per_cycle": 1},
                power_model=_model(0, 1),
            ),
        )
        layer = LayerActivity("layer", 100, 0, 0, {"gb": 100})

        power_w = _memory_power(component, layer, 100)

        self.assertEqual(power_w, 0.5)

    def test_component_power_interpolation(self) -> None:
        model = _model(1.0, 3.0)
        self.assertEqual(_component_power_w(2, 0.0, model), 2.0)
        self.assertEqual(_component_power_w(2, 0.5, model), 4.0)
        self.assertEqual(_component_power_w(2, 1.0, model), 6.0)


class WorkloadPowerTests(unittest.TestCase):
    def test_composite_pe_does_not_count_covered_rf_idle_power_twice(self) -> None:
        result = evaluate_workload_power(
            _composite_implemented(),
            WorkloadActivityProfile(
                layers=(LayerActivity("layer", 100, 1, 1, {}),)
            ),
            _dram(),
        )

        self.assertEqual(result.power_w, 24)

    def test_composite_pe_adds_only_covered_rf_access_increment(self) -> None:
        result = evaluate_workload_power(
            _composite_implemented(),
            WorkloadActivityProfile(
                layers=(
                    LayerActivity(
                        "layer",
                        100,
                        1,
                        1,
                        {"rf_i1": 100},
                    ),
                )
            ),
            _dram(),
        )

        self.assertEqual(result.power_w, 26)

    def test_pe_power_uses_active_and_idle_counts_from_mapping(self) -> None:
        profile = WorkloadActivityProfile(
            layers=(LayerActivity("layer", 1000, 16000, 16, {}),)
        )

        result = evaluate_workload_power(
            _implemented(_model(1, 3), pe_count=32),
            profile,
            _dram(),
        )

        self.assertEqual(result.power_w, 16 * 3 + 16 * 1)

    def test_reference_frequency_sets_time_and_fmax_only_checks_viability(self) -> None:
        profile = WorkloadActivityProfile(
            layers=(LayerActivity("layer", 1000, 1000, 1, {}),)
        )

        result = evaluate_workload_power(
            _implemented(_model(0, 1), fmax_mhz=160.0),
            profile,
            _dram(),
        )

        self.assertAlmostEqual(result.latency_s, 1000 / 100e6)
        self.assertAlmostEqual(result.power_w, 1.0)
        self.assertAlmostEqual(result.energy_j, 1000 / 100e6)
        self.assertEqual(result.operating_frequency_mhz, 100.0)

        with self.assertRaisesRegex(ValueError, "below power reference frequency"):
            evaluate_workload_power(
                _implemented(_model(0, 1), fmax_mhz=80.0),
                profile,
                _dram(),
            )

    def test_power_includes_dram_energy_and_is_weighted_by_duration(self) -> None:
        profile = WorkloadActivityProfile(
            layers=(
                LayerActivity("active", 10, 10, 1, {"dram": 5}),
                LayerActivity("idle", 30, 0, 0, {}),
            )
        )

        result = evaluate_workload_power(
            _implemented(_model(0, 1)),
            profile,
            _dram(_model(0.2, 4.2)),
        )

        self.assertAlmostEqual(result.latency_s, 4e-7)
        self.assertAlmostEqual(result.dram_energy_j, 2.8e-7)
        self.assertAlmostEqual(result.energy_j, 3.8e-7)
        self.assertAlmostEqual(result.power_w, 0.95)


if __name__ == "__main__":
    unittest.main()
