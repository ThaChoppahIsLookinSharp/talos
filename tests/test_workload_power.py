from __future__ import annotations

import unittest

from talos.architecture.abstract_accelerator import AbstractComponent
from talos.evaluation.workload_activity import LayerActivity, WorkloadActivityProfile
from talos.ip import IPBlock, PowerCharacterization
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent
from talos.level2.workload_power import (
    _component_power_w,
    _memory_utilization,
    _pe_utilization,
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


def _implemented(pe_model: PowerCharacterization) -> ImplementedAccelerator:
    components: list[ImplementedComponent] = []
    for name, ip_type, count, metadata, model in (
        ("pe_array", "pe", 1, {"macs_per_cycle": 1}, pe_model),
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
                    fmax_mhz=100.0,
                    metadata=metadata,
                    power_model=model,
                ),
            )
        )
    return ImplementedAccelerator(components=components)


class UtilizationTests(unittest.TestCase):
    def test_pe_utilization_cases(self) -> None:
        common = {
            "latency_cycles": 10,
            "instance_count": 10,
            "macs_per_cycle": 1,
            "spatially_used_pes": 10,
        }
        self.assertEqual(_pe_utilization(mac_count=0, **common), 0)
        self.assertEqual(_pe_utilization(mac_count=50, **common), 0.5)
        self.assertEqual(_pe_utilization(mac_count=100, **common), 1)
        with self.assertRaisesRegex(ValueError, "exceeds 1"):
            _pe_utilization(mac_count=101, **common)
        with self.assertRaisesRegex(ValueError, "uses 11 PEs"):
            _pe_utilization(mac_count=100, **{**common, "spatially_used_pes": 11})
        with self.assertRaisesRegex(ValueError, "macs_per_cycle"):
            _pe_utilization(mac_count=100, **{**common, "macs_per_cycle": 0})

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

    def test_component_power_interpolation(self) -> None:
        model = _model(1.0, 3.0)
        self.assertEqual(_component_power_w(2, 0.0, model), 2.0)
        self.assertEqual(_component_power_w(2, 0.5, model), 4.0)
        self.assertEqual(_component_power_w(2, 1.0, model), 6.0)


class WorkloadPowerTests(unittest.TestCase):
    def test_power_includes_dram_energy_and_is_weighted_by_duration(self) -> None:
        profile = WorkloadActivityProfile(
            layers=(
                LayerActivity("active", 10, 10, 1, {}, dram_access_energy_j=4e-8),
                LayerActivity("idle", 30, 0, 0, {}),
            )
        )

        result = evaluate_workload_power(_implemented(_model(0, 1)), profile)

        self.assertAlmostEqual(result.latency_s, 4e-7)
        self.assertAlmostEqual(result.energy_j, 1.4e-7)
        self.assertAlmostEqual(result.power_w, 0.35)
        self.assertNotAlmostEqual(result.power_w, 0.5)


if __name__ == "__main__":
    unittest.main()
