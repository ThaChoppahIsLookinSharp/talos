from __future__ import annotations

from dataclasses import replace
import io
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from talos.architecture.genome import (
    GB_BW_OPTIONS,
    GB_SIZE_OPTIONS,
    RF_BW_OPTIONS,
    RF_SIZE_OPTIONS,
    default_genome,
    decode_genome,
)
from talos.evaluation.cacti_costs import (
    CactiMemoryCost,
    Level1EnergyCalibration,
    REFERENCE_GB_CAPACITY_BYTES,
    REFERENCE_WORD_BITS,
    _included_cacti_master,
    _write_cacti_config,
    calibrate_synthetic_dram_ip,
    calibrate_synthetic_dram_power_model,
    characterize_level1_energy,
    parse_cacti_output,
    resolve_dram_ip,
)
from talos.evaluation.area_calibration import Level1AreaCalibration
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator
from talos.ga.pymoo_runner import TalosPymooProblem, run_nsga2_pymoo
from talos.ip import IPBlock, IPPool, PowerCharacterization


def calibration() -> Level1EnergyCalibration:
    return Level1EnergyCalibration(
        technology_nm=65,
        reference_gb_capacity_bytes=REFERENCE_GB_CAPACITY_BYTES,
        reference_word_bits=REFERENCE_WORD_BITS,
        reference_gb_read_energy_pj=24,
        reference_gb_write_energy_pj=48,
        mac_energy_pj=6,
        gb_costs=tuple(
            CactiMemoryCost(
                capacity_bits=size,
                bandwidth_bits=bandwidth,
                read_energy_pj_per_access=size / 8192 + bandwidth / 64,
                write_energy_pj_per_access=size / 4096 + bandwidth / 32,
                standby_power_w=0.002,
            )
            for size in GB_SIZE_OPTIONS
            for bandwidth in GB_BW_OPTIONS
        ),
        rf_costs=tuple(
            CactiMemoryCost(
                size,
                bandwidth,
                size / 64 + bandwidth / 8,
                size / 32 + bandwidth / 4,
                standby_power_w=0.001,
            )
            for size in RF_SIZE_OPTIONS
            for bandwidth in RF_BW_OPTIONS
        ),
    )


def area_calibration() -> Level1AreaCalibration:
    return Level1AreaCalibration(
        pe_rf_area_mm2={
            (size, bandwidth): 1.0
            for size in RF_SIZE_OPTIONS
            for bandwidth in RF_BW_OPTIONS
        },
        gb_area_mm2={
            (size, bandwidth): 1.0
            for size in GB_SIZE_OPTIONS
            for bandwidth in GB_BW_OPTIONS
        },
    )


def synthetic_dram_model() -> PowerCharacterization:
    return PowerCharacterization(
        source="synthetic",
        activity_method="access_rate",
        reference_frequency_mhz=500,
        p_idle_w=0.02,
        p_active_w=4.5,
        voltage_v=1,
        temperature_c=25,
        corner="tt",
    )


class CactiParserTests(unittest.TestCase):
    def test_parser_ignores_secondary_na_and_converts_nj_to_pj(self) -> None:
        source = io.StringIO(
            "Capacity (bytes), Output width (bits), Dynamic search energy (nJ),"
            " Dynamic read energy (nJ), Dynamic write energy (nJ),"
            " Standby leakage per bank(mW)\n"
            "1024,64,N/A,0.0125,0.025,2.5\n"
        )

        read, write, standby = parse_cacti_output(
            source,
            expected_capacity_bytes=1024,
            expected_bandwidth_bits=64,
        )

        self.assertEqual((read, write, standby), (12.5, 25, 0.0025))

    def test_parser_rejects_mismatch_ambiguity_and_invalid_energy(self) -> None:
        header = (
            "Capacity (bytes),Output width (bits),"
            "Dynamic read energy (nJ),Dynamic write energy (nJ),"
            "Standby leakage per bank(mW)\n"
        )
        for body, message in (
            ("2048,64,1,1,1\n", "capacity mismatch"),
            ("1024,128,1,1,1\n", "bandwidth mismatch"),
            (
                "1024,64,1,1,1\n1024,64,2,2,2\n",
                "exactly one",
            ),
            ("1024,64,N/A,1,1\n", "invalid required"),
            ("1024,64,nan,1,1\n", "finite and positive"),
            ("1024,64,0,1,1\n", "finite and positive"),
            ("1024,64,1,1,nan\n", "finite and non-negative"),
        ):
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValueError, message):
                    parse_cacti_output(
                        io.StringIO(header + body),
                        expected_capacity_bytes=1024,
                        expected_bandwidth_bits=64,
                    )

    def test_config_uses_requested_pool_technology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.cfg"
            _write_cacti_config(
                path,
                capacity_bytes=1024,
                bandwidth_bits=64,
                line_size_bytes=8,
                technology_um=0.040,
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("-size (bytes) 1024", text)
        self.assertIn("-block size (bytes) 8", text)
        self.assertIn("-output/input bus width 64", text)
        self.assertIn("-technology (u) 0.04", text)


class EnergyCalibrationTests(unittest.TestCase):
    def test_characterizes_memory_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            master = Path(temporary) / "cacti_master"
            master.mkdir()
            (master / "cacti").touch()

            def fake_cost(
                _master: Path,
                *,
                role: str,
                capacity_bits: int,
                bandwidth_bits: int,
                technology_um: float,
            ) -> CactiMemoryCost:
                self.assertEqual(technology_um, 0.040)
                energy = 60 if role == "reference_global_buffer" else 10
                return CactiMemoryCost(
                    capacity_bits,
                    bandwidth_bits,
                    energy,
                    energy + 12,
                    standby_power_w=0.001,
                )

            with (
                patch(
                    "talos.evaluation.cacti_costs._characterize_memory",
                    side_effect=fake_cost,
                ) as characterize,
                patch(
                    "talos.evaluation.cacti_costs.shutil.copytree",
                    wraps=shutil.copytree,
                ) as copytree,
            ):
                result = characterize_level1_energy(master, technology_nm=40)

        self.assertEqual(copytree.call_count, 1)
        self.assertEqual(characterize.call_count, 62)
        self.assertEqual(result.technology_nm, 40)
        self.assertEqual(len(result.gb_costs), 25)
        self.assertEqual(len(result.rf_costs), 36)
        self.assertEqual(
            {
                (cost.capacity_bits, cost.bandwidth_bits)
                for cost in result.gb_costs
            },
            {
                (size, bandwidth)
                for size in GB_SIZE_OPTIONS
                for bandwidth in GB_BW_OPTIONS
            },
        )
        self.assertEqual(
            {
                (cost.capacity_bits, cost.bandwidth_bits)
                for cost in result.rf_costs
            },
            {
                (size, bandwidth)
                for size in RF_SIZE_OPTIONS
                for bandwidth in RF_BW_OPTIONS
            },
        )
        reference_call = characterize.call_args_list[-1].kwargs
        self.assertEqual(
            reference_call["capacity_bits"],
            REFERENCE_GB_CAPACITY_BYTES * 8,
        )
        self.assertEqual(reference_call["bandwidth_bits"], REFERENCE_WORD_BITS)
        self.assertEqual(result.mac_energy_pj, 11)

    def test_eyeriss_scaling(self) -> None:
        result = calibration()

        self.assertEqual(result.mac_energy_pj, 6)
        self.assertEqual(result.rf_energy_pj_per_access(8), 3)
        self.assertEqual(result.rf_energy_pj_per_access(16), 6)
        self.assertEqual(result.rf_energy_pj_per_access(64), 24)
        self.assertEqual(result.rf_energy_pj_per_access(256), 96)
        self.assertEqual(result.dram_energy_pj_per_16b, 1200)
        self.assertEqual(result.dram_energy_pj_per_access(16), 1200)
        self.assertEqual(result.dram_energy_pj_per_access(512), 38_400)

    def test_synthetic_dram_reconstructs_dynamic_energy_and_preserves_idle(self) -> None:
        result = calibration()
        model = synthetic_dram_model()
        calibrated = calibrate_synthetic_dram_power_model(
            model,
            dram_bandwidth_bits=512,
            accesses_per_cycle=2,
            calibration=result,
        )

        reconstructed_pj = (
            (calibrated.p_active_w - calibrated.p_idle_w)
            / (calibrated.reference_frequency_mhz * 1e6 * 2)
            * 1e12
        )
        self.assertEqual(calibrated.p_idle_w, model.p_idle_w)
        self.assertAlmostEqual(
            reconstructed_pj,
            result.dram_energy_pj_per_access(512),
        )

    def test_real_dram_is_not_modified(self) -> None:
        model = replace(synthetic_dram_model(), source="genus")
        dram = IPBlock(
            id="dram",
            type="dram",
            area=0,
            throughput=1,
            delay=1,
            bandwidth_bits=512,
            metadata={"accesses_per_cycle": 1},
            power_model=model,
        )

        self.assertIs(calibrate_synthetic_dram_ip(dram, calibration()), dram)

    def test_dram_copy_leaves_original_pool_and_pe_power_untouched(self) -> None:
        pe_model = replace(
            synthetic_dram_model(),
            activity_method="vectorless",
            p_idle_w=0.1,
            p_active_w=0.3,
        )
        pe = IPBlock(
            id="pe",
            type="pe",
            area=1,
            throughput=1,
            delay=1,
            power_model=pe_model,
        )
        dram = IPBlock(
            id="dram",
            type="dram",
            area=0,
            throughput=1,
            delay=1,
            bandwidth_bits=512,
            metadata={"accesses_per_cycle": 1},
            power_model=synthetic_dram_model(),
        )
        original = IPPool([pe, dram])
        calibrated_dram = calibrate_synthetic_dram_ip(dram, calibration())
        calibrated = IPPool([pe, calibrated_dram])

        self.assertIs(original.by_type("pe")[0].power_model, pe_model)
        self.assertIs(calibrated.by_type("pe")[0].power_model, pe_model)
        self.assertEqual(
            original.by_type("dram")[0].power_model.p_active_w,
            4.5,
        )
        self.assertNotEqual(
            calibrated.by_type("dram")[0].power_model.p_active_w,
            4.5,
        )

    def test_pool_dram_has_priority_over_fallback(self) -> None:
        pe = IPBlock(
            id="pe",
            type="pe",
            area=1,
            throughput=1,
            delay=1,
            fmax_mhz=500,
            power_model=replace(
                synthetic_dram_model(),
                source="genus",
            ),
        )
        dram = IPBlock(
            id="measured_dram",
            type="dram",
            area=0,
            throughput=1,
            delay=1,
            fmax_mhz=500,
            bandwidth_bits=256,
            metadata={"accesses_per_cycle": 2},
            power_model=replace(
                synthetic_dram_model(),
                source="measured",
            ),
        )

        self.assertIs(
            resolve_dram_ip(IPPool([pe, dram]), calibration()),
            dram,
        )

    def test_platform_dram_is_shared_idle_proxy(self) -> None:
        pe = IPBlock(
            id="pe",
            type="pe",
            area=1,
            throughput=1,
            delay=1,
            fmax_mhz=500,
            power_model=replace(
                synthetic_dram_model(),
                source="genus",
            ),
        )

        dram = resolve_dram_ip(IPPool([pe]), calibration())

        self.assertEqual(dram.id, "dram_platform_512b")
        self.assertEqual(dram.bandwidth_bits, 512)
        self.assertEqual(dram.power_model.p_idle_w, 0.02)
        self.assertEqual(
            dram.power_model.reference_frequency_mhz,
            500,
        )
        self.assertGreater(
            dram.power_model.p_active_w,
            dram.power_model.p_idle_w,
        )


class Level1EnergyIntegrationTests(unittest.TestCase):
    def test_calibration_failure_prevents_workers_and_pymoo(self) -> None:
        with (
            patch(
                "talos.ga.pymoo_runner.characterize_level1_energy",
                side_effect=RuntimeError("CACTI failed"),
            ),
            patch("talos.ga.pymoo_runner.mp.get_context") as get_context,
            patch("talos.ga.pymoo_runner.minimize") as minimize,
            self.assertRaisesRegex(RuntimeError, "CACTI failed"),
        ):
            run_nsga2_pymoo(
                workload_path="unused.onnx",
                area_calibration=area_calibration(),
                pop_size=2,
                n_gen=1,
                n_workers=2,
                save_csv=False,
            )

        get_context.assert_not_called()
        minimize.assert_not_called()

    def test_worker_state_reuses_calibration_without_adapter(self) -> None:
        result = calibration()
        problem = TalosPymooProblem(
            workload_path="unused.onnx",
            objective_names=["energy"],
            area_calibration=area_calibration(),
            adapter=object(),
            energy_calibration=result,
        )

        state = problem.__getstate__()

        self.assertIsNone(state["_adapter"])
        self.assertIs(state["energy_calibration"], result)
        self.assertEqual(state["area_calibration"], area_calibration())

    def test_accelerator_uses_calibrated_mac_rf_gb_and_dram(self) -> None:
        result = calibration()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = ZigZagEvaluator(
                workload="unused.onnx",
                workdir=temporary,
                dram_power_model=synthetic_dram_model(),
                energy_calibration=result,
                area_calibration=area_calibration(),
            )
            config = decode_genome(default_genome())
            accelerator = evaluator._build_accelerator(config)

        self.assertEqual(
            accelerator["operational_array"]["unit_energy"],
            result.mac_energy_pj,
        )
        expected_rf = result.rf_cost(
            config.rf_size_bits,
            config.rf_bw_bits,
        )
        for name in ("rf_i1", "rf_i2", "rf_o"):
            self.assertEqual(
                accelerator["memories"][name]["r_cost"],
                expected_rf.read_energy_pj_per_access,
            )
            self.assertEqual(
                accelerator["memories"][name]["w_cost"],
                expected_rf.write_energy_pj_per_access,
            )
        gb = accelerator["memories"]["gb"]
        expected_gb = result.gb_cost(config.gb_size_bits, config.gb_bw_bits)
        self.assertEqual(gb["r_cost"], expected_gb.read_energy_pj_per_access)
        self.assertEqual(gb["w_cost"], expected_gb.write_energy_pj_per_access)
        dram = accelerator["memories"]["dram"]
        self.assertEqual(
            dram["r_cost"],
            result.dram_energy_pj_per_access(512),
        )
        self.assertEqual(dram["r_cost"], dram["w_cost"])
        self.assertTrue(math.isfinite(dram["r_cost"]))

    def test_changing_gb_changes_only_its_own_cost(self) -> None:
        result = calibration()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = ZigZagEvaluator(
                workload="unused.onnx",
                workdir=temporary,
                energy_calibration=result,
                area_calibration=area_calibration(),
            )
            first = evaluator._build_accelerator(
                decode_genome([0, 0, 0, 0, 0, 0, 0])
            )
            second = evaluator._build_accelerator(
                decode_genome([0, 0, 0, 0, 4, 4, 0])
            )

        self.assertNotEqual(
            first["memories"]["gb"]["r_cost"],
            second["memories"]["gb"]["r_cost"],
        )
        self.assertEqual(
            first["operational_array"]["unit_energy"],
            second["operational_array"]["unit_energy"],
        )
        self.assertEqual(
            first["memories"]["dram"]["r_cost"],
            second["memories"]["dram"]["r_cost"],
        )


class RealCactiSmokeTests(unittest.TestCase):
    def test_bundled_cacti_characterizes_full_catalog_and_reference(self) -> None:
        source = _included_cacti_master()
        if not (source / "cacti").is_file():
            self.skipTest("Bundled CACTI executable is unavailable.")
        result = characterize_level1_energy(source)

        self.assertEqual(len(result.gb_costs), 25)
        self.assertEqual(len(result.rf_costs), 36)
        self.assertEqual(
            result.gb_cost(8192, 512).cacti_capacity_bits,
            16384,
        )
        self.assertEqual(
            result.gb_cost(8192, 1024).cacti_capacity_bits,
            32768,
        )
        self.assertEqual(
            result.gb_cost(16384, 1024).cacti_capacity_bits,
            32768,
        )
        for cost in result.gb_costs:
            self.assertGreater(cost.read_energy_pj_per_access, 0)
            self.assertGreater(cost.write_energy_pj_per_access, 0)
            self.assertTrue(math.isfinite(cost.read_energy_pj_per_access))
            self.assertTrue(math.isfinite(cost.write_energy_pj_per_access))
            self.assertGreater(cost.standby_power_w, 0)
        for cost in result.rf_costs:
            self.assertGreater(cost.standby_power_w, 0)
            self.assertTrue(math.isfinite(cost.standby_power_w))
        self.assertGreater(result.reference_gb_read_energy_pj, 0)
        self.assertGreater(result.reference_gb_write_energy_pj, 0)


if __name__ == "__main__":
    unittest.main()
