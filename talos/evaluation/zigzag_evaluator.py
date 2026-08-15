from __future__ import annotations

from collections.abc import Iterable, Iterator
import copy
import contextlib
from dataclasses import dataclass
import io
import logging
from math import isfinite, prod
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from typing import Any

import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper
import yaml
from zigzag.parser.onnx.utils import get_onnx_tensor_type

from talos.architecture.genome import (
    DEFAULT_DRAM_BW_BITS,
    ArchitectureConfig,
    decode_genome,
)
from talos.evaluation.cacti_costs import (
    Level1EnergyCalibration,
    calibrate_synthetic_dram_power_model,
)
from talos.evaluation.workload_activity import (
    WorkloadActivityProfile,
    extract_workload_activity_profile,
)
from talos.ip.ip_characterization import PowerCharacterization

DEFAULT_DRAM_ACCESSES_PER_CYCLE = 1.0
DEFAULT_DRAM_POWER_MODEL = PowerCharacterization(
    source="synthetic",
    activity_method="access_rate",
    reference_frequency_mhz=500.0,
    p_idle_w=0.02,
    p_active_w=4.5,
    voltage_v=1.0,
    temperature_c=25.0,
    corner="tt",
)
ZIGZAG_MAPPING_OBJECTIVES = {"energy", "latency", "EDP"}
ZIGZAG_ONNX_OPERATORS = {"Conv", "QLinearConv", "Gemm", "MatMul"}
ZIGZAG_PRECISION_ATTRIBUTES = {
    "act_size",
    "weight_size",
    "output_size",
}
ONNX_NUMERIC_FORMATS = {
    TensorProto.FLOAT: ("float32", 32),
    TensorProto.UINT8: ("uint8", 8),
    TensorProto.INT8: ("int8", 8),
    TensorProto.UINT16: ("uint16", 16),
    TensorProto.INT16: ("int16", 16),
    TensorProto.INT32: ("int32", 32),
    TensorProto.INT64: ("int64", 64),
    TensorProto.FLOAT16: ("float16", 16),
    TensorProto.DOUBLE: ("float64", 64),
    TensorProto.UINT32: ("uint32", 32),
    TensorProto.UINT64: ("uint64", 64),
    TensorProto.BFLOAT16: ("bfloat16", 16),
    TensorProto.FLOAT8E4M3FN: ("float8e4m3fn", 8),
    TensorProto.FLOAT8E4M3FNUZ: ("float8e4m3fnuz", 8),
    TensorProto.FLOAT8E5M2: ("float8e5m2", 8),
    TensorProto.FLOAT8E5M2FNUZ: ("float8e5m2fnuz", 8),
    TensorProto.UINT4: ("uint4", 4),
    TensorProto.INT4: ("int4", 4),
    TensorProto.FLOAT4E2M1: ("float4e2m1", 4),
    TensorProto.FLOAT8E8M0: ("float8e8m0", 8),
    TensorProto.UINT2: ("uint2", 2),
    TensorProto.INT2: ("int2", 2),
}


def mapping_objective_for_level1(objective_names: Iterable[str]) -> str:
    """Choose the one ZigZag mapping criterion matching Level-1 objectives."""
    names = set(objective_names)
    energy = bool(names & {"energy", "eap"})
    latency = bool(names & {"latency", "alp"})
    if "edp" in names or (energy and latency):
        return "EDP"
    if energy:
        return "energy"
    if latency:
        return "latency"
    return "EDP"


def prepare_onnx_workload(
    workload: str | Path,
) -> tuple[ModelProto, dict[int, dict[str, str]]]:
    """Infer ONNX types and make ZigZag use their bit widths."""
    path = Path(workload)
    try:
        model = onnx.load(path, load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise ValueError(
            f"Unable to load and infer ONNX workload {path}: {exc}"
        ) from exc

    formats_by_layer: dict[int, dict[str, str]] = {}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in ZIGZAG_ONNX_OPERATORS:
            continue
        weight_index = 3 if node.op_type == "QLinearConv" else 1
        if len(node.input) <= weight_index or not node.output:
            raise ValueError(
                f"ONNX node {_node_label(node, node_index)!r} has no "
                "activation, weight or output tensor."
            )
        activation = _tensor_numeric_format(
            model,
            node.input[0],
            node,
            node_index,
        )
        weight = _tensor_numeric_format(
            model,
            node.input[weight_index],
            node,
            node_index,
        )
        output = _tensor_numeric_format(
            model,
            node.output[0],
            node,
            node_index,
        )
        _replace_zigzag_precision_attributes(
            node,
            activation_bits=activation[1],
            weight_bits=weight[1],
            output_bits=output[1],
        )
        formats_by_layer[node_index] = {
            "I": activation[0],
            "W": weight[0],
            "O": output[0],
        }
    return model, formats_by_layer


def _tensor_numeric_format(
    model: ModelProto,
    tensor_name: str,
    node: NodeProto,
    node_index: int,
) -> tuple[str, int]:
    try:
        elem_type = get_onnx_tensor_type(tensor_name, model).elem_type
    except KeyError as exc:
        raise ValueError(
            f"Unable to infer type for tensor {tensor_name!r} "
            "in ONNX "
            f"node {_node_label(node, node_index)!r}."
        ) from exc
    try:
        return ONNX_NUMERIC_FORMATS[elem_type]
    except KeyError as exc:
        try:
            type_name = TensorProto.DataType.Name(elem_type)
        except ValueError:
            type_name = str(elem_type)
        raise ValueError(
            f"Unsupported ONNX tensor type {type_name!r} for tensor "
            f"{tensor_name!r} in node "
            f"{_node_label(node, node_index)!r}."
        ) from exc


def _replace_zigzag_precision_attributes(
    node: NodeProto,
    *,
    activation_bits: int,
    weight_bits: int,
    output_bits: int,
) -> None:
    kept = [
        copy.deepcopy(attribute)
        for attribute in node.attribute
        if attribute.name not in ZIGZAG_PRECISION_ATTRIBUTES
    ]
    node.ClearField("attribute")
    node.attribute.extend(kept)
    node.attribute.extend(
        [
            helper.make_attribute("act_size", activation_bits),
            helper.make_attribute("weight_size", weight_bits),
            helper.make_attribute("output_size", output_bits),
        ]
    )


def _node_label(node: NodeProto, node_index: int) -> str:
    return node.name or f"Op{node_index}"


@dataclass
class EvaluationResult:
    latency: float
    energy: float
    area: float
    valid: bool
    error_message: str | None = None
    activity_profile: WorkloadActivityProfile | None = None
    mapping_objective: str | None = None


class ZigZagEvaluator:
    """
    TALOS -> ZigZag compatibility layer.

    Genome semantics live in talos.architecture.genome. This evaluator
    only consumes decoded architecture configs and runs ZigZag.
    """

    def __init__(
        self,
        workload: str,
        mapping: list[dict[str, Any]] | None = None,
        opt: str = "EDP",
        use_mock_area: bool = True,
        workdir: str | None = None,
        debug: bool = False,
        lpf_limit: int = 6,
        nb_spatial_mappings_generated: int = 3,
        dram_bandwidth_bits: int = DEFAULT_DRAM_BW_BITS,
        dram_accesses_per_cycle: float = DEFAULT_DRAM_ACCESSES_PER_CYCLE,
        dram_power_model: PowerCharacterization = DEFAULT_DRAM_POWER_MODEL,
        energy_calibration: Level1EnergyCalibration | None = None,
    ) -> None:
        if dram_bandwidth_bits <= 0:
            raise ValueError("DRAM bandwidth must be > 0.")
        if not isfinite(dram_accesses_per_cycle) or dram_accesses_per_cycle <= 0:
            raise ValueError("DRAM accesses_per_cycle must be finite and > 0.")
        if not isinstance(dram_power_model, PowerCharacterization):
            raise ValueError(
                "DRAM power_model must be a PowerCharacterization."
            )
        if not isinstance(energy_calibration, Level1EnergyCalibration):
            raise ValueError(
                "Level 1 energy_calibration must be provided before evaluation."
            )
        if opt not in ZIGZAG_MAPPING_OBJECTIVES:
            raise ValueError(
                f"ZigZag opt must be one of {sorted(ZIGZAG_MAPPING_OBJECTIVES)}."
            )
        self.workload = workload
        self.mapping = mapping if mapping is not None else self._default_mapping()
        self.opt = opt
        self.use_mock_area = use_mock_area
        self.workdir = (
            Path(workdir) if workdir is not None else Path.cwd() / ".talos_zigzag"
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.lpf_limit = lpf_limit
        self.nb_spatial_mappings_generated = nb_spatial_mappings_generated
        self.dram_bandwidth_bits = dram_bandwidth_bits
        self.dram_accesses_per_cycle = dram_accesses_per_cycle
        self.energy_calibration = energy_calibration
        self.dram_power_model = calibrate_synthetic_dram_power_model(
            dram_power_model,
            dram_bandwidth_bits=dram_bandwidth_bits,
            accesses_per_cycle=dram_accesses_per_cycle,
            calibration=energy_calibration,
        )
        self.mapping_yaml_path = self._write_mapping_yaml(self.mapping)
        self._evaluation_counter = 0
        self._onnx_workload: ModelProto | None = None
        self._operand_numeric_formats_by_layer: dict[
            int,
            dict[str, str],
        ] = {}

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        try:
            cfg = decode_genome(genome)
            accelerator = self._build_accelerator(cfg)
            accelerator_yaml_path = self._write_accelerator_yaml(accelerator)

            if self.debug:
                self._print_debug_yaml(accelerator_yaml_path)
                energy, latency, cme = self._run_zigzag(accelerator_yaml_path)
            else:
                with self._quiet_zigzag():
                    energy, latency, cme = self._run_zigzag(accelerator_yaml_path)

            activity_profile = extract_workload_activity_profile(
                cme,
                operand_numeric_formats_by_layer=(
                    self._operand_numeric_formats_by_layer
                ),
            )
            energy += self._dram_idle_energy_pj(latency)
            energy += self._onchip_idle_energy_pj(
                cfg,
                activity_profile,
            )
            area = self._extract_area(cme, cfg)

            return EvaluationResult(
                latency=float(latency),
                energy=float(energy),
                area=float(area),
                valid=True,
                activity_profile=activity_profile,
                mapping_objective=self.opt,
            )

        except Exception as exc:
            if self.debug:
                import traceback

                print("ZigZag evaluation failed:")
                traceback.print_exc()

            return EvaluationResult(
                latency=float("inf"),
                energy=float("inf"),
                area=float("inf"),
                valid=False,
                error_message=str(exc),
                mapping_objective=self.opt,
            )

    def evaluate_many(
        self,
        genomes: Iterable[list[float]],
        n_workers: int = 1,
    ) -> list[EvaluationResult]:
        rows = list(genomes)
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1.")
        if n_workers == 1 or len(rows) < 2:
            return [self.evaluate(genome) for genome in rows]

        worker_evaluator = copy.copy(self)
        worker_evaluator._onnx_workload = None
        worker_evaluator._operand_numeric_formats_by_layer = {}
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(n_workers, len(rows)),
            initializer=_initialize_zigzag_worker,
            initargs=(worker_evaluator,),
        ) as pool:
            return pool.map(_evaluate_zigzag_worker, rows)

    def _write_mapping_yaml(self, mapping: list[dict[str, Any]]) -> str:
        mapping_path = self.workdir / "mapping.yaml"
        with open(mapping_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapping, f, sort_keys=False)

        return str(mapping_path)

    def _write_accelerator_yaml(self, accelerator: dict[str, Any]) -> str:
        accelerator_path = self.workdir / "accelerator.yaml"
        with open(accelerator_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(accelerator, f, sort_keys=False)

        return str(accelerator_path)

    def _run_zigzag(self, accelerator_yaml_path: str) -> tuple[float, float, Any]:
        from zigzag.api import get_hardware_performance_zigzag

        dump_folder = self._next_dump_folder()
        pickle_path = Path(dump_folder) / "list_of_cmes.pickle"
        energy, latency, _cumulative_cme = get_hardware_performance_zigzag(
            workload=self._prepared_workload(),
            accelerator=accelerator_yaml_path,
            mapping=self.mapping_yaml_path,
            opt=self.opt,
            dump_folder=dump_folder,
            pickle_filename=str(pickle_path),
            lpf_limit=self.lpf_limit,
            nb_spatial_mappings_generated=self.nb_spatial_mappings_generated,
            loma_show_progress_bar=self.debug,
        )
        with pickle_path.open("rb") as handle:
            layer_cmes = pickle.load(handle)
        return energy, latency, layer_cmes

    def _prepared_workload(self) -> ModelProto:
        if self._onnx_workload is None:
            (
                self._onnx_workload,
                self._operand_numeric_formats_by_layer,
            ) = prepare_onnx_workload(self.workload)
        return copy.deepcopy(self._onnx_workload)

    def _next_dump_folder(self) -> str:
        """
        ZigZag's default dump folder includes datetime strings with ':'.
        Those paths are invalid on Windows, so TALOS always provides a
        portable per-evaluation output folder.
        """
        self._evaluation_counter += 1
        folder_name = f"run_{os.getpid()}_{self._evaluation_counter:06d}"
        dump_folder = self.workdir / "zigzag_outputs" / folder_name
        return str(dump_folder)

    @contextlib.contextmanager
    def _quiet_zigzag(self) -> Iterator[None]:
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                yield
        finally:
            logging.disable(previous_disable_level)

    def _print_debug_yaml(self, accelerator_yaml_path: str) -> None:
        print("Using mapping file:", self.mapping_yaml_path)
        print(Path(self.mapping_yaml_path).read_text(encoding="utf-8"))

        print("Using accelerator file:", accelerator_yaml_path)
        print(Path(accelerator_yaml_path).read_text(encoding="utf-8"))

    def _rw_port(self, name: str, bw: int, allocations: list[str]) -> dict[str, Any]:
        return {
            "name": name,
            "type": "read_write",
            "bandwidth_min": bw,
            "bandwidth_max": bw,
            "allocation": allocations,
        }

    def _build_accelerator(self, cfg: ArchitectureConfig) -> dict[str, Any]:
        dram_energy_pj_per_access = self._dram_dynamic_energy_pj_per_access()
        rf_cost = self.energy_calibration.rf_cost(
            cfg.rf_size_bits,
            cfg.rf_bw_bits,
        )
        gb_cost = self.energy_calibration.gb_cost(
            cfg.gb_size_bits,
            cfg.gb_bw_bits,
        )
        accelerator = {
            "name": "talos_candidate",
            "operational_array": {
                "is_imc": False,
                "unit_energy": self.energy_calibration.mac_energy_pj,
                "unit_area": 1.0,
                "dimensions": ["D1", "D2"],
                "sizes": [cfg.pe_x, cfg.pe_y],
                "imc_type": None,
                "adc_resolution": 0,
                "bit_serial_precision": None,
            },
            "memories": {
                "rf_i1": {
                    "size": cfg.rf_size_bits,
                    "r_cost": rf_cost.read_energy_pj_per_access,
                    "w_cost": rf_cost.write_energy_pj_per_access,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I1"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["I1, tl", "I1, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_i2": {
                    "size": cfg.rf_size_bits,
                    "r_cost": rf_cost.read_energy_pj_per_access,
                    "w_cost": rf_cost.write_energy_pj_per_access,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I2"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["I2, tl", "I2, fh"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "rf_o": {
                    "size": cfg.rf_size_bits,
                    "r_cost": rf_cost.read_energy_pj_per_access,
                    "w_cost": rf_cost.write_energy_pj_per_access,
                    "area": 1.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.rf_bw_bits,
                            ["O, fh", "O, fl", "O, th", "O, tl"],
                        )
                    ],
                    "served_dimensions": [],
                },
                "gb": {
                    "size": cfg.gb_size_bits,
                    "r_cost": gb_cost.read_energy_pj_per_access,
                    "w_cost": gb_cost.write_energy_pj_per_access,
                    "area": 10.0,
                    "latency": 1,
                    "mem_type": "sram",
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.gb_bw_bits,
                            [
                                "I1, tl", "I1, fh",
                                "I2, tl", "I2, fh",
                                "O, fh", "O, fl", "O, th", "O, tl",
                            ],
                        )
                    ],
                    "served_dimensions": cfg.gb_served_dims,
                },
                # DRAM is a fixed platform IP, not an on-chip Level-2 gene.
                "dram": {
                    "size": 10**12,
                    "r_cost": dram_energy_pj_per_access,
                    "w_cost": dram_energy_pj_per_access,
                    "area": 0.0,
                    "latency": 1,
                    "mem_type": "dram",
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            self.dram_bandwidth_bits,
                            [
                                "I1, tl", "I1, fh",
                                "I2, tl", "I2, fh",
                                "O, fh", "O, fl", "O, th", "O, tl",
                            ],
                        )
                    ],
                    "served_dimensions": ["D1", "D2"],
                },
            },
        }

        return accelerator

    def _dram_dynamic_energy_pj_per_access(self) -> float:
        return self.energy_calibration.dram_energy_pj_per_access(
            self.dram_bandwidth_bits
        )

    def _dram_idle_energy_pj(self, latency_cycles: float) -> float:
        model = self.dram_power_model
        latency_s = latency_cycles / (
            model.reference_frequency_mhz * 1_000_000.0
        )
        return model.p_idle_w * latency_s * 1e12

    def _onchip_idle_energy_pj(
        self,
        cfg: ArchitectureConfig,
        profile: WorkloadActivityProfile,
    ) -> float:
        pe_count = cfg.pe_x * cfg.pe_y
        idle_pe_cycles = sum(
            max(
                0.0,
                pe_count * layer.latency_cycles - layer.mac_count,
            )
            for layer in profile.layers
        )
        pe_idle_energy = (
            idle_pe_cycles
            * self.energy_calibration.pe_idle_energy_pj_per_cycle
        )

        rf = self.energy_calibration.rf_cost(
            cfg.rf_size_bits,
            cfg.rf_bw_bits,
        )
        gb = self.energy_calibration.gb_cost(
            cfg.gb_size_bits,
            cfg.gb_bw_bits,
        )
        gb_count = prod(
            size
            for dimension, size in (
                ("D1", cfg.pe_x),
                ("D2", cfg.pe_y),
            )
            if dimension not in cfg.gb_served_dims
        )
        standby_power_w = (
            3 * pe_count * rf.standby_power_w
            + gb_count * gb.standby_power_w
        )
        frequency_hz = (
            self.dram_power_model.reference_frequency_mhz * 1_000_000.0
        )
        latency_s = profile.total_latency_cycles / frequency_hz
        leakage_energy_pj = standby_power_w * latency_s * 1e12
        memory_idle = (
            self.energy_calibration
            .memory_clock_idle_energy_pj_per_cycle
        )
        clock_idle_energy_pj = profile.total_latency_cycles * (
            3
            * pe_count
            * memory_idle(rf)
            + gb_count
            * memory_idle(gb)
        )
        return (
            pe_idle_energy
            + leakage_energy_pj
            + clock_idle_energy_pj
        )

    def _default_mapping(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "default",
                "memory_operand_links": {
                    "O": "O",
                    "W": "I2",
                    "I": "I1",
                },
            }
        ]

    def _extract_area(self, cme: Any, cfg: ArchitectureConfig) -> float:
        """
        First try to recover area from ZigZag's returned object.
        Fall back to a very rough analytical estimate.
        """
        candidate_attrs = [
            "area_total",
            "total_area",
            "area",
        ]

        for attr in candidate_attrs:
            if hasattr(cme, attr):
                value = getattr(cme, attr)
                if isinstance(value, (int, float)):
                    return float(value)

        if isinstance(cme, dict):
            for key in candidate_attrs:
                if key in cme and isinstance(cme[key], (int, float)):
                    return float(cme[key])

        if self.use_mock_area:
            return self._estimate_area(cfg)

        raise ValueError("ZigZag did not return an area value.")

    def _estimate_area(self, cfg: ArchitectureConfig) -> float:
        """
        Very rough placeholder area model.

        Replace this later with your own Level-2 IP characterization model.
        """
        mac_count = cfg.pe_x * cfg.pe_y
        gb_count = prod(
            size
            for dimension, size in (("D1", cfg.pe_x), ("D2", cfg.pe_y))
            if dimension not in cfg.gb_served_dims
        )

        mac_area = mac_count * 1.0
        rf_area = 3 * mac_count * cfg.rf_size_bits * 0.001
        gb_area = gb_count * cfg.gb_size_bits * 0.0005

        return float(mac_area + rf_area + gb_area)


_ZIGZAG_WORKER: ZigZagEvaluator | None = None


def _initialize_zigzag_worker(
    evaluator: ZigZagEvaluator,
) -> None:
    global _ZIGZAG_WORKER
    evaluator.workdir /= f"worker_{os.getpid()}"
    evaluator.workdir.mkdir(parents=True, exist_ok=True)
    evaluator.mapping_yaml_path = evaluator._write_mapping_yaml(
        evaluator.mapping
    )
    evaluator._evaluation_counter = 0
    _ZIGZAG_WORKER = evaluator


def _evaluate_zigzag_worker(
    genome: list[float],
) -> EvaluationResult:
    if _ZIGZAG_WORKER is None:
        raise RuntimeError("ZigZag worker was not initialized.")
    return _ZIGZAG_WORKER.evaluate(genome)
