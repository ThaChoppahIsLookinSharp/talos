from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import sys
from typing import Any
import yaml

from talos.architecture.genome import ArchitectureConfig, decode_genome


@dataclass
class EvaluationResult:
    latency: float
    energy: float
    area: float
    valid: bool
    error_message: str | None = None
    area_source: str = "missing"
    area_is_proxy: bool = False
    raw_zigzag_area: float | None = None
    zigzag_area_path: str | None = None
    memory_cost_mode: str = "manual"


class ZigZagEvaluator:
    """
    TALOS -> ZigZag compatibility layer.

    Genome semantics live in talos.architecture.genome. This evaluator
    only consumes decoded architecture configs and runs ZigZag.

    Level 1 keeps the third objective exposed as ``area`` for compatibility,
    but its provenance is tracked explicitly. The area policy decides whether
    TALOS uses a usable ZigZag area estimate, its internal architectural cost
    proxy, or a preference order between both. That proxy is only meant to
    guide the Level-1 search, not to represent physical implementation area.
    """

    VALID_AREA_POLICIES = {
        "zigzag_only",
        "prefer_zigzag_then_proxy",
        "proxy_only",
    }
    VALID_MEMORY_COST_MODES = {
        "manual",
        "zigzag_auto",
    }

    def __init__(
        self,
        workload: str,
        mapping: list[dict[str, Any]] | None = None,
        opt: str = "EDP",
        area_policy: str = "prefer_zigzag_then_proxy",
        memory_cost_mode: str = "manual",
        workdir: str | None = None,
        debug: bool = False,
        debug_cme: bool = False,
        lpf_limit: int = 6,
        nb_spatial_mappings_generated: int = 3,
    ) -> None:
        self.workload = workload
        self.mapping = mapping if mapping is not None else self._default_mapping()
        self.opt = opt
        if area_policy not in self.VALID_AREA_POLICIES:
            valid = ", ".join(sorted(self.VALID_AREA_POLICIES))
            raise ValueError(
                f"Unknown area_policy {area_policy!r}. Expected one of: {valid}."
            )
        self.area_policy = area_policy
        if memory_cost_mode not in self.VALID_MEMORY_COST_MODES:
            valid = ", ".join(sorted(self.VALID_MEMORY_COST_MODES))
            raise ValueError(
                f"Unknown memory_cost_mode {memory_cost_mode!r}. Expected one of: {valid}."
            )
        self.memory_cost_mode = memory_cost_mode
        self.workdir = (
            Path(workdir) if workdir is not None else Path.cwd() / ".talos_zigzag"
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.debug_cme = debug_cme
        self.lpf_limit = lpf_limit
        self.nb_spatial_mappings_generated = nb_spatial_mappings_generated
        self.mapping_yaml_path = self._write_mapping_yaml(self.mapping)
        self._evaluation_counter = 0

    def evaluate(self, genome: list[float]) -> EvaluationResult:
        try:
            cfg = decode_genome(genome)
            accelerator = self._build_accelerator(cfg)
            accelerator_yaml_path = self._write_accelerator_yaml(accelerator)

            if self.debug:
                self._print_debug_yaml(accelerator_yaml_path)
                with self._zigzag_runtime_env():
                    energy, latency, cme = self._run_zigzag(accelerator_yaml_path)
            else:
                with self._quiet_zigzag():
                    with self._zigzag_runtime_env():
                        energy, latency, cme = self._run_zigzag(accelerator_yaml_path)

            if self.debug_cme:
                self._print_cme_inspection(cme)

            area, area_source, raw_zigzag_area, zigzag_area_path = self._extract_area(
                cme,
                cfg,
            )

            if self.debug and area_source == "proxy":
                print(
                    "Using TALOS area proxy for the Level-1 third objective."
                )
            if self.debug_cme and zigzag_area_path is not None:
                print(f"Resolved ZigZag area from path: {zigzag_area_path}")

            return EvaluationResult(
                latency=float(latency),
                energy=float(energy),
                area=float(area),
                valid=True,
                area_source=area_source,
                area_is_proxy=(area_source == "proxy"),
                raw_zigzag_area=raw_zigzag_area,
                zigzag_area_path=zigzag_area_path,
                memory_cost_mode=self.memory_cost_mode,
            )

        except Exception as exc:
            if self.debug:
                import traceback

                print("ZigZag evaluation failed:")
                traceback.print_exc()

            error_message = str(exc)
            if self.memory_cost_mode == "zigzag_auto":
                error_message = self._format_memory_cost_mode_error(exc)

            return EvaluationResult(
                latency=float("inf"),
                energy=float("inf"),
                area=float("inf"),
                valid=False,
                error_message=error_message,
                area_source="missing",
                area_is_proxy=False,
                raw_zigzag_area=None,
                zigzag_area_path=None,
                memory_cost_mode=self.memory_cost_mode,
            )

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

        return get_hardware_performance_zigzag(
            workload=self.workload,
            accelerator=accelerator_yaml_path,
            mapping=self.mapping_yaml_path,
            opt=self.opt,
            dump_folder=self._next_dump_folder(),
            pickle_filename=None,
            lpf_limit=self.lpf_limit,
            nb_spatial_mappings_generated=self.nb_spatial_mappings_generated,
            loma_show_progress_bar=self.debug,
        )

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

    @contextlib.contextmanager
    def _zigzag_runtime_env(self) -> Iterator[None]:
        """
        Ensure ZigZag/CACTI subprocesses inherit the active Python environment.

        ZigZag 3.8.5 launches CACTI helpers through a bare ``python`` command.
        When the PATH points to a different interpreter, auto memory cost
        extraction can fail even if the current TALOS venv is correctly set up.
        """
        if self.memory_cost_mode != "zigzag_auto":
            yield
            return

        python_executable = Path(sys.executable).resolve()
        python_dir = str(python_executable.parent)
        site_packages = str(python_executable.parent.parent / "Lib" / "site-packages")
        repo_root = str(Path(__file__).resolve().parents[2])
        previous_path = os.environ.get("PATH", "")
        previous_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PATH"] = os.pathsep.join([python_dir, previous_path])
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [site_packages, repo_root, previous_pythonpath]
            if previous_pythonpath
            else [site_packages, repo_root]
        )
        try:
            yield
        finally:
            os.environ["PATH"] = previous_path
            if previous_pythonpath:
                os.environ["PYTHONPATH"] = previous_pythonpath
            else:
                os.environ.pop("PYTHONPATH", None)

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

    def build_accelerator_from_genome(self, genome: list[float]) -> dict[str, Any]:
        """Build the accelerator description used for the given TALOS genome."""
        return self._build_accelerator(decode_genome(genome))

    def render_accelerator_yaml(self, genome: list[float]) -> str:
        """Render the accelerator YAML for inspection/debug without running ZigZag."""
        accelerator = self.build_accelerator_from_genome(genome)
        return yaml.safe_dump(accelerator, sort_keys=False)

    def _memory_cost_fields(
        self,
        *,
        size_bits: int,
        r_cost: float,
        w_cost: float,
        area: float,
        latency: int,
        mem_type: str,
    ) -> dict[str, Any]:
        if self.memory_cost_mode == "manual":
            return {
                "size": size_bits,
                "r_cost": r_cost,
                "w_cost": w_cost,
                "area": area,
                "latency": latency,
                "mem_type": mem_type,
                "auto_cost_extraction": False,
            }

        return {
            "size": size_bits,
            "r_cost": None,
            "w_cost": None,
            "area": None,
            "latency": latency,
            "mem_type": mem_type,
            "auto_cost_extraction": True,
        }

    def _build_accelerator(self, cfg: ArchitectureConfig) -> dict[str, Any]:
        accelerator = {
            "name": "talos_candidate",
            "operational_array": {
                "is_imc": False,
                "unit_energy": 1.0,
                "unit_area": 1.0,
                "dimensions": ["D1", "D2"],
                "sizes": [cfg.pe_x, cfg.pe_y],
                "imc_type": None,
                "adc_resolution": 0,
                "bit_serial_precision": None,
            },
            "memories": {
                "rf_i1": {
                    **self._memory_cost_fields(
                        size_bits=cfg.rf_size_bits,
                        r_cost=1.0,
                        w_cost=1.0,
                        area=1.0,
                        latency=1,
                        mem_type="sram",
                    ),
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
                    **self._memory_cost_fields(
                        size_bits=cfg.rf_size_bits,
                        r_cost=1.0,
                        w_cost=1.0,
                        area=1.0,
                        latency=1,
                        mem_type="sram",
                    ),
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
                    **self._memory_cost_fields(
                        size_bits=cfg.rf_size_bits,
                        r_cost=1.0,
                        w_cost=1.0,
                        area=1.0,
                        latency=1,
                        mem_type="sram",
                    ),
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
                    **self._memory_cost_fields(
                        size_bits=cfg.gb_size_bits,
                        r_cost=10.0,
                        w_cost=10.0,
                        area=10.0,
                        latency=1,
                        mem_type="sram",
                    ),
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
                "dram": {
                    **self._memory_cost_fields(
                        size_bits=10**12,
                        r_cost=1000.0,
                        w_cost=1000.0,
                        area=0.0,
                        latency=1,
                        mem_type="dram",
                    ),
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        self._rw_port(
                            "rw_port_1",
                            cfg.dram_bw_bits,
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

    def _coerce_numeric_area(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _extract_area_from_accelerator(self, accelerator: Any) -> tuple[float | None, str | None]:
        operational_array = getattr(accelerator, "operational_array", None)
        operational_area = self._coerce_numeric_area(
            getattr(operational_array, "total_area", None)
        )
        memory_hierarchy = getattr(accelerator, "memory_hierarchy", None)
        mem_level_list = getattr(memory_hierarchy, "mem_level_list", None)
        if operational_area is None or not isinstance(mem_level_list, list):
            return None, None

        total_area = operational_area
        seen_instances: set[int] = set()
        for level in mem_level_list:
            instance = getattr(level, "memory_instance", None) or getattr(level, "instance", None)
            if instance is None:
                continue
            instance_id = id(instance)
            if instance_id in seen_instances:
                continue
            seen_instances.add(instance_id)
            instance_area = self._coerce_numeric_area(getattr(instance, "area", None))
            if instance_area is None:
                return None, None
            total_area += instance_area

        return (
            float(total_area),
            "accelerator.operational_array.total_area + "
            "accelerator.memory_hierarchy.mem_level_list[*].instance.area",
        )

    def _iter_zigzag_area_candidates(
        self,
        obj: Any,
        *,
        path: str,
        depth: int,
        max_depth: int,
        seen: set[int],
    ) -> Iterator[tuple[float, str]]:
        """
        Yield candidate area values from common ZigZag result structures.
        """
        obj_id = id(obj)
        if obj_id in seen or depth > max_depth:
            return
        seen.add(obj_id)

        candidate_names = ("area_total", "total_area", "area")

        if isinstance(obj, dict):
            for key in candidate_names:
                value = self._coerce_numeric_area(obj.get(key))
                if value is not None:
                    yield value, f"{path}.{key}" if path else key
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else str(key)
                yield from self._iter_zigzag_area_candidates(
                    value,
                    path=next_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                    seen=seen,
                )
            return

        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj[:8]):
                next_path = f"{path}[{idx}]"
                yield from self._iter_zigzag_area_candidates(
                    item,
                    path=next_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                    seen=seen,
                )
            return

        for name in candidate_names:
            try:
                value = self._coerce_numeric_area(getattr(obj, name, None))
            except Exception:
                value = None
            if value is not None:
                yield value, f"{path}.{name}" if path else name

        accelerator = getattr(obj, "accelerator", None)
        if accelerator is not None:
            area, area_path = self._extract_area_from_accelerator(accelerator)
            if area is not None and area_path is not None:
                prefix = f"{path}." if path else ""
                yield area, f"{prefix}{area_path}"

        for name in ("results", "result", "summary", "cme", "core", "accelerator"):
            try:
                value = getattr(obj, name, None)
            except Exception:
                value = None
            if value is None:
                continue
            next_path = f"{path}.{name}" if path else name
            yield from self._iter_zigzag_area_candidates(
                value,
                path=next_path,
                depth=depth + 1,
                max_depth=max_depth,
                seen=seen,
            )

    def _extract_zigzag_area(self, cme: Any) -> tuple[float | None, str | None]:
        for value, path in self._iter_zigzag_area_candidates(
            cme,
            path="",
            depth=0,
            max_depth=4,
            seen=set(),
        ):
            cleaned_path = path[1:] if path.startswith(".") else path
            return value, cleaned_path
        return None, None

    def _summarize_cme(self, cme: Any) -> str:
        lines: list[str] = [f"type={type(cme)}"]
        if isinstance(cme, list):
            lines.append(f"len={len(cme)}")
            for idx, item in enumerate(cme[:3]):
                lines.append(f"[{idx}] type={type(item)}")
                if isinstance(item, tuple):
                    lines.append(f"[{idx}] tuple_len={len(item)}")
        elif isinstance(cme, dict):
            lines.append(f"keys={list(cme.keys())[:12]}")
        else:
            attrs = [a for a in dir(cme) if not a.startswith("_")]
            interesting = [
                a for a in attrs if any(tok in a.lower() for tok in ("area", "energy", "lat", "mem", "cost"))
            ]
            lines.append(f"attrs={interesting[:12]}")
        return " | ".join(lines)

    def _print_cme_inspection(self, cme: Any) -> None:
        print(f"CME inspection: {self._summarize_cme(cme)}")
        area, path = self._extract_zigzag_area(cme)
        if area is None:
            print("CME area inspection: no usable ZigZag area found.")
        else:
            print(f"CME area inspection: found area={area} at {path}")

    def _extract_area(
        self,
        cme: Any,
        cfg: ArchitectureConfig,
    ) -> tuple[float, str, float | None, str | None]:
        """
        Resolve the Level-1 third objective value and its provenance.

        TALOS keeps returning ``area`` for compatibility with the current
        pipeline, but the metadata records whether the value came from ZigZag
        or from TALOS' internal architectural cost proxy fallback.
        """
        if self.area_policy == "proxy_only":
            return self._estimate_area_proxy(cfg), "proxy", None, None

        zigzag_area, zigzag_area_path = self._extract_zigzag_area(cme)
        if zigzag_area is not None:
            return zigzag_area, "zigzag", zigzag_area, zigzag_area_path

        if self.area_policy == "prefer_zigzag_then_proxy":
            return self._estimate_area_proxy(cfg), "proxy", None, None

        raise ValueError(
            "ZigZag did not return a usable area value and area_policy='zigzag_only'."
        )

    def _estimate_area_proxy(self, cfg: ArchitectureConfig) -> float:
        """
        Very rough architectural cost proxy for Level 1.

        This is not physical implementation area. It is only a coarse
        heuristic based on array and memory sizes so TALOS can still guide the
        search when ZigZag does not expose a usable area metric. Level 2 is
        expected to replace this with a more serious PPA re-estimation flow.
        """
        mac_count = cfg.pe_x * cfg.pe_y

        mac_area = mac_count * 1.0
        rf_area = mac_count * cfg.rf_size_bits * 0.001
        gb_area = cfg.gb_size_bits * 0.0005

        return float(mac_area + rf_area + gb_area)

    def _format_memory_cost_mode_error(self, exc: Exception) -> str:
        base = str(exc)
        if os.name == "nt":
            return (
                f"{base} "
                "ZigZag 3.8.5 memory auto-cost extraction is currently experimental in this "
                "Windows environment: its CACTI helper uses Unix-specific commands and path "
                "assumptions, so `memory_cost_mode='zigzag_auto'` may fail at runtime."
            )
        return base
