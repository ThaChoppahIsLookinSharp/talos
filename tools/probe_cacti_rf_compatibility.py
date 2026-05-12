from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.architecture.genome import default_genome  # noqa: E402
from talos.architecture.memory_specs import (  # noqa: E402
    RF_BANDWIDTH_MIN_BITS,
    RF_SIZE_OPTIONS,
    bits_to_bytes,
)
from talos.evaluation.zigzag_evaluator import ZigZagEvaluator  # noqa: E402


RF_SIZE_BYTES = [bits_to_bytes(size_bits) for size_bits in RF_SIZE_OPTIONS]
RF_BANDWIDTH_BITS = [8, 16, 32, 64, 128, 256]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cacti_master_path() -> Path:
    import zigzag.cacti.cacti_parser as cacti_parser

    return Path(cacti_parser.CactiParser.CACTI_TOP_PATH).resolve().parent


def build_rf_only_auto_yaml(size_bytes: int, bandwidth_bits: int, path: Path) -> None:
    evaluator = ZigZagEvaluator(
        workload=str(repo_root() / "workloads" / "alexnet.onnx"),
        memory_cost_mode="manual",
        workdir=str(repo_root() / ".talos_zigzag" / "cacti_rf_compatibility" / "yaml_workdir"),
        lpf_limit=1,
        nb_spatial_mappings_generated=1,
    )
    accelerator = evaluator.build_accelerator_from_genome(default_genome())

    for mem_name, memory in accelerator["memories"].items():
        if mem_name.startswith("rf_"):
            memory["size"] = size_bytes * 8
            memory["r_cost"] = None
            memory["w_cost"] = None
            memory["area"] = None
            memory["auto_cost_extraction"] = True
            for port in memory["ports"]:
                port["bandwidth_min"] = min(RF_BANDWIDTH_MIN_BITS, bandwidth_bits)
                port["bandwidth_max"] = bandwidth_bits
        else:
            memory["auto_cost_extraction"] = False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(accelerator, sort_keys=False), encoding="utf-8")


def parse_cacti_pool(mem_pool_path: Path) -> dict[str, Any] | None:
    if not mem_pool_path.exists() or mem_pool_path.stat().st_size == 0:
        return None
    data = yaml.safe_load(mem_pool_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        return None
    first_key = next(iter(data))
    value = data[first_key]
    return value if isinstance(value, dict) else None


def compact_error(stdout: str, stderr: str, returncode: int) -> str:
    combined = "\n".join(part for part in (stdout, stderr) if part).strip()
    if not combined:
        return f"CACTI subprocess failed with return code {returncode}."

    interesting: list[str] = []
    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "Number of sets is too small" in stripped
            or "Need to either" in stripped
            or "FileNotFoundError" in stripped
            or "ChildProcessError" in stripped
            or "Traceback" in stripped
            or "No such file" in stripped
            or "ValueError" in stripped
            or "could not convert string to float" in stripped
            or "ERROR: no valid data array organizations found" in stripped
        ):
            interesting.append(stripped)
    if interesting:
        return " | ".join(interesting[:4])

    return " | ".join(line.strip() for line in combined.splitlines()[:4] if line.strip())


def parse_cache_cfg_out(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None

    headers = [header.strip() for header in rows[0]]
    values = [value.strip() for value in rows[1]]
    data = dict(zip(headers, values, strict=False))

    try:
        return {
            "size_byte": int(float(data["Capacity (bytes)"])),
            "area": float(data["Area (mm2)"]),
            "cost": {
                "read_word": float(data["Dynamic read energy (nJ)"]),
                "write_word": float(data["Dynamic write energy (nJ)"]),
            },
            "IO_bus_width": int(float(data["Output width (bits)"])),
        }
    except (KeyError, TypeError, ValueError):
        return None


def run_cacti(size_bytes: int, bandwidth_bits: int, mem_pool_path: Path) -> tuple[bool, bool, dict[str, Any] | None, str]:
    master = cacti_master_path()
    cacti_top = master / "cacti_top.py"
    cache_out_path = master / "self_gen" / "cache.cfg.out"
    mem_pool_path.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    repo = str(repo_root())
    site = str(Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    env["PYTHONPATH"] = os.pathsep.join([site, repo, env.get("PYTHONPATH", "")])

    completed = subprocess.run(
        [
            "python",
            str(cacti_top),
            "--mem_type",
            "sram",
            "--cache_size",
            str(size_bytes),
            "--IO_bus_width",
            str(bandwidth_bits),
            "--ex_rd_port",
            "1",
            "--ex_wr_port",
            "1",
            "--rd_wr_port",
            "0",
            "--bank_count",
            "1",
            "--mem_pool_path",
            str(mem_pool_path),
            "--technology",
            "0.022",
        ],
        cwd=str(master),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    parsed_pool = parse_cacti_pool(mem_pool_path)
    parsed_cache_out = parse_cache_cfg_out(cache_out_path)
    parsed = parsed_pool or parsed_cache_out
    wrapper_valid = completed.returncode == 0 and parsed_pool is not None
    cacti_output_valid = parsed is not None
    if wrapper_valid:
        return True, True, parsed, ""
    return False, cacti_output_valid, parsed, compact_error(completed.stdout, completed.stderr, completed.returncode)


def main() -> None:
    out_dir = repo_root() / ".talos_zigzag" / "cacti_rf_compatibility"
    yaml_dir = out_dir / "yamls"
    pool_dir = cacti_master_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "rf_cacti_compatibility.csv"
    fieldnames = [
        "rf_size_bytes",
        "rf_bandwidth_max_bits",
        "zigzag_wrapper_valid",
        "cacti_output_valid",
        "area_mm2",
        "read_energy_pj",
        "write_energy_pj",
        "read_energy_nj",
        "write_energy_nj",
        "error_cacti",
        "yaml_path",
        "mem_pool_path",
    ]

    rows: list[dict[str, Any]] = []
    for size_bytes in RF_SIZE_BYTES:
        for bandwidth_bits in RF_BANDWIDTH_BITS:
            yaml_path = yaml_dir / f"rf_{size_bytes}B_bw_{bandwidth_bits}b.yaml"
            build_rf_only_auto_yaml(size_bytes, bandwidth_bits, yaml_path)

            mem_pool_path = pool_dir / f"pool_rf_{size_bytes}B_bw_{bandwidth_bits}b.yaml"
            wrapper_valid, cacti_output_valid, data, error = run_cacti(size_bytes, bandwidth_bits, mem_pool_path)

            read_nj = data["cost"]["read_word"] if data else ""
            write_nj = data["cost"]["write_word"] if data else ""
            rows.append(
                {
                    "rf_size_bytes": size_bytes,
                    "rf_bandwidth_max_bits": bandwidth_bits,
                    "zigzag_wrapper_valid": wrapper_valid,
                    "cacti_output_valid": cacti_output_valid,
                    "area_mm2": data["area"] if data else "",
                    "read_energy_pj": read_nj * 1000 if data else "",
                    "write_energy_pj": write_nj * 1000 if data else "",
                    "read_energy_nj": read_nj,
                    "write_energy_nj": write_nj,
                    "error_cacti": error,
                    "yaml_path": str(yaml_path),
                    "mem_pool_path": str(mem_pool_path),
                }
            )
            if wrapper_valid:
                status = "OK"
            elif cacti_output_valid:
                status = "CACTI_OUTPUT_ONLY"
            else:
                status = "FAIL"
            print(f"{status}: RF={size_bytes} B, bw={bandwidth_bits} b")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
