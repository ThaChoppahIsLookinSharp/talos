#!/usr/bin/env python3
"""
Select winning Talos solutions from a target-sweep results directory.

Expected layout (recursively searched):

  <sweep>/energy/full_flow_summary.csv
  <sweep>/energy_area/full_flow_summary.csv
  <sweep>/area/full_flow_summary.csv
  <sweep>/performance/full_flow_summary.csv
  <sweep>/area_performance/full_flow_summary.csv
  <sweep>/energy_performance/full_flow_summary.csv
  <sweep>/energy_area_performance/full_flow_summary.csv

Winner selection used by default:
  - energy: minimum workload_energy_j
  - area: minimum area
  - performance: minimum workload_latency_s
  - multi-objective:

        r_i(x) = ln(f_i(x) / f_i_min)
        S(x)   = 100 * sqrt(sum_i r_i(x)^2)

    where f_i_min is computed over the union of all feasible rows found in
    the target CSVs. The minimum-S solution wins.

IMPORTANT:
  * By default, selection uses TOTAL workload energy (workload_energy_j),
    including DRAM, exactly as in the original sweep.
  * onchip_energy_j = workload_energy_j - dram_energy_j is also reported.
  * Existing/local balanced_score columns are ignored.
  * Exact ties are preserved as separate winner rows.

Examples:
  python select_talos_winners.py results/alexnet_int16_sweeps/20260816_155356

  python select_talos_winners.py results/alexnet_int16_sweeps/20260816_155356 \
      --output winners.csv

  # Re-select using on-chip energy instead of total energy:
  python select_talos_winners.py results/.../20260816_155356 \
      --energy-mode onchip
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "energy": ("energy",),
    "energy_area": ("energy", "area"),
    "area": ("area",),
    "performance": ("performance",),
    "area_performance": ("area", "performance"),
    "energy_performance": ("energy", "performance"),
    "energy_area_performance": ("energy", "area", "performance"),
}

TARGET_ORDER = list(TARGETS)

ALIASES = {
    "energy": ["workload_energy_j", "energy_j", "energy"],
    "dram": ["dram_energy_j", "workload_dram_energy_j", "dram_energy"],
    "latency": ["workload_latency_s", "latency_s", "latency"],
    "throughput": ["workload_throughput_ips", "throughput_ips", "throughput"],
    "area": ["area", "area_mm2", "total_area_mm2"],
    "power": ["power", "power_w", "workload_power_w", "average_power_w"],
    "arch": ["architecture_index", "arch_index", "architecture", "arch"],
    "solution": ["level2_solution_index", "solution_index", "level2_index", "solution"],
    "genome": ["level2_genome", "genome"],
    "pe": ["pe", "pe_name", "selected_pe", "pe_impl", "pe_ip"],
    "rf_i1": ["rf_i1", "rf_i1_name", "selected_rf_i1", "rf_i1_impl"],
    "rf_i2": ["rf_i2", "rf_i2_name", "selected_rf_i2", "rf_i2_impl"],
    "rf_o": ["rf_o", "rf_o_name", "selected_rf_o", "rf_o_impl"],
    "gb": ["gb", "gb_name", "selected_gb", "gb_impl", "global_buffer"],
}


def find_col(df: pd.DataFrame, key: str, required: bool = False) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in ALIASES[key]:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise KeyError(
            f"Could not find column for {key!r}. Tried {ALIASES[key]}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def infer_target(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in TARGETS:
            return part
    if path.stem in TARGETS:
        return path.stem
    return None


def discover_csvs(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {t: [] for t in TARGETS}

    paths = [root] if root.is_file() else list(root.rglob("*.csv"))
    for path in paths:
        target = infer_target(path)
        if target:
            candidates[target].append(path)

    selected: dict[str, Path] = {}
    for target, files in candidates.items():
        if not files:
            continue
        # Prefer the sweep-wide summary over per-architecture CSVs.
        files.sort(
            key=lambda p: (
                0 if p.name == "full_flow_summary.csv" else 1,
                len(p.parts),
                str(p),
            )
        )
        selected[target] = files[0]
    return selected


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    e_col = find_col(out, "energy", required=True)
    d_col = find_col(out, "dram", required=False)

    out[e_col] = pd.to_numeric(out[e_col], errors="coerce")
    if d_col is not None:
        out[d_col] = pd.to_numeric(out[d_col], errors="coerce")
        out["onchip_energy_j"] = out[e_col] - out[d_col]
    else:
        out["onchip_energy_j"] = np.nan
    return out


def positive_min(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    v = v[np.isfinite(v) & (v > 0)]
    if v.empty:
        raise ValueError("No finite positive values available for an objective.")
    return float(v.min())


def metric(df: pd.DataFrame, objective: str, energy_mode: str) -> pd.Series:
    if objective == "energy":
        if energy_mode == "onchip":
            if df["onchip_energy_j"].isna().all():
                raise KeyError("--energy-mode onchip requires a DRAM-energy column.")
            return pd.to_numeric(df["onchip_energy_j"], errors="coerce")
        return pd.to_numeric(df[find_col(df, "energy", True)], errors="coerce")
    if objective == "area":
        return pd.to_numeric(df[find_col(df, "area", True)], errors="coerce")
    if objective == "performance":
        return pd.to_numeric(df[find_col(df, "latency", True)], errors="coerce")
    raise ValueError(objective)


def global_ideal(dfs: dict[str, pd.DataFrame], energy_mode: str) -> dict[str, float]:
    return {
        obj: positive_min(
            pd.concat([metric(df, obj, energy_mode) for df in dfs.values()], ignore_index=True)
        )
        for obj in ("energy", "area", "performance")
    }


def select_winners(
    df: pd.DataFrame,
    target: str,
    ideal: dict[str, float],
    energy_mode: str,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> pd.DataFrame:
    objectives = TARGETS[target]
    work = df.copy()

    for obj in objectives:
        work[f"__{obj}"] = metric(work, obj, energy_mode)

    valid = np.ones(len(work), dtype=bool)
    for obj in objectives:
        x = work[f"__{obj}"].to_numpy(dtype=float)
        valid &= np.isfinite(x) & (x > 0)
    work = work.loc[valid].copy()

    if work.empty:
        raise ValueError(f"No valid rows for target {target}")

    if len(objectives) == 1:
        obj = objectives[0]
        x = work[f"__{obj}"].to_numpy(dtype=float)
        best = np.min(x)
        winners = work.loc[np.isclose(x, best, rtol=rtol, atol=atol)].copy()
        winners["selection_score"] = np.nan
        return winners

    score_sq = np.zeros(len(work), dtype=float)
    for obj in objectives:
        x = work[f"__{obj}"].to_numpy(dtype=float)
        r = np.log(x / ideal[obj])
        score_sq += r * r

    work["selection_score"] = 100.0 * np.sqrt(score_sq)
    best = work["selection_score"].min()
    return work.loc[
        np.isclose(work["selection_score"], best, rtol=rtol, atol=atol)
    ].copy()


def get_value(row: pd.Series, df: pd.DataFrame, key: str):
    col = find_col(df, key, required=False)
    if col is None:
        return np.nan
    return row[col]


def result_row(target: str, path: Path, df: pd.DataFrame, idx, row: pd.Series) -> dict:
    e_col = find_col(df, "energy", True)
    d_col = find_col(df, "dram", False)
    l_col = find_col(df, "latency", True)
    t_col = find_col(df, "throughput", False)
    a_col = find_col(df, "area", True)
    p_col = find_col(df, "power", False)

    onchip = row.get("onchip_energy_j", np.nan)

    return {
        "target": target,
        "architecture": get_value(row, df, "arch"),
        "level2_solution": get_value(row, df, "solution"),
        "genome": get_value(row, df, "genome"),
        "workload_energy_j": row[e_col],
        "dram_energy_j": row[d_col] if d_col else np.nan,
        "onchip_energy_j": onchip,
        "onchip_energy_mj": onchip * 1000 if pd.notna(onchip) else np.nan,
        "workload_latency_s": row[l_col],
        "workload_throughput_ips": row[t_col] if t_col else np.nan,
        "area_mm2": row[a_col],
        "power_w": row[p_col] if p_col else np.nan,
        "selection_score": row.get("selection_score", np.nan),
        "PE": get_value(row, df, "pe"),
        "RF_I1": get_value(row, df, "rf_i1"),
        "RF_I2": get_value(row, df, "rf_i2"),
        "RF_O": get_value(row, df, "rf_o"),
        "GB": get_value(row, df, "gb"),
        "source_csv": str(path),
        "csv_line": int(idx) + 2 if isinstance(idx, (int, np.integer)) else idx,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument(
        "--energy-mode",
        choices=("total", "onchip"),
        default="total",
        help="Energy used to SELECT winners. Default: total workload energy including DRAM.",
    )
    ap.add_argument("--output", type=Path, help="Optional output CSV.")
    args = ap.parse_args()

    root = args.results_dir.resolve()
    csvs = discover_csvs(root)
    if not csvs:
        raise SystemExit(f"No target CSVs found below {root}")

    missing = [t for t in TARGET_ORDER if t not in csvs]
    if missing:
        print("WARNING: missing targets:", ", ".join(missing))

    dfs = {target: add_derived_columns(pd.read_csv(path)) for target, path in csvs.items()}
    ideal = global_ideal(dfs, args.energy_mode)

    print("Ideal point used by the log-ratio score:")
    print(f"  energy  = {ideal['energy']:.12g} J")
    print(f"  area    = {ideal['area']:.12g} mm^2")
    print(f"  latency = {ideal['performance']:.12g} s")
    print(f"Energy mode for selection: {args.energy_mode}\n")

    rows = []
    for target in TARGET_ORDER:
        if target not in dfs:
            continue
        winners = select_winners(dfs[target], target, ideal, args.energy_mode)
        for idx, row in winners.iterrows():
            rows.append(result_row(target, csvs[target], dfs[target], idx, row))

    out = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(TARGET_ORDER)}
    out["__order"] = out["target"].map(order)
    out = out.sort_values(["__order", "csv_line"]).drop(columns="__order")

    display_cols = [
        "target",
        "architecture",
        "level2_solution",
        "workload_energy_j",
        "dram_energy_j",
        "onchip_energy_mj",
        "workload_latency_s",
        "workload_throughput_ips",
        "area_mm2",
        "power_w",
        "selection_score",
        "PE",
        "GB",
    ]
    display_cols = [c for c in display_cols if c in out.columns]

    shown = out[display_cols].copy()
    for col, digits in {
        "workload_energy_j": 6,
        "dram_energy_j": 6,
        "onchip_energy_mj": 3,
        "workload_latency_s": 6,
        "workload_throughput_ips": 3,
        "area_mm2": 6,
        "power_w": 4,
        "selection_score": 4,
    }.items():
        if col in shown:
            shown[col] = pd.to_numeric(shown[col], errors="coerce").round(digits)

    try:
        print(shown.to_markdown(index=False))
    except ImportError:
        print(shown.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
