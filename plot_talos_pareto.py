#!/usr/bin/env python3
"""Plot the best E/A/P sweep cases in a 3x3 Pareto scatter matrix.

Example:
  .venv/bin/python plot_talos_pareto.py results/vgg16_fp16_sweeps/20260822_025107
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OBJECTIVES = {
    "energy": ("energy",),
    "area": ("area",),
    "performance": ("latency",),
    "energy_area": ("energy", "area"),
    "area_performance": ("area", "latency"),
    "energy_performance": ("energy", "latency"),
    "energy_area_performance": ("energy", "area", "latency"),
}
ORDER = tuple(OBJECTIVES)
SHORT = {
    "energy": "E", "area": "A", "performance": "P", "energy_area": "E+A",
    "area_performance": "A+P", "energy_performance": "E+P",
    "energy_area_performance": "E+A+P",
}
METRICS = {
    "energy": ("workload_energy_j", "Energía total (J)"),
    "area": ("level2_area", "Área (mm²)"),
    "latency": ("workload_latency_s", "Latencia (s)"),
}


def valid_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [name for name, _ in METRICS.values()]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path}: faltan columnas {set(required) - set(df.columns)}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "constraints_satisfied" in df:
        df = df[df["constraints_satisfied"].astype(str).str.lower().eq("true")]
    return df[np.isfinite(df[required]).all(axis=1) & (df[required] > 0).all(axis=1)].copy()


def score(df: pd.DataFrame, objectives: tuple[str, ...]) -> pd.Series:
    """Distance logarítmica al ideal; todas las métricas se minimizan."""
    return np.sqrt(sum(np.log(df[METRICS[obj][0]] / df[METRICS[obj][0]].min()) ** 2 for obj in objectives))


def local_winners(path: Path, target: str) -> pd.DataFrame:
    df = valid_rows(path)
    if df.empty:
        return df
    objectives = OBJECTIVES[target]
    if len(objectives) == 1:
        values = df[METRICS[objectives[0]][0]]
        selected = df[np.isclose(values, values.min())].iloc[:1].copy()
    else:
        values = score(df, objectives)
        selected = df[np.isclose(values, values.min())].iloc[:1].copy()
    selected["run"] = path.parent.parent.name
    selected["target"] = target
    return selected


def select_case(candidates: pd.DataFrame, target: str) -> pd.DataFrame:
    """Choose one representative for each target across pools/seeds."""
    objectives = OBJECTIVES[target]
    if len(objectives) == 1:
        values = candidates[METRICS[objectives[0]][0]]
    else:
        values = score(candidates, objectives)
    return candidates[np.isclose(values, values.min())].iloc[:1].copy()


def nondominated(df: pd.DataFrame) -> np.ndarray:
    values = df[[name for name, _ in METRICS.values()]].to_numpy()
    return np.array([
        not np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))
        for point in values
    ])


def collect(root: Path) -> pd.DataFrame:
    per_target: dict[str, list[pd.DataFrame]] = {target: [] for target in ORDER}
    for path in sorted(root.rglob("full_flow_summary.csv")):
        target = path.parent.name
        if target in per_target:
            winners = local_winners(path, target)
            if not winners.empty:
                per_target[target].append(winners)

    chosen = []
    for target in ORDER:
        if not per_target[target]:
            print(f"WARNING: no hay resultados válidos para {target}")
            continue
        chosen.append(select_case(pd.concat(per_target[target], ignore_index=True), target))
    if not chosen:
        raise ValueError(f"No se encontraron full_flow_summary.csv válidos bajo {root}")
    result = pd.concat(chosen, ignore_index=True)
    result["pareto_3d"] = nondominated(result)
    return result


def plot(cases: pd.DataFrame, output: Path) -> None:
    metric_keys = tuple(METRICS)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for row, y_metric in enumerate(metric_keys):
        for col, x_metric in enumerate(metric_keys):
            ax = axes[row, col]
            if row == col:
                ax.axis("off")
                ax.text(.5, .5, METRICS[x_metric][1], ha="center", va="center", fontsize=15)
                continue
            x_col, x_label = METRICS[x_metric]
            y_col, y_label = METRICS[y_metric]
            for index, case in cases.reset_index(drop=True).iterrows():
                ax.scatter(case[x_col], case[y_col], s=70, color=colors(index),
                           edgecolor="black" if case.pareto_3d else "0.55", linewidth=1.5,
                           label=SHORT[case.target] if row == 0 and col == 1 else None)
                ax.annotate(SHORT[case.target], (case[x_col], case[y_col]), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=.25)
            if row == 2:
                ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel(y_label)
    fig.suptitle("Mejor caso por objetivo — borde negro: frente de Pareto E/A/P")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    probe = pd.DataFrame({"workload_energy_j": [1., 2., 1.], "level2_area": [1., 2., 2.],
                          "workload_latency_s": [1., 2., 1.]})
    assert nondominated(probe).tolist() == [True, False, False]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, help="PNG de salida (por defecto: <results_dir>/pareto_best_cases.png)")
    parser.add_argument("--csv", type=Path, help="CSV de los casos dibujados")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check: OK")
        return 0
    if args.results_dir is None:
        parser.error("results_dir es obligatorio salvo con --self-check")

    root = args.results_dir.resolve()
    cases = collect(root)
    output = args.output or root / "pareto_best_cases.png"
    plot(cases, output)
    csv = args.csv or output.with_suffix(".csv")
    columns = ["target", "run", "architecture_index", "level2_solution_index", *[v[0] for v in METRICS.values()], "pareto_3d"]
    cases[[column for column in columns if column in cases]].to_csv(csv, index=False)
    print(cases[[column for column in columns if column in cases]].to_string(index=False))
    print(f"\nPNG: {output}\nCSV: {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
