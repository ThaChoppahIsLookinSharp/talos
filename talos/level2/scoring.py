from __future__ import annotations

import math
from collections.abc import Sequence


AUGMENTED_TCHEBYCHEFF_RHO = 0.05


def augmented_tchebycheff_scores(
    objective_values: Sequence[Sequence[float]],
) -> list[float]:
    if not objective_values:
        return []

    values = [[float(value) for value in row] for row in objective_values]
    objective_count = len(values[0])
    if not objective_count or any(len(row) != objective_count for row in values):
        raise ValueError(
            "Augmented Tchebycheff scoring requires equally sized objective vectors."
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for row in values
        for value in row
    ):
        raise ValueError(
            "Logarithmic augmented Tchebycheff scoring requires strictly positive, finite "
            "minimization objective values."
        )

    log_minimums = [
        math.log(min(row[index] for row in values))
        for index in range(objective_count)
    ]
    scores = []
    for row in values:
        ratios = [
            math.log(value) - log_minimum
            for value, log_minimum in zip(row, log_minimums, strict=True)
        ]
        scores.append(
            100.0
            * (max(ratios) + AUGMENTED_TCHEBYCHEFF_RHO * sum(ratios))
        )
    return scores
