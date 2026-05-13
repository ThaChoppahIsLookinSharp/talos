from __future__ import annotations

from typing import Any


def run_hierarchical_pipeline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "Hierarchical pipeline is not implemented yet. "
        "Use `python -m talos level1` and `python -m talos level2` independently."
    )
