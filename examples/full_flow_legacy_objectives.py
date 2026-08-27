"""Run the full flow with the requested Level-1/ZigZag objective set."""

from __future__ import annotations

import full_flow_example as full_flow


def main() -> int:
    args = full_flow.parse_args()
    full_flow.LEVEL1_SCREENING_OBJECTIVES = list(args.level1_objectives)
    return full_flow.main()


if __name__ == "__main__":
    raise SystemExit(main())
