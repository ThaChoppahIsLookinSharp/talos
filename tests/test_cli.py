from __future__ import annotations

import contextlib
import io
import unittest

from talos.cli.main import build_parser, main


class CLITests(unittest.TestCase):
    def test_parser_builds_expected_subcommands(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["level2", "--pop-size", "4", "--generations", "1"])

        self.assertEqual(args.command, "level2")
        self.assertEqual(args.pop_size, 4)
        self.assertEqual(args.generations, 1)

    def test_pipeline_placeholder_does_not_crash(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            main(["pipeline"])

        self.assertIn("Hierarchical pipeline is not implemented yet", output.getvalue())


if __name__ == "__main__":
    unittest.main()
