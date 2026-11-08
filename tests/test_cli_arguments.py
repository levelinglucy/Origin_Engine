"""CLI input-contract checks: invalid requests must not masquerade as successful runs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


def invoke(*arguments, cwd):
    return subprocess.run(
        [sys.executable, str(ENTRY), *arguments],
        cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


class CLIBoundsChecks(unittest.TestCase):
    def test_zero_and_negative_cycles_are_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for count in ("0", "-3"):
                with self.subTest(cycles=count):
                    run = invoke("--cycles", count, "--seed", "7", cwd=folder)
                    self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                    self.assertIn("--cycles", run.stderr)
                    self.assertNotIn("=== Ψ Emissions ===", run.stdout)
                    self.assertFalse((folder / "oe_graphs").exists())

    def test_nonfinite_and_negative_novelty_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for value in ("nan", "inf", "-inf", "-0.01"):
                with self.subTest(dmin=value):
                    run = invoke("--cycles", "1", "--dmin", value,
                                 "--seed", "7", cwd=folder)
                    self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                    self.assertIn("--dmin", run.stderr)
                    self.assertNotIn("=== Ψ Emissions ===", run.stdout)

    def test_positive_cycles_and_zero_novelty_are_valid(self):
        with tempfile.TemporaryDirectory() as d:
            run = invoke("--cycles", "1", "--dmin", "0", "--seed", "7",
                         cwd=Path(d))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("01:", run.stdout)
        self.assertIn("Overall Avg:", run.stdout)


if __name__ == "__main__":
    unittest.main()
