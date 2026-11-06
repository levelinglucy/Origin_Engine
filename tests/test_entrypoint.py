"""Origin Engine CLI smoke checks; isolated, standard-library only."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "origin_engine_monolith.py"
ARCHIVE = ROOT / "python3 origin_engine_monolith.py"


class EntrypointChecks(unittest.TestCase):
    def invoke(self, *arguments: str, cwd: Path):
        return subprocess.run(
            [sys.executable, str(ENTRY), *arguments],
            cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", timeout=45, check=False,
        )

    def test_historical_source_is_unchanged(self):
        self.assertTrue(ENTRY.is_file())
        self.assertTrue(ARCHIVE.is_file())
        self.assertEqual(ENTRY.read_bytes(), ARCHIVE.read_bytes())

    def test_help_runs(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.invoke("--help", cwd=Path(d))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--cycles", run.stdout)
        self.assertIn("--graph-format", run.stdout)

    def test_two_cycles_print_output_and_scorecard(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.invoke("--cycles", "2", cwd=Path(d))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("=== Ψ Emissions ===", run.stdout)
        self.assertIn("01:", run.stdout)
        self.assertIn("02:", run.stdout)
        self.assertIn("=== Tests ===", run.stdout)
        self.assertIn("Overall Avg:", run.stdout)

    def test_invalid_pipeline_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = root / "bad-spec.json"
            spec.write_text(json.dumps({"pipeline": ["P", "Ψ"]}), encoding="utf-8")
            run = self.invoke("--spec", str(spec), "--cycles", "1", cwd=root)
        self.assertEqual(run.returncode, 2)
        self.assertIn("Pipeline mismatch", run.stdout)


if __name__ == "__main__":
    unittest.main()
