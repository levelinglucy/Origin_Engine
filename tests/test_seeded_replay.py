"""Opt-in seeded replay regression checks for Origin Engine.

No-seed mode must keep OS-backed entropy; replay must not patch process-wide
random state or depend on wall-clock filenames for comparison.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "origin_engine_monolith.py"
module_spec = importlib.util.spec_from_file_location("oe_seeded_replay", ENTRY)
assert module_spec is not None and module_spec.loader is not None
oe = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = oe
module_spec.loader.exec_module(oe)


class SeededReplayChecks(unittest.TestCase):
    def invoke(self, seed: int) -> str:
        with tempfile.TemporaryDirectory() as d:
            run = subprocess.run(
                [sys.executable, str(ENTRY), "--cycles", "3", "--seed", str(seed)],
                cwd=d, capture_output=True, text=True, encoding="utf-8",
                timeout=45, check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Overall Avg:", run.stdout)
            return run.stdout

    def test_same_seed_repeats_in_independent_python_processes(self):
        self.assertEqual(self.invoke(0), self.invoke(0))

    def test_different_seeds_diverge_in_independent_processes(self):
        self.assertNotEqual(self.invoke(12), self.invoke(13))

    def test_default_still_uses_os_entropy_and_seed_does_not(self):
        with patch.object(oe.os, "urandom", side_effect=RuntimeError("OS entropy requested")):
            with self.assertRaisesRegex(RuntimeError, "OS entropy requested"):
                oe.OriginEngine().cycle()
            # Seeded cycle must avoid os.urandom in both P and C.
            seeded = oe.OriginEngine(seed=5)
            self.assertTrue(seeded.cycle().payload)
            self.assertTrue(seeded.cycle().payload)

    def test_separate_engine_streams_are_not_affected_by_interleaving(self):
        a = oe.OriginEngine(seed=99)
        b = oe.OriginEngine(seed=123)
        first = a.cycle().payload
        b.cycle()
        b.cycle()
        second = a.cycle().payload
        control = oe.OriginEngine(seed=99)
        self.assertEqual([first, second], [control.cycle().payload, control.cycle().payload])

    @staticmethod
    def graph_snapshot(seed: int, fmt: str):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(oe.os, "getcwd", return_value=d):
                engine = oe.OriginEngine(seed=seed)
                engine.phase.level = 4
                engine.phase.crit = float("inf")
                engine.phase.graph_format = fmt
                emissions = [engine.cycle() for _ in range(2)]
                normalized = []
                for item in emissions:
                    meta = dict(item.meta)
                    path = Path(meta.pop("graph_file"))
                    assert path.is_file()
                    disk = (json.loads(path.read_text(encoding="utf-8"))
                            if fmt == "json" else oe._parse_graphml(str(path)))
                    assert oe._graphs_equivalent(meta["graph"], disk)
                    normalized.append({"payload": item.payload, "meta": meta})
                return normalized

    def test_graph_outputs_replay_with_normalized_paths(self):
        for fmt in ("json", "graphml"):
            with self.subTest(fmt=fmt):
                first = self.graph_snapshot(719, fmt)
                self.assertEqual(first, self.graph_snapshot(719, fmt))
                self.assertNotEqual(first, self.graph_snapshot(720, fmt))


if __name__ == "__main__":
    unittest.main()
