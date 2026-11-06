"""End-to-end OriginEngine phase evolution and export/readback checks.

Runs the real P→S→U→Σ→C→Ψ cycle chain in temporary output storage.
The phase threshold is lowered only in test instances to reach Φ⁴ quickly;
normal entropy and default phase parameters remain unchanged in the product.
The phase-3→4 transition reports the *new* level but emits the phase-3
grammar; the following cycle is the first to emit an actual Φ⁴ graph.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oe_phase_integration", ROOT / "origin_engine_monolith.py")
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe  # dataclasses look up the module during import
spec.loader.exec_module(oe)


class HighPhaseIntegration(unittest.TestCase):
    def check_evolved_graphs(self, fmt: str):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(oe.os, "getcwd", return_value=d):
                engine = oe.OriginEngine()
                engine.phase.graph_format = fmt

                # Move through *real* engine cycles, rather than directly
                # calling _dump_graph_file or constructing a fake sidecar.
                engine.phase.crit = 0.0
                expected_grammars = [
                    "grammar_phase0", "grammar_phase1",
                    "grammar_phase2", "grammar_phase3plus",
                ]
                for next_level, expected_grammar in enumerate(expected_grammars, 1):
                    emission = engine.cycle()
                    self.assertEqual(emission.meta["phase_level"], next_level)
                    self.assertEqual(emission.meta["grammar_mode"], expected_grammar)
                    self.assertTrue(emission.meta["phase_fired"])
                    self.assertNotIn("graph", emission.meta)

                # Hold at Φ⁴ to verify repeated true graph grammar emissions.
                engine.phase.crit = float("inf")
                outputs = oe.run_prompts(engine, [], cycles=3)

            paths = []
            for output in outputs:
                meta = output["meta"]
                self.assertEqual(meta["phase_level"], 4)
                self.assertFalse(meta["phase_fired"])
                self.assertEqual(meta["grammar_mode"], "grammar_phase4plus_with_graph")
                self.assertNotIn("graph_dump_error", meta)
                self.assertIn("graph", meta)
                self.assertIn("graph_file", meta)
                self.assertEqual(len(meta["graph"]["nodes"]), 5)
                path = Path(meta["graph_file"])
                self.assertEqual(path.parent, Path(d) / "oe_graphs")
                self.assertEqual(path.suffix, "." + fmt)
                self.assertTrue(path.is_file())
                if fmt == "json":
                    disk = json.loads(path.read_text(encoding="utf-8"))
                else:
                    disk = oe._parse_graphml(str(path))
                self.assertTrue(oe._graphs_equivalent(meta["graph"], disk))
                paths.append(path)

            self.assertEqual(len(set(paths)), 3)
            # Unlike scorecard's no-graph default, this must actually have
            # checked real, non-vacuous exports from evolving cycles.
            report = oe.test_graph_dump_consistency(outputs)
            self.assertEqual(report.details["checked"], 3)
            self.assertEqual(report.details["passed"], 3)
            self.assertEqual(report.score, 3.0)

    def test_real_evolution_generates_json_sidecars(self):
        self.check_evolved_graphs("json")

    def test_real_evolution_generates_graphml_sidecars(self):
        self.check_evolved_graphs("graphml")

    def test_report_counts_a_failed_high_phase_export(self):
        with tempfile.TemporaryDirectory() as d:
            engine = oe.OriginEngine()
            engine.phase.level = 4
            engine.phase.crit = float("inf")
            with patch.object(oe.os, "getcwd", return_value=d):
                with patch.object(oe, "_dump_graph_file", side_effect=OSError("simulated disk failure")):
                    emission = engine.cycle()

        self.assertIn("graph", emission.meta)
        self.assertNotIn("graph_file", emission.meta)
        self.assertIn("simulated disk failure", emission.meta["graph_dump_error"])
        report = oe.test_graph_dump_consistency([{"payload": emission.payload, "meta": emission.meta}])
        self.assertEqual(report.details["checked"], 1)
        self.assertEqual(report.details["passed"], 0)
        self.assertEqual(report.score, 0.0)
        self.assertIn("simulated disk failure", str(report.details["fail_examples"]))


if __name__ == "__main__":
    unittest.main()
