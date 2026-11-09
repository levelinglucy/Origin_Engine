"""Graph scorecards must not hide changes to identifier or label types.

Only disposable, synthetic JSON graph files are used. The historical engine and
any user data are left untouched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_graph_types", SCRIPT)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


def graph():
    return {
        "nodes": [{"id": 0, "label": "5"}, {"id": 1, "label": "second"}],
        "edges": [{"source": 0, "target": 1}],
    }


class GraphTypeFidelityTests(unittest.TestCase):
    def test_json_scorecard_rejects_changed_scalar_types(self):
        original = graph()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "graph.json"
            tampered = [
                ("string node ID", {"nodes": [{"id": "0", "label": "5"}, original["nodes"][1]],
                                    "edges": original["edges"]}),
                ("boolean node ID", {"nodes": [{"id": False, "label": "5"}, original["nodes"][1]],
                                     "edges": original["edges"]}),
                ("numeric label", {"nodes": [{"id": 0, "label": 5}, original["nodes"][1]],
                                   "edges": original["edges"]}),
                ("string edge endpoint", {"nodes": original["nodes"],
                                          "edges": [{"source": "0", "target": 1}]}),
                ("boolean edge endpoint", {"nodes": original["nodes"],
                                           "edges": [{"source": False, "target": 1}]}),
            ]
            for description, changed in tampered:
                with self.subTest(description=description):
                    out.write_text(json.dumps(changed), encoding="utf-8")
                    report = oe.test_graph_dump_consistency([
                        {"meta": {"phase_level": 4, "graph": original,
                                  "graph_file": str(out)}}
                    ])
                    self.assertEqual(report.details["checked"], 1)
                    self.assertEqual(report.details["passed"], 0)
                    self.assertEqual(report.score, 0.0)

    def test_graph_structure_rejects_boolean_node_and_edge_ids(self):
        bad_node = {"nodes": [{"id": False, "label": "a"},
                              {"id": True, "label": "b"}],
                    "edges": [{"source": False, "target": True}]}
        bad_edge = graph()
        bad_edge["edges"] = [{"source": False, "target": True}]
        for candidate in (bad_node, bad_edge):
            with self.subTest(candidate=candidate):
                ok, _details = oe._graph_connectivity_ok(candidate)
                self.assertFalse(ok)

    def test_normal_integer_graph_remains_valid(self):
        original = graph()
        ok, _details = oe._graph_connectivity_ok(original)
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "graph.json"
            out.write_text(json.dumps(original), encoding="utf-8")
            report = oe.test_graph_dump_consistency([
                {"meta": {"phase_level": 4, "graph": original,
                          "graph_file": str(out)}}
            ])
            self.assertEqual(report.details["passed"], 1)
            self.assertEqual(report.score, 3.0)


if __name__ == "__main__":
    unittest.main()
