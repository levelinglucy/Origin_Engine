"""The graph scorecard must reject parallel edges in its simple-graph contract.

Synthetic in-memory graphs only. The generator emits each undirected pair
at most once, so this regression protects diagnostics without changing output.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_simple_edge_score", SOURCE)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


def connected_path():
    return {
        "nodes": [{"id": i, "label": f"v{i}"} for i in range(5)],
        "edges": [{"source": i, "target": i + 1} for i in range(4)],
    }


class SimpleEdgeScoreTests(unittest.TestCase):
    def test_rejects_repeated_undirected_pair_in_both_orders(self):
        for copy in ({"source": 0, "target": 1},
                     {"source": 1, "target": 0}):
            with self.subTest(duplicate=copy):
                graph = connected_path()
                graph["edges"].append(copy)
                ok, details = oe._graph_connectivity_ok(graph)
                self.assertFalse(ok, details)
                self.assertEqual(details["reason"], "duplicate_undirected_edge")

                report = oe.test_graph_output([
                    {"meta": {"phase_level": 4, "graph": graph}}
                ])
                self.assertEqual(report.details["checked"], 1)
                self.assertEqual(report.details["passed"], 0)
                self.assertEqual(report.score, 0)

    def test_legitimate_simple_connected_graph_still_passes(self):
        graph = connected_path()
        ok, details = oe._graph_connectivity_ok(graph)
        self.assertTrue(ok, details)
        report = oe.test_graph_output([
            {"meta": {"phase_level": 4, "graph": graph}}
        ])
        self.assertEqual(report.details["passed"], 1)
        self.assertEqual(report.score, 3.0)


if __name__ == "__main__":
    unittest.main()
