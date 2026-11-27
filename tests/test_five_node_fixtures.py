"""Exhaustive labelled five-node fixtures for the flat graph scorecards.

Connectivity and export fidelity are different properties: even a disconnected
graph must survive a faithful export, while an altered edge must fail fidelity.
All fixtures are synthetic; no generator sampling distribution is assumed.
"""
import importlib.util
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_five_node_fixtures", SOURCE)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)

PAIRS = tuple(itertools.combinations(range(5), 2))
LABELS = ("", "a & b", "Ω", "line\rreturn", "🦊")


def fixture(mask):
    return {
        "nodes": [{"id": i, "label": label} for i, label in enumerate(LABELS)],
        "edges": [{"source": a, "target": b}
                  for bit, (a, b) in enumerate(PAIRS) if mask & (1 << bit)],
    }


def connected_by_cuts(mask):
    # Independent oracle: every nontrivial partition must have an edge
    # crossing it. Anchor vertex 0 on one side to avoid complementary cuts.
    for side in range(1, 31, 2):
        if not any(mask & (1 << bit) and
                   bool(side & (1 << a)) != bool(side & (1 << b))
                   for bit, (a, b) in enumerate(PAIRS)):
            return False
    return True


def output(graph, path=None):
    meta = {"phase_level": 4, "graph": graph}
    if path is not None:
        meta["graph_file"] = str(path)
    return {"meta": meta}


class FiveNodeFixtures(unittest.TestCase):
    def test_all_labelled_topologies_against_independent_cut_oracle(self):
        connected = 0
        outputs = []
        for mask in range(1 << len(PAIRS)):
            graph = fixture(mask)
            expected = connected_by_cuts(mask)
            connected += expected
            outputs.append(output(graph))
            with self.subTest(mask=mask):
                actual, details = oe._graph_connectivity_ok(graph)
                self.assertEqual(actual, expected, details)
        self.assertEqual(connected, 728)
        report = oe.test_graph_output(outputs)
        self.assertEqual(report.details["checked"], 1024)
        self.assertEqual(report.details["passed"], connected)
        self.assertEqual(report.score, round(3 * connected / 1024, 3))

    def test_both_exports_preserve_all_topologies_and_reject_one_edge_change(self):
        for fmt in ("json", "graphml"):
            # Only disposable files; unique identifiers avoid collision scans.
            with tempfile.TemporaryDirectory() as temp:
                with patch.object(oe.os, "getcwd", return_value=temp):
                    for mask in range(1 << len(PAIRS)):
                        with self.subTest(fmt=fmt, mask=mask):
                            graph = fixture(mask)
                            path = Path(oe._dump_graph_file(graph, 4, f"{mask:04x}", fmt))
                            self.assertTrue(path.is_file())
                            good = oe.test_graph_dump_consistency([output(graph, path)])
                            self.assertEqual((good.details["checked"], good.details["passed"], good.score),
                                             (1, 1, 3.0), good.details)
                            # Toggle one fixed edge in the expected sidecar;
                            # disk bytes are unchanged. This covers addition
                            # and removal across the complete labelled corpus.
                            bad = oe.test_graph_dump_consistency([output(fixture(mask ^ 1), path)])
                            self.assertEqual((bad.details["checked"], bad.details["passed"], bad.score),
                                             (1, 0, 0.0), bad.details)
                self.assertEqual(len(list((Path(temp) / "oe_graphs").iterdir())), 1024)


if __name__ == "__main__":
    unittest.main()
