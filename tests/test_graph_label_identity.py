"""A repeated generated glyph label must not collapse distinct graph vertices.

Uses a deterministic synthetic RNG and a mocked output path; no user files or
graphs are written. The graph still has five independent node identities when
their textual labels happen to coincide.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_repeated_graph_labels", SOURCE)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


class RepeatedTokenRng:
    """Always produce the same token and accept every proposed edge."""

    def randint(self, lower, upper):
        return lower

    def random(self):
        return 0.0


class GraphLabelIdentityTests(unittest.TestCase):
    def test_repeated_labels_keep_distinct_edge_endpoints(self):
        phase = oe.Phase(level=4, crit=float("inf"))
        iota = oe.Iota(ident="0123456789abcdef", orbit=3, entropy_mark="cafebabe")
        with patch.object(oe, "_dump_graph_file", return_value="synthetic.json"):
            emitted = oe.Ψ(iota, [], phase, rng=RepeatedTokenRng())

        graph = emitted.meta["graph"]
        self.assertEqual(len(graph["nodes"]), 5)
        self.assertEqual(len({n["label"] for n in graph["nodes"]}), 1)
        self.assertEqual(
            graph["edges"],
            [{"source": i, "target": j}
             for i in range(5) for j in range(i + 1, 5)],
        )
        okay, details = oe._graph_connectivity_ok(graph)
        self.assertTrue(okay, details)


if __name__ == "__main__":
    unittest.main()
