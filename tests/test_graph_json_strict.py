"""Strict JSON graph evidence regressions, using only disposable synthetic files.

A comparison must not certify ambiguous or nonstandard JSON as a faithful
copy even if Python's permissive decoder chooses the expected last value.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_strict_graph_json", SOURCE)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


def sample_graph():
    return {
        "nodes": [{"id": 0, "label": "original"}, {"id": 1, "label": "second"}],
        "edges": [{"source": 0, "target": 1}],
    }


class StrictGraphJSONReadback(unittest.TestCase):
    def test_ambiguous_json_and_nonfinite_tokens_never_receive_perfect_score(self):
        graph = sample_graph()
        valid = json.dumps(graph, ensure_ascii=False)
        cases = {
            "duplicate nested label": valid.replace(
                '"label": "original"', '"label": "hidden change", "label": "original"', 1
            ),
            "duplicate top-level nodes": '{"nodes": [], ' + valid[1:],
            "nonstandard NaN extra": valid[:-1] + ', "unchecked": NaN}',
            "nonstandard Infinity extra": valid[:-1] + ', "unchecked": Infinity}',
            "nonstandard negative Infinity": valid[:-1] + ', "unchecked": -Infinity}',
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph.json"
            for case, payload in cases.items():
                with self.subTest(case=case):
                    path.write_text(payload, encoding="utf-8")
                    report = oe.test_graph_dump_consistency([{
                        "meta": {"phase_level": 4, "graph": graph,
                                 "graph_file": str(path)}
                    }])
                    self.assertEqual(report.details["checked"], 1)
                    self.assertEqual(report.details["passed"], 0)
                    self.assertEqual(report.score, 0.0)
                    self.assertTrue(report.details["fail_examples"])

    def test_unambiguous_standard_json_still_scores_perfectly(self):
        graph = sample_graph()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph.json"
            path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            report = oe.test_graph_dump_consistency([{
                "meta": {"phase_level": 4, "graph": graph,
                         "graph_file": str(path)}
            }])
            self.assertEqual((report.details["checked"], report.details["passed"]), (1, 1))
            self.assertEqual(report.score, 3.0)


if __name__ == "__main__":
    unittest.main()
