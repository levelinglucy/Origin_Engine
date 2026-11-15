"""Directed GraphML must not be credited as an undirected export.

Only synthetic graphs in temporary directories are read or written.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import xml.etree.ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_graphml_direction", SCRIPT)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)

NS = "{http://graphml.graphdrawing.org/xmlns}"
GRAPH = {
    "nodes": [{"id": 0, "label": "alpha"}, {"id": 1, "label": "beta"}],
    "edges": [{"source": 0, "target": 1}],
}


class GraphMLDirectionFidelityTests(unittest.TestCase):
    def report(self, path):
        return oe.test_graph_dump_consistency([
            {"meta": {"phase_level": 4, "graph": GRAPH, "graph_file": str(path)}}
        ])

    def export(self, temp):
        with mock.patch.object(oe.os, "getcwd", return_value=temp):
            return Path(oe._dump_graph_file(GRAPH, 4, "direction-test", "graphml"))

    def test_rejects_changed_missing_and_invalid_graph_defaults(self):
        for default in ("directed", None, "invalid"):
            with self.subTest(edgedefault=default), tempfile.TemporaryDirectory() as temp:
                path = self.export(temp)
                tree = ET.parse(path)
                graph = tree.getroot().find(NS + "graph")
                if default is None:
                    del graph.attrib["edgedefault"]
                else:
                    graph.set("edgedefault", default)
                tree.write(path, encoding="utf-8")
                report = self.report(path)
                self.assertEqual(report.details["checked"], 1)
                self.assertEqual(report.details["passed"], 0)
                self.assertEqual(report.score, 0.0)
                self.assertIn("undirected", report.details["fail_examples"][0]["reason"])

    def test_rejects_directed_or_invalid_edge_overrides(self):
        for directed in ("true", "1", "invalid", ""):
            with self.subTest(directed=directed), tempfile.TemporaryDirectory() as temp:
                path = self.export(temp)
                tree = ET.parse(path)
                tree.getroot().find(NS + "graph/" + NS + "edge").set("directed", directed)
                tree.write(path, encoding="utf-8")
                report = self.report(path)
                self.assertEqual(report.details["passed"], 0)
                self.assertEqual(report.score, 0.0)

    def test_preserves_export_and_explicit_undirected_edges(self):
        for directed in (None, "false", "0"):
            with self.subTest(directed=directed), tempfile.TemporaryDirectory() as temp:
                path = self.export(temp)
                if directed is not None:
                    tree = ET.parse(path)
                    tree.getroot().find(NS + "graph/" + NS + "edge").set("directed", directed)
                    tree.write(path, encoding="utf-8")
                self.assertEqual(oe._parse_graphml(path), GRAPH)
                report = self.report(path)
                self.assertEqual(report.details["passed"], 1)
                self.assertEqual(report.score, 3.0)

    def test_mixed_files_keep_failed_direction_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            valid = self.export(temp)
            changed = self.export(temp)
            tree = ET.parse(changed)
            tree.getroot().find(NS + "graph").set("edgedefault", "directed")
            tree.write(changed, encoding="utf-8")
            report = oe.test_graph_dump_consistency([
                {"meta": {"phase_level": 4, "graph": GRAPH, "graph_file": str(path)}}
                for path in (valid, changed)
            ])
            self.assertEqual(report.details["checked"], 2)
            self.assertEqual(report.details["passed"], 1)
            self.assertEqual(report.score, 1.5)


if __name__ == "__main__":
    unittest.main()
