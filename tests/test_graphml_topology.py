"""Regression tests for topology lost by partial GraphML readback."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import xml.etree.ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_graphml_topology", SCRIPT)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)

NS = "{http://graphml.graphdrawing.org/xmlns}"
GRAPH = {
    "nodes": [{"id": 0, "label": "alpha"}, {"id": 1, "label": "beta"}],
    "edges": [{"source": 0, "target": 1}],
}


class GraphMLTopologyTests(unittest.TestCase):
    def export(self, temp):
        with mock.patch.object(oe.os, "getcwd", return_value=temp):
            return Path(oe._dump_graph_file(GRAPH, 4, "topology", "graphml"))

    def report(self, *paths):
        return oe.test_graph_dump_consistency([
            {"meta": {"phase_level": 4, "graph": GRAPH, "graph_file": str(path)}}
            for path in paths
        ])

    def alter(self, path, kind):
        tree = ET.parse(path)
        root = tree.getroot()
        graph = root.find(NS + "graph")
        node = graph.find(NS + "node")
        if kind == "second":
            other = ET.SubElement(root, NS + "graph",
                                  {"id": "other", "edgedefault": "undirected"})
            ET.SubElement(other, NS + "node", {"id": "n2"})
        elif kind == "nested":
            other = ET.SubElement(node, NS + "graph",
                                  {"id": "nested", "edgedefault": "undirected"})
            ET.SubElement(other, NS + "node", {"id": "n2"})
        elif kind == "hyperedge":
            edge = ET.SubElement(graph, NS + "hyperedge")
            for n in ("n0", "n1"):
                ET.SubElement(edge, NS + "endpoint", {"node": n})
        elif kind == "port":
            ET.SubElement(node, NS + "port", {"name": "p"})
            graph.find(NS + "edge").set("sourceport", "p")
        elif kind == "wrong_root":
            root.tag = "wrapper"
        else:
            raise AssertionError(kind)
        tree.write(path, encoding="utf-8")

    def test_rejects_unrepresented_topology(self):
        for kind in ("second", "nested", "hyperedge", "port", "wrong_root"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                path = self.export(temp)
                self.alter(path, kind)
                report = self.report(path)
                self.assertEqual(report.details["checked"], 1)
                self.assertEqual(report.details["passed"], 0)
                self.assertEqual(report.score, 0.0)

    def test_valid_export_and_failed_graph_both_count(self):
        with tempfile.TemporaryDirectory() as temp:
            valid, altered = self.export(temp), self.export(temp)
            self.assertEqual(oe._parse_graphml(valid), GRAPH)
            self.alter(altered, "nested")
            report = self.report(valid, altered)
            self.assertEqual(report.details["checked"], 2)
            self.assertEqual(report.details["passed"], 1)
            self.assertEqual(report.score, 1.5)

    def test_unmodified_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.report(self.export(temp))
            self.assertEqual(report.details["passed"], 1)
            self.assertEqual(report.score, 3.0)


if __name__ == "__main__":
    unittest.main()
