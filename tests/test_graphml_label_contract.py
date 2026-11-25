"""Regression tests for the engine's own GraphML node-label contract."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import xml.etree.ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_graphml_labels", SCRIPT)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)

NS = "{http://graphml.graphdrawing.org/xmlns}"
GRAPH = {
    "nodes": [{"id": 0, "label": ""}, {"id": 1, "label": "alpha & β"}],
    "edges": [{"source": 0, "target": 1}],
}


class GraphMLLabelContractTests(unittest.TestCase):
    def export(self, temp):
        with mock.patch.object(oe.os, "getcwd", return_value=temp):
            return Path(oe._dump_graph_file(GRAPH, 4, "labels", "graphml"))

    def report(self, *paths):
        return oe.test_graph_dump_consistency([
            {"meta": {"phase_level": 4, "graph": GRAPH, "graph_file": str(path)}}
            for path in paths
        ])

    def alter(self, path, kind):
        tree = ET.parse(path)
        root = tree.getroot()
        key = root.find(NS + "key")
        nodes = root.find(NS + "graph").findall(NS + "node")
        data = nodes[1].find(NS + "data")
        if kind == "renamed":
            key.set("attr.name", "color")
        elif kind == "wrong_domain":
            key.set("for", "edge")
        elif kind == "wrong_type":
            key.set("attr.type", "int")
        elif kind == "missing_key":
            root.remove(key)
        elif kind == "duplicate_key":
            ET.SubElement(root, NS + "key", dict(key.attrib))
        elif kind == "missing_empty":
            nodes[0].remove(nodes[0].find(NS + "data"))
        elif kind == "duplicate_data":
            ET.SubElement(nodes[1], NS + "data", {"key": "label"}).text = "different"
        elif kind == "nested_data":
            ET.SubElement(data, "{urn:extension}extra").text = "not a scalar"
        else:
            raise AssertionError(kind)
        tree.write(path, encoding="utf-8")

    def test_refuses_lost_or_ambiguous_label_semantics(self):
        for kind in ("renamed", "wrong_domain", "wrong_type", "missing_key",
                     "duplicate_key", "missing_empty", "duplicate_data", "nested_data"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                path = self.export(temp)
                self.alter(path, kind)
                report = self.report(path)
                self.assertEqual(report.details["checked"], 1)
                self.assertEqual(report.details["passed"], 0)
                self.assertEqual(report.score, 0.0)

    def test_real_writer_preserves_empty_and_unicode_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.export(temp)
            self.assertEqual(oe._parse_graphml(path), GRAPH)
            self.assertEqual(self.report(path).score, 3.0)

    def test_mixed_files_keep_invalid_label_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            valid, altered = self.export(temp), self.export(temp)
            self.alter(altered, "renamed")
            report = self.report(valid, altered)
            self.assertEqual(report.details["checked"], 2)
            self.assertEqual(report.details["passed"], 1)
            self.assertEqual(report.score, 1.5)


if __name__ == "__main__":
    unittest.main()
