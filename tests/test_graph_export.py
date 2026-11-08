"""Graph export regression tests for identity collisions and valid XML.

Historical code is imported without editing the predecessor. Each test uses
temporary output directories; frozen timestamps make collisions deterministic.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass resolves module globals
    spec.loader.exec_module(module)
    return module


current = load_source(ROOT / "origin_engine_monolith.py", "oe_current_graph")
historical = load_source(ROOT / "python3 origin_engine_monolith.py", "oe_historical_graph")


class FrozenDateTime(datetime.datetime):
    @classmethod
    def utcnow(cls):
        return cls(2026, 11, 6, 9, 0, 0)


def sample(label):
    return {
        "nodes": [{"id": 0, "label": label}, {"id": 1, "label": "Unicode Ω🦊"}],
        "edges": [{"source": 0, "target": 1}],
    }


class GraphExportIntegrity(unittest.TestCase):
    def test_original_overwrite_reproduced_in_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(historical.os, "getcwd", return_value=temp):
                with patch.object(historical.datetime, "datetime", FrozenDateTime):
                    first = historical._dump_graph_file(sample("first"), 4, "same-ident", "json")
                    second = historical._dump_graph_file(sample("second"), 4, "same-ident", "json")
            self.assertEqual(first, second)
            self.assertEqual(json.loads(Path(first).read_text(encoding="utf-8")), sample("second"))

    def test_repeated_exports_preserve_all_json_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(current.os, "getcwd", return_value=temp):
                with patch.object(current.datetime, "datetime", FrozenDateTime):
                    paths = [current._dump_graph_file(sample(str(i)), 4, "same-ident", "json")
                             for i in range(3)]
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(paths[0].endswith("_20261106T090000Z.json"))
            self.assertTrue(paths[1].endswith("_20261106T090000Z_0001.json"))
            for i, path in enumerate(paths):
                self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8")), sample(str(i)))

    def test_legacy_graphml_corruption_reproduced(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(historical.os, "getcwd", return_value=temp):
                path = historical._dump_graph_file(sample('A&B <C> "quoted"'), 4, "abc", "graphml")
            with self.assertRaises(ET.ParseError):
                ET.parse(path)

    def test_graphml_escapes_labels_and_round_trips(self):
        originals = [sample('A&B <C> "quoted"'), sample("Ω🦊 > & <")]
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(current.os, "getcwd", return_value=temp):
                with patch.object(current.datetime, "datetime", FrozenDateTime):
                    paths = [current._dump_graph_file(g, 4, "same-ident", "graphml")
                             for g in originals]
            self.assertNotEqual(*paths)
            for expected, path in zip(originals, paths):
                tree = ET.parse(path)
                namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
                self.assertIsNotNone(tree.getroot().find("g:key[@id='label']", namespace))
                self.assertTrue(current._graphs_equivalent(expected, current._parse_graphml(path)))
            outputs = [{"meta": {"phase_level": 4, "graph": g, "graph_file": p}}
                       for g, p in zip(originals, paths)]
            score = current.test_graph_dump_consistency(outputs)
            self.assertEqual(score.details["checked"], 2)
            self.assertEqual(score.details["passed"], 2)

    def test_invalid_format_does_not_create_a_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(current.os, "getcwd", return_value=temp):
                with self.assertRaisesRegex(ValueError, "Unsupported graph format"):
                    current._dump_graph_file(sample("x"), 4, "same-ident", "txt")
            self.assertFalse((Path(temp) / "oe_graphs").exists())


    def test_graph_is_complete_before_first_destination_becomes_visible(self):
        # A final filename must not be claimed until its JSON is fully written.
        graph = sample("ready")
        with tempfile.TemporaryDirectory() as temp:
            real_link = current.os.link
            observed = []

            def inspect_publish(staged, final):
                self.assertFalse(Path(final).exists())
                self.assertEqual(json.loads(Path(staged).read_text(encoding="utf-8")), graph)
                observed.append((staged, final))
                return real_link(staged, final)

            with patch.object(current.os, "getcwd", return_value=temp):
                with patch.object(current.os, "link", side_effect=inspect_publish):
                    path = current._dump_graph_file(graph, 4, "ident", "json")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][1], path)
            self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8")), graph)
            self.assertEqual(list((Path(temp) / "oe_graphs").glob(".oe_graph_stage_*")), [])

    def test_publication_failure_does_not_claim_or_corrupt_final_name(self):
        # A failed hard-link publication must leave prior exports intact.
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(current.os, "getcwd", return_value=temp):
                with patch.object(current.datetime, "datetime", FrozenDateTime):
                    first = current._dump_graph_file(sample("first"), 4, "same-ident", "json")
                    with patch.object(current.os, "link",
                                      side_effect=OSError("simulated publication failure")):
                        with self.assertRaisesRegex(OSError, "simulated publication failure"):
                            current._dump_graph_file(sample("second"), 4, "same-ident", "json")
            directory = Path(temp) / "oe_graphs"
            self.assertEqual(json.loads(Path(first).read_text(encoding="utf-8")), sample("first"))
            self.assertEqual(sorted(path.name for path in directory.iterdir()),
                             [Path(first).name])


if __name__ == "__main__":
    unittest.main()
