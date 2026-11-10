"""Bounded-memory phase-shift report scoring over synthetic histories.

No engine run, graph export, network, personal data or persistent file writes.
The predecessor materialized the Cartesian product of pre/post payloads.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tracemalloc
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_phase_streaming", SOURCE)
oe = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


def record(text, grammar, fired=False):
    return {"payload": text, "meta": {
        "phase_fired": fired, "grammar_mode": grammar,
        "target_d": 0.4, "alphabet_len": 25,
        "alphabet_hash": "synthetic", "alphabet_sample": "ABCDE",
    }}


class PhaseShiftStreamingTests(unittest.TestCase):
    def test_cross_divergence_scores_without_quadratic_pair_storage(self):
        # 500 x 500 comparison pairs were previously held in a list in RAM.
        # Construct the history before recording peak temporary allocations.
        outputs = (
            [record("before", "grammar0") for _ in range(500)]
            + [record("switch", "grammar1", fired=True)]
            + [record("after", "grammar2") for _ in range(500)]
        )
        # A plain lambda avoids mock call-history allocations in this stress test.
        with mock.patch.object(oe, "norm_dist", new=lambda a, b: 0.5):
            tracemalloc.start()
            try:
                result = oe.test_phase_shift(outputs)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(result.name, "Phase-Shift")
        self.assertEqual(result.details["cross_divergence"], 0.5)
        self.assertEqual(result.details["post_grammars"], ["grammar2"])
        self.assertLess(
            peak, 5 * 1024 * 1024,
            "Phase-shift scoring must not store all Cartesian comparison pairs",
        )


if __name__ == "__main__":
    unittest.main()
