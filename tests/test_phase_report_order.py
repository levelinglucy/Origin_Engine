"""Reproducible phase-shift scorecard ordering under process hash randomization.

Only constructed synthetic outputs are used; no graph exports or user data.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"

CHILD = r'''
import importlib.util
import json
import sys
source = sys.argv[1]
spec = importlib.util.spec_from_file_location("oe_phase_report_order", source)
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)

def record(payload, grammar, fired=False):
    return {"payload": payload, "meta": {
        "phase_fired": fired, "grammar_mode": grammar,
        "target_d": 0.4, "alphabet_len": 25,
        "alphabet_hash": "synthetic", "alphabet_sample": "ABCDE"
    }}

outputs = [
    record("pre", "grammar_phase0"),
    record("transition", "grammar_phase1", True),
    record("after A", "grammar_phase3plus"),
    record("after B", "grammar_phase2"),
    record("after C", "grammar_phase4plus_with_graph"),
]
result = oe.test_phase_shift(outputs)
print(json.dumps(result.details["post_grammars"]))
'''


class PhaseReportOrderTests(unittest.TestCase):
    def test_seeded_report_orders_distinct_post_phase_grammars_canonically(self):
        """Hash seed must not permute an otherwise identical scorecard."""
        outputs = []
        for hash_seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            completed = subprocess.run(
                [sys.executable, "-c", CHILD, str(SOURCE)],
                capture_output=True, text=True, env=env, timeout=10, check=True,
            )
            outputs.append(json.loads(completed.stdout))
        expected = sorted([
            "grammar_phase3plus", "grammar_phase2",
            "grammar_phase4plus_with_graph",
        ])
        self.assertEqual(outputs, [expected, expected])


if __name__ == "__main__":
    unittest.main()
