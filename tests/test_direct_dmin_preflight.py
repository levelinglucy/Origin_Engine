"""Direct engine novelty targets must obey the same input contract as the CLI.

Synthetic API checks: do not write graphs, mutate source history or claim
cross-platform reproducibility. Invalid constructor arguments must fail before
an engine can enter a poisoned state.
"""
import importlib.util
import sys
import unittest
from pathlib import Path


ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_direct_dmin_preflight", ENTRY)
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


class DirectNoveltyTargetChecks(unittest.TestCase):
    def test_bad_direct_targets_are_rejected_at_construction(self):
        # NaN can otherwise skip the candidate loop and append a None
        # payload; infinity or a negative threshold silently changes
        # retries. Strictly refuse unsupported types too.
        for bad in (float("nan"), float("inf"), float("-inf"), -0.01,
                    True, None, "0.35", 10 ** 400):
            with self.subTest(d_min=repr(bad)):
                with self.assertRaisesRegex(ValueError, "d_min"):
                    oe.OriginEngine(d_min=bad, seed=219)

    def test_valid_direct_targets_replay_without_changing_defaults(self):
        for target in (0, 0.0, 0.35, 1.5):
            with self.subTest(d_min=target):
                first = oe.OriginEngine(d_min=target, seed=219)
                second = oe.OriginEngine(d_min=target, seed=219)
                one = first.cycle()
                two = second.cycle()
                self.assertIsInstance(one.payload, str)
                self.assertTrue(one.payload)
                self.assertEqual((one.payload, one.meta),
                                 (two.payload, two.meta))


if __name__ == "__main__":
    unittest.main()
