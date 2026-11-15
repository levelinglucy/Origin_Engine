"""Regression: explicit Python cycle counts must not silently emit a cycle.

The CLI already refuses nonpositive counts; this protects programmatic callers
before they consume a seeded draw or publish a graph. In-memory only.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_direct_cycle_counts", SOURCE)
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


class DirectCycleCountTests(unittest.TestCase):
    def test_invalid_explicit_counts_fail_without_engine_mutation(self):
        for invalid in (0, -1, -7, True, 1.5, "2"):
            with self.subTest(cycles=repr(invalid)):
                engine = oe.OriginEngine(seed=219)
                initial_rng = engine._seed_rng.getstate()
                with self.assertRaisesRegex(ValueError, "cycles"):
                    oe.run_prompts(engine, [], cycles=invalid)
                self.assertIsNone(engine.iota)
                self.assertEqual(engine.history, [])
                self.assertEqual(engine.phase.level, 0)
                self.assertEqual(engine._seed_rng.getstate(), initial_rng)

    def test_conditioned_invalid_count_cannot_advance_seeded_engine(self):
        for invalid in (0, -2):
            with self.subTest(cycles=invalid):
                engine = oe.OriginEngine(seed=219)
                initial_rng = engine._seed_rng.getstate()
                with self.assertRaisesRegex(ValueError, "cycles"):
                    oe.run_prompts(engine, [{"prompt": "valid synthetic input"}],
                                   cycles=invalid, condition_prompts=True)
                self.assertEqual(engine.history, [])
                self.assertIsNone(engine.iota)
                self.assertEqual(engine._seed_rng.getstate(), initial_rng)

    def test_positive_explicit_counts_preserve_seeded_replay(self):
        for conditioned in (False, True):
            with self.subTest(conditioned=conditioned):
                args = {"cycles": 2, "condition_prompts": conditioned}
                pack = [{"prompt": "valid synthetic input"}]
                a = oe.OriginEngine(seed=219)
                b = oe.OriginEngine(seed=219)
                first = oe.run_prompts(a, pack, **args)
                second = oe.run_prompts(b, pack, **args)
                self.assertEqual(len(first), 2)
                self.assertEqual(first, second)
                self.assertEqual(len(a.history), 2)


if __name__ == "__main__":
    unittest.main()
