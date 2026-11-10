"""Direct cycle() prompt validation must not advance deterministic engine state.

These synthetic API tests do not touch user files or exercise graph output.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


class CyclePromptPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("origin_cycle_preflight", ENTRY)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        cls.module_name = spec.name

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.module_name, None)

    def test_rejected_direct_prompt_does_not_consume_seed_or_advance_state(self):
        for bad in ("", " \t ", 7, {"prompt": "synthetic"}, "\ud800", "\udc00"):
            with self.subTest(bad=repr(bad)):
                engine = self.module.OriginEngine(seed=918)
                fresh = self.module.OriginEngine(seed=918)
                with self.assertRaises(ValueError):
                    engine.cycle(prompt=bad)
                self.assertEqual(engine.history, [])
                self.assertIsNone(engine.iota)
                self.assertEqual(engine.phase, fresh.phase)
                self.assertEqual(engine._seed_rng.getstate(),
                                 fresh._seed_rng.getstate())
                # A rejected call must not poison a subsequent valid run.
                resumed = engine.cycle(prompt="synthetic amber")
                control = fresh.cycle(prompt="synthetic amber")
                self.assertEqual((resumed.payload, resumed.meta),
                                 (control.payload, control.meta))

    def test_valid_unicode_and_none_remain_seeded(self):
        for prompt in (None, "café 💜"):
            with self.subTest(prompt=repr(prompt)):
                left = self.module.OriginEngine(seed=119)
                right = self.module.OriginEngine(seed=119)
                a = left.cycle(prompt=prompt)
                b = right.cycle(prompt=prompt)
                self.assertEqual((a.payload, a.meta), (b.payload, b.meta))


if __name__ == "__main__":
    unittest.main()
