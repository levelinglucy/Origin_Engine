"""The dependency-bound scorecard must honor its public epsilon_max control.

The old score always used an overlap allowance of 0.05 and reported 0.02,
even when a caller explicitly passed a different bound. Synthetic records
avoid generation, personal data, graph files and external services.
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


class DependencyBoundThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("origin_dependency_threshold", ENTRY)
        cls.engine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.engine
        spec.loader.exec_module(cls.engine)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("origin_dependency_threshold", None)

    def test_custom_epsilon_changes_score_and_target_without_changing_default(self):
        # Only "abc" overlaps in the right-hand record's four unique trigrams:
        # overlap = 1/4 = 0.25. Historical default allowance = 0.05.
        outputs = [{"payload": "abcdefghij"}, {"payload": "abcQQQQQQ"}]
        default = self.engine.test_dependency_bound(outputs)
        explicit_default = self.engine.test_dependency_bound(outputs, epsilon_max=0.02)
        wider = self.engine.test_dependency_bound(outputs, epsilon_max=0.12)

        self.assertEqual(default, explicit_default)
        self.assertEqual(default.score, 2.4)
        self.assertEqual(default.details["target≈ε"], 0.02)
        self.assertEqual(default.details["overlap_max"], 0.25)
        self.assertEqual(wider.score, 3.0)
        self.assertEqual(wider.details["target≈ε"], 0.12)
        self.assertEqual(wider.details["overlap_max"], 0.25)

    def test_nonnumeric_or_impossible_bounds_are_rejected(self):
        outputs = [{"payload": "abc"}, {"payload": "abd"}]
        for invalid in (-0.01, 1.01, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    self.engine.test_dependency_bound(outputs, epsilon_max=invalid)


if __name__ == "__main__":
    unittest.main()
