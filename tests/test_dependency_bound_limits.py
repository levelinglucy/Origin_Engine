"""Direct dependency-proxy threshold validation, without running generation."""
import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
spec = importlib.util.spec_from_file_location("oe_dependency_bound_limits", SOURCE)
assert spec is not None and spec.loader is not None
oe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oe
spec.loader.exec_module(oe)


class DependencyBoundLimits(unittest.TestCase):
    def test_out_of_range_integer_does_not_overflow_float_conversion(self):
        for target in (10**1000, -(10**1000)):
            with self.subTest(sign=target > 0):
                with self.assertRaisesRegex(ValueError, "finite number from 0 to 1"):
                    oe.test_dependency_bound([], epsilon_max=target)

    def test_existing_invalid_targets_remain_invalid(self):
        for target in (True, False, None, "0.02", -1, 2, -0.01, 1.01,
                       float("nan"), float("inf"), -float("inf")):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "finite number from 0 to 1"):
                    oe.test_dependency_bound([], epsilon_max=target)

    def test_valid_bounds_and_default_proxy_scores(self):
        outputs = [{"payload": "abcdef"}, {"payload": "abcdef"}]
        for target, expected in ((0, 0.0), (0.02, 0.15), (1, 3.0),
                                 (0.0, 0.0), (1.0, 3.0)):
            with self.subTest(target=target):
                result = oe.test_dependency_bound(outputs, epsilon_max=target)
                self.assertEqual(result.score, expected)
                self.assertEqual(result.details["overlap_max"], 1.0)
        self.assertEqual(oe.test_dependency_bound(outputs).score, 0.15)


if __name__ == "__main__":
    unittest.main()
