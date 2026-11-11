"""External JSON may not silently discard conflicting keys or accept NaN.

Tests use only synthetic input and disposable working directories.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
PIPELINE = ["P", "S", "U", "Σ", "C", "Ψ"]


def run_cli(root, *args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *args, "--cycles", "1", "--seed", "19"],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


class AmbiguousExternalJSON(unittest.TestCase):
    def assert_refused(self, run, root, option):
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn(option, run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        self.assertNotIn("=== Ψ Emissions ===", run.stdout)
        self.assertFalse((root / "oe_graphs").exists())

    def test_spec_refuses_ambiguous_or_nonstandard_json(self):
        pipeline = json.dumps(PIPELINE, ensure_ascii=False)
        cases = (
            ("shadowed pipeline", '{"pipeline":["X"],"pipeline":' + pipeline + '}'),
            ("nested duplicate", '{"pipeline":' + pipeline +
             ',"axioms":{"claim":"first","claim":"second"}}'),
            ("nonfinite constant", '{"pipeline":' + pipeline +
             ',"axioms":{"claim":NaN}}'),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            for label, data in cases:
                with self.subTest(label=label):
                    spec.write_text(data, encoding="utf-8")
                    self.assert_refused(run_cli(root, "--spec", str(spec)), root, "--spec")

    def test_prompt_pack_refuses_ambiguous_or_nonstandard_json(self):
        cases = (
            ("shadowed prompt", '{"prompt":"first","prompt":"second"}\n'),
            ("nested duplicate", '{"prompt":"ok","meta":{"a":1,"a":2}}\n'),
            ("nonfinite constant", '{"prompt":"ok","metadata":Infinity}\n'),
            ("nonfinite later entry", '{"prompt":"ok"}\n{"prompt":"next","x":-Infinity}\n'),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            for label, data in cases:
                with self.subTest(label=label):
                    prompts.write_text(data, encoding="utf-8")
                    for conditioned in (False, True):
                        args = ["--prompts", str(prompts)]
                        if conditioned:
                            args.append("--condition-prompts")
                        self.assert_refused(run_cli(root, *args), root, "--prompts")


if __name__ == "__main__":
    unittest.main()
