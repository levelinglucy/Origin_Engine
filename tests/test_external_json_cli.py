"""CLI errors for corrupt/missing synthetic JSON inputs.

Fixtures and subprocesses remain in disposable temporary directories.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"
PIPELINE = ["P", "S", "U", "Σ", "C", "Ψ"]


def invoke(root, *args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *args, "--cycles", "1", "--seed", "19"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


class ExternalJSONInputErrors(unittest.TestCase):
    def assert_rejected_before_generation(self, run, root, option):
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn(option, run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        self.assertNotIn("=== Ψ Emissions ===", run.stdout)
        self.assertFalse((root / "oe_graphs").exists())

    def test_invalid_spec_inputs_are_cli_errors(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "spec.json"
            cases = [
                ("missing", None),
                ("invalid JSON", '{"pipeline":'),
                ("wrong root shape", "[]"),
                ("invalid UTF-8", b"\xff"),
            ]
            for label, payload in cases:
                with self.subTest(label=label):
                    if path.exists():
                        path.unlink()
                    if isinstance(payload, bytes):
                        path.write_bytes(payload)
                    elif payload is not None:
                        path.write_text(payload, encoding="utf-8")
                    run = invoke(root, "--spec", str(path))
                    self.assert_rejected_before_generation(run, root, "--spec")

    def test_invalid_prompt_pack_inputs_are_cli_errors(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "prompts.jsonl"
            cases = [
                ("missing", None),
                ("malformed second record", '{"prompt":"ok"}\n{"prompt":\n'),
                ("invalid UTF-8", b"\xff"),
            ]
            for label, payload in cases:
                with self.subTest(label=label):
                    if path.exists():
                        path.unlink()
                    if isinstance(payload, bytes):
                        path.write_bytes(payload)
                    elif payload is not None:
                        path.write_text(payload, encoding="utf-8")
                    run = invoke(root, "--prompts", str(path), "--condition-prompts")
                    self.assert_rejected_before_generation(run, root, "--prompts")

    def test_valid_external_json_remains_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = root / "spec.json"
            prompts = root / "prompts.jsonl"
            spec.write_text(json.dumps({"pipeline": PIPELINE}), encoding="utf-8")
            prompts.write_text(json.dumps({"prompt": "synthetic amber"}) + "\n", encoding="utf-8")
            run = invoke(root, "--spec", str(spec), "--prompts", str(prompts),
                         "--condition-prompts")
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("=== Ψ Emissions ===", run.stdout)


if __name__ == "__main__":
    unittest.main()
