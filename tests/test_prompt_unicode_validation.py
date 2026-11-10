"""Unpaired surrogate escapes must not start partly conditioned experiments."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


class PromptUnicodeValidationTests(unittest.TestCase):
    def test_cli_refuses_surrogate_before_any_emissions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "synthetic.jsonl"
            pack.write_text(
                json.dumps({"prompt": "valid first"}) + "\n"
                + json.dumps({"prompt": "\ud800"}) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(ENTRY), "--seed", "12", "--cycles", "4",
                 "--prompts", str(pack), "--condition-prompts"],
                cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--condition-prompts", result.stderr)
            self.assertIn("surrogate", result.stderr.lower())
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("=== Ψ Emissions ===", result.stdout)
            self.assertFalse((root / "oe_graphs").exists())

    def test_valid_non_bmp_unicode_still_conditions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "unicode.jsonl"
            pack.write_text(json.dumps({"prompt": "café 💜"}) + "\n",
                            encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ENTRY), "--seed", "12", "--cycles", "2",
                 "--prompts", str(pack), "--condition-prompts"],
                cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("=== Ψ Emissions ===", result.stdout)

    def test_api_validates_entire_pack_before_first_cycle(self):
        spec = importlib.util.spec_from_file_location("origin_prompt_unicode", ENTRY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            class Recorder:
                def __init__(self):
                    self.seen = []
                def cycle(self, prompt=None):
                    self.seen.append(prompt)
                    return mod.Construct(payload="synthetic", meta={})
            for surrogate in ("\ud800", "\udc00"):
                with self.subTest(surrogate=repr(surrogate)):
                    engine = Recorder()
                    with self.assertRaisesRegex(ValueError, "surrogate"):
                        mod.run_prompts(
                            engine, [{"prompt": "good"}, {"prompt": surrogate}],
                            cycles=3, condition_prompts=True,
                        )
                    self.assertEqual(engine.seen, [])
        finally:
            sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
