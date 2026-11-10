"""Verify prompt-pack CLI messaging matches current experimental behaviour.

No source files or personal data are read; subprocess runs use temporary input.
The legacy engine does not condition generation on prompt content.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


def run(*arguments, cwd):
    return subprocess.run(
        [sys.executable, str(ENTRY), *arguments],
        cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


class PromptPackHonestyTests(unittest.TestCase):
    def test_prompt_pack_warns_about_nonconditioning_without_changing_seeded_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({"prompt": "first synthetic input"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"prompt": "different synthetic input"}) + "\n", encoding="utf-8")
            args = ("--cycles", "2", "--seed", "73")
            original = run(*args, cwd=root)
            a = run(*args, "--prompts", str(first), cwd=root)
            b = run(*args, "--prompts", str(second), cwd=root)

        for result in (original, a, b):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(original.stdout, a.stdout)
        self.assertEqual(a.stdout, b.stdout)
        self.assertNotIn("ignored", original.stderr.lower())
        for result in (a, b):
            self.assertIn("--prompts", result.stderr)
            self.assertIn("not used to condition generation", result.stderr)


if __name__ == "__main__":
    unittest.main()
