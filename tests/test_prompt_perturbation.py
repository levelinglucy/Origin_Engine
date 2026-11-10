"""Opt-in, hashed prompt perturbation must be reproducible and isolated.

Every test uses synthetic prompts in disposable directories. The default path
must remain unchanged so older seeded experiments are directly comparable.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "origin_engine_monolith.py"


def invoke(*arguments, cwd):
    return subprocess.run(
        [sys.executable, str(ENTRY), *arguments],
        cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


class PromptPerturbationTests(unittest.TestCase):
    def test_opt_in_changes_seeded_glyphs_without_changing_legacy_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({"prompt": "amber solar cells"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"prompt": "violet underwater caves"}) + "\n", encoding="utf-8")
            base = ("--cycles", "2", "--seed", "73")
            control = invoke(*base, cwd=root)
            old = invoke(*base, "--prompts", str(first), cwd=root)
            a = invoke(*base, "--prompts", str(first), "--condition-prompts", cwd=root)
            replay = invoke(*base, "--prompts", str(first), "--condition-prompts", cwd=root)
            b = invoke(*base, "--prompts", str(second), "--condition-prompts", cwd=root)

        for result in (control, old, a, replay, b):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(control.stdout, old.stdout)
        self.assertEqual(a.stdout, replay.stdout)
        self.assertNotEqual(a.stdout, b.stdout)
        self.assertNotEqual(a.stdout, control.stdout)
        self.assertIn("not used to condition", old.stderr)
        self.assertNotIn("not used to condition", a.stderr)

    def test_opt_in_requires_a_nonempty_pack_of_prompt_strings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing = invoke("--cycles", "1", "--condition-prompts", cwd=root)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("--prompts", missing.stderr)
            for invalid in ([], [{"prompt": ""}], [{"prompt": 4}],
                            [{"not_prompt": "text"}], ["bare string"]):
                with self.subTest(invalid=invalid):
                    pack = root / "invalid.jsonl"
                    pack.write_text("".join(json.dumps(value) + "\n" for value in invalid),
                                    encoding="utf-8")
                    result = invoke("--cycles", "1", "--condition-prompts",
                                    "--prompts", str(pack), cwd=root)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertNotIn("=== Ψ Emissions ===", result.stdout)
                    self.assertFalse((root / "oe_graphs").exists())

    def test_prompt_round_robin_is_explicit_and_nonmutating(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("origin_prompt_module", ENTRY)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            class Recorder:
                def __init__(self):
                    self.seen = []
                def cycle(self, prompt=None):
                    self.seen.append(prompt)
                    return module.Construct(payload="synthetic", meta={})
            recorder = Recorder()
            pack = [{"prompt": "alpha"}, {"prompt": "beta"}]
            result = module.run_prompts(recorder, pack, cycles=5,
                                        condition_prompts=True)
            self.assertEqual(recorder.seen, ["alpha", "beta", "alpha", "beta", "alpha"])
            self.assertEqual(len(result), 5)
            self.assertEqual(pack, [{"prompt": "alpha"}, {"prompt": "beta"}])
        finally:
            sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
