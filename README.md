# Origin Engine — Monolith Edition

## Development

The proposed `v0.1.1` remains an untagged, non-stable source candidate. See [developer notes for the current goal and validation](./DEV_NOTES.md), [the changelog](./CHANGELOG.md) and [the milestone roadmap](./ROADMAP.md).

## What it does

Origin Engine is a **novelty-oriented experimental generator**, using six operators (`P → S → U → Σ → C → Ψ`) to produce symbolic glyph sequences. Its phase accumulator can change grammars and introduce additional glyphs, and later phases can write graph representations. The metrics are experimental proxies, **not proof of consciousness, originality independent of inherited code, or safety**.

The historical monolithic source remains in this repository under the filename `python3 origin_engine_monolith.py`; that filename looks like a shell command but is actually a file name. The canonical `origin_engine_monolith.py` began as an exact copy to restore the documented command. It now includes isolated graph-export integrity changes; the historical file is preserved unchanged.

## Quick start

Requires **Python 3.10 or newer**, standard library only. From the repository root:

```bash
python3 origin_engine_monolith.py --cycles 3
```

Or save just `origin_engine_monolith.py` in a directory and run it there. Output includes glyph emissions followed by experimental score summaries. For options:

```bash
python3 origin_engine_monolith.py --help
```

Optional flags: `--spec` (JSON spec), `--prompts` (JSONL parsed only by default), `--condition-prompts` (opt-in exact-text perturbation), `--cycles`, `--dmin`, `--graph-format json|graphml`, and `--seed` (integer). CLI `--cycles` requires a positive integer and `--dmin` a finite, non-negative number (zero is allowed). Invalid arguments are refused before generation with exit status 2. The `OE-Compliance` scorecard uses the selected `--dmin` baseline; its edit-distance score is an exploratory proxy, not a safety or originality certification.

**Prompt-pack behaviour:** By default, `--prompts path.jsonl` parses but **ignores** prompt text and warns on stderr; `--cycles` (default 16) still controls the emission count. This preserves earlier experiments. A separate, explicitly experimental mode makes each cycle's **exact prompt bytes** influence its glyph alphabet through a SHA-256-derived, cycle-local state perturbation:

```bash
python3 origin_engine_monolith.py --prompts sample.jsonl --condition-prompts --cycles 3 --seed 42
```

Use JSONL objects such as `{"prompt": "synthetic amber seed"}`, one per line; at least one nonempty string prompt is required. Conditioned packs also refuse escaped unpaired surrogate code points *before any cycle*; ordinary Unicode (including emoji) is valid. Multiple prompts repeat in file order if cycles exceed entries. Identical seeds and files replay deterministically on a matching implementation; different text may change emissions. This is **not semantic language understanding** or proof of a scientific prompt effect. Do not use sensitive prompt text: output may reveal information about the chosen input. The opt-in mode does not change the default path, random draw sequence or the preserved historical monolith. See [the disclosure decision](./DEV_NOTES.md#decision-2026-11-10-disclose-unused-jsonl-prompt-content) and [the opt-in design](./DEV_NOTES.md#decision-2026-11-10-opt-in-exact-text-prompt-perturbation).

For repeatable research, run `python3 origin_engine_monolith.py --cycles 3 --seed 42`. Equal seeds and inputs reproduce glyphs and metrics on a matching implementation. Graph filenames embed timestamps and directory paths, so compare normalized graph records and actual graph contents rather than raw paths. Without `--seed`, the existing OS-backed randomness is preserved. **Seeded replay is not cryptographic and cross-version identical output is not guaranteed.** Later phases may write files to `oe_graphs/` relative to the **working directory**. Run in a disposable directory if you do not want output files in your working tree. Exports are written to private staging files and published with an exclusive hard link only after the complete JSON/GraphML payload is closed; systems without same-directory hard-link support report an export error rather than leaving a partial final graph. Abrupt termination can leave a hidden `.oe_graph_stage_*.tmp` file for manual inspection/cleanup. Published files are owner-only by default. Graph exports refuse XML 1.0 forbidden characters or non-finite JSON numbers instead of writing malformed files; GraphML preserves carriage returns in labels. This is **not** a power-loss durability guarantee, and support on iPhone/Pythonista is unverified.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

This checks the historical source's Git blob identity, CLI help, bounded generation, invalid-spec rejection, deterministic export collisions, JSON preservation and GraphML label round trips. The new seeded replay tests check independent subprocess runs, per-engine isolation and normalized Φ⁴ JSON/GraphML graph outputs; they do not establish cross-platform byte identity or device-specific Pythonista execution. The bundled scorecard is informative, not an exhaustive quality or safety test.

See [developer notes](./DEV_NOTES.md) for historical context, trade-offs and [ROADMAP.md](./ROADMAP.md) for the release gate.
