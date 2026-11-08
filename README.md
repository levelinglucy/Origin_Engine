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

Optional flags: `--spec` (JSON spec), `--prompts` (JSONL), `--cycles`, `--dmin`, `--graph-format json|graphml`, and `--seed` (integer). CLI `--cycles` requires a positive integer and `--dmin` a finite, non-negative number (zero is allowed). Invalid arguments are refused before generation with exit status 2.

For repeatable research, run `python3 origin_engine_monolith.py --cycles 3 --seed 42`. Equal seeds and inputs reproduce glyphs and metrics on a matching implementation. Graph filenames embed timestamps and directory paths, so compare normalized graph records and actual graph contents rather than raw paths. Without `--seed`, the existing OS-backed randomness is preserved. **Seeded replay is not cryptographic and cross-version identical output is not guaranteed.** Later phases may write files to `oe_graphs/` relative to the **working directory**. Run in a disposable directory if you do not want output files in your working tree.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

This checks the historical source's Git blob identity, CLI help, bounded generation, invalid-spec rejection, deterministic export collisions, JSON preservation and GraphML label round trips. The new seeded replay tests check independent subprocess runs, per-engine isolation and normalized Φ⁴ JSON/GraphML graph outputs; they do not establish cross-platform byte identity or device-specific Pythonista execution. The bundled scorecard is informative, not an exhaustive quality or safety test.

See [developer notes](./DEV_NOTES.md) for historical context, trade-offs and [ROADMAP.md](./ROADMAP.md) for the release gate.
