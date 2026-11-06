# Origin Engine — Monolith Edition

## Development summary

**Status:** proposed `v0.1.1` integrated candidate; optional seeded replay is a separate development slice awaiting CI and review, not yet stable.
**Current goal:** make reproducible research runs possible without changing normal OS-backed randomness.
**Latest development (2026-11-06):** an isolated per-engine integer seed option now feeds P/U/C/Ψ; added checks for clean-process replay, independent streams and graph round trips. Verification pending on this branch.
**Next:** (1) verify PR CI and source readback; (2) review replay limitations before integration; (3) separately seek trusted Pythonista/device evidence before stable promotion.

[Changelog](./CHANGELOG.md) · [Developer notes and decisions](./DEV_NOTES.md) · [Milestones](./ROADMAP.md)

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

Optional flags: `--spec` (JSON spec), `--prompts` (JSONL), `--cycles`, `--dmin`, `--graph-format json|graphml`, and `--seed` (integer).

For repeatable research, run `python3 origin_engine_monolith.py --cycles 3 --seed 42`. Equal seeds and inputs reproduce glyphs and metrics on a matching implementation. Graph filenames embed timestamps and directory paths, so compare normalized graph records and actual graph contents rather than raw paths. Without `--seed`, the existing OS-backed randomness is preserved. **Seeded replay is not cryptographic and cross-version identical output is not guaranteed.** Later phases may write files to `oe_graphs/` relative to the **working directory**. Run in a disposable directory if you do not want output files in your working tree.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

This checks the historical source's Git blob identity, CLI help, bounded generation, invalid-spec rejection, deterministic export collisions, JSON preservation and GraphML label round trips. The new seeded replay tests check independent subprocess runs, per-engine isolation and normalized Φ⁴ JSON/GraphML graph outputs; they do not establish cross-platform byte identity or device-specific Pythonista execution. The bundled scorecard is informative, not an exhaustive quality or safety test.

See [developer notes](./DEV_NOTES.md) for historical context, trade-offs and [ROADMAP.md](./ROADMAP.md) for the release gate.
