# Origin Engine — Monolith Edition

## Development summary

**Status:** proposed `v0.1.1` integrated candidate; no stable tag or release. **Current goal:** verify genuine Φ⁴+ engine evolution and make failed graph writes visible in the consistency report.
**Latest development (2026-11-06):** the candidate now counts failed Φ⁴+ graph dumps as failures rather than silently omitting them from its score. A new integration suite drives actual engine cycles from Φ through Φ⁴ and reads back three JSON or GraphML outputs in isolated directories. **Candidate validation awaits CI**; earlier graph export checks passed on Python 3.11/3.12.
**Next:** (1) run and review integration CI; (2) pursue opt-in seeded replay only after the integrity gate; (3) verify Pythonista on a trusted device before claiming that platform is tested.

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

Optional flags: `--spec` (JSON spec), `--prompts` (JSONL), `--cycles`, `--dmin`, and `--graph-format json|graphml`. Later phases may write files to `oe_graphs/` relative to the **working directory**. Run in a disposable directory if you do not want output files in your working tree.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

This checks the historical source's Git blob identity, CLI help, bounded generation, invalid-spec rejection, deterministic export collisions, JSON preservation and GraphML label round trips. It does not check seeded replay or device-specific Pythonista execution. The bundled scorecard is informative, not an exhaustive quality or safety test.

See [developer notes](./DEV_NOTES.md) for historical context, trade-offs and [ROADMAP.md](./ROADMAP.md) for the release gate.
