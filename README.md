# Origin Engine — Monolith Edition

## Development summary

**Status:** proposed `v0.1.1` integrated candidate plus graph-integrity fix under review; no stable release or new tag. **Current goal:** preserve every exported graph and round-trip valid JSON/GraphML labels.  
**Latest development (2026-11-06):** the canonical standalone engine now selects an exclusive unused filename instead of overwriting graphs with a matching identity/second, and uses XML-safe GraphML serialization. The preserved historical source is unchanged. **Candidate CI is pending.**  
**Next:** (1) verify graph integrity regressions on Python 3.11/3.12; (2) consider release validation on a trusted iPhone/Pythonista environment; (3) prototype optional seeded replay without changing default entropy.

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
