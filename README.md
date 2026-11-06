# Origin Engine — Monolith Edition

## Development summary

**Status:** entrypoint repair candidate; not tagged or released. **Current goal:** a working, understandable, self-contained Python launch path.  
**Latest change (2026-11-06):** restored the documented `origin_engine_monolith.py` filename as an exact source copy, leaving the historically named file intact. Added a bounded CLI regression suite and automated checks.  
**Next:** (1) verify candidate CI on supported Python versions; (2) assess deterministic, seeded replay as a separate experiment; (3) validate JSON/GraphML export under controlled fixtures.  
**Version policy:** proposed `0.1.1` patch after verification; no existing version is represented as stable by this update.

[Changelog](./CHANGELOG.md) · [Developer notes and decision](./DEV_NOTES.md) · [Milestones](./ROADMAP.md)

## What it does

Origin Engine is a **novelty-oriented experimental generator**, using six operators (`P → S → U → Σ → C → Ψ`) to produce symbolic glyph sequences. Its phase accumulator can change grammars and introduce additional glyphs, and later phases can write graph representations. The metrics are experimental proxies, **not proof of consciousness, originality independent of inherited code, or safety**.

The historical monolithic source remains in this repository under the filename `python3 origin_engine_monolith.py`; that filename looks like a shell command but is actually a file name. The canonical `origin_engine_monolith.py` is an exact copy to make the documented command work without deleting the original.

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

This checks the historical/canonical source match, CLI help, bounded generation and invalid-spec rejection. It does not yet test seeded replay or comprehensive graph exports. The bundled scorecard is informative, not an exhaustive quality or safety test.

See [developer notes](./DEV_NOTES.md) for historical context, trade-offs and [ROADMAP.md](./ROADMAP.md) for the release gate.
