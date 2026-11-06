# Origin Engine — developer notes

## Development summary

**Current stage:** unreleased entrypoint repair candidate for proposed `0.1.1`.  
**Why:** the README named `origin_engine_monolith.py`, but the repository provided only `python3 origin_engine_monolith.py`, so copying or invoking the advertised path failed.  
**What changed:** add an exact duplicate under the canonical path, preserve its predecessor; add CLI tests, CI, corrected quick start, roadmap and changelog.  
**Next:** verify CI; then investigate deterministic replay and export consistency separately.

[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Roadmap](./ROADMAP.md)

## Historical source and current behaviour

At the inspected baseline commit `f7d59d13f0d8b3a5f88d222e785156c7410ca37e`, the sole monolith was the file literally named `python3 origin_engine_monolith.py` (Git blob `8f59d6cf147be28403f8fe00d686f1904d989561`). The old README was also missing the closing Markdown code fence. This is a naming/documentation integration defect; we are **not claiming the numerical engine algorithm was broken**.

The monolith uses `os.urandom` and `random.SystemRandom` for stochastic operators, retains glyph history in memory during each invocation, and uses a phase accumulator to evolve the emitted grammar. The `--cycles` CLI drives the engine and prints the emissions and a toy scorecard. Later phases can produce graph files in the working directory. The source itself was not edited in this milestone. See the older documents in this repository for design context; they are preserved as historical material, not rewritten into retrospective claims.

## Decision 2026-11-06: exact copy versus rename or wrapper

**Options:** (a) rename the historical file; (b) introduce a wrapper which loads that oddly named file; (c) copy the exact blob to a canonical name.

**Choice:** (c). It restores a simple portable one-file entrypoint while leaving the historically named artifact untouched and the engine's behaviour unchanged. It avoids complexity or package import dependencies in an iPhone/Pythonista-oriented project.

**Trade-offs:** this duplicates ~29 KB of source; later updates could diverge. A test asserts the two tracked files remain byte-for-byte equal for this repair stage. Future independent changes should deliberately select one canonical source and preserve the other as a versioned historical artifact, rather than silently editing only one. This decision is reversible without rewriting history.

**Evidence:** filenames and baseline blob were inspected through GitHub. Runtime validation is delegated to the included isolated subprocess regression suite and GitHub Actions; passing smoke tests do not prove broader correctness or compatibility with every device.

**Revisit:** if the two copies need divergent evolution, formalise an archival path and migration guide; do not treat the archive as a second actively maintained source.

## Continuation

1. Confirm candidate CI and read back the canonical file's blob SHA. Only then consider this patch verified.
2. Investigate a deliberately opt-in seeded mode with a deterministic replay contract, while preserving the old default entropy source and recording incompatible output differences. Do not claim reproducibility before testing it.
3. Exercise higher phase graph output using bounded temporary directories and compare JSON/GraphML graph structures; check repeated-path collisions and XML escaping. Prioritise any confirmed data corruption over feature work.

**Compatibility:** Python 3.10+ syntax is currently used in type hints; Pythonista/device-specific execution has not been checked. External JSON/JSONL inputs should be treated as data, not instructions. Nothing here attests to artificial sentience.
