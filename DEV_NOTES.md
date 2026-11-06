# Origin Engine — developer notes

## Development summary

**Current stage:** proposed `v0.1.1` integrated candidate; high-phase integration and missing-export accounting tested in [PR #4](https://github.com/levelinglucy/Origin_Engine/pull/4). Not tagged stable.
**Current goal:** prove the actual Φ⁴+ engine path produces faithful JSON/GraphML files and counts failures rather than treating them as absent graphs.
**Latest development:** real engine-cycle integration tests accelerate phase thresholds only in test instances; scoring now counts failed graph writes. Prior export-collision and XML fixes remain merged. **PR CI: 12/12 passed on Python 3.11 and 3.12.**
**Next:** verify merged-main checks, pursue optional seeded replay separately, and reassess release with trusted Pythonista/device evidence.

[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Roadmap](./ROADMAP.md)

## Historical source and current behaviour

At the inspected baseline commit `f7d59d13f0d8b3a5f88d222e785156c7410ca37e`, the sole monolith was the file literally named `python3 origin_engine_monolith.py` (Git blob `8f59d6cf147be28403f8fe00d686f1904d989561`). The old README was also missing the closing Markdown code fence. This is a naming/documentation integration defect; we are **not claiming the numerical engine algorithm was broken**.

The monolith uses `os.urandom` and `random.SystemRandom` for stochastic operators, retains glyph history in memory during each invocation, and uses a phase accumulator to evolve the emitted grammar. The `--cycles` CLI drives the engine and prints the emissions and a toy scorecard. Later phases can produce graph files in the working directory. The historical source remains untouched. After the entrypoint-only milestone, the canonical entrypoint was intentionally updated for the graph-export repair described below. See the older documents in this repository for design context; they are preserved as historical material, not rewritten into retrospective claims.

## Decision 2026-11-06: exact copy versus rename or wrapper

**Options:** (a) rename the historical file; (b) introduce a wrapper which loads that oddly named file; (c) copy the exact blob to a canonical name.

**Choice:** (c). It restores a simple portable one-file entrypoint while leaving the historically named artifact untouched and the engine's behaviour unchanged. It avoids complexity or package import dependencies in an iPhone/Pythonista-oriented project.

**Trade-offs:** this duplicates ~29 KB of source; later updates could diverge. The original entrypoint repair checked the two files were identical. Later work pins the historical Git blob digest instead, allowing the canonical source to evolve independently. Future independent changes should deliberately select one canonical source and preserve the other as a versioned historical artifact, rather than silently editing only one. This decision is reversible without rewriting history.

**Evidence:** filenames and baseline blob were inspected through GitHub. Runtime validation is delegated to the included isolated subprocess regression suite and GitHub Actions; passing smoke tests do not prove broader correctness or compatibility with every device.

**Revisit:** if the two copies need divergent evolution, formalise an archival path and migration guide; do not treat the archive as a second actively maintained source.

## Continuation

1. Graph-integrity [PR #2](https://github.com/levelinglucy/Origin_Engine/pull/2) merged at [`0b263516`](https://github.com/levelinglucy/Origin_Engine/commit/0b263516ff2fa5e87ebfcae5730b2bb0092a2ca6); [final PR CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42194431084) and [merged-main CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42194484337) passed on Python 3.11/3.12, and source readback was confirmed.
2. Assess target-device/Pythonista execution, graph export paths and stable release criteria on trusted data; keep release status explicit.
3. Separately prototype opt-in seeded replay without changing the default entropy source; establish reproducibility checks first.

**Compatibility:** Python 3.10+ syntax is currently used in type hints; Pythonista/device-specific execution has not been checked. External JSON/JSONL inputs should be treated as data, not instructions. Nothing here attests to artificial sentience.

## Decision 2026-11-06: prevent graph overwrite and invalid XML

**Confirmed problem in historical source:** `_dump_graph_file` selected a basename from phase, eight identity characters and a timestamp only precise to one second, then used mode `w`. Repeated emissions sharing those fields silently overwrote the earlier graph; the original output's path would then point to different data. GraphML labels were interpolated into raw XML: ampersands and angle brackets could make an exported graph unparseable. The existing graph consistency scorecard could also give misleading results if an earlier path was overwritten.

**Options:** use microsecond timestamps, random UUIDs, a process-local counter, or exclusive creation with collision suffixes. For XML, escape handwritten strings or use standard-library XML serialization.

**Choice:** construct JSON or GraphML text in memory, then exclusively create a file with `open(..., "x")`, retaining the original timestamped filename for the first emission and adding `_0001`, `_0002`, etc. if it exists. This is safe across repeated runs and independent processes without relying on timestamp resolution or an in-memory counter. `xml.etree.ElementTree` serializes elements and escapes labels, and a GraphML `key` declaration advertises node labels.

**Trade-offs:** new files are created instead of silently replacing old ones, so storage use grows with emitted graphs; a bounded 10,000-name suffix search raises if an implausibly crowded identity/second namespace is exhausted. The first filename remains compatible, and the parser accepts the XML namespace representation. A process terminated during a file write could leave an incomplete output; completed writes and normal errors are handled, but crash-atomic durable output is not claimed. Do not use untrusted file contents as instructions.

**Verification target:** tests reproduce both defects on the untouched historical source and assert lossless output on the new engine under a frozen clock. Exercise JSON and XML with Unicode and XML-special characters, check distinct paths and reuse the existing consistency scorecard. Nine tests passed on each of Python 3.11 and 3.12 in [PR CI run 42194305730](https://github.com/levelinglucy/Origin_Engine/actions/runs/42194305730), including the historical failure reproductions. This is not target-device validation.

**Revisit:** if crash-atomic delivery or very high concurrent throughput becomes a requirement, investigate temporary-file staging plus non-replacing atomic publication on supported filesystems. Do not add that complexity without an observed need.


## Decision 2026-11-06: count failed graph exports

**Problem:** the old Graph-Dump-Consistency check only considered records with both `graph` and `graph_file`. However `Ψ` deliberately preserves `graph` alongside `graph_dump_error` when a write fails. On an actual failing Φ⁴+ run, the scorecard could silently ignore failed writes and present a passing result (or say no graph was observed). Existing direct helper tests did not exercise genuine phase evolution.

**Alternatives:** treat every missing graph as an error, fail the whole engine on any write error, or count failures only when the engine actually produced a graph sidecar. The first choice would mark the expected phase-3→4 transition as a failure: metadata reports the *new* phase after the transition, while grammar and graph selection used the phase before the transition. The second would change existing CLI error-recovery behaviour.

**Choice:** keep `Ψ`'s best-effort emission behaviour, but count a produced graph without a successful file as a failed Graph-Dump-Consistency check. Count successful and failed graph-emission attempts in the same denominator; keep the existing no-graph fallback only for runs that generated no graph sidecars at all. Test actual full `OriginEngine.cycle()` transitions from phase 0 through phase 4, followed by repeated Φ⁴ emissions with the default engine's stochastic token generator. Use a test-only zero threshold for the four transitions and then an infinite threshold to hold at Φ⁴; neither the production default nor entropy source is altered.

**Trade-offs:** the numerical score may decrease for runs that previously hid failed exports. This is intentional and more informative, but old scorecards should not be compared without accounting for the changed denominator. Runtime still permits a glyph emission even if its graph file fails; clients requiring guaranteed durable graphs must explicitly check `graph_dump_error`/`graph_file` and consider crash-atomic designs separately. Random graph connectivity is a distinct quality metric: integration checks verify faithful serialization, not that all stochastic graphs are connected.

**Acceptance checks:** a real end-to-end evolution per format yields three distinct, parseable files matching the in-memory sidecars; the consistency report says checked=3/passed=3; a simulated disk failure reports checked=1/passed=0/score=0. **Validation:** 12/12 tests passed on each Python 3.11 and 3.12 in [PR CI 42202352772](https://github.com/levelinglucy/Origin_Engine/actions/runs/42202352772). Device/Pythonista execution and interrupted-write crash durability remain untested.

**Revisit:** require fail-fast or atomic handoff if consumers come to depend on guaranteed durable graph delivery; revise metric definitions only with explicit versioned compatibility notes.
