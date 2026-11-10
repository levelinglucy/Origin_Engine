# Origin Engine — developer notes

## Development summary

**Goal:** reproducible symbolic generation with honest, complete graph exports and bounded-memory phase-shift evidence. Proposed v0.1.1 source candidate on Python 3.11/3.12; iPhone/Pythonista support unverified.

**Integration:** Strict graph JSON readback [PR #13](https://github.com/levelinglucy/Origin_Engine/pull/13) and seeded grammar ordering [PR #14](https://github.com/levelinglucy/Origin_Engine/pull/14) are merged. The pair-streaming change is separately proposed in [PR #15](https://github.com/levelinglucy/Origin_Engine/pull/15), **not merged until exact revision CI passes**.

**Validation:** Exact main `84d5776b` passed 33/33 tests on Python 3.11 and 3.12 ([CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42815671399)). Streaming test-first PR #15 revision `48cef85e` reproduced one memory-budget failure out of 34 tests on **each** Python version ([red CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42865335324)); its repaired candidate remains to be validated.

**Publication:** untagged, non-stable source candidate; no verified GitHub release. Hostile filesystem races, abrupt power failure, large-cycle runtime feasibility and iPhone/Pythonista remain unverified. **Next:** validate PR #15 after implementation, observe branch and PR gates, and merge only if current/allowed. Preserve the historical monolith and earlier complete exports.

[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Roadmap](./ROADMAP.md)

## Decision 2026-11-10: bound phase-shift scorecard's temporary memory

**Confirmed predecessor:** `test_phase_shift()` constructed a `list(itertools.product(pre_payloads, post_payloads))` only to compute its mean. For 500 pre/post records, the previous Python 3.11/3.12 code allocated about **15.3 MiB peak temporary memory (16,079,620 bytes on Python 3.11; 16,080,260 on 3.12)** under `tracemalloc`, exceeding the 5 MiB synthetic acceptance budget, even though all 33 prior tests passed. [Test-first CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42865335324) shows the failure on both versions; the harness uses an allocation-free distance stub and synthetic histories without writing graphs.

**Choice:** Use nested generator iteration for the same order of distance comparisons and divide by the product of the two nonzero input counts. No pair tuples are accumulated. The `norm_dist` implementation, exact pair order, empty-side result, score formula, grammar names, emitted sequences and graph files are unchanged. Alternatives included retaining a list with a documented cycle limit (unnecessary memory use) and sampling only some pairs (would change score semantics). 

**Acceptance:** The same synthetic 500×500 calculation returns `cross_divergence=0.5`, preserves sorted post-transition grammar names and stays below a 5 MiB peak measured allocation; all 34 synthetic and historical tests pass on exact PR head under Python 3.11/3.12. This removes **quadratic pair storage**, not quadratic distance-computation time or the memory held by the source history itself. Tracing is implementation-specific and the 5 MiB limit is a regression tripwire, not an application memory guarantee. Do not claim all large-cycle jobs are feasible or iPhone/Pythonista certified. [PR #15](https://github.com/levelinglucy/Origin_Engine/pull/15) owns exact validation/integration evidence.

## Decision 2026-11-10: stabilize phase-shift report ordering

**Confirmed predecessor:** `test_phase_shift` collected distinct post-transition grammar names in a set and returned `list(post_grammars)`. Identical synthetic phase traces, launched under process hash seeds 0 and 1, printed different name orders. The test-first case failed on both CPython 3.11/3.12 in [PR #14 red CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42815290741), while prior unrelated tests passed.

**Choice:** Sort only the presented `post_grammars` list. Its membership, phase-shift score, underlying generated emissions, graph output, random sampling and CLI flags are unchanged. Alternative hash-dependent set order has no useful semantics and needlessly complicates comparing seeded scorecards.

**Acceptance and limits:** A synthetic cross-process test asserts the exact stable, alphabetically ordered name list under distinct `PYTHONHASHSEED` values, and the entire existing suite must pass on supported CPython versions. Prior reports may differ only in this informational field's order. This does not claim that every printed field is canonical, certify seeded cross-version equivalence, or validate iPhone/Pythonista or durable writes. [PR #14](https://github.com/levelinglucy/Origin_Engine/pull/14) owns exact final candidate evidence.

## Decision 2026-11-10: refuse ambiguous saved graph JSON evidence

**Confirmed predecessor:** Python's default `json.load` silently takes the last value for repeated object keys, and accepts nonstandard `NaN` and `Infinity` constants. The graph readback comparison ignores irrelevant extra fields, so a file containing a contradictory duplicate label, shadowed top-level graph, or nonstandard constant could still earn a perfect graph consistency score. Test-first disposable JSON cases and exact red CI are recorded in [PR #13](https://github.com/levelinglucy/Origin_Engine/pull/13).

**Choice:** keep generated graph JSON unchanged, but use `object_pairs_hook` to reject duplicate keys at every nesting depth and `parse_constant` to reject nonstandard numeric tokens on readback. Errors count as failed saved evidence, not exceptions escaping the scorecard. Valid JSON and existing GraphML comparison still work; no historical source or user data are modified.

**Trade-offs:** previously tolerated ambiguous or invalid JSON must be regenerated from trusted source before it can pass verification. This does not reject unknown but otherwise valid extra graph attributes, authenticate authorship, constrain file sizes, prove atomic readback against concurrent changes, or certify iPhone/Pythonista execution. The graph export already refuses non-finite values during writing; this change closes the readback gap rather than claiming a general JSON validator.

## Decision 2026-11-10: keep graph vertex identity independent of glyph labels

**Confirmed predecessor:** In the Φ⁴ grammar, generated vertex labels were converted to edge endpoints with `nodes.index(label)`. When two different vertices produced the same text, both endpoints resolved to the first occurrence. This could create self-loops or isolated vertices even when the sampled index pairs were distinct. The synthetic test-first [PR #12](https://github.com/levelinglucy/Origin_Engine/pull/12) forces all five labels equal and selects every edge; the predecessor fails with repeated `(0,0)` endpoints on both Python 3.11/3.12.

**Choice:** keep integer vertex indices through edge selection and JSON/GraphML serialization; look up the labels by index only when formatting the readable glyph-pair text. This preserves the five distinct vertices, edge sampling order and random draw count, and changes no graph schema or historical monolith. Identical-label cases can yield different corrected edge topology and glyph-pair text stays visually ambiguous; the graph exports are the authoritative topology.

**Trade-offs and boundaries:** This protects topology when generated labels coincide, not scientific significance, uniqueness of labels, cryptographic graph provenance, iPhone/Pythonista compatibility or power-loss safety. Valid ordinary distinct-label cases retain their prior mapping. [PR #12](https://github.com/levelinglucy/Origin_Engine/pull/12) owns exact-revision test and integration evidence; the source remains an untagged candidate until a release is actually published.

## Decision 2026-11-09: compare graph evidence without lossy scalar coercion

**Confirmed predecessor:** graph readback normalized `id` and edge endpoints with `int(...)`, and labels with `str(...)`. Thus a JSON export altered from integer `0` to string `"0"`, or string label `"5"` to number `5`, could still receive a perfect match. The graph structure check also accepted Boolean node/edge IDs because Python treats `bool` as an integer subclass. [PR #11](https://github.com/levelinglucy/Origin_Engine/pull/11) contains test-first reproductions from disposable synthetic JSON and the canonical validation record.

**Choice:** check exact integer IDs/endpoints and string labels before comparing, while retaining node/edge order normalization. Badly typed structures become mismatches rather than exceptions or passing scores. The GraphML parser already maps its textual identifiers into integer graph IDs, so legitimate JSON/GraphML round trips remain comparable. Generated graphs, graph file formats and the historical monolith are unchanged.

**Alternatives and limits:** accepting convertible values hides evidence changes; comparing raw serialized JSON would falsely reject harmless order differences and GraphML's representation. This is a narrow typed-scalar check, not a full schema validator or tamper-proof signature; it does not check ignored extra fields or authenticate the provenance of a graph. Real device, race and power-loss behaviour remains unverified. Revisit if the graph schema intentionally gains additional typed fields.

## Decision 2026-11-09: score the chosen CLI novelty baseline

**Verified source-level mismatch:** `--dmin` already changed the engine's novelty loop, but the final `OE-Compliance` scorecard always compared emitted glyphs against the built-in default `0.35`. A custom CLI value could therefore be accepted yet produce a misleading printed `target`. The earlier `test_oe_compliance` API itself already accepted an explicit baseline; the mismatch was in `main()`.

**Choice:** pass the validated `args.dmin` into the compliance scorecard. Keep the default behaviour, the phase-incremented emission targets, other scoring functions, seed replay and historical monolith unchanged. A minimal subprocess regression covers `--dmin 0.0` and `0.8`. [PR #10](https://github.com/levelinglucy/Origin_Engine/pull/10) is the authoritative test-first, CI and integration record.

**Boundary:** the score remains a toy normalized edit-distance proxy, not a proof of compliance, creativity, consciousness or safety. This does not redefine `test_non_reference` or the deliberately separate dependency proxy. Revisit only if a larger scorecard semantics redesign is justified.

## Decision 2026-11-08: reject graphs that cannot be represented faithfully

**Confirmed predecessor behaviour:** `json.dumps` silently emitted `NaN` and `Infinity` for non-finite floats; those are not standard JSON numbers. ElementTree escaped `&` and `<` correctly but wrote forbidden XML 1.0 controls such as NUL into GraphML labels/IDs/endpoints. Literal carriage return in GraphML text parsed back as line feed. [PR #9 test-first CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42471435625) reproduced failures on Python 3.11/3.12 with disposable inputs. [W3C XML 1.0 character and line-ending rules](https://www.w3.org/TR/xml/#charsets).

**Choice:** refuse non-finite JSON scalars using `allow_nan=False`; validate XML 1.0 scalar codepoints in dynamic node/edge identifiers and labels before creating output directories; serialize literal carriage returns from XML text as `&#13;` references to preserve round-trip value. These checks happen before staging/publishing. The previous graph naming, complete-only publication, historical monolith and valid Unicode samples remain unchanged.

**Alternatives:** silently replace bad controls (data loss), allow malformed files (false evidence), or post-parse everything after publication (too late). Explicit refusal is safer than claiming a corrupted graph is valid. JSON is strict at serialization; GraphML checks the relevant scalar values but does not claim full GraphML schema validation (for example, arbitrary custom IDs may violate downstream schema constraints).

**Trade-offs/revisit:** previously emitted malformed exports may require regeneration from trusted source; this change does not repair historic files or validate unknown third-party extensions. JSON/GraphML encoding is checked on synthetic fixtures; do not infer device/Pythonista support or power-loss durability. [PR #9](https://github.com/levelinglucy/Origin_Engine/pull/9) owns exact-commit validation and integration evidence.

## Decision 2026-11-08: publish only fully written graph outputs

**Confirmed predecessor behaviour:** export used `open(..., "x")` on the final JSON/GraphML path before writing data. Exclusive mode prevented overwriting an existing graph, but a process termination or simultaneous reader could observe an empty/truncated file under a finished-looking filename. The test-first [PR #8](https://github.com/levelinglucy/Origin_Engine/pull/8) reproduces the absent publication barrier with two failures in Python 3.12 CI; the other supported version also failed.

**Choice:** serialize before filesystem side effects, write to a private same-directory `.oe_graph_stage_*.tmp` file and close it before publishing with `os.link(stage, final)` (atomic exclusive hard-link creation). Preserve the legacy filename and collision suffix strategy; clean up the staging path on ordinary success/failure. Filesystems disallowing hard links **fail closed** instead of producing partial final outputs. Staged files are owner-only by default; output permission is correspondingly more restrictive than the old umask-based creation. The historical filename/source stays untouched.

**Alternatives:** writing directly to `x` (incomplete final visible); `os.replace` (overwrites previous graphs); and creating the final exclusively before streaming bytes (same visibility problem). Hard links are the portable standard-library approach for same-filesystem atomic no-replace publication on supported systems, not a promise about every platform.

**Limits and revisit:** a hard process kill can leave a hidden staging file requiring manual cleanup; forced power loss, directory sync, hostile concurrent modifications and iPhone/Pythonista/hard-link availability are untested. No fsync durability guarantee is claimed. Revisit for a verified target filesystem lacking hard-link support, or if owner-only export permissions conflict with a supported workflow. Keep the previous stable source revision recoverable.

## Decision 2026-11-08: make invalid CLI requests fail before work

**Verified predecessor:** `--cycles 0` and negative values exited successfully after emitting one glyph, because `run_prompts` used `max(1, N)`. Non-finite `--dmin` values reached novelty and score calculations; `nan` could crash after printing emissions while `inf` could misleadingly exit successfully. This is a CLI correctness problem, not evidence of corruption of historical engine files.

**Choice:** reject non-positive CLI cycle counts and non-finite or negative CLI novelty thresholds with argparse exit status 2, before input loading or engine execution. Keep `--dmin 0` valid for experimentation; do not impose an arbitrary upper cycle cap or alter the programmatic `run_prompts` API in this patch. This does not solve memory/time costs of very large valid runs.

**Alternatives:** silently clamp invalid values to defaults (misleading), repair only error messages after computation (too late), or add a broad resource policy (requires separate evidence). Three small subprocess regressions cover failure and valid boundary behaviour. [PR #7](https://github.com/levelinglucy/Origin_Engine/pull/7) is the canonical test-first and integration record. The source-only change affects CLI requests; historical monolith, seed algorithm, graph exports, JSON formats and prior research outputs are untouched.

## Decision 2026-11-06: opt-in per-engine seeded replay

**Problem:** the canonical engine uses `os.urandom` for P/C and global `SystemRandom` for U/Ψ. `random.seed` cannot replay these OS-backed sources. Experimental comparisons therefore depended on irreproducible draws.

**Alternatives:** patch global entropy functions temporarily; replace module-level global RNG; or inject a private pseudorandom stream into the operators. Global replacement risks affecting unrelated engines and default randomness during interleaved runs.

**Choice:** `OriginEngine(seed=integer)` owns a single `random.Random` instance. P/C byte draws use its `getrandbits(8)`, U/Ψ use its numeric methods. `seed=None` still uses `os.urandom` and `SystemRandom` in the same operators as before. Existing positional operator calls remain compatible with optional keyword-only `rng`. The CLI exposes `--seed INTEGER`, defaulting to no seed. The historical monolith is unchanged.

**Trade-offs:** seeded draws are predictable and must never be used for cryptographic secrets. Repeatability depends on matching source revision, Python/random implementation, parameters and call order. Timestamped file paths and collision suffixes are intentionally not deterministic; graph content and metadata excluding file paths can be compared. A future source refactor may change draw order; durable cross-version replay would need a versioned PRNG and manifest contract. No globals are patched or seeded.

**Acceptance:** equal seeds reproduce output in independent subprocesses; different seeds diverge; interleaved engines are independent; unseeded mode still requests OS entropy; seeded Φ⁴ JSON/GraphML sidecars round-trip under normalized paths. 17/17 checks passed on Python 3.11 and 3.12 on exact final PR head `d18dbbde` and merged-main head `ab27ea30`; these results do not prove target-device support. Pythonista/device execution remains untested.

**Revisit:** if durable archival replay is required, define an explicit stream version, PRNG and canonical serialization protocol before making a cross-version guarantee.

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
