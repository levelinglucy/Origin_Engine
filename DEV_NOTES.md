# Origin Engine — developer notes

## Development summary

**Current goal:** [PR #29](https://github.com/levelinglucy/Origin_Engine/pull/29) preserves explicit string node-label semantics in GraphML export consistency checks.

**Integration:** Draft candidate; see PR #29 for live status. Predecessor main `02597a21` includes PR #28 topology fidelity. **Validation:** [Test-only hosted red evidence](https://github.com/levelinglucy/Origin_Engine/actions/runs/45721044452): nine failed assertions/subcases in 67 tests on each Python 3.11/3.12; actual writer empty/Unicode roundtrip passed. Candidate validation remains outstanding. **Release:** Unreleased.

**Decision (2026-11-26):** Require exactly one writer-style key declaration (`id=label`, `for=node`, `attr.name=label`, `attr.type=string`), no second applicable label declaration, and one explicit childless label data element per node. Empty elements remain empty strings; absent data is not an empty label. Other named attributes are not compared. The [GraphML primer §2.4](https://graphml.graphdrawing.org/primer/graphml-primer.html#s2.4) distinguishes key identifiers from attribute names. A generic importer could resolve alternate key IDs or defaults, but this is a verifier for our own writer, not a general importer. External equivalent files using other key IDs or implicit defaults are outside the supported contract. Preserve generation, writer, JSON, direction/topology checks and historical source.

**Next:** Inspect exact candidate hosted Python 3.11/3.12 results (including additional ambiguity and unrelated-attribute controls); verify branch and required checks before integration. No schema validation, arbitrary extension fidelity, security, device or durability certification.

### Previous GraphML topology decision

**Current goal:** [PR #28](https://github.com/levelinglucy/Origin_Engine/pull/28) prevents partial GraphML readback from awarding full export-consistency credit to documents containing topology the engine does not represent.

**Integration:** See PR #28 for current merge status; predecessor main `1592d9cd` includes direction-fidelity PR #27. **Validation:** [Hosted red/green evidence](https://github.com/levelinglucy/Origin_Engine/pull/28#issuecomment-7140516720) for the tested PR merge, including the real writer regressions. Final integration checks are tracked in the PR. **Release:** Unreleased.

**Decision (2026-11-24):** Require the GraphML root and exactly one flat graph, reject hyperedges, node ports and edge port attributes. The [GraphML primer](https://graphml.graphdrawing.org/primer/graphml-primer.html#s3) explains these richer graph models. A general importer can intentionally project them away, but this scorecard is checking fidelity of the engine's own flat undirected export, so silent omission is inappropriate. Preserve the writer, JSON reader, explicit undirected overrides, generation, and existing historical source. This is not a complete XML/GraphML schema validator or security boundary; arbitrary extension data and other malformed cases are outside this narrow change.

**Next:** Confirm the final PR head and required checks before integration, then verify merged-main validation. No device, release, external graph authenticity or durability claim.

### Historical development summaries

**Current reviewable candidate:** [PR #26](https://github.com/levelinglucy/Origin_Engine/pull/26) corrects the direct Python `run_prompts(..., cycles=...)` API: an explicitly zero or negative count previously generated *one* emission and changed seeded state. A non-integer or Boolean count is now rejected with `ValueError` before any engine cycle. Valid positive explicit counts and the historic `cycles=None` default are unchanged. The CLI already rejected nonpositive `--cycles`.

**Integration:** Draft, unmerged; this proposal must remain separate from the previously merged direct-`d_min` [PR #25](https://github.com/levelinglucy/Origin_Engine/pull/25). **Validation:** test-first regression and a narrow source guard were committed, but no new hosted test result is claimed here. **Release:** no tag, device test or crash-durability claim. **Next:** inspect exact PR-head Python 3.11/3.12 tests, then consider low-risk integration only after reviewing required checks.

**Recent correctness repair:** [PR #25](https://github.com/levelinglucy/Origin_Engine/pull/25) guards the direct Python API's novelty target. `OriginEngine(d_min=NaN)` could skip all Ψ candidates, append `None` to history and make later calls fail. The constructor now refuses invalid targets before allocating engine state, without changing valid seeded streams, CLI flags, graph formats or the historical source.

**Integration:** PR #25 is the canonical change record; check its current merge status before claiming integration. [PR #24](https://github.com/levelinglucy/Origin_Engine/pull/24) was merged into main at `b45ecd5b`; its simple-undirected edge validation is independent. [PR #23](https://github.com/levelinglucy/Origin_Engine/pull/23) remains an unmerged research draft without generator changes.

**Validation:** test-first `d909c744` failed eight expected invalid-target subcases out of 54 tests per Python 3.11/3.12 in [red CI 43647873667](https://github.com/levelinglucy/Origin_Engine/actions/runs/43647873667); valid-target replay controls passed. Source-and-test revision `8d4d840` and documented candidate `a0f9d628` passed **54/54 per Python 3.11 and 3.12** ([final pre-merge CI 43647966918](https://github.com/levelinglucy/Origin_Engine/actions/runs/43647966918)); both job logs inspected. Candidate validation is not merged-main validation. **Release:** no verified new tag, client/device or power-loss certification.

**Next:** inspect PR #25's live status and exact main tests if integrated. Keep PR #23's connectivity research separate, and do not force graph connectivity without a use case.

**Limits:** Direct construction argument validation is not a global state transaction. Later mutation of `engine.d_min`, callback failures and filesystem crashes remain outside this fix. Scorecards are structural and experimental proxies, not scientific guarantees.

[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Roadmap](./ROADMAP.md)

## Decision 2026-11-15: preserve GraphML direction evidence

**Confirmed defect:** The GraphML readback helper ignored both graph `edgedefault` and per-edge `directed` attributes. A synthetic export changed from undirected to directed still earned a perfect Graph-Dump-Consistency score against the original undirected sidecar. Before the fix, the new four-test regression had eight failing assertions/subcases, including a mixed valid/altered pair incorrectly counted as two passes.

**Choice:** Require the export contract's explicit `edgedefault="undirected"` and reject directed or invalid edge overrides. Keep absent overrides and explicit `false`/`0` undirected booleans. The [GraphML primer](https://graphml.graphdrawing.org/primer/graphml-primer.html#s2.3) explains graph defaults and edge overrides; the [schema](https://graphml.graphdrawing.org/specification/element_edge.xsd.htm) declares the latter as XML Schema booleans. Do not erase contradictory direction evidence before comparison.

**Scope and trade-offs:** This is a reader for this engine's own exports, not a complete GraphML schema validator or general mixed-graph importer. A directed-default external document is refused even if all its edges explicitly override to undirected. Generator, seed stream, graph writer, JSON readback and historical monolith remain unchanged. Tests use synthetic temporary files, including actual exporter round trips, explicit undirected controls and mixed-file scoring. Local Python 3.12.14 passed all 61 tests; hosted results belong in the draft PR. No integration, release, iPhone or crash-durability claim.

## Decision 2026-11-15: validate explicit direct API cycle counts

**Confirmed source defect:** The CLI rejects zero/negative `--cycles`, but `run_prompts` computed `N` then called `range(max(1, N))`. A direct Python call requesting `cycles=0` or a negative integer therefore generated an unwanted emission and advanced the engine's Iota, PRNG and history. `cycles=True` also generated one emission. This is a source-level reproducible contract mismatch, not an observed external data-loss incident.

**Narrow repair:** If `cycles is not None`, require an exact built-in positive integer and raise `ValueError` before prompt-pack preflight or mutation. Leave implicit `cycles=None` historical behaviour alone, including its empty-pack fallback. Valid explicit counts, generation, graph format, seeded random stream and the historical monolith are untouched. The new synthetic regression tests invalid counts for state preservation and valid conditioned/unconditioned seeded replay.

**Trade-off / verification gate:** Direct callers using an explicit nonpositive or Boolean count for a legacy one-cycle fallback will now get a diagnostic error. A broader type/pack redesign is out of scope. PR #26 owns any exact-commit test results, review and merge state; do not infer hosted CI or a release from these notes.

## Decision 2026-11-14: preflight direct Python novelty thresholds

**Observed predecessor:** Unlike the CLI, the Python `OriginEngine` constructor stored any `d_min` value. NaN makes the first novelty-loop comparison false, leaving `best=None`; a subsequent cycle can then process a non-string history item. Negative or infinite values select misleading candidate-search behaviour, and Booleans are technically numbers in Python but not meaningful thresholds. This is a reproducible source-level API defect, not an observed customer incident. The original constructor accepted every test's invalid argument in the hosted red test.

**Choice:** Require an exact built-in int or float, a nonnegative value and finite real magnitude; handle huge ints that overflow `math.isfinite` conversion as invalid. Validate before `Kernel`, `Phase` and random state initialization. Accept `d_min=0`, default 0.35, and values above 1 exactly as before, preserving the old novelty algorithm and seeded output for valid callers. `ValueError` mentions `d_min` for immediate diagnosis. Avoid constraining upper bounds or silently coalescing arbitrary objects.

**Trade-offs and revisit:** Direct callers relying on inappropriate Boolean, text, huge-int or nonfinite targets will now receive an explicit error. The CLI already rejected negative and nonfinite values. This does not prevent external code mutating `engine.d_min` later; enforce a stronger immutable configuration contract only after establishing a concrete need. Hosted red/final CI and review evidence belong in PR #25.

## Decision 2026-11-14: reject parallel undirected edges in the graph scorecard

**Confirmed predecessor:** `_graph_connectivity_ok` accepted a five-node connected path plus a repeated edge as `{'nodes': 5, 'edges': 5}`, whether the second copy was ordered 0→1 or reversed 1→0. The source generator considers each unordered pair exactly once, so these inputs are malformed under its simple-graph contract. The scorecard otherwise gave them full Graph-Output credit. Synthetic red tests `2c6eb786` failed twice on hosted Python 3.11 and 3.12. No real user data incident was observed.

**Choice:** canonicalize each endpoint pair for duplicate detection after existing type, endpoint and self-loop checks; refuse a repeated pair with `duplicate_undirected_edge`. Preserve the original endpoint orientation in the graph and do not modify the generator. Rejecting duplicates in the diagnostic is more truthful than silently interpreting repeated records as additional independent edges. This also avoids inventing a multigraph format or deduplicating malformed evidence.

**Trade-offs:** This check deliberately treats repeated records as invalid rather than harmless redundant serialization. GraphML/JSON readback equivalence still judges file/sidecar equality separately, not structure; use Graph-Output to check the generated graph's structural contract. This does not enforce graph connectivity in the generator, prove graph source integrity, bound arbitrary user-supplied graph sizes, or change seeded outputs. Evidence for the corrected candidate belongs in PR #24.

## Decision 2026-11-11: reject ambiguous external JSON before generation

**Observed predecessor:** `load_spec` and `read_jsonl` use ordinary `json.load(s)`, which picks the last value for a duplicate key, including a `pipeline` or `prompt`, and accepts Python-only constants `NaN` and `Infinity`. The caller could receive a successful report for contradictory or nonstandard input. This is a deterministic synthetic code-path issue, not evidence of an actual owner data incident.

**Choice:** use shared `object_pairs_hook` and `parse_constant` checks at both external decoding boundaries, without changing Python's JSON output formats or the graph readback parser. Reject every duplicate key, including nested fields that the engine might otherwise ignore, and refuse nonstandard numeric constants. The existing CLI exception handling should emit an option-specific error and status 2 before constructing an engine.

**Alternatives and trade-offs:** leave last-key-wins as an undocumented parser convenience (misleading), or normalize ambiguous inputs after decoding (discarded values cannot be recovered). Strict refusal may reject previously tolerated nonconforming packs; authors must fix or regenerate those files. Whitespace, ordering, valid Unicode and standard JSON retain their meaning. This does not add semantic conditioning, integrity signatures, schema validation or power-loss protection. PR #22 owns the actual tests and validation evidence.

## Decision 2026-11-11: fail cleanly for invalid external JSON inputs

**Observed predecessor:** Optional `--spec` and `--prompts` files were opened and decoded outside the CLI's exception handling. Missing files, malformed JSON/JSONL or invalid UTF-8 escaped as Python tracebacks and exit code 1; a spec with a JSON array as its root raised an `AttributeError`. The synthetic test-first [PR #21](https://github.com/levelinglucy/Origin_Engine/pull/21) demonstrates seven failing cases on each hosted Python version, while a valid spec and opt-in prompt pack still work.

**Choice:** Require the spec JSON root to be an object, and catch file, decoding and JSON/value errors for each optional input at the CLI boundary. Report the offending option through `argparse.error` (exit 2) before creating an engine or generating output; keep the historical invalid-pipeline explanation and the valid prompt semantics unchanged. Preserve unmodified historical source and already accepted JSON formats.

**Trade-off and limits:** This is error reporting, not a security sandbox, validation of every optional spec field, strict duplicate-key JSON policy, an output transaction or protection against faults during later generation. Unconditioned JSONL contents are still parsed without influencing glyphs. Revisit input-schema tightening only for a concrete ambiguous-input use case. Candidate revision-specific validation remains in PR #21.

## Decision 2026-11-10: respect the dependency-bound scorer's epsilon argument

**Observed predecessor:** `test_dependency_bound(outputs, epsilon_max=...)` accepted an epsilon argument but always evaluated an adjacent-output 3-gram overlap against `0.05` and reported `target≈ε: 0.02`. Callers requesting a stricter or looser bound received the same score and advertised target. Synthetic outputs whose adjacent overlap is exactly 0.25 expose this defect: historical default scores 2.4, but selecting ε=0.12 ought to change the proxy allowance and score.

**Choice and acceptance:** Retain historical default scoring by documenting the existing heuristic proxy as **2.5 × epsilon_max** (default ε=0.02 corresponds to 0.05). Reject non-finite, nonnumeric, Boolean and out-of-range ε values; use the selected proxy allowance for scoring and report the actual requested epsilon. A test fixes the exact default score and asserts that a looser explicit bound changes the synthetic score and displayed target. Only scorecard API logic changes: generated glyphs, PRNG state, graphs and the CLI default are unaffected. [PR #20](https://github.com/levelinglucy/Origin_Engine/pull/20) owns revision-specific validation and integration status.

**Limits and trade-offs:** The factor 2.5 is a retained historical toy heuristic, **not** a derived confidence level, probability of dependence or meaningful causal independence criterion. Different epsilon requests can now change scores, but this does not make the underlying metric scientifically validated. For actual statistical evidence, replace the proxy in a separately versioned experiment with an explicit calibration and controlled baseline. Existing default reports stay score-equivalent; custom epsilon users may see corrected results.

## Decision 2026-11-10: reject invalid direct prompts before mutation

**Observed predecessor:** Although `run_prompts(..., condition_prompts=True)` checked an entire pack up front, direct callers of `OriginEngine.cycle(prompt=...)` hit the prompt type/blank/UTF-8 checks *after* `P`, `U` and `C` had consumed PRNG draws and assigned `self.iota`. A rejected request therefore changed the seeded engine even with no emission in history. [PR #19](https://github.com/levelinglucy/Origin_Engine/pull/19) contains the synthetic failed-state reproduction.

**Choice:** Reuse a single strict prompt-to-UTF-8 validator in both the full pack preflight and direct `cycle()`. On direct calls, validation now precedes the first operator, so rejected blank/non-string/unpaired-surrogate inputs do not consume seed state, change Iota or start a graph export. Valid prompt bytes feed the same versioned digest as before; `None` and default seeded runs are untouched. No snapshot/rollback machinery was added.

**Limits and alternatives:** This protects against *invalid prompt validation failures*, not arbitrary failures later inside the engine, graph publication or a crash halfway through a valid cycle. Wrapping the entire cycle in state-copy/restore would be a larger, unproven transaction change. This does not certify concurrent use, graph durability, scientific understanding or iPhone support. Revisit broader rollback only for a concrete failure and acceptance criteria; use the PR as the authoritative validation record.

## Decision 2026-11-10: opt-in exact-text prompt perturbation

**Problem and goal:** [PR #16](https://github.com/levelinglucy/Origin_Engine/pull/16) correctly exposed that `--prompts` was inert; the default must stay compatible with prior seeded experiments. A separate falsifiable experiment should let users measure sensitivity to supplied text rather than inventing a language model.

**Opt-in method:** `--condition-prompts` requires a nonempty JSONL pack of objects with a nonblank string `prompt` field. Each of `--cycles` emissions takes the next prompt in file order, repeating round-robin. Exact UTF-8 text (no normalization) is SHA-256 hashed with a versioned domain separator. That digest modifies a **cycle-local** `Iota` identity/orbit/entropy mark before `Ψ` emits symbols. The persistent base `Iota` and PRNG draw sequence are unaltered. History and phase can diverge naturally after differently conditioned outputs. Invalid packs fail before the first cycle; this includes escaped unpaired surrogate code points that Python's JSON parser accepts but UTF-8 encoding cannot represent. Prior to PR #18, a surrogate in a later pack entry could let earlier engine cycles run before the error was reported; all conditioned prompts now receive this UTF-8 preflight. No prompt text or hashes are added to logs by the mode itself.

**Acceptance and limitations:** Same seed, same pack and matching implementation produce identical stdout; changing a synthetic pack changes glyphs; without the opt-in flag, same-seed runs with or without a pack remain identical. Empty/invalid inputs are rejected. This is a deterministic *hash perturbation*, **not** semantic comprehension, meaningful prompt following, causality evidence, secrecy (output may disclose aspects of the input), cryptographic authentication, cross-platform byte identity or iPhone certification. Existing JSONL packs are not migrated. Alternative designs include feeding tokens into all six operators (larger behavioural change) or silently changing default `--prompts` (breaks established comparisons); both were rejected here. Consider versioning the method if a later semantic operator is designed. [PR #17](https://github.com/levelinglucy/Origin_Engine/pull/17) holds exact validation and status.

## Decision 2026-11-10: disclose unused JSONL prompt content

**Observed code path:** `--prompts` accepts and parses JSONL, but `run_prompts(engine, prompts, cycles=args.cycles)` only uses the cycle count, never the individual entries; `OriginEngine.cycle()` has no prompt argument. The CLI always passes a positive `--cycles` value (default 16). Therefore two different same-length, valid packs with identical `--seed` and `--cycles` generate the same output. Treating their text as conditioning would be a misleading claim.

**Choice:** Keep the existing generator and historical monolith unchanged. When a pack is supplied and successfully parsed, emit a succinct stderr notice that the JSONL content is not used for conditioning, and clarify `--help` and README. The notice reveals no prompt text or file location; stdout, seed draws, phase and graph formats stay the same. New tests explicitly pin non-conditioning and the warning, without claiming the pack does anything it does not.

**Alternatives and reconsideration:** Quietly ignoring data hides an important functional limitation; silently mixing text into P/C/Ψ would invent experimental semantics and invalidate seeded comparisons. A separately scoped milestone could define a conditional operator with tests for content sensitivity, repeatability, backward-compatible defaults and provenance. This patch is a disclosure, **not** prompt conditioning, scientific validation or iPhone certification. [PR #16](https://github.com/levelinglucy/Origin_Engine/pull/16) is the validation and integration record.

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
