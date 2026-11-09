# Origin Engine changelog

## Unreleased — CLI novelty scorecard accuracy (2026-11-09)

- Use the validated custom `--dmin` baseline in the printed `OE-Compliance` scorecard. Previously the engine respected the chosen novelty target while the scorecard silently evaluated and displayed the built-in `0.35`.
- Add a seeded, isolated CLI regression for both zero and a non-default target; preserve the default scorecard and all other metrics. No graph format or historical source changes.

[Decision and limitations](./DEV_NOTES.md#decision-2026-11-09-score-the-chosen-cli-novelty-baseline) · [PR #10 validation and integration](https://github.com/levelinglucy/Origin_Engine/pull/10). Unreleased source candidate, no iPhone/Pythonista or power-loss assertion.


## Unreleased — strict graph scalars (2026-11-08)

- Reject GraphML forbidden XML 1.0 characters in node IDs, edge endpoints and labels *before* creating an output directory, instead of publishing XML another parser cannot open.
- Preserve carriage returns in GraphML text with character references; default XML line-ending normalization previously changed label content.
- Refuse non-finite floats in JSON exports rather than writing nonstandard `NaN`/`Infinity` tokens. Previously generated malformed files are not silently rewritten.
- Preserve complete-only exclusive publication, ordinary valid graph data and the historical monolith.

[Encoding rationale](./DEV_NOTES.md#decision-2026-11-08-reject-graphs-that-cannot-be-represented-faithfully) · [PR #9 evidence and integration](https://github.com/levelinglucy/Origin_Engine/pull/9). Still untagged and unreleased; CPython tests do not certify Pythonista or filesystem durability.

## Unreleased — complete graph publication (2026-11-08)

- Stage complete JSON/GraphML bytes in a private same-directory temporary file before atomically claiming the final filename with an exclusive hard link. Previously the final path was visible while being written, so interruption or concurrent inspection could encounter incomplete JSON/XML.
- Preserve pre-existing graphs, collision suffixes and the unchanged historical monolith. A failed publication leaves no claimed final file; ordinary cleanup removes the stage. An abrupt kill may leave a hidden staging file. Hard-link-disabled filesystems fail closed, and published files now default to owner-only permissions.

[Design and limitations](./DEV_NOTES.md#decision-2026-11-08-publish-only-fully-written-graph-outputs) · [PR #8 / validation record](https://github.com/levelinglucy/Origin_Engine/pull/8). Not tagged stable; device, power-loss and hostile filesystem-race behaviour remain unverified.

## Unreleased — honest CLI input contracts (2026-11-08)

- Refuse `--cycles` of zero or less, which previously caused a seemingly successful one-emission run.
- Refuse non-finite or negative `--dmin` before processing, preventing NaN/Infinity from causing misleading output or a late failure. Zero remains supported.
- Add isolated CLI regression checks with valid-boundary coverage. Historical source and programmatic engine controls remain unchanged.

[Input-contract decision](./DEV_NOTES.md#decision-2026-11-08-make-invalid-cli-requests-fail-before-work) · [PR #7 / validation evidence](https://github.com/levelinglucy/Origin_Engine/pull/7). The proposed `v0.1.1` remains unreleased; no stable tag or Pythonista/device certification.

## Unreleased — opt-in seeded replay candidate (2026-11-06)

**Status:** separate development branch for proposed `v0.1.1`. Merged in [PR #5](https://github.com/levelinglucy/Origin_Engine/pull/5) at [`ab27ea30`](https://github.com/levelinglucy/Origin_Engine/commit/ab27ea30c22b93da9a0625c2c22bc5d2d96e204f); 17/17 tests passed on Python 3.11/3.12 in final PR and merged-main CI. Not tagged stable.

### Added
- Optional integer `--seed` and `OriginEngine(seed=...)` introduce private pseudorandom replay across P/U/C/Ψ. Default operator calls continue to use `os.urandom` and `SystemRandom`.
- Seeded replay checks cover fresh Python processes, different seeds, independent interleaved engines, default OS entropy and normalized Φ⁴ JSON/GraphML round trips.
- README usage and developer decision note explain trade-offs and continuation.

**Reason:** deterministic research comparisons without altering global random state or reducing default entropy. **Validation:** [final PR CI 42210535894](https://github.com/levelinglucy/Origin_Engine/actions/runs/42210535894) and [merged-main CI 42210586116](https://github.com/levelinglucy/Origin_Engine/actions/runs/42210586116) both passed 17/17 tests on each Python 3.11 and 3.12, including five new replay tests. No target-device check or stable release. Seeded draws are not cryptographic; timestamps/paths and cross-version output identities are not guaranteed.

[Engineering rationale](./DEV_NOTES.md#decision-2026-11-06-opt-in-per-engine-seeded-replay) · [Roadmap](./ROADMAP.md)

## Unreleased — high-phase integration and honest export metrics (2026-11-06)

**Status:** included in [PR #4](https://github.com/levelinglucy/Origin_Engine/pull/4) for proposed `v0.1.1`. Tested candidate; not a tagged stable release.

### Fixed
- Graph-Dump-Consistency now counts Φ⁴+ graphs whose disk export raised an error or lacked a file path as failed checks. Previously those graphs were omitted, allowing misleading passing scores or a no-graph default after failed writes.

### Tests
- Added integration checks running the real six-operator engine through phase transitions to Φ⁴ using a test-only threshold, then verifying three actual graph emissions per format, distinct paths, JSON/GraphML round trips and non-vacuous consistency scores.
- Added a simulated disk failure test proving the sidecar is retained, the dump error is reported and the failure contributes to the score denominator.

**Reason:** helper-level serialization tests cannot prove the engine's normal high-phase pipeline emits correct sidecars, and omitting failed dumps from metrics hides data loss. **Validation:** 12/12 tests passed on each Python 3.11 and 3.12 at [PR CI run 42202352772](https://github.com/levelinglucy/Origin_Engine/actions/runs/42202352772), including all three new integration tests. Actual device/Pythonista execution and crash-atomic writes remain unverified.

[Engineering rationale](./DEV_NOTES.md#decision-2026-11-06-count-failed-graph-exports) · [Roadmap](./ROADMAP.md)

## Unreleased — graph export integrity candidate (2026-11-06)

**Status:** merged into main via [PR #2](https://github.com/levelinglucy/Origin_Engine/pull/2) at [`0b263516`](https://github.com/levelinglucy/Origin_Engine/commit/0b263516ff2fa5e87ebfcae5730b2bb0092a2ca6); `v0.1.1` remains an untagged candidate, not stable.

### Fixed
- Graph exports with identical phase, identity prefix and UTC second no longer truncate previous emissions. Each new export uses exclusive creation and a deterministic collision suffix.
- GraphML now escapes XML-special node labels through standard-library element serialization and declares the GraphML label key.

### Tests and documentation
- Added isolated regression checks reproducing original overwrite/malformed-XML behavior and validating corrected JSON/GraphML paths, labels, edges and scorecard consistency.
- Pinned the historical source by its original Git blob digest, letting the actively maintained entrypoint evolve while preserving provenance.
- Updated project overview, developer decisions, release roadmap and continuation guidance.

**Reason:** silent loss of an earlier graph is a data-integrity defect; malformed XML defeats the advertised export format. **Validation:** nine checks passed on Python 3.11 and 3.12 in [final PR CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42194431084) and [merged-main CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42194484337). Target-device testing and formal stable promotion remain unverified.

[Engineering rationale](./DEV_NOTES.md#decision-2026-11-06-prevent-graph-overwrite-and-invalid-xml) · [Release gates](./ROADMAP.md)

## Earlier integrated candidate — entrypoint repair (2026-11-06)

**Status:** development/verification candidate; `0.1.1` is a *proposed* next patch version, not a published release.

### Fixed
- Provided `origin_engine_monolith.py` under the name advertised by the README, preserving exact contents and the existing historical file `python3 origin_engine_monolith.py`. The CLI source itself is unchanged.
- Repaired the truncated Markdown quick-start code fence and clarified actual run instructions.

### Added
- Standard-library subprocess smoke tests for CLI help, two-cycle execution, invalid pipeline handling, and byte-for-byte historical source identity.
- Python 3.11/3.12 GitHub Actions verification workflow.
- Top-level development summary, staged targets, decision rationale, continuation instructions and stated compatibility limits.

### Why / checks
- Reason: the original repository could not execute the filename shown in its README because that name was absent. The copied file preserves provenance without performing a risky source rewrite.
- Required acceptance: tests pass on the candidate commit in CI, canonical Git blob equalled the historical blob at PR #1 and the README launch path worked. The initial entrypoint candidate passed all four checks on Python 3.11 and 3.12 in PR and merged-main GitHub Actions on 2026-11-06; see PR #1. Additional graph integrity coverage belongs to the later candidate above.
- Not covered: reproducibility, advanced graph export, iPhone/Pythonista runtime and scientific claims.

[Decision and trade-offs](./DEV_NOTES.md) · [Release gate](./ROADMAP.md)

Earlier history remains in the original repository files and Git history; no prior changelog has been inferred.
