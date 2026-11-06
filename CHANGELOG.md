# Origin Engine changelog

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
