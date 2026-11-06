# Origin Engine changelog

## Unreleased — entrypoint repair candidate (2026-11-06)

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
- Required acceptance: tests pass on the candidate commit in CI, canonical Git blob equals historical blob and README launch path works. Initial test status is **pending** until the workflow is observed.
- Not covered: reproducibility, advanced graph export, iPhone/Pythonista runtime and scientific claims.

[Decision and trade-offs](./DEV_NOTES.md) · [Release gate](./ROADMAP.md)

Earlier history remains in the original repository files and Git history; no prior changelog has been inferred.
