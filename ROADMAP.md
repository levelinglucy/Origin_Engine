# Origin Engine roadmap

## Development summary

**Goal:** a runnable, recoverable symbolic generator with faithful graph exports and optional reproducible experiments.
**Current stage:** proposed `v0.1.1` candidate; entrypoint, export integrity and Φ⁴ integration are merged. Seeded replay is a tested PR candidate, not yet merged or tagged stable.
**Current priority:** verify per-engine seeded replay, default entropy invariance and both high-phase graph formats; then revisit target-device validation and a separate release decision.
[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Developer notes](./DEV_NOTES.md)

| Milestone | Deliverable | Acceptance check | Status |
| --- | --- | --- | --- |
| Entry-point repair | Canonical monolith, CLI docs, isolated smoke tests | Original source blob intact; Python 3.11/3.12 CLI help, two-cycle run and invalid spec exit pass | Completed and merged in PR #1; candidate not released |
| Export-integrity candidate | Exclusive graph naming, XML-safe GraphML with key declaration | Isolated historical repro; three same-second JSON writes preserved; XML-special Unicode labels parse and match sidecars; Python 3.11/3.12 CI | Completed and merged in PR #2; 9/9 checks on Python 3.11/3.12 on PR and main |
| Real high-phase integration | Accelerated test-only phase transitions; actual engine cycles emit JSON/GraphML sidecars; failed writes counted in scorecard | Three real Φ⁴ graph emissions per format round-trip and have distinct paths; mocked write failure is not scored as success | [PR #4](https://github.com/levelinglucy/Origin_Engine/pull/4) candidate, 12/12 tests passed in PR CI on Python 3.11/3.12; stable not tagged |
| Candidate promotion | Confirm final merged head and stable-entrypoint criteria | Relevant checks pass on exact merged commit, known limitations documented and predecessor recoverable | CI gate met; stable release still pending |
| Opt-in seeded replay | Private per-engine PRNG, `--seed INTEGER`, unchanged default entropy | Same seed reproduces across fresh processes and normalized Φ⁴ files; different seeds diverge; interleaved streams isolated | 17/17 tests passed on Python 3.11/3.12 in [PR CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42210406967); awaiting final review/merge, not stable |
| Device validation | Run on trusted target iPhone/Pythonista device | CLI launch, bounded cycles, JSON/GraphML writes and readback verified on device | Not tested; does not block research branches |

**Release gate:** do not declare a stable patch when applicable data-integrity defects remain. Validate exact candidate revisions, preserve the historical monolith, and document incomplete platform testing. A CI success is not a claim that all possible defects are absent. No release or tag is created by this roadmap.
