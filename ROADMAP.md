# Origin Engine roadmap

## Development summary

**Goal:** a runnable, recoverable symbolic generator with faithful graph exports and optional reproducible experiments.
**Target:** proposed `v0.1.1` source candidate on Python 3.11/3.12, with optional iPhone/Pythonista support unverified.
**Integration, validation and release:** see the single [current development summary](./DEV_NOTES.md#development-summary) and exact evidence linked from individual PRs; the roadmap is not a second status dashboard.
[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Developer notes](./DEV_NOTES.md)

| Milestone | Deliverable | Acceptance check | Status |
| --- | --- | --- | --- |
| Entry-point repair | Canonical monolith, CLI docs, isolated smoke tests | Original source blob intact; Python 3.11/3.12 CLI help, two-cycle run and invalid spec exit pass | Completed and merged in PR #1; candidate not released |
| Export-integrity candidate | Exclusive graph naming, XML-safe GraphML with key declaration | Isolated historical repro; three same-second JSON writes preserved; XML-special Unicode labels parse and match sidecars; Python 3.11/3.12 CI | Completed and merged in PR #2; 9/9 checks on Python 3.11/3.12 on PR and main |
| Real high-phase integration | Accelerated test-only phase transitions; actual engine cycles emit JSON/GraphML sidecars; failed writes counted in scorecard | Three real Φ⁴ graph emissions per format round-trip and have distinct paths; mocked write failure is not scored as success | [PR #4](https://github.com/levelinglucy/Origin_Engine/pull/4) merged; 12/12 prior PR tests per Python version; untagged |
| Candidate promotion | Confirm final merged head and stable-entrypoint criteria | Relevant checks pass on exact merged commit, known limitations documented and predecessor recoverable | CI gate met; stable release still pending |
| Opt-in seeded replay | Private per-engine PRNG, `--seed INTEGER`, unchanged default entropy | Same seed reproduces across fresh processes and normalized Φ⁴ files; different seeds diverge; interleaved streams isolated | Merged PR #5; 17/17 tests passed on Python 3.11/3.12 in [merged-main CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42210586116); not stable |
| CLI input honesty | Refuse non-positive `--cycles` and non-finite/negative `--dmin` before generation | Rejected with exit 2, valid boundary accepted; full suite passes | [PR #7](https://github.com/levelinglucy/Origin_Engine/pull/7) merged, [main CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42438215235) 20/20 per Python version; untagged |
| CLI scorecard target fidelity | Use the validated `--dmin` in the displayed `OE-Compliance` baseline rather than silently reporting `0.35` | Isolated three-cycle seeded CLI regression for zero and overridden targets; full 3.11/3.12 workflow | Proposed in [PR #10](https://github.com/levelinglucy/Origin_Engine/pull/10); see PR for actual validation and merge status, unreleased |
| Complete-only graph publication | Stage a closed graph then exclusively claim its final filename | No final path visible during write; prior exports intact after error; 22 tests per version | [PR #8](https://github.com/levelinglucy/Origin_Engine/pull/8) merged, [main CI](https://github.com/levelinglucy/Origin_Engine/actions/runs/42467061939); untagged |
| Strict graph scalar serialization | Reject non-XML control values and nonstandard JSON floats; retain XML carriage returns | Invalid inputs refused before filesystem side effects; GraphML CR round-trips; previous 22 tests remain valid on both CPython versions | [PR #9](https://github.com/levelinglucy/Origin_Engine/pull/9) contains live integration/validation state; untagged |
| Device validation | Run on trusted target iPhone/Pythonista device | CLI launch, bounded cycles, JSON/GraphML writes and readback verified on device | Not tested; does not block research branches |

**Release gate:** do not declare a stable patch when applicable data-integrity defects remain. Validate exact candidate revisions, preserve the historical monolith, and document incomplete platform testing. A CI success is not a claim that all possible defects are absent. No release or tag is created by this roadmap.
