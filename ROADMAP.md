# Origin Engine roadmap

## Development summary

**Goal:** a runnable, recoverable symbolic generator with lossless graph exports and transparent experimental limits.  
**Current stage:** proposed `v0.1.1` candidate; entrypoint and graph-integrity fixes merged with passing CI. No tagged stable release.  
**Current priority:** trusted target-device checks and optional seeded replay research, before a separate stable release decision.  
[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Developer notes](./DEV_NOTES.md)

| Milestone | Deliverable | Acceptance check | Status |
| --- | --- | --- | --- |
| Entry-point repair | Canonical monolith, CLI docs, isolated smoke tests | Original source blob intact; Python 3.11/3.12 CLI help, two-cycle run and invalid spec exit pass | Completed and merged in PR #1; candidate not released |
| Export-integrity candidate | Exclusive graph naming, XML-safe GraphML with key declaration | Isolated historical repro; three same-second JSON writes preserved; XML-special Unicode labels parse and match sidecars; Python 3.11/3.12 CI | Completed and merged in PR #2; 9/9 checks on Python 3.11/3.12 on PR and main |
| Candidate promotion | Confirm final merged head and stable-entrypoint criteria | Relevant checks pass on exact merged commit, known limitations documented and predecessor recoverable | CI gate met; stable release still pending |
| Opt-in seeded replay | Explicit seed option, unchanged default entropy | Same seed in clean directories reproduces normalized outputs; different seeds diverge; default behaviour unaffected | Optional, separate follow-on |
| Device validation | Run on trusted target iPhone/Pythonista device | CLI launch, bounded cycles, JSON/GraphML writes and readback verified on device | Not tested; does not block research branches |

**Release gate:** do not declare a stable patch when applicable data-integrity defects remain. Validate exact candidate revisions, preserve the historical monolith, and document incomplete platform testing. A CI success is not a claim that all possible defects are absent. No release or tag is created by this roadmap.
