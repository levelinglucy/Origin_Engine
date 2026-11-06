# Origin Engine roadmap

## Development summary

**Goal:** a runnable, recoverable symbolic generator with lossless graph exports and transparent experimental limits.  
**Current stage:** proposed `v0.1.1` candidate; entrypoint patch merged, graph-integrity repair under review. No tagged stable release.  
**Current priority:** confirm reproducible graph export fixes with CI and remote readback before new features.  
[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Developer notes](./DEV_NOTES.md)

| Milestone | Deliverable | Acceptance check | Status |
| --- | --- | --- | --- |
| Entry-point repair | Canonical monolith, CLI docs, isolated smoke tests | Original source blob intact; Python 3.11/3.12 CLI help, two-cycle run and invalid spec exit pass | Completed and merged in PR #1; candidate not released |
| Export-integrity candidate | Exclusive graph naming, XML-safe GraphML with key declaration | Isolated historical repro; three same-second JSON writes preserved; XML-special Unicode labels parse and match sidecars; Python 3.11/3.12 CI | In review; CI pending |
| Candidate promotion | Confirm final merged head and stable-entrypoint criteria | All relevant checks pass on exact commit, known limitations documented, previous version recoverable | Planned, not a published release |
| Opt-in seeded replay | Explicit seed option, unchanged default entropy | Same seed in clean directories reproduces normalized outputs; different seeds diverge; default behaviour unaffected | Optional, separate follow-on |
| Device validation | Run on trusted target iPhone/Pythonista device | CLI launch, bounded cycles, JSON/GraphML writes and readback verified on device | Not tested; does not block research branches |

**Release gate:** do not declare a stable patch when applicable data-integrity defects remain. Validate exact candidate revisions, preserve the historical monolith, and document incomplete platform testing. A CI success is not a claim that all possible defects are absent. No release or tag is created by this roadmap.
