# Origin Engine roadmap

## Development summary

**Goal:** make the experimental generator easy to run, test and reproduce without changing its historical source.  
**Stage:** entrypoint repair candidate, proposed patch `0.1.1`, **unreleased**.  
**Next:** establish passing candidate CI, evaluate opt-in seeded mode, then test graph exports.  
[Overview](./README.md) · [Changelog](./CHANGELOG.md) · [Developer notes](./DEV_NOTES.md)

| Stage | Deliverable | Acceptance | Status |
| --- | --- | --- | --- |
| Entry point candidate | Correct standalone filename, docs, isolated CLI checks | Historical source preserved; canonical source exact match; `--help`, 2-cycle execution and invalid spec check pass under Python 3.11/3.12 | In development; CI pending |
| Patch verification | Reviewed and merged source/README/tests | Candidate checks green; head unchanged; remote readback | Planned |
| Experimental reproducibility | Optional explicit seed without altering default behaviour | Identical outputs for repeated seeded runs in clean directories; different seeds diverge; existing default still works | Optional next milestone |
| Graph export hardening | Compare emitted JSON/GraphML at high phases | Parse both formats and validate label fidelity, edges and absence of filename collisions in a bounded fixture | Optional next milestone |

**Release rules:** keep the historical blob recoverable. Do not label this work released/stable merely because tests were added or a PR merged; verify the exact candidate and note limitations first. Functional or data-integrity defects take priority over cosmetic changes. No release or tag is created by this roadmap.
