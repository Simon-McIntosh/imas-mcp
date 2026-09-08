# Did the deliverable get written incrementally, or in one closing generation?

The variable two fleets converged on, and which no ledger row carries. Measured
here with `deliverable_write_shape.py`, which reads a run's stream and places
every write against its **declared** write paths in the run's own timeline.

**A first version of this instrument counted every write, including scratch, and
separated nothing** — a node that writes ten scratch query scripts and no report
scores as "incremental" by write count. Filtering to the declared deliverable is
what makes it discriminate, and it is worth stating because the broken version
looked plausible.

## Result over the eight report-shaped nodes of the first wave

| node | deliverable writes | span | outcome |
|---|---:|---|---|
| quarantine-instruments | **0** | — | died, total loss |
| every-export-exit | **0** | — | died, total loss |
| release-residues | **0** | — | died, total loss |
| reasonless-failures | 1 | 0.007 | died, total loss |
| which-section-rows | **0** | — | reported `complete`, artifact absent |
| bare-bases | 6 | 0.10 → 0.98 | landed |
| unsourced-72 | 5 | 0.03 → 1.00 | landed |
| edits-flag-scoping | 4 | **0.94 → 1.00** | **landed** |

## What this supports, and what it does not

**It does not support incremental writing as a survival predictor.** The
`edits-flag` node wrote its deliverable entirely inside the final 6% of its run —
one closing generation by any definition — and **landed cleanly**. So a closing
generation is not fatal.

**Every total loss never wrote its deliverable at all.** Three touched it zero
times; the fourth created it once in the first 1% and never returned. They spent
their whole lives accumulating investigation with nothing on disk. The distinction
is therefore not *how* the deliverable was written but *whether the run ever
reached its writing phase* — and a node that dies before that phase is not
diagnosable by write shape, because there is no write to shape.

**So the recommendation both fleets sent upstream needs splitting in two.**

- As a **recovery** measure it holds, and this wave is evidence for it: the
  `which-section-rows` node's findings survived intact because it wrote them into
  its *manifest* before the report it never produced, and one implement node's work
  survived because it *committed* before dying. The peer fleet's four rescues had
  complete deliverables on disk for the same reason.
- As a **survival** measure it is unsupported. Writing incrementally does not make
  a node finish. It makes a node that dies leave something behind.

Those are different claims and conflating them oversells the change. "Commit each
deliverable and write the manifest before any long output" is right — as
loss-mitigation, not as death-prevention.

## The instrument's other use, which is immediate

It detects **`status: complete` with a never-written deliverable from the stream
alone**, with no worktree inspection: `which-section-rows` reports zero
deliverable writes. That class cost this session a near-miss — a passing gate
would have hidden it — and it is cheap to check automatically at promotion.
