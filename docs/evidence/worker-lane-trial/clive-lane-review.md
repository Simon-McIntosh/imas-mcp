# The locally served worker lane, reviewed over 14 runs

One coordinator session (`ship-s10-20260907`), 14 nodes dispatched to the `clive`
backend — DeepSeek-v4-flash on a 4-GPU serve — while a peer coordinator held 20
nodes on the same endpoint. Written because a lane's economics are only visible
across a whole wave, and because a review that records only the failures would
misdescribe this one.

## Outcomes

| | count |
|---|---|
| gate passed | 7 |
| failed on a defective **brief** (work sound, landed as a pair) | 1 |
| infrastructure failure | 6 |

Of the six infrastructure failures: four produced nothing, one committed real work
before dying, and one wrote its findings into its manifest but never its declared
report. **Four of the six had their deliverable recovered** — three by the
coordinator reading the question directly, one by merging the orphaned commit.

## The throughput split, which is the finding worth carrying

Sorting all 14 by generated tokens per second:

| outcome | range | mean |
|---|---|---|
| passed | 2.04 – 4.18 tok/s | 2.62 |
| infrastructure failure | 1.43 – 2.03 tok/s | 1.82 |

**No overlap.** Every failure at or below 2.03; every pass at or above 2.04. The
fastest run of the wave, 4.18, was also the most tightly briefed — `exact` spec
level, two files, both root causes pre-isolated so it had nothing to re-derive.

If the split holds on another fleet, tokens/second is usable as an **early-abort
signal**: a coordinator could redispatch a slow node rather than discover at hour
seven that its response never arrived.

`machine_seconds` is *not* the discriminator. Failures spent 65–1,684 s of machine
time, passes 109–935 s. One node computed for **137 s across 26,879 s of life** —
0.5%. The work is not heavy; it is queued.

## Mechanism

Every stream carries `cache_read_input_tokens: 0` and
`cache_creation_input_tokens: 0` — **no prompt caching on this lane**. One node
re-sent 9.07M input tokens across 25 turns. Per-turn cost therefore grows with an
accumulating context, and every death is a response that stopped arriving
(`subtype: success` with `is_error: true`, or `Request timed out`).

Turn counts at death: **81, 61, 43, 25**. Every loss was a long report-shaped or
multi-file brief. **Not one narrow single-outcome node died.** That is the shape
the mechanism predicts, and it is the shape the wave produced.

## What these workers did well

Recorded because it changed outcomes, not as balance.

- **Refused a stale premise instead of inventing work.** Briefed to fix a cascade
  defect, one worker found the fix had landed six days earlier, verified the commit
  was an ancestor, and wrote the missing regression suite instead. Judgement, not
  compliance.
- **Corrected the coordinator's briefs three times, all correct** — a wrong line
  reference and the predicate behind it; a stale figure of ~137 replaced by a
  measured 105; a plan's own counts.
- **Falsified a plan premise by measurement.** The plan held that a single-token
  name cannot clear a 0.55 similarity gate; a worker found an accepted one that
  cleared it at 0.699 on a real quorum. Every field re-queried and exact.
- **Reported added failures rather than weakening its own change.** A contract
  change exposed five tests pinning the retired contract; the worker reported
  5 → 7 plus three collateral instead of making its gate pass. The most valuable
  single behaviour observed — and the exposure was the brief's fault, not the
  work's.
- **Counterfactual proofs by default, by name**: "5 of 8 fail by name when
  reverted", "4 of 5 on a literal revert". Verified from the logs; exact.
- **Virtual-merge discipline when asked** — overlaid an unmerged peer commit,
  measured there, confirmed byte-identity, restored a clean tree.
- **Respected the compute rule unprompted**, running a 7,555-test gate on a SLURM
  debug partition rather than the login node.
- **Wrote findings into the manifest rather than deferring them**, which is the
  only reason one node's work survived its own delivery failure.
- **Left a latent hazard recorded rather than fudged**, naming why it could not be
  closed inside its fence.

## Where they struggled

- **Long multi-part briefs do not survive this lane.** Six failures, all
  report-shaped or multi-file.
- **A fabricated symbol.** One report named `_NonCanonicalParseError`, which
  exists nowhere; the real raise is a bare `ValueError`. Stated as fact and
  load-bearing for a downstream remediation, so the plan was fenced against it.
- **`status: complete` with an absent deliverable.** Caught only by the cheap
  untrust checks; a passing gate would have hidden it entirely.
- **Ephemeral evidence paths.** Two nodes cited `/tmp` for gate logs and
  reproduction scripts — nine scripts and four logs. They survived long enough to
  verify and preserve, but a ledger row citing them would have rotted.
- **Imprecise quantifiers that changed scope.** "Trailing <token>" for tokens
  sitting mid-name; "nine automated refines" where 20 was measurable and the
  window ran ninety minutes past the stated cutoff. A remediation sized on the
  reported figure would have under-scoped by half.
- **Duplicated helpers instead of importing them**, leaving one module with a
  same-named private helper carrying a different signature from its sibling.
- **One figure unreconciled** — 53 against 473 for the same-sounding population,
  its narrower definition never stated.

## What the coordinator changes as a result

1. **Split multi-file briefs per file**, and prefer `exact` spec level with causes
   pre-isolated. The one node briefed that way was the fastest of the wave.
2. **Measure what rests on a contract before briefing its change.** One node was
   recorded `malformed-node` for exactly this omission; the failures it exposed
   were legitimate and predictable.
3. **Never cite `/tmp` as gate evidence.** Preserve logs into the repository
   before a ledger row points at them.
4. **Read the diff and the logs, never the gate verdict.** Of eight landings,
   every one had something the summary did not say: a fabricated symbol, an absent
   artifact, a doubled exposure figure, a duplicated helper.

## Cross-fleet corroboration — nova S19, 26 clive runs on the same serve

The peer coordinator ran 26 clive nodes and 6 codex nodes in the same window on
the same endpoint. Three findings arrived **independently from both fleets**,
which is why they are worth acting on rather than noting:

| Finding | this fleet | nova S19 |
|---|---|---|
| **shape decides survival, not role tier** | code nodes 5/6 landed; report-shaped 3/8 | implement 8/8 landed; investigate scouts **0/5** |
| **manifests under-report real work** | one read `commits: (pending)` over a real commit; one read `complete` with the artifact absent | three passed runs left commits, artifacts and evidence fields empty |
| **no prompt caching, so context is paid in full every turn** | `cache_read` and `cache_creation` 0 on every stream; 9.07M input re-sent over 25 turns | broad reading balloons toward the 1M window and dies there — a grep-only, write-incrementally brief still reached **1.1M** and died with 6 of 9 rows |

Nova's additions, which this fleet could not have measured:

- **Codex did the same three scouts in 11, 27 and 11 minutes.** Against 0 of 5 on
  clive, that settles where reading-shaped work belongs. "Use local workers" should
  not be read as covering scouts.
- **Image reads make a session unresumable.** Four long implement workers (7–12 h)
  finished their computation and wrote receipts, then died before committing; three
  could not be resumed because they had read their own PNG and the lane answers
  `400 not a multimodal model` on replay. That silently converts a recoverable
  session into an unrecoverable one — the most dangerous item in either fleet's
  report.
- **Single generations run 15–25 minutes** under roughly 30 concurrent consumers.
  That explains a stall this fleet chose not to act on: a node tripped the 900 s
  window at 935 s quiet and returned to `working` seven minutes later. **On this
  lane a stall event is a long write until proven otherwise, and the 900 s window
  is mis-tuned rather than the node being sick.**
- **Review-role nodes are reliable here**: 7 of 7 scoring reviews delivered,
  recomputing headline numbers independently and raising unrequested caveats.
- A test node lost both arms of a full-suite run by launching them into one-hour
  debug allocations without chunking.

One distinction this fleet can add to nova's throttling observations: **zero
`api_retry` records across all 14 streams here**, while two other projects' runs
on the same lane in the same window carried 10 and 5. Retry pressure and queue
starvation are separable on this endpoint, and only the second killed nodes here.
