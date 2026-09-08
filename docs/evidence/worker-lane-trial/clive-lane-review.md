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

## The throughput split — RETRACTED

This section originally reported that generated tokens per second separated every
success from every failure in this wave (passes 2.04–4.18, failures 1.43–2.03, no
overlap across 14 runs) and proposed it as an early-abort signal.

**That does not hold, and no rule should be built on it.** The peer fleet tested it
against 26 clive runs and found **full overlap**: passes 1.69–6.54 output-tokens/s,
failures 1.79–6.12, with its *fastest* failure (6.12 out/s, 68 turns) dying and its
*slowest* pass (1.69 out/s) landing clean. The split here was a small-sample
artifact — in this wave tokens/second happened to track node **shape**, and shape
is the real variable. The larger sample breaks the throughput correlation while
preserving the shape result.

What survives from the measurement is the negative half: **`machine_seconds` is not
the discriminator either.** Failures spent 65–1,684 s of machine time, passes
109–935 s, and one node computed for **137 s across 26,879 s of life** — 0.5%. The
work is not heavy; it is queued.

## Mechanism — corrected

Every stream carries `cache_read_input_tokens: 0` and
`cache_creation_input_tokens: 0` — **no prompt caching on this lane**, confirmed on
all 14 streams here and all 26 of the peer fleet's. One node here re-sent 9.07M
input tokens across 25 turns; the peer's heaviest re-sent 37M across 173 turns.
That uncached per-turn re-send is the genuine cost driver.

**But context accumulation is not what kills a node, and the original claim here
that it "walks itself into the window" was wrong.** The peer fleet's figures invert
it: per-turn context is *higher* in the survivors — dead scouts carried 41k–101k
input tokens per turn against 107k–215k for passing implement nodes — and its
largest run of all passed at 173 turns and 37.1M cumulative input. Turn count and
cumulative input separate nothing.

The mechanism that explains **both** fleets is a **long single output late in a
long session under load**. Deaths cluster at or near the report-writing turn,
40–150 minutes in, and never hit a node whose brief made the deliverable a small,
early, incremental write.

That accounts for the two partial survivors here, which had been filed as luck: one
wrote its findings **into the manifest** and then died before its report, so the
findings survived; another **committed** and then died, so the commit survived.
Both survived because the durable write came before the long output. The four total
losses all had the report as one final generation.

Turn counts at the four total losses here: 81, 61, 43, 25 — but the peer's passing
run at 173 turns shows that is a symptom of shape, not a threshold.

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

1. **Make the deliverable a small, early, incremental write**, and write the
   manifest before any long output. This is the highest-value change in either
   fleet's data: both partial survivors here survived because a durable write
   preceded the long generation, and the peer fleet's four rescues had complete
   deliverables on disk with no manifest.
2. **Route reading-shaped work off this lane.** 0 of 5 scouts finished on the peer
   fleet against 3 of 3 on codex in 11–27 minutes; report-shaped nodes here landed
   3 of 8. Split multi-file briefs per file and prefer `exact` spec level with
   causes pre-isolated.
3. **Measure what rests on a contract before briefing its change.** One node was
   recorded `malformed-node` for exactly this omission; the failures it exposed
   were legitimate and predictable.
4. **Never cite `/tmp` as gate evidence.** Preserve logs into the repository
   before a ledger row points at them.
5. **Read the diff and the logs, never the gate verdict.** Of eight landings,
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
