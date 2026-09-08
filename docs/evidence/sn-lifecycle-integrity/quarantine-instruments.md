# The two quarantine instruments, and the WHERE clause that separates them

**Recovered by the coordinator.** The node dispatched for this ran 81 turns over
7.5 hours and was abandoned by lane saturation leaving no commit, no report and
no manifest at all. The question blocks the cohort tail drain's first step and the
withheld names on the release path, so it is answered here from the code and the
live graph. Base `69f24046b`; graph read 2026-09-08.

## The answer: they are not two readings of one question, they are a confirm and a clear

| Instrument | Question it answers | What it writes |
|---|---|---|
| `default_revalidate` (`campaign.py:694`) | "of the names I just touched **that were not quarantined**, which are now clean?" | non-quarantined → `valid`; still-dirty → re-`quarantined` |
| `default_clear_quarantine` (`campaign.py:648`) | "lift the audit quarantine on **accepted** names so the docs pools can claim them" | `quarantined` → `pending`, a deliberate transient |

**The whole divergence is one line.** `default_revalidate`'s confirm branch is:

```cypher
WHERE coalesce(sn.validation_status, '') <> 'quarantined'
SET sn.validation_status = 'valid'
```

A quarantined name is **excluded by construction**. So a "revalidation sweep" run
through this instrument can never lift a quarantine: it confirms only rows that
were already clean, silently skips every quarantined one, and returns a
`confirmed` count that reads exactly like success. That is why the recorded sweep
"reported 23 of 35 quarantines as stale and clearable" and then "cleared zero of
23" — it used the confirm instrument where the clear instrument was required, and
the confirm instrument's own filter made the no-op invisible.

The code already states this, at `campaign.py:998-1000`: *"the prose grep is a
campaign-specific signal the ISN audit does not carry, and it must not overwrite
an audit quarantine (`default_revalidate` only confirms non-quarantined members to
'valid')"*. The rule was written down; nothing enforced that a sweep used the
right instrument.

## The sequence the code encodes, which the sweep skipped

`default_clear_quarantine` resets to `pending` rather than `valid`, and its
docstring says why: the docs review and accept path excludes
`validation_status='quarantined'`, so an accepted-but-quarantined name never
surfaces a refined doc. So the order is **clear → drain → revalidate**: lift to
the honest transient first, let the drain do the work, then let
`default_revalidate` restore `valid` or re-quarantine. Running revalidate alone,
first, cannot start that chain.

## Live population, and a correction to the plan's figure

612 names carry `validation_status='quarantined'`:

| name_stage | count |
|---|---|
| superseded | 303 |
| exhausted | 274 |
| drafted | 16 |
| **accepted** | **14** |
| reviewed | 4 |
| pending | 1 |

The 577 superseded and exhausted rows are terminal and are already excluded from
the quality gate's numerator and denominator, so the live cohort that matters is
the **14 accepted** ones — exactly the set `default_clear_quarantine` admits,
since it requires `name_stage='accepted'`.

**The plan's §9b figure has moved.** It records "5 … accepted stage … scores 0.844
to 0.906". Measured now: 14 accepted-and-quarantined, of which **2** sit in that
score band. Re-derive before acting on 5, as with the 78→93 unsourced drift.

**A second defect, found while counting: 12 of the 14 carry
`quarantine_reason = null`.** A quarantine with neither an observation time nor a
stated reason cannot be triaged by inspection at all — this is §5's
reasonless-failure footgun reappearing on the validation axis. The one row that
does carry a reason reads `campaign: banned prose persisted after refine`, written
by `default_revalidate`'s re-quarantine branch, which confirms that path is live.

## What this licenses, and what it does not

Licensed now: a sweep over the 14 may proceed through `default_clear_quarantine`,
in the clear → drain → revalidate order, and its result is readable because the
instruments' roles are settled.

Not licensed: treating a `confirmed` count from `default_revalidate` as a
clearance figure, ever. The plan's measure — "both surfaces answer the same
question, or each states which question it answers and refuses the other" — is
half met: they now demonstrably answer different questions, but **neither refuses
the other's question**, so the same wrong call remains available to the next
caller. Making `default_revalidate` refuse a quarantined id outright, rather than
filtering it away, is the remaining work.
