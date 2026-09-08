# Quarantine instrument authority

Measured on the production `codex` graph from the login-node-local graph
tunnel on 2026-09-08. Every live read was a bounded `StandardName` query; the
repair used the normal deterministic validation path and made no model call.

## Verdict

`validated_at` plus the verdict written in the same transaction by
`mark_names_validated` is authoritative for whether a name is currently
quarantined. A stored `validation_status = 'quarantined'` with
`validated_at IS NULL` is an undated historical value, not a current
quarantine finding. The scalar still carries the outcome, but it has no
authority without the observation that produced it.

The two surfaces that were treated as quarantine instruments do not measure the
same thing:

| Code path | Predicate it evaluates | Effect |
|---|---|---|
| `audits.run_audits` | returns the issues produced by its fixed list of codex audit checks for the candidate fields supplied by its caller | produces evidence only; it neither runs the complete ISN admission gate nor writes a verdict or observation time |
| `default_revalidate` | for proposed clean ids, first matches `coalesce(sn.validation_status, '') = 'quarantined'` | is a campaign prose-confirmation step; it refuses those matched ids and therefore cannot clear a quarantine |
| `default_audit_revalidate` → `drain_validation_for_ids` | claims explicit ids whose `validated_at IS NULL`, then runs `validate_name_candidate` | calls `mark_names_validated`, which atomically writes `validation_issues`, `validation_status`, and `validated_at`; this is the quarantine authority |

The error in the earlier sweep was treating a fresh `run_audits` issue list as a
current graph verdict, then using the final confirmation helper as if it were
the deterministic validator. The confirmation guard is designed to prevent
that: it collects the requested ids whose predicate is
`coalesce(sn.validation_status, '') = 'quarantined'` and raises instead of
setting them `valid`. It cannot clear a quarantine. The correct sequence is
clear the validation stamp → run the complete deterministic admission gate →
atomically record the verdict and observation time. For a documentation
campaign, `default_clear_quarantine` additionally moves an accepted quarantine
to `pending`; it now clears the old `validated_at` at that same transition.

## Bounded live census before revalidation

Property coverage proved that the predicates could match before interpreting
the cohort:

| accepted names | `validation_status` present | `validated_at` present | stored quarantines | quarantines with an observation |
|---:|---:|---:|---:|---:|
| 2,360 | 2,360 | 2,094 | 14 | 0 |

The stored-status/stamp cross-tab was `quarantined + unvalidated = 14`,
`valid + unvalidated = 252`, and `valid + validated = 2,094`. Thus the prior
fourteen quarantine scalars were all stale as current verdicts; this conclusion
does not assert that the underlying names are valid.

## Revalidation cohort

The independent export ledger identifies exactly five candidate rows excluded
because their stored status was quarantined. All five were passed to
`default_audit_revalidate`, which cleared their validation stamps and delegated
to `drain_validation_for_ids`; no direct status-setting query released a name.
The receipt was `cleared=5`, four re-quarantined ids, and one valid id.

Scores below are the live `review_mean_score` before and after the run. Two
drafted identities have no score. This corrects the older five-scored-name
description without changing the export-side count of five.

| identity | score before / after | before | authoritative result | after |
|---|---:|---|---|---|
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | none / none | drafted, quarantined, undated | grammar round-trip failure | drafted, quarantined, dated |
| `radial_outline_of_plasma_boundary` | 0.8444 / 0.8444 | accepted, quarantined, undated | ISN semantic error: `outline` does not name the represented entity | accepted, quarantined, dated |
| `radial_outline_of_wall` | 0.9063 / 0.9063 | accepted, quarantined, undated | same ISN semantic error | accepted, quarantined, dated |
| `vertical_coordinate_of_line_of_sight` | 0.8828 / 0.8828 | accepted, quarantined, undated | valid, no issues | accepted, valid, dated |
| `vertical_outline_of_plasma_boundary` | 0.7792 / 0.7792 | drafted, quarantined, undated | ISN semantic error plus missing length-language evidence | drafted, quarantined, dated |

`vertical_coordinate_of_line_of_sight` is released from the quarantine filter.
The other four remain withheld for newly observed reasons rather than for stale
scalars. Their grammar, semantic, and documentation repairs belong to their
own owners; a status writer must not erase them.

## Corrected census

For the five export exclusions, the corrected census is **4 genuinely
quarantined and 1 valid**. Restricting to accepted names, the post-run graph has
2,360 accepted rows: 13 retain the quarantine scalar, but only **2 are dated
current quarantines** and 11 remain undated historical values outside this
five-id operation. The code now makes the temporal invariant explicit:
campaign clearing removes the old observation time, and only the complete
deterministic validator restores a dated verdict.

The live graph operation ran on the login node because its Bolt endpoint is a
login-node-local tunnel. The scope was five indexed ids, every query stayed
below ten seconds, and no heavy local computation or model call accompanied it.
