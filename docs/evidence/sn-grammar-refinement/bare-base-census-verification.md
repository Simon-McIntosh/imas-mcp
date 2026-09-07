Coordinator verification of n-sgr-how-many-accepted-names-are-bare-bases
run at 2026-09-07T23:25:56Z against the live graph, base 69f24046b

--- accepted population ---
MATCH (sn:StandardName) WHERE sn.name_stage='accepted' RETURN count(sn)
  => 2360   (report claims 2360: MATCH)

--- bare single-token accepted ---
MATCH (sn:StandardName) WHERE sn.name_stage='accepted' AND NOT sn.id CONTAINS '_' RETURN sn.id, sn.origin, sn.source_types, sn.reviewer_model_name, sn.reviewer_score_name, sn.semantic_sim
  => 3 rows (report claims exactly 3: MATCH)
     beta      origin=derived     source_types=['dd']      model=structural-inheritance      score=0.49375  semantic_sim=0.8295314908027649
     momentum  origin=derived     source_types=None        model=structural-inheritance      score=None     semantic_sim=None
     vorticity origin=catalog_edit source_types=['catalog'] model=openrouter/x-ai/grok-4.5    score=0.8625   semantic_sim=0.6986501812934875
  every field matches the report's table.

--- name-axis review history ---
MATCH (sn:StandardName {id:$n})-[:HAS_REVIEW]->(r:StandardNameReview) WHERE r.review_axis='name' RETURN r.resolution_method, toString(r.reviewed_at) ORDER BY r.reviewed_at
  vorticity => 3 rows, one authoritative_escalation, ZERO semantic_similarity_gate rows
               (report claims passed on merit with no gate-firing row: MATCH)
  beta      => 12 rows, SIX semantic_similarity_gate rows at
               2026-07-22T13:15:13, 2026-07-23T18:35:30, 2026-08-11T06:39:40,
               2026-08-25T09:41:08, 2026-09-02T14:15:32, 2026-09-05T14:13:20
               (report claims 6 pre-definition-coupling firings: MATCH on count and dates)
               and the 2026-09-05T21:22:39 rows carry NO gate row, only quorum_consensus
               (report claims the definitions-coupled gate stopped firing: MATCH)

--- source-line claims, read at base ---
workers.py:8368   is_derived = item.get("origin") == "derived"                 EXACT
defaults.py:24    SEMANTIC_SIM_GATE_RESOLUTION_METHOD = 'semantic_similarity_gate'  EXACT
defaults.py:27    SEMANTIC_SIM_CRITICAL = 0.55                                   EXACT
defaults.py:35    SEMANTIC_SIM_SYNTHETIC_SCORE = 0.30                            EXACT
graph_ops.py:25632 def _structural_accept_route(...) documenting a source_free route  EXACT

VERDICT: every checkable claim in the report holds. EXIT=0
