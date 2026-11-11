# FMEA — Failure Mode and Effects Analysis (verification layer)

Scope: the LIGHT_platform data path, ordered by data flow. For every
component: how it can fail silently, which gate detects it, and the
fallback. Standing principle: **a detected failure escalates to more
human review, never to more automation.** Gates are read-only; they
flag, isolate-recommend, and log — they never rewrite pipeline data.

| # | Component | Silent failure mode | Detection | Fallback (human escalation) |
|---|-----------|--------------------|-----------|-----------------------------|
| 1 | Literature extraction (PDF → tables) | Unit-convention drift between sources (e.g. Pa vs kPa logged values landing in one column) | `physical_plausibility` YM gel-scope rules (>4.0 flag, >5.5 suspect), externally anchored | Tier A: isolate-review recommendation; collaborator verifies against source paper |
| 2 | Property databases (YM / SR CSVs) | Values outside physical definition domain (ratio < 1, placeholder zeros); extreme-but-possible outliers | `physical_plausibility` SR domain + suspect rules | Tier A/B per rule; run log (append-only) is evidence for triage |
| 3 | Feature generation (Morgan pooling) | Feature column set or order drifts between training and prediction time | `schema_check` fingerprint (stamp → check, set + order) | Hard stop before predict; no zero-padding, no truncation |
| 4 | Unsupervised clustering → candidate tables | Row permutation between paired tables (counts equal, alignment wrong) | `schema_check assert_join_alignment` (key sequence, position by position) + `assert_join_rowcounts` | Hard stop at the join; both tables re-exported by hand |
| 5 | Supervised training (RF regression / classification) | Hyperparameter or seed drift making runs incomparable | `reproducibility_check` double-run, field-by-field, all consecutive pairs | FAIL verdict; training rerun under reviewed config |
| 6 | Deployment prediction (best model on candidates) | Model/artifact mismatch (trained under different library version or feature contract) | `schema_check` stamp from model payload + version note in acceptance log | Predict refused; pin environment (task-9 requirements freeze) |
| 7 | LLM consensus / final selection | Nondeterministic aggregation or silent partial failure of a run | `reproducibility_check` on archived-result regeneration; consensus numbers verified field-exact against archived JSON | FAIL verdict; archived results remain the citable snapshot |
| 8 | Verification gates themselves | Gate regression (a rule stops firing, or fires spuriously) | `fault_injection` mutation suite (5/5 families), real-DB baseline pinned in tests, coverage >= 80% | Suite red = gate change reverted before merge |

Known accepted limits (documented, not hidden):

- Gates validate structure, determinism, and physical range — not
  chemical correctness; that judgment stays human (Tier A exists for it).
- `reproducibility_check` treats int and float as equal when numerically
  equal (documented policy); dtype-only drift is not flagged.
- Run logs are single-writer (JSONL append); concurrent gate processes
  on one machine are not supported.
