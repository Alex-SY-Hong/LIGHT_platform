# Acceptance criteria — Agent/verification gates

Every criterion below is machine-checked by the test suite
(`python -m pytest Agent/verification`, 45 tests, coverage >= 80%,
current 86%). A criterion only counts as met when its test is green on
the branch under review.

## reproducibility_check (task 4)

- Double-run of `reliability_analysis.analyze_reliability` is
  bit-identical, and regeneration is value-identical (0 field diffs)
  to the archived `reliability_analysis_results.json`.
- `runs >= 2` and non-empty artifacts are enforced; vacuous PASS is
  impossible (tested).
- All consecutive run pairs are compared; an intermittent flip in run 1
  of 3 cannot hide (tested).
- NaN-vs-value and bool-vs-int drift FAIL (regression-tested); any gate
  crash still appends the run log before exiting non-zero.

## schema_check (task 5)

- Stamping the deployed rf_cv10 best-fold payload yields the 1024-column
  contract; the 630-candidate feature table matches it in set AND order
  (validated once on the real artifacts).
- Missing, extra, or reordered columns FAIL loudly; padding/truncation
  paths do not exist in the gate.
- Join row counts: 630 == 630 across the candidate tables.
- Key-sequence alignment: a same-count row permutation FAILs at the
  first differing position.

## physical_plausibility (task 3)

- YM rules are gel-scope only; on the real database they flag exactly
  the four mixed-unit rows caught by the gel-scope rules (Tier A) plus
  one suspect row (Tier B) — pinned by a slow-marked
  test; any rule change that moves this baseline turns the suite red.
- SR rules: sub-1 domain findings and the SR=300 suspect row are
  recorded as run evidence (append-only log); they are triage input,
  not acceptance blockers.
- Zero false positives on a clean synthetic corpus (fault-injection
  suite asserts this every run).

## fault_injection (task 6)

- Mutation score 100%: all five corruption families (log10 shift,
  ratio scale, unseeded randomness, column shuffle, row shuffle) are
  intercepted by the gate named in the FMEA.
- Clean corpus: zero false positives across all gates.
- Every suite run appends an immutable append-only record.

## Package-level

- `python -m pytest Agent/verification` exits 0.
- Coverage of `Agent.verification` >= 80% (CLI entry points are
  exercised by CI rather than unit tests; their logic is thin).
