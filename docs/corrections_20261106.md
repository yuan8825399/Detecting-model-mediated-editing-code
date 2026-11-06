# Corrections to the 1.0.0 analysis

This directory is a separate corrected analysis. The released corpus, source-group
partitions, original release and original manuscript are preserved. All corrected
results are regenerated from the corpus in a previously empty output directory.

## Corrected computations

- Hierarchical inference evaluates the conditional second head for every test text.
  True test labels are used only for evaluation. The resulting five-class posterior
  is normalized and no longer uses labels to route test inputs.
- Transfer comparisons use a matching in-domain fused detector (C=1, balanced binary
  class weights). The single-generator matrix remains a separate matched comparison.
- Subfield labels are computed from original titles/abstracts and broadcast to all
  variants of the source before source-subfield holdout or metadata controls.
- A conformal quantile whose required rank exceeds the calibration count is infinity.
  All analysis scripts use the same implementation.
- Effective-only presence tasks exclude unchanged edits under every instruction, while
  retaining every original. The corrected test denominator is 1,332.
- Character edit counts recover integer distance using original, not output, length.
- Robustness operating thresholds use calibration originals. Cross-era models retain
  the original pooling of all non-test sources, now disclosed in the manuscript.
- Coverage uncertainty uses source-document bootstrap intervals (2,000 resamples).
  Per-task performance intervals resample fixed test predictions, without refitting.

## Interpretation and implementation boundaries

Character operation decompositions use SequenceMatcher alignment. They are not a
decomposition of the optimal Levenshtein path; the supplied normalized Levenshtein
distance is a separate quantity. The surface-feature control includes lexical diversity
as well as length and punctuation. The isolation audit checks source membership and exact
duplicate titles/abstracts; it does not implement near-duplicate screening.

PAL-CP reports test precision and retrospective measured edit magnitude. It does not
provide a deployed edit-magnitude estimator or calibration-only precision estimates.
With multiple correlated variants per source, source-disjoint partitions do not alone
establish record-level exchangeability. Coverage and repeated partitions are empirical
evaluations, not verification of a finite-sample guarantee for this corpus design.

The legacy CSV fields `ceiling` and `normalised_tpr` in
`per_condition_ceiling.csv`, `max_recall` in `ceiling.csv`, and `pooled_ceiling` in
`recalib.json` are retained for file compatibility. They mean the
non-identical fraction and recall divided by that fraction. This fraction is not an
unconditional recall upper bound when false positives are allowed.

## Reproduction

`analysis/06_main_experiments.py` writes stable `base` and `bert` result and prediction
filenames. The pipeline declares `exploratory/18_method_probe.py` as the producer of
the PAL-CP comparison dependency. Missing inputs or BERT embeddings fail explicitly.

Run `python scripts/run_pipeline.py --stage all` to rebuild all stages. For the exact
analysis environment used in this correction, see `requirements-reproduction.txt`.
Frozen BERT uses `bert-base-chinese` revision
`8f23c25b06e129b6c986331a13d8d025a92cf0ea`; set `HF_HUB_OFFLINE=1` to use its local cache.
The embedding manifest records the model revision, corpus hash and record ordering.

Missing generation requests and original annotation-process provenance have not been
reconstructed. Corpus completeness and numeric agreement do not validate those processes.
