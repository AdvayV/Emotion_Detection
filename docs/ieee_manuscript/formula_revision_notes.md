# Formula revision integration notes

Use the ordered records in `formula_explanations.json` as display Equations (1)-(11). The `equation_latex` value is an equation body: it has neither dollar delimiters nor an explicit number. Wrap each body in the renderer's numbered equation environment. Preserve `aligned` line breaks, center the mathematical block and place its number consistently at the right margin. Each `section_anchor` exactly matches a current manuscript heading without its Markdown prefix.

| Number | Key | Section |
|---|---|---|
| 1 | softmax_decision | 3.1 |
| 2 | temperature_scaling | 3.2 |
| 3 | jensen_shannon | 3.3 |
| 4 | weighted_cross_entropy | 3.4 |
| 5 | one_vs_rest_counts | 5.1 |
| 6 | precision_recall_f1 | 5.1 |
| 7 | macro_f1_accuracy | 5.1 |
| 8 | sadness_worked_example | 5.1 |
| 9 | negative_log_likelihood | 5.3 |
| 10 | multiclass_brier | 5.3 |
| 11 | expected_calibration_error | 5.3 |

Replace redundant inline code-style formulas with references to these displays. Introduce symbols in compact adjacent prose using each record's `symbols` field; avoid eleven large standalone notation tables. Keep the sadness calculation visibly labelled **illustrative**. It uses TP=8, FP=2 and FN=4 to obtain precision 0.8000, recall approximately 0.6667 and F1 approximately 0.7273; it is not a model result.

Implementation checks used `hinglish_emotion/evaluation.py`, `calibration.py`, `disagreement.py`, `training/train_transformer.py`, `training/train_advanced.py` and `hinglish_emotion/fusion.py`. Important conventions are the following:

- Per-class precision, recall and F1 return zero on a zero denominator. Macro-F1 averages all seven target classes.
- Temperature is bounded to [0.05,20], keeps zero probabilities at zero and preserves the class ordering.
- NLL uses natural logarithms and a target-probability floor of 1e-300. Brier sums the seven class errors before averaging rows; there is no additional division by seven.
- ECE uses ten equal-width, left-inclusive/right-exclusive bins, except that the final bin includes one. Empty bins are skipped.
- Jensen-Shannon comparison uses aligned normalized vectors, base-two logarithms and a square-root distance with a zero floor against numerical round-off.
- Balanced training weights are n_train/(7*max(n_c,1)). The microbatch loss divides by the sum of target weights. Accumulation averages scaled microbatch losses; it should not be described as an exactly identical weighted reduction over one larger batch.

Numerical checks against the actual functions confirmed the illustrative sadness result, clipped natural-log NLL, class-sum Brier convention, confidence-one ECE bin, identical/disjoint Jensen-Shannon cases, temperature handling of zero entries and PyTorch's target-weight denominator. These checks performed no model training. No empirical result, calibration measurement, dataset scope or generalization claim is changed by this revision.
