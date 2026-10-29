# Shared-data emotion model comparison

The **Model comparison** workspace performs new local inference on one supplied
file. **Emotion lab > Encoder results** displays saved training-run results;
**Model disagreement** remains the small-phrase review and correction queue.

## Run it

From the project root:

```powershell
python -m pip install -e ".[model,dashboard]"
python -m streamlit run dashboard/app.py
```

Open http://127.0.0.1:8501 and choose **Model comparison**. Models must already be
installed under models/. GitHub does not contain the large trained weights. Use
[the Colab and installation guide](../GET_STARTED.md) when preparing a fresh copy.
No model is downloaded or trained by this comparison page.
Streamlit file watching is disabled in the project configuration to avoid inspecting
lazy Transformers modules after inference. Restart the server after code edits;
checkpoint-version checks still run whenever the page reruns.

Select Hindi-English for XLM-R, mBERT and MuRIL. Select English-French for XLM-R
and mBERT. Missing models are omitted from the available choices, never assigned
zero scores. Choose at least two installed checkpoints.

## Input file

CSV must use UTF-8; XLSX is also accepted. Uploads are limited to 25 MB and 50,000
input rows; at most 10,000 selected phrases can be compared in one run.

```csv
text,emotion,language_pair,split
Aaj main bahut khush hoon,joy,hi-en,test
Kal ke exam se darr lag raha hai,fear,hi-en,test
```

The text column is required. Map other column names using **Map comparison
columns**. Expected emotion labels are joy, sadness, anger, fear, surprise,
disgust and neutral. Existing emotion aliases such as happy and sad are accepted;
positive/negative sentiment labels cannot be treated as seven-emotion labels.
Language aliases such as en-fr are normalized. Without a language column, the
chosen comparison language applies to the file.

Only the chosen language enters the run. Test rows are selected by default when
train/validation/test metadata is present. A file containing only training rows
cannot silently become a test benchmark. Disable test-only selection only for an
explicit diagnostic run; the changed scope is recorded.

On this local installation, the normal test input is:

- data/training/multilingual_bootstrap_v1/hindi_english/test.csv: 1,000 synthetic Hindi-English test phrases.
- data/training/multilingual_bootstrap_v1/english_french/test.csv: 1,000 synthetic English-French test phrases.
- data/training/multilingual_bootstrap_v1/test.csv: both language slices; each comparison filters its chosen mix.
- [evaluation_set](../evaluation_set/README.md): separate 100-phrase checking drafts per mix, with unreviewed labels.

Generated training/test files are local and ignored by Git. On a fresh checkout,
`python -m training.package_multilingual` prepares the documented synthetic data
and Colab package. This generates examples; it does not train model weights.
The tracked checking drafts remain available without generating the full corpus.

## What happens after Run

1. Validate the file, seven-label schema, selected language, test partition and checkpoints.
2. Clear shared classifier caches and load one compatible CPU checkpoint.
3. Predict in small batches and retain all seven scores with the actual checkpoint identity.
4. Release that classifier before loading the next, reducing simultaneous memory use.
5. Find the union of training/validation overlaps across selected model guards.
6. By default, exclude each flagged phrase from every model's score. Score only identical labelled eligible rows.
7. Store the completed result in Streamlit session memory. A failed selected model aborts the run; demo rules are not substituted.

The default batch size is four and CPU thread count is two. Lower the batch size
if memory is constrained. Timing records checkpoint loading separately from
prediction/preprocessing; latency is specific to the machine and settings.

## Read the five tabs

| Tab | Meaning |
| --- | --- |
| Scoreboard | Accuracy, seven-class macro-F1, correct/scored counts, truncation and timing for each model |
| All model outputs | Expected label, each model's prediction/confidence/correctness, disagreement and scoring eligibility; searchable, with filters |
| Per-emotion metrics | Precision, recall, F1, class support and one confusion matrix per model |
| Paired mistakes | Both correct, only the left model correct, only the right correct and both wrong on the same scored rows |
| Run details | Dataset hashes, selection/exclusion settings, checkpoint paths/versions, training scope and runtime settings |

Accuracy is correct predictions divided by scored labelled rows. Macro-F1 gives
each of the seven emotion classes equal weight; zero-support classes remain part
of that fixed average and are identified. Neither the top prediction probability
nor model agreement is accuracy. The phrase-detail panel shows all seven scores
from every model. Paired counts are descriptive; they are not a statistical
significance test and do not create a voting classifier.

Files without labels still produce predictions, confidence distributions and
agreement diagnostics. They never receive accuracy, F1 or confusion matrices.
Rows excluded by overlap remain in the output with included_in_scoring=false.
Missing guards are warnings, not certification of independent held-out data.

## Exports and freshness

- model_comparison_scores.csv: each model's scores, scored counts and timing.
- model_comparison_predictions.csv: aligned phrases, expected labels, every model's prediction and seven probabilities, source/warning/overlap flags.
- model_comparison_report.json: full metrics, confusion matrices, paired counts, hashes, checkpoint identities, scope and runtime settings.

Use the download buttons to save results. No predictions are written to disk
automatically. Inputs, column mappings, selected language, models, checkpoint
versions and CPU/exclusion settings bind the displayed result. Changing them
requires a new run. Search and display filters do not rerun inference or rescore.

## Interpret the experiment honestly

The current installed XLM-R and mBERT checkpoints used 16,000 bilingual training
examples; MuRIL used 8,000 Hindi-English examples. A common test set supports a
comparison of these configurations, but does not isolate the effect of encoder
architecture. Scores on generated templates or unreviewed checking labels are
not validated population-level emotion accuracy. Human-reviewed independent data
is needed for stronger claims. Review labels before treating either a high or a
low model score as conclusive.

## Verification

```powershell
python -m pytest tests/test_model_comparison.py tests/test_model_comparison_ui.py -q
```

Tests exercise identical row selection, hand-computed metrics, shared overlap
exclusions, duplicated source IDs, no-label behavior, missing/failing models,
classifier release, language restrictions, stale-result invalidation and exports.
Local full-test replay results, when available, are recorded separately in
[comparison_verification.json](evaluation/comparison_verification.json); they
recompute predictions rather than copying the saved score table.
