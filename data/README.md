# Dataset guide

## New two-language XLM-R bootstrap

`training/multilingual_bootstrap_v1` contains the combined Hindi-English and
English-French bootstrap: 16,000 training, 2,000 validation and 2,000 synthetic
test rows. Each language contributes 8,000/1,000/1,000. Its `hindi_english` and
`english_french` subfolders provide the separate-language files and audits.
These are generated templates, not independently reviewed research data.
The original dataset below and its trained checkpoint are preserved.

Use `templates/labelled_example.csv` to try file accuracy and
`templates/unlabelled_example.csv` to try prediction without expected labels.
These are illustrative format examples; they are not the training corpus.

## The dataset used for the current trained model

Open **[training/synthetic_hinglish_8000](training/synthetic_hinglish_8000)**.
This is the working copy of the Hindi-English dataset used for the first XLM-R
experiment. It contains seven labels: joy, sadness, anger, fear, surprise,
disgust and neutral.

| File | Rows | Use |
| --- | ---: | --- |
| [train.csv](training/synthetic_hinglish_8000/train.csv) | 8,000 | Training examples that update weights |
| [validation.csv](training/synthetic_hinglish_8000/validation.csv) | 1,000 | Checkpoint selection and validation |
| [test.csv](training/synthetic_hinglish_8000/test.csv) | 1,000 | Synthetic test examples held out from training |
| [all_splits.csv](training/synthetic_hinglish_8000/all_splits.csv) | 10,000 | Actual trainer input, with the fixed split column |
| [DATASET_CARD.md](training/synthetic_hinglish_8000/DATASET_CARD.md) | — | Generation method and limitations |
| [dataset_audit.json](training/synthetic_hinglish_8000/dataset_audit.json) | — | Counts, provenance and the CSV hash |

**The labels are synthetic and have not been independently human-reviewed.**
They enable an initial training experiment; its scores are not verified
real-world accuracy. Keep the split column and related examples together.
The audit records the exact combined CSV, so do not silently edit labels or
partitions and retain the old audit. A revised dataset needs a new version and
an updated audit before it is packaged or trained.

The combined CSV is the file to pass to `training.compare_encoders`. The
individual CSVs make each partition easy to inspect; they are not extra data.

## Demo datasets

These small files exercise the interface and rules. They are **not** the
8,000-phrase training dataset.

| Folder | Contents |
| --- | --- |
| [demo/sentiment](demo/sentiment) | The 75-phrase workbook and the small `sample_messages.csv` |
| [demo/emotions](demo/emotions) | The 84-phrase illustrative emotion CSV and its label notes; Emotion lab currently selects its 42 Hindi-English rows |

Regenerate only the sentiment demonstration workbook with
`python scripts/build_phrase_workbook.py` from the project root.

## Human-reviewed research data

- [annotation](annotation/README.md) holds collection and independent-review
  instructions. Its collection is unfinished; it is not the current training set.
- `audits` holds public-source and annotation checks. The MaSaC audit records
  unresolved reuse verification. That corpus was not included in this synthetic run.
- `raw` and `prepared` are reserved for future verified source data and prepared
  research datasets. They are not created or filled with placeholder data here.

## Other files you may need

- **Upload to Colab:** [../colab](../colab/README.md). The upload ZIP contains a
  verified snapshot of the working dataset, so it is self-contained.
- **Recorded training results:** `../outputs/training/xlmr_synthetic_8000`.
  Its frozen manifest and predictions are original run evidence, not a second
  working training set.
- **Weights used by the dashboard:** `../models/xlmr_emotion`.
- **Historical files and approved cleanup records:** `../archive`. Redundant
  download ZIPs were deleted after the installed model and datasets were verified.

Rebuild the Colab upload with `python -m training.package_synthetic --overwrite`.
Existing working CSVs are verified and copied; they are not silently regenerated.
On a fresh checkout, the same command without `--overwrite` creates the generated
dataset and upload package. Generated data and model artifacts are ignored by Git.

## Separate checking phrases

[evaluation_set](../evaluation_set/README.md) is outside training: 100 Hindi-English
and 100 English-French challenge drafts, plus unlabelled prediction copies.
Their proposed labels are assistant-authored and not independently reviewed.
The exact-overlap audit found no matches against the 20,000-row frozen source
or installed checkpoint guards; that is not evidence of human label quality.
