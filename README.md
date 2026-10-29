# Hinglish Emotion Detection

A local Streamlit application for exploring sentiment and seven emotions in
mixed-language phrases, with optional transformer checkpoints and local Qwen
review. Training runs separately in a GPU notebook; the dashboard loads the
resulting checkpoint for prediction.

**Review documents:** [Documents for review](Documents%20for%20review/README.md).

**Start here:** [Run, file evaluation and Colab instructions](GET_STARTED.md).
The current emotion workspace supports Hindi-English and English-French, CSV/XLSX
prediction and labelled-file scoring. The installed combined XLM-R checkpoint
covers both language pairs while the original Hindi-English checkpoint remains
available separately. The current training handoff covers all three encoders and a separate sarcasm task:
[colab/project_training](colab/project_training/START_HERE.md).

See the [23 October review report](Documents%20for%20review/Review%20reports/23OctReview%20AdvayV%20KaavishG%20CD.docx)
for the system map, saved synthetic results and evidence limits.

The [IEEE journal manuscript](Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal.pdf)
and [editable Overleaf project](Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal_Overleaf.zip)
include 31 verified references, a 20-paper implementation comparison, four module
algorithms, the seven-emotion checkpoint specification and 18 explained editable
equations with a worked F1 example. See
[manuscript sources and editing instructions](docs/ieee_manuscript/README.md).

## Current dashboard and evidence

The emotion-first dashboard has eight workspaces. **Overview** derives readiness
from `dashboard/readiness.py` across three encoder families and two language
profiles. Four installed checkpoint directories represent three families: XLM-R
and mBERT support Hindi-English and English-French; MuRIL supports Hindi-English
only. Three-class sentiment remains a separate legacy task. Four module flows
cover live prediction, matched comparison, Colab training and human review.

Training code, installed weights and effectiveness are distinct. Existing emotion
weights are installed, but fusion, learned context and supervised sarcasm weights
are not; there is no reinforcement learning. Accuracy on independently labelled
human messages and any Qwen review gain remain unmeasured. The revised IEEE
report contains 16 pages after the verified alignment revision.

A checkpoint with known unsupported `trained_language_pairs` is rejected for
single, context and batch emotion prediction; `evaluate-emotions` preflights all
language pairs. Legacy checkpoints with unknown coverage remain experimental and
warned. CLI `predict-file` allows fallback only when explicitly requested via
`--allow-fallback`; matched comparison never falls back.

## Compare XLM-R, mBERT and MuRIL in one place

Open **Model comparison** in the Streamlit sidebar. Select **Hindi-English** to
score all three installed emotion encoders together; **English-French** compares
XLM-R and mBERT because MuRIL has no supported French checkpoint.

1. Upload one CSV/XLSX, or choose a bundled test/checking dataset.
2. Map the phrase, expected emotion and language columns if needed.
3. Keep test-only selection and shared training-overlap exclusion enabled.
4. Click **Run and score selected models** once. Models load sequentially on CPU.
5. Inspect the **Scoreboard**, **All model outputs**, **Per-emotion metrics**,
   **Paired mistakes** and **Run details** tabs. Download the aligned CSV and JSON
   report to retain the same predictions, labels, model versions and settings.

Every selected model receives identical phrases and is scored on the same eligible
labelled rows. If any selected model flags a phrase as training/validation overlap,
the default policy excludes it from every score. Missing overlap guards are shown
explicitly. Without expected labels the page shows outputs and disagreement, with
no fabricated accuracy or F1. No Qwen review, demo fallback or consensus vote is
mixed into these encoder scores.

Use accuracy alongside seven-class macro-F1 (the mean balance of precision and
recall across the seven emotions), per-emotion F1 and confusion matrices. Model
agreement and a high phrase confidence are not evidence that a label is correct.
Synthetic and unreviewed checking labels remain marked as diagnostic data.
See the [comparison guide](docs/MODEL_COMPARISON.md) for schema, interpretation,
CPU settings, exports and reproducible verification.

The 29 October 2026 local replay used all 1,000 shared Hindi-English synthetic
test rows, with no detected training/validation overlap. All three results matched
their saved language-specific test metrics:

| Model | Accuracy | Seven-class macro-F1 |
| --- | ---: | ---: |
| XLM-R | 95.6% | 0.9563 |
| mBERT | 84.3% | 0.8439 |
| MuRIL | 80.3% | 0.8123 |

These are configuration results on synthetic held-out rows; training scope differs
between MuRIL and the two bilingual models. See the
[replay evidence](docs/evaluation/comparison_verification.json) for counts,
paired mistakes, runtime settings and the dataset fingerprint.

## Find the datasets and Colab files

Start with [the dataset guide](data/README.md). The current combined-language
Colab package uses [multilingual_bootstrap_v1](data/training/multilingual_bootstrap_v1):
20,000 synthetic rows (16,000 train, 2,000 validation, 2,000 test), split equally
between Hindi-English and English-French. These are generated examples and need
independent human evaluation.

The earlier Hindi-English working CSVs remain
under [data/training/synthetic_hinglish_8000](data/training/synthetic_hinglish_8000).
Use `train.csv` for the 8,000 training phrases, `validation.csv` for the 1,000
validation phrases and `test.csv` for the 1,000 synthetic test phrases. The trainer
uses `all_splits.csv`, whose split column keeps these roles separate.

Colab uploads are under [colab](colab/README.md), trained weights are under
`models/`, and the downloaded run reports are under
`outputs/training/xlmr_synthetic_8000`. Small UI fixtures live under `data/demo`;
they are separate from the training set. Historical working files and cleanup
records are under `archives`; the latest organisation retained downloaded ZIPs,
previews and caches without deletion.

## What works today

| Capability | Implementation and current limits |
| --- | --- |
| Three-class sentiment | Positive, neutral and negative; always-available rule baseline and optional local transformer |
| Shared-data model comparison | One scoring run for compatible XLM-R/mBERT/MuRIL checkpoints; aligned predictions, shared overlap exclusions, accuracy/macro-F1, per-class results, paired mistakes and CSV/JSON exports |
| Seven-emotion detection | Joy (happy), sadness, anger, fear, surprise, disgust and neutral; demo rules or a trained seven-class checkpoint |
| Language scope | Hindi-English and English-French; XLM-R routes to language-compatible installed weights and labels any explicit demo fallback |
| Expressive spelling | Conservative normalization, negation preservation and elongation cues such as `acchaaa` |
| Emoji and sarcasm | Word–emoji agreement/conflict cues, a rule-based sarcasm baseline, and an optional separately trained binary sarcasm checkpoint |
| Local Qwen | Optional Ollama `qwen3:0.6b` review for sentiment and seven emotions; context, uncertainty and conflicts can trigger review; suggestions retain the original classifier scores |
| Research workbench | Input/reviewer ablations, experimental calibration, blind review and versioned corrections, conversation traces, and the advanced GPU training handoff |
| Model training | Colab jobs for combined-language XLM-R/mBERT, Hindi-English MuRIL and reviewed Hindi-English binary sarcasm; fixed splits, Drive recovery and inference export |

The installed XLM-R, mBERT and MuRIL emotion checkpoints are trained weights;
they are separate from the new advanced-fusion model. Synthetic and demo scores
do not establish performance on real user messages. The advanced fusion notebook
is available, but no advanced checkpoint has been trained or installed.

## Current mBERT run and model review

Use [the focused mBERT Colab guide](colab/mbert_study/START_HERE.md). Upload
`colab/mbert_study/TRAIN_MBERT_STUDY.ipynb`, select a T4 GPU, run all cells and
upload `colab/mbert_study/mbert_study.zip` at the first prompt. The package
contains source and the same bilingual 16,000/2,000/2,000 frozen partitions as
combined XLM-R. The completed run has now been installed and its held-out results reproduced on CPU. The downloaded
`mbert_combined.zip` installs with `python -m training.install_checkpoint` and
is discovered automatically for both language mixes after installation.

**Model disagreement** is a separate dashboard workspace. Select Hindi-English
and compare the installed XLM-R and MuRIL checkpoints on up to 20 phrases. Models
load sequentially to limit RAM use. The page separates top-label disagreement,
score uncertainty and distribution distance; agreement is not proof of correctness.
Choose a reviewed label, declare the phrase origin and explicitly confirm it to
add a candidate to the session queue. Download the candidate CSV before closing
the session. It leaves the split unspecified and never changes training data or
model weights automatically. Keep frozen evaluation examples out of later training.
English-French comparison needs the completed mBERT export alongside XLM-R;
MuRIL is restricted to Hindi-English.

[Evaluation challenge drafts](evaluation_set/README.md) provide 100 phrases per
language and unlabelled copies. They have no normalized exact overlap with the
frozen 20,000-row source or the installed guards, but their assistant-proposed
labels are unreviewed. They are not a human benchmark. They are selectable in
Emotion lab's file evaluation tab and stay separate from all training packages.

The [submission report](Documents%20for%20review/Review%20reports/CodeMood_Hindi_English_Project_Progress_Report.docx)
has a full-page architecture and four module diagrams. Editable Mermaid sources
and SVG artwork are in [docs/submission](docs/submission/README.md). Saved results
remain synthetic-only; current live inference still makes confident mistakes.

## Install and run

Use Python 3.10 or newer and run commands from the repository root, where
`pyproject.toml` is located, rather than from the `hinglish_emotion` subfolder.
For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard]"
python -m streamlit run dashboard/app.py
```

Open the local address printed by Streamlit. Install `.[model,dashboard]` instead
if this machine will load transformer checkpoints or run training utilities.
The default demo requires no model download or cloud API. Downloading base
encoders for training does require internet access.

## Dashboard pages

- **Analyze text:** three-class sentiment, cue explanations, sarcasm/conflict
  flags, review status, normalization and baseline valence/arousal intensity.
- **Emotion lab:** seven emotion scores, a representative emotion icon,
  Hindi-English examples, checkpoint selection and dataset
  evaluation, normalization, emoji/sarcasm diagnostics and optional context-aware Qwen review. It warns about missing language validation, synthetic evaluation
  and truncated inputs. Confidence scores are not calibrated correctness odds.
- **Model effectiveness:** live sentiment comparisons, class metrics, confusion
  matrices and reliability charts for models actually available on the machine.
- **Dataset explorer:** validated CSV/XLSX uploads, filters, source table, class
  balance, sarcasm rate and message-length distributions.
- **Research workbench:** bounded experiments, confidence calibration, blind
  human review and versioned corrections, conversation traces, and GPU handoff.
- **Model comparison:** matched evaluation with aligned outputs and common scoring exclusions.
- **Model disagreement:** compare outputs and collect explicitly reviewed candidates.
- **Overview:** current encoder readiness, pipeline and evidence limits.

Recent live analyses stay in Streamlit session state and can be downloaded as
CSV. They are not automatically written to the dataset. Sentiment starts with
rules; Emotion lab selects a complete local XLM-R checkpoint when available.
Selecting an unavailable checkpoint does not train or download it.

## Research workbench

Open **Research workbench** and choose a workflow from its navigation:

- **Experiments** compares baseline, normalized and emoji-removed inputs, with
  optional context/no-context variants. Choose one compatible local model and a
  CSV or the bundled assistant-authored drafts. The row limit defaults to 10 and
  stops at 200. Optional local Qwen review is capped at five calls; suggestions
  remain separate from base scores. Meaning-changing variants report label
  stability, and accuracy requires labels declared valid for that variant. Draft
  labels are diagnostic only, not benchmark evidence.
- **Confidence calibration** fits an experimental temperature profile from a
  declared validation split or dedicated calibration file. Frozen-evaluation
  overlaps are blocked; synthetic or unreviewed labels require explicit opt-in.
  Temperature changes confidence, not the predicted emotion, and profiles are
  not applied to the installed model automatically.
- **Human review** creates two blank reviewer sheets from a queue. Use two
  different named reviewers and adjudicate disagreements or context flags. Upload
  the frozen base split file, reviewed corrections and matching audit to build a
  versioned dataset. Corrections go to train only; test rows remain protected and
  synthetic origin is retained. Exports are explicit and do not update model
  weights.
- **Conversation** shows turn-level base/reviewed labels. Earlier turns can be
  sent to local Qwen when enabled; existing encoders use context only if their
  checkpoint was trained for it. Score curves are model outputs, not a person’s
  emotional-state trajectory.
- **GPU training** links to the optional advanced XLM-R learned emoji-fusion
  workflow. Use Colab or Kaggle; no advanced weights are included. The default
  run uses synthetic emotion data only. Context and sarcasm heads stay off until
  suitable human-reviewed labels are supplied.

Start with [GET_STARTED.md](GET_STARTED.md), the
[advanced GPU guide](colab/advanced_training/START_HERE.md), or the
[23 October review report](Documents%20for%20review/Review%20reports/23OctReview%20AdvayV%20KaavishG%20CD.docx).

For optional sentiment review, install Ollama separately and run:

```powershell
ollama pull qwen3:0.6b
python -m hinglish_emotion analyze "wah kya service hai, 2 ghante late 😂" --reviewer ollama
```

The reviewer calls `http://localhost:11434` and requires Ollama to be running.

## Use the downloaded seven-emotion model

The original Hindi-English checkpoint remains under `models/xlmr_emotion`, including
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`emotion_metadata.json` and `metrics.json`. Install `.[model,dashboard]` and
start Streamlit from this project root. In the sidebar open **Emotion lab**,
choose Hindi-English, leave **XLM-R** selected, type a phrase and press
**Detect emotion**. Inference uses local files and does not need Ollama or a cloud API.

This original seven-class checkpoint is separate from the sentiment models on
**Analyze text**. Its synthetic-data warning is intentional. The first run reports 90.7%
accuracy and 0.9078 macro-F1 on 1,000 generated test phrases, but local manual
checks exposed confident mistakes on ordinary neutral, negated and elongated
phrases. Treat it as an initial experiment requiring better data and evaluation.
Weights and downloaded archives are ignored by Git; a code checkout alone does
not include the trained checkpoint.

### Combined Hindi-English and English-French XLM-R checkpoint

Install the downloaded `xlmr_combined.zip` from the project root with
`python -m training.install_checkpoint xlmr_combined.zip`. The installer places
it in `models/xlmr_multilingual_emotion` and preserves the original
`models/xlmr_emotion`. In **Emotion lab**, XLM-R automatically selects the
language-compatible checkpoint for Hindi-English (`hi-en`) and French-English
(`fr-en`). This seven-class emotion model is separate from three-class sentiment
on **Analyze text**.

The run used full fine-tuning for three epochs and selected epoch 1. Its synthetic
test accuracy was 0.925 and macro-F1 was 0.9250594 across 2,000 generated rows.
For `hi-en` (1,000 rows), accuracy and macro-F1 were 0.956 and 0.9562989; for
`fr-en` (1,000 rows), they were 0.894 and 0.8937038. All data was synthetic-only,
with no human validation. These results do not establish real-world accuracy.
GitHub omits the model weights and ZIP, so download the checkpoint and install it
locally; installing it is not new model training.

The new UI separates experimental scope from score ambiguity, clears stale
results when inputs change, and fingerprints weights/tokenizer/configuration
for cache invalidation. The Training and Colab tab displays saved run metrics
for the selected checkpoint, without presenting them as a typed phrase's accuracy.

For the two-language bootstrap and notebook, run
`python -m training.package_multilingual`. It creates the separate upload folder
and uses 8,000/1,000/1,000 synthetic rows per language. For labelled or unlabelled
file inference use `python -m hinglish_emotion predict-file your_file.csv`.
See [the base-paper comparison](training/BASE_PAPER_COMPARISON.md) for the actual
scope above Team 5 G2 and the limitations of its confidence-based intensity mapping.

## Start the 8,000-phrase Colab experiment

This is the self-contained experimental route. It generates original synthetic
Hindi-English template combinations, includes current training source and
creates an upload notebook. It does not need the unfinished MaSaC review files.

```powershell
python -m training.package_synthetic
```

If that output already exists, add `--overwrite` to refresh its known package
files. The generated dataset is reproducible with seed 42.

| Partition | Distinct phrases | Purpose |
| --- | ---: | --- |
| Training | 8,000 | Update model parameters |
| Validation | 1,000 | Select the best checkpoint |
| Synthetic test | 1,000 | Evaluate after model selection |

All seven emotions are approximately balanced. Exact duplicate phrases and
cross-split scenario groups are checked, and scenario/reaction template families
are separated across partitions. The phrases are compositional synthetic
variants, not 10,000 independent human-authored messages. Some contain contextual
emoji; emoji are not assigned from emotion labels. Sarcasm and mixed emotions
are not adequately represented by this set.

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `colab/colab_ready_8000/TRAIN_ON_COLAB.ipynb` through **File > Upload notebook**.
3. Select a GPU runtime (T4 if offered) before running the notebook.
4. Click **Run all** and upload `colab/colab_ready_8000.zip` at the first cell's prompt.
5. Approve mounting your own Google Drive when prompted.
6. Leave `MODELS = ["xlmr"]` for the first run, or choose `["xlmr", "muril"]`
   for a sequential matched comparison.
7. Download the exported inference ZIP and extract its model folder under this
   repository's `models/` directory. Select it in **Emotion lab**.

The default Drive location is `MyDrive/HinglishEmotionSynthetic8000`. The recipe
uses up to three epochs, batch size 2, gradient accumulation 8, maximum length
128 and learning rate 2e-5. Three epochs mean 24,000 training phrase exposures
per model, still only 8,000 distinct training rows.

Each completed epoch saves recovery state to Drive. After a runtime reset,
rerun setup from the first cell with the same package and training settings;
`--resume` restores the latest completed epoch. An interrupted partial epoch is
repeated. Runtime storage is temporary and GPU allocation is not guaranteed.
If `PACKAGE` is undefined, the upload/extraction cell has not been run in the
current runtime. If the GPU check fails, fix the runtime before training.

The generated folder includes `START_HERE.md`, a dataset card, an audit, file
hashes and the notebook. The folder and ZIP are intentionally ignored by Git:
regenerate them from the committed generator rather than uploading weights,
optimizer state or browser downloads. Synthetic training requires an explicit
`--allow-synthetic` opt-in, and its checkpoint retains an experimental scope.

## Human-reviewed research workflow

Use [training/WORKFLOW.md](training/WORKFLOW.md) and
[notebooks/train_emotion_pilot.ipynb](notebooks/train_emotion_pilot.ipynb) for the
separate research-data route. It requires a reviewed dataset and matching audit,
not the synthetic experiment above.

The recorded [MaSaC audit](data/audits/masac.json) has unverified reuse conditions;
export remains blocked pending applicable publisher terms. It excludes contempt
rather than relabelling it and quarantines normalized phrases appearing across
partitions. Follow [the independent annotation guide](data/annotation/README.md)
to collect and reconcile the 350-phrase supplement. Raw source data and annotation
working CSVs are kept out of Git.

Emotion training preserves case, accents, negation, expressive spelling and emoji
by default. Checkpoints record preprocessing and token limits so inference uses
the same settings. Legacy checkpoints retain their older normalized input path
and report unknown provenance. Related groups remain in one split, both models
share a frozen manifest, and changed data/settings are rejected during recovery.

Each run saves validation history, test metrics, per-class support, per-language
and per-source results, test predictions, a model card and lexical-baseline
comparisons. Inference exports omit optimizer state and raw phrases.

## Dataset formats and command-line examples

Sentiment CSVs require `text,label` with positive, neutral or negative labels.
Optional annotations include `sarcasm`, `valence`, `arousal`, `conversation_id`,
`speaker_id`, `turn_id` and `evidence`.

Emotion CSVs require `text,emotion,language_pair`. Use `hi-en` or `fr-en` and the
seven labels above. Optional `group_id`, `source`, `split`, `annotation_status`
and `human_reviewed` fields support provenance and split checks. Related
translations, paraphrases and conversations must stay in the same partition.
See [the emotion demo notes](data/demo/emotions/README.md) for label definitions.

```powershell
python -m hinglish_emotion normalize "movie acchaaa thi 😭"
python -m hinglish_emotion emotions "aaj main bahut khush hoon 😊" --language-pair hi-en
python -m hinglish_emotion evaluate-emotions data/demo/emotions/mixed_emotions_demo.csv
python -m hinglish_emotion emotions "aaj main bahut khush hoon" --model-path models/xlmr_emotion
```

The bundled sentiment workbook has 75 illustrative phrases. The mixed-emotion
CSV has 84 illustrative phrases across both language mixes. Neither is an
independent benchmark, and the emotion demo is blocked from transformer training.

For reviewed emotion data, preview the matched experiment before starting it:

```powershell
python -m training.compare_encoders --task emotion --data data/prepared/pilot_hi.csv --data-audit data/audits/pilot_hi.json --dry-run
```

Remove `--dry-run` to train after the data checks pass. Emotion comparison
normally selects XLM-R and MuRIL on Hindi-English. mBERT is optional; use
`--models mbert xlmr muril` for all three. MuRIL is excluded from French-English
experiments. Full fine-tuning should use a GPU; CPU execution needs explicit
`--allow-cpu` or `--freeze-encoder`, and may still be slow or memory-constrained.

Sentiment comparison uses the same runner with `--task sentiment` and a
sentiment CSV. The base IDs are `google-bert/bert-base-multilingual-cased`,
`google/muril-base-cased` and `FacebookAI/xlm-roberta-base`. Model folders are
separate for each task, such as `models/xlmr_sentiment` and `models/xlmr_emotion`.

## Project layout and validation

```text
dashboard/          Streamlit pages, upload handling and live evaluation
hinglish_emotion/   Normalization, inference, routing and metrics
training/           Training, data audits, annotation and Colab packaging
notebooks/          Human-reviewed pilot notebook
data/training/      Working training, validation and synthetic test CSVs
data/demo/          Small sentiment and emotion examples for software checks
data/annotation/    Human collection and review workflow
data/audits/        Public-source and annotation audits
colab/              Ready-to-upload notebook and ZIP, plus research source bundle
models/             Trained checkpoints used by local prediction
outputs/            Project documents and original training-run reports
scripts/            Dataset workbook builder
archives/           Retained model downloads, previews and historical files
Documents for review/ Papers, Word reports and editable LaTeX sources
tests/              Core, dashboard, data, packaging and recovery checks
```

```powershell
python -m unittest discover -s tests -v
```

The full suite exercises the UI, seven-class validation, leakage checks,
synthetic counts/provenance, notebook syntax, exports and a tiny offline
transformer training/resume experiment. These software checks do not validate
the accuracy of a trained research model.

Independent human-labelled evaluation, measured calibration/reviewer benefit,
controlled ablation results and an advanced GPU run remain pending. The binary
sarcasm trainer exists, but no reviewed sarcasm corpus or trained sarcasm weights
are supplied. Qwen review is uncalibrated; context goes to that reviewer, not a
learned conversation encoder. Intensity remains heuristic. See
[implementation status](IMPLEMENTATION_STATUS.md) for current limits.

## MuRIL Hindi-English study

The separate [MuRIL Colab guide](colab/muril_study/START_HERE.md) fine-tunes `google/muril-base-cased` for the seven emotion labels on synthetic Hindi-English data. Its default job uses 8,000 training, 1,000 validation and 1,000 test examples, with three epochs. It shares the test identities used by the installed combined XLM-R checkpoint, which was trained on 16,000 combined Hindi-English and English-French examples. This is an operational same-test comparison, not a controlled encoder comparison. The guide explains the optional matched 8,000-row XLM-R run and how to install either exported checkpoint.

Scores are synthetic-only and do not establish real-world performance. The saved combined XLM-R Hindi-English baseline is 95.6% accuracy and 0.9563 macro-F1. The completed MuRIL Colab report records 80.3% accuracy and 0.8123 macro-F1 on the same 1,000 Hindi-English test phrases.

Open **Emotion lab → XLM-R vs MuRIL** in the dashboard for summary scores, per-emotion F1 charts and downloadable comparison tables. With `models/muril_hi_en_emotion` installed, the dashboard checks its original training metrics against the study contract and baseline, and **Try a phrase → MuRIL → Hindi–English** runs the trained seven-emotion classifier locally. If only `muril_hi_en_comparison.zip` is present, saved comparison results remain available but live MuRIL inference requires the separate checkpoint export. Offline prediction was verified with the installed checkpoint; synthetic test scores do not guarantee correct predictions on new phrases.

Download `muril_hi_en.zip` from Colab or Google Drive under `HinglishEmotionProject/MuRILStudy/exports`. From the project folder, install it with:

```powershell
python -m training.install_checkpoint "muril_hi_en.zip"
```

The installer validates the export and creates `models/muril_hi_en_emotion`. Refresh the dashboard and choose **MuRIL** with **Hindi-English** for live predictions. The comparison ZIP alone cannot enable these predictions.

### Installed bilingual mBERT results

`models/mbert_multilingual_emotion` contains the completed seven-emotion model.
Select mBERT in Emotion lab for Hindi-English or English-French. The saved
Encoder results tab compares XLM-R and mBERT on matching bilingual partitions
and adds MuRIL for its verified Hindi-English study. Model disagreement now
compares all three encoders for Hindi-English and XLM-R/mBERT for English-French.

CPU inference reproduced all 2,000 held-out synthetic test results: mBERT
accuracy 81.4%, macro-F1 0.8119; Hindi-English 84.3% / 0.8439 and English-French
78.5% / 0.7708. Validation selected epoch 1 of three. These are generated-data
results and do not establish accuracy on natural messages; a simple happy
Hinglish phrase was confidently misclassified as sadness. Original exported
metrics and metadata snapshots are in `colab/mbert_study/completed_*.json`.
Model weights and the downloaded ZIP remain local and ignored by Git.
