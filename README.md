# Hinglish Emotion Detection

A local Streamlit application for exploring sentiment and seven emotions in
mixed-language phrases, with optional transformer checkpoints and local Qwen
review. Training runs separately in a GPU notebook; the dashboard loads the
resulting checkpoint for prediction.

## What works today

| Capability | Implementation and current limits |
| --- | --- |
| Three-class sentiment | Positive, neutral and negative; always-available rule baseline and optional local transformer |
| Seven-emotion detection | Joy (happy), sadness, anger, fear, surprise, disgust and neutral; demo rules or a trained seven-class checkpoint |
| Language mixes | Hindi-English and experimental French-English; the matched XLM-R/MuRIL training experiment uses Hindi-English |
| Expressive spelling | Conservative normalization, negation preservation and elongation cues such as `acchaaa` |
| Emoji and sarcasm | Rule-based conflict/sarcasm cues and review routing in the sentiment pipeline; not a trained sarcasm classifier |
| Local Qwen | Optional Ollama `qwen3:0.6b` review of routed three-class sentiment cases; not a seven-emotion reviewer |
| Research utilities | Dataset validation, class/per-language metrics, confusion matrices, calibration diagnostics, intensity, conversation memory and evidence-masking baselines |
| Model training | mBERT, MuRIL and XLM-R runners, fixed group-aware splits, checkpoint recovery and inference export |

The repository contains software, illustrative data and a synthetic-data
experiment. It does not ship trained model weights or claim verified research
accuracy. Base encoders are not finished emotion classifiers. Synthetic and demo
scores do not establish performance on real user messages. French-English
remains experimental until independently reviewed test coverage is available.

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
  Hindi-English/French-English examples, checkpoint selection and dataset
  evaluation. It warns about missing language validation, synthetic evaluation
  and truncated inputs. Confidence scores are not calibrated correctness odds.
- **Model effectiveness:** live sentiment comparisons, class metrics, confusion
  matrices and reliability charts for models actually available on the machine.
- **Dataset explorer:** validated CSV/XLSX uploads, filters, source table, class
  balance, sarcasm rate and message-length distributions.
- **Overview:** current pipeline, proposed improvements and the distinction
  between implemented baselines and experiments requiring training.

Recent live analyses stay in Streamlit session state and can be downloaded as
CSV. They are not automatically written to the dataset. The app starts with
rules; selecting an unavailable checkpoint does not train or download it.

For optional sentiment review, install Ollama separately and run:

```powershell
ollama pull qwen3:0.6b
python -m hinglish_emotion analyze "wah kya service hai, 2 ghante late 😂" --reviewer ollama
```

The reviewer calls `http://localhost:11434` and requires Ollama to be running.

## Start the 8,000-phrase Colab experiment

This is the self-contained experimental route. It generates original synthetic
Hindi-English template combinations, includes current training source and
creates an upload notebook. It does not need the unfinished MaSaC review files.

```powershell
python -m training.package_synthetic --output-dir colab_ready_8000
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
2. Upload `colab_ready_8000/TRAIN_ON_COLAB.ipynb` through **File > Upload notebook**.
3. Select a GPU runtime (T4 if offered) before running the notebook.
4. Click **Run all** and upload `colab_ready_8000.zip` at the first cell's prompt.
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
See [data/MIXED_EMOTIONS.md](data/MIXED_EMOTIONS.md) for label definitions.

```powershell
python -m hinglish_emotion normalize "movie acchaaa thi 😭"
python -m hinglish_emotion emotions "aaj main bahut khush hoon 😊" --language-pair hi-en
python -m hinglish_emotion evaluate-emotions data/mixed_emotions_demo.csv
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
data/               Illustrative datasets, annotation instructions and audit
outputs/            Project proposal and revision documents
tests/              Core, dashboard, data, packaging and recovery checks
```

```powershell
python -m unittest discover -s tests -v
```

The full suite exercises the UI, seven-class validation, leakage checks,
synthetic counts/provenance, notebook syntax, exports and a tiny offline
transformer training/resume experiment. These software checks do not validate
the accuracy of a trained research model.

The next research steps are independent real-data evaluation, matched encoder
comparisons and controlled emoji/normalization ablations. Learned sarcasm,
calibrated seven-emotion Qwen review, supervised intensity and conversation
context still need suitable annotations and experiments.
