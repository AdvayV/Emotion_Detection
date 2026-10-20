# Seven-emotion training: experimental and reviewed-data routes

## Quick experiment with 8,000 synthetic training phrases

The self-contained synthetic route is available now:

```powershell
python -m training.package_synthetic
```

Add `--overwrite` only when refreshing an existing package. The generated folder
contains `TRAIN_ON_COLAB.ipynb`, current source, 8,000 training phrases, 1,000
validation phrases and 1,000 synthetic test phrases. Upload the notebook to Colab,
select a GPU, then run all cells and supply `colab/colab_ready_8000.zip` when prompted.
Connect your own Drive to retain recovery checkpoints. Start with XLM-R; change
`MODELS` to `["xlmr", "muril"]` for sequential comparison.

This route does not use MaSaC or require its pending review files. Labels follow
authored template intent, with no independent human review. The phrases are
compositional variants, not independent naturally collected messages. Template
families and scenario groups are separated across splits, but shared vocabulary
and synthetic style limit generalization. Contextual emoji are not generated
from emotion labels. Sarcasm and mixed emotions are not adequately represented.

Synthetic training requires `--allow-synthetic`; the notebook supplies it. The
dataset card, audit, SHA256 manifest and exported checkpoint keep the synthetic
scope explicit. Synthetic test scores do not establish real-world accuracy.
The package and ZIP are ignored by Git and can be rebuilt from committed source.

The local working dataset is `data/training/synthetic_hinglish_8000`.
The notebook and upload ZIP are under `colab`. Packaging copies a verified
snapshot of the working CSVs; it does not silently regenerate existing data.
See [the dataset index](../data/README.md) for exact file roles.

If the runtime restarts, rerun setup from the first cell with the same package
and settings. `PACKAGE` is created by the upload cell. The trainer resumes from
the latest completed epoch on Drive; a partial interrupted epoch is repeated.

## Human-reviewed pilot

The remaining sections describe the separate research-data route and
`notebooks/train_emotion_pilot.ipynb`, not the synthetic upload notebook.

Training and prediction are separate. Labelled phrases train a classifier in a
cloud notebook. The saved model then predicts an emotion for a new phrase in the
local Streamlit dashboard. Uploading an unlabelled file does not train a model.

This first experiment trains XLM-R and MuRIL on the same Hindi–English data.
French–English stays experimental until a bilingual reviewer and independent
test coverage are available. No model weights for the research experiment have
been trained or downloaded by this implementation.

## 1. Read the source audit

The official Task A resource was inspected at revision
b28a3673b2561ffc7d5b11149266bf8b63cba5ab.

The checked repository contains no README, licence or reuse-terms file.
The audit therefore records reuse_conditions_verified=false and blocks CSV
export for training. Establish applicable publisher permission/terms first;
public visibility alone was not treated as approval.

The source has eight labels. The adapter excludes contempt rather than
reinterpreting it as disgust or anger. It handles emotions in train/validation
and labels in the test JSON. It quarantines 765 rows whose normalized text occurs
across partitions. Seven-class coverage remains, but the filtered corpus is not
the unchanged SemEval benchmark. The test set has only 16 disgust examples after
filtering, so report per-class support and avoid overinterpreting its score.

Episodes recur across the official conversation-segment partitions; this is not
episode-disjoint evaluation. These limitations and original class counts are
recorded in data/audits/masac.json.

Audit a local copy of the official Task A files:

~~~powershell
python -m training.audit_masac --source-dir "data/raw/EDiReF-SemEval2024/Data/Task A" --report data/audits/masac.json
~~~

After applicable reuse conditions are established, retain a document recording
them. Export with its path and an explicit attestation that they were reviewed:

~~~powershell
python -m training.audit_masac --source-dir "data/raw/EDiReF-SemEval2024/Data/Task A" --report data/audits/masac.json --output data/prepared/masac_7.csv --reuse-evidence "path/to/publisher_terms.txt" --reuse-approved
~~~

The evidence hash and exact exported CSV hash are saved in the audit.
--reuse-approved is not a substitute for actually obtaining applicable terms.
The trainer refuses a MaSaC CSV without its matching approved audit.

Source: https://github.com/LCS2-IIITD/EDiReF-SemEval2024

## 2. Complete independent annotation

Follow data/annotation/README.md. The generated collection has 140 training,
70 validation and 140 challenge-test slots, with a balanced seven-emotion target.
Two reviewers label independently; disagreement/context flags require adjudication.
The 84 demo phrases remain illustrative fixtures, not training material.

Merge the approved public subset and completed supplement:

~~~powershell
python -m training.annotations build-pilot --public-csv data/prepared/masac_7.csv --public-audit data/audits/masac.json --supplement data/prepared/pilot_supplement.csv
~~~

This produces pilot_hi.csv and its matching audit in data/prepared and
data/audits. Cross-partition groups and duplicate texts cause an error.
No automatic label replacement or silent resplitting is performed.

## 3. Use the cloud notebook

Create a source-only ZIP from the current local code:

~~~powershell
python -m training.cloud_bundle --output colab/research_source/emotion_cloud_source.zip
~~~

Open notebooks/train_emotion_pilot.ipynb in Google Colab and execute cells manually.
Upload the source ZIP, then the reviewed pilot CSV and matching audit.
Select a GPU and verify that PyTorch detects CUDA. A failed GPU check stops the
notebook before a slow full CPU fine-tuning run. Use Kaggle as an availability
fallback; GPU hardware and session quotas are not guaranteed.

Mount your own Drive and keep sufficient storage for both model weights and the
latest optimizer/RNG recovery state. Temporary runtime storage is not durable.
Nothing in this workflow upgrades your AWS account or launches an AWS resource.

The starting recipe is three epochs, batch size 2, accumulation 8, token limit
128, learning rate 2e-5 and training/split seed 42. On a supported GPU, auto
precision uses fp16. CPU experiments require fp32 and explicit --allow-cpu or
--freeze-encoder. Head-only training uses less memory but adapts fewer parameters.

The model sees minimally cleaned raw text: case, accents, negation, repeated
letters, punctuation and complete emoji sequences are preserved. Check the
tokenizer's emoji coverage in the notebook. Do not fabricate emoji from labels.

An epoch is one pass through training phrases. The classifier produces seven
scores; cross-entropy measures error against the label. Backpropagation and
AdamW update the model's parameters. Validation macro-F1 selects the checkpoint;
test labels do not select it.

## 4. Recovery and reproducibility

One frozen manifest is reused by both models. --seed controls training randomness;
--split-seed controls the initial partition, and an existing manifest wins only
after dataset and manifest hashes pass validation. Later training seeds must not
silently change the test set.

Each completed epoch saves latest_training_state.pt, including weights, optimizer,
mixed-precision scaler, Python/NumPy/Torch RNG state and loader shuffle state.
--resume restores an epoch boundary. A disconnect in the middle of an epoch
repeats that epoch. Changed data, splits, batch size, precision, preprocessing or
optimizer settings cause resume to fail. --epochs is the total target, including
already completed epochs.

Best model plus latest recovery state are retained. Early stopping uses patience
2. Do not edit an output folder's recovery state or frozen manifest by hand.

## 5. Inspect results and deploy locally

metrics.json contains measured accuracy, macro-F1, class precision/recall/F1,
support, confusion matrices, language results and the lexical baseline.
It reports public-corpus and team challenge-test results separately in
test_by_source, in addition to the pooled test result.
MODEL_CARD.md records the base encoder, task, preprocessing and fingerprints.
Never transfer an accuracy claim from a publisher's model card to this task.

XLM-R is a multilingual RoBERTa encoder pretrained with masked language modelling.
MuRIL is an Indic-focused BERT encoder exposed to transliteration and translation
pairs. Neither downloaded base checkpoint is a finished seven-emotion classifier.
The new classification head learns the labels during fine-tuning.

Export a completed inference checkpoint:

~~~powershell
python -m training.cloud_bundle --model-dir models/xlmr_emotion --output outputs/exports/xlmr_emotion.zip
~~~

Inference ZIPs exclude optimizer state and raw split/test phrases. Extract the
model folder under models in the VIT project and select it in Emotion lab.
Legacy seven-class checkpoints retain their original normalized-text behaviour
and show unknown provenance. Three-class sentiment checkpoints are rejected.

The dashboard shows all seven scores, an emotion icon and uncertainty warnings.
Scores are not calibrated correctness probabilities. Truncated input and
unverified language coverage also trigger warnings. French–English use of an
unvalidated checkpoint stays experimental; MuRIL is restricted to the matched
Hindi–English experiment in this interface.

## Subsequent experiments

After the trained pilot and credible data are available:

1. Add reviewed French–English data and train a separate multilingual XLM-R run.
2. Compare raw spelling with normalization and emoji retained with emoji removed
   using the same frozen groups and clear annotation conditions.
3. Calibrate confidence on validation data, then compare seven-emotion Qwen
   review of all cases with selective review, reporting accuracy and latency.

Conversation context, learned sarcasm supervision and faithful word evidence need
their own annotations and controlled experiments. The existing sentiment Qwen
reviewer is not silently reused for seven emotions.
