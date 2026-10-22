# Run and train the multilingual emotion project

## Open the application

Run commands from the personal HinglishEmotionProject folder containing pyproject.toml.

```powershell
python -m pip install -e ".[model,dashboard]"
python -m streamlit run dashboard/app.py
```

Open http://127.0.0.1:8501. The default page is **Emotion lab**.
Choose Hindi-English or English-French and a classifier. XLM-R automatically
selects a compatible installed checkpoint: the combined XLM-R model supports
both `hi-en` and `fr-en`, and the original Hindi-English checkpoint remains
available separately. Demo lexical rules are an explicitly labelled fallback,
not a replacement claimed to be trained XLM-R.

The combined checkpoint is not included in GitHub downloads. If you have the
downloaded `xlmr_combined.zip`, install it from the project root:

```powershell
python -m training.install_checkpoint xlmr_combined.zip
```

It installs to `models/xlmr_multilingual_emotion` and leaves the original
`models/xlmr_emotion` intact. Then start or refresh Streamlit and select XLM-R
in **Emotion lab**. This installs existing trained weights; it does not train a
new model. See the results and evaluation limits below.

## Predict phrases or score a file

**Try a phrase:** choose an example or type your message and click Detect emotion.
Changing the phrase, model or language clears the previous result. The score
is not an accuracy percentage. Experimental-data notices and score ambiguity
are separate signals.

**Predict and evaluate a file:** upload CSV or XLSX. Expand Map file columns
if your columns have different names. Required: a phrase/text column. Optional:
an expected emotion and language_pair. Codes hi-en and fr-en are canonical;
en-fr and common aliases are also accepted for dashboard uploads.

Labels: joy, sadness, anger, fear, surprise, disgust, neutral. Happy/happiness
and sad are accepted aliases. Positive and negative are sentiment labels and
cannot be silently converted into the seven emotions.

With labels, the app reports accuracy, seven-class macro-F1, per-language results,
per-class precision/recall and a confusion matrix. Without labels, it reports
predictions only. Download the predictions and JSON evaluation report. A declared
train/validation/test file defaults to test rows only. Training overlap, missing
classes, synthetic data and demo fallback are reported explicitly.

Small upload examples are in data/templates. Existing test data is under
data/training/synthetic_hinglish_8000/test.csv. The new combined bootstrap is
data/training/multilingual_bootstrap_v1/test.csv. These are synthetic examples,
not independent proof of accuracy on natural messages.

## Start the current mBERT comparison

1. Upload `colab/mbert_study/TRAIN_MBERT_STUDY.ipynb` to Google Colab.
2. Choose Runtime > Change runtime type > T4 GPU, then Run all.
3. Upload the sibling `mbert_study.zip` when prompted and mount your own Drive.
4. Keep the existing settings. It uses the same bilingual data/splits as combined XLM-R.
5. Download `mbert_combined.zip` when the notebook finishes, then run:

```powershell
python -m training.install_checkpoint "C:/Users/Advay/Downloads/mbert_combined.zip"
```

Refresh Emotion lab and choose mBERT. The installer uses
`models/mbert_multilingual_emotion` and preserves existing encoders.
See `colab/mbert_study/START_HERE.md` for recovery and comparison details.
The upload package is generated locally and is not committed with the weights.

## Review model disagreements

Open **Model disagreement**, choose Hindi-English and both installed encoders,
enter one phrase per line and compare. Review the outputs, select your label,
declare the phrase origin, confirm and add it to the session queue. Download
candidate corrections before closing. This does not retrain the model. For
English-French, install the trained mBERT checkpoint first; MuRIL is unavailable.

Separate challenge drafts are in `evaluation_set/`; choose them in the Emotion
lab file tab. The expected labels are unreviewed proposals. Use the unlabelled
copies for prediction only, or obtain independent reviews before reporting accuracy.

## Train the models on Google Colab

The combined XLM-R checkpoint described above is already available for local
prediction. Use this optional Colab workflow only when you want to train another
model or create a new run. Use the separate **colab/project_training** folder. You need just two files:
`PROJECT_TRAINING.ipynb` and `project_training.zip`. No model is trained on your laptop.

1. Open https://colab.research.google.com/ and choose File > Upload notebook.
   Upload `colab/project_training/PROJECT_TRAINING.ipynb`.
2. Choose Runtime > Change runtime type > GPU (T4 if offered), then Save.
3. In the notebook's second code cell, keep `JOB="xlmr_combined"` for the first run.
   Choose Runtime > Run all. When prompted, upload
   `colab/project_training/project_training.zip`, then approve your own Drive mount.
4. Let training finish. The final cell shows held-out metrics and downloads the
   inference ZIP. Drive retains the full run under `MyDrive/HinglishEmotionProject`.
   If disconnected, reconnect and rerun the same notebook with the same ZIP,
   job, data and settings; it resumes from a completed checkpoint. Work since
   the last saved checkpoint may need to run again.
5. Install the downloaded emotion ZIP from the project root:

```powershell
python -m training.install_checkpoint "C:\Users\Advay\Downloads\xlmr_combined.zip"
```

6. Refresh Streamlit, choose the language and encoder, and test a labelled file.
   Repeat in a fresh Colab session with `JOB="mbert_combined"`, then
   `JOB="muril_hi_en"`. Install each downloaded ZIP using the same installer.
   MuRIL supports Hindi-English in this project; it is not offered for French.

The installer validates the task, language metadata and classifier head and
refuses to overwrite an existing model. Newly installed compatible checkpoints
are discovered automatically. The combined XLM-R checkpoint was installed at
`models/xlmr_multilingual_emotion`; the original Hindi-only XLM-R run remains at
`models/xlmr_emotion`.

The emotion bootstrap contains **20,000 synthetic phrases: 16,000 training,
2,000 validation and 2,000 test**, split equally between Hindi-English and
English-French. MuRIL uses only the Hindi-English half. Generated labels are
not human-validated. Set `USE_OWN_EMOTION_DATA=True` to upload your reviewed
canonical CSV. A high synthetic test score does not prove accuracy on natural
messages. Free GPU availability and session duration are variable.

### Separate sarcasm training

The same notebook supports `JOB="sarcasm_hi_en"`. It asks for your own
human-reviewed CSV: `text,sarcasm,language_pair,split,group_id,human_reviewed`.
A header-only template is in `colab/project_training/package/data/sarcasm_template.csv`.
Use `sarcastic` / `not_sarcastic`, `hi-en`, explicit train/validation/test
partitions, and `human_reviewed=true`. Both classes must be present in each
partition; groups must not cross partitions and duplicate phrases are rejected.
Do not relabel positive/negative sentiment as sarcasm. No reviewed sarcasm
corpus has been provided, so this job is not ready to run until you supply one.

Install its downloaded export separately:

```powershell
python -m training.install_sarcasm_checkpoint "C:\Users\Advay\Downloads\sarcasm_hi_en.zip"
```

It installs under `models/xlmr_sarcasm`. Enable **Use trained sarcasm model** in
Emotion lab. If absent or incompatible, the interface identifies the rule-based
cue fallback. This binary model does not replace the seven-emotion model.

### How the layers work

The chosen encoder makes the primary seven-emotion prediction. Normalization,
emoji agreement/conflict, sarcasm and intensity provide separate diagnostic
signals. The encoder uses the input preparation recorded in its checkpoint;
raw-trained weights continue to receive raw text. Diagnostic normalization is
not silently applied to them. Scores in the chart always belong to the primary
classifier, even if optional Qwen review suggests a different final emotion.

To use Qwen, keep Ollama running with `qwen3:0.6b` installed and enable local
review in Emotion lab. Missing Ollama/model and invalid responses produce a
visible status while retaining the primary result. Conversation context is
used by this reviewer only. Qwen is not fine-tuned by this training notebook.
File evaluation scores the selected primary encoder so comparisons remain
consistent; it does not measure the optional reviewer's effect.

The detailed training guide is `colab/project_training/START_HERE.md`.
The earlier `colab/xlmr_multilingual` notebook remains an XLM-R-only alternative.

## Command-line file prediction

```powershell
python -m hinglish_emotion predict-file data/templates/labelled_example.csv --demo --output outputs/demo_predictions.csv
python -m hinglish_emotion predict-file data/training/synthetic_hinglish_8000/test.csv --output outputs/xlmr_test_predictions.csv --report-json outputs/xlmr_test_report.json
```

Use --allow-fallback only if you want unavailable language models to use labelled
demo rules. Each exported row identifies its actual prediction source.

## What the existing training result means

The combined multilingual XLM-R run used full fine-tuning for three epochs and
selected epoch 1. On its 2,000 synthetic test phrases, accuracy was 0.925 and
macro-F1 was 0.9250594. For `hi-en` (1,000 test phrases), accuracy was 0.956 and
macro-F1 was 0.9562989; for `fr-en` (1,000), accuracy was 0.894 and macro-F1 was
0.8937038. All examples were synthetic-only, with no human validation. These
scores do not show expected accuracy on real messages.

The installed Hindi-English XLM-R completed three epochs and selected epoch 1
by validation macro-F1. The recorded synthetic test accuracy is 90.7% on 1,000
examples; ordinary phrases can still be confidently wrong. This is primarily
a data/generalization limitation, not evidence that changing a UI label can fix
the learned model. This is the original checkpoint at `models/xlmr_emotion`,
distinct from the combined checkpoint at `models/xlmr_multilingual_emotion`.
Inspect saved run metrics on the Training and Colab tab. **Analyze text** remains
the separate three-class sentiment page; the seven labels belong to **Emotion lab**.

Team 5 - G2.docx is the base paper and remains unchanged. See
training/BASE_PAPER_COMPARISON.md for its relationship to this implementation.
