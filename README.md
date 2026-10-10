# Local Hinglish Emotion Detection

This repository is the implementation workspace for the revised English–Hinglish emotion-detection project. The first milestone is intentionally runnable without downloading any model weights. It establishes the preprocessing, output contracts, local-review routing, dataset checks, and evaluation functions that every later model must use.

## Current milestone

- Conservative Hinglish spelling normalization.
- Expressive elongation detection (`acchaaa → accha` while retaining an emphasis feature).
- Negation preservation.
- A transparent rule baseline for end-to-end smoke testing.
- A confidence router for difficult, sarcastic, or text–emoji-conflicting examples.
- Optional local Qwen review through Ollama.
- Accuracy, macro-F1, per-class metrics, calibration, spelling consistency, and sentiment-flip evaluation.
- Dataset-schema validation for sentence and conversation experiments.
- Optional conversation memory for speaker-specific emotion shifts.
- A transparent valence/arousal intensity baseline.
- Evidence masking checks for faithful explanations and counterfactual tests.

The rule baseline is not the proposed research model. It exists so that the complete software path can be tested before MuRIL is trained.

## Run locally

```powershell
python -m unittest discover -s tests -v
python -m hinglish_emotion normalize "movie acchaaa thi 😭"
python -m hinglish_emotion analyze "wah kya service hai, 2 ghante late 😂"
```

To use a locally running Qwen model through Ollama:

```powershell
ollama pull qwen3:0.6b
python -m hinglish_emotion analyze "wah kya service hai, 2 ghante late 😂" --reviewer ollama
```

No cloud API is used. The Ollama integration calls `http://localhost:11434`.

## Run the research dashboard

Install the optional dashboard dependencies once, then start the local interface:

```powershell
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

The dashboard opens with four views: an overview, interactive single-message
analysis, live model-effectiveness evaluation, and a dataset explorer. It starts
with the transparent rule baseline and can optionally use a locally saved
transformer checkpoint or local Qwen3 0.6B through Ollama. Analysis history
remains in the current browser session and can be downloaded as CSV.

The effectiveness view evaluates only models that are actually available on
the machine. It always measures the rule baseline, adds the configured local
transformer when its folder exists, and adds the routed Qwen variant when
Ollama and the selected model are ready. Comparison scores, per-class metrics,
confusion matrices, and reliability graphs are calculated from live predictions
rather than placeholder benchmark values.

## Dataset format

The minimum CSV schema is:

```text
text,label
movie acchi thi,positive
service acchi nahi thi,negative
```

Supported optional columns are:

```text
sarcasm,valence,arousal,conversation_id,speaker_id,turn_id,evidence
```

Conversation experiments must keep complete conversations in one dataset split. Counterfactual variants must also remain in the same split as their source example.

The dashboard accepts both CSV and XLSX. The bundled
`data/hinglish_emotion_phrases.xlsx` workbook contains 75 common phrases, balanced
across positive, neutral, and negative labels, with spelling, negation, emoji,
and sarcasm examples. Regenerate it deterministically with:

```powershell
python data/build_phrase_workbook.py
```

The workbook is an implementation smoke-test set, not a published benchmark.
Use a separately held-out, independently annotated dataset for research claims.

## Planned build sequence

1. Validate and profile the real project dataset.
2. Reproduce BERT and RoBERTa baselines.
3. Fine-tune MuRIL, mBERT, and XLM-R under the same split.
4. Add contextual emoji fusion and sarcasm supervision.
5. Calibrate the primary classifier and tune the local-Qwen routing threshold.
6. Add supervised intensity, faithful evidence, and conversation-memory experiments as separate ablations.

## Follow-up research modules

The next three ideas are implemented as small, testable baselines so they can
be evaluated without changing the main classifier:

```python
from hinglish_emotion.context import ConversationMemory, TurnRecord
from hinglish_emotion.intensity import IntensityEstimator
from hinglish_emotion.evidence import check_evidence
```

`ConversationMemory` compares a speaker's current prediction with their earlier
turns and marks interpretable shifts such as positive-to-negative. It keeps a
bounded history and does not mix speakers. `IntensityEstimator` exposes
valence, arousal, and cues such as elongation, punctuation, and emotional emoji;
it is a baseline for a later supervised valence/arousal head. `check_evidence`
masks the highlighted words and checks whether the prediction responds, which
provides a simple faithfulness test for model explanations.

These are deliberately separate ablations. They can be reported as added
capabilities only after the real dataset contains the corresponding
conversation, intensity, or evidence annotations.

## Train and compare mBERT, MuRIL, and XLM-R

Install the optional training dependencies from the project root:

```powershell
python -m pip install -e ".[model,dashboard]"
```

The three encoder IDs are `google-bert/bert-base-multilingual-cased`,
`google/muril-base-cased`, and `FacebookAI/xlm-roberta-base`. Supply a real
labelled CSV with `text,label` columns and at least six examples per class.
Preview the sequential training jobs first:

```powershell
python -m training.compare_encoders --data data/your_dataset.csv --freeze-encoder --dry-run
```

Remove `--dry-run` to train. `--freeze-encoder` trains only the classification
head and uses less memory on an 8 GB CPU laptop; omit it for full fine-tuning
when adequate hardware is available. Use `--models muril` for one model, or
`--local-files-only` once the base weights are cached. The first download needs
internet access, and CPU training may take a long time.

The runner uses the same normalization, seed, split, epochs, batch size, and
token limit for all three models. It verifies that the saved split manifests
match and writes `models/encoder_comparison.csv` with held-out accuracy,
macro-F1, and negative recall. Individual checkpoints and metrics are saved in
`models/mbert_sentiment`, `models/muril_sentiment`, and `models/xlmr_sentiment`.
Select a trained checkpoint from the dashboard sidebar to use it for analysis.
Untrained choices are marked as unavailable. The bundled workbook is an
implementation smoke test, not a research benchmark.

## Fine-tune MuRIL directly

Once the real labelled CSV is available, the first research model can be trained with:

```powershell
python -m training.train_transformer --data data/your_dataset.csv --model google/muril-base-cased --output models/muril_sentiment
```

To guarantee that training remains offline after the checkpoint has been downloaded, add `--local-files-only`.

Use the saved model locally with:

```powershell
python -m hinglish_emotion analyze "movie acchaaa thi 😊" --model-path models/muril_sentiment
```

The training command creates a fixed split manifest, saves the best validation checkpoint, stops early when macro-F1 no longer improves, and writes test metrics to `metrics.json`.
