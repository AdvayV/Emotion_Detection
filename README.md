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
ollama run qwen3:4b
python -m hinglish_emotion analyze "wah kya service hai, 2 ghante late 😂" --reviewer ollama
```

No cloud API is used. The Ollama integration calls `http://localhost:11434`.

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

## Fine-tune MuRIL

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
