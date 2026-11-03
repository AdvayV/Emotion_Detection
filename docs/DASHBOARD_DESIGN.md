# CodeMood dashboard design

The dashboard remains a Streamlit application. It uses native buttons, selectors,
tabs and downloads; decorative HTML contains escaped text and no executable
controls. The redesign changes presentation and information hierarchy, not model
weights, probabilities, comparison eligibility or evaluation metrics.

## Main workflows

- **Emotion lab:** language and checkpoint selection first; phrase input and
  prediction side by side on wide screens, stacked on narrow screens. Context,
  Qwen, sarcasm and calibration controls stay available in an expander. Editing
  the phrase or settings invalidates its previous result. Demo results remain
  labelled, and original base scores remain visible beside any reviewed label.
- **Model comparison:** select language and compatible models, supply a dataset,
  inspect the scoring safeguards, and run the existing sequential comparison.
  Accuracy cards supplement macro-F1, per-emotion scores and paired mistakes.
  They use the same eligible labelled rows as the existing report. Missing
  labels never produce invented scores, and MuRIL remains Hindi-English only.
- **Overview:** direct actions open prediction, model comparison or the dataset
  explorer. Checkpoint readiness continues to come from installed metadata.
- **Other workspaces:** share typography, charts, page headers and navigation,
  retaining their original controls and session-state behavior.

## Visual decisions

Porcelain backgrounds and white surfaces separate tasks without strong visual
noise. A dark indigo header supplies hierarchy; purple identifies primary
actions, and seven consistent emotion colors support chart reading. Text and
numbers accompany color cues. Keyboard focus outlines remain visible, navigation
is not hidden, and reduced-motion preferences disable decorative transitions.
The interface uses the existing system/Streamlit fonts, with no external font
or analytics service added.

## Verification

The complete local regression run passed **261 tests and 33 subtests** on
3 November 2026. These include prediction-source labels, optional controls,
settings persistence, stale-result invalidation, file evaluation, comparison
safeguards and existing model functionality. Run from the repository root:

```powershell
python -m pytest tests -q
python -m streamlit run dashboard/app.py
```

Browser acceptance checks cover a 1440-pixel desktop and a 390-pixel phone
viewport, example selection, a demo prediction, clearing stale scores after an
input edit, comparison readiness, Overview navigation and horizontal overflow.
These checks validate the interface; they are not new model-accuracy experiments.
