# Implementation status

The software pipeline is connected. This does not mean all proposed models have
been trained or that research accuracy has been established.

| Feature | Working implementation | What remains |
| --- | --- | --- |
| Seven emotions | Joy, sadness, anger, fear, surprise, disgust, neutral; phrase and file prediction | Independent human-labelled evaluation |
| Model selection | Local XLM-R, mBERT and MuRIL registry; language checks; explicit demo fallback | Combined mBERT and XLM-R plus Hindi-English MuRIL are installed; independent evaluation remains pending |
| Existing learned weights | Original XLM-R (90.7% synthetic); combined XLM-R (92.5% overall, 95.6% hi-en, 89.4% fr-en); bilingual mBERT (81.4% overall); MuRIL hi-en (80.3%) | Real-data generalization; all recorded scores are synthetic-only |
| Expressive spelling | Conservative normalization, elongation and negation cues | Controlled normalization ablation; normalized text only used when checkpoint metadata requires it |
| Emoji conflict | Word-only sentiment cues compared with emoji cues; ambiguous symbols marked uncertain | A trained fusion checkpoint and independent emoji validation; training architecture is implemented |
| Sarcasm | Explicit rule baseline; supervised XLM-R binary trainer, importer and local adapter | Reviewed sarcasm data and the Colab run; no trained sarcasm checkpoint supplied |
| Local Qwen | Optional seven-emotion structured review, routing, exact target evidence spans and graceful unavailable state | Measured reviewer benefit; calibrated confidence; no Qwen fine-tuning in this scope |
| Conversation context | Optional context passed to the Qwen reviewer | No context-trained checkpoint installed; optional context-conditioned training/adapter code is available |
| Input/reviewer experiments | Research workbench compares raw, normalized, emoji-removed, and optional context variants; exports row-level results with latency, coverage and label stability | Independent labels for variant accuracy; measured reviewer benefit remains pending |
| Confidence calibration | Research workbench fits an explicitly experimental temperature profile from validation or dedicated calibration data; blocks frozen evaluation overlap | Human-reviewed calibration sample and transfer evaluation; profiles do not auto-deploy |
| Blind human review | Two independent blank reviewer sheets, adjudication, audit receipt and versioned training dataset export | Human reviewers and a completed annotation round; no label is inferred from model output |
| Conversation trace | Turn-by-turn base/reviewer output, bounded history and optional Qwen review | Context-trained encoder and human evaluation; current model curves are not a measure of mental state |
| Advanced learned emoji fusion | XLM-R fusion model and Colab/Kaggle training package are implemented | GPU run and exported checkpoint; default uses synthetic emotion labels, context/sarcasm heads remain off without reviewed labels |
| Intensity and evidence | Labelled heuristic intensity and lexical evidence | Human annotations; no claim of transformer attribution |
| Dataset evaluation | CSV/XLSX mapping, accuracy, macro-F1, per-class/language scores, confusion matrix, leakage/source notices | Representative unseen human test set |
| Training handoff | Separate Colab upload package, model cards, group-aware frozen splits, best validation checkpoint, test metrics, Drive recovery and inference ZIP | User-run GPU sessions and inspection of resulting metrics |
| Session history | Live analysis history in session state, CSV export, clear action | No automatic dataset writes |

## Current status

The emotion-first dashboard has eight workspaces. Overview reads checkpoint
readiness from `dashboard/readiness.py` for three encoder families and two
language profiles (`hi-en` and `fr-en`); four installed checkpoint directories
represent three families, not four different encoders. XLM-R and mBERT cover
both profiles; MuRIL covers Hindi-English only. Three-class sentiment remains a
separate legacy task. The four module flows cover live prediction, matched
comparison, Colab training and human review.

Known unsupported `trained_language_pairs` are rejected by `EmotionDetector`
for single, context and batch prediction. `evaluate-emotions` checks every
language pair before evaluation. Legacy checkpoints with unknown coverage remain
experimental and produce a warning. CLI `predict-file` permits demo fallback only
when explicitly requested with `--allow-fallback`; matched comparison never uses
fallback. Its default common overlap exclusion and aligned outputs support scoring
on the same eligible labelled rows.

Training code, installed weights and demonstrated effectiveness are separate:
the four checkpoint directories contain existing trained emotion weights, not a
new training run. Fusion, learned context and supervised sarcasm weights are not
installed, and there is no reinforcement learning. Independent human-labelled
performance and any benefit from Qwen review remain unmeasured. Current software
validation: 254 tests and 33 subtests;
software tests do not establish model effectiveness. The revised IEEE report is
16 pages after the alignment revision.

## Training jobs

- `xlmr_combined`: XLM-R, Hindi-English and English-French, seven emotions.
- `mbert_combined`: mBERT, same language/data scope for a matched comparison.
- `muril_hi_en`: MuRIL, Hindi-English only.
- `sarcasm_hi_en`: separate binary XLM-R, user-supplied reviewed data only.

The bundled emotion bootstrap is 20,000 synthetic rows: 16,000 train, 2,000
validation and 2,000 test. Each language mix contributes half. Software tests
and synthetic benchmarks do not establish research novelty or human accuracy.
The base paper predicts three sentiment classes, so its numbers cannot be
directly compared with this seven-emotion task. Use matched tasks and held-out
data to claim an improvement. See training/BASE_PAPER_COMPARISON.md.

## Files to use

Start with GET_STARTED.md. Upload the notebook and ZIP from
colab/project_training; inspect START_HERE.md in that folder. The package
contains source and data only, never downloaded model weights, optimizer
states or private credentials. Training runs on Colab; local execution is
inference, analysis and software verification.

## October 23 research-workbench update (historical snapshot)

The focused bilingual mBERT package is in colab/mbert_study. It verifies the
same data and split hashes as combined XLM-R and exports a checkpoint accepted
by the existing installer. The completed mBERT export is now installed; no local training was performed.

At that snapshot, model disagreement was a sixth dashboard workspace, with sequential local model
loading, base scores, disagreement/uncertainty separation and human-confirmed
session corrections. Exports are candidate data with an unspecified split;
there is no automatic model update. The two evaluation_set challenge files each
contain 100 unreviewed assistant-draft phrases, with separate unlabelled copies
and a zero-exact-overlap audit. Real-world benchmark validation is still pending.

Bilingual mBERT now works for both language mixes; CPU test inference reproduced 81.4% synthetic accuracy and 0.8119 macro-F1 on 2,000 rows. Encoder results and disagreement use its actual installed checkpoint. Representative human test data and reliable generalization remain open work.

The Research workbench now exposes bounded input/reviewer experiments,
experimental confidence calibration, blind two-reviewer correction export,
conversation traces and the advanced GPU handoff. The optional fusion trainer is
not evidence of a trained fusion model: no advanced checkpoint is installed.
Challenge-draft labels remain assistant-authored and unreviewed. The 23 October
review report and saved diagnostics are linked from [README.md](README.md).

Project validation at this snapshot: 221 tests and 33 subtests passed. Software
tests verify the workflows and do not establish model accuracy on human data.
