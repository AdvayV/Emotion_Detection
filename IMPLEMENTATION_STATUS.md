# Implementation status

The software pipeline is connected. This does not mean all proposed models have
been trained or that research accuracy has been established.

| Feature | Working implementation | What remains |
| --- | --- | --- |
| Seven emotions | Joy, sadness, anger, fear, surprise, disgust, neutral; phrase and file prediction | Independent human-labelled evaluation |
| Model selection | Local XLM-R, mBERT and MuRIL registry; language checks; explicit demo fallback | New Colab weights for combined-language XLM-R/mBERT and Hindi-English MuRIL |
| Existing learned weights | Hindi-English synthetic XLM-R run, three epochs, best validation epoch 1 | Real-data generalization; saved synthetic test accuracy 90.7% is not a real-world estimate |
| Expressive spelling | Conservative normalization, elongation and negation cues | Controlled normalization ablation; normalized text only used when checkpoint metadata requires it |
| Emoji conflict | Word-only sentiment cues compared with emoji cues; ambiguous symbols marked uncertain | Learned multimodal fusion and annotated validation |
| Sarcasm | Explicit rule baseline; supervised XLM-R binary trainer, importer and local adapter | Reviewed sarcasm data and the Colab run; no trained sarcasm checkpoint supplied |
| Local Qwen | Optional seven-emotion structured review, routing, exact target evidence spans and graceful unavailable state | Measured reviewer benefit; calibrated confidence; no Qwen fine-tuning in this scope |
| Conversation context | Optional context passed to the Qwen reviewer | No learned temporal/context encoder |
| Intensity and evidence | Labelled heuristic intensity and lexical evidence | Human annotations; no claim of transformer attribution |
| Dataset evaluation | CSV/XLSX mapping, accuracy, macro-F1, per-class/language scores, confusion matrix, leakage/source notices | Representative unseen human test set |
| Training handoff | Separate Colab upload package, model cards, group-aware frozen splits, best validation checkpoint, test metrics, Drive recovery and inference ZIP | User-run GPU sessions and inspection of resulting metrics |
| Session history | Live analysis history in session state, CSV export, clear action | No automatic dataset writes |

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
