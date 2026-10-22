# Optional advanced XLM-R training

1. Upload TRAIN_ADVANCED.ipynb to https://colab.research.google.com/.
2. Select Runtime > Change runtime type > GPU, then run the notebook cells.
3. Upload xlmr_advanced_training.zip from this folder at the first prompt.
4. The default experiment trains an actual learned text-emoji gate using XLM-R,
   on both Hindi-English and French-English. GPU training happens in your runtime.
5. Mount your own Drive for completed-epoch recovery. Keep EPOCHS=3 initially.
6. Inspect per-language and per-emotion support and scores. Download the inference ZIP.
7. Install from the laptop project root:
   python -m training.install_checkpoint "C:/path/to/xlmr_fusion_emotion.zip"
8. Refresh the Emotion lab and select the optional learned fusion profile.

The existing base XLM-R, mBERT and MuRIL checkpoints remain available. This package
contains source and data, no trained fusion weights. The learned profile becomes
available only after a completed GPU run is exported and installed.

The default data is the existing synthetic bootstrap: 16,000 train / 2,000
validation / 2,000 test, half from each language pair. It is not human-labelled.
Its scores measure generated examples and do not establish real-world accuracy
or improvement over the base checkpoints. Use the same independent held-out labels
and task when comparing models. No accuracy target or published-paper parity is claimed.

## Optional independently supervised tasks

Default USE_CONTEXT=False and SARCASM_COLUMN=None are intentional: the bootstrap
has no independently reviewed context or sarcasm annotations. No labels are derived
from emoji, sentiment, emotion classes or model guesses.

For your own canonical CSV, set USE_OWN_DATA=True. Required columns:
text,emotion,language_pair; labels joy,sadness,anger,fear,surprise,disgust,neutral;
language_pair hi-en or fr-en. Optional row_id, group_id, conversation_id and split
follow the existing frozen split/audit workflow. A source requiring an approved
matching data audit must include its audit JSON in the upload.

To train context conditioning, supply context and human_reviewed=true for every
row with context, plus group_id or conversation_id tying related turns together.
Set USE_CONTEXT=True. Every split must contain reviewed contextual emotion rows.
The emotion label must describe the current message with the provided context.
A corrected CSV retaining any synthetic-origin rows remains mixed/synthetic data:
set ALLOW_SYNTHETIC=True explicitly and provide its matching audit when required.
Human correction of selected rows does not make all generated rows human-labelled.
Do not paste held-out future turns into training context. Current text is the first
XLM-R tokenizer sequence and context is the second. Current text receives the
available token budget first; context receives the remaining budget. An overlong
current message is bounded safely and truncation is disclosed at inference.

To train the auxiliary sarcasm head, provide an independent sarcasm column of
0/1 or not_sarcastic/sarcastic and human_reviewed=true for its labelled rows.
Set SARCASM_COLUMN="sarcasm". Leave missing labels blank; their auxiliary loss
is masked. Both binary classes are required in train and validation. A trained
emotion model does not imply a trained sarcasm head. Context and sarcasm require
suitable real labels before they can be represented as trained capabilities.

## Kaggle alternative when Colab cannot assign a GPU

Open TRAIN_ADVANCED_KAGGLE.ipynb in Kaggle Notebooks. Upload this same training ZIP
as a private Kaggle Dataset and attach it with Add Input. In notebook Settings,
select a GPU accelerator and enable Internet to fetch the public base model and
Python dependencies. No credential API or paid service is used by the notebook.
The notebook locates the ZIP below /kaggle/input and saves the run below
/kaggle/working/CodeMood_Advanced_XLMR. Use Save Version with output persistence;
retain its model/recovery files and the inference ZIP. For a later session, attach
the saved notebook output as input, set RESTORE_RUN_FROM to that prior run folder,
and rerun the same settings. Kaggle runtime resets do not preserve unsaved working
files. Availability, account verification and quotas depend on Kaggle.

## Architecture and recovery

Text CLS plus mean learned emoji-sequence embeddings are fused through a learned
residual vector gate. Emoji sequences (including ZWJ, modifiers and flags) use
versioned SHA256 buckets, so embeddings may collide; they are trained parameters,
not hand-authored sentiment scores. Missing emoji contribution is exactly zero.
The optional binary head shares the fused representation. Emoji embeddings, gate,
classification head and encoder receive gradients during training.

Only CUDA training is permitted. --dry-run validates data, groups, frozen hashes
and supervision before importing Torch, querying CUDA or downloading a base model.
Validation macro-F1 chooses the best seven-emotion checkpoint. Test is evaluated
once after selection. A completed run returns saved results when resumed; tuning
after inspecting test requires a separately preregistered experiment and fresh holdout.

Drive recovery saves the newest fully completed epoch with model/optimizer/RNG
SafeTensors plus JSON, without pickle. A partial epoch repeats after interruption.
Rerun the notebook from the upload cell after a runtime reset, with the same data
and training settings. EPOCHS is the total target count. Keep the best checkpoint
and recovery folder together. The inference ZIP excludes optimizer state, source
training rows and recovery files. GPU allocation and runtime durations vary.
