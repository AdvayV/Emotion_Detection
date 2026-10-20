# XLM-R training handoff

1. Open https://colab.research.google.com/ and upload TRAIN_XLMR.ipynb.
2. Runtime > Change runtime type > GPU (T4 if available). Run all.
3. At the upload prompt choose xlmr_multilingual.zip from this same folder.
4. Leave MODE="combined" for one model supporting both language mixes. Use
   MODE="separate" only if you want two independent XLM-R runs.
5. Approve your own Drive mount. Training and recovery state save under
   MyDrive/CodeMood_XLMR_v2. Free GPU availability and session length vary.
6. Inspect per-language/per-emotion results. Download the inference ZIP(s).
7. On the laptop, from HinglishEmotionProject, run:
   python -m training.install_checkpoint "C:/path/to/downloaded/model.zip"
8. Open http://127.0.0.1:8501, select Emotion lab, XLM-R, and the language.
   The app discovers language-compatible checkpoints. Existing models are preserved.

## Dataset and limitations

The supplied bootstrap is synthetic: 8,000 training, 1,000 validation and 1,000
synthetic test phrases per language mix. Combined: 16,000/2,000/2,000. It is not
independently human-labelled and cannot establish real-world or published-paper
accuracy. The existing Hinglish examples are retained; French-English examples
are authored template compositions. No MaSaC corpus is included.

Working CSVs: data/training/multilingual_bootstrap_v1 from the project root.
Use all_splits.csv for training. Language codes: hi-en and fr-en (UI also accepts
en-fr uploads). Seven labels: joy, sadness, anger, fear, surprise, disgust, neutral.
USE_OWN_DATA=True accepts a canonical reviewed CSV and optional source audit.
Do not automatically relabel positive/negative sentiment as the seven emotions.

The package folder is a source/data snapshot for reproducibility; upload only
the notebook and ZIP. The ZIP contains no model weights, credentials or Drive data.

## Recovery and troubleshooting

After a reset, rerun from the first cell and upload the same package again.
PACKAGE undefined means the upload cell has not run. CUDA unavailable means the
runtime has no visible GPU. A partial epoch is repeated after disconnection.
Resume requires unchanged data and settings. Increasing EPOCHS sets the total,
not additional epochs. Keep both best checkpoint and latest epoch state on Drive.

Unlabelled CSV/XLSX files can be predicted in the dashboard, but accuracy needs
expected labels. Test rows are selected by default when a split column exists.
The model may confidently misclassify everyday phrases; training data diversity
and independently reviewed evaluation are still required. Optional Qwen review
is available for phrases; file evaluation scores the primary emotion model.
