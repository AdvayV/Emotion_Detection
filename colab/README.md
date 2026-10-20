# Files to upload to Google Colab

**Current full workflow:** open [project_training/START_HERE.md](project_training/START_HERE.md).
Upload [PROJECT_TRAINING.ipynb](project_training/PROJECT_TRAINING.ipynb), then
[project_training.zip](project_training/project_training.zip) when prompted.
The same notebook offers combined-language XLM-R and mBERT, Hindi-English
MuRIL, and a separate reviewed-data sarcasm job. Run one job per GPU session.
All full training and resume state stays in your Google Drive. The final cell
shows metrics and downloads an inference-only ZIP for safe local installation.

The bundled emotion data is synthetic. The sarcasm template contains headers
only; that job needs your own human-reviewed data. See ../GET_STARTED.md for
the complete sequence and ../IMPLEMENTATION_STATUS.md for remaining experiments.

The [xlmr_multilingual](xlmr_multilingual/START_HERE.md) folder remains an
alternative XLM-R-only workflow with shared or separate language models.

## Earlier Hindi-only experiment

Use these two files for the current 8,000-phrase synthetic experiment:

1. Open Colab and upload
   **[colab_ready_8000/TRAIN_ON_COLAB.ipynb](colab_ready_8000/TRAIN_ON_COLAB.ipynb)**
   as the notebook.
2. Select a GPU runtime. Run the first cell and upload
   **[colab_ready_8000.zip](colab_ready_8000.zip)** when prompted.

The ZIP already contains the source and a snapshot of the dataset. You do not
need to upload individual training, validation or test CSVs separately.
See [START_HERE.md](colab_ready_8000/START_HERE.md) for the full sequence.

The local working CSVs are under
`../data/training/synthetic_hinglish_8000`. To refresh the upload after source
changes, run this from the project root:

```powershell
python -m training.package_synthetic --overwrite
```

The builder verifies the working dataset audit and copies the same partitions.
An incomplete or inconsistent dataset stops packaging instead of being replaced.
Generated labels remain experimental and need human evaluation.

`research_source/emotion_cloud_source.zip` is a **different** source-only bundle
for `../notebooks/train_emotion_pilot.ipynb`. That route still requires separately
reviewed research data and an approved audit; it is not the current ready-to-train
synthetic package.

The obsolete upload ZIP and original Colab downloads were removed with approval;
`../archive/cleanup_manifest.json` records the cleanup. The current files above
remain available. Inference weights are installed under `../models`.
