# MuRIL Hindi-English Colab study

This notebook fine-tunes `google/muril-base-cased` with a seven-emotion classification head: joy, sadness, anger, fear, surprise, disgust and neutral. MuRIL is used for Hindi-English; there is no French MuRIL run.

## Run it in Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File > Upload notebook** and upload `colab/muril_study/TRAIN_MURIL_STUDY.ipynb` from this project.
3. In Colab, choose **Runtime > Change runtime type**, select **T4 GPU** if available, and save.
4. Choose **Runtime > Run all**.
5. At the first upload prompt, upload `muril_study.zip`.
6. Approve access to your own Google Drive when prompted. The notebook saves checkpoints under `MyDrive/HinglishEmotionProject/MuRILStudy/<JOB>` so training can resume after a runtime reset.

The default settings are `JOB='muril_hi_en'` and `EPOCHS=3`. The package has 8,000 synthetic Hindi-English training rows, 1,000 validation rows and 1,000 test rows. Its test identities match those used by the installed combined XLM-R run. That XLM-R run trained on 16,000 combined Hindi-English and English-French rows, while this MuRIL run trains on 8,000 Hindi-English rows, so comparing these two scores is not a controlled encoder comparison.

For an optional 8,000-row Hindi-English XLM-R comparison, change `JOB='muril_hi_en'` to `JOB='xlmr_hi_en_matched'` in the notebook configuration cell, then run all cells again. This creates a separate job and does not overwrite the combined-language XLM-R run.

## Downloads and local use

The notebook stores the run at `MyDrive/HinglishEmotionProject/MuRILStudy/<JOB>` and exports two files under `MyDrive/HinglishEmotionProject/MuRILStudy/exports/`:

- `<JOB>.zip`: inference checkpoint, installed locally as `models/muril_hi_en_emotion` for the default MuRIL job.
- `<JOB>_comparison.zip`: run report, including per-class metrics and accuracy/macro-F1 SVG charts.

The export downloader downloads both artifacts. From the project root in Windows PowerShell, install the default checkpoint:

```powershell
python -m training.install_checkpoint "C:/Users/Advay/Downloads/muril_hi_en.zip"
```

For the optional matched XLM-R export, install its downloaded archive; it is placed at `models/xlmr_hi_en_emotion`. The installed combined checkpoint remains separate.

Then start the dashboard from the project root:

```powershell
python -m streamlit run dashboard/app.py
```

Open **Emotion lab**, select **MuRIL**, and use the new **XLM-R vs MuRIL** tab to view saved comparison metrics. The accuracy and macro-F1 values appear only after the run completes; no MuRIL score exists beforehand.

## Resume and interpret results

If Colab disconnects, reopen the same notebook, upload the same `muril_study.zip` at the first prompt, keep the same `JOB`, authorize Drive again if asked, and choose **Runtime > Run all**. Drive recovery files let the notebook resume the job.

Accuracy is the share of test phrases classified correctly overall. Macro-F1 calculates precision and recall for each emotion, combines them into an F1 score, then gives every emotion equal weight. Per-class results show precision, recall, F1 and support for each label; support is the number of test examples for that label.

The data is synthetic, so these scores measure performance on generated phrases only. They are not validation on real-world conversations and do not establish how the model will perform in use. The saved combined XLM-R Hindi-English baseline is 95.6% accuracy and 0.9563 macro-F1; that is an existing synthetic-run result, not a MuRIL estimate.
