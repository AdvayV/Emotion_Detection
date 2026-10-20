# START HERE

1. Upload `PROJECT_TRAINING.ipynb` to Google Colab and choose Runtime > Change runtime type > GPU.
2. Run the first cell and upload `project_training.zip` when prompted. Extraction is path-checked and every packaged file is SHA-256 checked before install. No weights are in this ZIP.
3. Keep the default `JOB="xlmr_combined"` for one XLM-R seven-emotion run across Hindi-English and English-French; choose `mbert_combined` for mBERT, `muril_hi_en` for Hindi-English MuRIL, or `sarcasm_hi_en` for a separate supervised binary run. Run one job per Colab session.
4. Approve Drive mount. Frozen splits, training states and checkpoints save under `MyDrive/HinglishEmotionProject`; rerun the same job to resume. Do not change data or settings when resuming.
5. Sarcasm requires your own labelled dataset. Template: `package/data/sarcasm_template.csv`; rows need human review, explicit independent train/validation/test split, unique group_id and both classes in each split. No labels or corpus are provided or generated.
6. Model cards: https://huggingface.co/FacebookAI/xlm-roberta-base ; https://huggingface.co/google-bert/bert-base-multilingual-cased ; https://huggingface.co/google/muril-base-cased

Emotion bootstrap data is synthetic (see `package/data/DATASET_CARD.md`) and does not demonstrate human-world accuracy. `training/train_transformer.py` exports safetensors and emits task metadata. Inspect validation selection and test metrics before any use.


7. After training, download the best inference export. Install an emotion export with `python -m training.install_checkpoint "C:/path/to/xlmr_combined.zip"`; it is placed in a distinct model folder. Install a sarcasm export with `python -m training.install_sarcasm_checkpoint "C:/path/to/sarcasm_hi_en.zip"`; it uses its own separate sarcasm model folder and does not replace an emotion model.