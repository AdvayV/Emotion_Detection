# mBERT combined Colab study

1. In a fresh clone, regenerate the deterministic multilingual bootstrap data from the project root: `python -c "from training.multilingual_data import save_multilingual; save_multilingual('data/training/multilingual_bootstrap_v1')"`.
2. Rebuild this handoff with `python -m training.package_mbert_study`. The builder uses installed XLM-R metrics and metadata when available, or verifies the small snapshots in this folder against `study_contract.json` when model files are absent.
3. Upload `TRAIN_MBERT_STUDY.ipynb` to Google Colab, select **Runtime > Change runtime type > T4 GPU**, and run all cells.
4. Upload `mbert_study.zip` at the first prompt. The notebook path-checks extraction, verifies every file hash, installs the project dependencies, and confirms CUDA is enabled.
5. The default job fine-tunes mBERT on the existing combined Hindi-English and English-French seven-emotion dataset. It requires the exact frozen 16,000 train / 2,000 validation / 2,000 test split and SHA-256 hashes recorded by the XLM-R baseline.
6. Approve the Google Drive mount. Frozen split, model checkpoint and resume state save under `MyDrive/HinglishEmotionProject/MBERTStudy/mbert_combined`. Reopen the same notebook and upload the same ZIP to resume after a runtime disconnect.
7. The last notebook cell compares validation-selected mBERT against the combined XLM-R baseline on the same held-out synthetic test examples, then downloads `mbert_combined.zip`. From the project root, install it with `python -m training.install_checkpoint "C:/path/to/mbert_combined.zip"`.

The phrases are synthetic and template-based. The test split is shared intentionally with XLM-R; it is not an independent evaluation. The project contains no separate human-labelled evaluation set. The dataset audit reports zero cross-split group and template-family overlap. Report the synthetic limitation whenever sharing scores. Do not tune hyperparameters against test results.

## Completed run

The exported mBERT checkpoint is installed locally. Saved synthetic test accuracy is 81.4% overall, 84.3% Hindi-English and 78.5% English-French. The full 2,000-row result was reproduced using CPU inference. `completed_metrics.json` and `completed_metadata.json` preserve the original run evidence without weights. Validation selected epoch 1; later epochs did not improve validation macro-F1. These results are not a human benchmark.
