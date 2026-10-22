# Multilingual emotion evaluation challenge drafts

This folder contains two 100-row draft files and unlabelled copies for prediction:

- `hindi_english_100.csv` and `english_french_100.csv` include an assistant-proposed `emotion` label.
- `hindi_english_100_unlabelled.csv` and `english_french_100_unlabelled.csv` keep the same text with a blank `emotion` column.

The phrases were authored for this challenge draft and include code-switching, negation, spelling variation, emoji, conflicting cues, indirect emotion, and neutral factual messages. The label column is a proposal, not ground truth. Every row has `annotation_status=assistant_draft`, `source=assistant_draft`, and `human_reviewed=false`. Do not report these files as a human evaluation benchmark or use draft labels to make performance claims. Have independent annotators review and adjudicate them first.

`overlap_audit.json` records row counts and SHA-256 hashes and checks normalized exact-phrase overlap against all 20,000 rows in the frozen multilingual training CSV and every installed model's `evaluation_guard.json`. The current audit requires zero overlap. Zero exact normalized overlap does not prove independent authorship, statistical independence, translation independence, or human ground truth. Do not add these phrases to training data.
