# Sentiment demo data

`hinglish_emotion_phrases.xlsx` contains 75 illustrative phrases, balanced across
positive, neutral and negative labels. The dashboard uses it for sentiment
exploration and software checks. `sample_messages.csv` is a smaller introductory
example file.

These files are not the seven-emotion training dataset. Open
`../../training/synthetic_hinglish_8000` for the 8,000 training phrases, or read
[the dataset guide](../../README.md).

Rebuild the workbook from the project root with
`python scripts/build_phrase_workbook.py`.
