# File format examples

`labelled_example.csv` contains 14 illustrative examples, one per emotion for
each language mix. It is a small format demonstration with suggested labels,
not an independently annotated benchmark and not a training dataset.

`unlabelled_example.csv` contains text and language codes only. The dashboard
predicts labels but correctly omits accuracy and F1 because expected labels are absent.

`training_template.csv` contains only the canonical training headers. Fill it
with your own appropriately sourced, reviewed phrases before selecting
USE_OWN_DATA in the Colab notebook. Provide all seven labels in each language
and keep related paraphrases/translations/conversations in one split.

Use `hi-en` for Hindi-English and `fr-en` for English-French. For dashboard
uploads, `en-fr` is also accepted. Emotion labels are joy, sadness, anger, fear,
surprise, disgust and neutral. Positive/negative belong to the separate
sentiment task and are not automatically mapped into the seven emotions.
