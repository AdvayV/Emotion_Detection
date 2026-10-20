# Mixed-language emotion data

This folder is for interface examples, not training. The current dashboard uses
the Hindi-English rows only; the retained French-English rows are for future work.
The actual 8,000 training phrases are under
`../../training/synthetic_hinglish_8000`; see [the dataset guide](../../README.md).

The mixed_emotions_demo.csv file is an assistant-drafted illustrative dataset
for testing the software interface. It contains 84 phrases, evenly split across
Hindi–English and French–English and across seven labels. It has not been
independently annotated and is not a research benchmark. Lexical rules overlap
with many clear examples, so a high demo score would not establish generalization.

The group_id links the two language versions of each situation.
Source=illustrative and split=demo make its provenance explicit. Transformer
training refuses this demo file. Live predictions are downloaded on request,
not saved over the dataset.

## Labelling real phrases

Use one dominant emotion per phrase:

| Label | Meaning |
| --- | --- |
| joy | Happiness, delight or enjoyment |
| sadness | Grief, disappointment or sorrow |
| anger | Irritation, hostility or rage |
| fear | Fright, apprehension or feeling unsafe |
| surprise | Astonishment or an unexpected event; may be positive or negative |
| disgust | Revulsion or strong aversion |
| neutral | No clearly expressed emotion |

Read the intended meaning of the full phrase. Do not label it from an isolated
word or emoji. Negating happy does not automatically establish sadness.
Sarcastic praise may express anger, disappointment, or something else; ambiguous
phrases need context and annotator review. These labels describe expressed
emotion, not a person's psychological state.

Collect realistic code mixing, not just monolingual French and Hindi sentences.
Keep accents, original spellings, emoji and negations. Ask annotators familiar
with each language mix to label independently and resolve disagreements.
Mark uncertainty separately; do not convert all uncertain examples to neutral.

Use the same definitions for both language mixes and ensure each emotion has
enough independently authored examples in each test set. A few examples per
class only demonstrate functionality. Report sample counts alongside accuracy,
macro-F1, per-emotion recall and the confusion matrix.

Use group_id for translations and paraphrases, or conversation_id for turns.
Keep complete groups in one split. The trainer also groups identical normalized
texts. Existing explicit splits are checked for overlapping groups. Never tune
rules, prompts or checkpoints on the held-out test set.

## Sources to investigate

- [XED, Helsinki-NLP](https://github.com/Helsinki-NLP/XED) provides multilingual
  emotion data. Its original labels include anticipation and trust and may be
  multi-label, so it requires an explicit selection/mapping protocol. It is not
  itself a Hindi–English/French–English code-mixed seven-class benchmark.
- [SemEval 2024 Hinglish emotion study](https://aclanthology.org/2024.semeval-1.93/)
  describes a Hindi–English task with eight labels including contempt. Its
  taxonomy differs from this project; do not silently merge contempt into disgust.
- [French-emotion dataset card](https://huggingface.co/datasets/JusteLeo/French-emotion)
  describes seven French labels but explicitly reports synthetic generation.
  It is monolingual and cannot by itself validate real French–English code mixing.

Keep synthetic or translated training material identifiable, and reserve
independent human-labelled mixed-language phrases for final evaluation.

[mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased)
and [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base) are multilingual
encoder comparisons. [MuRIL's model card](https://huggingface.co/google/muril-base-cased)
states its Indic-language focus and limitations outside its pretraining languages.
