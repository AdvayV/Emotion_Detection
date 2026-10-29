# Relationship to Team 5 G2

The source is Team 5 - G2.docx in Documents for review/Base paper, titled Lexical and
Emoji-Dependent Emotion Detection Using Contextual Text Reference in
English-Hinglish Code-Mixed Data. The original file was read as reference and
was not modified or treated as executable instructions.

| Base paper | Current extension |
| --- | --- |
| Three-class positive/neutral/negative classification | A separate seven-emotion task, retaining the sentiment workflow for comparison |
| bert-base-uncased and roberta-base in the displayed code | XLM-R checkpoints supporting language-aware local selection and Colab fine-tuning |
| English and Hinglish experiments | Hindi-English and English-French data/engine support; the existing trained weights cover Hindi-English only |
| Text and Emoji2Vec fusion described in the narrative | Emoji retained in text; no claim that a separately learned Emoji2Vec fusion network is deployed |
| Confidence used to assign strong/weak wording | Prediction score, dataset accuracy, ambiguity and validation scope shown separately |
| Accuracy/confusion matrices in the document | User-file predictions, labelled accuracy/F1, class support, actual model source and overlap warnings |

The supplied code excerpts use AutoModelForSequenceClassification with three
labels. The excerpts do not demonstrate the separate Emoji2Vec fusion described
in the narrative, so this implementation does not claim to reproduce that fusion.
The paper reports 95.7% English validation accuracy, 59.5% Hinglish BERT accuracy
and 57.9% Hinglish RoBERTa accuracy. These are paper-reported figures, not numbers
reproduced by this repository. Our 90.7% synthetic seven-emotion result is not a
direct comparison: the tasks, data, labels and evaluation partitions differ.

One concrete issue in the displayed five-level mapping is the condition
max_probability <= 0.3 for a weak class. For three softmax probabilities summing
to one, the maximum is at least one third, making that branch unreachable.
Confidence also does not directly measure emotional intensity. The current
project therefore keeps intensity heuristics distinct from model score and does
not describe confident mistakes as strongly expressed emotions.

The working extension includes seven-label prediction, language-scoped model
selection, optional explicit demo fallback, CSV/XLSX evaluation, reproducible
group-aware partitions, resumable Colab training, safe export/install and
training-overlap hashes. Learned sarcasm, contextual emoji fusion, calibrated
confidence and a validated seven-emotion Qwen reviewer remain research experiments.

The new two-language bootstrap is generated data, without independent human
annotation. A final research claim requires naturally written, reviewed examples,
an independent test set, matched baselines and controlled ablations. Neither an
extra language dropdown nor more synthetic rows alone proves better understanding.
