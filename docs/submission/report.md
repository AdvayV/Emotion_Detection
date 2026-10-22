# CodeMood – Sarcasm-Aware Hinglish Analysis
## Research and Implementation Report

**Course:** Compiler Design (CD)
**Team:** CodeMood – Sarcasm-Aware Hinglish Analysis
**Prepared by:** Advay Vivek (24BCT0136) and Kaavish Gogia (24BCT0103)
**Project snapshot:** 22 October 2026

> This report documents the repository implementation and saved experiments. It separates working software, completed synthetic runs and pending research. A checkpoint or dataset card describes an artifact; it does not establish human-world accuracy.

<!-- PAGE -->

## Abstract

CodeMood is a local analysis workbench for informal code-mixed text. It provides separate three-class sentiment and seven-class emotion tasks, primarily for Hindi-English (Hinglish), with an experimental English-French route. The dashboard and CLI support phrase analysis, CSV/XLSX inference, labelled-file evaluation, local checkpoint selection, optional local Qwen review and export of explicitly human-reviewed candidate corrections.

The work evolved from the team reference report, *Lexical and Emoji-Dependent Emotion Detection Using Contextual Text Reference in English-Hinglish Code-Mixed Data*. The reference describes BERT/RoBERTa, Emoji2Vec fusion and confidence-based intensity wording. The included code excerpts implement three sentiment classes and report 95.7% English, 59.5% Hinglish BERT and 57.9% Hinglish RoBERTa validation accuracy. They do not demonstrate a learned Emoji2Vec fusion network. CodeMood retains the code-mixed text and emoji motivation but accurately describes its text encoders, rule diagnostics and optional local reviewer.

An installed seven-emotion XLM-R checkpoint trained on 8,000 synthetic Hindi-English examples reports 90.7% test accuracy and 0.9078 macro-F1 on 1,000 generated phrases. A combined-language XLM-R checkpoint reports 92.5% / 0.9251 on 2,000 generated test rows: 95.6% / 0.9563 for Hindi-English and 89.4% / 0.8937 for English-French. A separate MuRIL Hindi-English study reports 80.3% / 0.8123 on the same 1,000 Hindi-English test identities. These are synthetic-only outcomes. Templates and generated language patterns, without independent human labels, limit what they support. They cannot be directly compared with the reference report’s three-class sentiment results.

The contribution is an integrated local research workflow with task separation, language and provenance warnings, data guards, artifact export/import and human review controls. Real-world validity, learned sarcasm, learned emoji fusion, calibrated confidence and measured Qwen benefit remain open work. The bilingual mBERT run is now complete and installed.

## 1. Introduction

Hinglish may combine Hindi in Latin script, English, transliteration, expressive spelling, punctuation, emoji and conversational context. “haan wah, 2 ghante late 😂” cannot safely be interpreted by positive-word counts. “khush nahi hoon” includes negation; “acchaaa” may signal elongation. Emoji may reinforce or contradict text. A hard label can hide ambiguity.

The project separates sentiment from emotion, confidence from emotional intensity, demo rules from learned prediction, and synthetic engineering tests from human validation. It is a student prototype, not an authoritative reading of a person’s feelings. Sentiment labels are positive/neutral/negative; emotion labels are joy, sadness, anger, fear, surprise, disgust and neutral. These schemas are distinct.

The engineering contribution includes modular local software, checkpoint metadata, file prediction/evaluation, split and overlap checks, saved metrics, language-aware routing, synthetic warnings, sequential model comparison, a human-review queue and exports without optimizer state or raw test phrases. It does not establish a new general theory of multilingual emotion or reproduce Emoji2Vec fusion.

<!-- PAGE -->

## 2. Reference paper and project lineage

The supplied Team 5 - G2.docx is titled *Lexical and Emoji-Dependent Emotion Detection Using Contextual Text Reference in English-Hinglish Code-Mixed Data*. Its narrative describes English/Hinglish, BERT and RoBERTa, emoji mapping, Emoji2Vec, fused vectors and five-level confidence wording. It reports 95.7% accuracy for both models on English, 59.5% BERT on Hinglish and 57.9% RoBERTa on Hinglish. These are claims from that document, not numbers reproduced here.

The code excerpts use a sequence-classification head with three labels. They do not show a learned Emoji2Vec branch or fusion layer. One five-level rule uses maximum probability at or below 0.3 as a low-confidence branch. For three softmax labels summing to one, at least one value must be at least one third, so that branch cannot occur. Confidence also does not measure emotional intensity.

CodeMood creates distinct sentiment and emotion workflows, a local checkpoint registry, file operations, validation, group-aware split manifests, hashes, metadata and scope warnings. Emoji can remain in model text; separate rules can flag word/emoji conflict. No learned emoji embedding fusion is deployed. Sarcasm has a rule baseline and binary training adapter, but no reviewed corpus or trained checkpoint. Optional Qwen is a reviewer, not a project-fine-tuned model; its suggestions have not been validated as more accurate than the encoder.

The reference and current scores differ in labels, data, origin, training and test construction. Shared use of “accuracy” does not support a comparison. That requires the same reviewed examples, labels, split identities, preprocessing and protocol.

## 3. System overview

**Figure 1. CodeMood architecture.** See [master_architecture.mmd](diagrams/master_architecture.mmd). Solid routes represent implemented work; dashed routes show optional, pending or unvalidated work.

The Streamlit app provides six workspaces: Emotion lab, Model disagreement, Analyze text, Model effectiveness, Dataset explorer and Overview. CLI operations support normalization, phrase analysis and file prediction/evaluation. Models load from local folders. Choosing an unavailable model does not train or download it: the UI warns, uses an explicitly declared demo fallback or refuses a comparison requiring trained checkpoints.

A softmax score is a model distribution over labels, not the probability an answer is correct. A separate top-score/margin heuristic flags uncertainty; it is not calibration. Disagreement is separate because models may agree and still be wrong. Sentiment has rules, optional transformer, cue explanations, conflict flags and optional Qwen review. Emotion has a lexical demo or trained seven-class model. Language pairs without verified tests are marked experimental. Context may be sent to Qwen, but no learned conversation encoder exists. Intensity and evidence are heuristic, not psychological measures or faithful neural attribution.

### Twelve functional work areas and current status

| Work area | Current state | Evidence boundary |
|---|---|---|
| Sentiment | Three-class rules and optional local model | Scores are not calibrated affect |
| Emotion | Seven-class demo and local checkpoint support | Synthetic-only results; validation pending |
| Model registry | Completeness, task and language checks | Missing weights do not trigger training |
| Text preparation | Conservative normalization, saved checkpoint modes | Negation and spelling remain difficult |
| Emoji diagnostics | Cue rules and conflict flags | No Emoji2Vec fusion network |
| Sarcasm | Narrow baseline and separate binary adapter | No reviewed corpus or trained weights |
| Qwen review | Optional local structured suggestion | Uncalibrated; benefit unmeasured |
| Context/intensity | Reviewer context and heuristic intensity | No learned conversation encoder |
| File evaluation | CSV/XLSX metrics and predictions | Results depend on supplied test data |
| Provenance | Audits, manifests, source and overlap notices | Hashes do not prove representativeness |
| Colab training | XLM-R, MuRIL and mBERT exports installed | Independent human evaluation remains pending |
| Disagreement | Sequential models and confirmed review queue | No consensus, auto-labelling or learning |

These states separate available code from completed experiments. A trainer or installer makes a run possible; neither means weights were produced. A UI control does not constitute validation evidence.


<!-- PAGE -->

## 4. Seven-emotion inference

The user enters a phrase and selects hi-en or fr-en. The detector validates the pair. A lexical demo uses a small cue list and marks its source; its score is not a calibrated probability. A trained model reads checkpoint metadata, prepares text using its saved mode/version, tokenizes to the maximum length and runs locally. Metadata records encoder, labels, input mode, max length, data/split hashes, data kind, evaluation scope and validated languages. Current synthetic checkpoints list no validated language pair: training coverage is not independent testing. This distinction affects both user experience and result reporting. A checkpoint can contain training examples from a language pair and still have no independent evidence that the classifier handles naturally written examples from that pair. Language selection is therefore an experimental scope label, not a detector that verifies the user's language. The interface depends on the user's selection and the metadata attached to the model.

The combined XLM-R model also mixes language scope within one parameter set. Its Hindi-English slice score describes predictions on that subset after training across both language pairs. It does not isolate Hindi-English-only training. Conversely, the MuRIL study is limited to Hindi-English. Both can be operationally compared on shared test identities, but an observed score gap conflates encoder choice with training scope and potentially with optimization/run differences. Reporting this plainly prevents a tempting but unsupported conclusion that one encoder is categorically better.

Output contains top emotion, seven scores, source/model, warnings, truncation and heuristic uncertainty. Optional diagnostics show cues or request review. Scores are not translated to emotion intensity. Truncation is flagged because omitted text can contain negation or context.

The original Hindi-English XLM-R reports 90.7% accuracy / 0.9078 macro-F1 on 1,000 generated test phrases. Combined XLM-R reports 92.5% / 0.9251 overall; 95.6% / 0.9563 on 1,000 generated Hindi-English examples; 89.4% / 0.8937 on 1,000 English-French examples. MuRIL reports 80.3% / 0.8123 on those Hindi-English test identities. All are synthetic only.

**Figure 2. Seven-emotion inference.** See [emotion_inference.mmd](diagrams/emotion_inference.mmd).

A cue can be negated, quoted, sarcastic or refer to someone else. Transliteration varies, emotions are subjective, and templates underrepresent mixed affect, slang, sarcasm and discourse. A model can be confidently wrong or use template shortcuts. Warnings and human inspection are central.

<!-- PAGE -->

## 5. Data, training and export

The original synthetic Hindi-English dataset has 8,000 train, 1,000 validation and 1,000 test rows. The combined bootstrap has 20,000 rows (16,000/2,000/2,000), equally divided between Hindi-English and English-French. Labels are approximately balanced. Examples are generated templates, not independent natural messages. Duplicate and scenario checks reduce obvious leakage but do not remove synthetic style.

Combined XLM-R fully fine-tunes FacebookAI/xlm-roberta-base for three epochs, selecting epoch 1 by validation macro-F1. Its contract records seed 42, split seed 42, max length 128, batch 2, accumulation 8, learning rate 2e-5, balanced weights, fp16, CUDA and synthetic-only scope. Validation macro-F1 falls after epoch 1; test remains held out.

MuRIL uses synthetic hi-en 8,000/1,000/1,000. It shares test identities with combined XLM-R’s hi-en slice, but combined XLM-R trained on both language pairs while MuRIL trained on hi-en only. This is not a controlled encoder comparison. Matched experiments must align rows, preprocessing and budget. The completed mBERT run uses the same 16,000/2,000/2,000 data and split contract as combined XLM-R. It selected epoch 1 by validation macro-F1 and reports 81.4% accuracy with 0.8119 macro-F1 on 2,000 synthetic test phrases. CPU inference from the downloaded weights reproduced these metrics exactly. Hindi-English accuracy is 84.3% with macro-F1 0.8439; English-French accuracy is 78.5% with macro-F1 0.7708.

Packaging assembles notebook, source, data, cards, audits and hashes. Colab training uses GPU and user Drive for epoch recovery. Export contains inference weights and metadata, not optimizer state or raw test phrases. Installers validate task-specific artifacts. Hashes/splits detect drift but do not make data representative. No full fine-tuning is claimed on this 8 GB CPU machine; training is intended for Colab GPU.


The learning loop starts with an already pretrained encoder, not random language-model weights. A new seven-output head is fine-tuned together with the encoder. Each mini-batch contains two tokenized phrases and their target labels. The forward pass produces seven logits; cross-entropy compares these with the supplied labels. Backpropagation computes gradients, which accumulate over eight mini-batches before an AdamW update. This gives an effective batch of sixteen phrases and normally 1,000 optimizer updates per epoch for the combined 16,000-row training partition.

An epoch is one pass through the training rows. Validation uses no gradient updates and selects the best checkpoint by macro-F1. The frozen test partition is evaluated after that selection. The mBERT notebook keeps the original three-epoch budget, 128-token limit, learning rate 0.00002, seeds 42 and fp16 precision so the comparison remains interpretable. A failed CUDA check stops training; a disconnected runtime can recover from the latest completed epoch on Drive after the setup cells are rerun. GPU availability and session length are not guaranteed by free Colab [9].

The encoders differ in pretraining rather than in their task labels. mBERT uses multilingual Wikipedia and a BERT architecture; XLM-R uses multilingual CommonCrawl and the RoBERTa training approach. MuRIL focuses on Indian languages and includes transliterated and translated pairs in pretraining. These are reasons to compare them, not guarantees about this emotion task. Their official cards describe base encoders; our saved metrics describe the task-specific checkpoints.

**Figure 3. Colab train, validate and export.** See [colab_training.mmd](diagrams/colab_training.mmd).

The separate human-data route requires review. MaSaC reuse terms are unresolved and export is blocked until verified. The adapter excludes contempt rather than relabelling and quarantines normalized cross-split duplicates. Independent annotation instructions exist; collection is incomplete.

A reviewed dataset should document who may contribute, how examples were gathered, whether text was paraphrased or translated, and what the label represents: expressed emotion, perceived emotion, or annotator interpretation. These are not interchangeable targets. Annotators need examples for neutral messages, mixed affect, negation, quoted speech, sarcasm and context dependence. Unresolvable items should have an ambiguity option or adjudication record rather than being forced into a confident class. Store both the original annotation and any adjudicated label so that later analysis can distinguish disagreement from model error.

The current synthetic generator is useful for testing the full training loop, split validation, export, installation, and dashboard display without relying on private messages. Its controlled templates also permit repeatable software regressions. However, generated variation can accidentally create shortcuts—for example, a class may share characteristic words or punctuation patterns—and compositional uniqueness does not ensure semantic independence. This makes generated data a development instrument, not a substitute for naturally authored examples.

<!-- PAGE -->

## 6. File evaluation and provenance

Prediction files require a text column. Expected emotion labels are optional, and language_pair can be mapped from a column or supplied by the selected language. Optional fields include group, source, split, annotation status and human review. Unlabelled prediction cannot report accuracy. By default, declared test rows are selected when a split column exists; users can explicitly choose a broader scope, which is labelled accordingly.

Metrics include accuracy, macro-F1, per-class precision/recall/F1, support, confusion matrix, language summaries, warnings, counts and prediction rows. Macro-F1 weights classes evenly; support matters. Small or unrepresentative files are unstable, and demo files are fixtures rather than benchmarks.

Where a checkpoint includes its matching evaluation guard or split manifest, evaluation checks normalized-text overlap with train/validation identities. This catches exact-normalization duplicates recorded in that manifest, not all paraphrase, translation, author, template or topic leakage. Define groups before splitting and freeze test identities.

Synthetic origin is recorded in cards, audits, metadata and UI. Human rows should record source, group, language, status and provenance. Dataset changes require new version/hash. Inference checkpoint exports omit raw test phrases. These practices aid traceability but do not establish annotation quality.

The evaluator also retains row-level predicted label, confidence-like score, source, truncation and experimental status. This makes it possible to inspect which records produced a summary and to segment errors after evaluation. Such record-level inspection should follow the dataset’s access rules: a downloadable evaluation file can contain sensitive text even when model weights do not. Aggregate metrics should never be presented without data scope, class counts and known overlap checks. If rows lack reliable provenance, the result should be described as a score on the provided file rather than as a benchmark.


**Figure 4. File evaluation and provenance.** See [file_evaluation.mmd](diagrams/file_evaluation.mmd).

Each result should identify task, checkpoint fingerprint, dataset version/hash, language, data origin, split/group policy, row support, preprocessing, selection rule, metric definitions and limits. Evaluation metadata identifies the exact checkpoint and file scope used for each run.

<!-- PAGE -->


A separate evaluation_set folder now contains 100 Hindi-English and 100 English-French challenge phrases, plus unlabelled prediction copies. Their proposed labels are explicitly assistant-authored drafts and human_reviewed is false. The overlap audit found zero normalized exact matches against all 20,000 frozen source rows and the four installed model guards. This rules out those particular duplicates, not semantic overlap or annotation error. These files are for inspection and subsequent independent annotation; they are not a new human benchmark and must stay outside training.

## 7. Model disagreement and human feedback

The disagreement lab accepts up to 20 phrases, a language pair and at least two compatible installed seven-emotion checkpoints. Demo rules are not votes. The shared classifier cache is cleared before comparison. Each model then loads, predicts and is released before the next model loads. Results retain checkpoint/version/source, seven scores and top label. A signature of phrase, language and versions guards against stale output.

Disagreement means different top labels. Uncertainty uses separate score/margin heuristics. Jensen-Shannon divergence summarizes distribution differences. None is accuracy; no consensus or majority label is computed. Models may share failures.

A human selects a seven-emotion label, may add notes and must confirm before adding a candidate. The session queue records group, source, note, checkpoint provenance and synthetic status, deduplicates phrase/language and leaves split blank. Only confirmed rows export. The queue does not change training data, weights, metrics or test labels.

This is not auto-learning, auto-labelling, adjudication or demonstrated active learning. Candidates require review, versioning, provenance, group-aware splits and exclusion of frozen test examples before training. Export is explicit; the queue is session state.

**Figure 5. Human-reviewed disagreement queue.** See [disagreement_review.mmd](diagrams/disagreement_review.mmd).

Optional Qwen sends text to local Ollama. Users should know that path. Exports should not become hidden conversation archives; remove identifying content where suitable and collect only under approved conditions.

<!-- PAGE -->

## 8. Diagnostics and optional reviewers

Emoji can remain in model text. Rules compare selected word/emoji cues and flag conflict; this is not Emoji2Vec fusion. Meaning varies by context and community. A controlled ablation on matched reviewed data is needed.

The sentiment baseline catches only a narrow pattern of positive wording, negative situation and laughter emoji. It is not a general sarcasm detector. A separate binary XLM-R route expects labelled data; no reviewed corpus or trained weights exist. The title signals intended scope, not demonstrated sarcasm performance.

Optional Qwen reviewers return structured suggestions with context/evidence. Routing can request review for uncertainty, truncation, negation or conflict. Original scores remain separate. The reviewer is not ground truth and has not been fine-tuned for this project; benefit is unmeasured. Context may be supplied, but no learned temporal model exists. Future study should compare no review, review-all and selective review on the same human test data.

Intensity is heuristic valence/arousal. Evidence cues are lexical matches. Normalization handles selected elongation/negation, with checkpoint metadata retaining preparation mode. Negation scope, contrast, litotes, double negatives and quotes remain risks.

<!-- PAGE -->

## 9. Evaluation summary

| Experiment | Test data | Result | Scope |
|---|---|---|---|
| Original XLM-R | 1,000 generated hi-en | 90.7%; macro-F1 0.9078 | Synthetic only |
| Combined XLM-R | 2,000 generated rows | 92.5%; macro-F1 0.9251 | Synthetic; epoch 1 selected |
| Combined XLM-R hi-en | 1,000 generated | 95.6%; macro-F1 0.9563 | Training used both languages |
| Combined XLM-R fr-en | 1,000 generated | 89.4%; macro-F1 0.8937 | No human validation |
| MuRIL hi-en | Same test identities | 80.3%; macro-F1 0.8123 | Different training scope |
| Combined mBERT | 2,000 generated rows | 81.4%; macro-F1 0.8119 | Same bilingual partitions; CPU result reproduced |
| mBERT hi-en | 1,000 generated | 84.3%; macro-F1 0.8439 | Synthetic only |
| mBERT fr-en | 1,000 generated | 78.5%; macro-F1 0.7708 | Synthetic only |

The combined XLM-R report also records a synthetic lexical baseline at 37.55% accuracy / 0.3765 macro-F1. It is an engineering reference, not a universal lower bound. Metrics are transcribed from saved artifacts and depend on those artifacts’ integrity and scope.

Manual checks found confident errors on neutral, negated and elongated phrases. Synthetic scores and high softmax do not establish robust inference. Template shortcuts, transliteration, ambiguity, sarcasm and discourse remain risks.

Software checks validate app behavior, guards, packaging, recovery, formats and checkpoint compatibility. They do not measure research accuracy. The current verification is recorded in the final validation section.

<!-- PAGE -->

## 10. Limitations

**Data:** synthetic examples have limited variety, template-intended labels and possible lexical shortcuts. Natural annotators may disagree. Demos are not benchmarks. Human collection is unfinished; public reuse terms need verification.

**Measurement:** accuracy can hide class errors; macro-F1 does not remove sampling uncertainty. Balanced generated tests may not resemble use. No confidence interval or external replication is claimed. MuRIL and XLM-R training scopes differ. Original sentiment scores use other data and labels. Fair comparison needs matched human-reviewed data.

**Model:** encoders were pretrained for language modeling, not these emotion labels. Synthetic fine-tuning does not guarantee dialect, slang, spelling, emoji pragmatics or context. Truncation may remove cues; probabilities are uncalibrated; rules miss language; Qwen may be wrong. Do not infer mental health or make consequential judgments.

**Systems:** training depends on Colab GPU, runtime, Drive, package versions and pretrained weights. Local 8 GB CPU supports app/inference, but full fine-tuning may be slow or memory-bound. Hashes/manifests help, but code revision, libraries, hardware and logs matter. No fresh training is claimed.

## 11. Ethics and responsible use

Emotion labels interpret text and do not measure internal state. Transliteration/dialect is not speaker error. High scores are not certainty, intensity or clinical significance. Use is exploratory pending human evaluation and intended-use review.

Collection context governs reuse. MaSaC export remains blocked pending terms. Predictions are never automatically queued; test data stays isolated. Future collection should explain purpose/retention, minimize personal data and obtain required consent/approval. Local inference avoids external APIs, but local Qwen receives submitted text.

<!-- PAGE -->

## 12. Twelve shared project responsibilities

These are team work areas. Available records do not verify person-by-person assignments.

1. **Problem definition:** distinguish sentiment and fine-grained emotion.
2. **Task design:** maintain independent three- and seven-class schemas.
3. **Input preparation:** preserve negation, emoji, spelling and checkpoint metadata.
4. **Data workflow:** create synthetic data, cards, audits and splits.
5. **Human data:** document source terms, review, adjudication and provenance.
6. **Model training:** support XLM-R, mBERT and MuRIL as scope allows.
7. **Experiment control:** freeze manifests, settings, recovery and hashes.
8. **Inference:** install task-specific models, check metadata, batch and flag truncation.
9. **Evaluation:** compute class/language metrics, support, confusion and overlap.
10. **Application:** deliver dashboard/CLI local phrase and file flows.
11. **Diagnostics:** expose cues, review and disagreement inspection.
12. **Reporting:** distinguish evidence from plans, protect privacy, avoid unsupported claims.

## 13. Future work

Collect representative, independently reviewed Hinglish data with documented conditions and class support. Use independent reviewers, retain disagreements, adjudicate ambiguity, group conversations/paraphrases/templates, and freeze test identities.

Repeat the matched mBERT and XLM-R experiments with additional seeds and reviewed data. Match MuRIL/XLM-R on identical partitions, preprocessing and budget. Expand language scope after human review. Analyze negation, spellings, emoji, sarcasm, length and code-mix ratio.

Run raw/normalized and emoji-retained/removed ablations. Calibrate only on human-labelled validation data. Review/version queue candidates; never train on test data. Learned sarcasm requires its own reviewed corpus; contextual learning requires conversation labels and privacy review.

A staged study would first estimate annotator agreement and revise label guidance before any model comparison. It would then publish the dataset card, source restrictions, de-identified examples, group policy and frozen test hash. A development split would support model selection; validation would support any calibration choice; test results would be run once per frozen protocol. If the test set is reused to tune a system, a new untouched test set is needed.

For encoder comparison, train XLM-R, mBERT and MuRIL only where each model supports the language pair, using identical rows, preprocessing, epochs/compute budget and selection metric. Report per-class support and confusion, language slices, run seeds, uncertainty intervals where feasible, and paired error analysis. If a model cannot cover a language, report it as unavailable rather than substituting another task checkpoint. A synthetic-to-human transfer experiment can be useful, but it should state that the test distribution is human-reviewed and compare against simple lexical baselines.

Ablations should change one factor at a time: raw versus normalized text; emoji preserved versus removed; single phrase versus reviewed context; and no reviewer versus selective Qwen review. Keep annotation and partition identities fixed. Report selective-review coverage and latency along with quality, since review may improve a small subset while leaving many cases untouched. Qualitative examples should include both successful predictions and confident failures, with enough context to interpret the case but without exposing identities.

These design choices make the future claim falsifiable: a model or component earns support only if it improves a predeclared metric on unseen reviewed examples without unacceptable class-specific regressions.

Until matched human experiments show a modeling gain, frame the contribution around reproducibility and human-controlled review. More synthetic rows or another encoder/language option alone does not prove generalization or novelty.

<!-- PAGE -->

## 14. Conclusion

CodeMood grew from a Hinglish sentiment/emoji concept into a local seven-emotion workbench: separate tasks, local inference, synthetic training packages, Colab recovery/export, provenance checks, file evaluation, diagnostics, optional review and a human-controlled queue.

Saved XLM-R, MuRIL and mBERT scores demonstrate generated-data experiments, not accuracy on natural messages. Original sentiment metrics are context, not a seven-emotion baseline. Emoji2Vec fusion, trained sarcasm, calibrated confidence and measured Qwen benefit are not claimed. The defensible next step is human-reviewed data and matched frozen-split experiments with error analysis.

## References

1. Team 5 – G2. *Lexical and Emoji-Dependent Emotion Detection Using Contextual Text Reference in English-Hinglish Code-Mixed Data*. Supplied Team 5 - G2.docx; reported figures not independently reproduced.
2. Conneau et al. “Unsupervised Cross-lingual Representation Learning at Scale.” ACL 2020. https://arxiv.org/abs/1911.02116
3. Khanuja et al. “MuRIL: Multilingual Representations for Indian Languages.” 2021. https://arxiv.org/abs/2103.10730
4. Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” NAACL-HLT 2019. https://aclanthology.org/N19-1423/
5. Liu et al. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” 2019. https://arxiv.org/abs/1907.11692
6. Eisner et al. “emoji2vec: Learning Emoji Representations from their Description.” SocialNLP 2016. https://aclanthology.org/W16-6208/
7. Encoder cards: XLM-R https://huggingface.co/FacebookAI/xlm-roberta-base ; mBERT https://huggingface.co/google-bert/bert-base-multilingual-cased ; MuRIL https://huggingface.co/google/muril-base-cased . These describe pretrained encoders, not this project’s fine-tuning.
8. Project references: README.md, IMPLEMENTATION_STATUS.md, training/BASE_PAPER_COMPARISON.md, training/WORKFLOW.md, data/README.md, colab/muril_study/xlmr_baseline_metrics.json and outputs/training artifacts.

9. Google Colab. Resource and runtime guidance. https://research.google.com/colaboratory/faq.html

## Final validation record

On 22 October 2026, the complete offline suite passed 173 tests and 33 subtests. Coverage includes routing, seven-label schemas, file mapping invalidation, dataset validation, checkpoint installation, review failures, sequential disagreement inference, session correction export and Colab packaging. The mBERT handoff also passed isolated imports using only its exported source tree. A live Streamlit test loaded the installed XLM-R and MuRIL weights sequentially and produced seven scores for each phrase without writing a review queue. Both predicted joy for Aaj main bahut khush hoon; both incorrectly predicted sadness for the fear phrase Mujhe bahut darr lag raha hai. This example shows why agreement is not correctness. Ollama was initially stopped. After starting it, a live Qwen3 0.6B call returned a valid structured review in 10.56 seconds using a lexical-demo primary prediction. It suggested neutral for the fear phrase, which was not a quality improvement. The call verifies connectivity and response validation, not reviewer accuracy; failure and unavailable states are also tested. No model was trained locally in this update.

The newly installed mBERT model was tested in Emotion lab for both language mixes. All 2,000 held-out phrases reproduced the exported accuracy and macro-F1, with zero recorded training/validation overlap. The happy phrase Aaj main bahut khush hoon was nevertheless classified as sadness with score 0.9994. Successful export and matching synthetic scores verify deployment consistency; they do not establish natural-language reliability.
