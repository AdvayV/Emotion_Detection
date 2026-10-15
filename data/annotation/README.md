# Independent review of the pilot supplement

On a fresh clone, generate the blank local collection before following this guide:

```powershell
python -m training.annotations init --output-dir data/annotation
```

The command refuses to overwrite an existing collection. Working CSVs are
ignored by Git; the generator and quota definitions are committed.

The collection.csv file contains 350 blank slots. It does not contain emotion
labels or invented human annotations. Fill it with original or appropriately
sourced Hindi–English phrases. Keep public source material identifiable.

| Partition | Phrases | Target per emotion |
| --- | ---: | ---: |
| train | 140 | 20 |
| validation | 70 | 10 |
| test | 140 | 20 |

Each neighbouring pair is assigned one scenario group. You may update group IDs
to reflect real related variants, but related phrases must never cross partitions.
Include both emoji and non-emoji phrases, negation, indirect emotion, expressive
spelling and conflicting cues. Do not infer a label from an emoji alone.

## Collection and blind review

1. Fill text in collection.csv without adding emotion answers. Use one dominant
   expressed emotion: joy, sadness, anger, fear, surprise, disgust, or neutral.
2. Generate reviewer sheets after the collection is complete:

~~~powershell
python -m training.annotations review-sheets --collection data/annotation/collection.csv
~~~

3. Give reviewer_a.csv and reviewer_b.csv to two different Hindi–English readers.
   They enter a pseudonymous reviewer_id, emotion, needs_context and notes.
   They should not see each other's labels. Text, identifiers and splits stay fixed.
4. Run reconciliation:

~~~powershell
python -m training.annotations reconcile --collection data/annotation/collection.csv --reviewer-a data/annotation/reviewer_a.csv --reviewer-b data/annotation/reviewer_b.csv
~~~

If reviewers disagree or flag missing context, a third person resolves it in a
CSV with row_id, emotion, adjudicator_id, and notes. Pass --adjudications with that
file. A disputed phrase needs a clear written resolution, not an automatic vote.
If it cannot be resolved, replace it with a usable phrase and repeat review.

The report lists pending rows and class quotas. Do not change a correct label
to fill a quota: collect a suitable additional/replacement phrase instead.
The exporter checks identifiers and grouping and refuses incomplete reviews.
Distinct identifiers are a consistency check; the team is responsible for
genuinely independent readers.

Successful reconciliation exports data/prepared/pilot_supplement.csv.
The collection and review CSVs are local working data. They are not included in
the source-only cloud bundle or inference ZIP.

## French–English

Do not certify French labels without a bilingual reviewer. The engine and UI
keep French–English experimental while that reviewer and reviewed test coverage
are unavailable. The same collection/reconciliation tool can be used later with
--language-pair fr-en in a separate output directory.
