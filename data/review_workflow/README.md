# Human review and GPU retraining handoff

This workflow creates blind CSV sheets from an explicitly uploaded disagreement queue or challenge draft. It keeps proposed/model labels in a separate file. Both reviewers independently enter a reviewer ID and one of the seven canonical emotion labels. Label conflicts and any `needs_context` flag require a third reviewer, a canonical decision, and written notes. Incomplete or changed rows cannot be exported.

Set `dataset_role=train` for correction candidates. Set it to `test` only for independent evaluation examples; these rows remain evaluation only and are never merged into training data. Existing source and synthetic-origin provenance stay attached after review, so reviewed generated text remains marked experimental. Human review records annotation status but does not establish that an assistant-authored phrase is natural or representative.

Use the CLI:

```powershell
python -m training.review_workflow create queue.csv review_sheets
python -m training.review_workflow reconcile queue.csv review_sheets/reviewer_a.csv review_sheets/reviewer_b.csv --adjudications adjudications.csv --output reviewed.csv --audit audit.json
python -m training.review_workflow build-corrections base_all_splits.csv reviewed.csv versioned_dataset --audit audit.json
```

The UI function is `dashboard.review_panel.render_review_panel(root, style_chart)`. It accepts uploads, creates downloadable blank sheets, accepts completed reviewer/adjudication sheets, and offers a reviewed-only ZIP. It does not silently save a dataset into the project.

Assistant drafts, illustrative rows, rows marked as evaluation data, and content matching the project's evaluation set default to `dataset_role=test`; explicitly verify the role before creating sheets. Reviewed output uses a compatible `synthetic_human_reviewed` source for synthetic-origin material while preserving `original_source`, `original_annotation_status`, and `synthetic_origin` fields. The two-reviewer consensus status is `human_reviewed`; disagreement/context resolution is `human_adjudicated`.

The versioned dataset export requires a completed audit bound to the exact reconciled CSV and reviewer sheet hashes. It adds eligible corrections to `train` and carries forward base `validation` and `test` rows. It rejects normalized-text duplicates, overlap with any base or independent evaluation phrase, row collisions, and group overlap with frozen partitions while allowing multiple rows in the same new training group. Export contains `all_splits.csv`, `audit.json`, a dataset card, and GPU handoff notes. The audit includes `prepared_csv_sha256`, compatible with the training data validation contract. If synthetic-origin rows exist, the audit retains their experimental flag and run training only with the existing explicit `--allow-synthetic` option. This workflow collects and packages human labels; it does not perform annotation or train weights by itself.
