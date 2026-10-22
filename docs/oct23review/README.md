# 23 October project review

The final Word report is [23OctReview AdvayV KaavishG CD.docx](../../outputs/23OctReview%20AdvayV%20KaavishG%20CD.docx).
It has 15 pages, six architecture/workflow diagrams and two scientific chart figures.
Team: Advay Vivek (24BCT0136) and Kaavish Gogia (24BCT0103).

## Evidence and editable sources

- `report.md`: report text and table source.
- `diagrams/*.mmd`: editable Mermaid sources; PNG/SVG artwork is in `figures`.
- `results.json`: checked saved checkpoint metrics, epoch histories and data scope.
- `challenge_diagnostics.json`: separate checking-set diagnostics with input hashes.
- `calibration_example.json`: measured 98-row synthetic validation fit, not independent evaluation.

All existing checkpoint scores are from synthetic data. MuRIL used a smaller Hindi-English training scope than bilingual XLM-R and mBERT. No advanced fusion weights, human benchmark, learned sarcasm result or reviewer accuracy gain is reported.

## Rebuild

Use the configured document runtime and run `scripts/build_oct23_charts.py`, render the Mermaid sources with `scripts/render_oct23_diagrams.cjs`, then run `scripts/build_oct23_review.py`. The chart builder verifies the installed model metric copies against saved Colab study records. It requires the local installed checkpoints. Word tables and prose remain editable; diagrams can be changed in their Mermaid source and rebuilt.

The final report was rendered to 15 pages using the packaged document rasterizer with Word PDF export, and every page was visually checked. Rendered previews remain in ignored `outputs/verification/oct23review_final`; they are not submission deliverables. The full software suite passed 221 tests and 33 subtests.
