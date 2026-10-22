# Submission report and editable diagrams

The latest report is [the project submission DOCX](../../outputs/CodeMood_Hindi_English_Project_Progress_Report.docx).
It contains one full-page master architecture plus four module diagrams:
seven-emotion inference, Colab training/export, file evaluation and model disagreement with human review.
Team details are retained. mBERT is installed for both language mixes; its synthetic test results were reproduced locally.

`report.md` is the editable report source. `diagrams/*.mmd` are Mermaid sources;
`figures/*.svg` are vector artwork and `figures/*.png` are 3x print-quality renders.
Word embeds the high-resolution PNGs for compatibility. Edit the Mermaid source
and rerender to change diagram structure; images are not Word-native shapes.
`validation.json` records current offline software checks and actual local model outputs.
Synthetic training metrics do not establish real-world accuracy.

The report builder is `scripts/build_submission_report.py` (python-docx and Pillow).
The diagram renderer is `scripts/render_submission_diagrams.cjs` (Mermaid 11 and Playwright).
Pass the Playwright module and mermaid.min.js paths to the renderer. The document
renderer uses the Codex bundled document helper and native Word PDF export on Windows;
set CODEMOOD_DOCUMENT_RUNTIME if the bundled runtime is installed elsewhere.
The previous report is preserved locally under archive/reports and omitted from Git.
