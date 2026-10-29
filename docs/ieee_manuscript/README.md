# CodeMood IEEE journal manuscript

- [Current Overleaf upload ZIP](../../Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal_Overleaf.zip)
- [Compiled journal PDF](../../Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal.pdf)
- [Editable LaTeX document](../../Documents%20for%20review/Latest%20IEEE%20paper/LaTeX%20source/main.tex)
- [Authors and correspondence](../../Documents%20for%20review/Latest%20IEEE%20paper/LaTeX%20source/authors.tex)
- [Result verification record](verification_report.json)

The current source uses IEEEtran journal format: a first-page title and author
list, affiliation and correspondence footnotes, abstract, Index Terms, numbered
sections, equations and IEEE references. Advay Vivek, Kaavish Gogia and Suganthini C
are listed as authors. Suganthini C is the corresponding author, using the
user-supplied address suganthini.c@vit.ac.in and VIT Vellore affiliation.

The manuscript includes three literature subsections, four explained modules
and algorithms, six tables, a full-page system architecture, two module flow
diagrams, two measured performance/training graphs, eighteen explained equations,
and 31 references. The 20-paper comparison matrix and literature source links
are included in the upload package. The earlier project-cover ZIP and existing
Word documents are preserved; this revision updates the journal LaTeX source.

## Edit in Overleaf

Choose New Project > Upload Project, upload the current ZIP, select main.tex as
Main document and pdfLaTeX as Compiler, and click Recompile. Edit authors.tex for
author/contact details, main.tex for prose, references.bib for references and
figures/*.tex for the vector diagrams. The journal source has no coursework cover.
IEEE provides [article templates and its template selector](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).
Apply the chosen journal's additional submission requirements when selecting a venue.

## Evidence and rebuild

Run scripts/audit_ieee_manuscript.py using the report Python environment, then
scripts/build_ieee_charts.py and scripts/build_ieee_latex.py. These use existing
saved results and do not train or download models. The PDF compiles with Tectonic;
all eleven pages have been rendered and visually checked. There are no missing
glyphs, unresolved citations or overfull boxes. Local previews are under ignored
archives/verification/ieee-journal.

The audit recomputes accuracy, per-class precision/recall/F1 and macro-F1 from
saved confusion matrices, checks model metadata and Colab metric copies, and
reproduces the dataset/split hashes from the current CSV. Manuscript class-table
values and diagnostic-file hashes also match. XLM-R and mBERT use the same
16,000/2,000/2,000 bilingual synthetic partitions; MuRIL uses 8,000/1,000/1,000
Hindi-English rows. Calibration uses a 98-row fitting subset. These scopes remain
explicit, and no empirical gain is assigned to an unevaluated optional branch.
The reported 221-test/33-subtest result is the earlier completed software suite,
not a new test run for this manuscript revision.
