# CodeMood IEEE journal manuscript - editable Overleaf source

1. In Overleaf choose New Project > Upload Project and upload the supplied ZIP.
2. Select main.tex as the main document and pdfLaTeX as the compiler.
3. Recompile; Overleaf runs BibTeX for the numbered IEEE references.

The source uses the standard IEEEtran journal class, with the title, three authors,
affiliation, corresponding-author footnote, abstract and Index Terms on page one.
There is no separate coursework cover. Suganthini C is the corresponding author;
her supplied contact is suganthini.c@vit.ac.in. Edit authors.tex to update contact
or affiliation details. No unprovided academic title or IEEE membership is asserted.

Edit main.tex for prose and eighteen numbered equations, references.bib for the
31 references, and figures/*.tex for three editable vector TikZ diagrams. The
full-system architecture occupies a separate figure page. Tables, algorithms,
plots and diagrams remain black and white. Additional Mermaid/SVG sources are
included when available. literature_sources.csv and related_work_comparison.csv
record the source links and the implemented extensions discussed in the survey.

verification_report.json records arithmetic, metadata, citation and data-file
checks against saved local artifacts. Model scores concern synthetic held-out
sets only. XLM-R and mBERT share the 16k/2k/2k bilingual split; MuRIL uses the
8k/1k/1k Hindi-English scope. Calibration scores are fitting-set diagnostics.
Optional fusion/context/sarcasm training paths are not credited with encoder gains.
No model was retrained for this manuscript revision.

This is a generic IEEE journal manuscript, not a template certified for a specific
journal. Check the chosen journal's page limits, anonymization and submission
requirements before upload. The earlier journal PDF/ZIP is preserved in Documents for review/Previous versions.

The GitHub repository and pinned evaluated source revision appear in Code Availability.
The current affiliation address was verified against VIT's official mandatory disclosure:
https://vit.ac.in/sites/default/files/iqac/VIT-Vellore-Mandatory-Disclosure.pdf
Author names and corresponding email were supplied by the team. No ORCID, department,
funding or acceptance dates have been invented. Venue selection and author ORCIDs
are still administrative submission tasks; independent natural-message evaluation
is an empirical limitation, not a formatting issue.

Final aligned edition: 16 pages, seven figures, ten tables, eighteen equations and 31 references. All pages were reviewed and all fonts are embedded. The evaluated implementation is revision2885df0; the recorded suite passes254tests and33subtests.
