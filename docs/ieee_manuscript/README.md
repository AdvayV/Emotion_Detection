# CodeMood IEEE manuscript

- [Editable Word manuscript](../../outputs/23OctReview%20AdvayV%20KaavishG%20CD%20IEEE.docx)
- [Overleaf upload ZIP](../../outputs/23OctReview%20AdvayV%20KaavishG%20CD%20Overleaf.zip)
- [LaTeX main document](overleaf/main.tex)

The manuscript contains the requested cover, abstract, introduction, three literature survey subsections, four explained modules and pseudocode algorithms, implementation and results, six numbered tables, a full-page architecture diagram, two workflow diagrams, two measured chart figures, conclusion and 31 IEEE-numbered references. The cover names Advay Vivek (24BCT0136), Kaavish Gogia (24BCT0103), guide Suganthini C and VIT Vellore.

`references.json` records the 31 verified primary paper sources. `related_work_matrix.json` maps 20 papers to implemented extensions, with actual code evidence paths. All comparisons in that matrix concern implementation; no unmatched published benchmark is represented as an accuracy gain. The two CSV lists are included in the Overleaf project.

## Edit in Overleaf

Choose New Project > Upload Project and select the ZIP. Set main.tex as Main document, choose pdfLaTeX and click Recompile. Edit prose and tables in main.tex, bibliography in references.bib, and diagrams in figures/*.tex. The standalone cover and full-page architecture are project-specific additions to the IEEEtran conference layout. Word prose, tables and pseudocode are editable; its diagrams are rebuilt from the included source assets.

## Rebuild and verification

Use the configured bundled document Python runtime with scripts/build_ieee_charts.py, scripts/build_ieee_manuscript_docx.py and scripts/build_ieee_latex.py. Mermaid fallbacks use scripts/render_ieee_diagrams.cjs. The saved model metrics provide all plotted values. The existing application code was unchanged in this manuscript task.

The Word file was rendered to 13 pages and every page visually reviewed; a split reference was corrected and the changed pages rechecked. The Overleaf source compiled successfully with Tectonic and was visually reviewed, with no missing glyphs, undefined citations or overfull boxes. QA PDFs and page images are under ignored outputs/verification. Scores remain tied to synthetic test partitions, and no measured improvement is assigned to an unevaluated optional model branch.
