# CodeMood IEEE-style Overleaf project

1. Open Overleaf and choose New Project > Upload Project.
2. Upload the supplied Overleaf ZIP.
3. Set `main.tex` as Main document and choose pdfLaTeX as Compiler.
4. Click Recompile. Overleaf runs BibTeX for the numbered references.

Edit `main.tex` for prose, `references.bib` for the 31 verified references, and
`figures/*.tex` for the three editable TikZ diagrams. Eighteen numbered equation
blocks include nearby symbol definitions; edit them directly in main.tex. The included
`related_work_comparison.csv` maps 20 papers to implemented extensions and code
paths; `literature_sources.csv` lists all verified source links.

 The source uses the standard
`IEEEtran` conference class and editable LaTeX, BibTeX, tables, and pseudocode.

The cover page and the full-page architecture insert are project-specific
presentation choices. The manuscript is IEEE style and is not represented as a
certified IEEE conference template. Mermaid, SVG, and TikZ sources are included
where available; raster PNGs provide portable figure fallbacks.
