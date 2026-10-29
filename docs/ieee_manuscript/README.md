# Aligned CodeMood IEEE journal manuscript

- [Current PDF](../../Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal.pdf)
- [Direct-upload Overleaf ZIP](../../Documents%20for%20review/Latest%20IEEE%20paper/CodeMood_IEEE_Journal_Overleaf.zip)
- [Editable main.tex](../../Documents%20for%20review/Latest%20IEEE%20paper/LaTeX%20source/main.tex)
- [Authors and correspondence](../../Documents%20for%20review/Latest%20IEEE%20paper/LaTeX%20source/authors.tex)

The expanded alignment edition targets 15–18 pages and compiles to 16 pages in
local verification. It retains standard IEEEtran journal typography, two columns,
the full-page architecture, seven figures, ten tables, eighteen explained equations,
four algorithms and 31 references. Added sections cover encoder/tokenizer details,
feature traceability, language contracts, Colab recovery and worked metric examples.
The probability-quality table explicitly distinguishes loss/error from confidence.

Advay Vivek, Kaavish Gogia and Suganthini C are listed as authors. Suganthini C is
corresponding author at suganthini.c@vit.ac.in. The affiliation is Vellore Institute
of Technology, Katpadi-Thiruvalam Road, Vellore, Tamil Nadu 632014, India, verified
from the official institutional disclosure. No department, ORCID or funding was invented.

Upload the ZIP using Overleaf > New Project > Upload Project. Select main.tex as
Main document, pdfLaTeX as compiler, then Recompile. Text, tables, mathematics and
TikZ diagrams remain editable. Vector PDF plots and high-resolution previews are
included. Earlier PDF/ZIP editions are preserved in Documents for review/Previous versions.

The source repository and evaluated implementation revision appear in Code Availability.
The local inference replay and software regression records are under docs/evaluation.
Model results are synthetic-only, calibration uses its fitting subset, and optional
fusion/context/sarcasm/reviewer gains remain unmeasured. Software tests do not imply
human-message accuracy. This revision adds no RL, training data or research weights.

Rebuild using scripts/build_ieee_charts.py, scripts/audit_ieee_manuscript.py and
scripts/build_ieee_latex.py in the report Python environment. Compile and render
for page review before replacing deliverables. The GIF is intentionally unchanged.
