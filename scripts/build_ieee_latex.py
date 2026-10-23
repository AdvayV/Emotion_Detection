"""Package the IEEE-style manuscript as editable Overleaf/LaTeX sources.

This builder emits source files and a portable ZIP; it does not compile TeX.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "ieee_manuscript"
SOURCE = BASE / "manuscript.md"
REFERENCES = BASE / "references.json"
FIGURES = BASE / "figures"
DIAGRAMS = BASE / "diagrams"
OVERLEAF = BASE / "overleaf"
MAIN_TEX = OVERLEAF / "main.tex"
BIB_FILE = OVERLEAF / "references.bib"
README = OVERLEAF / "README.md"
OUTPUT_ZIP = ROOT / "outputs" / "23OctReview AdvayV KaavishG CD Overleaf.zip"

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CITATION_RE = re.compile(r"\[@([^]]+)\]")
INLINE_RE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`|\[@[^]]+\]|\[[^\]]+\]\([^)]+\)|\$[^$]+\$)")
FENCE_RE = re.compile(r"^```(?:[A-Za-z0-9_-]+)?\s*$", re.I)
ASSET_EXTENSIONS = {".png", ".svg", ".mmd", ".tex", ".tikz"}


def read_references():
    if not REFERENCES.is_file():
        raise FileNotFoundError(f"Missing citation source: {REFERENCES}")
    raw = json.loads(REFERENCES.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "references" in raw:
        raw = raw["references"]
    if isinstance(raw, dict):
        records = [{"key": key, **(value if isinstance(value, dict) else {"title": str(value)})} for key, value in raw.items()]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("references.json must be a list or key-to-reference object")
    result = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every reference must be a JSON object")
        key = str(record.get("key") or record.get("id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", key):
            raise ValueError(f"Reference key is empty or invalid for BibTeX: {key!r}")
        if key in seen:
            raise ValueError(f"Duplicate reference key: {key}")
        if not record.get("title"):
            raise ValueError(f"Reference {key} has no title")
        seen.add(key)
        result.append({**record, "key": key})
    return result


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _citation_keys(raw: str) -> list[str]:
    inside = raw.strip().strip("[]")
    return [piece.lstrip("@").strip() for piece in re.split(r"[,;\s]+", inside) if piece.lstrip("@").strip()]


def latex_inline(text: str, known_keys: set[str]) -> str:
    out = []
    cursor = 0
    for match in INLINE_RE.finditer(text):
        out.append(latex_escape(text[cursor:match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            out.append(r"\textbf{" + latex_inline(token[2:-2], known_keys) + "}")
        elif token.startswith("*"):
            out.append(r"\emph{" + latex_inline(token[1:-1], known_keys) + "}")
        elif token.startswith("`"):
            maths = {
                "p(y|x) = softmax(W h + b)": r"\(\mathbf{p}(y\mid x)=\operatorname{softmax}(Wh+b)\)",
                "p_T = softmax(log(p) / T)": r"\(\mathbf{p}_T=\operatorname{softmax}(\log\mathbf{p}/T)\)",
                "JS(P,Q) = [KL(P||M) + KL(Q||M)] / 2": r"\(\mathrm{JS}(P,Q)=[\mathrm{KL}(P\Vert M)+\mathrm{KL}(Q\Vert M)]/2\)",
                "TP / (TP + FP)": r"\(\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})\)",
                "TP / (TP + FN)": r"\(\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})\)",
            }
            out.append(maths.get(token[1:-1], r"\texttt{" + latex_escape(token[1:-1]) + "}"))
        elif token.startswith("[@"):
            keys = _citation_keys(token)
            missing = [key for key in keys if key not in known_keys]
            if missing:
                raise ValueError(f"Citation key(s) not in references.json: {', '.join(missing)}")
            out.append(r"\cite{" + ",".join(dict.fromkeys(keys)) + "}")
        elif token.startswith("$"):
            # Keep inline math as LaTeX math rather than escaping its operators.
            out.append(token)
        else:
            label, _target = LINK_RE.fullmatch(token).groups()
            out.append(latex_escape(label))
        cursor = match.end()
    out.append(latex_escape(text[cursor:]))
    return "".join(out)


def _bib_escape(value) -> str:
    if isinstance(value, list):
        value = " and ".join(map(str, value))
    text = str(value)
    import unicodedata
    for code, replacement in {0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"', 0x2013: "-", 0x2014: "--", 0x00a0: " ", 0xfeff: ""}.items():
        text = text.replace(chr(code), replacement)
    text = text.replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    accents = {0x0301: "'", 0x0300: "`", 0x0302: "^", 0x0308: '"', 0x030c: "v", 0x0303: "~", 0x0327: "c", 0x030a: "r"}
    output = []
    for char in text:
        if ord(char) in {0x0141, 0x0142}:
            output.append(r"{\L}" if ord(char) == 0x0141 else r"{\l}")
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if len(decomposed) == 2 and ord(decomposed[1]) in accents:
            output.append("{\\" + accents[ord(decomposed[1])] + "{" + decomposed[0] + "}}")
        else: output.append(char)
    return "".join(output)


def render_bibtex(records: list[dict]) -> str:
    entries = []
    for record in records:
        fields = []
        authors = record.get("authors", record.get("author"))
        if authors:
            fields.append(f"  author = {{{_bib_escape(authors)}}}")
        title = str(record["title"])
        protected = {}
        acronyms = ("CodeMood", "Hinglish", "MuRIL", "mBERT", "XLM-R", "RoBERTa", "BERT", "XLNet", "SVM", "NLP", "LLMs", "LLM", "APIs", "API", "ECE", "GPU", "CPU", "CNN", "RNN", "LSTM", "AI", "XLM")
        for index, acronym in enumerate(acronyms):
            placeholder = f"ZXACRONYM{index}ZX"
            title = re.sub(rf"(?<![A-Za-z0-9]){re.escape(acronym)}(?![A-Za-z0-9])", placeholder, title)
            if placeholder in title:
                protected[placeholder] = acronym
        escaped_title = _bib_escape(title)
        for placeholder, acronym in protected.items():
            escaped_title = escaped_title.replace(placeholder, "{{" + acronym + "}}")
        fields.append("  title = {{" + escaped_title + "}}")
        venue = record.get("venue", record.get("journal", record.get("booktitle", record.get("conference", record.get("publisher")))))
        explicit_type = str(record.get("entry_type", record.get("type", ""))).lower()
        venue_l = str(venue or "").lower()
        explicit_type = {"conference": "inproceedings", "journal": "article", "arxiv": "misc"}.get(explicit_type, explicit_type)
        if explicit_type in {"article", "inproceedings", "book", "misc"}:
            entry_type = explicit_type
        elif record.get("journal") or any(term in venue_l for term in ("ieee transactions", "ieee journal", "acm transactions", "plos", "communications of the acm")):
            entry_type = "article"
        elif record.get("booktitle") or record.get("conference") or "proceedings" in venue_l or any(term in venue_l for term in ("acl", "emnlp", "naacl", "coling", "lrec", "interspeech", "icassp", "neurips", "icml", "iclr", "ijcai", "aaai", "kdd")):
            entry_type = "inproceedings"
        elif "arxiv" in venue_l:
            entry_type = "misc"
        elif record.get("publisher") and "book" in explicit_type:
            entry_type = "book"
        else:
            entry_type = "misc"
        venue_field = {"article": "journal", "inproceedings": "booktitle", "book": "publisher", "misc": "howpublished"}[entry_type]
        if venue:
            fields.append(f"  {venue_field} = {{{_bib_escape(venue)}}}")
        for key in ("year", "volume", "number", "pages", "doi"):
            if record.get(key):
                fields.append(f"  {key} = {{{_bib_escape(record[key])}}}")
        if record.get("url"):
            fields.append(f"  url = {{{_bib_escape(record['url'])}}}")
        entries.append(f"@{entry_type}{{{record['key']},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + "\n"


def _figure(line: str):
    match = IMAGE_RE.search(line)
    if not match:
        return None
    return match.group(1).strip(), unquote(match.group(2).split("#", 1)[0])


def _figure_assets(target: str) -> tuple[str, str | None, Path]:
    relative = Path(target.replace("\\", "/"))
    if relative.is_absolute() or relative.suffix.lower() != ".png" or ".." in relative.parts:
        raise ValueError(f"Figure must be a relative PNG path: {target}")
    stem = relative.stem
    png = (BASE / relative).resolve()
    if png.parent != FIGURES.resolve():
        png = (FIGURES / relative.name).resolve()
    if png.parent != FIGURES.resolve() or not png.is_file():
        raise FileNotFoundError(f"Figure PNG is missing or outside the figure folder: {target}")
    # TikZ source takes precedence for vector-editable diagrams.
    tex = None
    for directory, prefix in ((FIGURES, "figures"), (DIAGRAMS, "diagrams")):
        for suffix in (".tex", ".tikz"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                tex = f"{prefix}/{candidate.name}"
                break
        if tex:
            break
    return f"figures/{png.name}", tex, png


def _figure_latex(caption: str, target: str) -> str:
    image_ref, tikz_path, png_path = _figure_assets(target)
    stem = Path(target).stem
    caption = re.sub(r"^(?:Fig(?:ure)?\.?\s*)\d+\s*[.:]\s*", "", caption.strip(), flags=re.I)
    caption_tex = latex_inline(caption, KNOWN_KEYS)
    label = re.sub(r"[^A-Za-z0-9:.-]+", "-", stem)
    if stem == "master_architecture":
        if tikz_path:
            visual = rf"\resizebox{{.96\textwidth}}{{!}}{{\input{{{tikz_path}}}}}"
        else:
            visual = rf"\includegraphics[width=.96\textwidth,height=.78\textheight,keepaspectratio]{{{image_ref}}}"
        return (
            "\\clearpage\n\\onecolumn\n\\thispagestyle{plain}\n"
            "\\begin{center}\n" + visual + "\n\\end{center}\n"
            "\\refstepcounter{figure}\\begin{center}\\footnotesize\\textbf{Fig. \\thefigure.} "
            + caption_tex + "\\end{center}\n"
            f"\\label{{fig:{label}}}\n\\clearpage\n\\twocolumn\n"
        )
    is_wide = stem in {"model_comparison", "training_history"}
    env = "figure*" if is_wide else "figure"
    width = ".88\\textwidth" if is_wide else ".98\\columnwidth"
    max_h = ".38\\textheight" if is_wide else ".34\\textheight"
    visual = rf"\resizebox{{{width}}}{{!}}{{\input{{{tikz_path}}}}}" if tikz_path else rf"\includegraphics[width={width},height={max_h},keepaspectratio]{{{image_ref}}}"
    return (
        f"\\begin{{{env}}}[!t]\n\\centering\n{visual}\n"
        f"\\caption{{{caption_tex}}}\\label{{fig:{label}}}\n\\end{{{env}}}\n"
    )


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        content = line.strip().strip("|")
        if re.fullmatch(r"\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*", content):
            continue
        rows.append([cell.strip() for cell in content.split("|")])
    count = max(map(len, rows), default=0)
    return [row + [""] * (count - len(row)) for row in rows]


TABLE_TITLES = ["Seven emotion checkpoint outputs", "Dataset resources and partitions", "Per class F1 within each model test scope", "Validation subset confidence scaling", "Checking phrase diagnostics", "Matched bilingual synthetic test comparison"]
TABLE_COUNTER = 0

def _table_latex(lines: list[str]) -> str:
    global TABLE_COUNTER
    title = TABLE_TITLES[TABLE_COUNTER]
    TABLE_COUNTER += 1
    rows = _table_rows(lines)
    if not rows:
        return ""
    columns = len(rows[0])
    # X columns wrap long descriptors; right-aligned X is used for numeric data.
    specs = []
    for ci in range(columns):
        values = [row[ci].strip() for row in rows[1:] if row[ci].strip()]
        numeric = bool(values) and all(re.fullmatch(r"[<>=~+-]?[\d][\d,.%/() -]*", value) for value in values)
        alignment = r"\raggedleft" if numeric and len(rows[0][ci]) <= 18 else r"\raggedright"
        specs.append(r">{" + alignment + r"\arraybackslash}X")
    spec = "@{}" + "".join(specs) + "@{}"
    rendered = ["\\begin{table}[!t]", "\\caption{" + title + "}", "\\centering", "\\footnotesize", rf"\begin{{tabularx}}{{\columnwidth}}{{{spec}}}", "\\toprule"]
    for ri, row in enumerate(rows):
        cells = [latex_inline(value, KNOWN_KEYS) for value in row]
        if ri == 0:
            cells = [r"\textbf{" + value + "}" for value in cells]
        rendered.append(" & ".join(cells) + r" \\")
        if ri == 0:
            rendered.append("\\midrule")
    rendered.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}"])
    return "\n".join(rendered) + "\n"


def _heading(text: str, level: int) -> str:
    clean = text.strip()
    clean = re.sub(r"^\d+(?:\.\d+)*(?:[.)])?\s+", "", clean)
    if level == 2:
        barrier = "\\FloatBarrier\n" if clean.lower() == "conclusion" else ""
        return barrier + rf"\section{{{latex_inline(clean, KNOWN_KEYS)}}}"
    if level == 3:
        return rf"\subsection{{{latex_inline(clean, KNOWN_KEYS)}}}"
    return rf"\subsubsection{{{latex_inline(clean, KNOWN_KEYS)}}}"


def render_body(markdown: str, records: list[dict]) -> str:
    global KNOWN_KEYS
    KNOWN_KEYS = {record["key"] for record in records}
    lines = markdown.splitlines()
    output = []
    in_abstract = False
    in_keywords = False
    in_code = False
    code_lines = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if FENCE_RE.match(line):
            if in_code:
                output.append("\\begin{lstlisting}\n" + "\n".join(code_lines) + "\n\\end{lstlisting}")
                code_lines, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw.rstrip())
            i += 1
            continue
        if not line:
            i += 1
            continue
        if line.startswith("<!--"):
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("## "):
            title = re.sub(r"^\d+(?:\.\d+)*(?:[.)])?\s+", "", line[3:].strip())
            lowered = title.lower()
            if in_abstract:
                output.append("\\end{abstract}")
                in_abstract = False
            if in_keywords:
                output.append("\\end{IEEEkeywords}")
                in_keywords = False
            if lowered == "references":
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("## "):
                    i += 1
                continue
            if lowered == "abstract":
                output.append("\\begin{abstract}")
                in_abstract = True
            elif lowered in {"keywords", "index terms"}:
                output.append("\\begin{IEEEkeywords}")
                in_keywords = True
            else:
                output.append(_heading(title, 2))
            i += 1
            continue
        if line.startswith("### "):
            output.append(_heading(line[4:], 3))
            i += 1
            continue
        if line.startswith("#### "):
            output.append(_heading(line[5:], 4))
            i += 1
            continue
        image = _figure(line)
        if image:
            caption, target = image
            output.append(_figure_latex(caption, target))
            i += 1
            continue
        if line.startswith("|"):
            group = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i].strip())
                i += 1
            output.append(_table_latex(group))
            continue
        if re.match(r"^[-*+]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*+]\s+", "", lines[i].strip()))
                i += 1
            output.append("\\begin{itemize}\n" + "\n".join(r"\item " + latex_inline(item, KNOWN_KEYS) for item in items) + "\n\\end{itemize}")
            continue
        if re.match(r"^\d+[.)]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+[.)]\s+", "", lines[i].strip()))
                i += 1
            output.append("\\begin{enumerate}\n" + "\n".join(r"\item " + latex_inline(item, KNOWN_KEYS) for item in items) + "\n\\end{enumerate}")
            continue
        if line.startswith("> "):
            output.append("\\begin{quote}" + latex_inline(line[2:], KNOWN_KEYS) + "\\end{quote}")
            i += 1
            continue
        # Join Markdown soft-wrapped prose lines into one paragraph.
        paragraph = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "|", "![") ) or FENCE_RE.match(nxt) or re.match(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s)", nxt):
                break
            paragraph.append(nxt)
            i += 1
        output.append(latex_inline(" ".join(paragraph), KNOWN_KEYS))
    if in_code:
        raise ValueError("Unclosed Markdown pseudocode fence")
    if in_abstract:
        output.append("\\end{abstract}")
    if in_keywords:
        output.append("\\end{IEEEkeywords}")
    return "\n\n".join(output) + "\n"


def _bib_for(record: dict) -> str:
    def val(key, fallback=None):
        value = record.get(key, fallback)
        if value is None:
            return ""
        if isinstance(value, list):
            value = " and ".join(map(str, value))
        return str(value)
    authors = val("authors", record.get("author"))
    venue = val("venue", record.get("journal", record.get("publisher")))
    fields = []
    for key, value in (("author", authors), ("title", val("title")), ("howpublished", venue),
                       ("year", val("year")), ("volume", val("volume")), ("number", val("number")),
                       ("pages", val("pages")), ("doi", val("doi")), ("url", val("url"))):
        if value:
            fields.append(f"  {key} = {{{_bib_escape(value)}}}")
    return f"@misc{{{record['key']},\n" + ",\n".join(fields) + "\n}\n"


def render_main(records: list[dict], body: str) -> str:
    keys = ",".join(record["key"] for record in records)
    if not keys:
        raise ValueError("At least one reference is required")
    return r"""\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{cite}
\usepackage{placeins}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,frame=single,framesep=3pt,showstringspaces=false}
\title{CodeMood Multilingual Emotion Detection}
\author{\IEEEauthorblockN{Advay Vivek\IEEEauthorrefmark{1} and Kaavish Gogia\IEEEauthorrefmark{1}}\\
\IEEEauthorblockA{\IEEEauthorrefmark{1}VIT Vellore, Compiler Design}}
\begin{document}
\onecolumn
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{0.22\textheight}
{\LARGE\bfseries CodeMood Multilingual Emotion Detection\par}
\vspace{2.2cm}
{\large Prepared by\par}
\vspace{0.4cm}
{\large Advay Vivek\\24BCT0136\par}
\vspace{0.35cm}
{\large Kaavish Gogia\\24BCT0103\par}
\vspace{1.4cm}
{\normalsize Guided by Suganthini C\\VIT Vellore\\Compiler Design\\23 October 2026\par}
\end{titlepage}
\twocolumn
\maketitle
""" + rf"\nocite{{{keys}}}" + "\n" + body + r"""
\begingroup
\interlinepenalty=10000
\bibliographystyle{IEEEtran}
\bibliography{references}
\endgroup
\end{document}
"""


def _copy_assets(destination: Path):
    copied = []
    for source_dir, rel_base in ((FIGURES, Path("figures")), (DIAGRAMS, Path("diagrams"))):
        if not source_dir.is_dir():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file() or source.suffix.lower() not in ASSET_EXTENSIONS:
                continue
            rel = source.relative_to(source_dir)
            target = destination / rel_base / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target.relative_to(destination).as_posix())
    return copied


def build():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Manuscript Markdown not found: {SOURCE}")
    markdown = SOURCE.read_text(encoding="utf-8")
    if re.search(r"\{\{[^}]+\}\}", markdown):
        raise ValueError("Unresolved manuscript placeholder remains")
    records = read_references()
    global KNOWN_KEYS
    KNOWN_KEYS = {record["key"] for record in records}
    body = render_body(markdown, records)
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    MAIN_TEX.write_text(render_main(records, body), encoding="utf-8")
    BIB_FILE.write_text(render_bibtex(records), encoding="utf-8")
    copied = _copy_assets(OVERLEAF)
    import csv
    with (OVERLEAF / "literature_sources.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference", "key", "title", "year", "venue", "url"])
        writer.writeheader()
        for index, record in enumerate(records, 1):
            writer.writerow({"reference": index, **{name: record.get(name, "") for name in ["key", "title", "year", "venue", "url"]}})
    matrix = json.loads((BASE / "related_work_matrix.json").read_text(encoding="utf-8"))
    with (OVERLEAF / "related_work_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader(); writer.writerows(matrix)
    readme = """# CodeMood IEEE-style Overleaf project

1. Open Overleaf and choose New Project > Upload Project.
2. Upload the supplied Overleaf ZIP.
3. Set `main.tex` as Main document and choose pdfLaTeX as Compiler.
4. Click Recompile. Overleaf runs BibTeX for the numbered references.

Edit `main.tex` for prose, `references.bib` for the 31 verified references, and
`figures/*.tex` for the three editable TikZ diagrams. The included
`related_work_comparison.csv` maps 20 papers to implemented extensions and code
paths; `literature_sources.csv` lists all verified source links.

 The source uses the standard
`IEEEtran` conference class and editable LaTeX, BibTeX, tables, and pseudocode.

The cover page and the full-page architecture insert are project-specific
presentation choices. The manuscript is IEEE style and is not represented as a
certified IEEE conference template. Mermaid, SVG, and TikZ sources are included
where available; raster PNGs provide portable figure fallbacks.
"""
    README.write_text(readme, encoding="utf-8")
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OVERLEAF.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OVERLEAF).as_posix())
    print(f"{MAIN_TEX}\n{BIB_FILE}\n{README}\n{OUTPUT_ZIP}")
    print(f"Copied {len(copied)} optional figure/source assets")


if __name__ == "__main__":
    build()
