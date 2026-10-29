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
OVERLEAF = ROOT / "Documents for review" / "Latest IEEE paper" / "LaTeX source"
MAIN_TEX = OVERLEAF / "main.tex"
BIB_FILE = OVERLEAF / "references.bib"
README = OVERLEAF / "README.md"
OUTPUT_ZIP = ROOT / "Documents for review" / "Latest IEEE paper" / "CodeMood_IEEE_Journal_Overleaf.zip"

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CITATION_RE = re.compile(r"\[@([^]]+)\]")
INLINE_RE = re.compile(r"(\[\[(?:fig|tab|eq):[A-Za-z0-9_.-]+\]\]|\*\*.+?\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`|\[@[^]]+\]|\[[^\]]+\]\([^)]+\)|\$[^$]+\$)")
FENCE_RE = re.compile(r"^```(?:[A-Za-z0-9_-]+)?\s*$", re.I)
ASSET_EXTENSIONS = {".png", ".pdf", ".svg", ".mmd", ".tex", ".tikz"}


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
        if token.startswith("[["):
            kind, key = token[2:-2].split(":", 1)
            if kind == "fig": out.append(r"Fig.~\ref{fig:" + key + "}")
            elif kind == "tab": out.append(r"Table~\ref{tab:" + key + "}")
            else: out.append(r"\eqref{eq:" + key + "}")
        elif token.startswith("**"):
            out.append(r"\textbf{" + latex_inline(token[2:-2], known_keys) + "}")
        elif token.startswith("*"):
            out.append(r"\emph{" + latex_inline(token[1:-1], known_keys) + "}")
        elif token.startswith("`"):
            from ieee_equations import INLINE_LATEX
            if token[1:-1] in INLINE_LATEX:
                out.append(r"\(" + INLINE_LATEX[token[1:-1]] + r"\)")
                cursor = match.end()
                continue
            maths = {
                "p(y|x) = softmax(W h + b)": r"\(\mathbf{p}(y\mid x)=\operatorname{softmax}(Wh+b)\)",
                "p_T = softmax(log(p) / T)": r"\(\mathbf{p}_T=\operatorname{softmax}(\log\mathbf{p}/T)\)",
                "JS(P,Q) = [KL(P||M) + KL(Q||M)] / 2": r"\(\mathrm{JS}(P,Q)=[\mathrm{KL}(P\Vert M)+\mathrm{KL}(Q\Vert M)]/2\)",
                "TP / (TP + FP)": r"\(\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})\)",
                "TP / (TP + FN)": r"\(\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})\)",
            }
            code = latex_escape(token[1:-1]).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
            out.append(maths.get(token[1:-1], r"\texttt{" + code + "}"))
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
            label, target = LINK_RE.fullmatch(token).groups()
            if target.startswith(("https://", "http://")):
                if label == target:
                    out.append(r"\url{" + target + "}")
                else:
                    out.append(r"\href{" + target.replace("#", r"\#").replace("%", r"\%") + "}{" + latex_escape(label) + "}")
            else:
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
    entries = [r"""@IEEEtranBSTCTL{IEEEcontrol,
  CTLuse_forced_etal = {yes},
  CTLmax_names_forced_etal = {6},
  CTLnames_show_etal = {1}
}"""]
    for original in records:
        record = dict(original)
        # Separate bibliographic fields instead of italicizing page ranges as venues.
        venue = str(record.get("venue", ""))
        pages = re.search(r", pp\. ([0-9]+)-([0-9]+)", venue)
        if pages:
            record["pages"] = pages[1] + "--" + pages[2]
            venue = venue[:pages.start()]
        journals = {
            "ghosh2018": ("Computational Linguistics", "44", "4"),
            "lin1991": ("IEEE Transactions on Information Theory", "37", "1"),
            "davani2022": ("Transactions of the Association for Computational Linguistics", "10", ""),
            "bender2018": ("Transactions of the Association for Computational Linguistics", "6", ""),
            "kraljnovak2015": ("PLOS ONE", "10", "12"),
            "gebru2021": ("Communications of the ACM", "64", "12"),
        }
        key = record["key"]
        if key in journals:
            venue, volume, issue = journals[key]
            record.update(entry_type="article", volume=volume, number=issue)
            if key == "kraljnovak2015": record["pages"] = "e0144296"
        elif key in {"vaswani2017", "ovadia2019", "lakshminarayanan2017", "geifman2017"}:
            venue = "Advances in Neural Information Processing Systems"
            record.update(entry_type="inproceedings", volume="32" if key=="ovadia2019" else "30")
        elif key in {"hu2020", "guo2017"}:
            venue = "Proc. ICML"
            record.update(entry_type="inproceedings", series="Proceedings of Machine Learning Research", volume="119" if key=="hu2020" else "70")
        elif key != "khanuja2021":
            record["entry_type"] = "inproceedings"
        record["venue"] = venue
        fields = []
        authors = record.get("authors", record.get("author"))
        if isinstance(authors, list):
            family_names = {"Hal Daum\u00e9 III": "Daum\u00e9, III, Hal", "Petra Kralj Novak": "Kralj Novak, Petra", "Cynthia Van Hee": "Van Hee, Cynthia"}
            authors = [family_names.get(name, name) for name in authors]
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
        for key in ("series", "year", "volume", "number", "pages", "doi"):
            if record.get(key):
                fields.append(f"  {key} = {{{_bib_escape(record[key])}}}")
        if record.get("url"):
            fields.append(f"  url = {{{record['url']}}}")
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
    vector_pdf = png.with_suffix(".pdf")
    image_ref = f"figures/{vector_pdf.name}" if vector_pdf.is_file() else f"figures/{png.name}"
    return image_ref, tex, png


def _figure_latex(caption: str, target: str) -> str:
    image_ref, tikz_path, png_path = _figure_assets(target)
    stem = Path(target).stem
    caption = re.sub(r"^(?:Fig(?:ure)?\.?\s*)\d+\s*[.:]\s*", "", caption.strip(), flags=re.I)
    caption_tex = latex_inline(caption, KNOWN_KEYS)
    label = re.sub(r"[^A-Za-z0-9_:.-]+", "-", stem)
    if stem == "master_architecture":
        visual = (rf"\resizebox{{.96\textwidth}}{{!}}{{\input{{{tikz_path}}}}}" if tikz_path else
                  rf"\includegraphics[width=.96\textwidth,height=.82\textheight,keepaspectratio]{{{image_ref}}}")
        return ("\\begin{figure*}[p]\n\\centering\n" + visual + "\n" +
                "\\caption{" + caption_tex + "}\\label{fig:" + label + "}\n\\end{figure*}\n")
    is_wide = stem in {"model_comparison", "training_history", "encoder_adaptation", "matched_evaluation"}
    env = "figure*" if is_wide else "figure"
    width = ".84\\textwidth" if stem == "encoder_adaptation" else ".94\\textwidth" if is_wide else ".98\\columnwidth"
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


TABLE_LAYOUTS = {
    "capability": ("feature_traceability", "Capability, Implementation Location, and Evidence Status"),
    "encoder": ("encoder_properties", "Installed Encoder Properties and Tokenizers"),
    "checkpoint output": ("emotion_labels", "Seven-Emotion Label Definitions"),
    "data resource": ("dataset_partitions", "Dataset Resources and Frozen Partitions"),
    "emotion": ("class_f1", "Per-Class F1 in Each Recorded Test Scope"),
    "confidence measure": ("calibration_fit", "Probability Quality Before and After Temperature Scaling"),
    "model": ("matched_scores", "Matched Hindi-English Replay on 1,000 Synthetic Test Rows"),
    "model pair": ("paired_errors", "Paired Correctness on the Shared Hindi-English Scoring Set"),
    "checking set": ("checking_diagnostics", "Unreviewed Checking-Phrase Diagnostics"),
    "method on bilingual test": ("bilingual_scores", "Matched Bilingual Synthetic Test Comparison"),
}
TABLE_COUNTER = 0

def _table_latex(lines: list[str]) -> str:
    global TABLE_COUNTER
    TABLE_COUNTER += 1
    rows = _table_rows(lines)
    if not rows:
        return ""
    key = rows[0][0].casefold()
    if key not in TABLE_LAYOUTS:
        raise ValueError(f"Table header needs a caption/label mapping: {rows[0][0]}")
    label, title = TABLE_LAYOUTS[key]
    columns = len(rows[0])
    # X columns wrap long descriptors; right-aligned X is used for numeric data.
    specs = []
    for ci in range(columns):
        values = [row[ci].strip() for row in rows[1:] if row[ci].strip()]
        numeric = bool(values) and all(re.fullmatch(r"[<>=~+-]?[\d][\d,.%/() -]*", value) for value in values)
        alignment = r"\raggedleft" if numeric and len(rows[0][ci]) <= 18 else r"\raggedright"
        specs.append(r">{" + alignment + r"\arraybackslash}X")
    spec = "@{}" + "".join(specs) + "@{}"
    wide = label in {"feature_traceability", "encoder_properties"}
    environment = "table*" if wide else "table"
    table_width = r"\textwidth" if wide else r"\columnwidth"
    rendered = [f"\\begin{{{environment}}}[!t]", "\\caption{" + title + "}\\label{tab:" + label + "}", "\\centering", "\\footnotesize", rf"\begin{{tabularx}}{{{table_width}}}{{{spec}}}", "\\toprule"]
    for ri, row in enumerate(rows):
        cells = [latex_inline(value, KNOWN_KEYS) for value in row]
        if ri == 0:
            cells = [r"\textbf{" + value + "}" for value in cells]
        rendered.append(" & ".join(cells) + r" \\")
        if ri == 0:
            rendered.append("\\midrule")
    rendered.extend(["\\bottomrule", "\\end{tabularx}", f"\\end{{{environment}}}"])
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
    markdown = re.sub(r"(?m)^\*\*(?:Keywords|Index Terms):\*\*\s*(.+)$", r"## Keywords\n\n\1", markdown)
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
                listing = "\\begin{lstlisting}\n" + "\n".join(code_lines) + "\n\\end{lstlisting}"
                if output and output[-1].startswith(r"\textbf{Algorithm"):
                    heading = output.pop()
                    listing = "\\par\\smallskip\\noindent\\begin{minipage}[t]{\\columnwidth}\n" + heading + "\n" + listing + "\n\\end{minipage}\\par\\smallskip"
                output.append(listing)
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
        equation = re.fullmatch(r"\[\[equation:([a-z_0-9]+)\]\]", line)
        if equation:
            from ieee_equations import LATEX
            key = equation.group(1)
            expression = LATEX[key]
            output.append("\\begin{equation}\n" + expression + "\n\\label{eq:" + key + "}\n\\end{equation}")
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
            if not nxt or nxt.startswith(("#", "|", "![", "[[equation:") ) or FENCE_RE.match(nxt) or re.match(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s)", nxt):
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


def render_main(records: list[dict], body: str, title: str) -> str:
    if not records:
        raise ValueError("At least one reference is required")
    template = r"""\documentclass[journal]{IEEEtran}
% Generic IEEE journal manuscript; apply the selected venue's extra requirements.
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{tabularx,booktabs,array}
\usepackage{url,cite,placeins}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc}
\usepackage{listings}
\usepackage[hidelinks]{hyperref}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,frame=single,framesep=3pt,showstringspaces=false}
\title{__PAPER_TITLE__}
\input{authors}
\hypersetup{pdftitle={__PAPER_TITLE__},pdfauthor={Advay Vivek, Kaavish Gogia, Suganthini C}}
\begin{document}
\bstctlcite{IEEEcontrol}
\raggedbottom
% Permit the dedicated architecture page without deferring the figure queue.
\renewcommand{\dblfloatpagefraction}{0.6}
\renewcommand{\floatpagefraction}{0.6}
\maketitle
""" + body + r"""
% Balance the final reference columns in this expanded edition.
\IEEEtriggeratref{20}
\begingroup
\interlinepenalty=10000
\bibliographystyle{IEEEtran}
\bibliography{references}
\endgroup
\end{document}
"""
    return template.replace("__PAPER_TITLE__", latex_escape(title))


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
    global TABLE_COUNTER
    TABLE_COUNTER = 0
    body = render_body(markdown, records)
    title = markdown.splitlines()[0].removeprefix("# ").strip()
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    MAIN_TEX.write_text(render_main(records, body, title), encoding="utf-8")
    (OVERLEAF / "authors.tex").write_text(r"""% Author identities and correspondence supplied by the project team.
\author{Advay~Vivek, Kaavish~Gogia, and Suganthini~C%
\thanks{\textit{Corresponding author: Suganthini C.}}%
\thanks{The authors are with Vellore Institute of Technology, Katpadi--Thiruvalam Road, Vellore, Tamil Nadu 632014, India (e-mail: \href{mailto:suganthini.c@vit.ac.in}{suganthini.c@vit.ac.in}).}}
""", encoding="utf-8")
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
    readme = """# CodeMood IEEE journal manuscript - editable Overleaf source

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
"""
    README.write_text(readme, encoding="utf-8")
    audit = BASE / "verification_report.json"
    if audit.is_file():
        shutil.copy2(audit, OVERLEAF / audit.name)
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OVERLEAF.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OVERLEAF).as_posix())
    print(f"{MAIN_TEX}\n{BIB_FILE}\n{README}\n{OUTPUT_ZIP}")
    print(f"Copied {len(copied)} optional figure/source assets")


if __name__ == "__main__":
    build()
