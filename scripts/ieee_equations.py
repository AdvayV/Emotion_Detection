"""Native editable Word equations for the IEEE manuscript.

The finite registry deliberately mirrors the manuscript's LaTeX equations.
No raster images, external math renderer, Word COM, or formula parser is used.

Usage::
    paragraph = document.add_paragraph()
    append_equation(paragraph, "softmax")
    paragraph.add_run("\\t(1)")  # Caller owns tabs and equation numbering.

``append_equation`` appends an inline ``m:oMath`` to a normal Word paragraph.
This permits a right-aligned equation number in the same paragraph.  For an
unnumbered standalone display, use ``append_display_equation`` instead.  The
caller should use automatic/at-least line spacing, never an exact line height:
stacked fractions and summation limits must be allowed to increase the line.
"""
from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


LATEX = {
    "kl": r"\mathrm{KL}(P\Vert M)=\sum_{k:P_k>0}P_k\log_2\!\left(\frac{P_k}{M_k}\right)",
    "worked_f1": r"F_{1,\mathrm{sad}}=\frac{16}{22}\approx0.7273",
    "class_weights": r"w_c=\frac{n_{\mathrm{train}}}{7\max(n_c,1)}",
    "weighted_ce": r"L_{\mathcal B}=-\frac{\sum_{i\in\mathcal B}w_{y_i}\ln(p_{i,y_i})}{\sum_{i\in\mathcal B}w_{y_i}}",
    "softmax": r"p_k=\frac{e^{z_k}}{\sum_{j=1}^{7}e^{z_j}},\quad k=1,\ldots,7",
    "argmax": r"\hat y=\operatorname*{arg\,max}_{k\in\{1,\ldots,7\}}p_k",
    "precision": r"P=\frac{TP}{TP+FP}",
    "recall": r"R=\frac{TP}{TP+FN}",
    "f1": r"F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}",
    "macro_f1": r"F_{1,\mathrm{macro}}=\frac{1}{7}\sum_{k=1}^{7}F_{1,k}",
    "accuracy": r"\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\hat y_i=y_i)",
    "temperature": r"p_k^{(T)}=\frac{p_k^{1/T}}{\sum_{j=1}^{7}p_j^{1/T}}",
    "nll": r"\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\ln\!\left(\max(p_{i,y_i},\epsilon)\right)",
    "brier": r"\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{7}(p_{ik}-y_{ik})^2",
    "ece": r"\mathrm{ECE}=\sum_{b=1}^{B}\frac{|B_b|}{N}\left|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\right|",
    "js": r"\mathrm{JS}(P,Q)=\frac{\mathrm{KL}(P\Vert M)+\mathrm{KL}(Q\Vert M)}{2}",
    "js_mixture": r"M=\frac{P+Q}{2}",
    "js_distance": r"d_{\mathrm{JS}}(P,Q)=\sqrt{\mathrm{JS}(P,Q)}",
}

ALIASES = {
    "macro-f1": "macro_f1", "macrof1": "macro_f1", "f1_score": "f1",
    "temperature_scaling": "temperature", "negative_log_likelihood": "nll",
    "brier_score": "brier", "jensen_shannon": "js", "djs": "js_distance",
    "weighted_cross_entropy": "weighted_ce", "class_weight": "class_weights",
}


def _el(name: str, *children, **attributes):
    node = OxmlElement(name)
    for key, value in attributes.items():
        node.set(qn("m:" + key), str(value))
    for child in children:
        if child is not None:
            node.append(child)
    return node


def _parts(value):
    if isinstance(value, str):
        return [_run(value)]
    if hasattr(value, "tag"):
        return [value]
    return list(value)


def _box(name, value):
    return _el(name, *_parts(value))


def _run(text, *, upright=False):
    node = _el("m:r")
    if upright:
        node.append(_el("m:rPr", _el("m:sty", val="p")))
    value = _el("m:t")
    if text.startswith(" ") or text.endswith(" "):
        value.set(qn("xml:space"), "preserve")
    value.text = text
    node.append(value)
    return node


def _seq(*values):
    result = []
    for value in values:
        result.extend(_parts(value))
    return result


def _frac(top, bottom):
    return _el("m:f", _el("m:fPr", _el("m:type", val="bar")),
               _box("m:num", top), _box("m:den", bottom))


def _sub(base, index):
    return _el("m:sSub", _box("m:e", base), _box("m:sub", index))


def _sup(base, power):
    return _el("m:sSup", _box("m:e", base), _box("m:sup", power))


def _subsup(base, index, power):
    return _el("m:sSubSup", _box("m:e", base), _box("m:sub", index),
               _box("m:sup", power))


def _delim(value, left="(", right=")"):
    return _el("m:d", _el("m:dPr", _el("m:begChr", val=left),
                          _el("m:endChr", val=right),
                          _el("m:grow", val="1")), _box("m:e", value))


def _sum(index, upper, body):
    properties = _el("m:naryPr", _el("m:chr", val="\u2211"),
                     _el("m:limLoc", val="undOvr"),
                     _el("m:grow", val="1"),
                     _el("m:subHide", val="0"), _el("m:supHide", val="0"))
    return _el("m:nary", properties, _box("m:sub", index),
               _box("m:sup", upper), _box("m:e", body))


def _sum_over(index, body):
    properties = _el("m:naryPr", _el("m:chr", val="\u2211"),
                     _el("m:limLoc", val="undOvr"), _el("m:grow", val="1"),
                     _el("m:subHide", val="0"), _el("m:supHide", val="1"))
    return _el("m:nary", properties, _box("m:sub", index),
               _el("m:sup"), _box("m:e", body))


def _hat(value):
    return _el("m:acc", _el("m:accPr", _el("m:chr", val="\u0302")),
               _box("m:e", value))


def _sqrt(value):
    return _el("m:rad", _el("m:radPr", _el("m:degHide", val="1")),
               _el("m:deg"), _box("m:e", value))


def _name(text):
    return _run(text, upright=True)


def _call(name, arguments):
    return _seq(_name(name), _delim(arguments))


def _kl(first, second):
    return _call("KL", _seq(first, "\u2225", second))


def _formula(key):
    if key == "kl":
        index = _seq("k:", _sub("P", "k"), ">0")
        ratio = _frac(_sub("P", "k"), _sub("M", "k"))
        body = _seq(_sub("P", "k"), _sub(_name("log"), "2"), _delim(ratio))
        return _seq(_kl("P", "M"), "=", _sum_over(index, body))
    if key == "worked_f1":
        return _seq(_sub("F", _seq("1,", _name("sad"))), "=", _frac("16", "22"), "\u22480.7273")
    if key == "class_weights":
        return _seq(_sub("w", "c"), "=", _frac(
            _sub("n", _name("train")),
            _seq("7", _call("max", _seq(_sub("n", "c"), ",1")))))
    if key == "weighted_ce":
        # PyTorch weighted CrossEntropyLoss mean reduction divides by the
        # sum of target-class weights in the actual batch, not batch size.
        def index():
            return _seq("i\u2208", _run("\u212c"))
        def weight():
            return _sub("w", _sub("y", "i"))
        probability = _sub("p", _seq("i,", _sub("y", "i")))
        return _seq(_sub("L", _run("\u212c")), "=\u2212", _frac(
            _sum_over(index(), _seq(weight(), _call("ln", probability))),
            _sum_over(index(), weight())))
    if key == "softmax":
        return _seq(_sub("p", "k"), "=", _frac(
            _sup("e", _sub("z", "k")),
            _sum("j=1", "7", _sup("e", _sub("z", "j")))),
            ",  k=1,\u2026,7")
    if key == "argmax":
        maximum = _el("m:limLow", _box("m:e", _name("arg max")),
                      _box("m:lim", _seq("k\u2208", _delim("1,\u2026,7", "{", "}"))))
        return _seq(_hat("y"), "=", maximum, "\u2009", _sub("p", "k"))
    if key == "precision":
        return _seq("P=", _frac(_name("TP"), _seq(_name("TP"), "+", _name("FP"))))
    if key == "recall":
        return _seq("R=", _frac(_name("TP"), _seq(_name("TP"), "+", _name("FN"))))
    if key == "f1":
        return _seq(_sub("F", "1"), "=", _frac("2PR", "P+R"), "=",
                    _frac(_seq("2", _name("TP")),
                          _seq("2", _name("TP"), "+", _name("FP"), "+", _name("FN"))))
    if key == "macro_f1":
        return _seq(_sub("F", _seq("1,", _name("macro"))), "=", _frac("1", "7"),
                    _sum("k=1", "7", _sub("F", "1,k")))
    if key == "accuracy":
        indicator = _seq(_name("\U0001d540"), _delim(_seq(_sub(_hat("y"), "i"), "=", _sub("y", "i"))))
        return _seq(_name("Accuracy"), "=", _frac("1", "N"), _sum("i=1", "N", indicator))
    if key == "temperature":
        return _seq(_subsup("p", "k", "(T)"), "=", _frac(
            _subsup("p", "k", _frac("1", "T")),
            _sum("j=1", "7", _subsup("p", "j", _frac("1", "T")))))
    if key == "nll":
        probability = _sub("p", _seq("i,", _sub("y", "i")))
        maximum = _call("max", _seq(probability, ",", "\u03b5"))
        return _seq(_name("NLL"), "=\u2212", _frac("1", "N"),
                    _sum("i=1", "N", _call("ln", maximum)))
    if key == "brier":
        squared_error = _sup(_delim(_seq(_sub("p", "ik"), "\u2212", _sub("y", "ik"))), "2")
        return _seq(_name("Brier"), "=", _frac("1", "N"),
                    _sum("i=1", "N", _sum("k=1", "7", squared_error)))
    if key == "ece":
        bin_size = _delim(_sub("B", "b"), "|", "|")
        difference = _seq(_call("acc", _sub("B", "b")), "\u2212", _call("conf", _sub("B", "b")))
        return _seq(_name("ECE"), "=", _sum("b=1", "B", _seq(
            _frac(bin_size, "N"), _delim(difference, "|", "|"))))
    if key == "js":
        return _seq(_call("JS", "P,Q"), "=", _frac(_seq(_kl("P", "M"), "+", _kl("Q", "M")), "2"))
    if key == "js_mixture":
        return _seq("M=", _frac("P+Q", "2"))
    if key == "js_distance":
        return _seq(_sub("d", _name("JS")), _delim("P,Q"), "=", _sqrt(_call("JS", "P,Q")))
    raise KeyError("Unknown equation key: " + repr(key))


def _format_math(math, font_size):
    """Attach explicit Word run/control formatting without changing math style."""
    half_points = str(round(float(font_size) * 2))

    def properties():
        value = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn("w:" + attribute), "Cambria Math")
        value.append(fonts)
        for tag in ("w:sz", "w:szCs"):
            size = OxmlElement(tag)
            size.set(qn("w:val"), half_points)
            value.append(size)
        return value

    for run in math.iter(qn("m:r")):
        run.insert(1 if len(run) and run[0].tag == qn("m:rPr") else 0, properties())
    # ctrlPr supplies the same size/font to the fraction bar, delimiters,
    # radicals, accents and large operators rather than only their text.
    property_names = {"fPr", "sSubPr", "sSupPr", "sSubSupPr", "dPr", "naryPr", "accPr", "radPr", "limLowPr"}
    construct_names = {"sSub": "sSubPr", "sSup": "sSupPr", "sSubSup": "sSubSupPr", "limLow": "limLowPr"}
    for node in list(math.iter()):
        local_name = node.tag.rsplit("}", 1)[-1]
        if local_name in construct_names:
            name = construct_names[local_name]
            if node.find(qn("m:" + name)) is None:
                node.insert(0, _el("m:" + name))
    for node in math.iter():
        if node.tag.rsplit("}", 1)[-1] in property_names:
            node.append(_el("m:ctrlPr", properties()))


def append_equation(paragraph, key: str, *, font_size: float = 9.0):
    """Append and return an editable ``m:oMath`` for a registered equation.

    The caller owns paragraph placement, alignment and numbering. All glyphs
    and structures use Cambria Math; ``font_size`` is in points. No drawing,
    embedded object, or plain-text formula fallback is generated.
    """
    key = ALIASES.get(key, key)
    if key not in LATEX:
        raise KeyError("Unknown equation key: " + repr(key))
    if not 8 <= float(font_size) <= 14:
        raise ValueError("Equation font_size must be between 8 and 14 points")
    math = _el("m:oMath", *_formula(key))
    _format_math(math, font_size)
    paragraph._p.append(math)
    return math


def append_display_equation(paragraph, key: str, *, font_size: float = 9.0):
    """Append and return a centered standalone ``m:oMathPara`` display."""
    math = append_equation(paragraph, key, font_size=font_size)
    paragraph._p.remove(math)
    display = _el("m:oMathPara", _el("m:oMathParaPr", _el("m:jc", val="center")), math)
    paragraph._p.append(display)
    return display


__all__ = ["LATEX", "ALIASES", "append_equation", "append_display_equation"]

# Finite inline symbols keep prose notation consistent with the display equations.
INLINE_LATEX = {
    "epsilon_floor": r"\epsilon=10^{-300}",
    "p_k": r"p_k", "P_k": r"P_k", "n_train": r"n_{\mathrm{train}}",
    "n_c": r"n_c", "w_c": r"w_c", "y_i": r"y_i",
    "p_i,y_i": r"p_{i,y_i}", "y_ik": r"y_{ik}", "B_b": r"B_b",
    "acc(B_b)": r"\mathrm{acc}(B_b)", "conf(B_b)": r"\mathrm{conf}(B_b)",
}

def append_inline_symbol(paragraph, key, font_size=10.0):
    if key == "epsilon_floor": value = _seq("\u03b5=", _sup("10", "\u2212300"))
    elif key == "p_i,y_i": value = _sub("p", _seq("i,", _sub("y", "i")))
    elif key in {"acc(B_b)", "conf(B_b)"}: value = _call(key.split("(")[0], _sub("B", "b"))
    elif key == "n_train": value = _sub("n", _name("train"))
    else:
        base, index = key.split("_", 1)
        value = _sub(base, index)
    math = _el("m:oMath", *_parts(value))
    _format_math(math, font_size)
    paragraph._p.append(math)
    return math
