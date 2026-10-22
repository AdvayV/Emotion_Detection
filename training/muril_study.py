"""Compare saved Hindi-English emotion runs without loading model weights.

The contract is built only after reproducing the installed XLM-R dataset and
split hashes. MuRIL and the optional matched XLM-R must reproduce its Hindi-
English subset. Aggregate scores are evidence of a run, not new predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import zipfile
from pathlib import Path

from hinglish_emotion.tasks import EMOTION_LABELS
from training.data_workflow import records_hash


def test_fingerprint(frame):
    """Bind phrase identities, texts and labels independently of extra columns."""
    columns = ["row_id", "text", "emotion", "language_pair"]
    return records_hash(frame[columns].sort_values("row_id").reset_index(drop=True))


def _validate_run(metrics, expected, base_model):
    if metrics.get("task") != "emotion" or metrics.get("base_model") != base_model:
        raise ValueError("Unexpected task or encoder in saved study metrics")
    if list(metrics.get("labels", [])) != list(EMOTION_LABELS):
        raise ValueError("The study requires the same seven ordered emotion labels")
    for key in ("dataset_sha256", "split_sha256"):
        if metrics.get(key) != expected[key]:
            raise ValueError(f"Saved {base_model} {key} differs from the study contract; do not compare these runs")
    if metrics.get("split_sizes") != expected["split_sizes"]:
        raise ValueError("Saved split sizes differ from the study contract")
    if metrics.get("evaluation_scope") != "synthetic_only":
        raise ValueError("This contract covers the synthetic study only")
    settings = expected.get("settings", {})
    if any(metrics.get(key) != value for key, value in settings.items()):
        raise ValueError("Saved preparation or training settings differ from this study")
    scores = metrics.get("test_by_language", {}).get("hi-en")
    if not scores or scores.get("rows") != expected["hi_en_test_rows"]:
        raise ValueError("Missing Hindi-English test metrics or different test size")
    for key in ("accuracy", "macro_f1"):
        value = scores.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid held-out {key}")
    per_class = scores.get("per_class", {})
    if set(per_class) != set(EMOTION_LABELS):
        raise ValueError("Missing per-emotion test metrics")
    for label, values in per_class.items():
        if values.get("support") != expected["hi_en_test_support"][label]:
            raise ValueError("Different Hindi-English class supports")
        for key in ("precision", "recall", "f1"):
            value = values.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid per-emotion score")
    return scores


def validate_baseline(contract, baseline):
    """Validate the recorded baseline even when MuRIL has not returned yet."""
    if contract.get("schema_version") != 1 or contract.get("language_pair") != "hi-en":
        raise ValueError("Unsupported MuRIL study contract")
    if hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest() != contract["baseline_metrics_sha256"]:
        raise ValueError("The saved XLM-R baseline has changed")
    return _validate_run(baseline, contract["baseline"], "FacebookAI/xlm-roberta-base")


def compare_runs(contract, baseline, muril, matched=None):
    """Return serializable, language-specific scores after provenance checks."""
    validate_baseline(contract, baseline)
    runs = [("XLM-R combined", baseline, contract["baseline"], "FacebookAI/xlm-roberta-base", "Hindi-English + English-French"),
            ("MuRIL Hindi-English", muril, contract["matched"], "google/muril-base-cased", "Hindi-English")]
    if matched is not None:
        runs.append(("XLM-R Hindi-English (matched)", matched, contract["matched"], "FacebookAI/xlm-roberta-base", "Hindi-English"))
    rows, classes = [], []
    for name, metrics, expected, base, scope in runs:
        score = _validate_run(metrics, expected, base)
        sizes = metrics["split_sizes"]
        rows.append({"Model": name, "Training phrases": sizes["train"], "Validation phrases": sizes["validation"],
                     "Test phrases": score["rows"], "Accuracy": score["accuracy"], "Macro-F1": score["macro_f1"], "Training scope": scope})
        for label in EMOTION_LABELS:
            values = score["per_class"][label]
            classes.append({"Model": name, "Emotion": label, "Precision": values["precision"],
                            "Recall": values["recall"], "F1": values["f1"], "Support": values["support"]})
    notes = [
        "All scores use the same 1,000 held-out Hindi-English phrase identities and labels, bound by the study's dataset and split hashes.",
        "These are synthetic template results, not evidence of real-world emotion accuracy or generalization.",
        "Combined XLM-R used 16,000 training phrases across two language mixes; MuRIL uses 8,000 Hindi-English phrases. This comparison does not isolate the encoder's effect.",
        "Macro-F1 weights each of the seven emotions equally; accuracy counts the fraction of correctly classified phrases.",
        "No statistical significance, latency, memory-use or human-labelled benchmark claim is made by this report.",
    ]
    if matched is not None:
        notes.append("The additional Hindi-English XLM-R and MuRIL runs share data, splits and declared settings. This is a matched single-seed comparison, not a multi-seed significance study.")
    return {"comparison_type": "same_test_with_matched_pair" if matched is not None else "same_test_different_training_scope",
            "test_sha256": contract["hi_en_test_sha256"], "rows": rows, "per_class": classes, "notes": notes}


def load_comparison_export(archive, contract, baseline):
    """Read a Colab report without treating aggregate scores as model weights.

    Report identity and score consistency can be checked here. Full training
    provenance still requires the checkpoint's original metrics.json; this
    function never manufactures that file from the report.
    """
    baseline_score = validate_baseline(contract, baseline)
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            allowed = {"comparison.json", "comparison.csv", "comparison.svg", "per_emotion.csv", "READ_ME.txt"}
            if (len(entries) > 5 or len({item.filename for item in entries}) != len(entries)
                    or any(item.filename not in allowed or item.file_size > 1024 ** 2 for item in entries)):
                raise ValueError("Unexpected comparison export contents")
            report = json.loads(source.read("comparison.json"))
        if report["comparison_type"] != "same_test_different_training_scope" or report["test_sha256"] != contract["hi_en_test_sha256"]:
            raise ValueError("Exported comparison uses a different test set or study type")
        rows, classes = report["rows"], report["per_class"]
        if len(rows) != 2 or len(classes) != 14:
            raise ValueError("Expected two model summaries and fourteen emotion results")
        expected_runs = [("XLM-R combined", contract["baseline"], "Hindi-English + English-French"),
                         ("MuRIL Hindi-English", contract["matched"], "Hindi-English")]
        for row, (name, expected, scope) in zip(rows, expected_runs):
            if any(row[key] != value for key, value in {
                "Model": name, "Training phrases": expected["split_sizes"]["train"],
                "Validation phrases": expected["split_sizes"]["validation"],
                "Test phrases": expected["hi_en_test_rows"], "Training scope": scope}.items()):
                raise ValueError("Exported comparison differs from the study's model or split sizes")
            details = [item for item in classes if item["Model"] == name]
            if len(details) != 7 or {item["Emotion"] for item in details} != set(EMOTION_LABELS):
                raise ValueError("Missing or duplicate per-emotion results")
            for detail in details:
                label = detail["Emotion"]
                if detail["Support"] != expected["hi_en_test_support"][label]:
                    raise ValueError("Exported class supports differ from the study")
                for key in ("Precision", "Recall", "F1"):
                    value = detail[key]
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid per-emotion score in comparison export")
                p, r = detail["Precision"], detail["Recall"]
                if not math.isclose(detail["F1"], 2*p*r/(p+r) if p+r else 0, abs_tol=1e-8):
                    raise ValueError("Inconsistent per-emotion F1 in comparison export")
                if name == "XLM-R combined":
                    for key in ("precision", "recall", "f1", "support"):
                        if detail[key.capitalize() if key != "f1" else "F1"] != baseline_score["per_class"][label][key]:
                            raise ValueError("Exported XLM-R results differ from the saved baseline")
            calculated = {"Macro-F1": sum(item["F1"] for item in details)/7,
                          "Accuracy": sum(item["Recall"]*item["Support"] for item in details)/expected["hi_en_test_rows"]}
            for key, calculated_value in calculated.items():
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1 or not math.isclose(value, calculated_value, abs_tol=1e-8):
                    raise ValueError("Inconsistent aggregate scores in comparison export")
            if name == "XLM-R combined" and (row["Accuracy"] != baseline_score["accuracy"] or row["Macro-F1"] != baseline_score["macro_f1"]):
                raise ValueError("Exported XLM-R summary differs from the saved baseline")
        # Use local explanatory text rather than trusting claims in an imported report.
        report["notes"] = [
            "These are saved Colab report scores on 1,000 synthetic Hindi-English test phrases, not real-world accuracy.",
            "Combined XLM-R used 16,000 multilingual training phrases; MuRIL used 8,000 Hindi-English phrases. This comparison does not isolate the encoder's effect.",
            "Report test identity, baseline and score consistency were checked. Full MuRIL training provenance and live predictions require its separate checkpoint export.",
        ]
        return report
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read comparison export: {exc}") from exc


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def chart_svg(report):
    """Portable research figure; no model, browser or plotting dependency."""
    rows = report["rows"]
    height = 160 + 100 * len(rows)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#18283c">',
           '<text x="28" y="32" font-size="22">Hindi-English: same 1,000 synthetic test phrases</text>',
           '<text x="28" y="58" font-size="14">Blue: accuracy · teal: macro-F1 · scores from saved test runs</text>']
    for tick in range(0, 101, 20):
        x = 320 + tick * 5.5
        out.append(f'<line x1="{x}" y1="82" x2="{x}" y2="{height-65}" stroke="#e1e6ed"/>')
        out.append(f'<text x="{x}" y="{height-42}" text-anchor="middle" font-size="13">{tick}%</text>')
    for index, row in enumerate(rows):
        y = 100 + index * 100
        name = html.escape(row["Model"])
        out.append(f'<text x="28" y="{y+20}" font-size="16">{name}</text>')
        for offset, key, color in ((0, "Accuracy", "#355d91"), (30, "Macro-F1", "#27847a")):
            score = row[key]
            out.append(f'<rect x="320" y="{y+offset}" width="{score*550:.2f}" height="22" fill="{color}"/>')
            out.append(f'<text x="{330+score*550:.2f}" y="{y+offset+17}" font-size="14">{score:.2%}</text>')
    out.append(f'<text x="28" y="{height-12}" font-size="12">Synthetic study. Combined XLM-R has additional French-English training data.</text></g></svg>')
    return "\n".join(out)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "comparison.csv").write_text(csv_text(report["rows"]), encoding="utf-8-sig", newline="")
    (output / "per_emotion.csv").write_text(csv_text(report["per_class"]), encoding="utf-8-sig", newline="")
    (output / "comparison.svg").write_text(chart_svg(report), encoding="utf-8")
    (output / "READ_ME.txt").write_text("XLM-R / MuRIL Hindi-English study\n\n" + "\n\n".join(report["notes"]), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("colab/muril_study"))
    parser.add_argument("--muril", type=Path, default=Path("models/muril_hi_en_emotion/metrics.json"))
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/muril_study"))
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    report = compare_runs(read(args.study / "study_contract.json"), read(args.study / "xlmr_baseline_metrics.json"), read(args.muril), read(args.matched) if args.matched else None)
    print(write_report(report, args.output))


if __name__ == "__main__":
    main()
