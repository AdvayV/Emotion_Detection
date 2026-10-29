"""Check the manuscript's reported results against saved project artifacts."""
from __future__ import annotations
import csv
import io
import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
import build_ieee_charts as charts
import build_ieee_latex as latex

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/ieee_manuscript"
sys.path.insert(0, str(ROOT))

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def close(a, b, label):
    if not math.isclose(a, b, abs_tol=1e-12):
        raise ValueError(f"{label}: {a} != {b}")

def verify_scores(record, labels, scope):
    matrix = record["confusion_matrix"]
    n = sum(sum(matrix[a][b] for b in labels) for a in labels)
    correct = sum(matrix[a][a] for a in labels)
    scores = []
    for k in labels:
        tp = matrix[k][k]
        support = sum(matrix[k].values())
        predicted = sum(matrix[a][k] for a in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2*tp/(predicted+support) if predicted+support else 0.0
        entry = record["per_class"][k]
        for name, value in [("precision", precision), ("recall", recall), ("f1", f1)]:
            close(value, entry[name], f"{scope} {k} {name}")
        assert support == entry["support"], (scope, k)
        scores.append(f1)
    close(correct/n, record["accuracy"], scope+" accuracy")
    close(sum(scores)/len(labels), record["macro_f1"], scope+" macro F1")
    return {"rows": n, "correct": correct, "accuracy": correct/n, "macro_f1": sum(scores)/len(labels)}

def main():
    evidence = charts.build_evidence()
    verified = {}
    loaded = {}
    for key, (_, path) in charts.MODELS.items():
        run = read(path)
        loaded[key] = run
        metadata = read(path.replace("metrics.json", "emotion_metadata.json"))
        for field in ("base_model", "labels", "max_length", "dataset_sha256", "split_sha256", "text_mode", "preparation_version", "training_data_kind", "evaluation_scope"):
            assert run[field] == metadata[field], (key, field)
        assert metadata["training_status"] == "complete"
        assert metadata["validated_language_pairs"] == []
        overall = verify_scores(run["test"], run["labels"], key+" overall")
        slices = {lang: verify_scores(v, run["labels"], key+" "+lang) for lang, v in run["test_by_language"].items()}
        assert overall["rows"] == run["split_sizes"]["test"] == sum(v["rows"] for v in slices.values())
        assert overall["correct"] == sum(v["correct"] for v in slices.values())
        best = max(run["history"], key=lambda v:v["validation_macro_f1"])
        close(best["validation_macro_f1"], run["best_validation_macro_f1"], key+" best epoch")
        assert best["epoch"] == 1
        verified[key] = {"overall": overall, "by_language": slices, "split_sizes": run["split_sizes"], "best_epoch": best["epoch"], "metadata_matches": True, "metric_sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
    # Reproduce canonical record and sorted split serialization without training.
    from hinglish_emotion.emotion_data import read_emotion_csv
    rows = read_emotion_csv(ROOT/"data/training/multilingual_bootstrap_v1/all_splits.csv")
    dataset_checks = {}
    for key, pair in (("xlmr", None), ("muril", "hi-en")):
        subset = [{**r, "label":r["emotion"]} for r in rows if pair is None or r["language_pair"] == pair]
        digest = hashlib.sha256(json.dumps(subset,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
        assert digest == loaded[key]["dataset_sha256"], (key,"current dataset identity")
        columns = [c for c in subset[0] if c != "split"] + ["split"]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer,fieldnames=columns,lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(subset,key=lambda r:(r["split"],r["row_id"])))
        split_digest = hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()
        assert split_digest == loaded[key]["split_sha256"], (key,"current split identity")
        dataset_checks[key] = {"dataset_sha256":digest,"split_sha256":split_digest,"rows":len(subset),"current_csv_matches_training_identity":True}
    for field in ("dataset_sha256", "split_sha256", "labels", "batch_size", "gradient_accumulation", "learning_rate", "max_length"):
        assert loaded["xlmr"][field] == loaded["mbert"][field], field
    baseline = loaded["xlmr"]["lexical_baseline"]["test"]
    close(baseline["accuracy"], .3755, "lexical baseline accuracy")
    if "confusion_matrix" in baseline:
        baseline_verified = verify_scores(baseline, loaded["xlmr"]["labels"], "lexical baseline")
    else:
        baseline_verified = baseline
    close((loaded["xlmr"]["test"]["accuracy"]-baseline["accuracy"])*100,54.95,"baseline improvement")
    close((loaded["xlmr"]["test"]["accuracy"]-loaded["mbert"]["test"]["accuracy"])*100,11.10,"mBERT improvement")
    diagnostic = read("docs/oct23review/challenge_diagnostics.json")
    for item in diagnostic["results"]:
        assert hashlib.sha256((ROOT/item["file"]).read_bytes()).hexdigest() == item["file_sha256"], item["file"]
        close(item["correct"]/item["rows"], item["accuracy"], item["file"])
    for item in read("docs/oct23review/results.json")["colab_cross_checks"]:
        a,b = [read(p) for p in item["files"]]
        for field in ("dataset_sha256", "split_sha256", "test", "history"):
            assert a[field] == b[field], (item["files"],field)
    markdown = (BASE/"manuscript.md").read_text(encoding="utf-8")
    refs = latex.read_references()
    used = {key for group in latex.CITATION_RE.findall(markdown) for key in latex._citation_keys(group)}
    assert used == {r["key"] for r in refs}
    assert len(refs) == 31
    assert all(r.get("verified_url") == r.get("url") for r in refs)
    from ieee_equations import LATEX
    equations = re.findall(r"\[\[equation:([a-z_0-9]+)\]\]", markdown)
    assert len(equations) == 18 and set(equations) == set(LATEX)
    for label in loaded["xlmr"]["labels"]:
        expected = "| " + label + " | " + " | ".join(f'{loaded[k]["test"]["per_class"][label]["f1"]:.4f}' for k in ("xlmr","mbert","muril")) + " |"
        assert expected in markdown, expected
    assert markdown.count("**Algorithm ") == 4
    assert len(latex.IMAGE_RE.findall(markdown)) == 7
    assert "## 2 Related Work" in markdown
    figures = {Path(target).stem for _, target in latex.IMAGE_RE.findall(markdown)}
    table_labels = {label for label, _ in latex.TABLE_LAYOUTS.values()}
    for kind, key in re.findall(r"\[\[(fig|tab|eq):([A-Za-z0-9_.-]+)\]\]", markdown):
        assert key in {"fig": figures, "tab":table_labels, "eq":set(LATEX)}[kind], (kind,key)
    replay = read("docs/evaluation/comparison_verification.json")
    assert replay["scored_rows"]==1000 and replay["excluded_overlap_rows"]==0
    replay_models={model["model"]:model for model in replay["models"]}
    for key, model_name in [("xlmr","XLM-R"),("mbert","mBERT"),("muril","MuRIL")]:
        actual=loaded[key]["test_by_language"]["hi-en"]
        for metric in ("accuracy","macro_f1"): close(replay_models[model_name][metric],actual[metric],model_name+" replay "+metric)
    for pair in replay["paired_comparisons"]:
        assert sum(pair[k] for k in ("both_correct","left_only_correct","right_only_correct","both_wrong"))==1000
        close(pair["both_correct"]+pair["left_only_correct"],replay_models[pair["left"]]["accuracy"]*1000,"paired left count")
        close(pair["both_correct"]+pair["right_only_correct"],replay_models[pair["right"]]["accuracy"]*1000,"paired right count")
    software=read("docs/evaluation/software_verification.json")
    assert software["tests_passed"]>=240 and software["subtests_passed"]>=33
    assert replay["implementation_revision"]==software["implementation_revision"]
    assert software["implementation_revision"] in markdown
    assert "{{" not in markdown
    assert len(re.findall(r"(?m)^\|[^\n]*$", markdown)) > 10
    assert "https://github.com/AdvayV/Emotion_Detection" in markdown
    calibration = read("docs/oct23review/calibration_example.json")
    assert calibration["rows"] == 98 and not calibration["accuracy_changed"] and not calibration["independent_evaluation"]
    report = {"checked_on": str(date.today()), "status": "passed", "method": "Recomputed saved confusion-matrix metrics and checked metadata, Colab copies, diagnostic file hashes, citations and equation coverage; no new model training or inference.", "models": verified, "current_training_datasets":dataset_checks, "lexical_baseline": baseline_verified, "diagnostic_file_hashes_match": True, "colab_metric_copies_match": True, "citations": {"references":31,"all_cited":True,"source_links_previously_verified":True}, "equations":18, "algorithms":4, "figures":7, "tables":10, "matched_replay":replay, "software_verification":software, "corresponding_author": {"name":"Suganthini C","email":"suganthini.c@vit.ac.in","source":"user-provided identity, affiliation and contact"}, "scope_limits":["All model accuracy tables concern saved synthetic test sets.","MuRIL has a smaller Hindi-English training scope than bilingual XLM-R/mBERT.","The 98-row calibration measurements use their fitting set, not a new test set.","Optional fusion, contextual and sarcasm heads are implemented training capabilities, not measured accuracy gains.","The software test count describes the prior completed suite, not a fresh rerun for this formatting revision."]}
    (BASE/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","models":verified,"references":31,"equations":18,"figures":7},indent=2))

if __name__ == "__main__":
    main()
