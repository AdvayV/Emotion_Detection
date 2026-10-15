"""Collect phrases, obtain independent labels, and reconcile the pilot supplement."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from hinglish_emotion.emotion_data import validate_emotion_rows
from hinglish_emotion.tasks import EMOTION_LABELS, canonical_emotion
from hinglish_emotion.text_preparation import duplicate_key
from training.splits import split_data

QUOTAS = {"train": 20, "validation": 10, "test": 20}
FIELDS = ("row_id", "group_id", "text", "language_pair", "split")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initialise(directory, language_pair="hi-en"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "collection.csv"
    if target.exists():
        raise ValueError("Collection already exists; it was not overwritten")
    rows = []
    for split, per_class in QUOTAS.items():
        count = per_class * 7
        for index in range(count):
            rows.append({
                "row_id": f"pilot_{language_pair}_{split}_{index:03d}",
                "group_id": f"pilot_{language_pair}_{split}_scenario_{index // 2:03d}",
                "text": "", "language_pair": language_pair, "split": split,
            })
    write_csv(target, rows)
    (directory / "quotas.json").write_text(json.dumps({split: {label: count for label in EMOTION_LABELS} for split, count in QUOTAS.items()}, indent=2), encoding="utf-8")
    return target


def collection_rows(path):
    rows = read_csv(path)
    if not rows or not set(FIELDS) <= set(rows[0]):
        raise ValueError("Collection requires row_id, group_id, text, language_pair and split")
    if any(not row["text"].strip() or not row["row_id"].strip() or not row["group_id"].strip() for row in rows):
        raise ValueError("Fill all phrase text and identifiers before generating reviewer sheets")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Collection row identifiers must be unique")
    return [{key: row[key] for key in FIELDS} for row in rows]


def reviewer_sheets(collection, output_dir):
    rows = collection_rows(collection)
    output_dir = Path(output_dir)
    for name in ("reviewer_a.csv", "reviewer_b.csv"):
        if (output_dir / name).exists():
            raise ValueError("Existing reviewer sheets will not be overwritten")
    for name in ("reviewer_a.csv", "reviewer_b.csv"):
        write_csv(output_dir / name, [{**row, "reviewer_id": "", "emotion": "", "needs_context": "", "notes": ""} for row in rows])


def reconcile(collection, reviewer_a, reviewer_b, adjudications=None):
    collected = collection_rows(collection)
    ids = {row["row_id"] for row in collected}
    reviews = []
    for path in (reviewer_a, reviewer_b):
        rows = read_csv(path)
        if len(rows) != len(ids) or {row.get("row_id") for row in rows} != ids:
            raise ValueError("Each reviewer sheet must contain every collection row exactly once")
        reviews.append({row["row_id"]: row for row in rows})
    decisions = {row["row_id"]: row for row in read_csv(adjudications)} if adjudications else {}
    result, pending = [], []
    for original in collected:
        key = original["row_id"]
        left, right = reviews[0][key], reviews[1][key]
        if any(row.get(field) != original[field] for row in (left, right) for field in FIELDS):
            raise ValueError("Reviewers cannot change phrase text, grouping or partitions")
        if not left.get("reviewer_id") or not right.get("reviewer_id") or left["reviewer_id"] == right["reviewer_id"]:
            pending.append({"row_id": key, "reason": "Two distinct reviewer identifiers are required"})
            continue
        labels = [canonical_emotion(row.get("emotion", "")) for row in (left, right)]
        if any(label not in EMOTION_LABELS for label in labels):
            pending.append({"row_id": key, "reason": "Both reviewers must supply a valid emotion"})
            continue
        flagged = any(row.get("needs_context", "").strip().lower() in {"yes", "true", "1"} for row in (left, right))
        decision = decisions.get(key, {})
        if labels[0] != labels[1] or flagged:
            label = canonical_emotion(decision.get("emotion", ""))
            judge = decision.get("adjudicator_id", "")
            if label not in EMOTION_LABELS or not judge or judge in {left["reviewer_id"], right["reviewer_id"]} or not decision.get("notes", "").strip():
                pending.append({"row_id": key, "reason": "Disagreement/context flag requires a distinct adjudicator and a written resolution"})
                continue
        else:
            label = labels[0]
        result.append({**original, "emotion": label, "source": "team_reviewed_pilot", "annotation_status": "human_adjudicated"})
    counts = {split: dict(Counter(row["emotion"] for row in result if row["split"] == split)) for split in QUOTAS}
    shortfalls = [
        f"{split}/{label}: {counts[split].get(label, 0)} reviewed; target {target}"
        for split, target in QUOTAS.items() for label in EMOTION_LABELS if counts[split].get(label, 0) != target
    ]
    report = {"ready": not pending and not shortfalls, "reviewed_rows": len(result), "pending": pending, "counts": counts, "quota_issues": shortfalls}
    if report["ready"]:
        result = validate_emotion_rows(result)
        frame = pd.DataFrame(result)
        frame["label"] = frame["emotion"]
        frame["duplicate_key"] = frame["text"].map(duplicate_key)
        split_data(frame, 42)
    return result, report


def combine(public_csv, public_audit, supplement, supplement_audit):
    audit = json.loads(Path(public_audit).read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(public_csv).read_bytes()).hexdigest()
    if not audit.get("reuse_conditions_verified") or digest != audit.get("prepared_csv_sha256"):
        raise ValueError("Public data requires a matching approved reuse audit")
    receipt = json.loads(Path(supplement_audit).read_text(encoding="utf-8"))
    supplement_hash = hashlib.sha256(Path(supplement).read_bytes()).hexdigest()
    if not receipt.get("ready") or receipt.get("prepared_csv_sha256") != supplement_hash:
        raise ValueError("Supplement requires its matching completed reconciliation report")
    reviewed = validate_emotion_rows(read_csv(supplement))
    if any(row.get("annotation_status") != "human_adjudicated" for row in reviewed):
        raise ValueError("Supplement rows must come from completed human reconciliation")
    rows = validate_emotion_rows([*read_csv(public_csv), *reviewed])
    frame = pd.DataFrame(rows).fillna("")
    if frame["row_id"].duplicated().any():
        raise ValueError("Row identifiers collide between datasets")
    frame["label"] = frame["emotion"]
    frame["duplicate_key"] = frame["text"].map(duplicate_key)
    split_data(frame, 42)
    return frame.drop(columns=["label", "duplicate_key"]).to_dict("records"), {
        **audit, "public_csv_sha256": digest,
        "supplement_sha256": supplement_hash,
        "supplement_audit_sha256": hashlib.sha256(Path(supplement_audit).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output-dir", default="data/annotation")
    init.add_argument("--language-pair", choices=("hi-en", "fr-en"), default="hi-en")
    sheets = sub.add_parser("review-sheets")
    sheets.add_argument("--collection", required=True)
    sheets.add_argument("--output-dir", default="data/annotation")
    merge = sub.add_parser("reconcile")
    merge.add_argument("--collection", required=True)
    merge.add_argument("--reviewer-a", required=True)
    merge.add_argument("--reviewer-b", required=True)
    merge.add_argument("--adjudications")
    merge.add_argument("--output", default="data/prepared/pilot_supplement.csv")
    merge.add_argument("--report", default="data/audits/pilot_annotations.json")
    build = sub.add_parser("build-pilot")
    build.add_argument("--public-csv", required=True)
    build.add_argument("--public-audit", required=True)
    build.add_argument("--supplement", required=True)
    build.add_argument("--supplement-audit", default="data/audits/pilot_annotations.json")
    build.add_argument("--output", default="data/prepared/pilot_hi.csv")
    build.add_argument("--report", default="data/audits/pilot_hi.json")
    args = parser.parse_args()
    if args.command == "init":
        print(initialise(args.output_dir, args.language_pair))
    elif args.command == "review-sheets":
        reviewer_sheets(args.collection, args.output_dir)
    elif args.command == "reconcile":
        rows, report = reconcile(args.collection, args.reviewer_a, args.reviewer_b, args.adjudications)
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["ready"]:
            raise SystemExit(f"Review report saved to {target}; export blocked until labels and quotas are resolved.")
        write_csv(args.output, rows)
        report.update(
            prepared_csv_sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
            collection_sha256=hashlib.sha256(Path(args.collection).read_bytes()).hexdigest(),
            reviewer_a_sha256=hashlib.sha256(Path(args.reviewer_a).read_bytes()).hexdigest(),
            reviewer_b_sha256=hashlib.sha256(Path(args.reviewer_b).read_bytes()).hexdigest(),
        )
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(args.output)
    else:
        rows, report = combine(args.public_csv, args.public_audit, args.supplement, args.supplement_audit)
        write_csv(args.output, rows)
        report["prepared_csv_sha256"] = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
