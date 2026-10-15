"""Audit the official Task A resource; export only after reuse evidence is supplied."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from hinglish_emotion.tasks import EMOTION_LABELS, canonical_emotion
from hinglish_emotion.text_preparation import duplicate_key

SOURCE_URL = "https://github.com/LCS2-IIITD/EDiReF-SemEval2024"
FILES = {"train": "MaSaC_train_erc.json", "validation": "MaSaC_val_erc.json", "test": "MaSaC_test_erc.json"}


def audit_source(source_dir: str | Path):
    source_dir = Path(source_dir)
    files = {split: source_dir / name for split, name in FILES.items()}
    if not all(path.is_file() for path in files.values()):
        raise ValueError("Source directory must contain the three MaSaC Task A JSON files")
    rows, counts, excluded, episodes = [], {}, {}, {}
    hashes = {}
    for split, path in files.items():
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        conversations = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_counts, dropped = Counter(), Counter()
        episodes[split] = set()
        for index, conversation in enumerate(conversations):
            texts = conversation["utterances"]
            labels = conversation.get("emotions", conversation.get("labels"))
            if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) != len(labels):
                raise ValueError(f"{path.name}: mismatched utterance and emotion lists")
            episode = str(conversation.get("episode", "unknown"))
            episodes[split].add(episode)
            # A conversation segment is the grouping unit; episode reuse is reported separately.
            digest = hashlib.sha256(json.dumps([episode, texts], ensure_ascii=False).encode()).hexdigest()[:24]
            for turn, (text, raw_label) in enumerate(zip(texts, labels)):
                label = canonical_emotion(str(raw_label))
                raw_counts[label] += 1
                if label == "contempt":
                    dropped[label] += 1
                    continue
                if label not in EMOTION_LABELS or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path.name}: invalid phrase or label {raw_label!r}")
                rows.append({
                    "row_id": f"masac_{split}_{index:04d}_{turn:04d}",
                    "text": text.strip(), "emotion": label, "language_pair": "hi-en",
                    "group_id": digest, "conversation_id": digest, "episode": episode,
                    "source": "MaSaC_EDiReF_TaskA", "annotation_status": "public_annotation", "split": split,
                })
        counts[split] = dict(raw_counts)
        excluded[split] = dict(dropped)
    owners = defaultdict(set)
    for row in rows:
        owners[duplicate_key(row["text"])].add(row["split"])
    crossing = {key for key, splits in owners.items() if len(splits) > 1}
    safe_rows = [row for row in rows if duplicate_key(row["text"]) not in crossing]
    group_partitions = defaultdict(set)
    for row in safe_rows:
        group_partitions[row["group_id"]].add(row["split"])
    overlapping_groups = sum(len(parts) > 1 for parts in group_partitions.values())
    reuse_files = []
    # Check the repository root and its documentation without treating public visibility as permission.
    ancestor = source_dir
    for _ in range(4):
        reuse_files.extend(path.name for path in ancestor.glob("*") if path.is_file() and any(word in path.name.lower() for word in ("license", "terms", "readme", "copyright")))
        if (ancestor / ".git").exists() or ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    try:
        revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, OSError):
        revision = "unknown"
    coverage = {split: {label: sum(row["split"] == split and row["emotion"] == label for row in safe_rows) for label in EMOTION_LABELS} for split in FILES}
    report = {
        "source_url": SOURCE_URL, "source_revision": revision, "source_sha256": hashes,
        "original_class_counts": counts, "excluded_classes": excluded,
        "quarantined_cross_split_duplicate_rows": len(rows) - len(safe_rows),
        "cross_split_conversation_groups": overlapping_groups, "prepared_class_counts": coverage,
        "shared_episode_counts": {f"{left}/{right}": len(episodes[left] & episodes[right]) for left in episodes for right in episodes if left < right},
        "scope_note": "Official conversation-segment partitions; episodes recur across splits. This is not episode-disjoint evaluation.",
        "reuse_files_found": sorted(set(reuse_files)), "reuse_conditions_verified": False,
        "ready_for_export": False,
    }
    blockers = []
    if overlapping_groups:
        blockers.append("Identical conversation segments cross partitions")
    if any(count == 0 for per_class in coverage.values() for count in per_class.values()):
        blockers.append("Duplicate quarantine leaves an emotion missing in a split")
    report["data_integrity_ready"] = not blockers
    report["export_blockers"] = blockers
    return safe_rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--report", default="data/audits/masac.json")
    parser.add_argument("--output", help="CSV output; requires reviewed reuse evidence")
    parser.add_argument("--reuse-evidence", help="Local document recording applicable publisher permission or terms")
    parser.add_argument("--reuse-approved", action="store_true", help="Explicitly attest that the supplied terms were reviewed and permit this use")
    args = parser.parse_args()
    rows, report = audit_source(args.source_dir)
    if args.reuse_approved:
        evidence = Path(args.reuse_evidence or "")
        if not args.reuse_evidence or not evidence.is_file() or not evidence.read_bytes().strip():
            raise ValueError("Reuse approval requires a non-empty evidence document")
        report.update(reuse_conditions_verified=True, reuse_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(), ready_for_export=report["data_integrity_ready"])
    if args.output and report["ready_for_export"]:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["prepared_csv_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output and not report["ready_for_export"]:
        raise SystemExit("Audit saved. CSV export blocked until publisher reuse conditions are established.")


if __name__ == "__main__":
    main()
