"""Fit an experimental temperature profile from held-out emotion labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hinglish_emotion.calibration import fit_temperature, frozen_phrase_hashes
from hinglish_emotion.emotions import LocalEmotionClassifier
from hinglish_emotion.evaluation_guard import phrase_hash
from hinglish_emotion.checkpoints import checkpoint_version


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CSV containing text, emotion/label, and optional split columns")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Write profile JSON here")
    parser.add_argument("--split", choices=("validation", "calibration"), required=True,
                        help="Declare validation rows, or that this file is a dedicated calibration set")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly allow synthetic labels; profile remains experimental")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Explicitly allow unknown or unreviewed labels; profile remains experimental")
    parser.add_argument("--scope", default="unspecified")
    args = parser.parse_args(argv)
    records = _load_csv(args.data)
    if not records:
        parser.error("Calibration CSV is empty")
    if args.split == "validation" and "split" not in records[0]:
        parser.error("Validation mode requires a split column declaring validation rows")
    for row in records:
        declared = row.get("split", "").strip().lower()
        if declared in {"train", "test"}:
            parser.error(f"Row declares prohibited {declared} split")
        if declared and declared != args.split:
            parser.error(f"Row declares split={declared}; calibration accepts only {args.split} rows")
    if args.split == "calibration" and any(row.get("split", "").strip().lower() == "test" for row in records):
        parser.error("Test rows are prohibited")
    root = Path(__file__).resolve().parents[1]
    frozen = frozen_phrase_hashes(root)
    overlaps = [row.get("text", "") for row in records if row.get("text") and phrase_hash(row["text"]) in frozen]
    if overlaps:
        parser.error(f"Calibration data contains {len(overlaps)} frozen test/evaluation phrases")
    if any(not row.get("text") or not (row.get("emotion") or row.get("label")) for row in records):
        parser.error("Every row requires text and an emotion/label")
    synthetic = any((row.get("source", "") + row.get("annotation_status", "")).lower().find("synthetic") >= 0 for row in records)
    if synthetic and not args.allow_synthetic:
        parser.error("Synthetic data requires --allow-synthetic")
    human_reviewed = bool(records) and all(row.get("human_reviewed", "").strip().lower() in {"true", "yes", "1"} for row in records)
    if not human_reviewed and not args.allow_unreviewed:
        parser.error("Unknown or unreviewed labels require --allow-unreviewed")
    classifier = LocalEmotionClassifier(args.checkpoint, device="cpu")
    probabilities = [classifier.predict_text(row["text"]).probabilities for row in records]
    metadata = classifier.metadata
    identity = {
        "model_path": str(args.checkpoint.resolve()), "labels": list(classifier.model.config.id2label.values()),
        "checkpoint_version": checkpoint_version(args.checkpoint),
        "text_mode": metadata.get("text_mode"), "preparation_version": metadata.get("preparation_version"),
        "max_length": classifier.max_length, "scope": args.scope,
    }
    profile = fit_temperature(probabilities, [r.get("emotion") or r.get("label") for r in records],
                              model_identity=identity, split=args.split, scope=args.scope,
                              synthetic=synthetic, allow_synthetic=args.allow_synthetic,
                              human_reviewed=human_reviewed, allow_unreviewed=args.allow_unreviewed)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "temperature": profile["temperature"],
                      "metrics_before": profile["metrics_before"], "metrics_after": profile["metrics_after"],
                      "status": profile["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
