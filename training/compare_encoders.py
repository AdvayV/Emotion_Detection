"""Train mBERT, MuRIL and XLM-R sequentially on the same data split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hinglish_emotion.data_validation import validate_csv
from hinglish_emotion.model_catalog import ENCODER_BY_KEY, checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multilingual Hinglish sentiment encoders")
    parser.add_argument("--data", required=True, help="Labelled CSV with text and label columns")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--models", nargs="+", choices=tuple(ENCODER_BY_KEY), default=list(ENCODER_BY_KEY))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true", help="Lower-memory head-only training")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading or training")
    return parser.parse_args()


def training_command(args: argparse.Namespace, key: str) -> list[str]:
    spec = ENCODER_BY_KEY[key]
    command = [
        sys.executable, "-m", "training.train_transformer",
        "--data", args.data,
        "--model", spec.model_id,
        "--output", str(checkpoint_path(spec, args.output_root)),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]
    if args.freeze_encoder:
        command.append("--freeze-encoder")
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def main() -> None:
    args = parse_args()
    if len(args.models) != len(set(args.models)):
        raise ValueError("Each encoder should be selected only once")
    if min(args.epochs, args.batch_size, args.max_length) < 1:
        raise ValueError("Epochs, batch size and max length must be positive")
    report = validate_csv(args.data)
    if not report.valid or min(report.label_counts.values()) < 6:
        raise ValueError("Provide a valid labelled CSV with at least six examples per sentiment class")
    if report.rows < 100:
        print("Warning: this small dataset is suitable for a smoke test, not research claims.", file=sys.stderr)

    expected_manifest_hash: str | None = None
    rows: list[dict[str, object]] = []
    for key in args.models:
        spec = ENCODER_BY_KEY[key]
        output = checkpoint_path(spec, args.output_root)
        command = training_command(args, key)
        print(f"{spec.name}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        manifest_hash = hashlib.sha256((output / "split_manifest.csv").read_bytes()).hexdigest()
        if expected_manifest_hash is None:
            expected_manifest_hash = manifest_hash
        elif manifest_hash != expected_manifest_hash:
            raise RuntimeError("Encoder runs used different data splits; comparison stopped")
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        test = metrics["test"]
        rows.append({
            "encoder": spec.name,
            "base_model": spec.model_id,
            "checkpoint": str(output),
            "training_mode": metrics["training_mode"],
            "seed": args.seed,
            "test_size": metrics["split_sizes"]["test"],
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "negative_recall": test["per_class"]["negative"]["recall"],
            "split_sha256": manifest_hash,
        })

    if rows:
        destination = Path(args.output_root) / "encoder_comparison.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Comparison saved to {destination}")


if __name__ == "__main__":
    main()
