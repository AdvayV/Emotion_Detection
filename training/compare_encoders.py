"""Sequential matched-model experiments with one frozen data split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from hinglish_emotion.data_validation import validate_csv
from hinglish_emotion.emotion_data import read_emotion_csv
from hinglish_emotion.model_catalog import ENCODER_BY_KEY, checkpoint_path
from hinglish_emotion.tasks import EMOTION_LABELS
from training.data_workflow import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=("sentiment", "emotion"), default="sentiment")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--models", nargs="+", choices=tuple(ENCODER_BY_KEY))
    parser.add_argument("--language-pair", choices=("hi-en", "fr-en", "all"))
    parser.add_argument("--data-audit")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--text-mode", choices=("raw", "normalized"))
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume existing per-model epoch states")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without creating files or downloading models")
    return parser.parse_args()


def training_command(args, key):
    spec = ENCODER_BY_KEY[key]
    command = [
        sys.executable, "-m", "training.train_transformer", "--task", args.task,
        "--data", args.data, "--model", spec.model_id,
        "--output", str(checkpoint_path(spec, args.output_root, args.task)),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length), "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed), "--split-seed", str(args.split_seed),
        "--language-pair", args.language_pair, "--text-mode", args.text_mode,
        "--split-manifest", args.split_manifest, "--gradient-accumulation", str(args.gradient_accumulation),
        "--precision", args.precision,
    ]
    if args.data_audit:
        command.extend(["--data-audit", args.data_audit])
    for flag in ("freeze_encoder", "allow_cpu", "local_files_only", "allow_synthetic"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    output = checkpoint_path(spec, args.output_root, args.task)
    if args.resume and (output / "latest_training_state.pt").is_file():
        command.append("--resume")
    return command


def main():
    args = parse_args()
    args.models = args.models or (["xlmr", "muril"] if args.task == "emotion" else list(ENCODER_BY_KEY))
    args.language_pair = args.language_pair or ("hi-en" if args.task == "emotion" else "all")
    args.text_mode = args.text_mode or ("raw" if args.task == "emotion" else "normalized")
    args.split_manifest = args.split_manifest or str(Path(args.output_root) / f"{args.task}_frozen_split.csv")
    if len(set(args.models)) != len(args.models) or min(args.epochs, args.batch_size, args.max_length, args.gradient_accumulation) < 1:
        raise ValueError("Choose distinct models and positive training settings")
    if args.task == "emotion":
        rows = read_emotion_csv(args.data)
        rows = [row for row in rows if args.language_pair == "all" or row["language_pair"] == args.language_pair]
        if min(sum(row["emotion"] == label for row in rows) for label in EMOTION_LABELS) < 6:
            raise ValueError("Need at least six examples per emotion in the selected language scope")
        if "muril" in args.models and any(row["language_pair"] == "fr-en" for row in rows):
            raise ValueError("The agreed MuRIL experiment is Hindi–English only. Use --language-pair hi-en or choose XLM-R for multilingual training.")
        if any(row.get("source") in {"illustrative", "assistant_draft"} for row in rows):
            print("Illustrative data: these commands are a preview; real training will reject this file.", file=sys.stderr)
    else:
        report = validate_csv(args.data)
        if not report.valid or min(report.label_counts.values()) < 6:
            raise ValueError("Provide a valid dataset with at least six examples per sentiment")
    if args.dry_run:
        for key in args.models:
            print(f"{ENCODER_BY_KEY[key].name}: {subprocess.list2cmdline(training_command(args, key))}")
        return
    prepared = prepare_data(args.data, args.task, args.text_mode, args.split_seed, args.split_manifest, args.data_audit, args.language_pair, args.allow_synthetic)
    expected_hash = prepared["manifest_sha256"]
    results = []
    for key in args.models:
        command = training_command(args, key)
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
        output = checkpoint_path(ENCODER_BY_KEY[key], args.output_root, args.task)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if metrics["split_sha256"] != expected_hash:
            raise RuntimeError("Model used a different frozen split; comparison stopped")
        result = {
            "encoder": ENCODER_BY_KEY[key].name, "checkpoint": str(output),
            "task": args.task, "text_mode": args.text_mode, "training_seed": args.seed,
            "split_sha256": expected_hash, "best_validation_macro_f1": metrics["best_validation_macro_f1"],
            "test_size": metrics["split_sizes"]["test"],
            "training_data_kind": metrics["training_data_kind"], "evaluation_scope": metrics["evaluation_scope"],
            "accuracy": metrics["test"]["accuracy"], "macro_f1": metrics["test"]["macro_f1"],
        }
        for pair, scores in metrics["test_by_language"].items():
            result[f"{pair}_accuracy"] = scores["accuracy"]
            result[f"{pair}_macro_f1"] = scores["macro_f1"]
        results.append(result)
    destination = Path(args.output_root) / ("emotion_encoder_comparison.csv" if args.task == "emotion" else "encoder_comparison.csv")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Comparison saved to {destination}")


if __name__ == "__main__":
    main()
