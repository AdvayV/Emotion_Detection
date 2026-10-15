from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hinglish_emotion.evaluation import classification_metrics
from hinglish_emotion.emotions import EmotionDetector
from hinglish_emotion.text_preparation import PREPARATION_VERSION
from training.data_workflow import prepare_data


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, label2id, max_length):
        self.texts = frame["normalized_text"].tolist()
        self.labels = frame["label"].map(label2id).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], truncation=True, max_length=self.max_length)
        return {**encoded, "labels": self.labels[index]}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and resume a local sentiment or seven-emotion classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=("sentiment", "emotion"), default="sentiment")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=3, help="Total target epochs, including completed epochs on resume")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="Training seed; does not change the frozen split")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-manifest", help="Shared manifest reused across models and training seeds")
    parser.add_argument("--data-audit", help="Approved source audit matching a MaSaC CSV")
    parser.add_argument("--language-pair", choices=("all", "hi-en", "fr-en"), default="all")
    parser.add_argument("--text-mode", choices=("raw", "normalized"))
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly permit slow full emotion fine-tuning on CPU")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume an epoch-boundary state; defaults to output/latest_training_state.pt")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly permit synthetic experimental training; no human validation is claimed")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def class_weight_tensor(labels, device, num_labels=3):
    counts = torch.bincount(torch.tensor(labels), minlength=num_labels).float()
    return (counts.sum() / (num_labels * counts.clamp_min(1))).to(device)


def random_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
        "numpy": {"name": numpy_state[0], "values": numpy_state[1].tolist(), "position": numpy_state[2], "has_gauss": numpy_state[3], "cached": numpy_state[4]},
    }


def restore_random_state(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.array(numpy_state["values"], dtype=np.uint32), numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached"]))


def predict(model, loader, device, precision="fp32"):
    model.eval()
    actual, predicted, confidences = [], [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            actual.extend(targets.cpu().tolist())
            predicted.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return actual, predicted, confidences


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def write_model_card(output, result):
    scope = ", ".join(result["emotion_metadata"]["validated_language_pairs"]) if result["task"] == "emotion" else "sentiment dataset"
    text = f"""# Fine-tuned {result['task']} checkpoint

Training data kind: {result['training_data_kind']}

Evaluation scope: {result['evaluation_scope']}

Synthetic-only results measure generated examples, not real-world human-labelled accuracy.

Base model: {result['base_model']}

Labels: {', '.join(result['labels'])}

Text preparation: {result['text_mode']} (version {PREPARATION_VERSION}); token limit {result['max_length']}.

Dataset SHA-256: {result['dataset_sha256']}

Split SHA-256: {result['split_sha256']}

Best validation macro-F1: {result['best_validation_macro_f1']:.4f}

Test accuracy: {result['test']['accuracy']:.4f}

Test macro-F1: {result['test']['macro_f1']:.4f}

Verified labelled evaluation scope: {scope or 'not independently verified'}.

These scores apply only to the recorded dataset and split. They do not establish accuracy for other languages, code mixing, sarcasm, or missing context. Classifier scores are not calibrated correctness probabilities. See metrics.json for per-class support, language scores and baseline comparison.

The model selects one dominant expressed emotion. It does not diagnose a person's psychological state. French–English remains experimental without reviewed test coverage.

Publisher model cards:
- XLM-R: https://huggingface.co/FacebookAI/xlm-roberta-base
- MuRIL: https://huggingface.co/google/muril-base-cased
"""
    (output / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def train(args):
    if min(args.epochs, args.patience, args.batch_size, args.max_length, args.gradient_accumulation) < 1 or args.learning_rate <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.model = args.model or ("FacebookAI/xlm-roberta-base" if args.task == "emotion" else "google/muril-base-cased")
    args.output = args.output or ("models/xlmr_emotion" if args.task == "emotion" else "models/muril_sentiment")
    text_mode = args.text_mode or ("raw" if args.task == "emotion" else "normalized")
    output = Path(args.output)
    manifest_path = args.split_manifest or str(output / "frozen_split.csv")
    prepared = prepare_data(args.data, args.task, text_mode, args.split_seed, manifest_path, args.data_audit, args.language_pair, args.allow_synthetic)
    device = args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    if device == "cpu" and args.task == "emotion" and not (args.allow_cpu or args.freeze_encoder):
        raise RuntimeError("No CUDA GPU found. Select a cloud GPU, or explicitly use --freeze-encoder / --allow-cpu.")
    precision = args.precision if args.precision != "auto" else "fp16" if device == "cuda" else "fp32"
    if device == "cpu" and precision != "fp32":
        raise ValueError("This trainer requires fp32 for CPU runs")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The assigned GPU does not support bf16")
    labels = list(prepared["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    train_frame, validation_frame, test_frame = prepared["parts"]
    identity = {
        "base_model": args.model, "task": args.task, "labels": labels, "text_mode": text_mode,
        "preparation_version": PREPARATION_VERSION, "max_length": args.max_length,
        "dataset_sha256": prepared["dataset_sha256"], "split_sha256": prepared["manifest_sha256"],
        "seed": args.seed, "split_seed": args.split_seed, "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate,
        "class_weights": args.class_weights, "freeze_encoder": args.freeze_encoder, "precision": precision,
        "patience": args.patience,
        "allow_synthetic": args.allow_synthetic,
        "training_data_kind": prepared["training_data_kind"], "evaluation_scope": prepared["evaluation_scope"],
    }
    resume_path = output / "latest_training_state.pt" if args.resume == "latest" else Path(args.resume) if args.resume else None
    state = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    if state and state["identity"] != identity:
        raise ValueError("Resume refused: dataset, split or training settings changed")
    if state and args.epochs < state["completed_epoch"]:
        raise ValueError("Target epochs cannot be smaller than the completed epoch count")
    if not state and (output / "metrics.json").exists():
        raise ValueError("Output already has a completed run. Resume it or choose a new output folder.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), label2id=label2id, id2label=id2label,
        local_files_only=args.local_files_only,
    ).to(device)
    model.config.classification_task = args.task
    metadata = {
        "schema_version": 1, "task": args.task, "labels": labels, "text_mode": text_mode,
        "preparation_version": PREPARATION_VERSION, "max_length": args.max_length,
        "base_model": args.model, "trained_language_pairs": prepared["trained_language_pairs"],
        "training_data_kind": prepared["training_data_kind"], "evaluation_scope": prepared["evaluation_scope"],
        "validated_language_pairs": [], "training_status": "training",
        "dataset_sha256": prepared["dataset_sha256"], "split_sha256": prepared["manifest_sha256"],
    }
    if args.task == "emotion":
        model.config.emotion_metadata = metadata
    if args.freeze_encoder:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    generator = torch.Generator().manual_seed(args.seed)
    def collate(items):
        targets = torch.tensor([item["labels"] for item in items], dtype=torch.long)
        inputs = [{key: value for key, value in item.items() if key != "labels"} for item in items]
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = targets
        return batch
    def loader(frame, shuffle):
        return DataLoader(TextDataset(frame, tokenizer, label2id, args.max_length), batch_size=args.batch_size, shuffle=shuffle, generator=generator if shuffle else None, collate_fn=collate)
    train_loader = loader(train_frame, True)
    validation_loader = loader(validation_frame, False)
    test_loader = loader(test_frame, False)
    weights = class_weight_tensor(train_frame["label"].map(label2id).tolist(), device, len(labels)) if args.class_weights == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_f1, stale, history, completed = -1.0, 0, [], 0
    if state:
        if not (output / "config.json").exists():
            raise ValueError("Resume requires the saved best checkpoint in the original output folder")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_f1, stale, history, completed = state["best_f1"], state["stale"], state["history"], state["completed_epoch"]
        restore_random_state(state["random"], generator)
        del state
    output.mkdir(parents=True, exist_ok=True)
    prepared["manifest"].to_csv(output / "split_manifest.csv", index=False, lineterminator="\n")
    for epoch in range(completed + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        if args.freeze_encoder:
            model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index, batch in enumerate(train_loader):
            targets = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            window_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(train_loader) - window_start)
            with autocast_context(device, precision):
                loss = criterion(model(**inputs).logits.float(), targets)
            scaler.scale(loss / window_size).backward()
            total_loss += float(loss.detach().item())
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        actual_ids, predicted_ids, _ = predict(model, validation_loader, device, precision)
        scores = classification_metrics([id2label[index] for index in actual_ids], [id2label[index] for index in predicted_ids], labels)
        validation_f1 = scores["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), "validation_macro_f1": validation_f1})
        print(json.dumps(history[-1]), flush=True)
        if validation_f1 > best_f1:
            best_f1, stale = validation_f1, 0
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            atomic_json(output / "emotion_metadata.json", metadata)
        else:
            stale += 1
        checkpoint_state = {
            "identity": identity, "completed_epoch": epoch, "best_f1": best_f1, "stale": stale,
            "history": history, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "random": random_state(generator),
        }
        temporary = output / "latest_training_state.pt.tmp"
        torch.save(checkpoint_state, temporary)
        temporary.replace(output / "latest_training_state.pt")
        del checkpoint_state
    del optimizer, trainable, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True).to(device)
    # Deployment uses fp32, so final test scores use the same inference arithmetic.
    actual_ids, predicted_ids, confidences = predict(best_model, test_loader, device)
    actual = [id2label[index] for index in actual_ids]
    predicted = [id2label[index] for index in predicted_ids]
    metrics = classification_metrics(actual, predicted, labels)
    predictions = test_frame.drop(columns=["normalized_text", "duplicate_key"], errors="ignore").copy()
    predictions["predicted"], predictions["confidence"] = predicted, confidences
    predictions.to_csv(output / "test_predictions.csv", index=False)
    by_language = {}
    if "language_pair" in test_frame:
        for pair in sorted(test_frame["language_pair"].unique()):
            indices = test_frame.index[test_frame["language_pair"] == pair].tolist()
            by_language[pair] = {**classification_metrics([actual[index] for index in indices], [predicted[index] for index in indices], labels), "rows": len(indices)}
    by_source = {}
    if "source" in test_frame:
        for source in sorted(test_frame["source"].unique()):
            indices = test_frame.index[test_frame["source"] == source].tolist()
            by_source[source] = {**classification_metrics([actual[index] for index in indices], [predicted[index] for index in indices], labels), "rows": len(indices)}
    baseline = {}
    if args.task == "emotion":
        detector = EmotionDetector()
        for name, frame in (("validation", validation_frame), ("test", test_frame)):
            guesses = [detector.predict(row["text"], row["language_pair"]).emotion for _, row in frame.iterrows()]
            baseline[name] = classification_metrics(frame["label"].tolist(), guesses, labels)
        metadata.update(validated_language_pairs=prepared["validated_language_pairs"], training_status="complete")
        best_model.config.emotion_metadata = metadata
        best_model.config.save_pretrained(output)
        atomic_json(output / "emotion_metadata.json", metadata)
    result = {
        **identity, "device": device, "training_mode": "frozen_encoder" if args.freeze_encoder else "full_fine_tuning",
        "split_sizes": dict(zip(("train", "validation", "test"), map(len, prepared["parts"]))),
        "history": history, "best_validation_macro_f1": best_f1,
        "test": metrics, "test_by_language": by_language, "test_by_source": by_source, "test_confidences": confidences,
        "dataset_report": prepared["report"], "lexical_baseline": baseline, "emotion_metadata": metadata,
    }
    atomic_json(output / "metrics.json", result)
    write_model_card(output, result)
    print(json.dumps({"output": str(output), "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}), flush=True)
    return result


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
