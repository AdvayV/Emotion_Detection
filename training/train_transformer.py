from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hinglish_emotion.data_validation import validate_csv
from hinglish_emotion.evaluation import classification_metrics
from hinglish_emotion.normalization import HinglishNormalizer


LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


class SentimentDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local Hinglish sentiment encoder")
    parser.add_argument("--data", required=True, help="CSV containing text and label columns")
    parser.add_argument("--model", default="google/muril-base-cased")
    parser.add_argument("--output", default="models/muril_sentiment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="balanced")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and load an already downloaded base model",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=seed,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def class_weight_tensor(labels: Sequence[int], device: str) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=len(LABEL2ID)).float()
    weights = counts.sum() / (len(LABEL2ID) * counts.clamp_min(1))
    return weights.to(device)


def predict(model, loader: DataLoader, device: str) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    true: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**batch).logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            true.extend(labels.cpu().tolist())
            predicted.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return true, predicted, confidences


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    report = validate_csv(args.data)
    if not report.valid:
        raise ValueError(f"Dataset validation failed: {report.to_dict()}")
    if min(report.label_counts.values()) < 4:
        raise ValueError("Each sentiment class needs at least four examples for a stratified 70/15/15 split")

    frame = pd.read_csv(args.data)
    frame["label"] = frame["label"].str.strip().str.lower()
    normalizer = HinglishNormalizer()
    frame["normalized_text"] = frame["text"].astype(str).map(
        lambda value: normalizer.normalize(value).normalized_text
    )
    train_frame, validation_frame, test_frame = split_data(frame, args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for split_name, split_frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        for index, row in split_frame.iterrows():
            split_rows.append({"split": split_name, "text": row["text"], "label": row["label"]})
    pd.DataFrame(split_rows).to_csv(output / "split_manifest.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=args.local_files_only,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def make_loader(split_frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        labels = split_frame["label"].map(LABEL2ID).tolist()
        dataset = SentimentDataset(
            split_frame["normalized_text"].tolist(), labels, tokenizer, args.max_length
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_frame, True)
    validation_loader = make_loader(validation_frame, False)
    test_loader = make_loader(test_frame, False)
    train_labels = train_frame["label"].map(LABEL2ID).tolist()
    weights = class_weight_tensor(train_labels, device) if args.class_weights == "balanced" else None
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        validation_true, validation_predicted, _ = predict(model, validation_loader, device)
        validation_f1 = f1_score(validation_true, validation_predicted, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "validation_macro_f1": float(validation_f1),
            }
        )
        if validation_f1 > best_f1:
            best_f1 = float(validation_f1)
            epochs_without_improvement = 0
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True).to(device)
    test_true_ids, test_predicted_ids, test_confidences = predict(best_model, test_loader, device)
    test_true = [ID2LABEL[value] for value in test_true_ids]
    test_predicted = [ID2LABEL[value] for value in test_predicted_ids]
    metrics = classification_metrics(test_true, test_predicted)
    result = {
        "base_model": args.model,
        "device": device,
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "history": history,
        "best_validation_macro_f1": best_f1,
        "test": metrics,
        "test_confidences": test_confidences,
        "dataset_report": report.to_dict(),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

