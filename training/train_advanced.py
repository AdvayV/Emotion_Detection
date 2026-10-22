"""GPU-only learned text/emoji seven-emotion training, with optional reviewed tasks.

--dry-run validates source/audit, frozen group splits and supervision before any
Torch/CUDA import or pretrained download. Recovery uses SafeTensors and JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.data_workflow import prepare_data
from hinglish_emotion.text_preparation import PREPARATION_VERSION

BASE_MODEL = "FacebookAI/xlm-roberta-base"
ARCHITECTURE = "text_emoji_multitask_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/xlmr_fusion_emotion")
    parser.add_argument("--split-manifest")
    parser.add_argument("--data-audit")
    parser.add_argument("--language-pair", choices=("all", "hi-en", "fr-en"), default="all")
    parser.add_argument("--text-mode", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--context-column", default="context")
    parser.add_argument("--sarcasm-column", help="Optional independent 0/1 or not_sarcastic/sarcastic labels")
    parser.add_argument("--sarcasm-weight", type=float, default=0.2)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", nargs="?", const="latest", help="Completed recovery epoch directory, or latest")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def reviewed(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def supervision_frame(frame, use_context=False, context_column="context", sarcasm_column=None):
    """Do not derive annotations from the emotion label or emoji cues."""
    frame = frame.copy()
    frame["context_text"] = ""
    if use_context:
        if context_column not in frame:
            raise ValueError(f"Context training requires column {context_column!r}")
        values = frame[context_column].fillna("").astype(str).str.strip()
        for index in frame.index[values.ne("")]:
            if not reviewed(frame.loc[index].get("human_reviewed", "")):
                raise ValueError("Context training requires human_reviewed=true on contextual emotion rows")
            if not any(str(frame.loc[index].get(key, "")).strip() for key in ("group_id", "conversation_id")):
                raise ValueError("Context rows require group_id or conversation_id to prevent conversation leakage")
        frame["context_text"] = values
    frame["sarcasm_target"] = -100
    if sarcasm_column:
        if sarcasm_column not in frame:
            raise ValueError(f"Sarcasm training requires column {sarcasm_column!r}")
        mapping = {"0": 0, "1": 1, "not_sarcastic": 0, "sarcastic": 1}
        for index, value in frame[sarcasm_column].fillna("").astype(str).items():
            value = value.strip().lower()
            if not value:
                continue
            if value not in mapping:
                raise ValueError("Sarcasm labels must be 0/1 or not_sarcastic/sarcastic; leave unknown rows blank")
            if not reviewed(frame.loc[index].get("human_reviewed", "")):
                raise ValueError("Independent sarcasm labels require human_reviewed=true")
            frame.loc[index, "sarcasm_target"] = mapping[value]
    return frame


def prepare_inputs(args):
    if min(args.epochs, args.patience, args.batch_size, args.gradient_accumulation, args.max_length) < 1:
        raise ValueError("Training counts must be positive")
    if args.learning_rate <= 0 or args.sarcasm_weight < 0 or not 5 <= args.max_length <= 512:
        raise ValueError("Invalid learning rate, sarcasm weight or XLM-R token limit")
    if args.sarcasm_column and args.sarcasm_weight <= 0:
        raise ValueError("Supervised sarcasm training requires a positive auxiliary weight")
    manifest = args.split_manifest or str(Path(args.output) / "frozen_split.csv")
    prepared = prepare_data(args.data, "emotion", args.text_mode, args.split_seed, manifest,
                            args.data_audit, args.language_pair, args.allow_synthetic)
    prepared["parts"] = tuple(supervision_frame(part, args.use_context, args.context_column, args.sarcasm_column)
                              for part in prepared["parts"])
    if args.use_context and any(not part["context_text"].ne("").any() for part in prepared["parts"]):
        raise ValueError("Context training requires reviewed context examples in train, validation and test")
    if args.sarcasm_column:
        for name, part in zip(("train", "validation"), prepared["parts"][:2]):
            if set(part.loc[part["sarcasm_target"].ne(-100), "sarcasm_target"]) != {0, 1}:
                raise ValueError(f"Reviewed sarcasm training requires both binary classes in {name}")
    prepared["supervision_counts"] = {
        name: {"rows": len(part), "context_rows": int(part["context_text"].ne("").sum()),
               "sarcasm_rows": int(part["sarcasm_target"].ne(-100).sum())}
        for name, part in zip(("train", "validation", "test"), prepared["parts"])}
    return prepared


def save_tensor_tree(directory, payload):
    """Serialize optimizer/scaler/RNG tensors without executable pickle data."""
    import torch
    from safetensors.torch import save_file
    tensors = {}
    def encode(value):
        if isinstance(value, torch.Tensor):
            key = f"tensor_{len(tensors):06d}"
            tensors[key] = value.detach().cpu().contiguous()
            return {"tensor": key}
        if isinstance(value, dict):
            return {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
        if isinstance(value, tuple):
            return {"tuple": [encode(item) for item in value]}
        if isinstance(value, list):
            return {"list": [encode(item) for item in value]}
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise TypeError(f"Unsupported recovery state type: {type(value).__name__}")
    tree = encode(payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "training_state.safetensors"))
    (directory / "training_state.json").write_text(json.dumps(tree), encoding="utf-8")


def load_tensor_tree(directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    tensors = load_file(str(directory / "training_state.safetensors"))
    def decode(value):
        if not isinstance(value, dict):
            return value
        if set(value) == {"tensor"}:
            return tensors[value["tensor"]]
        if set(value) == {"dict"}:
            return {decode(key): decode(item) for key, item in value["dict"]}
        if set(value) == {"tuple"}:
            return tuple(decode(item) for item in value["tuple"])
        if set(value) == {"list"}:
            return [decode(item) for item in value["list"]]
        raise ValueError("Invalid recovery JSON structure")
    return decode(json.loads((directory / "training_state.json").read_text(encoding="utf-8")))


def train(args):
    prepared = prepare_inputs(args)
    report = {"architecture": ARCHITECTURE, "base_model": BASE_MODEL,
              "dataset_sha256": prepared["dataset_sha256"], "split_sha256": prepared["manifest_sha256"],
              "training_data_kind": prepared["training_data_kind"], "evaluation_scope": prepared["evaluation_scope"],
              "supervision_counts": prepared["supervision_counts"], "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return report
    import gc
    import shutil
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer
    from hinglish_emotion.fusion import TextEmojiFusionModel, emoji_batch, multitask_loss, encode_fusion_inputs, CONTEXT_INPUT_POLICY
    from hinglish_emotion.evaluation import classification_metrics
    from training.train_transformer import (set_seed, atomic_json, class_weight_tensor,
                                             autocast_context, random_state, restore_random_state)
    if not torch.cuda.is_available():
        raise RuntimeError("Advanced fine-tuning requires a CUDA GPU. Use the Colab GPU notebook; no CPU training is run.")
    precision = "fp16" if args.precision == "auto" else args.precision
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The assigned GPU does not support bf16")
    set_seed(args.seed)
    device = "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(prepared["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    identity = {**report, "dry_run": False, "text_mode": args.text_mode,
                "preparation_version": PREPARATION_VERSION, "max_length": args.max_length,
                "seed": args.seed, "split_seed": args.split_seed, "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate,
                "class_weights": args.class_weights, "precision": precision, "patience": args.patience,
                "context_input_policy": CONTEXT_INPUT_POLICY,
                "use_context": args.use_context, "context_column": args.context_column,
                "sarcasm_column": args.sarcasm_column, "sarcasm_weight": args.sarcasm_weight,
                "allow_synthetic": args.allow_synthetic, "labels": labels}
    if (output / "metrics.json").exists():
        existing = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if args.resume and existing.get("identity") == identity:
            print("Completed run already evaluated; returning its saved results without inspecting test again.")
            return existing
        raise ValueError("This run has already evaluated test. Use a new output for a preregistered experiment.")
    if (output / "test_evaluation_started.json").exists():
        raise ValueError("Test evaluation already started for this run; recovery must not repeatedly inspect test.")
    resume_dir = None
    if args.resume:
        if args.resume == "latest":
            pointer = json.loads((output / "latest_recovery.json").read_text(encoding="utf-8"))
            resume_dir = (output / "recovery" / pointer["directory"]).resolve()
            if not resume_dir.is_relative_to((output / "recovery").resolve()):
                raise ValueError("Unsafe recovery directory")
        else:
            resume_dir = Path(args.resume)
    elif (output / "latest_recovery.json").exists() or (output / "model.safetensors").exists():
        raise ValueError("An incomplete run exists. Resume it or choose a new output.")
    state = load_tensor_tree(resume_dir) if resume_dir else None
    if state and (state["identity"] != identity or args.epochs < state["completed_epoch"]):
        raise ValueError("Recovery refused: data, frozen split, supervision or training settings changed")
    tokenizer = AutoTokenizer.from_pretrained(resume_dir or BASE_MODEL, local_files_only=bool(resume_dir) or args.local_files_only,
                                              trust_remote_code=False)
    model = (TextEmojiFusionModel.from_pretrained(resume_dir) if resume_dir else
             TextEmojiFusionModel.from_base_pretrained(local_files_only=args.local_files_only)).to(device)
    metadata = {"schema_version": 1, "task": "emotion", "architecture": ARCHITECTURE,
                "base_model": BASE_MODEL, "labels": labels, "text_mode": args.text_mode,
                "preparation_version": PREPARATION_VERSION, "max_length": args.max_length,
                "trained_language_pairs": prepared["trained_language_pairs"], "validated_language_pairs": [],
                "dataset_sha256": prepared["dataset_sha256"], "split_sha256": prepared["manifest_sha256"],
                "training_data_kind": prepared["training_data_kind"], "evaluation_scope": prepared["evaluation_scope"],
                "training_status": "training", "trained_sarcasm_head": bool(args.sarcasm_column),
                "trained_with_context": bool(args.use_context), "context_encoding": "current_text_first_tokenizer_pair_v1",
                "context_input_policy": CONTEXT_INPUT_POLICY,
                "emoji_encoding": "sha256_sequence_v1", "supervision_counts": prepared["supervision_counts"],
                "sarcasm_supervision": {"human_reviewed": bool(args.sarcasm_column), "label_column": args.sarcasm_column,
                                        "labels": ["not_sarcastic", "sarcastic"],
                                        "language_pairs": sorted(prepared["parts"][0].loc[prepared["parts"][0]["sarcasm_target"].ne(-100), "language_pair"].unique()) if args.sarcasm_column else [],
                                        "validated_language_pairs": []}}
    model.config.emotion_metadata = metadata
    model.config.classification_task = "emotion"
    generator = torch.Generator().manual_seed(args.seed)
    class Rows(Dataset):
        def __init__(self, frame):
            self.rows = frame.to_dict("records")
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            return self.rows[index]
    def collate(rows):
        texts = [row["normalized_text"] for row in rows]
        from hinglish_emotion.text_preparation import prepare_text
        contexts = [prepare_text(row["context_text"], args.text_mode) for row in rows] if args.use_context else None
        inputs, _ = encode_fusion_inputs(tokenizer, texts, contexts, args.max_length)
        inputs["emoji_ids"] = emoji_batch([row["text"] for row in rows])
        inputs["emotion_targets"] = torch.tensor([label2id[row["label"]] for row in rows])
        inputs["sarcasm_targets"] = torch.tensor([row["sarcasm_target"] for row in rows])
        return inputs
    loaders = [DataLoader(Rows(frame), batch_size=args.batch_size, shuffle=index == 0,
                          generator=generator if index == 0 else None, collate_fn=collate)
               for index, frame in enumerate(prepared["parts"])]
    weights = class_weight_tensor(prepared["parts"][0]["label"].map(label2id).tolist(), device, 7) if args.class_weights == "balanced" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    best_f1, stale, history, completed = -1.0, 0, [], 0
    if state:
        if not (output / "model.safetensors").is_file():
            raise ValueError("Recovery requires the saved best model in the original output directory")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_f1, stale, history, completed = state["best_f1"], state["stale"], state["history"], state["completed_epoch"]
        restore_random_state(state["random"], generator)
        del state
    prepared["manifest"].to_csv(output / "split_manifest.csv", index=False, lineterminator="\n")
    def evaluate(loader, arithmetic="fp32"):
        model.eval()
        actual, predicted, confidence = [], [], []
        sarcasm_actual, sarcasm_predicted = [], []
        with torch.inference_mode():
            for batch in loader:
                targets = batch.pop("emotion_targets")
                aux = batch.pop("sarcasm_targets")
                with autocast_context(device, arithmetic):
                    values = model(**{key: value.to(device) for key, value in batch.items()})
                scores, guesses = values.logits.float().softmax(-1).max(-1)
                actual.extend(targets.tolist())
                predicted.extend(guesses.cpu().tolist())
                confidence.extend(scores.cpu().tolist())
                mask = aux.ne(-100)
                if args.sarcasm_column and mask.any():
                    sarcasm_actual.extend(aux[mask].tolist())
                    sarcasm_predicted.extend(values.sarcasm_logits.detach().cpu()[mask].ge(0).int().tolist())
        scores = classification_metrics([labels[index] for index in actual], [labels[index] for index in predicted], labels)
        auxiliary = classification_metrics(sarcasm_actual, sarcasm_predicted, [0, 1]) if sarcasm_actual else None
        return scores, predicted, confidence, auxiliary
    for epoch in range(completed + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(loaders[0]):
            targets = batch.pop("emotion_targets").to(device)
            auxiliary = batch.pop("sarcasm_targets").to(device)
            start = index // args.gradient_accumulation * args.gradient_accumulation
            window = min(args.gradient_accumulation, len(loaders[0]) - start)
            with autocast_context(device, precision):
                prediction = model(**{key: value.to(device) for key, value in batch.items()})
                loss = multitask_loss(prediction, targets, auxiliary, args.sarcasm_weight, weights)
            scaler.scale(loss / window).backward()
            total += float(loss.detach())
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(loaders[0]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation, _, _, auxiliary_validation = evaluate(loaders[1], precision)
        score = validation["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(loaders[0]), "validation_macro_f1": score,
                        "sarcasm_validation": auxiliary_validation})
        print(json.dumps(history[-1]), flush=True)
        if score > best_f1:
            best_f1, stale = score, 0
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            atomic_json(output / "emotion_metadata.json", metadata)
        else:
            stale += 1
        recovery_root = output / "recovery"
        recovery_root.mkdir(exist_ok=True)
        staging = recovery_root / f"epoch_{epoch:04d}.tmp"
        if staging.exists():
            # Only the fixed staging child under this run's recovery root is removed.
            if not staging.resolve().is_relative_to(recovery_root.resolve()):
                raise ValueError("Unsafe staging directory")
            shutil.rmtree(staging)
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        save_tensor_tree(staging, {"identity": identity, "completed_epoch": epoch, "best_f1": best_f1,
                                  "stale": stale, "history": history, "optimizer": optimizer.state_dict(),
                                  "scaler": scaler.state_dict(), "random": random_state(generator)})
        final = recovery_root / f"epoch_{epoch:04d}"
        # A disconnect after directory rename but before pointer commit leaves an
        # orphan of this same epoch; replay safely replaces only that fixed child.
        if final.exists():
            if final.resolve().parent != recovery_root.resolve():
                raise ValueError("Unsafe recovery epoch directory")
            shutil.rmtree(final)
        staging.replace(final)
        previous = json.loads((output / "latest_recovery.json").read_text())["directory"] if (output / "latest_recovery.json").exists() else None
        atomic_json(output / "latest_recovery.json", {"directory": final.name})
        # Retain the newest completed epoch, avoiding unbounded Drive storage.
        if previous and previous != final.name:
            old = (recovery_root / previous).resolve()
            if old.parent == recovery_root.resolve() and old.is_dir():
                shutil.rmtree(old)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    model = TextEmojiFusionModel.from_pretrained(output).to(device).eval()
    atomic_json(output / "test_evaluation_started.json", {"identity": identity, "selection": "best_validation_macro_f1"})
    test_scores, predicted_ids, confidences, sarcasm_test = evaluate(loaders[2])
    frame = prepared["parts"][2]
    actual = frame["label"].tolist()
    guesses = [labels[index] for index in predicted_ids]
    by_language = {}
    for pair in sorted(frame["language_pair"].unique()):
        indices = frame.index[frame["language_pair"].eq(pair)].tolist()
        by_language[pair] = {**classification_metrics([actual[i] for i in indices], [guesses[i] for i in indices], labels), "rows": len(indices)}
    metadata.update(training_status="complete", validated_language_pairs=prepared["validated_language_pairs"])
    if args.sarcasm_column:
        metadata["sarcasm_supervision"]["validated_language_pairs"] = [
            pair for pair in sorted(frame["language_pair"].unique())
            if set(frame.loc[frame["language_pair"].eq(pair) & frame["sarcasm_target"].ne(-100), "sarcasm_target"]) == {0, 1}]
    model.config.emotion_metadata = metadata
    model.config.save_pretrained(output)
    atomic_json(output / "emotion_metadata.json", metadata)
    from hinglish_emotion.evaluation_guard import write_guard
    write_guard(output / "evaluation_guard.json", prepared["manifest"].to_dict("records"), metadata)
    result = {**identity, **report, "task": "emotion", "device": device,
              "training_mode": "full_fine_tuning", "identity": identity, "emotion_metadata": metadata, "labels": labels,
              "history": history, "best_validation_macro_f1": best_f1, "test": test_scores,
              "test_by_language": by_language, "sarcasm_test": sarcasm_test,
              "test_confidences": confidences, "split_sizes": {name: len(part) for name, part in zip(("train", "validation", "test"), prepared["parts"])}}
    atomic_json(output / "metrics.json", result)
    (output / "MODEL_CARD.md").write_text(
        "# Experimental learned text-emoji fusion\n\n"
        f"Base: {BASE_MODEL}. Seven emotions; text CLS plus learned SHA256-bucket emoji embeddings and a learned residual vector gate. "
        "Absent emoji contribution is forced to zero. Current-message text is the first tokenizer sequence; optional reviewed context is the second. Current-message token IDs receive the available budget first; context receives only the remaining budget (target_first_budget_v1). "
        f"Context trained: {args.use_context}; supervised sarcasm head trained: {bool(args.sarcasm_column)}.\n\n"
        f"Training kind: {prepared['training_data_kind']}; evaluation scope: {prepared['evaluation_scope']}. "
        "Synthetic bootstrap scores do not establish human-labelled or real-world accuracy. No superiority to base models is claimed. "
        "The auxiliary head uses only independently provided reviewed binary labels; blank labels are masked. "
        "Emoji buckets can collide. This is an experimental representation, not a hand-built emoji sentiment score. "
        "Context may be truncated and does not supply facts absent from the supplied conversation. Classifier scores are uncalibrated.\n\n"
        f"Dataset SHA256: {prepared['dataset_sha256']}\nSplit SHA256: {prepared['manifest_sha256']}\n"
        "Best checkpoint selected by validation macro-F1. Test was evaluated once after selection. See metrics.json for support and scores.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "test_macro_f1": test_scores["macro_f1"]}), flush=True)
    return result


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
