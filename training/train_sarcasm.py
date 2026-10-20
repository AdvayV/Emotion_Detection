"""GPU-only supervised binary sarcasm fine-tuning from explicitly labelled CSV splits."""
from __future__ import annotations
import argparse, hashlib, json, inspect
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABELS = ["not_sarcastic", "sarcastic"]
REQUIRED = {"text", "sarcasm", "language_pair", "split", "group_id"}

def read_splits(path, allow_synthetic=False):
    frame = pd.read_csv(path).fillna("")
    missing = REQUIRED - set(frame.columns)
    if missing: raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "human_reviewed" not in frame.columns: raise ValueError("CSV requires human_reviewed=true on every row")
    reviewed = frame.human_reviewed.astype(str).str.lower().isin({"true", "1", "yes"})
    if not reviewed.all(): raise ValueError("Every example must be explicitly human_reviewed=true")
    if "synthetic" in frame and frame.synthetic.astype(str).str.lower().isin({"true", "1", "yes"}).any() and not allow_synthetic:
        raise ValueError("Synthetic examples require --allow-synthetic")
    frame["text"] = frame.text.astype(str).str.strip()
    frame["group_id"] = frame.group_id.astype(str).str.strip()
    frame["split"] = frame.split.astype(str).str.lower().str.strip()
    frame["sarcasm"] = frame.sarcasm.astype(str).str.lower().str.strip()
    aliases = {"0":"not_sarcastic", "false":"not_sarcastic", "no":"not_sarcastic", "1":"sarcastic", "true":"sarcastic", "yes":"sarcastic"}
    frame["sarcasm"] = frame.sarcasm.map(lambda v: aliases.get(v, v))
    if frame.text.eq("").any() or frame.group_id.eq("").any(): raise ValueError("text and group_id must be non-empty")
    if not frame.sarcasm.isin(LABELS).all(): raise ValueError("sarcasm values must be binary: not_sarcastic/sarcastic (or 0/1)")
    if not frame.split.isin({"train", "validation", "test"}).all(): raise ValueError("split must be train, validation, or test")
    normalized_languages = frame.language_pair.astype(str).str.lower().str.strip().replace({"en-hi":"hi-en", "hinglish":"hi-en", "hindi-english":"hi-en"})
    if normalized_languages.ne("hi-en").any(): raise ValueError("This handoff accepts Hindi-English sarcasm labels only")
    frame["language_pair"] = normalized_languages
    if frame.groupby("group_id").split.nunique().gt(1).any(): raise ValueError("Duplicate/group leakage across splits")
    parts = {name: frame[frame.split == name].copy() for name in ("train", "validation", "test")}
    for name, part in parts.items():
        if part.empty or set(part.sarcasm) != set(LABELS): raise ValueError(f"{name} must contain both sarcasm classes")
    if frame.text.map(lambda x: hashlib.sha256(" ".join(x.casefold().split()).encode()).hexdigest()).duplicated().any():
        raise ValueError("Duplicate text rows are not permitted")
    return parts

class Dataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.enc = tokenizer(frame.text.tolist(), truncation=True, padding="max_length", max_length=max_length)
        self.labels = [LABELS.index(v) for v in frame.sarcasm]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {**{k: torch.tensor(v[i]) for k,v in self.enc.items()}, "labels":torch.tensor(self.labels[i])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base", choices=["FacebookAI/xlm-roberta-base"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    if min(a.epochs, a.batch_size, a.gradient_accumulation) < 1 or not 1 <= a.max_length <= 512:
        raise ValueError("Training sizes must be positive; max-length must be 1..512")
    if not torch.cuda.is_available():
        raise RuntimeError("Sarcasm training requires a Colab GPU; no CPU fallback is provided")
    parts = read_splits(a.data, a.allow_synthetic)
    identity = {"data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                "model": a.model, "epochs": a.epochs, "batch_size": a.batch_size,
                "gradient_accumulation": a.gradient_accumulation, "max_length": a.max_length,
                "allow_synthetic": a.allow_synthetic, "seed": 42}
    out = Path(a.output).resolve()
    identity_path = out / "run_identity.json"
    if out.exists() and any(out.iterdir()):
        if not a.resume or not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing run differs or resume was not requested; choose a new Drive output folder")
        if (out / "sarcasm_metadata.json").is_file():
            print("Completed matching run retained:", out)
            return
    out.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    from transformers import set_seed
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2, label2id={x:i for i,x in enumerate(LABELS)},
        id2label=dict(enumerate(LABELS)), trust_remote_code=False, use_safetensors=True)
    ds = {k: Dataset(v, tok, a.max_length) for k,v in parts.items()}
    def metrics(pred):
        y, z = pred.label_ids, pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(y,z), "macro_f1": f1_score(y,z,average="macro",zero_division=0)}
    settings = dict(output_dir=str(out/"trainer_state"), num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
                    gradient_accumulation_steps=a.gradient_accumulation, learning_rate=2e-5,
                    save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
                    metric_for_best_model="macro_f1", greater_is_better=True,
                    save_safetensors=True, fp16=True, report_to="none", seed=42, data_seed=42)
    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    settings[eval_key] = "epoch"
    args = TrainingArguments(**settings)
    processor_key = "processing_class" if "processing_class" in inspect.signature(Trainer).parameters else "tokenizer"
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                      compute_metrics=metrics, **{processor_key: tok})
    state_dir = out / "trainer_state"
    checkpoint = get_last_checkpoint(str(state_dir)) if a.resume and state_dir.is_dir() else None
    trainer.train(resume_from_checkpoint=checkpoint)
    val, test = trainer.evaluate(ds["validation"]), trainer.predict(ds["test"])
    trainer.model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    y, z = test.label_ids, test.predictions.argmax(-1)
    metadata = {"schema_version":1, "task":"sarcasm_binary", "labels":LABELS,
                "base_model":a.model, "language_pairs":["hi-en"], "training_status":"complete",
                "human_reviewed":True, "synthetic_opt_in":a.allow_synthetic, "max_length":a.max_length,
                "selected_validation_macro_f1":val["eval_macro_f1"],
                "test_metrics":{"accuracy":accuracy_score(y,z), "macro_f1":f1_score(y,z,average="macro",zero_division=0),
                                "per_class":classification_report(y,z,labels=[0,1],target_names=LABELS,output_dict=True,zero_division=0)},
                "split_rows":{k:len(v) for k,v in parts.items()}, "split_sha256":identity["data_sha256"]}
    (out/"sarcasm_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))

if __name__ == "__main__":
    main()
