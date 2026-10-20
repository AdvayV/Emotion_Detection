"""Package current source for cloud training, or export a completed inference checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.model_catalog import checkpoint_ready

ROOT = Path(__file__).resolve().parents[1]


def code_bundle(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("hinglish_emotion", "training"):
            for path in sorted((ROOT / directory).glob("*.py")):
                archive.write(path, "HinglishEmotionProject/" + path.relative_to(ROOT).as_posix())
        for relative in ("pyproject.toml", "README.md", "notebooks/train_emotion_pilot.ipynb", "data/annotation/README.md"):
            path = ROOT / relative
            if path.is_file():
                archive.write(path, "HinglishEmotionProject/" + relative)
    return output


def export_checkpoint(model_dir, output):
    model_dir = Path(model_dir)
    metadata = emotion_checkpoint_metadata(model_dir)
    if metadata["legacy"] or metadata.get("training_status") != "complete" or not checkpoint_ready(model_dir):
        raise ValueError("Export requires a completed seven-emotion checkpoint")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(model_dir.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            # Never export optimizer state, raw split rows, or test phrases.
            if path.name in {"config.json", "emotion_metadata.json", "evaluation_guard.json", "metrics.json", "MODEL_CARD.md"} or path.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "sentencepiece", "spiece")) or path.suffix == ".safetensors" or path.name in {"pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"} or path.name.startswith("pytorch_model-") and path.suffix == ".bin":
                archive.write(path, model_dir.name + "/" + path.name)
    return output


def export_sarcasm_checkpoint(model_dir, output):
    """Export inference-only files from a completed human-reviewed XLM-R sarcasm run."""
    model_dir = Path(model_dir)
    try:
        metadata = json.loads((model_dir / "sarcasm_metadata.json").read_text(encoding="utf-8"))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Export requires readable sarcasm metadata and config") from exc
    if (metadata.get("task") != "sarcasm_binary" or metadata.get("training_status") != "complete"
            or metadata.get("human_reviewed") is not True or metadata.get("base_model") != "FacebookAI/xlm-roberta-base"
            or metadata.get("language_pairs") != ["hi-en"]
            or metadata.get("labels") != ["not_sarcastic", "sarcastic"]
            or config.get("model_type") != "xlm-roberta"
            or config.get("id2label") != {"0": "not_sarcastic", "1": "sarcastic"}):
        raise ValueError("Export requires a completed, human-reviewed Hindi-English XLM-R sarcasm checkpoint")
    max_length = metadata.get("max_length", 128)
    if type(max_length) is not int or not 1 <= max_length <= 512:
        raise ValueError("Invalid sarcasm checkpoint maximum sequence length")
    files = [path for path in model_dir.iterdir() if path.is_file() and not path.is_symlink() and (
        path.name in {"config.json", "sarcasm_metadata.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "added_tokens.json", "sentencepiece.bpe.model",
                      "spiece.model", "vocab.txt", "merges.txt", "model.safetensors.index.json"}
        or path.name.startswith("model-") and path.suffix == ".safetensors"
        or path.name == "model.safetensors")]
    if not any(path.name == "model.safetensors" or path.name.startswith("model-") and path.suffix == ".safetensors" for path in files):
        raise ValueError("Export requires SafeTensors model weights")
    if not any(path.name == "tokenizer.json" for path in files):
        raise ValueError("Export requires tokenizer.json for local installation")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            archive.write(path, model_dir.name + "/" + path.name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", help="Export a completed inference model; omit for a source-only cloud bundle")
    args = parser.parse_args()
    print(export_checkpoint(args.model_dir, args.output) if args.model_dir else code_bundle(args.output))


if __name__ == "__main__":
    main()
