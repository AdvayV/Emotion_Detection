"""Package current source for cloud training, or export a completed inference checkpoint."""

from __future__ import annotations

import argparse
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
            if path.name in {"config.json", "emotion_metadata.json", "metrics.json", "MODEL_CARD.md"} or path.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "sentencepiece", "spiece")) or path.suffix == ".safetensors" or path.name in {"pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"} or path.name.startswith("pytorch_model-") and path.suffix == ".bin":
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
