"""Safely install a completed human-reviewed XLM-R sarcasm inference ZIP."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = "xlmr_sarcasm"
MODEL_ID = "FacebookAI/xlm-roberta-base"
LABELS = ["not_sarcastic", "sarcastic"]
ALLOWED = {
    "config.json", "sarcasm_metadata.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "sentencepiece.bpe.model", "spiece.model",
    "merges.txt", "vocab.txt", "model.safetensors.index.json",
}


def install(archive, models_root=None):
    models_root = Path(models_root or ROOT / "models").resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 100 or sum(item.file_size for item in members) > 5 * 1024**3:
            raise ValueError("Unexpected inference archive size")
        entries, parents = {}, set()
        for item in members:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP member")
            if item.is_dir():
                continue
            if len(name.parts) != 2:
                raise ValueError("Expected one checkpoint folder inside the export ZIP")
            parents.add(name.parts[0])
            filename = name.name
            safe_weight = filename == "model.safetensors" or filename.startswith("model-") and filename.endswith(".safetensors")
            if filename not in ALLOWED and not safe_weight:
                raise ValueError(f"Unexpected inference file: {filename}")
            if filename in entries:
                raise ValueError("Duplicate ZIP member")
            entries[filename] = item
        required = {"config.json", "sarcasm_metadata.json", "tokenizer.json"}
        if len(parents) != 1 or not required <= set(entries) or not any(
            name == "model.safetensors" or name.startswith("model-") and name.endswith(".safetensors") for name in entries
        ):
            raise ValueError("Incomplete sarcasm checkpoint export")
        config = json.loads(z.read(entries["config.json"]))
        metadata = json.loads(z.read(entries["sarcasm_metadata.json"]))
        if (metadata.get("task") != "sarcasm_binary" or metadata.get("training_status") != "complete"
                or metadata.get("human_reviewed") is not True or metadata.get("base_model") != MODEL_ID
                or metadata.get("language_pairs") != ["hi-en"]):
            raise ValueError("Export must be a completed, human-reviewed Hindi-English XLM-R sarcasm run")
        if (config.get("model_type") != "xlm-roberta" or metadata.get("labels") != LABELS
                or config.get("id2label") != {str(i): label for i, label in enumerate(LABELS)}):
            raise ValueError("XLM-R config must declare the exact binary sarcasm labels")
        max_length = metadata.get("max_length", 128)
        if type(max_length) is not int or not 1 <= max_length <= 512:
            raise ValueError("Invalid sarcasm checkpoint maximum sequence length")
        destination = models_root / TARGET
        if destination.exists():
            raise ValueError(f"Preserving existing model at {destination}. Choose a different models root or archive it explicitly first.")
        stage = models_root / (".import_" + uuid.uuid4().hex)
        if not stage.resolve().is_relative_to(models_root) or not destination.resolve().is_relative_to(models_root):
            raise ValueError("Invalid model destination")
        stage.mkdir(parents=True)
        try:
            for filename, item in entries.items():
                with z.open(item) as src, (stage / filename).open("xb") as out:
                    shutil.copyfileobj(src, out, 4 * 1024 * 1024)
            from safetensors import safe_open
            head_found = False
            for file in stage.glob("*.safetensors"):
                with safe_open(file, framework="np") as weights:
                    if "classifier.out_proj.weight" in weights.keys():
                        head_found = True
                        if weights.get_slice("classifier.out_proj.weight").get_shape() != [2, config.get("hidden_size")]:
                            raise ValueError("Incorrect sarcasm classification head")
            if not head_found:
                raise ValueError("Missing trained binary sarcasm classification head")
            stage.rename(destination)
        except Exception:
            for file in stage.iterdir():
                if file.is_file() and file.resolve().parent == stage.resolve():
                    file.unlink()
            stage.rmdir()
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--models-root")
    args = parser.parse_args()
    print(install(args.archive, args.models_root))


if __name__ == "__main__":
    main()
