"""Discover language-compatible emotion checkpoints and invalidate stale caches."""
from pathlib import Path
import hashlib
from .model_catalog import checkpoint_ready
from .checkpoint_metadata import emotion_checkpoint_metadata
from .model_catalog import ENCODER_BY_KEY


def checkpoint_version(path) -> str:
    path = Path(path)
    names = {"config.json", "emotion_metadata.json", "sarcasm_metadata.json", "evaluation_guard.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"}
    files = [p for p in path.iterdir() if p.is_file() and (p.name in names or p.suffix == ".safetensors" or p.name.endswith(".index.json"))]
    identity = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files)]
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def emotion_checkpoint(root, language_pair, encoder="xlmr", ready=checkpoint_ready, read_metadata=emotion_checkpoint_metadata):
    if encoder not in ENCODER_BY_KEY:
        raise ValueError(f"Unknown encoder: {encoder}")
    root = Path(root)
    if encoder == "muril" and language_pair != "hi-en":
        return None
    candidates = [f"{encoder}_multilingual_emotion", f"{encoder}_{language_pair.replace('-', '_')}_emotion", f"{encoder}_emotion"]
    for name in candidates:
        path = root / name
        if not ready(path):
            continue
        try:
            metadata = read_metadata(path)
        except ValueError:
            continue
        expected_base = ENCODER_BY_KEY[encoder].model_id
        if encoder != "xlmr" and metadata.get("base_model") != expected_base:
            continue
        if metadata.get("training_status") == "complete" and language_pair in metadata.get("trained_language_pairs", []):
            return path
    return None


def xlmr_checkpoint(root, language_pair, ready=checkpoint_ready, read_metadata=emotion_checkpoint_metadata):
    """Backward-compatible XLM-R discovery entry point."""
    return emotion_checkpoint(root, language_pair, "xlmr", ready, read_metadata)
