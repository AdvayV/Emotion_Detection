import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from safetensors.numpy import save_file

from hinglish_emotion.checkpoints import emotion_checkpoint, xlmr_checkpoint
from hinglish_emotion.model_catalog import ENCODER_BY_KEY
from hinglish_emotion.tasks import EMOTION_LABELS
from training.install_checkpoint import install


class EncoderInstallTests(unittest.TestCase):
    def fixture(self, root, encoder, pairs=("hi-en",), provenance=None, model_type=None):
        source = root / "source"
        source.mkdir()
        key = encoder
        spec = ENCODER_BY_KEY[key]
        model_type = model_type or ("xlm-roberta" if key == "xlmr" else "bert")
        metadata = {
            "schema_version": 1, "task": "emotion", "labels": list(EMOTION_LABELS),
            "text_mode": "raw", "preparation_version": 1, "max_length": 128,
            "trained_language_pairs": list(pairs), "validated_language_pairs": [],
            "training_status": "complete", "base_model": provenance or spec.model_id,
        }
        (source / "config.json").write_text(json.dumps({
            "model_type": model_type, "hidden_size": 4,
            "id2label": dict(enumerate(EMOTION_LABELS)), "emotion_metadata": metadata,
        }), encoding="utf-8")
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        head = "classifier.out_proj.weight" if key == "xlmr" else "classifier.weight"
        save_file({head: np.zeros((7, 4), dtype=np.float32)}, source / "model.safetensors")
        if key != "xlmr":
            (source / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
        archive = root / f"{key}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for file in source.iterdir():
                z.write(file, f"export/{file.name}")
        return archive

    def test_installs_mbert_and_muril_with_bert_heads(self):
        for encoder in ("mbert", "muril"):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = install(self.fixture(root, encoder), root / "models")
                self.assertEqual(path.name, f"{encoder}_hi_en_emotion")

    def test_xlmr_install_remains_compatible_with_unknown_base_on_xlmr_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root, "xlmr", provenance="tiny")
            self.assertEqual(install(archive, root / "models").name, "xlmr_hi_en_emotion")

    def test_bert_config_with_wrong_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root, "mbert", provenance="unknown/bert-checkpoint")
            with self.assertRaisesRegex(ValueError, "provenance"):
                install(archive, root / "models")

    def test_muril_fr_en_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root, "muril", pairs=("fr-en",))
            with self.assertRaisesRegex(ValueError, "language scope"):
                install(archive, root / "models")

    def test_discovery_uses_encoder_specific_folder_names_and_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mbert_hi_en_emotion"
            path.mkdir()
            metadata = {"training_status": "complete", "trained_language_pairs": ["hi-en"], "base_model": ENCODER_BY_KEY["mbert"].model_id}
            self.assertEqual(emotion_checkpoint(root, "hi-en", "mbert", lambda p: p == path, lambda p: metadata), path)
            self.assertIsNone(emotion_checkpoint(root, "fr-en", "muril", lambda p: True, lambda p: metadata))
            self.assertIsNone(xlmr_checkpoint(root, "hi-en", lambda p: False, lambda p: metadata))


if __name__ == "__main__":
    unittest.main()
