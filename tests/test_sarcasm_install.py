import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
from safetensors.numpy import save_file

from hinglish_emotion.sarcasm import LocalSarcasmClassifier
from training.cloud_bundle import export_sarcasm_checkpoint
from training.install_sarcasm_checkpoint import install


LABELS = ["not_sarcastic", "sarcastic"]
MODEL_ID = "FacebookAI/xlm-roberta-base"


class SarcasmInstallTests(unittest.TestCase):
    def fixture(self, root, *, reviewed=True, status="complete", pairs=("hi-en",), labels=LABELS,
                model_type="xlm-roberta", head_rows=2, extra=None):
        source = root / "source"
        source.mkdir()
        metadata = {"schema_version": 1, "task": "sarcasm_binary", "labels": labels,
                    "base_model": MODEL_ID, "language_pairs": list(pairs),
                    "training_status": status, "human_reviewed": reviewed}
        config = {"model_type": model_type, "hidden_size": 4, "num_labels": 2,
                  "id2label": {str(i): label for i, label in enumerate(labels)}}
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (source / "sarcasm_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        save_file({"classifier.out_proj.weight": np.zeros((head_rows, 4), dtype=np.float32)}, source / "model.safetensors")
        for name, body in (extra or {}).items():
            (source / name).write_text(body, encoding="utf-8")
        archive = root / "sarcasm.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for file in source.iterdir():
                z.write(file, f"xlmr_sarcasm/{file.name}")
        return archive, source

    def test_install_and_preserve_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _ = self.fixture(root)
            path = install(archive, root / "models")
            self.assertEqual(path.name, "xlmr_sarcasm")
            with self.assertRaisesRegex(ValueError, "Preserving existing"):
                install(archive, root / "models")

    def test_invalid_review_scope_head_and_labels_are_rejected(self):
        cases = [
            ({"reviewed": False}, "human-reviewed"),
            ({"status": "training"}, "human-reviewed"),
            ({"pairs": ("fr-en",)}, "human-reviewed"),
            ({"labels": ["negative", "positive"]}, "exact binary sarcasm labels"),
            ({"head_rows": 3}, "classification head"),
            ({"model_type": "bert"}, "exact binary sarcasm labels"),
        ]
        for options, error in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                archive, _ = self.fixture(Path(directory), **options)
                with self.assertRaisesRegex(ValueError, error):
                    install(archive, Path(directory) / "models")

    def test_installer_rejects_unexpected_file_and_traversal(self):
        for member, expected in (("xlmr_sarcasm/trainer_state/optimizer.pt", "checkpoint folder"),
                                 ("../config.json", "Unsafe")):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive, _ = self.fixture(root)
                with zipfile.ZipFile(archive, "a") as z:
                    z.writestr(member, "bad")
                with self.assertRaisesRegex(ValueError, expected):
                    install(archive, root / "models")

    def test_export_contains_inference_files_but_not_trainer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source = self.fixture(root)
            (source / "optimizer.pt").write_text("secret", encoding="utf-8")
            output = export_sarcasm_checkpoint(source, root / "export.zip")
            with zipfile.ZipFile(output) as z:
                names = {Path(name).name for name in z.namelist()}
            self.assertIn("sarcasm_metadata.json", names)
            self.assertIn("model.safetensors", names)
            self.assertNotIn("optimizer.pt", names)

    def test_local_classifier_accepts_expected_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _ = self.fixture(root)
            model_path = install(archive, root / "models")
            fake_torch = MagicMock()
            fake_transformers = MagicMock()
            with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                classifier = LocalSarcasmClassifier(model_path)
            self.assertEqual(classifier.metadata["labels"], LABELS)
            fake_transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
