import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.emotions import LocalEmotionClassifier
from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.text_preparation import prepare_text

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def test_raw_preserves_case_negation_emphasis_and_complete_emoji(self):
        text = "  Je suis TRÈS triste\nbut khush nahi 👩🏽‍💻 acchaaa!!!  "
        self.assertEqual(prepare_text(text), "Je suis TRÈS triste but khush nahi 👩🏽‍💻 acchaaa!!!")
        self.assertEqual(prepare_text("acchaaa", "normalized"), "accha")
        with self.assertRaises(ValueError):
            prepare_text("hello", version=999)

    def test_sentiment_checkpoint_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"id2label": {0: "positive", 1: "neutral", 2: "negative"}}))
            with self.assertRaisesRegex(ValueError, "not a seven-emotion"):
                LocalEmotionClassifier(path)

    def test_legacy_metadata_keeps_actual_old_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"id2label": dict(enumerate(EMOTION_LABELS))}))
            metadata = emotion_checkpoint_metadata(path)
            self.assertTrue(metadata["legacy"])
            self.assertEqual(metadata["text_mode"], "normalized")
            self.assertEqual(metadata["validated_language_pairs"], [])

    def test_notebook_cells_parse_and_have_no_executed_outputs(self):
        notebook = json.loads((ROOT / "notebooks" / "train_emotion_pilot.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"), "Optional training dependencies")
class DataWorkflowTests(unittest.TestCase):
    def write_fixture(self, directory):
        path = Path(directory) / "labels.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_id", "text", "emotion", "language_pair", "group_id", "source"])
            writer.writeheader()
            for label in EMOTION_LABELS:
                for index in range(20):
                    writer.writerow({"row_id": f"{label}_{index}", "text": f"Phrase {label} acchaaa sample {index}",
                                     "emotion": label, "language_pair": "hi-en", "group_id": f"{label}_{index}",
                                     "source": "synthetic_smoke_fixture"})
        return path

    def test_frozen_split_survives_preprocessing_changes_and_rejects_data_changes(self):
        from training.data_workflow import prepare_data
        with tempfile.TemporaryDirectory() as directory:
            data = self.write_fixture(directory)
            manifest = Path(directory) / "split.csv"
            first = prepare_data(data, "emotion", "raw", 42, manifest, allow_synthetic=True)
            second = prepare_data(data, "emotion", "normalized", 999, manifest, allow_synthetic=True)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual([part["row_id"].tolist() for part in first["parts"]],
                             [part["row_id"].tolist() for part in second["parts"]])
            self.assertEqual(second["validated_language_pairs"], [])
            data.write_text(data.read_text(encoding="utf-8").replace("sample 0", "changed sample 0"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_data(data, "emotion", "raw", 42, manifest, allow_synthetic=True)

    def test_public_data_requires_matching_approved_audit(self):
        from training.data_workflow import prepare_data
        with tempfile.TemporaryDirectory() as directory:
            data = self.write_fixture(directory)
            data.write_text(data.read_text().replace("synthetic_smoke_fixture", "MaSaC_EDiReF_TaskA"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved dataset audit"):
                prepare_data(data, "emotion", "raw")

    def test_matching_status_column_without_review_receipt_cannot_build_pilot(self):
        from training.annotations import combine, write_csv
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = self.write_fixture(root)
            audit = root / "public_audit.json"
            audit.write_text(json.dumps({"reuse_conditions_verified": True, "prepared_csv_sha256": hashlib.sha256(public.read_bytes()).hexdigest()}))
            supplement = root / "supplement.csv"
            write_csv(supplement, [{"text": "reviewed phrase", "emotion": "joy", "language_pair": "hi-en", "annotation_status": "human_adjudicated"}])
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"ready": True, "prepared_csv_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "matching completed reconciliation"):
                combine(public, audit, supplement, receipt)

    def test_illustrative_demo_remains_blocked(self):
        from training.data_workflow import prepare_data
        with self.assertRaisesRegex(ValueError, "Illustrative"):
            prepare_data(ROOT / "data" / "demo" / "emotions" / "mixed_emotions_demo.csv", "emotion", "raw")

    def test_annotation_export_requires_independent_completed_reviews(self):
        from training.annotations import initialise, read_csv, write_csv, reviewer_sheets, reconcile
        with tempfile.TemporaryDirectory() as directory:
            collection = initialise(directory)
            rows = read_csv(collection)
            self.assertEqual(len(rows), 350)
            self.assertTrue(all(not row["text"] for row in rows))
            for row in rows:
                row["text"] = f"Phrase {row['row_id']}"
            write_csv(collection, rows)
            reviewer_sheets(collection, directory)
            counts = {"train": 20, "validation": 10, "test": 20}
            for suffix, reviewer in (("a", "reviewer_A"), ("b", "reviewer_B")):
                path = Path(directory) / f"reviewer_{suffix}.csv"
                reviews = read_csv(path)
                indices = {split: 0 for split in counts}
                for row in reviews:
                    row["reviewer_id"] = reviewer
                    row["emotion"] = EMOTION_LABELS[indices[row["split"]] // counts[row["split"]]]
                    indices[row["split"]] += 1
                write_csv(path, reviews)
            result, report = reconcile(collection, Path(directory) / "reviewer_a.csv", Path(directory) / "reviewer_b.csv")
            self.assertTrue(report["ready"])
            self.assertEqual(len(result), 350)
            wrong = read_csv(Path(directory) / "reviewer_b.csv")
            wrong[0]["reviewer_id"] = "reviewer_A"
            write_csv(Path(directory) / "reviewer_b.csv", wrong)
            _, report = reconcile(collection, Path(directory) / "reviewer_a.csv", Path(directory) / "reviewer_b.csv")
            self.assertFalse(report["ready"])
            self.assertEqual(len(report["pending"]), 1)

    def test_masac_import_handles_test_schema_and_quarantines_leakage(self):
        from training.audit_masac import audit_source, FILES
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            for split, filename in FILES.items():
                texts = [f"{split} phrase {label} {index}" for label in EMOTION_LABELS for index in range(2)]
                labels = [label for label in EMOTION_LABELS for index in range(2)]
                texts.append("same phrase in separate splits")
                labels.append("joy")
                texts.append(f"{split} excluded phrase")
                labels.append("contempt")
                data = [{"episode": "shared_episode", "utterances": texts,
                         "labels" if split == "test" else "emotions": labels}]
                (root / filename).write_text(json.dumps(data), encoding="utf-8")
            rows, report = audit_source(root)
            self.assertEqual(report["quarantined_cross_split_duplicate_rows"], 3)
            self.assertTrue(all(row["emotion"] != "contempt" for row in rows))
            self.assertFalse(report["reuse_conditions_verified"])
            self.assertFalse(report["ready_for_export"])

    def test_checkpoint_export_excludes_training_state_and_raw_phrases(self):
        from training.cloud_bundle import export_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "xlmr_emotion"
            model.mkdir()
            config = {"id2label": dict(enumerate(EMOTION_LABELS)), "emotion_metadata": {
                "schema_version": 1, "task": "emotion", "labels": list(EMOTION_LABELS),
                "text_mode": "raw", "preparation_version": 1, "max_length": 128,
                "trained_language_pairs": ["hi-en"], "validated_language_pairs": ["hi-en"],
                "training_status": "complete",
            }}
            (model / "config.json").write_text(json.dumps(config))
            for name in ("tokenizer.json", "model.safetensors", "latest_training_state.pt", "split_manifest.csv", "test_predictions.csv"):
                (model / name).write_text("fixture")
            export = export_checkpoint(model, root / "inference.zip")
            with zipfile.ZipFile(export) as archive:
                names = archive.namelist()
            self.assertIn("xlmr_emotion/model.safetensors", names)
            self.assertFalse(any(name.endswith((".pt", ".csv")) for name in names))
