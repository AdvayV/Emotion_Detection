import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.text_preparation import duplicate_key
from training.synthetic_phrases import generated_rows, save_dataset, COUNTS
from training.package_synthetic import DATASET_FILES, notebook, snapshot_dataset


class GeneratedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generated_rows()

    def test_exact_counts_nearly_balanced_and_no_text_padding(self):
        self.assertEqual(len(self.rows), 10000)
        self.assertEqual(Counter(row["split"] for row in self.rows), Counter(COUNTS))
        for split in COUNTS:
            counts = Counter(row["emotion"] for row in self.rows if row["split"] == split)
            self.assertEqual(set(counts), set(EMOTION_LABELS))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(row["annotation_status"] == "synthetic" and row["human_reviewed"] == "false" for row in self.rows))
        self.assertTrue(all("sample " not in row["text"] and "example " not in row["text"] for row in self.rows))

    def test_normalized_phrases_groups_and_families_do_not_cross_splits(self):
        self.assertEqual(len({duplicate_key(row["text"]) for row in self.rows}), 10000)
        for field in ("group_id", "scenario_family", "template_family"):
            owners = {}
            for row in self.rows:
                owners.setdefault(row[field], set()).add(row["split"])
            self.assertTrue(all(len(parts) == 1 for parts in owners.values()), field)
        self.assertTrue(any(row["has_emoji"] == "true" for row in self.rows))
        self.assertTrue(any(row["has_emoji"] == "false" for row in self.rows))

    def test_notebook_is_self_contained_and_explicitly_experimental(self):
        nb = notebook("colab_ready_8000")
        text = "".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("--allow-synthetic", text)
        self.assertIn("all_splits.csv", text)
        self.assertIn("[8000, 1000, 1000]", text)
        self.assertNotIn("pilot_hi.csv", text)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


class DatasetSnapshotTests(unittest.TestCase):
    def test_packaging_copies_existing_working_data_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "working", root / "upload" / "data"
            save_dataset(source)
            before = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in DATASET_FILES}
            snapshot_dataset(target, source)
            self.assertEqual(before, {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in DATASET_FILES})
            self.assertEqual(before, {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in DATASET_FILES})

    def test_inconsistent_working_partition_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "working"
            save_dataset(source)
            path = source / "train.csv"
            path.write_bytes(path.read_bytes() + b"unexpected row\n")
            changed = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "does not match"):
                snapshot_dataset(root / "upload" / "data", source)
            self.assertEqual(path.read_bytes(), changed)


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"), "Optional training dependencies")
class SyntheticPreflightTests(unittest.TestCase):
    def test_opt_in_preflight_and_no_false_human_validation(self):
        from training.data_workflow import prepare_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = save_dataset(root)
            data, audit = root / "all_splits.csv", root / "dataset_audit.json"
            self.assertEqual(report["prepared_csv_sha256"], hashlib.sha256(data.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "allow-synthetic"):
                prepare_data(data, "emotion", "raw", data_audit=audit)
            prepared = prepare_data(data, "emotion", "raw", data_audit=audit, allow_synthetic=True)
            self.assertEqual([len(part) for part in prepared["parts"]], [8000, 1000, 1000])
            self.assertEqual(prepared["validated_language_pairs"], [])
            self.assertEqual(prepared["evaluation_scope"], "synthetic_only")
            self.assertEqual(prepared["training_data_kind"], "synthetic_only")
            report["prepared_csv_sha256"] = "invalid"
            audit.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "audit does not match"):
                prepare_data(data, "emotion", "raw", data_audit=audit, allow_synthetic=True)
