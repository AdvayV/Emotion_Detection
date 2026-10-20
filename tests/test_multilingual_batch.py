import ast
from collections import Counter
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hinglish_emotion.batch import normalize_records, predict_dataset, read_frame
from hinglish_emotion.emotions import EmotionDetector, EmotionResult
from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.checkpoints import checkpoint_version, xlmr_checkpoint
from hinglish_emotion.evaluation_guard import write_guard, training_hashes
from training.package_multilingual import notebook
from training.multilingual_data import multilingual_rows


class FilePredictionTests(unittest.TestCase):
    def test_unlabelled_file_never_reports_accuracy(self):
        rows=normalize_records([{"message":"Main khush hoon","language":"en-hi"},{"message":"Je suis triste today","language":"en-fr"}])
        report=predict_dataset(rows,EmotionDetector())
        self.assertIsNone(report["overall"])
        self.assertEqual(report["total_rows"],2)
        self.assertEqual(report["labelled_rows"],0)
        self.assertEqual([x["predicted"] for x in report["predictions"]],["joy","sadness"])

    def test_accuracy_uses_labelled_rows_and_test_partition_only(self):
        rows=normalize_records([{"text":"Main khush hoon","label":"sad","split":"train"},{"text":"Je suis triste","label":"sad","language_pair":"fr-en","split":"test"},{"text":"Main khush hoon","label":"","split":"test"}])
        report=predict_dataset(rows,EmotionDetector())
        self.assertEqual(report["total_rows"],2)
        self.assertEqual(report["overall"]["rows"],1)
        self.assertEqual(report["overall"]["accuracy"],1)
        self.assertEqual(report["unlabelled_rows"],1)

    def test_bad_sentiment_labels_and_empty_phrases_are_rejected(self):
        for row in [{"text":"hello","label":"positive"},{"text":"","label":"joy"}]:
            with self.assertRaises(ValueError):normalize_records([row])

    def test_csv_and_xlsx_have_matching_schema(self):
        import pandas as pd
        frame=pd.DataFrame([{"phrase":"Bonjour, I am happy","emotion":"happy","language_pair":"en-fr"}])
        workbook=io.BytesIO();frame.to_excel(workbook,index=False)
        a=read_frame(frame.to_csv(index=False).encode(),"data.csv")
        b=read_frame(workbook.getvalue(),"data.xlsx")
        self.assertEqual(normalize_records(a.to_dict("records")),normalize_records(b.to_dict("records")))

    def test_duplicate_csv_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):
            read_frame(b"text,text\na,b\n","ambiguous.csv")

    def test_exported_predictions_retain_dataset_provenance(self):
        rows=normalize_records([{"text":"Main khush hoon","emotion":"joy","source":"synthetic_fixture","annotation_status":"synthetic","split":"test"}])
        first=predict_dataset(rows,EmotionDetector())
        second=predict_dataset(normalize_records(first["predictions"]),EmotionDetector())
        self.assertTrue(second["synthetic_dataset"])
        self.assertEqual(second["predictions"][0]["dataset_source"],"synthetic_fixture")

    def test_experimental_scope_is_separate_from_score_ambiguity(self):
        scores={label:.001 for label in EMOTION_LABELS};scores["joy"]=.994
        result=EmotionResult("joy",scores,experimental=True)
        self.assertFalse(result.needs_context)
        self.assertTrue(result.experimental)

    def test_guard_detects_training_overlap_and_rejects_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            metadata={"dataset_sha256":"data","split_sha256":"split"}
            write_guard(path/"evaluation_guard.json",[{"text":"Main khush hoon","split":"train"}],metadata)
            detector=EmotionDetector()
            detector.classifier.model_path=str(path);detector.classifier.metadata=metadata
            rows=normalize_records([{"text":"Main khush hoon","emotion":"joy"}])
            self.assertEqual(predict_dataset(rows,detector)["training_overlap_rows"],1)
            detector.classifier.metadata={**metadata,"dataset_sha256":"wrong"}
            with self.assertRaises(ValueError):training_hashes(detector)


class CheckpointSelectionTests(unittest.TestCase):
    def test_hindi_checkpoint_is_not_selected_for_french(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"xlmr_emotion";path.mkdir()
            reader=lambda path:{"training_status":"complete","trained_language_pairs":["hi-en"]}
            ready=lambda path:path.name=="xlmr_emotion"
            self.assertEqual(xlmr_checkpoint(root,"hi-en",ready,reader),path)
            self.assertIsNone(xlmr_checkpoint(root,"fr-en",ready,reader))

    def test_weight_changes_invalidate_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/"config.json").write_text("{}")
            weights=p/"model.safetensors";weights.write_bytes(b"old")
            before=checkpoint_version(p);weights.write_bytes(b"new and longer")
            self.assertNotEqual(before,checkpoint_version(p))


class MultilingualBootstrapTests(unittest.TestCase):
    def test_counts_labels_and_cross_language_groups_are_disjoint(self):
        rows=multilingual_rows()
        self.assertEqual(Counter((r["language_pair"],r["split"]) for r in rows),Counter({("hi-en","train"):8000,("fr-en","train"):8000,("hi-en","validation"):1000,("fr-en","validation"):1000,("hi-en","test"):1000,("fr-en","test"):1000}))
        self.assertEqual(len({row["row_id"] for row in rows}),20000)
        self.assertTrue(all(row["annotation_status"]=="synthetic" and row["human_reviewed"]=="false" for row in rows))
        for field in ["group_id","scenario_family","template_family"]:
            owners={}
            for row in rows:owners.setdefault(row[field],set()).add(row["split"])
            self.assertTrue(all(len(value)==1 for value in owners.values()),field)

    def test_colab_notebook_is_executable_python_and_does_not_claim_training_done(self):
        nb=notebook()
        for cell in nb["cells"]:
            if cell["cell_type"]=="code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"],[])
                self.assertIsNone(cell["execution_count"])
        text="".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("--allow-synthetic",text)
        self.assertIn("USE_OWN_DATA",text)
        self.assertIn('"fr-en"',text)
