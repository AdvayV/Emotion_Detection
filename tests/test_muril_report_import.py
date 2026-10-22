"""Validate authentic Colab report imports without installing model weights."""
import copy
import io
import json
from pathlib import Path
import unittest
import zipfile

from training.muril_study import load_comparison_export


ROOT = Path(__file__).resolve().parents[1]


class MurilReportImportTests(unittest.TestCase):
    def setUp(self):
        study = ROOT / "colab/muril_study"
        self.contract = json.loads((study / "study_contract.json").read_text(encoding="utf-8"))
        self.baseline = json.loads((study / "xlmr_baseline_metrics.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(ROOT / "muril_hi_en_comparison.zip") as archive:
            self.report = json.loads(archive.read("comparison.json"))

    def archive(self, report, extra=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("comparison.json", json.dumps(report))
            if extra:
                archive.writestr(*extra)
        stream.seek(0)
        return stream

    def test_real_colab_export_has_recorded_scores_and_local_scope_notes(self):
        report = load_comparison_export(ROOT / "muril_hi_en_comparison.zip", self.contract, self.baseline)
        self.assertEqual(report["rows"][1]["Accuracy"], .803)
        self.assertAlmostEqual(report["rows"][1]["Macro-F1"], .8123164552869595)
        self.assertEqual(len(report["per_class"]), 14)
        self.assertTrue(any("require its separate checkpoint" in note for note in report["notes"]))

    def test_wrong_test_identity_is_refused(self):
        self.report["test_sha256"] = "different-test"
        with self.assertRaisesRegex(ValueError, "different test"):
            load_comparison_export(self.archive(self.report), self.contract, self.baseline)

    def test_invalid_or_inconsistent_results_are_refused(self):
        for kind in ("duplicate_emotion", "nan", "aggregate", "training_size", "baseline"):
            with self.subTest(kind=kind):
                report = copy.deepcopy(self.report)
                if kind == "duplicate_emotion":
                    report["per_class"][-1]["Emotion"] = report["per_class"][-2]["Emotion"]
                elif kind == "nan":
                    report["per_class"][-1]["Precision"] = float("nan")
                elif kind == "aggregate":
                    report["rows"][1]["Accuracy"] = .99
                elif kind == "training_size":
                    report["rows"][1]["Training phrases"] = 16000
                else:
                    report["rows"][0]["Accuracy"] = .95
                with self.assertRaises(ValueError):
                    load_comparison_export(self.archive(report), self.contract, self.baseline)

    def test_unexpected_archive_files_and_invalid_json_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unexpected comparison"):
            load_comparison_export(self.archive(self.report, ("../weights.bin", "unused")), self.contract, self.baseline)
        with self.assertRaises(ValueError):
            load_comparison_export(self.archive([]), self.contract, self.baseline)


if __name__ == "__main__":
    unittest.main()
