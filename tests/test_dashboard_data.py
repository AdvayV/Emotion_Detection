import importlib.util
from pathlib import Path
import unittest


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(PANDAS_AVAILABLE, "Dashboard dependencies are optional")
class DashboardDataTests(unittest.TestCase):
    def test_valid_dataset_report(self):
        import pandas as pd

        from dashboard.data import dataframe_report

        report = dataframe_report(pd.DataFrame({"text": ["accha", "theek"], "label": ["positive", "neutral"]}))
        self.assertTrue(report.valid)
        self.assertEqual(report.label_counts["positive"], 1)

    def test_missing_columns_are_rejected(self):
        import pandas as pd

        from dashboard.data import dataframe_report

        with self.assertRaises(ValueError):
            dataframe_report(pd.DataFrame({"message": ["accha"]}))

    def test_bundled_xlsx_is_read_with_balanced_labels(self):
        from dashboard.data import dataframe_report, read_dataset

        path = Path(__file__).resolve().parents[1] / "data" / "hinglish_emotion_phrases.xlsx"
        frame = read_dataset(path)
        report = dataframe_report(frame)
        self.assertEqual(report.rows, 75)
        self.assertEqual(report.label_counts, {"negative": 25, "neutral": 25, "positive": 25})
        self.assertTrue(report.valid)

    def test_unsupported_dataset_extension_is_rejected(self):
        from dashboard.data import read_dataset

        with self.assertRaises(ValueError):
            read_dataset("dataset.json")
