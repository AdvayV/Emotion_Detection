import importlib.util
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
