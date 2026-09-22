import importlib.util
from pathlib import Path
import unittest


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(PANDAS_AVAILABLE, "Dashboard dependencies are optional")
class DashboardEvaluationTests(unittest.TestCase):
    def test_bundled_workbook_runs_through_real_pipeline(self):
        from dashboard.data import read_dataset
        from dashboard.evaluation import evaluate_pipeline
        from hinglish_emotion.pipeline import EmotionPipeline

        path = Path(__file__).resolve().parents[1] / "data" / "hinglish_emotion_phrases.xlsx"
        result = evaluate_pipeline(read_dataset(path), EmotionPipeline(), "Rule baseline")

        self.assertEqual(result.summary["rows"], 75)
        self.assertGreaterEqual(result.summary["accuracy"], 0.80)
        self.assertGreaterEqual(result.summary["macro_f1"], 0.80)
        self.assertEqual(set(result.per_class["sentiment"]), {"positive", "neutral", "negative"})
        self.assertEqual(result.confusion.to_numpy().sum(), 75)
        self.assertEqual(result.calibration["count"].sum(), 75)

    def test_evaluation_rejects_unlabelled_frames(self):
        import pandas as pd

        from dashboard.evaluation import evaluate_pipeline
        from hinglish_emotion.pipeline import EmotionPipeline

        with self.assertRaises(ValueError):
            evaluate_pipeline(pd.DataFrame({"text": ["accha"]}), EmotionPipeline(), "Rule")


if __name__ == "__main__":
    unittest.main()
