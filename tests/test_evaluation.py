import unittest

from hinglish_emotion.evaluation import (
    classification_metrics,
    expected_calibration_error,
    sentiment_flip_accuracy,
    variation_consistency,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_metrics(self) -> None:
        true = ["positive", "positive", "neutral", "negative"]
        predicted = ["positive", "neutral", "neutral", "negative"]
        metrics = classification_metrics(true, predicted)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["positive"]["neutral"], 1)

    def test_consistency_and_flip_metrics(self) -> None:
        self.assertEqual(variation_consistency([["positive", "positive"], ["negative", "neutral"]]), 0.5)
        pairs = [("positive", "positive", "negative", "negative")]
        self.assertEqual(sentiment_flip_accuracy(pairs), 1.0)

    def test_calibration_error(self) -> None:
        value = expected_calibration_error([0.9, 0.6], [True, False], bins=2)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

