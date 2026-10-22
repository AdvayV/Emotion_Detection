"""No-download tests for sequential disagreement scoring and queue exports."""
import csv
import io
import unittest

from hinglish_emotion.disagreement import add_reviewed_candidate, compare_phrases, jensen_shannon, queue_csv
from hinglish_emotion.tasks import EMOTION_LABELS


def distribution(label):
    values = {name: 0.0 for name in EMOTION_LABELS}
    values[label] = 1.0
    return values


class FakeResult:
    source = "trained_checkpoint"
    warnings = ("test warning",)
    truncated = False
    experimental = True
    model_name = "fake model"

    def __init__(self, label):
        self.probabilities = distribution(label)


class FakeClassifier:
    active = 0
    peak = 0

    def __init__(self, path):
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        self.label = str(path)

    def predict_text(self, text):
        return FakeResult(self.label)

    def close(self):
        type(self).active -= 1

    def __del__(self):
        if type(self).active:
            type(self).active -= 1


class DisagreementTests(unittest.TestCase):
    def test_disagreement_and_identical_probabilities(self):
        same = jensen_shannon(distribution("joy"), distribution("joy"))
        self.assertEqual(same, (0.0, 0.0))
        results = compare_phrases(
            ["A phrase"], "hi-en", [("A", "joy", "v1"), ("B", "sadness", "v2")],
            classifier_factory=FakeClassifier,
        )
        self.assertTrue(results[0]["disagreement"])
        self.assertFalse(results[0]["uncertainty"])
        self.assertEqual(len(results[0]["pairwise_js"]), 1)
        self.assertEqual(results[0]["pairwise_js"][0]["divergence"], 1.0)
        self.assertEqual(results[0]["predictions"][0]["probabilities"], distribution("joy"))

    def test_sequential_factory_and_cache_clear(self):
        cleared = []
        FakeClassifier.peak = 0
        compare_phrases(
            ["one", "two"], "hi-en", [("A", "joy", "1"), ("B", "fear", "2")],
            classifier_factory=FakeClassifier, clear_shared_cache=lambda: cleared.append(True),
        )
        self.assertEqual(FakeClassifier.peak, 1)
        self.assertEqual(cleared, [True])

    def test_model_count_phrase_limit_and_labels_are_validated(self):
        with self.assertRaises(ValueError):
            compare_phrases(["x"], "hi-en", [("A", "joy", "1")], classifier_factory=FakeClassifier)
        with self.assertRaises(ValueError):
            compare_phrases(["x"] * 21, "hi-en", [("A", "joy", "1"), ("B", "sadness", "2")], classifier_factory=FakeClassifier)
        with self.assertRaises(ValueError):
            compare_phrases(["x" * 20_001], "hi-en", [("A", "joy", "1"), ("B", "sadness", "2")], classifier_factory=FakeClassifier)

    def test_invalid_probabilities_and_nontrained_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            from hinglish_emotion.disagreement import probability_summary
            probability_summary({label: 0.0 for label in EMOTION_LABELS})
        too_large = distribution("joy")
        too_large["sadness"] = 0.1
        with self.assertRaises(ValueError):
            probability_summary(too_large)

        class DemoClassifier(FakeClassifier):
            def predict_text(self, text):
                result = FakeResult("joy")
                result.source = "lexical_demo"
                return result

        with self.assertRaisesRegex(ValueError, "trained-checkpoint"):
            compare_phrases(["x"], "hi-en", [("A", "joy", "1"), ("B", "sadness", "2")], classifier_factory=DemoClassifier)

    def test_truncation_counts_as_uncertainty_and_metadata_is_preserved(self):
        class TruncatedClassifier(FakeClassifier):
            def predict_text(self, text):
                result = FakeResult("joy")
                result.truncated = True
                return result

        rows = compare_phrases(["x"], "hi-en", [("A", "joy", "1"), ("B", "joy", "2")], classifier_factory=TruncatedClassifier)
        prediction = rows[0]["predictions"][0]
        self.assertTrue(rows[0]["uncertainty"])
        self.assertTrue(prediction["truncated"])
        self.assertTrue(prediction["experimental"])
        self.assertEqual(prediction["warnings"], ("test warning",))
        self.assertEqual(prediction["model_name"], "fake model")

    def test_queue_requires_human_label_deduplicates_and_csv_quotes(self):
        queue = add_reviewed_candidate([], {
            "text": 'She said, "khush hoon"', "emotion": "joy", "language_pair": "hi-en",
            "group_id": "g1", "review_notes": "comma, quote: \"ok\"", "source": "phrases.csv",
        })
        duplicate = add_reviewed_candidate(queue, {"text": '  SHE   SAID, "KHUSH HOON"  ', "emotion": "anger", "language_pair": "hi-en"})
        self.assertEqual(len(duplicate), 1)
        self.assertTrue(duplicate[0]["human_reviewed"])
        self.assertEqual(duplicate[0]["split"], "")
        self.assertIsNone(add_reviewed_candidate([], {"text": "unknown origin", "emotion": "joy", "language_pair": "hi-en", "synthetic": None})[0]["synthetic"])
        parsed = list(csv.DictReader(io.StringIO(queue_csv(duplicate))))
        self.assertEqual(parsed[0]["text"], 'She said, "khush hoon"')
        self.assertEqual(parsed[0]["review_notes"], 'comma, quote: "ok"')
        self.assertEqual(parsed[0]["split"], "")
        with self.assertRaises(ValueError):
            add_reviewed_candidate([], {"text": "x", "emotion": "positive", "language_pair": "hi-en"})


if __name__ == "__main__":
    unittest.main()
