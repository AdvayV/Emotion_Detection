import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from hinglish_emotion.experiments import run_experiments


class Predictor:
    def __call__(self, text, language_pair="hi-en", context=""):
        label = "joy" if "happy" in text.lower() else "sadness" if "sad" in text.lower() else "neutral"
        return {"emotion": label, "probabilities": {"joy": .8 if label == "joy" else .05,
                "sadness": .8 if label == "sadness" else .05, "anger": .02, "fear": .02,
                "surprise": .02, "disgust": .02, "neutral": .07}, "source": "fake-local"}


class Reviewer:
    def __init__(self, fail=False): self.calls, self.fail = [], fail
    def review(self, text, normalized, primary, **kwargs):
        self.calls.append(text)
        if self.fail: raise RuntimeError("offline")
        return {"emotion": "anger"}


class EmotionReviewer(Reviewer):
    def __init__(self, suggestion):
        super().__init__()
        self.suggestion = suggestion
        self.primary_types = []
    def availability(self): return True, "ready"
    def review(self, text, normalized, primary, **kwargs):
        from hinglish_emotion.emotions import EmotionResult
        self.primary_types.append(type(primary))
        assert isinstance(primary, EmotionResult)
        return {"emotion": self.suggestion, "sarcasm": False, "evidence": []}
    @staticmethod
    def validate(result, text):
        assert result["emotion"] in {"joy", "anger"}
        return result


class ExperimentTests(unittest.TestCase):
    def test_variant_accuracy_requires_explicit_label_or_invariance(self):
        report = run_experiments([{"text": "happy", "label": "joy", "split": "test"}], Predictor(), ["baseline", "normalized", "emoji_removed"])
        stats = report["summaries"]
        self.assertEqual(stats["baseline"]["accuracy"], 1.0)
        self.assertIsNone(stats["normalized"]["accuracy"])
        self.assertEqual(stats["normalized"]["accuracy_rows"], 0)
        invariant = run_experiments([{"text": "happy", "label": "joy", "label_invariant": True, "split": "test"}], Predictor(), ["normalized"])
        self.assertEqual(invariant["summaries"]["normalized"]["accuracy"], 1.0)

    def test_review_suggestion_never_overwrites_primary_and_attempts_capped(self):
        reviewer = Reviewer()
        rows = [{"text": "happy", "split": "test"} for _ in range(8)]
        result = run_experiments(rows, Predictor(), ["baseline"], review_mode="all", reviewer=reviewer)
        self.assertEqual(len(reviewer.calls), 5)
        self.assertEqual(result["review_summary"]["attempted"], 5)
        self.assertEqual(result["review_summary"]["coverage"], .625)
        self.assertEqual(result["rows"][0]["base_label"], "joy")
        self.assertEqual(result["rows"][0]["review_suggestion"], "anger")
        self.assertEqual(result["rows"][0]["scores"]["joy"], .8)

    def test_review_errors_and_language_errors_are_rows(self):
        result = run_experiments([{"text": "sad", "language_pair": "xx", "split": "test"}, {"text": "sad", "split": "test"}], Predictor(), ["baseline"], review_mode="all", reviewer=Reviewer(True))
        self.assertEqual(result["rows"][0]["review_status"], "unsupported_language")
        self.assertEqual(result["rows"][1]["review_status"], "error")
        self.assertEqual(result["review_summary"]["errors"], 1)

    def test_test_filter_and_training_overlap_flag(self):
        result = run_experiments([{"text": "happy", "split": "train"}, {"text": "happy", "split": "test"}], Predictor(), ["baseline"])
        self.assertEqual(len(result["rows"]), 1)
        self.assertTrue(result["rows"][0]["heldout_overlap"])
        self.assertEqual(result["heldout_overlap_rows"], 1)

    def test_emoji_removal_preserves_ordinary_punctuation(self):
        result = run_experiments([{"text": "happy! 😊", "split": "test"}], Predictor(), ["emoji_removed"])
        self.assertEqual(result["rows"][0]["input_text"], "happy!")

    def test_predictor_failure_is_recorded_and_scores_remain_json_safe(self):
        def broken(text): raise RuntimeError("checkpoint missing")
        report = run_experiments([{"row_id": "phrase-7", "text": "happy", "label": "joy", "split": "test"}], broken, ["baseline"])
        item = report["rows"][0]
        self.assertEqual(item["row_id"], "phrase-7")
        self.assertIn("checkpoint missing", item["prediction_error"])
        self.assertEqual(item["scores"], {})
        self.assertIsNone(report["summaries"]["baseline"]["accuracy"])

    def test_review_error_does_not_remove_base_accuracy_and_reports_latency(self):
        report = run_experiments([{"text": "happy", "label": "joy", "split": "test"}], Predictor(), ["baseline"], review_mode="all", reviewer=Reviewer(True))
        self.assertEqual(report["summaries"]["baseline"]["accuracy"], 1.0)
        self.assertEqual(report["review_summary"]["accuracy_before"], None)
        self.assertEqual(report["review_summary"]["accuracy_after"], None)
        self.assertGreaterEqual(report["rows"][0]["review_latency_ms"], 0)

    def test_reviewer_accuracy_uses_same_limited_successful_labelled_subset(self):
        rows = [{"row_id": f"r{i}", "text": "happy", "label": "joy", "split": "test"} for i in range(7)]
        reviewer = EmotionReviewer("anger")
        report = run_experiments(rows, Predictor(), ["baseline"], review_mode="all", reviewer=reviewer)
        self.assertEqual(len(reviewer.primary_types), 5)
        self.assertTrue(all(kind.__name__ == "EmotionResult" for kind in reviewer.primary_types))
        self.assertEqual(report["review_summary"]["labelled_rows_compared"], 5)
        self.assertEqual(report["review_summary"]["accuracy_before"], 1.0)
        self.assertEqual(report["review_summary"]["accuracy_after"], 0.0)
        self.assertEqual([row["row_id"] for row in report["rows"][:1]], ["r0"])

    def test_invariance_without_a_declared_label_does_not_create_accuracy(self):
        report = run_experiments([{"text": "happy", "label_invariant": True, "split": "test"}], Predictor(), ["normalized"])
        self.assertIsNone(report["summaries"]["normalized"]["accuracy"])
        self.assertEqual(report["rows"][0]["label_source"], "invariance_declared_without_label")


if __name__ == "__main__": unittest.main()
