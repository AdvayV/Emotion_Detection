import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from hinglish_emotion.emotions import EmotionDetector, EmotionResult
from hinglish_emotion.emotion_data import evaluate_emotions, read_emotion_csv, validate_emotion_rows
from hinglish_emotion.normalization import HinglishNormalizer
from hinglish_emotion.tasks import EMOTION_LABELS


ROOT = Path(__file__).resolve().parents[1]


class EmotionTests(unittest.TestCase):
    def test_all_seven_emotions_in_both_language_mixes(self):
        detector = EmotionDetector()
        phrases = {
            "joy": ("Main khush hoon today", "Je suis heureux today"),
            "sadness": ("Main udaas hoon today", "Je suis triste today"),
            "anger": ("Mujhe gussa aa raha hai today", "Je suis en colère today"),
            "fear": ("Mujhe darr lag raha hai today", "J'ai peur today"),
            "surprise": ("Main hairan hoon today", "Je suis étonnée today"),
            "disgust": ("Mujhe ghin aa rahi hai today", "Je suis dégoûtée today"),
            "neutral": ("Meeting kal hai at three", "La réunion is at three"),
        }
        for expected, texts in phrases.items():
            for pair, text in zip(("hi-en", "fr-en"), texts):
                with self.subTest(expected=expected, pair=pair):
                    result = detector.predict(text, pair)
                    self.assertEqual(result.emotion, expected)
                    self.assertEqual(set(result.probabilities), set(EMOTION_LABELS))
                    self.assertAlmostEqual(sum(result.probabilities.values()), 1)

    def test_french_unicode_and_apostrophes_are_preserved(self):
        text = "Je suis très dégoûtée et j'ai peur"
        result = HinglishNormalizer().normalize(text)
        self.assertEqual(result.normalized_text, text.lower())
        self.assertIn("dégoûtée", [token.raw for token in result.tokens])
        self.assertIn("j'ai", [token.raw for token in result.tokens])

    def test_negation_does_not_invent_the_opposite_emotion(self):
        detector = EmotionDetector()
        for text, pair in (("Mujhe dar nahi lag raha", "hi-en"), ("Je n'ai pas peur today", "fr-en"), ("Je n'ai jamais peur", "fr-en"), ("I am not happy", "hi-en"), ("I don't feel happy", "fr-en")):
            result = detector.predict(text, pair)
            self.assertEqual(result.emotion, "neutral")
            self.assertTrue(result.needs_context)

    def test_demo_is_balanced_and_evaluation_is_grouped_by_language(self):
        rows = read_emotion_csv(ROOT / "data" / "demo" / "emotions" / "mixed_emotions_demo.csv")
        report = evaluate_emotions(rows, EmotionDetector())
        self.assertTrue(report["illustrative"])
        self.assertEqual(report["overall"]["rows"], 84)
        self.assertEqual(set(report["by_language"]), {"hi-en", "fr-en"})
        for scores in report["by_language"].values():
            self.assertEqual(scores["rows"], 42)
            self.assertEqual(scores["missing_emotions"], [])
            self.assertEqual(len(scores["confusion_matrix"]), 7)
            for metrics in scores["per_class"].values():
                self.assertEqual(metrics["support"], 6)

    def test_dataset_accepts_happy_alias_and_rejects_bad_language(self):
        row = {"text": "I am happy yaar", "emotion": "happy", "language_pair": "hi-en"}
        self.assertEqual(validate_emotion_rows([row])[0]["emotion"], "joy")
        with self.assertRaisesRegex(ValueError, "language_pair"):
            validate_emotion_rows([{**row, "language_pair": "fr"}])
        with self.assertRaisesRegex(ValueError, "Missing emotion columns"):
            validate_emotion_rows([{"text": "hello", "label": "positive"}])

    def test_evaluation_scores_only_declared_test_rows(self):
        rows = [
            {"text": "I am angry yaar", "emotion": "anger", "language_pair": "hi-en", "split": "train"},
            {"text": "Je suis triste today", "emotion": "sadness", "language_pair": "fr-en", "split": "test"},
        ]
        report = evaluate_emotions(rows, EmotionDetector())
        self.assertEqual(report["overall"]["rows"], 1)
        self.assertEqual(report["scope"], "declared test split")
        self.assertNotIn("hi-en", report["by_language"])
        self.assertIn("joy", report["overall"]["missing_emotions"])


@unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("pandas"), "Training dependencies optional")
class EmotionSplitTests(unittest.TestCase):
    def make_frame(self):
        import pandas as pd
        rows = []
        for label in EMOTION_LABELS:
            for index in range(20):
                for pair in ("hi-en", "fr-en"):
                    text = f"{label} independent phrase {index} {pair}"
                    rows.append({"text": text, "normalized_text": text, "label": label, "language_pair": pair, "group_id": f"{label}_{index}"})
        return pd.DataFrame(rows)

    def test_translations_stay_together_and_all_labels_are_covered(self):
        from training.splits import split_data
        parts = split_data(self.make_frame(), 42)
        groups = [set(part["group_id"]) for part in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for part in parts:
            for pair in ("hi-en", "fr-en"):
                self.assertEqual(set(part.loc[part["language_pair"] == pair, "label"]), set(EMOTION_LABELS))
        again = split_data(self.make_frame(), 42)
        self.assertEqual([part["text"].tolist() for part in parts], [part["text"].tolist() for part in again])

    def test_declared_split_rejects_related_phrases_in_different_partitions(self):
        from training.splits import split_data
        frame = self.make_frame()
        frame["split"] = ["train", "test"] * (len(frame) // 2)
        with self.assertRaisesRegex(ValueError, "cross declared"):
            split_data(frame, 42)
