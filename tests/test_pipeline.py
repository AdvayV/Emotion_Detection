import unittest

from hinglish_emotion.pipeline import EmotionPipeline


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = EmotionPipeline()

    def test_positive_message(self) -> None:
        result = self.pipeline.predict("movie acchaaa thi 😊")
        self.assertEqual(result.normalized.normalized_text, "movie accha thi 😊")
        self.assertEqual(result.primary.label, "positive")

    def test_negation_flips_positive_word(self) -> None:
        result = self.pipeline.predict("service acchi nahi thi 😒")
        self.assertEqual(result.primary.label, "negative")

    def test_ambiguous_emoji_can_trigger_review(self) -> None:
        result = self.pipeline.predict("finally job mil gayi 😭")
        self.assertTrue(result.review_decision.should_review)

    def test_sarcastic_pattern_is_routed(self) -> None:
        result = self.pipeline.predict("wah kya service hai 2 ghante late 😂")
        self.assertGreaterEqual(result.primary.sarcasm_probability, 0.60)
        self.assertIn("possible_sarcasm", result.review_decision.reasons)


if __name__ == "__main__":
    unittest.main()
