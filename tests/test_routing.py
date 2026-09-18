import unittest

from hinglish_emotion.routing import ReviewRouter


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ReviewRouter()

    def test_clear_prediction_is_not_routed(self) -> None:
        decision = self.router.decide(confidence=0.91)
        self.assertFalse(decision.should_review)

    def test_low_confidence_is_routed(self) -> None:
        decision = self.router.decide(confidence=0.51)
        self.assertTrue(decision.should_review)
        self.assertIn("low_confidence", decision.reasons)

    def test_sarcasm_and_conflict_reasons_are_retained(self) -> None:
        decision = self.router.decide(
            confidence=0.80,
            sarcasm_probability=0.78,
            text_emoji_conflict=True,
        )
        self.assertEqual(decision.reasons, ("possible_sarcasm", "text_emoji_conflict"))


if __name__ == "__main__":
    unittest.main()

