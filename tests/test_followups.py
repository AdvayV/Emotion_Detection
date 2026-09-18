import unittest

from hinglish_emotion.context import ConversationMemory, TurnRecord
from hinglish_emotion.evidence import check_evidence
from hinglish_emotion.intensity import IntensityEstimator


class FollowUpTests(unittest.TestCase):
    def test_speaker_aware_shift(self):
        memory = ConversationMemory()
        memory.add(TurnRecord("c1", 1, "a", "fine", "positive", 0.8))
        shift = memory.add(TurnRecord("c1", 2, "a", "bad", "negative", 0.7))
        self.assertEqual(shift.direction, "positive_to_negative")

    def test_intensity_keeps_elongation_signal(self):
        estimate = IntensityEstimator().estimate("acchaaa!!!")
        self.assertIn("elongation", estimate.cues)
        self.assertGreater(estimate.arousal, 0)

    def test_evidence_check(self):
        check = check_evidence("movie accha", ("accha",), lambda text: "positive" if "accha" in text else "neutral")
        self.assertTrue(check.evidence_changed_prediction)


if __name__ == "__main__":
    unittest.main()
