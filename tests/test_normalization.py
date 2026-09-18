import unittest

from hinglish_emotion.normalization import HinglishNormalizer


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = HinglishNormalizer()

    def test_known_spelling_variants_share_canonical_form(self) -> None:
        outputs = {
            self.normalizer.normalize(value).normalized_text
            for value in ("acha", "achha", "accha", "acchaaa")
        }
        self.assertEqual(outputs, {"accha"})

    def test_elongation_is_preserved_as_metadata(self) -> None:
        result = self.normalizer.normalize("movie acchaaa thi")
        token = next(token for token in result.tokens if token.raw == "acchaaa")
        self.assertEqual(token.normalized, "accha")
        self.assertTrue(token.elongated)
        self.assertEqual(token.removed_characters, 2)

    def test_legitimate_double_is_not_destroyed(self) -> None:
        result = self.normalizer.normalize("goooood")
        self.assertEqual(result.normalized_text, "good")

    def test_negation_is_preserved(self) -> None:
        result = self.normalizer.normalize("movie achhiii nhi thi")
        self.assertEqual(result.normalized_text, "movie acchi nahi thi")
        self.assertTrue(result.has_negation)

    def test_unknown_token_is_not_aggressively_rewritten(self) -> None:
        result = self.normalizer.normalize("zabardastttt")
        self.assertEqual(result.normalized_text, "zabardastt")
        self.assertTrue(result.tokens[0].elongated)

    def test_numbers_are_preserved(self) -> None:
        result = self.normalizer.normalize("2 ghante late")
        self.assertEqual(result.normalized_text, "2 ghante late")


if __name__ == "__main__":
    unittest.main()
