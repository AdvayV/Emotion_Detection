import math
from unittest.mock import Mock
import pytest
from hinglish_emotion.emotions import EmotionDetector, EmotionResult
from hinglish_emotion.emotion_workbench import EmotionWorkbench
from hinglish_emotion.ollama_emotion_reviewer import OllamaEmotionReviewer
from hinglish_emotion.tasks import EMOTION_LABELS


def primary(emotion="joy", experimental=False):
    return EmotionResult(emotion, {x: .94 if x == emotion else .01 for x in EMOTION_LABELS}, experimental=experimental)


def test_base_called_once_and_keeps_probabilities_after_review():
    detector = Mock(); original = primary(); detector.predict.return_value = original
    reviewer = Mock(); reviewer.availability.return_value = (True, "ready")
    reviewer.review.return_value = {"emotion":"anger", "sarcasm":True, "evidence":["great"]}
    result = EmotionWorkbench(detector, reviewer).analyze("great service 😡", "hi-en")
    detector.predict.assert_called_once_with("great service 😡", "hi-en")
    assert result.cues["emoji_relation"] == "conflict"
    assert result.review["status"] == "applied" and result.final_emotion == "anger"
    assert result.primary is original and result.primary.confidence == .94
    assert result.to_dict()["primary"]["emotion"] == "joy"


@pytest.mark.parametrize("value", [math.nan, -1, 2])
def test_invalid_supervised_score_falls_back_to_explicit_rules(value):
    detector = Mock(); detector.predict.return_value = primary()
    sarcasm = Mock(); sarcasm.predict_text.return_value = value
    result = EmotionWorkbench(detector, sarcasm_classifier=sarcasm).analyze("great service")
    assert result.cues["sarcasm_source"] == "heuristic"
    assert "unavailable" in result.cues["sarcasm_message"]


def test_review_not_needed_for_confident_phrase_but_context_routes():
    detector = Mock(); detector.predict.return_value = primary()
    reviewer = Mock(); reviewer.availability.return_value = (False, "Ollama unavailable")
    workbench = EmotionWorkbench(detector, reviewer)
    assert workbench.analyze("great service").review["status"] == "not_needed"
    reviewer.availability.assert_not_called()
    result = workbench.analyze("great service", context="My delivery was late")
    assert result.review["status"] == "unavailable" and result.final_emotion == "joy"
    assert result.context == "My delivery was late"


def test_bad_review_does_not_overwrite_primary():
    detector = Mock(); detector.predict.return_value = primary(experimental=True)
    reviewer = Mock(); reviewer.availability.return_value = (True, "ready")
    reviewer.review.return_value = {"emotion":"anger", "sarcasm":"false", "evidence":[]}
    result = EmotionWorkbench(detector, reviewer).analyze("hello")
    assert result.review["status"] == "error" and result.final_emotion == "joy"


@pytest.mark.parametrize("response", [
    {"emotion":"positive", "sarcasm":False, "evidence":[]},
    {"emotion":"joy", "sarcasm":"false", "evidence":[]},
    {"emotion":"joy", "sarcasm":False, "evidence":["not in target"]},
    {"emotion":"joy", "sarcasm":False, "evidence":[], "extra":0}])
def test_reviewer_schema_rejects_invalid_labels_booleans_and_evidence(response):
    with pytest.raises(ValueError): OllamaEmotionReviewer.validate(response, "great service")


def test_normalization_negation_and_sarcasm_are_diagnostics():
    result = EmotionWorkbench(EmotionDetector()).analyze("acchaaa nahi service late 😂")
    assert result.normalization["elongated_token_count"] > 0
    assert result.normalization["has_negation"]
    assert result.cues["emoji_relation"] == "uncertain"
    assert result.cues["intensity"]["source"] == "heuristic"
    assert result.review["status"] == "disabled"
    assert result.final_emotion == result.primary.emotion
