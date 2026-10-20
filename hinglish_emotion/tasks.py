"""Separate label spaces for sentiment and phrase-level emotion detection."""

SENTIMENT_LABELS = ("positive", "neutral", "negative")
EMOTION_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")
EMOTION_NAMES = {
    "joy": "Happy / joy", "sadness": "Sadness", "anger": "Anger",
    "fear": "Fear", "surprise": "Surprise", "disgust": "Disgust", "neutral": "Neutral",
}
LANGUAGE_PAIRS = {"hi-en": "Hindi–English", "fr-en": "English–French"}
EMOTION_ALIASES = {
    "happy": "joy", "happiness": "joy", "sad": "sadness",
    "joie": "joy", "tristesse": "sadness", "colère": "anger",
    "peur": "fear", "dégoût": "disgust", "neutre": "neutral",
}


def canonical_emotion(value: str) -> str:
    value = value.strip().lower()
    return EMOTION_ALIASES.get(value, value)
