import sys
import pytest
from hinglish_emotion import cli
from hinglish_emotion.emotion_data import evaluate_emotions
from hinglish_emotion.emotions import EmotionDetector, EmotionResult
from hinglish_emotion.tasks import EMOTION_LABELS

class FakeClassifier:
    def __init__(self, metadata):
        self.metadata, self.calls = metadata, []
    def result(self):
        return EmotionResult('joy', {label: float(label == 'joy') for label in EMOTION_LABELS})
    def predict_text(self, text):
        self.calls.append('text'); return self.result()
    def predict_with_context(self, text, context):
        self.calls.append('context'); return self.result()
    def predict_batch(self, texts, batch_size):
        self.calls.append('batch'); return [self.result() for _ in texts]

def classifier():
    return FakeClassifier({'trained_language_pairs': ['hi-en'], 'validated_language_pairs': ['hi-en']})

@pytest.mark.parametrize('method,args', [('predict', ('hello', 'fr-en')), ('predict_context', ('hello', 'fr-en', 'earlier')), ('predict_many', (['one', 'two'], ['hi-en', 'fr-en']))])
def test_unsupported_rejected_before_classifier(method, args):
    model = classifier()
    with pytest.raises(ValueError, match='trained language coverage'):
        getattr(EmotionDetector(model), method)(*args)
    assert model.calls == []

def test_validated_and_unverified_trained_pairs():
    model = FakeClassifier({'trained_language_pairs': ['hi-en', 'fr-en'], 'validated_language_pairs': ['hi-en']})
    results = EmotionDetector(model).predict_many(['one', 'two'], ['hi-en', 'fr-en'])
    assert not results[0].experimental
    assert results[1].experimental and 'no verified' in results[1].warnings[0]

@pytest.mark.parametrize('metadata', [{}, {'trained_language_pairs': []}, {'validated_language_pairs': ['hi-en']}])
def test_unknown_training_scope_warns(metadata):
    result = EmotionDetector(FakeClassifier(metadata)).predict('hello', 'hi-en')
    assert result.experimental and any('coverage is unknown' in w for w in result.warnings)

def test_demo_french_is_still_available_and_warned():
    result = EmotionDetector().predict('je suis content', 'fr-en')
    assert result.experimental and any('French' in w for w in result.warnings)

def test_evaluation_preflights_all_pairs():
    model = classifier()
    rows = [{'text': 'one', 'emotion': 'joy', 'language_pair': 'hi-en'}, {'text': 'two', 'emotion': 'joy', 'language_pair': 'fr-en'}]
    with pytest.raises(ValueError, match='trained language coverage'):
        evaluate_emotions(rows, EmotionDetector(model))
    assert model.calls == []

def test_cli_single_and_evaluate_preflight(monkeypatch, tmp_path):
    model = classifier()
    monkeypatch.setattr(cli, 'LocalEmotionClassifier', lambda path: model)
    monkeypatch.setattr(sys, 'argv', ['codemood', 'emotions', 'bonjour', '--language-pair', 'fr-en', '--model-path', 'mock'])
    with pytest.raises(ValueError, match='trained language coverage'): cli.main()
    path = tmp_path / 'rows.csv'
    path.write_text('text,emotion,language_pair\none,joy,hi-en\ntwo,joy,fr-en\n', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['codemood', 'evaluate-emotions', str(path), '--model-path', 'mock'])
    with pytest.raises(ValueError, match='trained language coverage'): cli.main()
    assert model.calls == []

def test_predict_file_requires_explicit_fallback(monkeypatch, tmp_path, capsys):
    model = classifier()
    monkeypatch.setattr(cli, 'LocalEmotionClassifier', lambda path: model)
    path = tmp_path / 'rows.csv'
    path.write_text('text,language_pair\nbonjour,fr-en\n', encoding='utf-8')
    args = ['codemood', 'predict-file', str(path), '--model-path', 'mock']
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(ValueError, match='not trained'):
        cli.main()
    monkeypatch.setattr(sys, 'argv', args + ['--allow-fallback'])
    cli.main()
    assert model.calls == []
    assert capsys.readouterr().out
