"""Regression tests for fair, aligned seven-emotion model evaluation."""
import gc
from pathlib import Path
import weakref

import pytest

from hinglish_emotion.emotions import EmotionResult
from hinglish_emotion.evaluation_guard import write_guard
from hinglish_emotion.model_comparison import compare_models, comparison_rows
from hinglish_emotion.tasks import EMOTION_LABELS


@pytest.fixture
def records():
    return [
        {"text":"training", "emotion":"joy", "language_pair":"hi-en", "split":"train"},
        {"text":"alpha", "emotion":"joy", "language_pair":"hi-en", "split":"test", "row_id":"same"},
        {"text":"beta", "emotion":"sadness", "language_pair":"hi-en", "split":"test", "row_id":"same"},
        {"text":"gamma", "emotion":"fear", "language_pair":"hi-en", "split":"test"},
        {"text":"blank", "emotion":"", "language_pair":"hi-en", "split":"test"},
        {"text":"french", "emotion":"joy", "language_pair":"en-fr", "split":"test"},
    ]


@pytest.fixture
def factory(tmp_path):
    live=weakref.WeakSet()
    loaded=[]
    answers={"one":{"alpha":"joy","beta":"anger","gamma":"fear"},
             "two":{"alpha":"sadness","beta":"sadness","gamma":"anger"},
             "three":{"alpha":"joy","beta":"sadness","gamma":"anger"}}
    class Classifier:
        def __init__(self,path):
            gc.collect()
            assert not live, "A previous classifier is still retained"
            self.model_path=str(path)
            self.metadata={"trained_language_pairs":["hi-en","fr-en"], "validated_language_pairs":[],
                           "training_data_kind":"synthetic_only", "evaluation_scope":"synthetic_only",
                           "dataset_sha256":"data", "split_sha256":"split"}
            self.name=Path(path).name; loaded.append(self.name);live.add(self)
        def predict_batch(self,texts,batch_size):
            return [EmotionResult(answers[self.name].get(text,"neutral"),
                    {label:(.94 if label==answers[self.name].get(text,"neutral") else .01) for label in EMOTION_LABELS},
                    source="trained_checkpoint",model_name=self.name) for text in texts]
    def create(name): return Classifier(name)
    create.loaded=loaded
    create.live=live
    return create


def checkpoints(tmp_path):
    return [(name,tmp_path/folder,"v1") for name,folder in [("XLM-R","one"),("mBERT","two"),("MuRIL","three")]]


def test_matched_split_language_alignment_and_known_scores(records,factory,tmp_path):
    progress=[]
    result=compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=factory,progress=lambda *args:progress.append(args))
    assert result["selected_rows"]==4 and result["scored_rows"]==3
    assert result["other_language_rows_skipped"]==1
    assert [m["metrics"]["accuracy"] for m in result["models"]]==[2/3,1/3,2/3]
    assert result["models"][0]["metrics"]["macro_f1"]==pytest.approx(2/7)
    assert [r["text"] for r in result["predictions"]]==["alpha","beta","gamma","blank"]
    assert result["predictions"][0]["row_id"]==result["predictions"][1]["row_id"]
    assert result["predictions"][0]["XLM-R correct"] is True
    assert result["predictions"][1]["XLM-R correct"] is False
    assert result["predictions"][-1]["XLM-R correct"] is None
    assert progress[-1][:2]==(12,12)
    assert not factory.live
    pair=result["paired_comparisons"][0]
    assert (pair["both_correct"],pair["left_only_correct"],pair["right_only_correct"],pair["both_wrong"])==(0,2,1,0)


def test_overlap_excludes_same_rows_for_every_model(records,factory,tmp_path):
    path=tmp_path/'one';path.mkdir()
    write_guard(path/'evaluation_guard.json',[{"text":"alpha","split":"train"}],{"dataset_sha256":"data","split_sha256":"split"})
    result=compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=factory)
    assert result["scored_rows"]==2 and result["excluded_overlap_rows"]==1
    assert all(m["metrics"]["rows"]==2 for m in result["models"])
    assert all(m["metrics"]["accuracy"]==.5 for m in result["models"])
    assert result["predictions"][0]["included_in_scoring"] is False
    assert result["predictions"][0]["training_overlap_any_model"] is True
    unguarded=compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=factory,exclude_overlap=False)
    assert unguarded["scored_rows"]==3 and unguarded["excluded_overlap_rows"]==0


def test_unlabelled_data_has_predictions_but_no_accuracy(records,factory,tmp_path):
    rows=[{**r,"emotion":""} for r in records]
    result=compare_models(rows,"hi-en",checkpoints(tmp_path),classifier_factory=factory)
    assert result["scored_rows"]==0
    assert all(m["metrics"] is None for m in result["models"])
    assert all(pair["scored_rows"]==0 for pair in result["paired_comparisons"])
    assert len(result["predictions"])==4


def test_all_overlap_never_becomes_zero_accuracy(records,factory,tmp_path):
    path=tmp_path/'one';path.mkdir()
    write_guard(path/'evaluation_guard.json',[{"text":r["text"],"split":"train"} for r in records],{"dataset_sha256":"data","split_sha256":"split"})
    result=compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=factory)
    assert result["scored_rows"]==0
    assert all(m["metrics"] is None for m in result["models"])


def test_french_excludes_muril_and_retains_only_french(records,factory,tmp_path):
    with pytest.raises(ValueError,match="Hindi-English"):
        compare_models(records,"fr-en",checkpoints(tmp_path),classifier_factory=factory)
    assert not factory.loaded
    result=compare_models(records,"fr-en",checkpoints(tmp_path)[:2],classifier_factory=factory)
    assert result["selected_rows"]==1 and result["predictions"][0]["text"]=="french"


def test_selected_model_failure_is_not_a_rule_fallback(records,factory,tmp_path):
    def broken(path):
        if Path(path).name=='two': raise OSError("missing weights")
        return factory(path)
    with pytest.raises(RuntimeError,match="mBERT comparison failed: missing weights"):
        compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=broken)
    assert not factory.live


def test_rejects_duplicate_checkpoints_and_invalid_inputs(records,factory,tmp_path):
    with pytest.raises(ValueError,match="same checkpoint"):
        compare_models(records,"hi-en",[("A",tmp_path/'one','1'),("B",tmp_path/'one','1')],classifier_factory=factory)
    with pytest.raises(ValueError,match="positive batch"):
        compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=factory,batch_size=0)
    with pytest.raises(ValueError,match="No phrases"):
        comparison_rows([{"text":"hello","emotion":"joy","language_pair":"fr-en"}],"hi-en")


def test_rejects_demo_predictions_and_bad_probability_vectors(records,tmp_path):
    class Fake:
        metadata={"trained_language_pairs":["hi-en"],"validated_language_pairs":[]}
        def predict_batch(self,texts,batch_size):
            return [EmotionResult("joy",{label:.1 for label in EMOTION_LABELS},source=self.source) for text in texts]
    Fake.source='lexical_demo'
    with pytest.raises(RuntimeError,match="fallback"):
        compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=lambda path:Fake())
    Fake.source='trained_checkpoint'
    with pytest.raises(RuntimeError,match="sum to one"):
        compare_models(records,"hi-en",checkpoints(tmp_path),classifier_factory=lambda path:Fake())
