"""Cross-module checks for research tools and backward-compatible inference."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.emotions import EmotionDetector,EmotionResult,LocalEmotionClassifier
from hinglish_emotion.emotion_workbench import EmotionWorkbench
from hinglish_emotion.conversation_lab import analyze_turns,parse_turns

ROOT=Path(__file__).resolve().parents[1]


def scores(label): return {name:(.94 if name==label else .01) for name in EMOTION_LABELS}


def test_conversation_uses_past_turns_and_speaker_specific_changes():
    calls=[]
    def analyze(text,pair,context):
        calls.append(context)
        label='sadness' if 'sad' in text else 'joy'
        return SimpleNamespace(primary=EmotionResult(label,scores(label)),final_emotion=label,review={'status':'disabled','suggestion':None})
    turns=parse_turns('A: happy\nB: happy too\nA: sad now')
    result=analyze_turns(turns,SimpleNamespace(analyze=analyze),history_limit=1)
    assert calls==['','A: happy','B: happy too']
    assert result[2]['previous_speaker_emotion']=='joy' and result[2]['speaker_emotion_changed']
    assert not result[0]['speaker_emotion_changed']


def test_context_reaches_only_the_explicit_context_interface():
    class Classifier:
        metadata={'validated_language_pairs':['hi-en']}
        def predict_text(self,text): return EmotionResult('joy',scores('joy'))
        def predict_with_context(self,text,context):
            assert context=='earlier message'
            return EmotionResult('sadness',scores('sadness'))
    result=EmotionWorkbench(EmotionDetector(Classifier())).analyze('current phrase','hi-en','earlier message')
    assert result.primary.emotion=='sadness'


def test_research_pages_and_missing_advanced_model_are_explicit():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30)
    app.session_state['workspace']='Research workbench';app.run()
    assert not app.exception
    for section in ['Confidence calibration','Human review','Conversation','GPU training']:
        app.radio(key='research_workflow').set_value(section).run()
        assert not app.exception,section
    assert any('no reviewed sarcasm' in item.value for item in app.warning)
    assert any(item.label=='Download advanced Colab notebook' for item in app.get('download_button'))
    app.radio(key='workspace').set_value('Emotion lab').run()
    app.selectbox(key='emotion_classifier_choice').set_value('XLM-R learned fusion').run()
    assert not app.exception
    assert next(b for b in app.button if b.label=='Detect emotion').disabled
    assert any('not installed' in x.value for x in app.warning)


def test_new_fusion_export_installs_and_predicts_locally(tmp_path):
    import torch
    from transformers import XLMRobertaConfig,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from hinglish_emotion.fusion import TextEmojiFusionModel,ARCHITECTURE,EMOJI_BUCKETS,MAX_EMOJI,CONTEXT_INPUT_POLICY
    from training.cloud_bundle import export_checkpoint
    from training.install_checkpoint import install
    cfg=XLMRobertaConfig(vocab_size=6,hidden_size=768,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=64,
        id2label=dict(enumerate(EMOTION_LABELS)),label2id={x:i for i,x in enumerate(EMOTION_LABELS)})
    cfg.fusion_architecture=ARCHITECTURE;cfg.emoji_hash_version='sha256_sequence_v1';cfg.emoji_buckets=EMOJI_BUCKETS;cfg.max_emoji_sequences=MAX_EMOJI
    metadata={'schema_version':1,'task':'emotion','architecture':ARCHITECTURE,'labels':list(EMOTION_LABELS),'base_model':'FacebookAI/xlm-roberta-base',
        'text_mode':'raw','preparation_version':1,'max_length':32,'trained_language_pairs':['hi-en'],'validated_language_pairs':[],
        'training_status':'complete','training_data_kind':'synthetic_only','evaluation_scope':'synthetic_only','dataset_sha256':'fixture','split_sha256':'fixture',
        'trained_with_context':False,'trained_sarcasm_head':False,'context_input_policy':CONTEXT_INPUT_POLICY,'sarcasm_supervision':{'human_reviewed':False}}
    cfg.emotion_metadata=metadata
    model=TextEmojiFusionModel(cfg).eval();folder=tmp_path/'fixture';model.save_pretrained(folder)
    (folder/'emotion_metadata.json').write_text(json.dumps(metadata))
    backend=Tokenizer(WordLevel({'<unk>':0,'<s>':1,'</s>':2,'<pad>':3,'hello':4,'world':5},unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    backend.post_processor=TemplateProcessing(single='<s> $A </s>',pair='<s> $A </s> </s> $B </s>',special_tokens=[('<s>',1),('</s>',2)])
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',bos_token='<s>',eos_token='</s>',pad_token='<pad>');tokenizer.save_pretrained(folder)
    archive=export_checkpoint(folder,tmp_path/'fusion.zip');installed=install(archive,tmp_path/'models')
    assert installed.name=='xlmr_fusion_emotion'
    classifier=LocalEmotionClassifier(installed,device='cpu')
    result=classifier.predict_text('hello world 😊')
    assert result.source=='trained_checkpoint' and set(result.probabilities)==set(EMOTION_LABELS)
    assert abs(sum(result.probabilities.values())-1)<1e-5
    contextual=classifier.predict_with_context('hello','earlier')
    assert any('ignored' in w for w in contextual.warnings)


def test_calibrated_scores_route_review_without_replacing_base(monkeypatch,tmp_path):
    from unittest.mock import Mock
    from hinglish_emotion.calibration import fit_temperature
    raw=scores('joy')
    profile=fit_temperature([raw],['sadness'],model_identity={'model_fingerprint':'version'},split='validation',scope='hi-en',allow_unreviewed=True)
    class Classifier:
        model_path=tmp_path
        metadata={'validated_language_pairs':['hi-en']}
        def predict_text(self,text):return EmotionResult('joy',raw,source='trained_checkpoint')
    monkeypatch.setattr('hinglish_emotion.checkpoints.checkpoint_version',lambda path:'version')
    reviewer=Mock();reviewer.availability.return_value=(False,'off')
    result=EmotionWorkbench(EmotionDetector(Classifier()),reviewer,calibration_profile=profile).analyze('phrase','hi-en')
    assert result.primary.probabilities==raw and result.primary.emotion=='joy'
    assert result.review['status']=='unavailable'
    assert result.cues['confidence_scaling']['used_for_review_routing']
    assert result.cues['confidence_scaling']['probabilities']['joy']<raw['joy']
