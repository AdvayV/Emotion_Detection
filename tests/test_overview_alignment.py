"""Overview reflects discovered emotion checkpoints, not legacy sentiment counts."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from dashboard.readiness import overview_readiness
from hinglish_emotion.tasks import EMOTION_LABELS


def checkpoint(root,name,base,pairs):
    p=root/'models'/name;p.mkdir(parents=True)
    m={'schema_version':1,'task':'emotion','labels':list(EMOTION_LABELS),'text_mode':'raw','preparation_version':1,'max_length':128,'base_model':base,'trained_language_pairs':pairs,'validated_language_pairs':[],'training_status':'complete','evaluation_scope':'synthetic_only'}
    (p/'config.json').write_text(json.dumps({'id2label':{str(i):l for i,l in enumerate(EMOTION_LABELS)},'emotion_metadata':m}))
    (p/'model.safetensors').write_bytes(b'structural-fixture-not-loaded')
    (p/'tokenizer.json').write_text('{}')
    return p


def render(root):
    code=f"from pathlib import Path\nimport dashboard.app as app\napp.PROJECT_ROOT=Path({str(root)!r})\napp.render_overview()"
    return AppTest.from_string(code,default_timeout=30).run()


def test_readiness_counts_families_not_artifacts_and_preserves_language_scope(tmp_path):
    checkpoint(tmp_path,'xlmr_multilingual_emotion','FacebookAI/xlm-roberta-base',['hi-en','fr-en'])
    checkpoint(tmp_path,'xlmr_emotion','FacebookAI/xlm-roberta-base',['hi-en'])
    checkpoint(tmp_path,'mbert_multilingual_emotion','google-bert/bert-base-multilingual-cased',['hi-en','fr-en'])
    checkpoint(tmp_path,'muril_hi_en_emotion','google/muril-base-cased',['hi-en'])
    status=overview_readiness(tmp_path)
    assert status['installed_families']==3 and len(status['models'])==3
    muril=next(row for row in status['models'] if row['Encoder']=='MuRIL')
    assert muril['English–French']=='Not supported'
    app=render(tmp_path)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Installed emotion encoders').value=='3 / 3'
    assert not any(m.label=='Trained sentiment encoders' for m in app.metric)


def test_missing_or_malformed_checkpoints_do_not_look_installed(tmp_path):
    p=checkpoint(tmp_path,'muril_hi_en_emotion','google/muril-base-cased',['hi-en'])
    (p/'config.json').write_text('{"id2label":{"0":"positive"}}')
    app=render(tmp_path)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Installed emotion encoders').value=='0 / 3'


def test_untrained_extensions_are_separate_from_available_training_code(tmp_path):
    t=tmp_path/'training';t.mkdir()
    for name in ['train_advanced.py','train_sarcasm.py']:(t/name).write_text('# fixture')
    status=overview_readiness(tmp_path)
    for capability in ['Learned emoji fusion','Learned conversation context','Supervised sarcasm']:
        row=next(r for r in status['capabilities'] if r['Capability']==capability)
        assert 'Training code available; no trained checkpoint' in row['Available today']
    app=render(tmp_path)
    assert not app.exception
    assert len(app.get('graphviz_chart'))==3
    app.button(key='overview_to_comparison').click().run()
    assert app.session_state['workspace']=='Model comparison'
