"""Regression checks for real dashboard actions and rerun-safe downloads."""
import csv
import io
import zipfile
from pathlib import Path
from streamlit.testing.v1 import AppTest
from training.review_workflow import create_review_sheets, export_review_zip

ROOT=Path(__file__).resolve().parents[1]


def review_app(uploads):
    code = """
import io
import streamlit as st
from pathlib import Path
from unittest.mock import patch
from dashboard.review_panel import render_review_panel

def uploaded(label, *args, key=None, **kwargs):
    payload=st.session_state['uploads'].get(key)
    if payload is None: return None
    result=io.BytesIO(payload)
    result.name=key+'.csv'
    return result

with patch('dashboard.review_panel.st.file_uploader', side_effect=uploaded):
    render_review_panel(Path('.'))
"""
    app=AppTest.from_string(code,default_timeout=30)
    app.session_state['uploads']=uploads
    return app.run()


def test_blind_sheet_download_survives_rerun_and_expires_on_source_change():
    payload=b'row_id,text,language_pair\nnew-1,New phrase for review,hi-en\n'
    app=review_app({'review_workflow_queue':payload})
    next(b for b in app.button if b.label=='Create blind review sheets').click().run()
    assert not app.exception
    assert any(b.label=='Download blind sheets' for b in app.get('download_button'))
    app.run()
    assert any(b.label=='Download blind sheets' for b in app.get('download_button'))
    app.session_state['uploads']={'review_workflow_queue':payload.replace(b'New phrase',b'Changed phrase')}
    app.run()
    assert not any(b.label=='Download blind sheets' for b in app.get('download_button'))


def test_base_paper_download_uses_organised_review_folder():
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30)
    app.session_state['workspace']='Emotion lab'
    app.run()
    assert not app.exception
    assert any(b.label=='Download base paper' for b in app.get('download_button'))


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def test_versioned_dataset_build_is_independent_of_queue_and_survives_rerun(tmp_path):
    queue=tmp_path/'queue.csv'
    write_csv(queue,[{'row_id':'new-correction','group_id':'new-group','text':'New human phrase for correction','language_pair':'hi-en','dataset_role':'train','source':'human'}])
    create_review_sheets(queue,tmp_path/'sheets')
    for filename,reviewer in [('reviewer_a.csv','Alice'),('reviewer_b.csv','Bob')]:
        path=tmp_path/'sheets'/filename
        with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        for row in rows:row.update(reviewer_id=reviewer,emotion='joy')
        write_csv(path,rows)
    data,_=export_review_zip(queue,tmp_path/'sheets/reviewer_a.csv',tmp_path/'sheets/reviewer_b.csv')
    base=tmp_path/'base.csv'
    write_csv(base,[{'row_id':str(i),'group_id':'base-'+str(i),'text':text,'emotion':'joy','language_pair':'hi-en','source':'human','split':split} for i,(text,split) in enumerate([('Base train phrase','train'),('Base validation phrase','validation'),('Base test phrase','test')])])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        uploads={'review_workflow_base':base.read_bytes(),'review_workflow_corr':z.read('reviewed_corrections.csv'),'review_workflow_audit':z.read('audit.json')}
    app=review_app(uploads)
    assert not app.exception
    build=next(b for b in app.button if b.label=='Build versioned training dataset')
    assert not build.disabled
    build.click().run()
    assert not app.exception
    assert any(b.label=='Download versioned GPU dataset' for b in app.get('download_button'))
    app.run()
    assert any(b.label=='Download versioned GPU dataset' for b in app.get('download_button'))
    app.session_state['uploads']={**uploads,'review_workflow_corr':uploads['review_workflow_corr']+b'\n'}
    app.run()
    assert not any(b.label=='Download versioned GPU dataset' for b in app.get('download_button'))


def test_calibration_uses_only_the_selected_language():
    code="""
import io
import streamlit as st
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from dashboard.calibration_panel import render_calibration_panel
from hinglish_emotion.tasks import EMOTION_LABELS
class Classifier:
    metadata={'text_mode':'raw','preparation_version':1}
    def __init__(self,*args,**kwargs):pass
    def predict_batch(self,texts,**kwargs):
        st.session_state['calibration_seen_texts']=texts
        return [SimpleNamespace(probabilities={k:(.7 if k=='joy' else .05) for k in EMOTION_LABELS}) for text in texts]
with ExitStack() as stack:
    stack.enter_context(patch('hinglish_emotion.checkpoints.emotion_checkpoint',return_value=Path('fake-checkpoint')))
    stack.enter_context(patch('hinglish_emotion.checkpoints.checkpoint_version',return_value='fake-version'))
    stack.enter_context(patch('hinglish_emotion.calibration.frozen_phrase_hashes',return_value=set()))
    stack.enter_context(patch('hinglish_emotion.emotions.LocalEmotionClassifier',Classifier))
    payload=b'text,emotion,language_pair,split,human_reviewed\\nHindi validation phrase,joy,hi-en,validation,true\\nFrench validation phrase,joy,en-fr,validation,true\\n'
    stack.enter_context(patch('streamlit.file_uploader',side_effect=lambda *args,**kwargs:io.BytesIO(payload)))
    render_calibration_panel(Path('.'),lambda chart,height:chart)
"""
    app=AppTest.from_string(code,default_timeout=30).run()
    next(b for b in app.button if b.label=='Fit calibration').click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    assert app.session_state['calibration_seen_texts']==['Hindi validation phrase']
    assert app.session_state['calibration_profile']['profile']['rows']==1
    app.selectbox(key='cal_pair').set_value('fr-en').run()
    next(b for b in app.button if b.label=='Fit calibration').click().run()
    assert not app.exception
    assert app.session_state['calibration_seen_texts']==['French validation phrase']


def test_effectiveness_hides_old_metrics_and_routes_to_emotion_tools():
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30)
    app.session_state['workspace']='Model effectiveness'
    app.run()
    next(b for b in app.button if b.label=='Run model evaluation').click().run()
    assert not app.exception
    assert any(m.label=='Accuracy' for m in app.metric)
    next(c for c in app.selectbox if c.label=='Primary classifier').set_value('mBERT').run()
    assert not any(m.label=='Accuracy' for m in app.metric)
    next(b for b in app.button if b.label=='Open seven-emotion evaluation').click().run()
    assert app.radio(key='workspace').value=='Emotion lab'
    app.radio(key='workspace').set_value('Model effectiveness').run()
    next(b for b in app.button if b.label=='Compare seven-emotion model outputs').click().run()
    assert app.radio(key='workspace').value=='Model comparison'
    assert not app.exception


def test_sidebar_keeps_checked_settings_after_switching_workspace():
    from unittest.mock import patch
    with patch('hinglish_emotion.ollama_reviewer.OllamaReviewer.availability',return_value=(False,'Ollama is unavailable in this test')):
        app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30)
        app.session_state['workspace']='Analyze text'
        app.run()
        next(c for c in app.selectbox if c.label=='Primary classifier').set_value('Custom checkpoint').run()
        next(t for t in app.text_input if t.label=='Local transformer folder').set_value('models/my-sentiment-model').run()
        next(t for t in app.get('toggle') if t.label=='Ask Qwen for a second opinion').set_value(True).run()
        next(t for t in app.text_input if t.label=='Ollama model').set_value('reviewer-test:latest').run()
        app.radio(key='workspace').set_value('Overview').run()
        app.radio(key='workspace').set_value('Analyze text').run()
        assert not app.exception
        assert next(c for c in app.selectbox if c.label=='Primary classifier').value=='Custom checkpoint'
        assert next(t for t in app.text_input if t.label=='Local transformer folder').value=='models/my-sentiment-model'
        assert next(t for t in app.get('toggle') if t.label=='Ask Qwen for a second opinion').value is True
        assert next(t for t in app.text_input if t.label=='Ollama model').value=='reviewer-test:latest'
