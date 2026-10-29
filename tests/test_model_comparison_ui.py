"""Interaction tests for the unified dataset/model comparison workspace."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT=Path(__file__).resolve().parents[1]


def comparison_app(tmp_path):
    folder=tmp_path/'evaluation_set';folder.mkdir()
    data='text,emotion,alternative,language_pair,source\nalpha,joy,sadness,hi-en,assistant_draft\nbeta,sadness,joy,hi-en,assistant_draft\n'
    (folder/'hindi_english_100.csv').write_text(data,encoding='utf-8')
    (folder/'english_french_100.csv').write_text(data.replace('hi-en','en-fr'),encoding='utf-8')
    code="""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import streamlit as st
from dashboard.comparison import render_model_comparison
from hinglish_emotion.emotions import EmotionResult
from hinglish_emotion.tasks import EMOTION_LABELS
class Classifier:
    metadata={'trained_language_pairs':['hi-en','fr-en'],'validated_language_pairs':[],'training_data_kind':'synthetic_only'}
    def __init__(self,path):self.name=Path(path).name
    def predict_batch(self,texts,batch_size):
        if st.session_state.get('fail_comparison'): raise RuntimeError('injected failure')
        labels=[('joy' if text=='alpha' else 'sadness') if self.name=='xlmr' else 'neutral' for text in texts]
        return [EmotionResult(label,{k:(.94 if k==label else .01) for k in EMOTION_LABELS},source='trained_checkpoint',model_name=self.name) for label in labels]
def available(root,pair):
    names=[('XLM-R','xlmr'),('mBERT','mbert')]+([('MuRIL','muril')] if pair=='hi-en' else [])
    return [(name,root/folder,st.session_state.get('fixture_version','1')) for name,folder in names]
with ExitStack() as stack:
    stack.enter_context(patch('dashboard.comparison.available_checkpoints',available))
    stack.enter_context(patch('dashboard.comparison.local_comparison_classifier',Classifier))
    render_model_comparison(lambda title,desc:st.title(title),lambda fig,height=320:fig,Path(ROOT))
""".replace('Path(ROOT)',f'Path({str(tmp_path)!r})')
    return AppTest.from_string(code,default_timeout=30).run()


def run(app):
    app.button(key='comparison_run').click().run()
    assert not app.exception
    assert not app.error,[e.value for e in app.error]
    return app.session_state['model_comparison_result']['report']


def test_three_models_share_scores_outputs_and_exports(tmp_path):
    app=comparison_app(tmp_path)
    report=run(app)
    assert report['scored_rows']==2
    assert [m['name'] for m in report['models']]==['XLM-R','mBERT','MuRIL']
    assert report['models'][0]['metrics']['accuracy']==1
    assert len(app.get('plotly_chart'))==5
    assert {b.label for b in app.get('download_button')}=={'Download comparison scores (CSV)','Download aligned model outputs (CSV)','Download complete comparison report (JSON)'}
    app.run()
    assert len(app.get('download_button'))==3
    app.radio(key='comparison_row_filter').set_value('Models disagree').run()
    assert app.session_state['model_comparison_result']['report']['scored_rows']==2


def test_label_mapping_and_checkpoint_change_hide_stale_scores(tmp_path):
    app=comparison_app(tmp_path)
    run(app)
    next(w for w in app.selectbox if w.label=='Gold emotion column').set_value('alternative').run()
    assert not app.get('plotly_chart') and not app.get('download_button')
    assert any('changed' in w.value for w in app.info)
    report=run(app)
    assert report['models'][0]['metrics']['accuracy']==0
    app.session_state['fixture_version']='2'
    app.run()
    assert not app.get('plotly_chart')


def test_no_labels_never_produces_accuracy_and_language_controls_model_scope(tmp_path):
    app=comparison_app(tmp_path)
    app.selectbox(key='comparison_language').set_value('fr-en').run()
    assert app.multiselect(key='comparison_models_fr-en').value==['XLM-R','mBERT']
    next(w for w in app.selectbox if w.label=='Gold emotion column').set_value('No expected labels').run()
    report=run(app)
    assert report['scored_rows']==0 and len(report['models'])==2
    assert all(m['metrics'] is None for m in report['models'])
    assert not app.get('plotly_chart')
    assert any('no accuracy or F1' in w.value for w in app.info)


def test_failed_run_clears_prior_results_and_one_model_disables_run(tmp_path):
    app=comparison_app(tmp_path)
    run(app)
    app.session_state['fail_comparison']=True
    app.button(key='comparison_run').click().run()
    assert not app.exception
    assert any('injected failure' in w.value for w in app.error)
    assert not app.get('plotly_chart') and not app.get('download_button')
    app.multiselect(key='comparison_models_hi-en').set_value(['XLM-R']).run()
    assert app.button(key='comparison_run').disabled


def test_workspace_navigation_exposes_the_new_page():
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30)
    app.session_state['workspace']='Emotion lab';app.run()
    app.button(key='sidebar_model_comparison').click().run()
    assert not app.exception
    assert app.radio(key='workspace').value=='Model comparison'
    assert any(title.value=='Model comparison' for title in app.title)
