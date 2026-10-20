"""Combined-checkpoint dashboard presentation regression tests (no model download)."""
import json
from pathlib import Path
import tempfile
import unittest
from streamlit.testing.v1 import AppTest


class CombinedTrainingStatusTests(unittest.TestCase):
    def render(self, root, split_hash='split-a'):
        code = f'''from dashboard.emotions import render_training_status
render_training_status({str(root)!r}, {{'trained_language_pairs':['hi-en','fr-en'], 'training_status':'complete','text_mode':'raw','max_length':128,'dataset_sha256':'dataset-a','split_sha256':{split_hash!r}}})'''
        return AppTest.from_string(code, default_timeout=30).run()

    def write_metrics(self, root):
        scores={'accuracy':.925,'macro_f1':.9250594,'per_class':{'joy':{'precision':.9,'recall':.9,'f1':.9,'support':1}}}
        metrics={'dataset_sha256':'dataset-a','split_sha256':'split-a','evaluation_scope':'synthetic_only','split_sizes':{'train':16000,'validation':2000,'test':2000},'test':scores,'history':[{'epoch':1,'validation_macro_f1':.9666},{'epoch':2,'validation_macro_f1':.964}], 'test_by_language':{'hi-en':{'accuracy':.956,'macro_f1':.9563,'rows':1000},'fr-en':{'accuracy':.894,'macro_f1':.8937,'rows':1000}}}
        (root/'metrics.json').write_text(json.dumps(metrics),encoding='utf-8')

    def test_combined_scores_show_both_languages_and_synthetic_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.write_metrics(root)
            app=self.render(root)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.metric[0].value,'92.5%')
            languages=app.dataframe[0].value
            self.assertEqual(set(languages['Language mix']),{'Hindi–English','English–French'})
            self.assertEqual(languages['Test phrases'].sum(),2000)
            self.assertTrue(any('Synthetic test results' in x.value for x in app.warning))
            self.assertTrue(any('epoch 1' in x.value for x in app.caption))

    def test_metrics_from_another_split_are_not_displayed_as_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.write_metrics(root)
            app=self.render(root,'different-split')
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.metric),0)
            self.assertTrue(any('do not match' in x.value for x in app.warning))
