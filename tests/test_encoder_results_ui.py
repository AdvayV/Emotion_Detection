"""Dashboard checks for the saved encoder results panel."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class EncoderResultsUITests(unittest.TestCase):
    def install(self, root):
        files = [
            ("colab/mbert_study/study_contract.json", "colab/mbert_study/study_contract.json"),
            ("colab/mbert_study/xlmr_baseline_metrics.json", "colab/mbert_study/xlmr_baseline_metrics.json"),
            ("colab/mbert_study/xlmr_baseline_metrics.json", "models/xlmr_multilingual_emotion/metrics.json"),
            ("colab/mbert_study/completed_metrics.json", "models/mbert_multilingual_emotion/metrics.json"),
            ("colab/mbert_study/completed_metadata.json", "models/mbert_multilingual_emotion/emotion_metadata.json"),
        ]
        for source_relative, target_relative in files:
            target = root / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source_relative, target)

    def render(self, root):
        from streamlit.testing.v1 import AppTest
        code = ("from pathlib import Path\n"
                "from dashboard.encoder_results import render_encoder_results\n"
                f"render_encoder_results(Path({str(root)!r}), lambda fig, height: fig)\n")
        return AppTest.from_string(code, default_timeout=30).run()

    def test_verified_hi_en_results_show_metrics_chart_tables_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install(root)
            app = self.render(root)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(len(app.dataframe), 2)
        self.assertEqual(len(app.metric), 4)
        labels = {item.label for item in app.get("download_button")}
        self.assertIn("Download summary (CSV)", labels)
        self.assertIn("Download per-emotion results (CSV)", labels)
        self.assertTrue(any("synthetic-only" in item.value for item in app.warning))

    def test_missing_mbert_shows_pending_message_without_fake_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install(root)
            (root / "models/mbert_multilingual_emotion/metrics.json").unlink()
            app = self.render(root)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(any("not installed yet" in item.value for item in app.info))

    def test_french_selector_uses_verified_language_score_and_no_muril_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install(root)
            app = self.render(root)
            app.selectbox[0].select("fr-en").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertFalse(any("MuRIL" in item.value for item in app.warning))
