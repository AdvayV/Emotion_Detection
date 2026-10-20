import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
import tempfile


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class StreamlitAppTests(unittest.TestCase):
    def make_app(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30)
        app.session_state["workspace"] = "Analyze text"
        return app.run()

    def test_untrained_encoder_is_shown_as_unavailable(self):
        app = self.make_app()
        selector = next(item for item in app.selectbox if item.label == "Primary classifier")
        selector.set_value("mBERT").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("not trained here yet" in item.value for item in app.warning))
        self.assertTrue(next(button for button in app.button if button.label == "Analyze message").disabled)
        next(button for button in app.button if button.label == "Use rule-based baseline").click().run()
        self.assertFalse(next(button for button in app.button if button.label == "Analyze message").disabled)

    def test_custom_model_requires_a_checkpoint(self):
        app = self.make_app()
        next(item for item in app.selectbox if item.label == "Primary classifier").set_value("Custom checkpoint").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(next(button for button in app.button if button.label == "Analyze message").disabled)

    def test_analysis_persists_without_duplicating_history(self):
        app = self.make_app()
        next(item for item in app.selectbox if item.label == "Try an example").set_value("Positive with emphasis").run()
        self.assertIn("acchaaa", app.text_area[0].value)
        next(button for button in app.button if button.label == "Analyze message").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["last_analysis"]["prediction"].final_label, "positive")
        self.assertEqual(len(app.session_state["history"]), 1)
        app.run()
        self.assertEqual(len(app.session_state["history"]), 1)
        self.assertTrue(any("Positive" in item.value for item in app.subheader))
        app.text_area[0].set_value("service acchi nahi thi").run()
        self.assertTrue(any("previous result" in item.value for item in app.info))
        next(button for button in app.button if button.label == "Analyze message").click().run()
        self.assertEqual(app.session_state["last_analysis"]["prediction"].final_label, "negative")
        self.assertEqual(len(app.session_state["history"]), 2)
        self.assertEqual(len(app.get("download_button")), 1)

    def test_sarcasm_review_reasons_are_readable(self):
        app = self.make_app()
        next(item for item in app.selectbox if item.label == "Try an example").set_value("Possible sarcasm").run()
        next(button for button in app.button if button.label == "Analyze message").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("second look" in item.value for item in app.warning))
        self.assertTrue(any("sarcastically" in item.value for item in app.markdown))

    def test_dashboard_and_effectiveness_graphs_render(self):
        app = self.make_app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio[0].value, "Analyze text")
        self.assertEqual(len(app.get("plotly_chart")), 0)
        app.radio[0].set_value("Model effectiveness").run()

        evaluation_button = next(
            button for button in app.button if button.label == "Run model evaluation"
        )
        evaluation_button.click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        metric_values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metric_values["Accuracy"], "88.0%")
        self.assertEqual(metric_values["Macro-F1"], "87.5%")
        self.assertEqual(len(app.get("plotly_chart")), 3)

    def test_dataset_filter_and_overview_navigation(self):
        app = self.make_app()
        app.radio[0].set_value("Dataset explorer").run()
        self.assertEqual(len(app.exception), 0)
        next(item for item in app.multiselect if item.label == "Sentiment filter").set_value(["negative"]).run()
        self.assertEqual(len(app.dataframe[0].value), 25)
        next(item for item in app.text_input if item.label == "Search messages").set_value("no matching phrase 123").run()
        self.assertEqual(len(app.dataframe[0].value), 0)
        app.radio[0].set_value("Overview").run()
        self.assertEqual(len(app.exception), 0)
        next(button for button in app.button if button.label == "Open text analysis").click().run()
        self.assertEqual(app.radio[0].value, "Analyze text")

    def test_seven_emotion_lab_analyzes_and_evaluates_current_hinglish_scope(self):
        app = self.make_app()
        app.radio[0].set_value("Emotion lab").run()
        self.assertEqual(len(app.exception), 0)
        next(item for item in app.selectbox if item.label == "Emotion classifier").set_value("Demo lexical rules").run()
        next(item for item in app.selectbox if item.label == "Emotion example").set_value("Hindi–English · happy").run()
        next(button for button in app.button if button.label == "Detect emotion").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["emotion_result"]["result"].emotion, "joy")
        next(button for button in app.button if button.label == "Evaluate seven emotions").click().run()
        self.assertEqual(len(app.exception), 0)
        report = app.session_state["emotion_evaluation"]["report"]
        self.assertEqual(report["overall"]["rows"], 42)
        self.assertEqual(set(report["by_language"]), {"hi-en"})
        self.assertEqual(next(item for item in app.selectbox if item.label == "Language mix").options, ["Hindi–English", "English–French"])
        with patch("dashboard.emotions.checkpoint_ready", return_value=False):
            next(item for item in app.selectbox if item.label == "Emotion classifier").set_value("XLM-R").run()
        self.assertTrue(next(button for button in app.button if button.label == "Detect emotion").disabled)

    def test_emotion_lab_discovers_trained_checkpoint_and_retains_scope_warning(self):
        from hinglish_emotion.emotions import EmotionResult
        from hinglish_emotion.tasks import EMOTION_LABELS
        result = EmotionResult("joy", {label: 1.0 if label == "joy" else 0.0 for label in EMOTION_LABELS}, source="trained_checkpoint", experimental=True, warnings=("Synthetic test data; real-world accuracy is unverified.",))
        detector = Mock()
        detector.predict.return_value = result
        metadata = {"schema_version": 1, "text_mode": "raw", "max_length": 128, "trained_language_pairs": ["hi-en"], "validated_language_pairs": [], "training_status": "complete", "evaluation_scope": "synthetic_only"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models" / "xlmr_emotion"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            with patch("dashboard.emotions.ROOT", root), patch("dashboard.emotions.checkpoint_ready", side_effect=lambda path: Path(path).name == "xlmr_emotion"), patch("dashboard.emotions.emotion_checkpoint_metadata", return_value=metadata), patch("dashboard.emotions.detector_for", return_value=detector):
                app = self.make_app()
                app.radio[0].set_value("Emotion lab").run()
                selector = next(item for item in app.selectbox if item.label == "Emotion classifier")
                self.assertEqual(selector.value, "XLM-R")
                self.assertTrue(any("generated phrases" in item.value for item in app.warning))
                next(item for item in app.text_area if item.label == "Phrase to analyze").set_value("Aaj main bahut khush hoon").run()
                next(button for button in app.button if button.label == "Detect emotion").click().run()
                self.assertEqual(len(app.exception), 0)
                detector.predict.assert_called_once_with("Aaj main bahut khush hoon", "hi-en")
                self.assertEqual(app.session_state["emotion_result"]["result"].source, "trained_checkpoint")
                self.assertTrue(any("real-world accuracy" in item.value.lower() for item in app.warning))
                next(item for item in app.text_area if item.label == "Phrase to analyze").set_value("A different phrase").run()
                self.assertNotIn("emotion_result", app.session_state)


if __name__ == "__main__":
    unittest.main()
