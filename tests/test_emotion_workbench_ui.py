"""Focused UI checks for the emotion workbench controls and model routing."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class EmotionWorkbenchUITests(unittest.TestCase):
    def make_app(self):
        from streamlit.testing.v1 import AppTest

        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30)
        app.session_state["workspace"] = "Emotion lab"
        return app.run()

    def test_optional_review_controls_default_off_and_context_is_available(self):
        app = self.make_app()
        self.assertEqual(len(app.exception), 0)
        checkboxes = {item.label: item.value for item in app.checkbox}
        self.assertFalse(checkboxes["Use optional local Qwen review when needed"])
        self.assertFalse(checkboxes["Use trained sarcasm model"])
        self.assertTrue(any(item.label == "Optional conversation context" for item in app.text_area))

    def test_all_encoder_choices_use_emotion_checkpoint_registry(self):
        from dashboard.emotions import resolve_model
        from hinglish_emotion.model_catalog import ENCODERS

        metadata = {
            "trained_language_pairs": ["hi-en"], "training_status": "complete",
            "text_mode": "raw", "max_length": 128,
        }
        for encoder in ENCODERS:
            with self.subTest(encoder=encoder.name), patch("dashboard.emotions.emotion_checkpoint", return_value=Path("model")) as discover, patch("dashboard.emotions.checkpoint_ready", return_value=True), patch("dashboard.emotions.emotion_checkpoint_metadata", return_value=metadata):
                path, found, issue = resolve_model(encoder.name, "hi-en")
                self.assertEqual(path, Path("model"))
                self.assertEqual(found, metadata)
                self.assertEqual(issue, "")
                discover.assert_called_once()
                self.assertEqual(discover.call_args.args[2], encoder.key)

    def test_optional_sarcasm_path_and_colab_package_downloads_are_present(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30)
        app.session_state["workspace"] = "Emotion lab"
        app.run()
        app.checkbox(key="emotion_sarcasm_enabled").set_value(True)
        app.run()
        self.assertTrue(any(item.label == "Sarcasm checkpoint folder" and item.value == "models/xlmr_sarcasm" for item in app.text_input))
        download_labels = {item.label for item in app.get("download_button")}
        self.assertIn("Download project training notebook", download_labels)
        self.assertIn("Download full Colab training package", download_labels)
        self.assertIn("Download project training guide", download_labels)


if __name__ == "__main__":
    unittest.main()
