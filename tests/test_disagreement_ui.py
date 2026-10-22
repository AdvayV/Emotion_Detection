"""Streamlit-facing checks for missing models and the explicit correction gate."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class DisagreementUITests(unittest.TestCase):
    def render(self, root, available=None, fake=False):
        from streamlit.testing.v1 import AppTest

        code = """
from pathlib import Path
import dashboard.disagreement as module
from unittest.mock import patch
from contextlib import ExitStack
if MAKE_FAKE:
    class Result:
        source = "trained_checkpoint"
        probabilities = {label: (0.7 if label == "joy" else 0.05) for label in ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")}
    class Fake:
        def __init__(self, path): self.path = path
        def predict_text(self, text):
            if text == "FAIL": raise RuntimeError("injected comparison failure")
            return Result()
with ExitStack() as stack:
    stack.enter_context(patch.object(module, "available_checkpoints", lambda root, pair: AVAILABLE))
    if MAKE_FAKE:
        stack.enter_context(patch.object(module, "LocalEmotionClassifier", Fake))
    module.render_disagreement_lab(lambda title, desc: None, lambda fig, height: fig, Path(ROOT))
""".replace("AVAILABLE", repr(available or [])).replace("MAKE_FAKE", repr(fake)).replace("ROOT", repr(str(root)))
        return AppTest.from_string(code, default_timeout=30).run()

    def test_missing_models_never_falls_back_to_demo_votes(self):
        import dashboard.disagreement as module
        original = module.available_checkpoints
        with self.subTest(language="hi-en"):
            app = self.render(PROJECT_ROOT, [])
        self.assertIs(module.available_checkpoints, original)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("At least two compatible trained" in item.value for item in app.warning))
        self.assertTrue(any("Demo rules do not provide model votes" in item.value for item in app.warning))

    def test_human_confirmation_is_required_and_only_then_downloaded(self):
        import dashboard.disagreement as module
        original_classifier = module.LocalEmotionClassifier
        checkpoints = [("XLM-R", Path("xlmr"), "v1"), ("MuRIL", Path("muril"), "v2")]
        app = self.render(PROJECT_ROOT, [(name, str(path), version) for name, path, version in checkpoints], fake=True)
        self.assertIs(module.LocalEmotionClassifier, original_classifier)
        self.assertEqual(len(app.exception), 0)
        app.text_area[0].set_value("Aaj khush hoon").run()
        compare = next(button for button in app.button if button.label == "Compare checkpoints")
        compare.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertFalse(any(button.label == "Download candidate corrections (CSV)" for button in app.get("download_button")))
        confirmation = next(box for box in app.checkbox if box.label == "I reviewed this phrase and confirm the correction")
        confirmation.check().run()
        add = next(button for button in app.button if button.label == "Add reviewed correction")
        self.assertFalse(add.disabled)
        add.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIs(module.LocalEmotionClassifier, original_classifier)
        downloads = {button.label for button in app.get("download_button")}
        self.assertIn("Download candidate corrections (CSV)", downloads)
        self.assertTrue(any("current weights or accuracy" in item.value for item in app.info))

    def test_stale_results_are_invalidated_and_failed_run_clears_rows(self):
        checkpoints = [("XLM-R", "xlmr", "v1"), ("MuRIL", "muril", "v2")]
        app = self.render(PROJECT_ROOT, checkpoints, fake=True)
        app.text_area[0].set_value("Aaj khush hoon").run()
        next(button for button in app.button if button.label == "Compare checkpoints").click().run()
        self.assertEqual(len(app.get("plotly_chart")), 1)
        app.text_area[0].set_value("FAIL").run()
        self.assertEqual(len(app.get("plotly_chart")), 0)
        next(button for button in app.button if button.label == "Compare checkpoints").click().run()
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("injected comparison failure" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
