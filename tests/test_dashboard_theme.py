import importlib.util
from pathlib import Path
import unittest


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class DashboardThemeTests(unittest.TestCase):
    def make_app(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30)
        app.session_state["workspace"] = "Emotion lab"
        return app.run()

    def test_emotion_workspace_uses_its_own_sidebar_settings_and_theme(self):
        app = self.make_app()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Model disagreement", app.radio[0].options)
        self.assertTrue(any(item.value == "Seven-emotion settings" for item in app.subheader))
        self.assertFalse(any(item.label == "Primary classifier" for item in app.selectbox))
        styles = "\n".join(item.value for item in app.markdown)
        self.assertIn("stSidebar", styles)
        self.assertIn("stMetric", styles)
        self.assertIn("@media (max-width: 760px)", styles)

    def test_emotion_page_exposes_checkpoint_readiness_and_prediction_source(self):
        app = self.make_app()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Installed trained checkpoints for this language mix" in item.value for item in app.caption))
        next(item for item in app.selectbox if item.label == "Emotion classifier").set_value("Demo lexical rules").run()
        self.assertTrue(any("no trained emotion checkpoint is running" in item.value for item in app.info))

    def test_disagreement_page_routes_through_seven_emotion_sidebar(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30)
        app.session_state["workspace"] = "Model disagreement"
        app = app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(item.value == "Seven-emotion settings" for item in app.subheader))
        self.assertFalse(any(item.label == "Primary classifier" for item in app.selectbox))
        self.assertTrue(any(item.label == "Language mix" for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()
