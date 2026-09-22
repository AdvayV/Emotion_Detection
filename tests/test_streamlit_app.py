import importlib.util
from pathlib import Path
import unittest


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_and_effectiveness_graphs_render(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Overview", "Analyze text", "Model effectiveness", "Dataset explorer"],
        )

        evaluation_button = next(
            button for button in app.button if button.label == "Run model evaluation"
        )
        evaluation_button.click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        metric_values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metric_values["Accuracy"], "88.0%")
        self.assertEqual(metric_values["Macro-F1"], "87.5%")
        self.assertGreaterEqual(len(app.get("plotly_chart")), 6)


if __name__ == "__main__":
    unittest.main()
