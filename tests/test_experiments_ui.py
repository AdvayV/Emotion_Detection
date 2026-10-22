import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Dashboard dependencies are optional")
class ExperimentPanelUITests(unittest.TestCase):
    def test_panel_controls_and_explicit_bounded_reviewer(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string(f'from pathlib import Path\nfrom dashboard.experiments_panel import render_experiments_panel\nrender_experiments_panel(Path({str(ROOT)!r}))')
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any(x.label == "Maximum rows" and x.value == 10 for x in app.number_input))
        self.assertTrue(any(x.label == "Optional local Qwen comparison" and x.value == "none" for x in app.selectbox))
        self.assertTrue(any(x.label == "Run experiments" for x in app.button))

if __name__ == "__main__": unittest.main()
