"""Changing the expected-label column must invalidate old accuracy."""
from pathlib import Path
import tempfile
from streamlit.testing.v1 import AppTest


def test_label_mapping_change_clears_displayed_accuracy():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data/demo/emotions"
        data.mkdir(parents=True)
        (data / "mixed_emotions_demo.csv").write_text(
            "text,label_a,label_b,language_pair\nMain khush hoon,joy,sadness,hi-en\n", encoding="utf-8")
        code = f"""from pathlib import Path
from dashboard.emotion_batch import render_file_predictions
render_file_predictions(Path({str(root)!r}), "Demo lexical rules", "hi-en", lambda *args: (None,None,""), lambda figure,height=320: figure)
"""
        app = AppTest.from_string(code, default_timeout=30).run()
        assert not app.exception
        next(item for item in app.selectbox if item.label == "Expected emotion column").set_value("label_a").run()
        next(item for item in app.button if item.label == "Evaluate seven emotions").click().run()
        assert not app.exception
        assert next(item for item in app.metric if item.label == "Emotion accuracy").value == "100.0%"
        next(item for item in app.selectbox if item.label == "Expected emotion column").set_value("label_b").run()
        assert not any(item.label == "Emotion accuracy" for item in app.metric)
        assert any("File or settings changed" in item.value for item in app.info)
        next(item for item in app.button if item.label == "Evaluate seven emotions").click().run()
        assert next(item for item in app.metric if item.label == "Emotion accuracy").value == "0.0%"
