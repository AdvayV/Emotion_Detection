import io
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


class FakeStreamlit:
    def __init__(self, upload=None):
        self.upload = upload
        self.messages = []

    def subheader(self, value): pass
    def caption(self, value): pass
    def selectbox(self, label, options, **kwargs): return options[0]
    def radio(self, label, options, **kwargs): return options[0]
    def file_uploader(self, *args, **kwargs): return self.upload
    def info(self, value): self.messages.append(value)
    def error(self, value): self.messages.append(value)
    def file_uploader(self, *args, **kwargs): return self.upload


def test_calibration_panel_handles_no_installed_models(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("plotly")
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    import hinglish_emotion.checkpoints as checkpoints
    monkeypatch.setattr(checkpoints, "emotion_checkpoint", lambda *args: None)
    from dashboard.calibration_panel import render_calibration_panel
    render_calibration_panel(Path.cwd(), lambda chart, height: chart)
    assert any("No installed" in text for text in fake.messages)


def test_calibration_panel_explains_missing_labels(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("plotly")
    upload = io.BytesIO(b"text\nhello\n")
    upload.name = "unlabelled.csv"
    fake = FakeStreamlit(upload)
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    import hinglish_emotion.checkpoints as checkpoints
    monkeypatch.setattr(checkpoints, "emotion_checkpoint", lambda *args: Path("checkpoint"))
    from dashboard.calibration_panel import render_calibration_panel
    render_calibration_panel(Path.cwd(), lambda chart, height: chart)
    assert any("needs text and emotion" in text for text in fake.messages)
