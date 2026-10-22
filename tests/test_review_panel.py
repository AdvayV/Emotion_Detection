from pathlib import Path

from dashboard import review_panel


def test_panel_is_single_render_function():
    assert callable(review_panel.render_review_panel)
    assert "human_reviewed_corrections" in Path(review_panel.__file__).read_text(encoding="utf-8")
