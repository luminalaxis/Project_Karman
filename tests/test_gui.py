"""Smoke test of the Streamlit GUI (skipped when streamlit is not installed)."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_gui_renders_and_computes():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception and not at.error
    lap = {m.label: m.value for m in at.metric}
    assert float(lap["랩타임 [s]"]) > 0
    [b for b in at.button if b.label == "랩타임 계수 계산"][0].click()
    at.run()
    assert not at.exception
    assert any(m.label == "+1 % ClA" for m in at.metric)
