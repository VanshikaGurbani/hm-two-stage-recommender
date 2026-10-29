from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_streamlit_app_renders_on_synthetic_data():
    at = st_testing.AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("SYNTHETIC" in w.value for w in at.warning)
    # change the MMR slider and the customer; the app must still render
    at.sidebar.slider[1].set_value(0.5).run()
    at.sidebar.number_input[0].set_value(2).run()
    assert not at.exception, [e.value for e in at.exception]
