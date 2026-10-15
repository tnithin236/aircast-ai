"""Run every Streamlit page headlessly against the synthetic test artifacts."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app/streamlit_app.py"] + [f"app/pages/{p.name}" for p in sorted((ROOT / "app/pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page, trained_env):
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / page), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_page_shows_error_not_crash_without_models(tmp_path, monkeypatch, trained_env):
    import streamlit as st

    monkeypatch.setenv("AIRCAST_MODELS_DIR", str(tmp_path))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Missing model artifacts" in e.value for e in at.error)
