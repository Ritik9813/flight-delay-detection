"""Smoke test: every dashboard page renders without raising (needs trained model)."""
import pytest

from src.config import DATA_PATH, MODEL_PATH

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Home Dashboard", "Dataset Explorer", "Flight Delay Prediction", "Batch Prediction",
    "Analytics", "Model Performance", "Prediction History",
]


@pytest.mark.skipif(not (MODEL_PATH.exists() and DATA_PATH.exists()),
                    reason="Run generate_dataset.py and train_model.py first")
def test_all_pages_render_without_exceptions():
    at = AppTest.from_file(str(MODEL_PATH.parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
