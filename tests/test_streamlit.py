from pathlib import Path


def test_streamlit_script_exists():
    candidates = [Path("app.py"), Path("app.py"), Path("app(1).py")]
    assert any(path.exists() for path in candidates)
