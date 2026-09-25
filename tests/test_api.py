def test_api_module_imports():
    from app.api.server import app
    assert app.title.startswith("SAFE-AI")
