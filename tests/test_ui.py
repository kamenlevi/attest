"""UI layer tests: config persistence, key masking, and the API — no network."""

import pytest

from attest import config as cfg
from attest.ui.state import AppState


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    for var in ("ATTEST_API_KEY", "ATTEST_MODEL", "ATTEST_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_config_roundtrip_keeps_defaults(isolated_config):
    c = cfg.load_config()
    c["theme"] = "light"
    c["pipeline"]["k"] = 12
    cfg.save_config(c)
    again = cfg.load_config()
    assert again["theme"] == "light"
    assert again["pipeline"]["k"] == 12
    assert "expand" in again["pipeline"]  # unset keys still defaulted


def test_public_state_masks_api_key(isolated_config):
    st = AppState()
    st.config["provider"]["api_key"] = "secret-key-1234"
    pub = st.public_state()["config"]["provider"]
    assert pub["api_key_set"] is True
    assert "secret-key-1234" not in pub["api_key"]
    assert pub["api_key"].endswith("1234")


def test_update_settings_merges_and_persists(isolated_config):
    st = AppState()
    st.update_settings({"pipeline": {"rerank": True}})
    assert st.config["pipeline"]["rerank"] is True
    assert "expand" in st.config["pipeline"]  # other keys untouched
    assert cfg.load_config()["pipeline"]["rerank"] is True  # persisted to disk


def test_ask_without_index_returns_friendly_error(isolated_config):
    st = AppState()  # no index_path configured
    out = st.ask("anything")
    assert "error" in out and "indexed" in out["error"].lower()


def test_api_state_and_settings_roundtrip(isolated_config):
    from fastapi.testclient import TestClient

    from attest.ui.server import create_app

    client = TestClient(create_app())
    state = client.get("/api/state").json()
    assert "config" in state and "library" in state
    updated = client.post("/api/settings", json={"patch": {"theme": "light"}}).json()
    assert updated["config"]["theme"] == "light"


def test_api_settings_ignores_masked_key(isolated_config):
    from fastapi.testclient import TestClient

    from attest.ui.server import create_app

    client = TestClient(create_app())
    client.post("/api/settings", json={"patch": {"provider": {"api_key": "realkey123"}}})
    # the UI later sends back the masked placeholder — it must NOT overwrite the real key
    client.post("/api/settings", json={"patch": {"provider": {"api_key": "••••y123"}}})
    assert cfg.load_config()["provider"]["api_key"] == "realkey123"
