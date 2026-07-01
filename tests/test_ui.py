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


def _indexed_state(isolated_config, tmp_path):
    """An AppState with a tiny mock-embedded index — no models, no network."""
    st = AppState()
    st.config["models"]["embedder"] = "mock"
    st.config["index_path"] = str(tmp_path / "idx")
    doc = tmp_path / "doc.txt"
    doc.write_text("photosynthesis happens inside chloroplasts of plant cells")
    st.index_file(str(doc))
    return st


def test_ask_returns_verification_and_cited_details(isolated_config, tmp_path):
    st = _indexed_state(isolated_config, tmp_path)
    out = st.ask("where does photosynthesis happen?")
    assert out["abstained"] is False
    # Mock provider means no judge -> citations are valid but unconfirmed.
    assert out["verification"]["status"] == "unverified"
    assert out["cited"][0]["source"] == "doc.txt"
    assert "page" in out["passages"][0]


def test_ask_trap_abstains_with_abstained_status(isolated_config, tmp_path):
    st = _indexed_state(isolated_config, tmp_path)
    out = st.ask("what is the lorentz transformation?")
    assert out["abstained"] is True
    assert out["verification"]["status"] == "abstained"


def test_pipeline_is_cached_across_asks(isolated_config, tmp_path):
    st = _indexed_state(isolated_config, tmp_path)
    r1 = st.retriever()
    r2 = st.retriever()
    assert r1 is r2  # same object — no per-question rebuild
    st.update_settings({"pipeline": {"lexical": False}})
    assert st.retriever() is not r1  # relevant change -> rebuilt
    st.update_settings({"theme": "light"})
    kept = st.retriever()
    assert st.retriever() is kept  # theme change must NOT rebuild


def test_convert_file_writes_clean_text_without_overwriting(isolated_config, tmp_path):
    st = AppState()
    doc = tmp_path / "note.txt"
    doc.write_text("some plain text")
    out = st.convert_file(str(doc))
    assert out["ok"]
    from pathlib import Path
    dest = Path(out["out"])
    assert dest != doc  # original untouched
    assert dest.read_text() == "some plain text"
    assert "plain text" in out["preview"]


def test_run_eval_reports_trust_metrics(isolated_config, tmp_path):
    import json
    st = _indexed_state(isolated_config, tmp_path)
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps([
        {"question": "where does photosynthesis happen?", "answerable": True},
        {"question": "what is the lorentz transformation?", "answerable": False},
    ]))
    out = st.run_eval(str(qfile), judge=True)  # no provider -> judge skipped
    assert out["ok"]
    assert out["metrics"]["bluff_rate"] == 0.0
    assert out["metrics"]["answer_coverage"] == 1.0
    assert len(out["rows"]) == 2


def test_run_eval_bad_questions_file_is_friendly(isolated_config, tmp_path):
    st = _indexed_state(isolated_config, tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    assert "error" in st.run_eval(str(bad))
    assert "error" in st.run_eval(str(tmp_path / "missing.json"))


def test_api_convert_endpoint(isolated_config, tmp_path):
    from fastapi.testclient import TestClient

    from attest.ui.server import create_app

    doc = tmp_path / "note.txt"
    doc.write_text("endpoint text")
    client = TestClient(create_app())
    res = client.post("/api/convert", json={"path": str(doc)}).json()
    assert res["ok"] and res["chars"] > 0
