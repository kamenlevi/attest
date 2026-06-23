import os

from attest.cli import _load_dotenv


def test_loads_keys_but_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        'ATTEST_MODEL="some-model"\n'
        "ATTEST_BASE_URL=http://example/v1\n"
        "ATTEST_ALREADY_SET=from_file\n"
    )
    monkeypatch.delenv("ATTEST_MODEL", raising=False)
    monkeypatch.delenv("ATTEST_BASE_URL", raising=False)
    monkeypatch.setenv("ATTEST_ALREADY_SET", "from_env")  # should win over the file

    _load_dotenv(str(env_file))

    assert os.environ["ATTEST_MODEL"] == "some-model"      # quotes stripped
    assert os.environ["ATTEST_BASE_URL"] == "http://example/v1"
    assert os.environ["ATTEST_ALREADY_SET"] == "from_env"  # not overridden


def test_missing_file_is_noop():
    _load_dotenv("/no/such/.env")  # must not raise
