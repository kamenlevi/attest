"""Vision-extractor tests: payload shape + retry, and page-spec parsing. No network."""

import pytest

from attest.backends.vision_extract import VisionExtractor
from attest.cli import _parse_pages


def test_parse_pages_handles_ranges_lists_and_none():
    assert _parse_pages(None) is None
    assert _parse_pages("41") == [40]            # 1-based -> 0-based
    assert _parse_pages("41,46") == [40, 45]
    assert _parse_pages("40-46") == [39, 40, 41, 42, 43, 44, 45]  # inclusive


def test_transcribe_sends_image_and_returns_content():
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["payload"] = payload
        captured["auth"] = headers.get("Authorization")
        return {"choices": [{"message": {"content": "$E = mc^2$"}}]}

    vx = VisionExtractor("vmodel", "http://x/v1", api_key="K", post_fn=fake_post)
    assert vx._transcribe_png(b"PNGBYTES") == "$E = mc^2$"
    content = captured["payload"]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["auth"] == "Bearer K"


def test_transcribe_retries_transient_failures():
    calls = []

    def flaky(url, payload, headers, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return {"choices": [{"message": {"content": "ok"}}]}

    vx = VisionExtractor("m", "http://x", retries=3, backoff=0, post_fn=flaky)
    assert vx._transcribe_png(b"x") == "ok"
    assert len(calls) == 3


def test_transcribe_raises_after_exhausting_retries():
    def always_fail(url, payload, headers, timeout):
        raise RuntimeError("down")

    vx = VisionExtractor("m", "http://x", retries=2, backoff=0, post_fn=always_fail)
    with pytest.raises(RuntimeError):
        vx._transcribe_png(b"x")
