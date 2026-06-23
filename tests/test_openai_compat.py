from attest.backends.openai_compat import OpenAICompatibleGenerator


def test_builds_correct_request_and_parses_response():
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "the answer is X [1]"}}]}

    gen = OpenAICompatibleGenerator(
        model="gpt-test", base_url="https://api.example.com/v1/", api_key="secret",
        post_fn=fake_post,
    )
    out = gen.generate("hello")

    assert out == "the answer is X [1]"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_no_api_key_means_no_auth_header():
    def fake_post(url, payload, headers, timeout):
        assert "Authorization" not in headers  # local servers need no key
        return {"choices": [{"message": {"content": "ok"}}]}

    gen = OpenAICompatibleGenerator(
        model="llama", base_url="http://localhost:11434/v1", post_fn=fake_post
    )
    assert gen.generate("hi") == "ok"
