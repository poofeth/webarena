import openai
import pytest

from llms.providers import openai_utils


def test_eval_chat_completion_uses_separate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "correct"}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "action-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://action.example/v1")
    monkeypatch.setenv("WEBARENA_EVAL_OPENAI_API_KEY", "eval-key")
    monkeypatch.setenv("WEBARENA_EVAL_OPENAI_API_BASE", "https://eval.example/v1")
    monkeypatch.setenv("WEBARENA_EVAL_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)

    response = openai_utils.generate_from_openai_eval_chat_completion(
        messages=[{"role": "user", "content": "grade this"}],
    )

    assert response == "correct"
    assert openai.api_key == "eval-key"
    assert openai.api_base == "https://eval.example/v1"
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["temperature"] == 0


def test_gpt5_chat_completion_omits_custom_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "same"}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "action-key")
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)

    openai_utils.generate_from_openai_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5-mini",
        temperature=0,
        max_tokens=16,
        top_p=1.0,
        context_length=0,
    )

    assert "temperature" not in calls[0]
    assert "top_p" not in calls[0]


def test_gpt5_chat_completion_keeps_non_default_top_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "same"}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "action-key")
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)

    openai_utils.generate_from_openai_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5-mini",
        temperature=0,
        max_tokens=16,
        top_p=0.5,
        context_length=0,
    )

    assert "temperature" not in calls[0]
    assert calls[0]["top_p"] == 0.5


def test_eval_chat_completion_falls_back_to_default_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_create(**kwargs):
        return {"choices": [{"message": {"content": "correct"}}]}

    monkeypatch.delenv("WEBARENA_EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "action-key")
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)

    response = openai_utils.generate_from_openai_eval_chat_completion(
        messages=[{"role": "user", "content": "grade this"}],
    )

    assert response == "correct"
    assert openai.api_key == "action-key"


def test_default_openai_base_is_restored_when_no_base_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "action-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    openai.api_base = "https://previous.example/v1"

    def fake_create(**kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)
    response = openai_utils.generate_from_openai_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=16,
        top_p=1.0,
        context_length=0,
    )

    assert response == "ok"
    assert openai.api_base == "https://api.openai.com/v1"
