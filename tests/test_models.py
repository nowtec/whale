from __future__ import annotations

from types import SimpleNamespace

import pytest

import models


def _fake_completion(content, calls, exc=None):
    def _fake(**kwargs):
        calls.append(kwargs)
        if exc is not None:
            raise exc
        message = SimpleNamespace(content=content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
            id="r1",
        )

    return _fake


@pytest.fixture(autouse=True)
def _model_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "m")


def test_json_parsed(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(models, "completion", _fake_completion('{"a": 1}', calls))
    assert models.completion_default_json("hi") == {"a": 1}
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["model"] == "openai/m"


def test_model_override(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(models, "completion", _fake_completion("{}", calls))
    models.completion_default_json("hi", model="x/y")
    assert calls[0]["model"] == "x/y"


def test_invalid_json_raises(monkeypatch):
    monkeypatch.setattr(models, "completion", _fake_completion("nope", []))
    with pytest.raises(models.CompletionJSONError):
        models.completion_default_json("hi")


def test_none_content_raises(monkeypatch):
    monkeypatch.setattr(models, "completion", _fake_completion(None, []))
    with pytest.raises(models.CompletionJSONError):
        models.completion_default_json("hi")


def test_provider_error_propagates(monkeypatch):
    monkeypatch.setattr(
        models, "completion", _fake_completion(None, [], exc=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        models.completion_default_json("hi")


def test_completion_json_error_is_value_error():
    assert issubclass(models.CompletionJSONError, ValueError)
