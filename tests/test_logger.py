from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_logger(monkeypatch, tmp_path):
    """Reload logger.py with isolated env and redirected HOME."""
    import logging as _logging

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    sys.modules.pop("logger", None)
    _logging.getLogger("whale").handlers.clear()
    mod = importlib.import_module("logger")
    yield mod
    sys.modules.pop("logger", None)
    _logging.getLogger("whale").handlers.clear()


@pytest.fixture
def fresh_logger_with_level(monkeypatch, tmp_path):
    import logging as _logging

    def _factory(level: str | None) -> object:
        monkeypatch.setenv("HOME", str(tmp_path))
        if level is None:
            monkeypatch.delenv("LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("LOG_LEVEL", level)
        sys.modules.pop("logger", None)
        _logging.getLogger("whale").handlers.clear()
        return importlib.import_module("logger")

    yield _factory
    sys.modules.pop("logger", None)
    _logging.getLogger("whale").handlers.clear()


def _last_stdout_json(capsys) -> dict:
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected at least one log line on stdout"
    return json.loads(out[-1])


def test_default_level_is_info(fresh_logger, capsys):
    fresh_logger.logger.debug("hidden")
    fresh_logger.logger.info("hello")
    captured = capsys.readouterr().out
    assert "hidden" not in captured
    lines = [line for line in captured.strip().splitlines() if line]
    payload = json.loads(lines[-1])
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello"


def test_log_level_env_enables_debug(fresh_logger_with_level):
    mod = fresh_logger_with_level("DEBUG")
    assert mod.logger.level == logging.DEBUG


def test_invalid_log_level_falls_back_and_warns(fresh_logger_with_level, capsys):
    mod = fresh_logger_with_level("NONSENSE")
    assert mod.logger.level == logging.INFO
    err = capsys.readouterr().err
    assert "invalid LOG_LEVEL" in err


def test_log_line_is_valid_json(fresh_logger, capsys):
    fresh_logger.logger.info("hello")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    json.loads(line)


def test_core_fields_present(fresh_logger, capsys):
    fresh_logger.logger.info("hello")
    payload = _last_stdout_json(capsys)
    for key in ("ts", "level", "logger", "msg", "session_id", "pid", "hostname"):
        assert key in payload, f"missing field {key}"
    assert payload["logger"] == "whale"
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello"
    assert isinstance(payload["pid"], int)
    assert isinstance(payload["session_id"], str)
    assert len(payload["session_id"]) > 0


def test_caller_kwargs_merged(fresh_logger, capsys):
    fresh_logger.logger.info("call", model="openai/gpt-4o", tokens=123)
    payload = _last_stdout_json(capsys)
    assert payload["model"] == "openai/gpt-4o"
    assert payload["tokens"] == 123


def test_reserved_keys_ignored(fresh_logger, capsys):
    fresh_logger.logger.info("msg", level="FAKE", hostname="OVERRIDE")
    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(out[-1])
    assert payload["level"] == "INFO"
    assert payload["msg"] == "msg"
    assert payload["hostname"] != "OVERRIDE"


def test_exc_info_via_kwarg(fresh_logger, capsys):
    try:
        raise ValueError("boom")
    except ValueError:
        fresh_logger.logger.error("failed", exc_info=True)
    payload = _last_stdout_json(capsys)
    assert "exc_info" in payload
    assert "ValueError" in payload["exc_info"]
    assert "boom" in payload["exc_info"]


def test_log_file_created(fresh_logger_with_level, tmp_path):
    mod = fresh_logger_with_level(None)
    mod.logger.info("to file")
    log_file = Path(str(tmp_path)) / ".whale" / "logs" / "whale.log"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected at least one line in log file"
    payload = json.loads(lines[-1])
    assert payload["msg"] == "to file"


def test_critical_filters_info(fresh_logger_with_level, capsys):
    mod = fresh_logger_with_level("CRITICAL")
    mod.logger.info("should not appear")
    mod.logger.critical("boom")
    out = capsys.readouterr().out
    assert "should not appear" not in out
    assert "boom" in out


def test_unserializable_kwarg_skipped(fresh_logger, capsys):
    class Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    fresh_logger.logger.info("msg", opaque=Opaque(), ok=1)
    payload = _last_stdout_json(capsys)
    assert payload["ok"] == 1
    assert "opaque" not in payload


def test_all_levels_callable(fresh_logger, capsys):
    fresh_logger.logger.setLevel(logging.DEBUG)
    for fn, name in [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ]:
        getattr(fresh_logger.logger, fn)("x")
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 5
    levels = [json.loads(line)["level"] for line in out]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_log_level_warn_alias(fresh_logger_with_level, capsys):
    mod = fresh_logger_with_level("WARN")
    assert mod.logger.level == logging.WARNING
    assert "invalid LOG_LEVEL" not in capsys.readouterr().err
