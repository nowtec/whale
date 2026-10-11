from __future__ import annotations

import json
import logging
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_VALID_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_RESERVED_KEYS = frozenset(
    {
        "ts",
        "level",
        "logger",
        "msg",
        "session_id",
        "pid",
        "hostname",
        "exc_info",
    }
)

_STDLOG_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)

_SESSION_ID = uuid.uuid4().hex
_PID = os.getpid()
_HOSTNAME = socket.gethostname()


def _resolve_level() -> int:
    raw = os.environ.get("LOG_LEVEL", "INFO").upper()
    if raw in _VALID_LEVELS:
        return _VALID_LEVELS[raw]
    sys.stderr.write(f"[logger] invalid LOG_LEVEL={raw!r}, falling back to INFO\n")
    sys.stderr.flush()
    return logging.INFO


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": "whale",
            "msg": record.getMessage(),
            "session_id": _SESSION_ID,
            "pid": _PID,
            "hostname": _HOSTNAME,
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _STDLOG_ATTRS or key in _RESERVED_KEYS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                continue

        return json.dumps(payload, ensure_ascii=False)


class _SafeStreamHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:  # type: ignore[override]
        pass

    def handleError(self, record: logging.LogRecord) -> None:
        sys.stderr.write(f"[logger] handler error: {sys.exc_info()[1]}\n")
        sys.stderr.flush()


class _SafeFileHandler(RotatingFileHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        sys.stderr.write(f"[logger] file handler error: {sys.exc_info()[1]}\n")
        sys.stderr.flush()


def _add_file_handler(logger: logging.Logger) -> None:
    try:
        log_dir = Path.home() / ".whale" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = _SafeFileHandler(
            log_dir / "whale.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    except OSError as exc:
        sys.stderr.write(f"[logger] file handler disabled: {exc}\n")
        sys.stderr.flush()


logger = logging.getLogger("whale")
logger.setLevel(_resolve_level())
logger.propagate = False

_stream_handler = _SafeStreamHandler()
_stream_handler.setFormatter(_JsonFormatter())
logger.addHandler(_stream_handler)

_add_file_handler(logger)


def _log(
    level: int,
    msg: str,
    *args: Any,
    exc_info: Any = False,
    **kwargs: Any,
) -> None:
    extra = {k: v for k, v in kwargs.items() if k not in _RESERVED_KEYS}
    logger.log(level, msg, *args, exc_info=exc_info, extra=extra)


def _filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in _RESERVED_KEYS}


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.DEBUG, msg, *args, **_filter_kwargs(kwargs))


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.INFO, msg, *args, **_filter_kwargs(kwargs))


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    _log(logging.WARNING, msg, *args, **_filter_kwargs(kwargs))


def error(msg: str, *args: Any, exc_info: Any = False, **kwargs: Any) -> None:
    _log(logging.ERROR, msg, *args, exc_info=exc_info, **_filter_kwargs(kwargs))


def critical(msg: str, *args: Any, exc_info: Any = False, **kwargs: Any) -> None:
    _log(logging.CRITICAL, msg, *args, exc_info=exc_info, **_filter_kwargs(kwargs))


logger.debug = debug  # type: ignore[attr-defined]
logger.info = info  # type: ignore[attr-defined]
logger.warning = warning  # type: ignore[attr-defined]
logger.error = error  # type: ignore[attr-defined]
logger.critical = critical  # type: ignore[attr-defined]
