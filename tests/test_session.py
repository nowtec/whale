from __future__ import annotations

import json

import pytest

from procedural_memory.refiner import EvolutionResult
from session import SessionManager


class _FakeEngine:
    def __init__(self):
        self.calls: list[str] = []

    def evolve_from_trajectory(self, session_id):
        self.calls.append(session_id)
        return EvolutionResult(1, True, 0.9, 1.0, 2, 1, None)


@pytest.fixture
def manager(tmp_path):
    return SessionManager(tmp_path)


def test_close_marks_closed_at_and_keeps_messages(manager):
    session = manager.initSession()
    message = {"role": "user", "content": "hi"}
    manager.appendMessage(session.id, message)

    assert manager.close_session(session.id) is None

    data = json.loads(session.path.read_text())
    assert data["closed_at"].endswith("Z")
    assert data["messages"] == [message]
    assert data["id"] == session.id


def test_close_does_not_evolve_by_default(manager):
    session = manager.initSession()
    engine = _FakeEngine()
    manager.close_session(session.id, memory_engine=engine)
    assert engine.calls == []


def test_close_with_evolve_calls_engine_once(manager):
    session = manager.initSession()
    engine = _FakeEngine()
    result = manager.close_session(session.id, evolve=True, memory_engine=engine)
    assert engine.calls == [session.id]
    assert result.accepted is True and result.score_after == 1.0


def test_close_unknown_session_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.close_session("nope")


def test_session_still_readable_and_appendable_after_close(manager):
    session = manager.initSession()
    manager.close_session(session.id)
    manager.appendMessage(session.id, {"role": "user", "content": "late"})
    assert manager.loadSession(session.id) == [{"role": "user", "content": "late"}]
    assert "closed_at" in json.loads(session.path.read_text())
