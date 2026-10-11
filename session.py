from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger import logger

DEFAULT_ROOT = Path.home() / ".whale"
SESSIONS_DIR_NAME = "sessions"


@dataclass
class Session:
    id: str
    path: Path
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
        }


class SessionManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or DEFAULT_ROOT).expanduser()
        self.sessions_dir = self.root / SESSIONS_DIR_NAME
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def initSession(self, path: Path | None = None) -> Session:
        session_id = uuid.uuid4().hex
        now = self._now()
        session = Session(
            id=session_id,
            path=self._path_for(session_id),
            created_at=now,
            updated_at=now,
            messages=[],
        )
        session.path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("session.init", session_id=session_id, cwd=str(path or Path.cwd()))
        return session

    def appendMessage(self, session_id: str, message: dict[str, Any]) -> None:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("messages", []).append(message)
        data["updated_at"] = self._now()
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("session.append", session_id=session_id, role=message.get("role"))

    def close_session(
        self,
        session_id: str,
        *,
        evolve: bool = False,
        memory_engine: Any | None = None,
    ) -> Any | None:
        """Mark the session closed; optionally evolve the procedural graph."""
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["closed_at"] = self._now()
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not evolve:
            logger.info("session.close", session_id=session_id, evolve=False)
            return None

        engine = memory_engine
        if engine is None:
            import memory  # lazy: memory loads sessions through this module

            engine = memory.memoryEngine
        result = engine.evolve_from_trajectory(session_id)
        logger.info(
            "session.close",
            session_id=session_id,
            accepted=result.accepted,
            score_before=result.score_before,
            score_after=result.score_after,
        )
        return result

    def listSessions(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for file in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("session.read_failed", path=str(file), error=str(exc))
                continue
            items.append(
                {
                    "id": data.get("id", file.stem),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                }
            )
        items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return items

    def loadSession(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])


sessionManager = SessionManager()


def initSession(path: Path | None = None) -> Session:
    return sessionManager.initSession(path)


def appendMessage(session_id: str, message: dict[str, Any]) -> None:
    sessionManager.appendMessage(session_id, message)


def close_session(
    session_id: str, *, evolve: bool = False, memory_engine: Any | None = None
) -> Any | None:
    return sessionManager.close_session(
        session_id, evolve=evolve, memory_engine=memory_engine
    )


def listSessions() -> list[dict[str, Any]]:
    return sessionManager.listSessions()


def loadSession(session_id: str) -> list[dict[str, Any]]:
    return sessionManager.loadSession(session_id)
