import json
import os
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from litellm import completion

from logger import logger

load_dotenv()


DEFAULT_PROVIDER = "openai"


class CompletionJSONError(ValueError):
    """The model response was empty or not valid JSON."""


def _default_model() -> str:
    return f"{DEFAULT_PROVIDER}/{os.environ['OPENAI_MODEL']}"


def _complete(
    prompt: str, model: str, response_format: dict[str, Any] | None = None
) -> str | None:
    started_at = perf_counter()
    logger.info("completion request", model=model, prompt=prompt)

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        choice = response.choices[0]
        content = choice.message.content
        usage = getattr(response, "usage", None)

        logger.info(
            "completion response",
            model=model,
            response=content,
            response_id=getattr(response, "id", None),
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )
        return content
    except Exception as exc:
        logger.error(
            "completion error",
            model=model,
            error_type=type(exc).__name__,
            error=str(exc),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            exc_info=True,
        )
        raise


def completion_default(prompt: str) -> str | None:
    return _complete(prompt, _default_model())


def completion_default_json(
    prompt: str,
    response_format: dict[str, Any] | None = None,
    *,
    model: str | None = None,
) -> Any:
    content = _complete(
        prompt,
        model or _default_model(),
        response_format or {"type": "json_object"},
    )
    if content is None:
        raise CompletionJSONError("empty completion content")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise CompletionJSONError(f"invalid JSON: {exc}") from exc
