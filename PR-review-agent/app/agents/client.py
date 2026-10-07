"""OpenRouter chat client with structured responses and bounded tool execution."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@asynccontextmanager
async def _http_client(
    client: httpx.AsyncClient | None,
    timeout: httpx.Timeout,
    default_headers: dict[str, str],
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an injected HTTP client or a request-scoped client."""
    if client is not None:
        client.headers.update(default_headers)
        yield client
    else:
        async with httpx.AsyncClient(timeout=timeout, headers=default_headers) as owned_client:
            yield owned_client


class LLMError(RuntimeError):
    """Raised when the configured language-model service returns an unusable response."""

    pass


class LLMClient:
    """Asynchronous client for structured chat completions and controlled tool calls."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for LLM review")
        self.settings = settings
        self.api_key = settings.openrouter_api_key.get_secret_value()
        self.http_client = http_client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[T, int]:
        """Validate a chat completion against the requested Pydantic response schema."""
        conversation = messages or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tokens_used = 0
        tool_output_budget = 24_000
        tool_rounds = 0
        tool_count = 0
        while tools and tool_rounds < 3 and tool_count < 6:
            response = await self._post(
                {
                    "model": model or self.settings.llm_model,
                    "messages": conversation,
                    "tools": tools,
                    "tool_choice": "auto",
                }
            )
            tokens_used += _total_tokens(response)
            message = response["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                if message.get("content"):
                    conversation.append(message)
                break
            conversation.append(message)
            for call in calls[: 6 - tool_count]:
                tool_count += 1
                call_id = call.get("id", "")
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")
                try:
                    result = await self._execute_tool(name, arguments)
                except (ValueError, KeyError, ValidationError) as error:
                    result = {"error": f"Invalid tool request: {error}"}
                if tool_output_budget <= 0:
                    result = {"error": "Tool output budget exhausted"}
                    serialized = json.dumps(result)
                else:
                    serialized = json.dumps(result, ensure_ascii=True)
                    output_limit = min(6000, tool_output_budget)
                    if len(serialized) > output_limit:
                        serialized = json.dumps(
                            {
                                "truncated": True,
                                "excerpt": serialized[: max(0, output_limit - 100)],
                            },
                            ensure_ascii=True,
                        )
                    tool_output_budget -= len(serialized)
                serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e")
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"UNTRUSTED_TOOL_RESULT {serialized}",
                    }
                )
            tool_rounds += 1
        response = await self._post(
            {
                "model": model or self.settings.llm_model,
                "messages": conversation,
                "response_format": {"type": "json_object"},
            }
        )
        choice = response.get("choices", [{}])[0]
        raw = choice.get("message", {}).get("content")
        if not isinstance(raw, str):
            raise LLMError("LLM response did not contain structured JSON content")
        try:
            parsed = response_model.model_validate_json(raw)
        except ValidationError as error:
            logger.warning("LLM returned schema-invalid JSON: %s", error)
            raise LLMError("LLM response did not match the required schema") from error
        tokens_used += _total_tokens(response)
        return parsed, tokens_used

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a chat-completion request and normalize transport or response errors."""
        timeout = httpx.Timeout(90, connect=10)
        for attempt in range(4):
            async with _http_client(
                self.http_client, timeout, self._attribution_headers()
            ) as client:
                response = await client.post(
                    f"{self.settings.openrouter_base_url.rstrip('/')}/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
            if response.status_code < 400:
                try:
                    result = response.json()
                except json.JSONDecodeError as error:
                    raise LLMError("LLM returned invalid JSON transport data") from error
                if not isinstance(result, dict):
                    raise LLMError("LLM returned an unexpected response object")
                return cast(dict[str, Any], result)
            retryable = response.status_code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == 3:
                logger.warning("LLM request failed with HTTP %d", response.status_code)
                raise LLMError(f"LLM API returned HTTP {response.status_code}")
            await asyncio.sleep(min(2**attempt, 20))
        raise LLMError("LLM retry loop exited unexpectedly")

    def _attribution_headers(self) -> dict[str, str]:
        """Return optional application attribution headers supported by OpenRouter."""
        headers = {"X-OpenRouter-Title": self.settings.openrouter_app_name}
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        return headers

    @staticmethod
    async def _execute_tool(name: str, arguments: str) -> object:
        """Run a validated tool call with the active toolbox and enforce call limits."""
        from app.agents.tools import execute_tool

        return await execute_tool(name, arguments)


def _total_tokens(response: dict[str, Any]) -> int:
    """Sum token counts reported by the completion response."""
    return int(response.get("usage", {}).get("total_tokens", 0))
