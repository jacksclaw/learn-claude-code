#!/usr/bin/env python3
"""Shared backend adapter for learn-claude-code lessons."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error, request

from anthropic import Anthropic


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class NormalizedResponse:
    content: list[Any]
    stop_reason: str


def _block_type(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def _tool_id(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("id", ""))
    return str(getattr(block, "id", ""))


def _tool_name(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("name", ""))
    return str(getattr(block, "name", ""))


def _tool_input(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block.get("input", {}))
    return dict(getattr(block, "input", {}) or {})


def load_codex_openai_key() -> Optional[str]:
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("OPENAI_API_KEY")


def detect_backend() -> str:
    forced = (
        os.getenv("LEARN_CC_BACKEND")
        or os.getenv("S01_BACKEND")
        or os.getenv("AGENT_BACKEND")
    )
    if forced in {"anthropic", "openai", "ollama"}:
        return forced
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    if os.getenv("OPENAI_API_KEY") or load_codex_openai_key():
        return "openai"
    return "anthropic"


def _anthropic_tools_to_openai(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        })
    return converted


class _MessagesAPI:
    def __init__(self, backend: "BackendClient"):
        self._backend = backend

    def create(self, **kwargs) -> NormalizedResponse:
        return self._backend.create(**kwargs)


class BackendClient:
    def __init__(self):
        self.backend = detect_backend()
        self.messages = _MessagesAPI(self)

        if self.backend == "anthropic":
            if os.getenv("ANTHROPIC_BASE_URL"):
                os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            self._anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
            self.default_model = os.environ["MODEL_ID"]
            return

        if self.backend == "ollama":
            self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
            self._ollama_model = os.getenv("OLLAMA_MODEL", "nemotron-3-super:latest")
            self._ollama_api_key = os.getenv("OLLAMA_API_KEY", "")
            self.default_model = self._ollama_model
            return

        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY") or load_codex_openai_key()
        if not openai_api_key:
            raise RuntimeError(
                "OpenAI backend selected, but no OPENAI_API_KEY was found in the "
                "environment or ~/.codex/auth.json"
            )
        self._openai = OpenAI(
            api_key=openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 8000,
        system: Optional[str] = None,
    ) -> NormalizedResponse:
        if self.backend == "anthropic":
            return self._anthropic_create(
                model=model or self.default_model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                system=system,
            )
        if self.backend == "ollama":
            return self._ollama_create(
                model=model or self.default_model,
                messages=messages,
                tools=tools,
                system=system,
            )
        return self._openai_create(
            model=model or self.default_model,
            messages=messages,
            tools=tools,
            system=system,
        )

    def _anthropic_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        max_tokens: int,
        system: Optional[str],
    ) -> NormalizedResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._to_anthropic_messages(messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        response = self._anthropic.messages.create(**kwargs)
        return NormalizedResponse(
            content=self._normalize_anthropic_content(response.content),
            stop_reason=response.stop_reason or "end_turn",
        )

    def _openai_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        system: Optional[str],
    ) -> NormalizedResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._to_openai_messages(
                messages,
                system=system,
                include_tool_name=False,
                assistant_tool_args_as_object=False,
            ),
        }
        openai_tools = _anthropic_tools_to_openai(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools
        response = self._openai.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = []
        if message.content:
            content.append(TextBlock(text=message.content))
        for tool_call in message.tool_calls or []:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            content.append(ToolUseBlock(
                id=tool_call.id,
                name=tool_call.function.name,
                input=args,
            ))
        return NormalizedResponse(
            content=content,
            stop_reason="tool_use" if message.tool_calls else "end_turn",
        )

    def _ollama_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        system: Optional[str],
    ) -> NormalizedResponse:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": self._to_openai_messages(
                messages,
                system=system,
                include_tool_name=True,
                assistant_tool_args_as_object=True,
            ),
        }
        ollama_tools = _anthropic_tools_to_openai(tools)
        if ollama_tools:
            payload["tools"] = ollama_tools
        headers = {"Content-Type": "application/json"}
        if self._ollama_api_key:
            headers["Authorization"] = f"Bearer {self._ollama_api_key}"
        req = request.Request(
            f"{self._ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode())
        except error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        message = raw["message"]
        content = []
        if message.get("content"):
            content.append(TextBlock(text=message["content"]))
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call["function"]
            content.append(ToolUseBlock(
                id=tool_call.get("id", ""),
                name=fn["name"],
                input=dict(fn.get("arguments", {})),
            ))
        return NormalizedResponse(
            content=content,
            stop_reason="tool_use" if message.get("tool_calls") else "end_turn",
        )

    def _normalize_anthropic_content(self, content: list[Any]) -> list[Any]:
        normalized = []
        for block in content:
            if getattr(block, "type", None) == "text":
                normalized.append(TextBlock(text=block.text))
            elif getattr(block, "type", None) == "tool_use":
                normalized.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        return normalized

    def _to_anthropic_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    part_type = _block_type(part)
                    if part_type == "text":
                        parts.append({"type": "text", "text": _block_text(part)})
                    elif part_type == "tool_use":
                        parts.append({
                            "type": "tool_use",
                            "id": _tool_id(part),
                            "name": _tool_name(part),
                            "input": _tool_input(part),
                        })
                    elif part_type == "tool_result":
                        parts.append({
                            "type": "tool_result",
                            "tool_use_id": str(part.get("tool_use_id", "")),
                            "content": str(part.get("content", "")),
                        })
                converted.append({"role": message["role"], "content": parts})
            else:
                converted.append({"role": message["role"], "content": str(content)})
        return converted

    def _to_openai_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str],
        include_tool_name: bool,
        assistant_tool_args_as_object: bool,
    ) -> list[dict[str, Any]]:
        converted = []
        if system:
            converted.append({"role": "system", "content": system})
        tool_name_map: dict[str, str] = {}
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role == "assistant":
                if isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        part_type = _block_type(block)
                        if part_type == "text":
                            text = _block_text(block)
                            if text:
                                text_parts.append(text)
                        elif part_type == "tool_use":
                            tool_input = _tool_input(block)
                            tool_call = {
                                "id": _tool_id(block),
                                "type": "function",
                                "function": {
                                    "name": _tool_name(block),
                                    "arguments": (
                                        tool_input if assistant_tool_args_as_object
                                        else json.dumps(tool_input)
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)
                            tool_name_map[tool_call["id"]] = tool_call["function"]["name"]
                    assistant_message = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else "",
                    }
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    converted.append(assistant_message)
                else:
                    converted.append({"role": "assistant", "content": str(content)})
                continue

            if role != "user":
                converted.append({"role": role, "content": str(content)})
                continue

            if not isinstance(content, list):
                converted.append({"role": "user", "content": str(content)})
                continue

            tool_messages = []
            text_parts = []
            for part in content:
                part_type = _block_type(part)
                if part_type == "tool_result":
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": str(part.get("tool_use_id", "")),
                        "content": str(part.get("content", "")),
                    }
                    if include_tool_name:
                        tool_name = tool_name_map.get(tool_message["tool_call_id"])
                        if tool_name:
                            tool_message["tool_name"] = tool_name
                    tool_messages.append(tool_message)
                elif part_type == "text":
                    text = _block_text(part)
                    if text:
                        text_parts.append(text)
            converted.extend(tool_messages)
            if text_parts:
                converted.append({"role": "user", "content": "\n".join(text_parts)})
        return converted


def get_client() -> BackendClient:
    return BackendClient()
