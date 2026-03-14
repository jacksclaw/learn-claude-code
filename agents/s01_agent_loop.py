#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
from urllib import request, error

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
TOOL_SCHEMA = {
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
ANTHROPIC_TOOLS = [TOOL_SCHEMA]
OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_SCHEMA["name"],
        "description": TOOL_SCHEMA["description"],
        "parameters": TOOL_SCHEMA["input_schema"],
    },
}]
OLLAMA_TOOLS = OPENAI_TOOLS


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
    forced = os.getenv("S01_BACKEND")
    if forced in {"anthropic", "openai", "ollama"}:
        return forced
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    if os.getenv("OPENAI_API_KEY") or load_codex_openai_key():
        return "openai"
    return "anthropic"


BACKEND = detect_backend()

if BACKEND == "anthropic":
    if os.getenv("ANTHROPIC_BASE_URL"):
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    anthropic_client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
    ANTHROPIC_MODEL = os.environ["MODEL_ID"]
elif BACKEND == "ollama":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nemotron-3-super:latest")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
else:
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY") or load_codex_openai_key()
    if not openai_api_key:
        raise RuntimeError(
            "OpenAI backend selected, but no OPENAI_API_KEY was found in the "
            "environment or ~/.codex/auth.json"
        )
    openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # Detach tool commands from the REPL stdin so they cannot consume the
        # next prompt input or block waiting for terminal input.
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def ollama_chat(messages: list, tools: list) -> dict:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": messages,
        "tools": tools,
    }
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc


# -- The core pattern: a while loop that calls tools until the model stops --
def anthropic_agent_loop(messages: list) -> str:
    while True:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL, system=SYSTEM, messages=messages,
            tools=ANTHROPIC_TOOLS, max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            texts = [block.text for block in response.content
                     if getattr(block, "type", None) == "text"]
            return "\n".join(texts).strip()
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})


def openai_agent_loop(messages: list) -> str:
    while True:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        message = response.choices[0].message
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        messages.append(assistant_message)
        if not message.tool_calls:
            return message.content or ""
        for tool_call in message.tool_calls:
            if tool_call.function.name != "bash":
                continue
            args = json.loads(tool_call.function.arguments or "{}")
            command = args["command"]
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


def ollama_agent_loop(messages: list) -> str:
    while True:
        response = ollama_chat(messages, OLLAMA_TOOLS)
        message = response["message"]
        tool_calls = message.get("tool_calls") or []
        assistant_message = {
            "role": "assistant",
            "content": message.get("content", ""),
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        if not tool_calls:
            return message.get("content", "")
        for tool_call in tool_calls:
            fn = tool_call["function"]
            if fn["name"] != "bash":
                continue
            command = fn["arguments"]["command"]
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            tool_message = {
                "role": "tool",
                "content": output,
                "tool_name": "bash",
            }
            if tool_call.get("id"):
                tool_message["tool_call_id"] = tool_call["id"]
            messages.append(tool_message)


def agent_loop(messages: list) -> str:
    if BACKEND == "ollama":
        return ollama_agent_loop(messages)
    if BACKEND == "openai":
        return openai_agent_loop(messages)
    return anthropic_agent_loop(messages)


if __name__ == "__main__":
    history = []
    if BACKEND in {"openai", "ollama"}:
        history.append({"role": "system", "content": SYSTEM})
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip():
            print()
            continue
        if query.strip().lower() in ("q", "exit"):
            break
        history.append({"role": "user", "content": query})
        final_text = agent_loop(history)
        if final_text:
            print(final_text)
        print()
