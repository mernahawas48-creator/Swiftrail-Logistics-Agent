"""
SwiftrailAgent -- the MCP client side of this lab.

Covers, end to end:

  1. THE HANDSHAKE (initialize/initialized): connect() opens the transport,
     declares the client's own capabilities (elicitation + sampling support),
     and calls session.initialize(). We store the server's declared
     capabilities from the InitializeResult and gate everything else on
     them via `supports()` -- e.g. we only attempt resources/read if the
     server actually declared a resources capability, instead of assuming
     it and getting a protocol error.

  2. TOOL DISCOVERY + CALLING: discover_tools() calls tools/list and returns
     the live schema for every tool currently exposed (which changes at
     runtime -- see #4). call_tool() invokes one by name/arguments.

  3. ELICITATION HANDLING: `elicitation_callback` is the function the
     ClientSession invokes whenever the *server* calls elicitation/create
     mid-tool-call. This genuinely pauses the agent -- it blocks on
     `input()` and prints the server's question and schema to the terminal
     -- rather than auto-answering or silently proceeding. Nothing else in
     this file can run until a human types a response here.

  4. REACTING TO tools/list_changed: `_on_message` is registered as the
     ClientSession's generic message handler. When it sees a
     notifications/tools/list_changed message, it does NOT poll or guess --
     it flags the tool list as stale so the next discover_tools() call
     re-fetches the live set (demo.py shows this by calling authenticate,
     then immediately re-listing tools and printing the diff).

  5. SAMPLING: `sampling_callback` is what actually answers
     sampling/createMessage requests using the AGENT's own model (not the
     server's), per the sampling capability the agent declared in
     connect(). If MISTRAL_API_KEY is set it makes a real completion;
     otherwise it returns a clearly-labeled canned response so the demo
     still runs offline.

NOTE ON THE SDK: this targets the official `mcp` Python SDK's ClientSession
API. If your installed SDK version names things slightly differently
(message_handler vs. a subclassed session, etc.), the comments below flag
the exact spots to adjust.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageResult, ElicitResult, TextContent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = PROJECT_ROOT / "mcp_server"
SERVER_FILE = SERVER_DIR / "server.py"


class SwiftrailAgent:
    def __init__(self, transport: str, http_url: str | None = None):
        self.transport = transport
        self.http_url = http_url
        self.session: ClientSession | None = None
        self.server_capabilities = None
        self.server_info = None
        self.tool_list_dirty = True
        self._stack = AsyncExitStack()

    async def connect(self):
        if self.transport == "stdio":
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SERVER_DIR)
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(SERVER_FILE)],
                cwd=str(SERVER_DIR),
                env=env,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif self.transport == "http":
            if not self.http_url:
                raise ValueError("HTTP transport requires --url")
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(self.http_url)
            )
        else:
            raise ValueError(f"Unknown transport: {self.transport}")

        self.session = await self._stack.enter_async_context(
            ClientSession(
                read,
                write,
                elicitation_callback=self._elicitation_callback,
                sampling_callback=self._sampling_callback,
                message_handler=self._on_message,
            )
        )

        init_result = await self.session.initialize()
        self.server_capabilities = init_result.capabilities
        self.server_info = init_result.serverInfo

        tools_capability = getattr(self.server_capabilities, "tools", None)
        print("=" * 68)
        print("HANDSHAKE COMPLETE")
        print(f"Server: {self.server_info.name}")
        print(f"Protocol: {init_result.protocolVersion}")
        print(f"Tools capability: {tools_capability}")
        print(
            "tools.listChanged declared: "
            f"{bool(getattr(tools_capability, 'listChanged', False))}"
        )
        print(f"Resources: {getattr(self.server_capabilities, 'resources', None)}")
        print(f"Prompts: {getattr(self.server_capabilities, 'prompts', None)}")
        print("=" * 68)
        return init_result

    def supports(self, capability_name: str) -> bool:
        return bool(getattr(self.server_capabilities, capability_name, None))

    def supports_tool_list_changes(self) -> bool:
        tools = getattr(self.server_capabilities, "tools", None)
        return bool(getattr(tools, "listChanged", False))

    async def discover_tools(self):
        if not self.supports("tools"):
            print("[fallback] Server did not declare a tools capability.")
            return []
        result = await self.session.list_tools()
        self.tool_list_dirty = False
        return result.tools

    async def call_tool(self, name: str, arguments: dict, progress_callback=None):
        if not self.supports("tools"):
            raise RuntimeError("Server did not declare tool support.")
        if progress_callback is not None:
            return await self.session.call_tool(
                name,
                arguments,
                progress_callback=progress_callback,
            )
        return await self.session.call_tool(name, arguments)

    @staticmethod
    def decode_tool_result(result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        content = getattr(result, "content", None) or []
        if not content:
            return None
        text = getattr(content[0], "text", str(content[0]))
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    async def read_resource(self, uri: str):
        if not self.supports("resources"):
            print(f"[fallback] Resources are unavailable; skipped {uri}.")
            return None
        return await self.session.read_resource(uri)

    async def list_prompts(self):
        if not self.supports("prompts"):
            print("[fallback] Prompts are unavailable.")
            return []
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, name: str, arguments: dict):
        if not self.supports("prompts"):
            return None
        return await self.session.get_prompt(name, arguments)

    async def _elicitation_callback(self, context: RequestContext, params):
        print("\n" + "!" * 68)
        print("SERVER PAUSED THE CALL: elicitation/create")
        print(params.message)
        schema_props = (params.requestedSchema or {}).get("properties", {})
        answers: dict[str, Any] = {}
        for field_name, field_schema in schema_props.items():
            description = field_schema.get("description", "")
            raw = input(
                f"{field_name} ({field_schema.get('type', 'string')}) "
                f"- {description}: "
            )
            if field_schema.get("type") == "boolean":
                answers[field_name] = raw.strip().lower() in {
                    "y",
                    "yes",
                    "true",
                    "1",
                }
            else:
                answers[field_name] = raw

        confirm = input("Submit this human response? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            return ElicitResult(action="decline")
        return ElicitResult(action="accept", content=answers)

    async def _sampling_callback(self, context: RequestContext, params):
        prompt_text = ""
        if params.messages:
            last_content = params.messages[-1].content
            prompt_text = getattr(last_content, "text", str(last_content))

        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            try:
                from langchain_mistralai import ChatMistralAI
            except ImportError as exc:
                raise RuntimeError(
                    "Mistral LangChain integration is not installed."
                ) from exc

            model_name = os.environ.get(
                "MISTRAL_MODEL",
                "mistral-small-latest",
            )
            model = ChatMistralAI(
                model=model_name,
                api_key=api_key,
                temperature=0.1,
                max_tokens=params.maxTokens or 200,
                max_retries=2,
            )
            response = await asyncio.to_thread(model.invoke, prompt_text)
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                content = getattr(response, "content", "")
                text = content if isinstance(content, str) else str(content)
        else:
            text = (
                "[offline sampling fallback] Review active holds and high "
                "balance-to-limit accounts first. Prompt excerpt: "
                + prompt_text[:180]
            )
            model_name = "swiftrail-offline-sampling-fallback"

        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=text),
            model=model_name,
        )

    async def _on_message(self, message):
        method = getattr(message, "method", None)
        if method is None and hasattr(message, "root"):
            method = getattr(message.root, "method", None)
        if method == "notifications/tools/list_changed":
            if self.supports_tool_list_changes():
                print("\n>>> notifications/tools/list_changed received")
                self.tool_list_dirty = True
            else:
                print(
                    "\n>>> Warning: received tools/list_changed although the "
                    "server did not declare listChanged=true."
                )

    async def close(self):
        await self._stack.aclose()


async def _smoke_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--url", default=None)
    args = parser.parse_args()

    agent = SwiftrailAgent(args.transport, args.url)
    try:
        await agent.connect()
        tools = await agent.discover_tools()
        print("\nDiscovered tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
