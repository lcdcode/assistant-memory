#!/usr/bin/env python3
"""
CLI client for the assistant-memory MCP server (used by Hermes). Works on Debian and Termux.

Usage:
  mcp_memory.py store content="..." category="decision" tags='["tag1","tag2"]'
  mcp_memory.py retrieve query="..." max_results=3
  mcp_memory.py stats

The server runs under the first interpreter found of: $ASSISTANT_MEMORY_PYTHON,
the repo venv (venv/bin/python), or the interpreter running this script.
"""

import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

REPO_DIR = Path(__file__).resolve().parent
SERVER_PATH = REPO_DIR / "src" / "server.py"
VENV_PYTHON_RELPATH = Path("venv") / "bin" / "python"
PYTHON_OVERRIDE_VAR = "ASSISTANT_MEMORY_PYTHON"

EXIT_OK = 0
EXIT_TOOL_ERROR = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """Invalid command line."""


class ToolCallError(Exception):
    """The MCP server reported a failed tool call."""


def _parse_fraction(raw):
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"expected a number between 0 and 1, got {raw!r}")
    return value


def _parse_positive_int(raw):
    value = int(raw)
    if value < 1:
        raise ValueError(f"expected a positive integer, got {raw!r}")
    return value


def _parse_bool(raw):
    lowered = raw.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError(f"expected true or false, got {raw!r}")


def _parse_tags(raw):
    try:
        tags = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"expected a JSON list of strings, got {raw!r}") from e
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"expected a JSON list of strings, got {raw!r}")
    return tags


@dataclass(frozen=True)
class CommandSpec:
    tool_name: str
    params: dict
    required: tuple = ()
    defaults: dict = field(default_factory=dict)


COMMANDS = {
    "store": CommandSpec(
        tool_name="store_memory",
        params={
            "content": str,
            "category": str,
            "summary": str,
            "tags": _parse_tags,
            "importance": _parse_fraction,
            "project": str,
        },
        required=("content",),
        defaults={"category": "discovery", "importance": 0.5},
    ),
    "retrieve": CommandSpec(
        tool_name="retrieve_memories",
        params={
            "query": str,
            "max_results": _parse_positive_int,
            "category": str,
            "min_similarity": _parse_fraction,
            "tags": _parse_tags,
            "project": str,
            "include_forgotten": _parse_bool,
            "show_ids": _parse_bool,
        },
        required=("query",),
        defaults={"max_results": 5},
    ),
    "stats": CommandSpec(tool_name="memory_stats", params={}),
}


def parse_cli_args(command, raw_args):
    """Turn key=value arguments into typed tool arguments for the given command."""
    spec = COMMANDS.get(command)
    if spec is None:
        raise UsageError(f"unknown command {command!r}; expected one of: {', '.join(COMMANDS)}")

    arguments = dict(spec.defaults)
    for raw_arg in raw_args:
        key, separator, raw_value = raw_arg.partition("=")
        if not separator:
            raise UsageError(f"{command}: expected key=value, got {raw_arg!r}")
        parse_value = spec.params.get(key)
        if parse_value is None:
            allowed = ", ".join(spec.params) or "none"
            raise UsageError(f"{command}: unknown parameter {key!r}; allowed: {allowed}")
        try:
            arguments[key] = parse_value(raw_value)
        except ValueError as e:
            raise UsageError(f"{command}: invalid value for {key}: {e}") from e

    missing = [name for name in spec.required if name not in arguments]
    if missing:
        raise UsageError(f"{command}: missing required parameter(s): {', '.join(missing)}")
    return arguments


def resolve_python(environ=None, repo_dir=REPO_DIR):
    """Pick the server interpreter. The venv path is kept unresolved so the venv stays active."""
    environ = os.environ if environ is None else environ
    override = environ.get(PYTHON_OVERRIDE_VAR)
    if override:
        if not Path(override).is_file():
            raise UsageError(f"{PYTHON_OVERRIDE_VAR}={override!r} is not a file")
        return override

    venv_python = Path(repo_dir) / VENV_PYTHON_RELPATH
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def result_text(result):
    """Join the text blocks of a CallToolResult; raise ToolCallError if it reports an error."""
    text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
    if result.is_error:
        raise ToolCallError(text)
    return text


async def call_tool(server, tool_name, arguments):
    """Call one tool. `server` is anything mcp.Client accepts (stdio params, or an MCPServer)."""
    async with Client(server) as client:
        result = await client.call_tool(tool_name, arguments)
    # Raised outside the Client context: anyio would wrap it in an ExceptionGroup.
    return result_text(result)


def stdio_server_params(python):
    return StdioServerParameters(
        command=python, args=[str(SERVER_PATH)], env=get_default_environment(),
    )


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return EXIT_OK

    command = argv[1]
    try:
        arguments = parse_cli_args(command, argv[2:])
        python = resolve_python()
    except UsageError as e:
        print(f"mcp_memory: {e}", file=sys.stderr)
        return EXIT_USAGE

    tool_name = COMMANDS[command].tool_name
    try:
        output = asyncio.run(call_tool(stdio_server_params(python), tool_name, arguments))
    except ToolCallError as e:
        message = f"mcp_memory: {e} (server interpreter: {python})"
        print(message, file=sys.stderr)
        return EXIT_TOOL_ERROR

    print(output)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv))
