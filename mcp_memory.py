#!/usr/bin/env python3
"""
MCP memory client for Hermes — store & retrieve via the assistant-memory MCP server.
Usage:
  python3 mcp_memory.py store content="..." category="decision" tags='["tag1","tag2"]'
  python3 mcp_memory.py retrieve query="..." max_results=3
  python3 mcp_memory.py stats
"""
import asyncio, json, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/projects/assistant-memory/src')

from mcp import StdioServerParameters, stdio_client
from mcp.client.stdio import get_default_environment
from mcp.client.session import ClientSession

SERVER = "/data/data/com.termux/files/home/projects/assistant-memory/src/server.py"

async def call_tool(name, args=None):
    args = args or {}
    server_params = StdioServerParameters(
        command="python3", args=[SERVER], env=get_default_environment(),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return result.content[0].text if result.content else ""

async def store(content, category="discovery", summary=None, tags=None, importance=0.5, project=None):
    kwargs = {"content": content, "category": category, "importance": importance}
    if summary: kwargs["summary"] = summary
    if tags: kwargs["tags"] = tags
    if project: kwargs["project"] = project
    return await call_tool("store_memory", kwargs)

async def retrieve(query, max_results=5, category=None, tags=None, project=None):
    kwargs = {"query": query, "max_results": max_results}
    if category: kwargs["category"] = category
    if tags: kwargs["tags"] = tags
    if project: kwargs["project"] = project
    return await call_tool("retrieve_memories", kwargs)

async def stats():
    return await call_tool("memory_stats")

async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    kwargs = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # Parse JSON values
            try: kwargs[k] = json.loads(v)
            except: kwargs[k] = v
    if cmd == "store":
        print(await store(**kwargs))
    elif cmd == "retrieve":
        print(await retrieve(**kwargs))
    elif cmd == "stats":
        print(await stats())
    else:
        print(f"Unknown command: {cmd}")

asyncio.run(main())
