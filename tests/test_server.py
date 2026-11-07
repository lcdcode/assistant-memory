"""Protocol-level tests for the MCP server: tool registration and error reporting.

Run with the project venv:  ./venv/bin/python -m unittest discover -s tests
Tools are called in-process through mcp.Client. Postgres and Ollama are replaced with fakes,
and PG_* points at a nonexistent database so an unpatched call can never reach real data.
"""

import os
import sys
import unittest
from unittest import mock

os.environ.update({
    "PG_HOST": "127.0.0.1",
    "PG_PORT": "1",
    "PG_DATABASE": "assistant_memory_test_unused",
})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx  # noqa: E402
from mcp import Client  # noqa: E402

import server  # noqa: E402

EXPECTED_TOOLS = {
    "store_memory", "retrieve_memories", "list_memories",
    "delete_memory", "memory_stats", "memory_forgetting_cycle",
}
VALID_UUID = "00000000-0000-0000-0000-000000000000"


def _ollama_returning(payload, status_code=200):
    """Patch get_embedding's HTTP client to answer every request with the given JSON."""
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code, json=payload))
    real_client = httpx.AsyncClient
    return mock.patch.object(
        server.httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw)
    )


class ServerTestCase(unittest.IsolatedAsyncioTestCase):

    async def call(self, tool_name, arguments=None):
        async with Client(server.mcp) as client:
            result = await client.call_tool(tool_name, arguments or {})
        text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
        return result.is_error, text


class ToolRegistrationTest(ServerTestCase):

    async def test_all_tools_registered(self):
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
        self.assertEqual({tool.name for tool in tools.tools}, EXPECTED_TOOLS)


class ValidationErrorTest(ServerTestCase):
    """Bad input is reported with is_error=True before any backend is touched."""

    def setUp(self):
        self.pool_patch = mock.patch.object(server, "get_pool", side_effect=AssertionError)
        self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)

    async def test_store_invalid_category(self):
        is_error, text = await self.call("store_memory", {"content": "x", "category": "nope"})
        self.assertTrue(is_error)
        self.assertIn("Invalid category 'nope'", text)

    async def test_store_content_too_long(self):
        content = "x" * (server.MAX_CONTENT_CHARS + 1)
        arguments = {"content": content, "category": "learning"}
        is_error, text = await self.call("store_memory", arguments)
        self.assertTrue(is_error)
        self.assertIn("Content too long", text)

    async def test_list_invalid_category(self):
        is_error, text = await self.call("list_memories", {"category": "nope"})
        self.assertTrue(is_error)
        self.assertIn("Invalid category", text)

    async def test_delete_invalid_uuid(self):
        is_error, text = await self.call("delete_memory", {"memory_id": "not-a-uuid"})
        self.assertTrue(is_error)
        self.assertIn("valid UUID", text)

    async def test_wrong_argument_type_rejected(self):
        is_error, _ = await self.call("list_memories", {"limit": "many"})
        self.assertTrue(is_error)


class BackendErrorTest(ServerTestCase):
    """Expected backend failures become actionable tool errors, not fake successes."""

    async def test_database_unreachable(self):
        refused = ConnectionRefusedError(111, "Connection refused")
        with mock.patch.object(server, "get_pool", side_effect=refused):
            is_error, text = await self.call("memory_stats")
        self.assertTrue(is_error)
        self.assertIn("Error executing tool memory_stats: ConnectionRefusedError", text)

    async def test_ollama_http_error(self):
        with _ollama_returning({"error": "boom"}, status_code=500):
            is_error, text = await self.call("retrieve_memories", {"query": "q"})
        self.assertTrue(is_error)
        self.assertIn("Error executing tool retrieve_memories: HTTPStatusError", text)

    async def test_ollama_missing_embedding(self):
        with _ollama_returning({}):
            is_error, text = await self.call(
                "store_memory", {"content": "x", "category": "learning"}
            )
        self.assertTrue(is_error)
        self.assertIn("ollama pull", text)

    async def test_delete_not_found(self):
        connection = mock.AsyncMock()
        connection.execute.return_value = "DELETE 0"
        pool = mock.MagicMock()
        pool.acquire.return_value.__aenter__.return_value = connection
        with mock.patch.object(server, "get_pool", mock.AsyncMock(return_value=pool)):
            is_error, text = await self.call("delete_memory", {"memory_id": VALID_UUID})
        self.assertTrue(is_error)
        self.assertIn("not found", text)


class UnexpectedErrorTest(ServerTestCase):

    async def test_bug_reported_without_internal_details(self):
        bug = RuntimeError("secret internal detail")
        with mock.patch.object(server, "get_pool", side_effect=bug), \
                self.assertLogs(level="ERROR"):
            is_error, text = await self.call("memory_stats")
        self.assertTrue(is_error)
        self.assertNotIn("secret internal detail", text)


if __name__ == "__main__":
    unittest.main()
