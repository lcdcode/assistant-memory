"""Unit tests for the mcp_memory CLI client: argument parsing and interpreter selection.

Run with the project venv:  ./venv/bin/python -m unittest discover -s tests
Needs the venv because mcp_memory imports the mcp package. No server or database is used.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mcp_memory  # noqa: E402
from mcp_memory import (  # noqa: E402
    PYTHON_OVERRIDE_VAR,
    VENV_PYTHON_RELPATH,
    UsageError,
    parse_cli_args,
    resolve_python,
)


class ParseCliArgsTest(unittest.TestCase):

    def test_store_applies_defaults(self):
        args = parse_cli_args("store", ["content=hello"])
        self.assertEqual(args, {"content": "hello", "category": "discovery", "importance": 0.5})

    def test_values_may_contain_equals_signs(self):
        args = parse_cli_args("store", ["content=a=b=c"])
        self.assertEqual(args["content"], "a=b=c")

    def test_json_looking_content_stays_a_string(self):
        # The original client ran json.loads on every value, turning "42" into an int.
        for raw in ("42", "true", '["x"]', "null"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_cli_args("store", [f"content={raw}"])["content"], raw)

    def test_tags_parse_as_json_list(self):
        args = parse_cli_args("store", ["content=x", 'tags=["a","b"]'])
        self.assertEqual(args["tags"], ["a", "b"])

    def test_invalid_tags_rejected(self):
        for raw in ("a,b", '"a"', "[1, 2]", '{"a": 1}'):
            with self.subTest(raw=raw), self.assertRaises(UsageError):
                parse_cli_args("store", ["content=x", f"tags={raw}"])

    def test_importance_must_be_fraction(self):
        args = parse_cli_args("store", ["content=x", "importance=0.9"])
        self.assertEqual(args["importance"], 0.9)
        for raw in ("1.5", "-0.1", "nan", "inf", "high"):
            with self.subTest(raw=raw), self.assertRaises(UsageError):
                parse_cli_args("store", ["content=x", f"importance={raw}"])

    def test_retrieve_types(self):
        args = parse_cli_args("retrieve", ["query=q", "max_results=3", "show_ids=true"])
        self.assertEqual(args, {"query": "q", "max_results": 3, "show_ids": True})

    def test_retrieve_rejects_bad_max_results(self):
        for raw in ("0", "-1", "three"):
            with self.subTest(raw=raw), self.assertRaises(UsageError):
                parse_cli_args("retrieve", ["query=q", f"max_results={raw}"])

    def test_bool_rejects_garbage(self):
        with self.assertRaises(UsageError):
            parse_cli_args("retrieve", ["query=q", "show_ids=maybe"])

    def test_stats_takes_no_params(self):
        self.assertEqual(parse_cli_args("stats", []), {})
        with self.assertRaises(UsageError):
            parse_cli_args("stats", ["verbose=true"])

    def test_missing_required(self):
        with self.assertRaisesRegex(UsageError, "content"):
            parse_cli_args("store", ["category=decision"])

    def test_unknown_parameter(self):
        with self.assertRaisesRegex(UsageError, "unknown parameter"):
            parse_cli_args("store", ["content=x", "user_id=someone"])

    def test_argument_without_equals(self):
        with self.assertRaisesRegex(UsageError, "key=value"):
            parse_cli_args("store", ["content"])

    def test_unknown_command(self):
        with self.assertRaisesRegex(UsageError, "unknown command"):
            parse_cli_args("delete", [])


class ResolvePythonTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_venv_python(self):
        venv_python = self.repo_dir / VENV_PYTHON_RELPATH
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(sys.executable)
        return venv_python

    def test_override_wins(self):
        self._make_venv_python()
        environ = {PYTHON_OVERRIDE_VAR: sys.executable}
        self.assertEqual(resolve_python(environ, self.repo_dir), sys.executable)

    def test_override_must_exist(self):
        environ = {PYTHON_OVERRIDE_VAR: str(self.repo_dir / "no-such-python")}
        with self.assertRaisesRegex(UsageError, PYTHON_OVERRIDE_VAR):
            resolve_python(environ, self.repo_dir)

    def test_venv_used_without_resolving_symlink(self):
        # Resolving the symlink would point at the system python and bypass the venv.
        venv_python = self._make_venv_python()
        self.assertEqual(resolve_python({}, self.repo_dir), str(venv_python))

    def test_falls_back_to_current_interpreter(self):
        self.assertEqual(resolve_python({}, self.repo_dir), sys.executable)


class MainTest(unittest.TestCase):

    def _run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = mcp_memory.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help(self):
        code, out, _ = self._run_main(["mcp_memory.py", "--help"])
        self.assertEqual(code, mcp_memory.EXIT_OK)
        self.assertIn("Usage:", out)

    def test_usage_error_goes_to_stderr(self):
        code, out, err = self._run_main(["mcp_memory.py", "store", "bogus"])
        self.assertEqual(code, mcp_memory.EXIT_USAGE)
        self.assertEqual(out, "")
        self.assertIn("key=value", err)


if __name__ == "__main__":
    unittest.main()
