from __future__ import annotations

import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ..config import ROOT, Settings, settings as default_settings


class DatabaseUnavailable(RuntimeError):
    """The MCP server could not be reached. Research continues without it."""


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: str

    def cite(self) -> str:
        return "internal database (nlquery-agent)"


def _server_params(settings: Settings):
    from mcp import StdioServerParameters

    args = shlex.split(settings.nlquery_mcp_args)
    if not args:
        raise DatabaseUnavailable("NLQUERY_MCP_ARGS is not set - add it to .env")

    # Relative paths in .env are relative to this project, not the shell's cwd.
    script = Path(args[0])
    if not script.is_absolute():
        script = (ROOT / script).resolve()
    if not script.is_file():
        raise DatabaseUnavailable(f"MCP server script not found: {script}")

    return StdioServerParameters(
        command=settings.nlquery_mcp_command, args=[str(script), *args[1:]]
    )


class DatabaseTool:
    """Bridge to nlquery-agent over MCP.

    Every call spawns and tears down the server, which costs ~1s. Use
    `session()` to hold one open across several calls.
    """

    def __init__(self, settings: Settings | None = None, *, mock: bool = False) -> None:
        self.settings = settings or default_settings
        self.mock = mock
        self._session = None

    @asynccontextmanager
    async def session(self):
        """Keep one server process open for the duration of the block."""
        if self.mock:
            yield self
            return

        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = _server_params(self.settings)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    try:
                        yield self
                    finally:
                        self._session = None
        except DatabaseUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any startup failure is fatal here
            raise DatabaseUnavailable(f"could not start MCP server: {exc}") from exc

    async def query(self, question: str, database: str = "chinook") -> QueryResult:
        """Answer a question from the database. The reply carries the SQL used."""
        answer = await self._call(
            "query_database", {"question": question, "database": database}
        )
        return QueryResult(question=question, answer=answer)

    async def schema(self, database: str = "chinook") -> str:
        return await self._call("get_schema", {"database": database})

    async def tables(self, database: str = "chinook") -> str:
        return await self._call("list_tables", {"database": database})

    async def databases(self) -> str:
        return await self._call("list_available_databases", {})

    async def _call(self, tool: str, args: dict) -> str:
        if self.mock:
            return f"[mock] {tool}({args})"
        if self._session is None:
            async with self.session():
                return await self._call(tool, args)

        result = await self._session.call_tool(tool, args)
        return _text_of(result)


def _text_of(result) -> str:
    blocks = getattr(result, "content", None) or []
    text = "\n".join(b.text for b in blocks if getattr(b, "type", None) == "text")
    if getattr(result, "isError", False):
        raise DatabaseUnavailable(text or "MCP tool returned an error")
    return text
