import pytest

from deepresearch.config import Settings
from deepresearch.tools import database as db
from deepresearch.tools.database import DatabaseTool, DatabaseUnavailable


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("NLQUERY_MCP_COMMAND", "python")
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "../NLQuery-agent/mcp_server.py")
    return Settings.from_env()


async def test_mock_mode_needs_no_server(settings):
    answer = await DatabaseTool(settings, mock=True).query("top 5 artists")
    assert "query_database" in answer
    assert "top 5 artists" in answer


async def test_schema_and_databases_reach_their_tools(settings):
    tool = DatabaseTool(settings, mock=True)
    assert "get_schema" in await tool.schema("northwind")
    assert "list_available_databases" in await tool.databases()


async def test_mock_session_is_a_no_op(settings):
    tool = DatabaseTool(settings, mock=True)
    async with tool.session():
        assert "query_database" in await tool.query("x")


def test_missing_args_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "")
    with pytest.raises(DatabaseUnavailable, match="NLQUERY_MCP_ARGS"):
        db._server_params(Settings.from_env())


def test_missing_script_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "no/such/server.py")
    with pytest.raises(DatabaseUnavailable, match="not found"):
        db._server_params(Settings.from_env())


def test_relative_paths_resolve_against_the_project_not_the_cwd(
    monkeypatch, tmp_path
):
    """A relative path in .env must not break when run from another directory."""
    script = db.ROOT / "runs" / "fake_server.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text("")
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "runs/fake_server.py")
    monkeypatch.chdir(tmp_path)
    try:
        params = db._server_params(Settings.from_env())
        assert params.args[0] == str(script.resolve())
    finally:
        script.unlink()


def test_error_results_raise_rather_than_return_text():
    result = type(
        "R",
        (),
        {
            "content": [type("B", (), {"type": "text", "text": "boom"})()],
            "isError": True,
        },
    )()
    with pytest.raises(DatabaseUnavailable, match="boom"):
        db._text_of(result)


def test_text_blocks_are_joined():
    blocks = [
        type("B", (), {"type": "text", "text": "one"})(),
        type("B", (), {"type": "text", "text": "two"})(),
    ]
    result = type("R", (), {"content": blocks, "isError": False})()
    assert db._text_of(result) == "one\ntwo"


def test_non_text_blocks_are_skipped():
    blocks = [
        type("B", (), {"type": "image", "text": "ignored"})(),
        type("B", (), {"type": "text", "text": "kept"})(),
    ]
    result = type("R", (), {"content": blocks, "isError": False})()
    assert db._text_of(result) == "kept"


def test_nlquery_models_are_passed_to_the_server(monkeypatch):
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "../NLQuery-agent/mcp_server.py")
    monkeypatch.setenv("NLQUERY_FAST_MODEL", "fast-x")
    monkeypatch.setenv("NLQUERY_SMART_MODEL", "smart-y")
    params = db._server_params(Settings.from_env())
    assert params.env == {"LLM_FAST_MODEL": "fast-x", "LLM_SMART_MODEL": "smart-y"}


def test_unset_models_leave_the_server_env_alone(monkeypatch):
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "../NLQuery-agent/mcp_server.py")
    monkeypatch.setenv("NLQUERY_FAST_MODEL", "")
    monkeypatch.setenv("NLQUERY_SMART_MODEL", "")
    assert db._server_params(Settings.from_env()).env is None


class _FakeSession:
    """Stands in for the MCP client: wraps errors the way its task groups do."""

    def __init__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, kind, exc, tb):
        if exc is not None:
            raise ExceptionGroup("unhandled errors in a TaskGroup", [exc])

    async def initialize(self):
        pass


def _fake_mcp(monkeypatch, start_error=None):
    import contextlib

    import mcp
    import mcp.client.stdio

    @contextlib.asynccontextmanager
    async def stdio_client(params):
        if start_error:
            raise start_error
        async with _FakeSession():
            yield "read", "write"

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", _FakeSession)
    monkeypatch.setattr(db, "_server_params", lambda settings: None)


async def test_an_error_inside_the_session_comes_back_as_itself(settings, monkeypatch):
    """A budget stop mid-run must not look like a dead database and trigger a rerun."""
    _fake_mcp(monkeypatch)
    with pytest.raises(KeyError):
        async with DatabaseTool(settings).session():
            raise KeyError("over budget")


async def test_a_server_that_will_not_start_is_unavailable(settings, monkeypatch):
    _fake_mcp(monkeypatch, start_error=OSError("no python"))
    with pytest.raises(DatabaseUnavailable, match="no python"):
        async with DatabaseTool(settings).session():
            pass
