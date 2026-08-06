import pytest

from deepresearch.config import Settings
from deepresearch.tools import database as db
from deepresearch.tools.database import DatabaseTool, DatabaseUnavailable, QueryResult


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("NLQUERY_MCP_COMMAND", "python")
    monkeypatch.setenv("NLQUERY_MCP_ARGS", "../NLQuery-agent/mcp_server.py")
    return Settings.from_env()


async def test_mock_mode_needs_no_server(settings):
    result = await DatabaseTool(settings, mock=True).query("how many artists?")
    assert isinstance(result, QueryResult)
    assert "query_database" in result.answer


async def test_query_carries_the_question_through(settings):
    result = await DatabaseTool(settings, mock=True).query("top 5 artists")
    assert result.question == "top 5 artists"


async def test_result_names_its_source_for_citation(settings):
    result = await DatabaseTool(settings, mock=True).query("x")
    assert "database" in result.cite()


async def test_schema_and_tables_reach_their_tools(settings):
    tool = DatabaseTool(settings, mock=True)
    assert "get_schema" in await tool.schema("northwind")
    assert "list_tables" in await tool.tables()
    assert "list_available_databases" in await tool.databases()


async def test_mock_session_is_a_no_op(settings):
    tool = DatabaseTool(settings, mock=True)
    async with tool.session():
        assert "query_database" in (await tool.query("x")).answer


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
