from deepresearch.state import (
    Finding,
    SourceIndex,
    SubTask,
    render_findings,
)


def test_brief_states_the_objective():
    task = SubTask("t1", "find TWP90 figures", "web")
    assert "find TWP90 figures" in task.brief()


def test_brief_includes_boundaries_when_set():
    """Boundaries are what stop two subagents doing the same work."""
    task = SubTask("t1", "find figures", "web", boundaries="do not cover causes")
    assert "Not your job" in task.brief()
    assert "do not cover causes" in task.brief()


def test_brief_omits_the_boundary_line_when_empty():
    assert "Not your job" not in SubTask("t1", "x", "web").brief()


def test_a_url_gets_one_number_however_often_it_is_added():
    index = SourceIndex()
    first = index.add("Title", "https://a.com")
    second = index.add("Different title", "https://a.com")
    assert first.n == second.n == 1
    assert len(index.sources) == 1


def test_different_urls_get_different_numbers():
    index = SourceIndex()
    assert index.add("A", "https://a.com").n == 1
    assert index.add("B", "https://b.com").n == 2


def test_sources_without_a_url_are_kept_separate():
    """Database answers have no URL but still need their own citation."""
    index = SourceIndex()
    assert index.add("database (chinook)", "").n == 1
    assert index.add("database (northwind)", "").n == 2


def test_number_for_finds_a_known_url():
    index = SourceIndex()
    index.add("A", "https://a.com")
    assert index.number_for("https://a.com") == 1
    assert index.number_for("https://unknown.com") is None


def test_source_renders_without_a_trailing_dash_when_url_is_empty():
    index = SourceIndex()
    assert str(index.add("database (chinook)", "")).endswith("(chinook)")


def test_findings_render_with_their_citation_number():
    index = SourceIndex()
    index.add("A", "https://a.com")
    finding = Finding("TWP90 was 4.62%", "OJK data", "https://a.com", "high")
    assert "[1]" in render_findings([finding], index)
    assert "high" in render_findings([finding], index)


def test_rendering_no_findings_says_so_rather_than_returning_blank():
    assert "No findings" in render_findings([], SourceIndex())
