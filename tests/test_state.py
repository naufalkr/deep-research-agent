from deepresearch.state import (
    Finding,
    Source,
    SourceIndex,
    SubTask,
    compact_citations,
    database_url,
    render_findings,
)


def test_brief_states_the_objective():
    assert "find TWP90 figures" in SubTask("t1", "find TWP90 figures").brief()


def test_brief_includes_boundaries_when_set():
    brief = SubTask("t1", "find figures", boundaries="do not cover causes").brief()
    assert "Not your job: do not cover causes" in brief


def test_a_neutral_brief_names_no_side():
    assert "Your side" not in SubTask("t1", "x").brief()


def test_debate_briefs_name_the_side():
    assert "FOR" in SubTask("pro", "banks should use AI", "pro").brief()
    assert "AGAINST" in SubTask("con", "banks should use AI", "con").brief()


def test_a_url_gets_one_number_however_often_it_is_added():
    index = SourceIndex()
    assert index.add("Title", "https://a.com").n == index.add("Other", "https://a.com").n == 1
    assert len(index.sources) == 1


def test_each_database_gets_its_own_number():
    index = SourceIndex()
    assert index.add("db", database_url("chinook")).n == 1
    assert index.add("db", database_url("northwind")).n == 2
    assert index.number_for(database_url("northwind")) == 2


def test_an_empty_url_has_no_number():
    index = SourceIndex()
    index.add("t", "")
    assert index.number_for("") is None


def test_database_sources_render_without_the_internal_url():
    assert str(Source(1, "internal database (chinook)", "db://chinook")) == (
        "[1] internal database (chinook)"
    )


def test_findings_render_with_number_side_and_verdict():
    index = SourceIndex()
    index.add("A", "https://a.com")
    f = Finding("TWP90 was 4.62%", "OJK data", "https://a.com", "high", stance="con",
                rebuttal=True, verdict="supported")
    text = render_findings([f], index)
    assert "[1]" in text
    assert "side: against (rebuttal)" in text
    assert "verdict: supported" in text


def test_rendering_no_findings_says_so():
    assert "No findings" in render_findings([], SourceIndex())


# --- citation compaction ---


def sources(n):
    return [Source(i, f"title {i}", f"https://s.com/{i}") for i in range(1, n + 1)]


def test_only_cited_sources_are_kept_and_renumbered_in_order():
    text, kept = compact_citations("A [7]. B [3]. C [7].", sources(9))
    assert text == "A [1]. B [2]. C [1]."
    assert [(s.n, s.url) for s in kept] == [(1, "https://s.com/7"), (2, "https://s.com/3")]


def test_adjacent_markers_are_renumbered_and_sorted():
    text, _ = compact_citations("X [5][2]. Y [2]. Z [5].", sources(5))
    assert text == "X [1][2]. Y [2]. Z [1]."


def test_brackets_that_are_not_sources_are_left_alone():
    text, kept = compact_citations("In [2024] it rose [1].", sources(2))
    assert text == "In [2024] it rose [1]."
    assert len(kept) == 1


def test_an_uncited_report_keeps_no_sources():
    assert compact_citations("nothing cited", sources(3)) == ("nothing cited", [])
