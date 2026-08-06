from deepresearch.config import Settings


def _settings(monkeypatch, **env) -> Settings:
    for key in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "LEAD_MODEL",
        "SUBAGENT_MODEL",
        "SEARCH_PROVIDER",
        "EXA_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env()


def test_each_role_gets_its_own_model(monkeypatch):
    s = _settings(
        monkeypatch,
        LEAD_MODEL="model-lead",
        CRITIC_MODEL="model-critic",
        SUBAGENT_MODEL="model-sub",
        CITATION_MODEL="model-cite",
    )
    assert s.model_for("lead") == "model-lead"
    assert s.model_for("critic") == "model-critic"
    assert s.model_for("subagent") == "model-sub"
    assert s.model_for("citation") == "model-cite"


def test_roles_have_defaults_when_env_is_empty(monkeypatch):
    s = _settings(monkeypatch)
    # a frontier model plans, a cheap one does volume work
    assert s.model_for("lead") != s.model_for("subagent")


def test_missing_reports_the_provider_key(monkeypatch):
    s = _settings(monkeypatch, LLM_PROVIDER="anthropic", SEARCH_PROVIDER="exa")
    assert "ANTHROPIC_API_KEY" in s.missing()
    assert "EXA_API_KEY" in s.missing()


def test_missing_is_empty_when_configured(monkeypatch):
    s = _settings(
        monkeypatch,
        LLM_PROVIDER="groq",
        GROQ_API_KEY="gsk_test",
        SEARCH_PROVIDER="exa",
        EXA_API_KEY="exa_test",
    )
    assert s.missing() == []


def test_mock_provider_needs_no_keys(monkeypatch):
    s = _settings(monkeypatch, LLM_PROVIDER="mock", SEARCH_PROVIDER="groq_builtin")
    assert s.missing() == []


def test_tavily_key_only_required_when_selected(monkeypatch):
    s = _settings(
        monkeypatch,
        LLM_PROVIDER="mock",
        SEARCH_PROVIDER="tavily",
        EXA_API_KEY="ignored",
    )
    assert s.missing() == ["TAVILY_API_KEY"]
