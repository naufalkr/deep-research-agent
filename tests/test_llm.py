import pytest

from deepresearch import llm, observability
from deepresearch.config import Settings
from deepresearch.llm import LLMClient


@pytest.fixture
def mock_settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LEAD_MODEL", "model-lead")
    monkeypatch.setenv("SUBAGENT_MODEL", "model-sub")
    return Settings.from_env()


@pytest.fixture(autouse=True)
def clean_tracker():
    observability.tracker.reset(budget_usd=0.0)
    yield
    observability.tracker.reset(budget_usd=0.0)


def test_mock_provider_needs_no_api_key(mock_settings):
    client = LLMClient(settings=mock_settings)
    assert client.complete("hello").startswith("[mock]")


def test_role_selects_the_model(mock_settings):
    client = LLMClient(settings=mock_settings)
    assert "model-lead" in client.complete("x", role="lead")
    assert "model-sub" in client.complete("x", role="subagent")


def test_every_call_is_recorded(mock_settings):
    client = LLMClient(settings=mock_settings)
    client.complete("x", role="lead", agent="lead", step="plan")

    assert len(observability.tracker.calls) == 1
    rec = observability.tracker.calls[0]
    assert rec.agent == "lead"
    assert rec.step == "plan"
    assert rec.model == "model-lead"
    assert rec.latency_s >= 0


def test_agent_defaults_to_the_role(mock_settings):
    client = LLMClient(settings=mock_settings)
    client.complete("x", role="critic")
    assert observability.tracker.calls[0].agent == "critic"


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMClient(settings=Settings.from_env())


def test_deepseek_points_at_its_own_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = LLMClient(settings=Settings.from_env())
    assert "api.deepseek.com" in str(client._client.base_url)


def test_deepseek_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        LLMClient(settings=Settings.from_env())


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    with pytest.raises((ValueError, RuntimeError)):
        LLMClient(settings=Settings.from_env())


def test_transient_errors_are_retried(mock_settings, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)  # don't actually wait
    calls = {"n": 0}

    class RateLimitError(Exception):
        pass

    def flaky(self, model, prompt, system, max_tokens, temperature):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("slow down")
        return "recovered", 1, 1

    monkeypatch.setattr(LLMClient, "_call", flaky)
    client = LLMClient(settings=mock_settings)
    assert client.complete("x") == "recovered"
    assert calls["n"] == 3


def test_permanent_errors_are_not_retried(mock_settings, monkeypatch):
    calls = {"n": 0}

    def broken(self, model, prompt, system, max_tokens, temperature):
        calls["n"] += 1
        raise ValueError("bad request")

    monkeypatch.setattr(LLMClient, "_call", broken)
    client = LLMClient(settings=mock_settings)
    with pytest.raises(ValueError):
        client.complete("x")
    assert calls["n"] == 1


def test_sampling_params_are_dropped_for_models_that_reject_them(monkeypatch):
    """Current Claude models 400 on temperature — it must not be sent."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUBAGENT_MODEL", "claude-opus-5")
    seen = {}

    class FakeMessages:
        def create(self, **kwargs):
            seen.update(kwargs)
            return type(
                "R",
                (),
                {
                    "content": [type("B", (), {"type": "text", "text": "hi"})()],
                    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 1})(),
                },
            )()

    client = LLMClient(settings=Settings.from_env())
    client._client = type("C", (), {"messages": FakeMessages()})()

    client.complete("x", role="subagent")
    assert "temperature" not in seen


def test_sampling_params_are_kept_for_older_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUBAGENT_MODEL", "claude-haiku-4-5")
    seen = {}

    class FakeMessages:
        def create(self, **kwargs):
            seen.update(kwargs)
            return type(
                "R",
                (),
                {
                    "content": [type("B", (), {"type": "text", "text": "hi"})()],
                    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 1})(),
                },
            )()

    client = LLMClient(settings=Settings.from_env())
    client._client = type("C", (), {"messages": FakeMessages()})()

    client.complete("x", role="subagent")
    assert seen["temperature"] == 0.1


async def test_acomplete_runs_the_same_path(mock_settings):
    client = LLMClient(settings=mock_settings)
    reply = await client.acomplete("x", role="lead")
    assert "model-lead" in reply
    assert len(observability.tracker.calls) == 1
