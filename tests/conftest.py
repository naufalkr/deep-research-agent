import pytest

from deepresearch.tools import web


@pytest.fixture(autouse=True)
def search_cache_in_tmp(tmp_path, monkeypatch):
    """Mock results must never land in the real search cache."""
    monkeypatch.setattr(web, "CACHE_DIR", str(tmp_path / "search"))
