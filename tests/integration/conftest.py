"""Integration test setup.

These tests hit real provider APIs and cost money. They are gated behind
``-m integration``; the default ``pytest`` invocation skips them.

Each test is also individually skipped when its provider's API key isn't
present, so a partial ``.env`` (e.g. only ``ANTHROPIC_API_KEY``) still runs
the tests it can. Loading happens once per session via ``python-dotenv``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is a dev dep; skip the whole suite without it.
    load_dotenv = None


_REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    if load_dotenv is not None:
        load_dotenv(_REPO_ROOT / ".env", override=False)


def _require_env(var: str) -> None:
    """Skip the test unless ``var`` is set. Returns nothing — tests must
    let the provider read the key from the environment via ``_resolve_key``,
    so the secret never appears as a local in pytest tracebacks."""
    if not os.getenv(var):
        pytest.skip(f"{var} not set; skipping live test")


@pytest.fixture
def anthropic_api_key() -> None:
    _require_env("ANTHROPIC_API_KEY")


@pytest.fixture
def openai_api_key() -> None:
    _require_env("OPENAI_API_KEY")


@pytest.fixture
def google_api_key() -> None:
    _require_env("GOOGLE_API_KEY")


@pytest.fixture
def llamacpp_host() -> str:
    return os.getenv("LLAMACPP_HOST", "http://localhost:8080")


@pytest.fixture
def llamacpp_model() -> str:
    return os.getenv("LLAMACPP_MODEL", "qwen2.5-3b-instruct-q4_k_m")
