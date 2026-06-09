"""``Conversation.save_slot`` / ``restore_slot`` / ``erase_slot`` /
``warm_and_save`` plus the lock primitives on ``LlamaCppServerProvider``.

These tests exercise the high-level slot API on top of an external-mode
``LlamaCppServerProvider`` whose httpx clients are monkeypatched. Managed
mode is excluded by raising ``NotImplementedError`` from the conversation
methods (deferred to v2)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from llmfacade.exceptions import (
    ConversationStateError,
    ProviderError,
    UnsupportedFeature,
)
from llmfacade.providers.llamacpp import LlamaCppServerProvider

from .conftest import MockProvider


class _FakeHttpResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self) -> Any:
        return self._json


@pytest.fixture
def llama_provider() -> LlamaCppServerProvider:
    return LlamaCppServerProvider(base_url="http://invalid.local:0/v1")


def _convo(provider: LlamaCppServerProvider, **convo_kwargs):
    model = provider.new_model("qwen2.5", max_tokens=64)
    return model.new_conversation(log_dir=False, **convo_kwargs)


# ---- capability gating ----------------------------------------------------


def test_save_slot_unsupported_on_non_llamacpp_provider() -> None:
    p = MockProvider()
    convo = p.new_model("mock").new_conversation(log_dir=False)
    with pytest.raises(UnsupportedFeature, match="slot_save_restore"):
        convo.save_slot("warmup.bin")


def test_restore_slot_unsupported_on_non_llamacpp_provider() -> None:
    p = MockProvider()
    convo = p.new_model("mock").new_conversation(log_dir=False)
    with pytest.raises(UnsupportedFeature):
        convo.restore_slot("warmup.bin")


def test_erase_slot_unsupported_on_non_llamacpp_provider() -> None:
    p = MockProvider()
    convo = p.new_model("mock").new_conversation(log_dir=False)
    with pytest.raises(UnsupportedFeature):
        convo.erase_slot()


def test_warm_and_save_unsupported_on_non_llamacpp_provider() -> None:
    p = MockProvider()
    convo = p.new_model("mock").new_conversation(log_dir=False)
    with pytest.raises(UnsupportedFeature):
        convo.warm_and_save("warmup.bin")


def test_save_slot_not_implemented_in_managed_mode(tmp_path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), name="qwen-fast")
    convo = model.new_conversation(log_dir=False)
    with pytest.raises(NotImplementedError, match="external-mode only"):
        convo.save_slot("warmup.bin")


def test_warm_and_save_not_implemented_in_managed_mode(tmp_path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), name="qwen-fast")
    convo = model.new_conversation(log_dir=False)
    with pytest.raises(NotImplementedError):
        convo.warm_and_save("warmup.bin")


# ---- filename sanitisation ------------------------------------------------


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("", "non-empty string"),
        (".hidden", "must not start with"),
        ("..", "must not start with"),
        ("foo/bar.bin", "disallowed characters"),
        ("foo\\bar.bin", "disallowed characters"),
        ("../foo.bin", "must not start with"),
        ("foo bar.bin", "disallowed characters"),
        ("foo:bar.bin", "disallowed characters"),
        ("a..b.bin", "must not contain '..'"),
    ],
)
def test_save_slot_rejects_unsafe_filename(
    llama_provider: LlamaCppServerProvider, bad: str, reason: str
) -> None:
    convo = _convo(llama_provider)
    with pytest.raises(ValueError, match=reason):
        convo.save_slot(bad)


def test_save_slot_accepts_safe_filename(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen.update(path=path, params=params, json=json)
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    out = convo.save_slot("npc_guard-v2.bin")
    assert out == {"ok": True}
    assert seen == {
        "path": "/slots/0",
        "params": {"action": "save"},
        "json": {"filename": "npc_guard-v2.bin"},
    }


# ---- routing through the provider ----------------------------------------


def test_save_slot_routes_to_slot_zero(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen.update(path=path, params=params, json=json)
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    convo.save_slot("warm.bin")
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "save"}


def test_restore_slot_routes_to_slot_zero(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen.update(path=path, params=params, json=json)
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    convo.restore_slot("warm.bin")
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "restore"}
    assert seen["json"] == {"filename": "warm.bin"}


def test_erase_slot_sends_no_filename(monkeypatch, llama_provider: LlamaCppServerProvider) -> None:
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen.update(path=path, params=params, json=json)
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    convo.erase_slot()
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "erase"}
    assert seen["json"] is None


# ---- async equivalents ----------------------------------------------------


def test_asave_slot_routes_through_async_client(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    seen: dict[str, Any] = {}

    async def fake_apost(path, *, params=None, json=None):
        seen.update(path=path, params=params, json=json)
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._ahttp, "post", fake_apost)
    convo = _convo(llama_provider)
    asyncio.run(convo.asave_slot("warm.bin"))
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "save"}
    assert seen["json"] == {"filename": "warm.bin"}


# ---- error translation ----------------------------------------------------


def test_save_slot_wraps_provider_error_with_slot_save_path_hint(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    def fake_post(path, *, params=None, json=None):
        return _FakeHttpResponse(
            status_code=500,
            text='{"error":"This server does not support slots action."}',
        )

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    with pytest.raises(ProviderError, match="--slot-save-path"):
        convo.save_slot("warm.bin")


def test_save_slot_unrelated_provider_error_passes_through(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    def fake_post(path, *, params=None, json=None):
        return _FakeHttpResponse(status_code=503, text="server is busy")

    monkeypatch.setattr(llama_provider._http, "post", fake_post)
    convo = _convo(llama_provider)
    with pytest.raises(ProviderError) as exc_info:
        convo.save_slot("warm.bin")
    # Message preserved verbatim; no --slot-save-path hint added.
    assert "--slot-save-path" not in str(exc_info.value)


# ---- warm_and_save --------------------------------------------------------


def test_warm_and_save_requires_empty_history(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    convo = _convo(llama_provider)
    convo.add_user_message("seeded")
    with pytest.raises(ConversationStateError, match="empty conversation history"):
        convo.warm_and_save("warm.bin")


def test_warm_and_save_runs_send_then_save_and_rolls_back(
    monkeypatch, llama_provider: LlamaCppServerProvider
) -> None:
    """The warmup send should leave history empty (rolled back), and the
    persisted save call should fire after the send returns."""
    sequence: list[str] = []

    class _FakeChatCompletions:
        def create(self, **_kw):
            sequence.append("send")
            from tests.test_llamacpp import _FakeResponse

            return _FakeResponse(content="x")

    monkeypatch.setattr(llama_provider._client.chat, "completions", _FakeChatCompletions())

    def fake_post(path, *, params=None, json=None):
        sequence.append(f"post:{params['action']}")
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)

    convo = _convo(llama_provider, system_blocks=["you are X"])
    out = convo.warm_and_save("guard.bin")
    assert out == {"ok": True}
    assert sequence == ["send", "post:save"]
    # History must be empty so the next user send acts as a fresh first turn.
    assert convo.history == []


def test_warm_and_save_bypasses_response_cache(
    monkeypatch, llama_provider: LlamaCppServerProvider, tmp_path
) -> None:
    """A pre-populated response cache must not satisfy the warmup completion:
    a cache hit would mean no request ever reaches llama-server, so save_slot
    would persist a cold slot-0 KV. The warmup must hit the provider hook
    every time, and must not write a cache entry either."""
    create_calls: list[str] = []

    class _FakeChatCompletions:
        def create(self, **_kw):
            create_calls.append("create")
            from tests.test_llamacpp import _FakeResponse

            return _FakeResponse(content="x")

    monkeypatch.setattr(llama_provider._client.chat, "completions", _FakeChatCompletions())

    def fake_post(path, *, params=None, json=None):
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(llama_provider._http, "post", fake_post)

    model = llama_provider.new_model("qwen2.5", max_tokens=64)
    # Seed the cache with the exact warmup request (same prompt ".", same
    # per-call max_tokens) via a normal cached send.
    seed = model.new_conversation(log_dir=False, cache_dir=tmp_path)
    seed.send(".", max_tokens=1)
    assert create_calls == ["create"]
    cached_before = list(tmp_path.rglob("*.json"))
    assert cached_before, "seed send should have populated the cache"

    convo = model.new_conversation(log_dir=False, cache_dir=tmp_path)
    out = convo.warm_and_save("warm.bin", max_warmup_tokens=1)
    assert out == {"ok": True}
    assert create_calls == ["create", "create"], "warmup must reach the server despite the hit"
    # The warmup must not write to the cache either.
    assert list(tmp_path.rglob("*.json")) == cached_before
    assert convo.history == []


# ---- lock primitives ------------------------------------------------------


def test_slot_lock_returns_threading_lock(
    llama_provider: LlamaCppServerProvider,
) -> None:
    lock = llama_provider.slot_lock()
    # threading.Lock() returns a `_thread.lock` whose type isn't directly
    # importable, but ``acquire`` / ``release`` is the contract we care about.
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")
    # Same instance on repeat calls.
    assert llama_provider.slot_lock() is lock


def test_aslot_lock_returns_asyncio_lock(
    llama_provider: LlamaCppServerProvider,
) -> None:
    lock = llama_provider.aslot_lock()
    assert isinstance(lock, asyncio.Lock)
    assert llama_provider.aslot_lock() is lock


def test_slot_lock_serialises_concurrent_holders(
    llama_provider: LlamaCppServerProvider,
) -> None:
    """Two threads racing on the lock observe mutual exclusion."""
    lock = llama_provider.slot_lock()
    enter = threading.Event()
    held = [False]
    overlapped = [False]

    def critical():
        with lock:
            if held[0]:
                overlapped[0] = True
            held[0] = True
            enter.wait(timeout=0.1)
            held[0] = False

    t1 = threading.Thread(target=critical)
    t2 = threading.Thread(target=critical)
    t1.start()
    t2.start()
    enter.set()
    t1.join()
    t2.join()
    assert overlapped[0] is False


def test_aslot_lock_serialises_concurrent_coroutines(
    llama_provider: LlamaCppServerProvider,
) -> None:
    """Two coroutines racing on ``aslot_lock`` observe mutual exclusion."""
    lock = llama_provider.aslot_lock()
    state = {"held": False, "overlapped": False}

    async def critical() -> None:
        async with lock:
            if state["held"]:
                state["overlapped"] = True
            state["held"] = True
            # Yield to the event loop so the other coro gets a chance to
            # try acquire while we're inside the section.
            await asyncio.sleep(0)
            state["held"] = False

    async def driver() -> None:
        await asyncio.gather(critical(), critical())

    asyncio.run(driver())
    assert state["overlapped"] is False
