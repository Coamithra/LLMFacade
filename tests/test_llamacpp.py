"""LlamaCppServerProvider unit tests.

Drives ``CompletionRequest`` through ``_build_kwargs`` and asserts the
SDK-shaped payload, covers tool-call parsing, finish_reason translation, the
extra_body routing for llama.cpp-specific samplers (top_k/min_p/
repeat_penalty), the output_format JSON branch, and the introspection +
``count_tokens`` paths against a mocked httpx transport."""

from __future__ import annotations

from typing import Any

import pytest

from llmfacade import Message, tool
from llmfacade.exceptions import ProviderError
from llmfacade.provider import CompletionRequest
from llmfacade.providers.llamacpp import LlamaCppServerProvider
from llmfacade.settings import OutputFormat


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"Weather in {city}: sunny."


def _req(
    *,
    settings: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
    messages: list[Message] | None = None,
) -> CompletionRequest:
    s = {"max_tokens": 64}
    if settings:
        s.update(settings)
    return CompletionRequest(
        model="qwen2.5",
        messages=messages or [Message(role="user", content="hi")],
        system_blocks=[],
        tools=tools or [],
        stop=None,
        settings=s,
        settings_source={k: "convo" for k in s},
    )


@pytest.fixture
def provider() -> LlamaCppServerProvider:
    # Constructor builds the OpenAI client + httpx clients but doesn't fire
    # any requests; a fake base_url is fine for unit tests.
    return LlamaCppServerProvider(base_url="http://invalid.local:0/v1")


# ---- _build_kwargs --------------------------------------------------------


def test_build_kwargs_top_k_min_p_repeat_penalty_via_extra_body(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(
        _req(settings={"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1})
    )
    assert kwargs["extra_body"] == {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1}
    # And these keys do NOT leak into the top-level kwargs (the OpenAI SDK
    # would reject them as unknown).
    for k in ("top_k", "min_p", "repeat_penalty"):
        assert k not in kwargs


def test_build_kwargs_extra_body_omitted_when_no_llamacpp_knobs(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"temperature": 0.5}))
    assert "extra_body" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_build_kwargs_output_format_json_sets_response_format(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"output_format": OutputFormat.JSON}))
    assert kwargs["response_format"] == {"type": "json_object"}


def test_build_kwargs_output_format_text_omits_response_format(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"output_format": OutputFormat.TEXT}))
    assert "response_format" not in kwargs


def test_build_kwargs_tools_sets_tool_choice_default(provider: LlamaCppServerProvider):
    kwargs = provider._build_kwargs(_req(tools=[get_weather]))
    assert kwargs["tools"][0]["function"]["name"] == "get_weather"
    assert kwargs["tool_choice"] == "auto"


def test_build_kwargs_stop_passed_through(provider: LlamaCppServerProvider):
    req = CompletionRequest(
        model="qwen2.5",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=["END"],
        settings={"max_tokens": 16},
        settings_source={"max_tokens": "convo"},
    )
    kwargs = provider._build_kwargs(req)
    assert kwargs["stop"] == ["END"]


# ---- response parsing -----------------------------------------------------


class _FakeFn:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, *, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _FakeFn(name, arguments)


class _FakeMsg:
    def __init__(self, *, content: str = "", tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, *, message: _FakeMsg, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, *, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeResponse:
    def __init__(
        self,
        *,
        content: str = "ok",
        finish_reason: str = "stop",
        tool_calls: list[Any] | None = None,
        model: str = "qwen2.5",
        prompt: int = 5,
        completion: int = 2,
    ):
        self.choices = [
            _FakeChoice(
                message=_FakeMsg(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
        self.usage = _FakeUsage(prompt=prompt, completion=completion)
        self.model = model


def test_complete_finish_reason_length(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="hello", finish_reason="length"),
    )
    resp = provider._complete_raw(_req())
    assert resp.finish_reason == "length"
    assert resp.text == "hello"


def test_complete_finish_reason_stop(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="hello", finish_reason="stop"),
    )
    resp = provider._complete_raw(_req())
    assert resp.finish_reason == "stop"


def test_complete_tool_call_roundtrip(monkeypatch, provider: LlamaCppServerProvider):
    fake = _FakeResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            _FakeToolCall(
                id="call-1",
                name="get_weather",
                arguments='{"city": "Paris"}',
            )
        ],
    )
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: fake,
    )
    resp = provider._complete_raw(_req(tools=[get_weather]))
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}
    assert call.id == "call-1"


# ---- streaming ------------------------------------------------------------


class _FakeDelta:
    def __init__(self, *, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeStreamChoice:
    def __init__(self, *, delta: _FakeDelta, finish_reason: str | None = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeStreamChunk:
    def __init__(
        self,
        *,
        choices: list[_FakeStreamChoice] | None = None,
        usage: _FakeUsage | None = None,
    ):
        self.choices = choices or []
        self.usage = usage


def test_stream_finish_reason_length(monkeypatch, provider: LlamaCppServerProvider):
    chunks = [
        _FakeStreamChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content="hi"))]),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="length")],
            usage=_FakeUsage(prompt=5, completion=2),
        ),
    ]
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: iter(chunks),
    )
    events = list(provider._stream_raw(_req()))
    final = [e for e in events if e.done]
    assert len(final) == 1
    assert final[0].finish_reason == "length"


# ---- introspection (mocked httpx) ----------------------------------------


class _FakeHttpResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_health_returns_parsed_body(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._http, "get", lambda path: _FakeHttpResponse(json_body={"status": "ok"})
    )
    assert provider.health() == {"status": "ok"}


def test_health_raises_on_503(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._http,
        "get",
        lambda path: _FakeHttpResponse(status_code=503, text="loading"),
    )
    with pytest.raises(ProviderError):
        provider.health()


def test_slots_returns_list(monkeypatch, provider: LlamaCppServerProvider):
    fake = [{"id": 0, "is_processing": False, "n_ctx": 8192}]
    monkeypatch.setattr(provider._http, "get", lambda path: _FakeHttpResponse(json_body=fake))
    assert provider.slots() == fake


def test_save_slot_posts_with_action_query(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen["path"] = path
        seen["params"] = params
        seen["json"] = json
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(provider._http, "post", fake_post)
    out = provider.save_slot(0, "warmup.bin")
    assert out == {"ok": True}
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "save"}
    assert seen["json"] == {"filename": "warmup.bin"}


def test_erase_slot_no_body(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen["json"] = json
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(provider._http, "post", fake_post)
    provider.erase_slot(2)
    assert seen["json"] is None


# ---- count_tokens ---------------------------------------------------------


def test_count_tokens_hits_tokenize(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, json=None):
        seen["path"] = path
        seen["json"] = json
        return _FakeHttpResponse(json_body={"tokens": [1, 2, 3, 4, 5]})

    monkeypatch.setattr(provider._http, "post", fake_post)
    assert provider.count_tokens("hello world") == 5
    assert seen["path"] == "/tokenize"
    assert seen["json"] == {"content": "hello world"}


def test_count_tokens_falls_back_on_connection_error(
    monkeypatch, provider: LlamaCppServerProvider
):
    def boom(path, *, json=None):
        raise ConnectionError("server unreachable")

    monkeypatch.setattr(provider._http, "post", boom)
    # chars/4 of "hello world" (11 chars) → 2.
    assert provider.count_tokens("hello world") == 2


def test_tokenizer_name_is_llama_server(provider: LlamaCppServerProvider):
    assert provider.tokenizer_name() == "llama-server /tokenize"


# ---- managed mode (lazy supervisor, launch-knob validation) --------------


from pathlib import Path  # noqa: E402

from llmfacade.exceptions import UnsupportedFeature  # noqa: E402


def test_external_mode_launch_knob_in_init_rejected() -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        LlamaCppServerProvider(base_url="http://x:0/v1", context_size=8192)


def test_external_mode_new_model_with_launch_knobs_rejected(
    provider: LlamaCppServerProvider,
) -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        provider.new_model("qwen", context_size=8192)


def test_external_mode_new_model_without_id_raises(
    provider: LlamaCppServerProvider,
) -> None:
    with pytest.raises(ValueError, match="requires a positional model_id"):
        provider.new_model()


def test_managed_mode_constructor_no_supervisor_started(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    assert p._managed
    assert p._supervisor is not None
    assert not p._supervisor.is_started
    # No openai client built yet; only built once supervisor starts.
    assert p._client is None


def test_managed_mode_new_model_requires_gguf(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(ValueError, match="requires gguf="):
        p.new_model()


def test_managed_mode_new_model_missing_gguf_path(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(FileNotFoundError, match="gguf not found"):
        p.new_model(gguf=str(tmp_path / "nonexistent.gguf"))


def test_managed_mode_new_model_registers_entry(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), context_size=8192)
    entries = p._supervisor.entries  # type: ignore[union-attr]
    assert len(entries) == 1
    assert entries[0].gguf == str(gguf)
    assert entries[0].context_size == 8192
    # Model id is `<stem>-<hash8>` derived from launch config.
    assert model.model_id.startswith("qwen-")
    suffix = model.model_id.rsplit("-", 1)[1]
    assert len(suffix) == 8


def test_managed_mode_explicit_name_used_as_model_id(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), name="qwen-fast")
    assert model.model_id == "qwen-fast"


def test_managed_mode_provider_defaults_cascade_into_model(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(
        llmfacade_dir=tmp_path / "sess",
        n_gpu_layers=32,
        cache_type_k="q8_0",
    )
    p.new_model(gguf=str(gguf))
    entry = p._supervisor.entries[0]  # type: ignore[union-attr]
    assert entry.n_gpu_layers == 32
    assert entry.cache_type_k == "q8_0"


def test_managed_mode_model_overrides_provider_defaults(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", n_gpu_layers=32)
    p.new_model(gguf=str(gguf), n_gpu_layers=8)
    assert p._supervisor.entries[0].n_gpu_layers == 8  # type: ignore[union-attr]


def test_managed_mode_two_models_register_two_entries(tmp_path: Path) -> None:
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(a))
    p.new_model(gguf=str(b))
    assert len(p._supervisor.entries) == 2  # type: ignore[union-attr]


def test_managed_mode_running_unload_against_dead_supervisor_raises_useful_error(
    tmp_path: Path,
) -> None:
    """Without llama-swap on PATH, calling `running()` triggers ensure_started()
    which raises ProviderNotInstalledError. The user gets the clear install hint
    rather than a confusing AttributeError."""
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"x")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    # We can't reliably assume llama-swap isn't on PATH in CI, but if it is the
    # call would otherwise spawn it. Stub which() to None to force the
    # not-installed path deterministically.
    import llmfacade.providers._swap_lifecycle as ls

    original = ls.shutil.which
    ls.shutil.which = lambda b: None  # type: ignore[assignment]
    try:
        from llmfacade.exceptions import ProviderNotInstalledError

        with pytest.raises(ProviderNotInstalledError):
            p.running()
    finally:
        ls.shutil.which = original  # type: ignore[assignment]


def test_managed_mode_shutdown_no_op_when_never_started(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.shutdown()  # must not raise
    p.shutdown()


def test_external_mode_shutdown_no_op() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    p.shutdown()  # supervisor is None; should just no-op
