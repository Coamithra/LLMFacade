"""Live ``llama-swap`` subprocess integration test for managed-mode llamacpp.

Skipped unless the ``llama-swap`` binary is on PATH and ``LLAMACPP_GGUF`` is
set to a real GGUF model path the test can launch. Two ``new_model`` calls
register two distinct YAML entries; the test issues a send to each and
exercises the introspection + lifecycle endpoints.

Env vars:

* ``LLAMACPP_GGUF`` — path to a GGUF on disk (mandatory; test skipped if missing).
* ``LLAMACPP_GGUF_B`` — optional second GGUF; defaults to the same as
  ``LLAMACPP_GGUF`` so the swap tests still exercise two distinct entries
  via differing context_size.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from llmfacade import LLM
from llmfacade.exceptions import UnsupportedFeature
from llmfacade.providers._swap_lifecycle import _pid_alive_and_named
from llmfacade.providers.llamacpp import LlamaCppServerProvider

pytestmark = pytest.mark.integration


def _require_swap_and_gguf() -> tuple[str, str]:
    if shutil.which("llama-swap") is None:
        pytest.skip("llama-swap binary not on PATH")
    if shutil.which("llama-server") is None:
        pytest.skip("llama-server binary not on PATH")
    gguf = os.getenv("LLAMACPP_GGUF")
    if not gguf or not Path(gguf).exists():
        pytest.skip("LLAMACPP_GGUF env var must point at an existing GGUF file")
    gguf_b = os.getenv("LLAMACPP_GGUF_B", gguf)
    return gguf, gguf_b


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "llmfacade-swap"


def test_managed_mode_two_models_swap_and_introspection(session_dir: Path) -> None:
    gguf_a, gguf_b = _require_swap_and_gguf()

    llm = LLM(log_dir=False)
    provider: LlamaCppServerProvider = llm.new_provider("llamacpp", llmfacade_dir=session_dir)  # type: ignore[assignment]

    fast = provider.new_model(name="fast", gguf=gguf_a, context_size=2048, max_tokens=64)
    if gguf_a == gguf_b:
        # Differentiate entries by context_size so the YAML has two slots.
        quality = provider.new_model(name="quality", gguf=gguf_b, context_size=4096, max_tokens=64)
    else:
        quality = provider.new_model(name="quality", gguf=gguf_b, max_tokens=64)

    convo_a = fast.new_conversation()
    convo_b = quality.new_conversation()

    try:
        # First send triggers spawn.
        resp_a = convo_a.send("Reply with the single word OK.")
        assert resp_a.text.strip()
        # YAML and pidfile should now exist on disk.
        assert provider._supervisor is not None
        assert provider._supervisor.yaml_path.exists()
        assert provider._supervisor.pid_file.exists()

        # After send_a, `fast` should be the active loaded model.
        running_after_a = provider.running()
        assert isinstance(running_after_a, list)
        ids_after_a = {entry.get("model") or entry.get("id") for entry in running_after_a}
        assert "fast" in ids_after_a, f"`fast` not loaded after first send: {running_after_a!r}"

        # Second send to the other model triggers a swap.
        resp_b = convo_b.send("Reply with the single word HELLO.")
        assert resp_b.text.strip()

        running_after_b = provider.running()
        ids_after_b = {entry.get("model") or entry.get("id") for entry in running_after_b}
        assert "quality" in ids_after_b, (
            f"`quality` not loaded after second send: {running_after_b!r}"
        )

        # Introspection probe — record outcome rather than failing the test if
        # llama-swap doesn't proxy /slots through.
        slots_outcome: str
        try:
            slots = provider.slots()
            slots_outcome = f"works: {len(slots)} slot(s)"
        except Exception as e:  # noqa: BLE001
            slots_outcome = f"unavailable: {type(e).__name__}: {e}"
        print(f"[introspection-probe] /slots through llama-swap: {slots_outcome}")

        # unload() works against llama-swap.
        try:
            provider.unload("quality")
        except UnsupportedFeature:
            pytest.fail("unload() should be supported in managed mode")

        # Capture the supervisor PID before shutdown so we can verify it's gone.
        proc = provider._supervisor._proc
        assert proc is not None
        swap_pid = proc.pid
    finally:
        provider.shutdown()

    # Process cleanup verification: the PID is no longer alive.
    # Give the OS a brief moment to reap.
    for _ in range(20):
        if not _pid_alive_and_named(swap_pid, "llama-swap"):
            break
        time.sleep(0.1)
    assert not _pid_alive_and_named(swap_pid, "llama-swap"), (
        f"llama-swap pid {swap_pid} survived shutdown"
    )
    # PID file removed too.
    assert not provider._supervisor.pid_file.exists()  # type: ignore[union-attr]


def test_register_after_send_does_not_400(session_dir: Path) -> None:
    """Mirrors `plans/llmfacade-multi-variant-bug.md`: register model A, send to
    it, *then* register model B and send to it immediately. Without the
    `_wait_for_model_visible` step in `register()`, B's send would race
    llama-swap's 2s `-watch-config` poll and return HTTP 400 'could not find
    suitable inference handler'."""
    gguf_a, gguf_b = _require_swap_and_gguf()

    llm = LLM(log_dir=False)
    provider: LlamaCppServerProvider = llm.new_provider("llamacpp", llmfacade_dir=session_dir)  # type: ignore[assignment]

    model_a = provider.new_model(name="bench-a", gguf=gguf_a, context_size=2048, max_tokens=4)
    convo_a = model_a.new_conversation(log_dir=False)
    try:
        resp_a = convo_a.send("hi")
        assert resp_a.text is not None

        model_b = provider.new_model(
            name="bench-b",
            gguf=gguf_b,
            # Differentiate from A by context_size so we get a fresh YAML entry
            # even when LLAMACPP_GGUF_B is unset and falls back to gguf_a.
            context_size=4096 if gguf_a == gguf_b else 2048,
            max_tokens=4,
        )
        convo_b = model_b.new_conversation(log_dir=False)
        # No sleep — exercises the post-spawn registration path. Without the
        # fix this returns ProviderError wrapping a 400.
        resp_b = convo_b.send("hi")
        assert resp_b.text is not None
    finally:
        provider.shutdown()


def test_interrupt_aborts_in_flight_send_from_another_thread(session_dir: Path) -> None:
    """Acceptance for `provider.interrupt()`: park a long managed-mode send on a
    background thread, call `interrupt()` from the main thread mid-flight, and
    assert the send raises promptly (well under its natural duration) and that a
    subsequent send recovers (lazy respawn)."""
    import threading

    from llmfacade.exceptions import LLMError

    gguf_a, _ = _require_swap_and_gguf()

    llm = LLM(log_dir=False)
    provider: LlamaCppServerProvider = llm.new_provider("llamacpp", llmfacade_dir=session_dir)  # type: ignore[assignment]
    # A large max_tokens so the decode runs for many seconds — long enough that
    # an instant abort is unambiguously faster than natural completion.
    model = provider.new_model(name="long", gguf=gguf_a, context_size=2048, max_tokens=2048)
    convo = model.new_conversation(log_dir=False)

    result: dict[str, object] = {}
    started = threading.Event()

    def worker() -> None:
        started.set()
        try:
            convo.send("Write an extremely long, detailed essay about the history of computing.")
            result["ok"] = True
        except LLMError as e:  # transport error surfaces as a facade error
            result["error"] = e

    try:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        started.wait(timeout=5)
        # Give the backend a moment to load + actually be parked decoding, then
        # abort. (First send also lazily spawns llama-swap and loads the model.)
        time.sleep(15.0)
        killed = provider.interrupt()
        assert killed is True

        # The blocked send must unblock promptly after the kill — not run to the
        # natural end of a 2048-token decode.
        t.join(timeout=20)
        assert not t.is_alive(), "send() did not unblock after interrupt()"
        assert "error" in result, f"expected a transport error, got {result!r}"

        # Recovery: a fresh send respawns llama-swap and reloads the model.
        recover = model.new_conversation(log_dir=False)
        resp = recover.send("hi")
        assert resp.text is not None
    finally:
        provider.shutdown()
