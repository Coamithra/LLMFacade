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
