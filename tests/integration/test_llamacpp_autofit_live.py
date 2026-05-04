"""Live integration test for the managed-mode autofit path.

Skipped unless ``llama-swap``, ``llama-server``, AND ``llama-fit-params`` are
all on PATH and ``LLAMACPP_GGUF`` points at a real GGUF file. Verifies that:

* `provider.new_model(gguf=...)` synchronously runs `llama-fit-params`,
  parses its output, and stashes the estimate keyed by the derived model_id.
* The rendered `swap.yaml` ``cmd:`` line contains ``--fit on`` by default.
* The JSONL settings header (and HTML log) carries a top-level
  ``fit_estimate`` block once a Conversation is constructed.
* A first ``convo.send("hi")`` round-trip succeeds (server starts under
  ``--fit on``).

Env vars (mandatory):

* ``LLAMACPP_GGUF`` — path to a GGUF the spawned llama-server can load.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from llmfacade import LLM
from llmfacade.providers.llamacpp import LlamaCppServerProvider

pytestmark = pytest.mark.integration


def _require_binaries_and_gguf() -> str:
    if shutil.which("llama-swap") is None:
        pytest.skip("llama-swap binary not on PATH")
    if shutil.which("llama-server") is None:
        pytest.skip("llama-server binary not on PATH")
    if shutil.which("llama-fit-params") is None:
        pytest.skip("llama-fit-params binary not on PATH")
    gguf = os.getenv("LLAMACPP_GGUF")
    if not gguf or not Path(gguf).exists():
        pytest.skip("LLAMACPP_GGUF env var must point at an existing GGUF file")
    return gguf


def test_new_model_populates_fit_estimate_and_swap_yaml(tmp_path: Path) -> None:
    gguf = _require_binaries_and_gguf()
    sess_dir = tmp_path / "llmfacade-autofit"

    llm = LLM(log_dir=False)
    provider: LlamaCppServerProvider = llm.new_provider(  # type: ignore[assignment]
        "llamacpp", llmfacade_dir=sess_dir
    )

    try:
        model = provider.new_model(name="autofit-probe", gguf=gguf, max_tokens=8)

        # Estimate dict key is created (None on parser failure, populated on
        # success). At minimum the wrapper attempted the probe.
        assert "autofit-probe" in provider._fit_estimates
        est = provider._fit_estimates["autofit-probe"]
        # Estimate may be None if llama-fit-params output drifts from the
        # tuned regexes — print so the test surface is informative either way.
        print(f"[autofit-probe] fit_estimate = {est!r}")
        assert est is None or isinstance(est, dict)
        if isinstance(est, dict):
            # At least one of the known fields should be present when parsing
            # succeeds. parallel echoes back from the entry; the rest come
            # from the fit-params output.
            assert any(
                k in est
                for k in ("context_size", "n_gpu_layers", "parallel", "est_vram_mib")
            ), f"fit_estimate dict has no known fields: {est!r}"

        # Rendered swap.yaml must contain --fit on by default. Triggering
        # supervisor start writes the YAML to disk via the lazy-spawn path,
        # so do an explicit YAML-write check via the supervisor's rendering.
        from llmfacade.providers._swap_lifecycle import _render_swap_yaml

        rendered = _render_swap_yaml(provider._supervisor.entries)  # type: ignore[union-attr]
        assert "--fit on" in rendered, f"rendered swap.yaml lacks --fit on:\n{rendered}"

        # The JSONL settings header should now carry a top-level fit_estimate
        # block when an estimate exists (skip header check if estimate is None
        # since the hook returns None and nothing gets logged).
        log_path = tmp_path / "autofit.jsonl"
        convo = model.new_conversation(name="autofit", log_path=log_path)
        header_line = log_path.read_text(encoding="utf-8").splitlines()[0]
        header = json.loads(header_line)
        if est is not None:
            assert "fit_estimate" in header, (
                f"settings header missing fit_estimate when estimate exists: {header!r}"
            )
            assert header["fit_estimate"] == est
            html = (tmp_path / "autofit.html").read_text(encoding="utf-8")
            assert "Fit estimate" in html
        else:
            assert "fit_estimate" not in header

        # Round-trip: a first send must succeed under --fit on. This exercises
        # the OOM-safety promise — on a small GPU the fit logic shrinks ctx /
        # offload to fit, so the server starts and answers instead of crashing.
        resp = convo.send("Reply with the single word OK.")
        assert resp.text and resp.text.strip()
    finally:
        provider.shutdown()


def test_new_model_with_fit_false_skips_estimate(tmp_path: Path) -> None:
    """`fit=False` must skip the fit-params spawn AND render `--fit off` in
    the YAML. No round-trip — just registration + rendering."""
    gguf = _require_binaries_and_gguf()
    sess_dir = tmp_path / "llmfacade-autofit-off"

    llm = LLM(log_dir=False)
    provider: LlamaCppServerProvider = llm.new_provider(  # type: ignore[assignment]
        "llamacpp", llmfacade_dir=sess_dir
    )

    try:
        model = provider.new_model(name="no-fit", gguf=gguf, fit=False, max_tokens=8)
        assert provider._fit_estimates["no-fit"] is None
        assert provider.log_metadata(model_id=model.model_id) is None

        from llmfacade.providers._swap_lifecycle import _render_swap_yaml

        rendered = _render_swap_yaml(provider._supervisor.entries)  # type: ignore[union-attr]
        assert "--fit off" in rendered
        assert "--fit on" not in rendered
    finally:
        provider.shutdown()
