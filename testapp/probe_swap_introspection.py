"""Probe: managed-mode introspection through llama-swap's /upstream/<model>/.

The phase-2 plan flagged the introspection-routing question as unverified;
the followup plan (~/.claude/plans/starry-leaping-liskov.md) settled it: route
through `/upstream/<model>/...` in managed mode. This script smoke-tests both
the provider-level wrappers (with and without a `model=` arg) and the
Model-bound mirrors. Expected outcome post-fix: every wrapper succeeds, no
silent fallbacks, no FAILED lines.
"""

import os
import sys
import traceback

from llmfacade import LLM

# Override with `PROBE_GGUF=/path/to/model.gguf python testapp/probe_…`.
DEFAULT_GGUF = r"C:\Models\qwen2.5-3b.gguf"


FAILURES: list[str] = []


def try_call(label: str, fn):
    print(f"\n--- {label} ---")
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        FAILURES.append(label)
        return None
    if isinstance(result, list):
        print(f"  OK: list of {len(result)}")
        for i, item in enumerate(result[:2]):
            display = (
                item
                if not isinstance(item, dict)
                else {k: item[k] for k in list(item)[:6]}
            )
            print(f"    [{i}] {display}")
    elif isinstance(result, dict):
        keys = list(result)
        print(f"  OK: dict keys={keys[:8]}{'...' if len(keys) > 8 else ''}  {result if len(keys) <= 4 else ''}")
    else:
        print(f"  OK: {result!r}")
    return result


def main() -> None:
    llm = LLM.default()
    provider = llm.new_provider("llamacpp", temperature=0.7)
    gguf = os.environ.get("PROBE_GGUF", DEFAULT_GGUF)
    model = provider.new_model(
        name="qwen2.5-3b",
        gguf=gguf,
        n_gpu_layers=999,
        max_tokens=64,
    )

    try:
        convo = model.new_conversation(
            name="probe",
            system_blocks=["You are terse. Respond in one short sentence."],
        )
        print("Sending one warmup message to spawn server...")
        resp = convo.send("hi")
        print(f"  reply: {resp.text!r}")

        # ---- provider wrappers, swap-root form (no model=) ----
        print("\n=== provider wrappers — swap-root form (no model=) ===")
        result = try_call("provider.health()  # swap-root, expect {'status':'ok'}", provider.health)
        if result is not None and result != {"status": "ok"}:
            FAILURES.append("provider.health() returned unexpected shape")
        try_call("provider.running()  # llama-swap-native", provider.running)

        # ---- provider wrappers with single-model inference (model= omitted) ----
        print("\n=== provider wrappers — single-model inference (model= omitted) ===")
        try_call("provider.slots()  # infers single registered model", provider.slots)
        try_call(
            "provider.count_tokens('hello world')  # routes via inferred model",
            lambda: provider.count_tokens("hello world"),
        )

        # ---- provider wrappers with explicit model= ----
        print("\n=== provider wrappers — explicit model= ===")
        try_call(
            "provider.health(model='qwen2.5-3b')  # backend JSON via /upstream/",
            lambda: provider.health(model="qwen2.5-3b"),
        )
        try_call(
            "provider.slots(model='qwen2.5-3b')",
            lambda: provider.slots(model="qwen2.5-3b"),
        )
        try_call(
            "provider.count_tokens('hello', model_id='qwen2.5-3b')",
            lambda: provider.count_tokens("hello", model_id="qwen2.5-3b"),
        )

        # ---- Model-bound mirrors (auto-bind self._model_id) ----
        print("\n=== Model-bound mirrors ===")
        try_call("model.health()", model.health)
        try_call("model.slots()", model.slots)
        try_call("model.count_tokens('hello world')", lambda: model.count_tokens("hello world"))
        try_call("model.tokenizer_name()", model.tokenizer_name)

        # ---- save / restore / erase round-trip via Model ----
        print("\n=== Model-bound slot save/restore/erase round-trip ===")
        try_call("model.save_slot(0, 'probe.bin')", lambda: model.save_slot(0, "probe.bin"))
        try_call("model.restore_slot(0, 'probe.bin')", lambda: model.restore_slot(0, "probe.bin"))
        try_call("model.erase_slot(0)", lambda: model.erase_slot(0))
    finally:
        provider.shutdown()
        print("\nshutdown complete.")
        if FAILURES:
            print(f"\n*** {len(FAILURES)} FAILED probe(s):")
            for f in FAILURES:
                print(f"  - {f}")
            sys.exit(1)
        else:
            print("\nAll probes succeeded.")


if __name__ == "__main__":
    main()
