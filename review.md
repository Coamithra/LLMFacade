# Codebase Review - Action Items

Critical review of LLMFacade as of the initial commit. Tackle top-down within each section.

## Real bugs

- [x] **#1 Google tool-result roundtrip is broken.** `providers/google.py:188-200` sets `function_response.name = b.tool_use_id`, but `tool_use_id` is a synthetic ID. Gemini matches `function_response.name` against the original `function_call.name`. Fix: store the function name on the `ToolResultBlock` (or look it up from history) and emit it here. Add a Gemini integration test that does a tool roundtrip.

- [x] **#2 Tool-dispatch loop has no max iteration guard.** `conversation.py:207-223` (and `aComplete` mirror at 252-268) is `while True:`. A misbehaving model can spin forever and burn tokens. Add `max_tool_iterations` (default ~16) and raise on overflow. *(Resolved more thoroughly by #3: the loop was removed from Conversation entirely. The cap now lives in `helpers.run_to_completion`, which still raises `ToolIterationLimitError`.)*

- [x] **#3 Streaming silently drops tool dispatch.** Resolved by removing auto-dispatch from `Conversation` entirely. `Complete`/`aComplete` were renamed to `Send`/`aSend` and are now strict single round-trips matching `Stream`/`aStream`. Tool execution moved to `llmfacade.helpers` (`run_bound_tools`, `run_to_completion`, plus async equivalents), built on the public API. Wire-format invariant is now enforced: `Send`/`Stream` raise `ConversationStateError` if any `tool_use` in history lacks a matching `tool_result`. This fixes the silent-corruption failure mode and removes the streaming/non-streaming asymmetry at its root rather than papering over it.

- [x] **#4 `_log_request` writes the entire history every turn.** Resolved by switching to delta logging *and* moving the cache-waste diagnostic to the response side, where the provider's actual `cache_read_tokens` / `cache_creation_tokens` give ground truth. Request records now just carry `new_messages` (delta since last log) plus an abbreviated `prior_history` preview (head/...elision.../tail) — no more local LCP tracking, no `_last_cached_msgs` state. Response records gain a `cache_summary` block computed from `Usage`: `cache_read_tokens`, `cache_creation_tokens`, `uncached_input_tokens`, `hit_ratio`, `approximate_messages_cached` (mapped back via per-provider tokenizer — tiktoken for OpenAI, chars/4 fallback for Anthropic/Google), `tokenizer` label, and a context-aware `_note` (caching working / first-cacheable turn / markers off / TTL-or-divergence / not supported). Per-turn log size is bounded; total log is O(N). Clone treats inherited history as already-logged. OpenAI/Google providers now capture cache stats (`prompt_tokens_details.cached_tokens`, `cached_content_token_count`) so the diagnostic works uniformly across providers. The stats-based approach automatically covers TTL expiry, mid-prefix mutation, missing beta headers, etc., without local state — when something invalidates the cache, `cache_read_tokens` drops and the `_note` flags it.

- [x] **#5 `LLM._default` is a process-wide mutable singleton.** `facade.py:17,28-32`. Mutating `LLM.default().api_keys` leaks across tests/libraries. Add `LLM.reset_default()` and a pytest fixture that resets between tests, or drop the singleton.

- [x] **#6 `LLM.log_dir` is dead code.** `facade.py:26`. Stored, never read. Wire it up as the default root for `Conversation.SetLogging`, or remove the field (and the README/CLAUDE.md mentions).

- [x] **#7 `Provider.NewModel(**kwargs)` and `Model.NewConversation(**kwargs)` swallow typos.** `provider.py:104-107`, `model.py:47-50`. Replace with explicit kwargs.

- [x] **#8 `ToolCall._fn` and `Response.raw` participate in equality/hashing.** `models.py:71-75, 99-108`. Add `compare=False` to both fields.

## Architecture (will haunt us)

- [ ] **#9 `_complete_raw` has 12 keyword arguments.** `provider.py:135-150`. Replace with a `CompletionRequest` dataclass; adding a setting then touches one type instead of four providers + facade + tests.

- [ ] **#10 Three-level `_SettingsFacade` is reimplemented identically.** `provider.py:23-65`. Provider/Model/Conversation each hold their own facade with the same SUPPORTS set. Collapse to one settings store keyed by scope; drop the duplication and the "where do I read this from" branching in `_call_kwargs`.

- [ ] **#11 Three separate `Settings` enums.** `settings.py`. Users have to remember scope. Unify into one `Settings` enum with a private `_scope` attribute (or constants module).

- [ ] **#12 PascalCase methods on Python classes.** `NewProvider`, `Complete`, `AddTool`, `Snapshot`, etc. Conflicts with PEP 8 and mixes with snake_case properties (`provider.name`, `convo.history`) and SCREAMING_SNAKE constants (`SUPPORTS`, `NAME`). Convert methods to snake_case before any external users adopt the API.

- [ ] **#13 Per-call override kwargs are a closed enum.** `conversation.py:30-37` `_PER_CALL_OVERRIDE_KEYS` plus the explicit kwargs on `Complete`/`Stream`. Take `**overrides` (or `dict[Settings, Any]`) and validate via the existing capability check.

- [ ] **#14 Two sources of truth for `BaseURL`.** `provider.py:75-89`: `__init__` accepts `base_url=`, stores `self._base_url`, and writes to settings; SDK clients read `self._base_url`, not the setting. Mutating the setting after construction has no effect. Pick one path.

- [ ] **#15 Anthropic `_NO_THINKING_MODELS` substring match.** `providers/anthropic.py:54, 76`. Hardcoded list will rot. Either query the SDK for capabilities or document this as best-effort and make it user-overridable via `capability_override`.

- [x] **#16 Tool call IDs synthesized from `id(obj)`.** `google.py:175,245,247`, `ollama.py:190,266`. CPython reuses ids after GC. Switch to `uuid.uuid4().hex` or a per-conversation counter.

- [ ] **#17 Streamed assistant turns lose thinking content.** `conversation.py:368-382` `_finalize_stream` discards `thinking_buf`. Stream and non-stream produce different histories for the same response. Persist thinking in the assistant message (or document the asymmetry).

- [x] **#18 `tools.py` falls back to `{"type": "string"}` for unknown annotations.** `tools.py:90-91`. Pydantic/dataclass annotations silently become strings -> model passes a string -> function TypeErrors at call time. Raise at decoration time or warn loudly.

- [x] **#19 Lazy intra-package imports.** `provider.py:48`, `conversation.py:455`, `facade.py:100`. No circular-import risk. Move to module top.

- [ ] **#20 Sync and async stream parsers are copy-pasted in every provider.** `anthropic.py:199-237` vs `239-277`, similar in OpenAI/Google/Ollama. ~150 duplicated lines. Factor into a shared parser over sync/async event sources.

- [x] **#21 `MockProvider` `@dataclass` + explicit `__init__` does nothing useful.** `tests/conftest.py:29-80`. Drop the decorator.

- [x] **#22 `OpenAIProvider._message_to_api` silently drops images on assistant messages.** `providers/openai.py:274-277`. Round-tripping a vision response loses image blocks. Either raise or warn.

## Test gaps

- [ ] **#23 Zero integration tests.** All tests use `MockProvider`. Bugs #1 and #15 would never surface. Add at least one skip-if-no-key live test per provider that does a tool roundtrip.

- [x] **#24 No test for the `auto_tools` infinite-loop case.** Add once #2 is fixed. *(Now covered by `test_helpers_run_to_completion_caps_iterations`.)*

- [x] **#25 No test for streaming + tool calls.** ~~Add once #3 is decided.~~ Moot after #3: `Send` and `Stream` are both strict single round-trips, so streaming + tool calls is just "stream returns a Response with tool_calls; caller dispatches via `helpers.run_bound_tools` like the non-streaming path." `test_send_with_dangling_tool_use_raises` covers the wire-format guard that applies to both paths.

- [x] **#26 No test that `LLM.default()` stays clean across tests.** Add once #5 is fixed.

- [x] **#27 No test for capability override edge cases on `Complete`.** `_collect_overrides` skips validation when value is `None`; verify `top_k=0` still validates correctly.

## Repo hygiene

- [x] **#28 No `LICENSE` file.** `pyproject.toml` declares MIT - add `LICENSE`.

- [x] **#29 No CI.** Add `.github/workflows/ci.yml` running `ruff check`, `ruff format --check`, and `pytest` on push/PR.

- [ ] **#30 Bump SDK version floors.** `pyproject.toml` pins like `anthropic>=0.40` are pre-tool-use-2.0 era. Bump to current majors and re-test.

- [ ] **#31 `examples/migrations/` is unreferenced.** Either link from README or move out of the published distribution.

- [ ] **#32 Real tokenizers for Anthropic & Google `cache_summary` boundary estimate.** `conversation.py:_estimate_cached_boundary` maps `cache_read_tokens` back to a message index using `Provider._estimate_tokens`. OpenAI overrides with `tiktoken.encoding_for_model(model_id)` (precise). Anthropic and Google fall back to `chars/4` (English-biased, ±10–30% off), so `approximate_messages_cached` can be off by a message or two. Options: (a) Anthropic — community packages like `anthropic-tokenizer-python` aren't official; check the SDK for a local tokenizer at the next bump (network-bound `client.messages.count_tokens` is unsuitable for a logging hot path); or implement the GPT-4 `cl100k`-via-tiktoken proxy (closer than chars/4, still not exact). (b) Google — `vertexai`/`google-generativeai` have no offline tokenizer either; same tradeoff. Either commit to per-provider real tokenizers as optional extras (`pip install llmfacade[tokenizers]`) or document the imprecision. The `tokenizer` field in `cache_summary` already reports which one was used so users can calibrate.

## Suggested order

Do in this sequence to minimize churn:

1. Quick wins: **#2**, **#5**, **#6**, **#8**, **#19**, **#21**, **#28**, **#29** (each is one short PR).
2. Correctness: **#1**, **#16**, **#18**, **#22** (fix wire-format issues before more users hit them).
3. Architecture cleanup as one branch: **#9**, **#10**, **#11**, **#12**, **#13**. These touch every file; doing them together avoids three painful migrations.
4. Streaming pass: **#3**, **#17**, **#20**.
5. Coverage: **#23-#27**, then **#15**, **#30**, **#31**.
