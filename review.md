# Codebase Review - Action Items

Critical review of LLMFacade as of the initial commit. Tackle top-down within each section.

## Real bugs

- [x] **#1 Google tool-result roundtrip is broken.** `providers/google.py:188-200` sets `function_response.name = b.tool_use_id`, but `tool_use_id` is a synthetic ID. Gemini matches `function_response.name` against the original `function_call.name`. Fix: store the function name on the `ToolResultBlock` (or look it up from history) and emit it here. Add a Gemini integration test that does a tool roundtrip.

- [x] **#2 Tool-dispatch loop has no max iteration guard.** `conversation.py:207-223` (and `aComplete` mirror at 252-268) is `while True:`. A misbehaving model can spin forever and burn tokens. Add `max_tool_iterations` (default ~16) and raise on overflow.

- [ ] **#3 Streaming silently drops tool dispatch.** `Stream`/`aStream` collect `tool_call_delta`s but never dispatch tools, even though `Complete`'s default is `auto_tools=True`. Either implement the loop in streaming (yield tool events, then dispatch, then continue the stream) or document and remove the asymmetry.

- [ ] **#4 `_log_request` writes the entire history every turn.** `conversation.py:465-477`. JSONL grows quadratically. Log only the new turn (delta), or split request envelope from history.

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

- [x] **#24 No test for the `auto_tools` infinite-loop case.** Add once #2 is fixed.

- [ ] **#25 No test for streaming + tool calls.** Add once #3 is decided.

- [x] **#26 No test that `LLM.default()` stays clean across tests.** Add once #5 is fixed.

- [x] **#27 No test for capability override edge cases on `Complete`.** `_collect_overrides` skips validation when value is `None`; verify `top_k=0` still validates correctly.

## Repo hygiene

- [x] **#28 No `LICENSE` file.** `pyproject.toml` declares MIT - add `LICENSE`.

- [x] **#29 No CI.** Add `.github/workflows/ci.yml` running `ruff check`, `ruff format --check`, and `pytest` on push/PR.

- [ ] **#30 Bump SDK version floors.** `pyproject.toml` pins like `anthropic>=0.40` are pre-tool-use-2.0 era. Bump to current majors and re-test.

- [ ] **#31 `examples/migrations/` is unreferenced.** Either link from README or move out of the published distribution.

## Suggested order

Do in this sequence to minimize churn:

1. Quick wins: **#2**, **#5**, **#6**, **#8**, **#19**, **#21**, **#28**, **#29** (each is one short PR).
2. Correctness: **#1**, **#16**, **#18**, **#22** (fix wire-format issues before more users hit them).
3. Architecture cleanup as one branch: **#9**, **#10**, **#11**, **#12**, **#13**. These touch every file; doing them together avoids three painful migrations.
4. Streaming pass: **#3**, **#17**, **#20**.
5. Coverage: **#23-#27**, then **#15**, **#30**, **#31**.
