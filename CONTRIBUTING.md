# Contributing: Tackling a Trello Card

Step-by-step workflow for picking up and completing any card from the [LLMFacade Trello board](https://trello.com/b/WIanVfPx) (board id `69f86428`).

---

## Quick ship (no card / small change)

Not every change is a Trello card. For a quick fix or doc tweak that doesn't warrant the full runbook below, the default ship flow is **PR + auto self-merge**:

```
git checkout -b <prefix>/<short-name>     # off main
git add <files> && git commit -m "..."     # only the files you touched
git push -u origin <branch>
gh pr create --fill                        # PR record + URL, no clicking
gh pr merge --merge                        # self-merge (see note); use --merge, not --squash
git checkout main && git pull origin main  # fast-forward local main to the merge
```

**No approval needed.** `main` is an unprotected branch on this solo public repo, so GitHub disabling the "Approve" button on your *own* PR is irrelevant — a required review only applies under a branch-protection rule, and this repo has none. Don't stop to ask the user to approve or open the PR by hand. (If the user says "just merge / direct", skip the PR entirely and fast-forward `main`.) The full card runbook (Phase 6) uses this same merge step inside the worktree flow.

---

## Worktree Quick Reference

All work happens in an isolated **git worktree** under `.trees/`. This lets multiple agents work on different cards simultaneously without interfering with each other. The root checkout stays on `main` — never switch it to a feature branch.

| Command | What it does |
|---------|-------------|
| `git worktree add .trees/<name> -b <branch> main` | Create a new worktree + branch from main |
| `git worktree list` | Show all active worktrees |
| `git worktree remove .trees/<name>` | Remove a worktree (clean up) |
| `git worktree prune` | Clean up stale worktree references |

**Key rules:**
- Each worktree gets its own branch; a branch can only be checked out in one worktree at a time
- Gitignored files (`.env`, `venv/`, `.llmfacade/`) do NOT exist in new worktrees — recreate them manually if needed for local testing (e.g. provider API keys, llamacpp `swap.yaml` state)
- All worktree directories live under `.trees/` (gitignored)

---

## Phase 1: Pick Up the Card

> **Picking up the top card? Use the atomic `grab` command.** When you are told to "pick up the top card/ticket" (rather than a specific named card), claim it in one step:
>
> ```
> trello --board 69f86428 grab --from "To Do" --to "Doing"
> ```
>
> This pops the top card of To Do, moves it to Doing, and prints the card it got you (it exits 1 when To Do is empty). It is safe to fire from several agents at once: each gets a distinct card, so no two collide on the same ticket. On the remote Trello backend `grab` settles ties with a brief (~10-30s) claim-comment handshake. For a specific card you were named, skip this and use step 3 below.

1. **Pull latest main** — `git pull origin main` so you start from the newest code
2. **Read the card** — Read the card description and any linked spec under `plans/<file>.md`. The plan is the long-form source of truth; the card is a pointer
3. **Move card to Doing** — `trello --board 69f86428 card move <card_id> Doing`
4. **Create worktree and branch** — Branch off `main` with a descriptive prefix:
    - Bugs: `fix/<short-name>` (e.g. `fix/multi-variant-race`)
    - Features: `feat/<short-name>` (e.g. `feat/flash-attn-knob`)
    - Refactoring: `refactor/<short-name>`
    - Docs / plans only: `docs/<short-name>`
    ```
    git worktree add .trees/<branch> -b <branch> main
    cd .trees/<branch>
    git push -u origin <branch>
    ```
5. **All subsequent work happens inside `.trees/<branch>/`**

## Phase 2: Research

Dig into the problem before proposing solutions. Use `/research` for topics that need external context (e.g. provider SDK quirks, llama.cpp endpoint contracts, tokenizer behaviour).

6. **Read the referenced code** — Card descriptions and `plans/*.md` cite specific files and line numbers. Read them — descriptions can drift
7. **Trace the call chain** — For bugs, trace how the problematic code gets invoked. For features, trace the existing system the feature plugs into (the four-level cascade `LLM → Provider → Model → Conversation` is documented in `CLAUDE.md`)
8. **Identify the blast radius** — Which providers does this touch? Which knobs cascade through it? Cross-check `SUPPORTS` sets per provider, capability gating, and the JSONL/HTML log format if you're adding fields
9. **Research unknowns** — Use `/research` for anything that needs external knowledge: SDK contract details (Anthropic / OpenAI / Google), llama.cpp / llama-swap behaviour, tokenizer gotchas
10. **Summarize findings** — Brief writeup of what you learned: root cause (bugs), design options (features), or risk areas (refactors). Becomes input to the design phase

## Phase 3: Design

11. **Draft the approach** — Either update the existing `plans/<file>.md` or write one. Include:
    - **Context**: what the card is about and why it matters
    - **Design**: file-by-file changes; new public API; cascade behaviour; capability flags
    - **Tests**: which test files get new tests, with names
    - **Out of scope**: what you're explicitly *not* doing
12. **Check for reusable patterns** — Look for existing utilities and conventions before inventing new ones (e.g. `_validate_knobs`/`_filter_unsupported` cascade helpers, `SUPPORTS` capability sets, `@dataclass(frozen=True, slots=True)` for wire-format types, `LAUNCH_KNOBS` vs `RUNTIME_KNOBS`)
13. **Align with the user** — Present the plan, get approval before writing code

## Phase 4: Implement

14. **Make the changes** — Edit files per the approved plan. Follow project conventions:
    - **Style**: Ruff with rules `E, F, I, UP, B, SIM`; line length 99; full type annotations; snake_case throughout
    - **Wire-format types**: `@dataclass(frozen=True, slots=True)`
    - **Configuration is constructor-only** — identity (api_key, base_url, model_id, system_blocks, tools, log_dir, log_path) is immutable post-construction
    - **Cascade**: generation knobs cascade `provider < model < convo < per_call`; unknown kwarg names raise `TypeError`; knobs not in the layer's effective `SUPPORTS` raise `UnsupportedFeature` at the layer they're set
    - **Capability gating**: never catch `UnsupportedFeature` to branch — query via `is_available(...)` / `get_capabilities()` first
    - **No `Start()` step** — conversations are usable immediately after construction
    - **Comments**: default to none; only add when the *why* is non-obvious. Don't narrate what the code does; identifiers handle that
15. **Document new conventions** — Update `CLAUDE.md` if the change introduces new knobs, new provider quirks, new file layout, or modifies cascade behaviour. CLAUDE.md is the source of truth

## Phase 5: Verify

16. **Lint** — `ruff check src/` and `ruff format src/` must be clean
17. **Smoke import** — `python -c "import llmfacade"` to catch syntax errors and broken imports
18. **Run unit tests** — `pytest` with default markers (no integration). Single-test invocation: `pytest tests/test_conversation.py::test_xyz`
19. **Integration tests are gated** — `tests/integration/` hits real provider APIs (Anthropic, OpenAI, Google) and burns credits on every run. **Never** invoke `pytest -m integration`, `pytest tests/integration/`, or any variant that includes them, unless the user has explicitly asked for that specific run in the current turn. Past authorization does not carry over. The llamacpp integration test is local and free, but still requires explicit permission so the rule stays simple: never auto-run anything under `tests/integration/`
20. **Manual smoke for managed-mode llamacpp changes** — If you touched the llamacpp managed-mode supervisor, swap.yaml rendering, or `_LaunchEntry` shape, run a real `provider.new_model(gguf=<real GGUF>) → convo.send("hi")` round-trip and inspect `<llmfacade_dir>/swap.yaml` and the JSONL log. Document the steps in the plan's "Verification" section
21. **Spot-check the diff** — Read through one more time for typos, off-by-ones, missing `await`, dict keys that don't exist, and `# removed` / dead-code residue
22. **Flag what needs manual testing** — Leave a note for the user of anything that can't be unit-tested (e.g. "verify Gemma 4 actually loads with `--mmproj`")

## Phase 6: Review & Ship

23. **Commit** — Descriptive message in the project's existing style (imperative, single-line subject, body explains *why* not *what*). Reference the card if useful. Push to the feature branch
24. **Peer review** — Run `/review` (spawns a fresh agent against the branch diff vs `main` with no prior context). It catches logic errors, missed edge cases, convention violations, naming issues we've gone blind to. Fix every finding before proceeding — even minor ones — unless the fix is a major undertaking (in which case track it as a follow-up card)
25. **Pull main into the branch** — `git pull origin main` to pick up anything that landed while you were working. Resolve conflicts using the rules below

### Merge Conflict Rules

25.1. **Default to main's version.** If a conflict is in code you didn't intentionally change, accept main's side. Someone else fixed a bug or added a feature — don't silently revert their work
25.2. **Assume incoming changes are important.** Treat every conflict as "main has a critical fix" until you've read the diff and confirmed otherwise. Be very careful about overwriting new code with your version
25.3. **Only keep your side for lines you specifically wrote.** If you changed a function and main also changed it, read both versions carefully. Merge surgically — keep their fixes, layer your feature on top
25.4. **If the merge is messy, restart from main.** When conflicts are widespread or hard to reason about, it's safer to take main wholesale and reimplement your changes on top. A clean re-apply is better than a botched merge
25.5. **Re-read the final result.** After resolving, read through every conflicted file in full. Make sure the merged code actually makes sense — don't just trust the conflict markers

26. **Re-run lint + unit tests** — Make sure the merge didn't break anything: `ruff check src/`, `pytest`
27. **Return to the root checkout** — `cd` back to the project root (where `main` is checked out). Remaining steps run from here
28. **Open a PR and self-merge** — `gh pr create --fill` then `gh pr merge --merge` (real merge commit, not `--squash`, so the branch's commits stay reachable and step 29's `git branch -d` still works), then `git pull origin main` to fast-forward the root checkout. **No approval needed** — `main` is unprotected on this solo repo, so GitHub disabling "Approve" on your own PR is irrelevant; a required review only applies under a branch-protection rule, which there is none. The PR is a record/URL with no extra ceremony — don't wait on a human to approve. (Direct `git merge <branch> && git push` is the fallback if `gh` is unavailable.)
29. **Clean up the worktree and branch**
    ```
    git worktree remove .trees/<branch>
    git worktree prune
    git branch -d <branch>
    git push origin --delete <branch>
    ```
30. **Delete the plan file** — If the card has a `plans/<file>.md` behind it, delete it now (`git rm plans/<file>.md && git commit -m "Remove <name> plan; <feature/fix> is implemented" && git push`). The plans directory is for *open* work only
31. **Move card to Done** — `trello --board 69f86428 card move <card_id> Done`
32. **Comment on the card** — `trello --board 69f86428 comment add <card_id> "<summary>"`. Include: what changed, which files, what it fixes/adds, the commit hash(es), and what needs manual testing. Leaves a paper trail for future debugging
33. **Create follow-up cards** — If review, implementation, or testing surfaced issues that are out of scope for this card (pre-existing bugs, minor improvements, edge cases deferred as too risky to bundle), create new Trello cards. Reference the original card so there's a trail. Don't let follow-up work disappear into commit messages — if it's worth noting, it's worth tracking
34. **Write an overview of the changes made** — End the session with a concise overview for the user: what changed, which files were touched, why it was done, and anything they should know going forward (manual-test steps, follow-up cards filed, behaviour shifts). This is the last thing in the conversation — a clear summary so the user doesn't have to reconstruct the work from commits or scrollback
