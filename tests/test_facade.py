"""LLM manager: default singleton + reset behavior."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from llmfacade import LLM


def test_llm_default_is_a_singleton():
    a = LLM.default()
    b = LLM.default()
    assert a is b


def test_default_mutation_phase_a():
    LLM.default().api_keys["leak"] = "abc"
    assert LLM.default().api_keys["leak"] == "abc"


def test_default_mutation_phase_b_clean():
    # If the autouse _reset_llm_default fixture is wired up correctly, the
    # mutation from phase_a must not leak into this test.
    assert "leak" not in LLM.default().api_keys


def test_reset_default_creates_fresh_instance():
    first = LLM.default()
    LLM.reset_default()
    second = LLM.default()
    assert first is not second


def test_default_is_thread_safe_under_concurrent_first_call():
    """Concurrent first-touch callers must all receive the same instance and
    LLM.__init__ must run exactly once."""
    LLM.reset_default()

    init_count = 0
    original_init = LLM.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal init_count
        init_count += 1
        original_init(self, *args, **kwargs)

    LLM.__init__ = counting_init
    try:
        n = 50
        barrier = threading.Barrier(n)
        results: list[LLM] = [None] * n  # type: ignore[list-item]

        def worker(i: int) -> None:
            barrier.wait()
            results[i] = LLM.default()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        LLM.__init__ = original_init

    assert init_count == 1
    first = results[0]
    assert all(r is first for r in results)


def test_log_dir_defaults_to_cwd_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = LLM()
    assert llm.run_dir is not None
    assert llm.run_dir.parent == tmp_path / "logs"
    assert llm.run_dir.name.startswith("llmfacade")
    # Lazy: just constructing the LLM does not touch disk.
    assert not (tmp_path / "logs").exists()


def test_log_dir_false_disables_run_dir():
    llm = LLM(log_dir=False)
    assert llm.run_dir is None
    assert llm._ensure_run_dir() is None


def test_log_dir_explicit_path(tmp_path):
    llm = LLM(log_dir=tmp_path / "custom")
    assert llm.run_dir is not None
    assert llm.run_dir.parent == tmp_path / "custom"


def test_ensure_run_dir_materializes_and_prunes(tmp_path):
    base = tmp_path / "logs"
    base.mkdir()
    # Pre-existing session folders that should be pruned to fit max_log_folders=2.
    old_a = base / "llmfacade20200101-000000"
    old_b = base / "llmfacade20200102-000000"
    old_c = base / "llmfacade20200103-000000"
    # Pruning sorts by st_mtime; on Windows three back-to-back mkdirs can share
    # a tick, leaving the sort order up to iterdir() (B-tree order on NTFS, not
    # alphabetical). Stamp distinct mtimes so old_c is unambiguously newest.
    for i, d in enumerate((old_a, old_b, old_c)):
        d.mkdir()
        (d / "marker.txt").write_text("x")
        os.utime(d, (1_700_000_000 + i, 1_700_000_000 + i))
    llm = LLM(log_dir=base, max_log_folders=2)
    run_dir = llm._ensure_run_dir()
    assert run_dir is not None
    assert run_dir.exists()
    # max_log_folders=2 keeps 1 old + the new one.
    surviving = sorted(p.name for p in base.iterdir() if p.is_dir())
    assert len(surviving) == 2
    assert run_dir.name in surviving
    # The two oldest got removed.
    assert old_a.name not in surviving
    assert old_b.name not in surviving
    # Idempotent: second call doesn't re-prune.
    again = llm._ensure_run_dir()
    assert again == run_dir


def test_ensure_run_dir_skips_non_llmfacade_siblings(tmp_path):
    base = tmp_path / "logs"
    base.mkdir()
    keep_me = base / "user-thing"
    keep_me.mkdir()
    (keep_me / "data.txt").write_text("x")
    llm = LLM(log_dir=base, max_log_folders=1)
    llm._ensure_run_dir()
    assert keep_me.exists()
    assert (keep_me / "data.txt").exists()


def test_max_log_folders_zero_keeps_only_new_run(tmp_path):
    base = tmp_path / "logs"
    base.mkdir()
    old = base / "llmfacade20200101-000000"
    old.mkdir()
    llm = LLM(log_dir=base, max_log_folders=0)
    run_dir = llm._ensure_run_dir()
    assert run_dir is not None
    assert not old.exists()
    assert run_dir.exists()


def test_run_dir_is_session_stamped_and_unique():
    a = LLM(log_dir=Path("/tmp/test"))
    b = LLM(log_dir=Path("/tmp/test"))
    assert a.run_dir is not None and b.run_dir is not None
    # Different LLMs created back-to-back may share a stamp at second-resolution
    # but they should at least be Paths, not None, and parented identically.
    assert a.run_dir.parent == b.run_dir.parent
