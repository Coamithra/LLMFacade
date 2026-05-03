"""``_LlamaSwapSupervisor`` tests against stubbed subprocess + filesystem.

Covers lazy startup, PID-file write/sweep, idempotent shutdown, signal-handler
installation/restoration, and the orphan detection helper. Uses monkeypatch
to stub out ``subprocess.Popen`` / ``shutil.which`` / health polling so no
real ``llama-swap`` is needed."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from llmfacade.exceptions import ProviderError, ProviderNotInstalledError
from llmfacade.providers import _swap_lifecycle as ls
from llmfacade.providers._launch import _LaunchEntry


class _FakeProc:
    """Stand-in for subprocess.Popen with controllable poll/wait behaviour."""

    def __init__(self, pid: int = 12345, *, alive: bool = True):
        self.pid = pid
        self._alive = alive
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminate_called = True
        self._alive = False

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self._alive = False
        return 0


@pytest.fixture
def fake_supervisor(tmp_path: Path) -> ls._LlamaSwapSupervisor:
    return ls._LlamaSwapSupervisor(llmfacade_dir=tmp_path / "session", global_ttl=0)


# ---- registration ---------------------------------------------------------


def test_register_appends_entry(fake_supervisor: ls._LlamaSwapSupervisor) -> None:
    e = _LaunchEntry(model_id="m", gguf="x.gguf")
    fake_supervisor.register(e)
    assert fake_supervisor.entries == [e]


def test_register_same_id_same_config_is_noop(fake_supervisor: ls._LlamaSwapSupervisor) -> None:
    e = _LaunchEntry(model_id="m", gguf="x.gguf", context_size=8192)
    fake_supervisor.register(e)
    fake_supervisor.register(e)
    assert len(fake_supervisor.entries) == 1


def test_register_same_id_different_config_raises(
    fake_supervisor: ls._LlamaSwapSupervisor,
) -> None:
    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="a.gguf"))
    with pytest.raises(ValueError, match="already registered with different launch params"):
        fake_supervisor.register(_LaunchEntry(model_id="m", gguf="b.gguf"))


# ---- lazy startup ---------------------------------------------------------


def test_ensure_started_raises_when_no_entries(
    fake_supervisor: ls._LlamaSwapSupervisor,
) -> None:
    with pytest.raises(ProviderError, match="no models registered"):
        fake_supervisor.ensure_started()


def test_ensure_started_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    monkeypatch.setattr(ls.shutil, "which", lambda _: None)
    with pytest.raises(ProviderNotInstalledError, match="not found on PATH"):
        fake_supervisor.ensure_started()


def _patch_for_successful_start(
    monkeypatch: pytest.MonkeyPatch, *, proc: _FakeProc, port: int = 5555
) -> dict[str, Any]:
    """Patch shutil.which, _spawn_with_pdeathsig, _pick_free_localhost_port,
    and the health-polling httpx import. Returns a dict the test can inspect
    to check what got spawned."""
    captured: dict[str, Any] = {}

    monkeypatch.setattr(ls.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(ls, "_pick_free_localhost_port", lambda: port)

    def fake_spawn(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return proc, None

    monkeypatch.setattr(ls, "_spawn_with_pdeathsig", fake_spawn)

    class _FakeHealthResp:
        status_code = 200

    class _FakeHttpx:
        @staticmethod
        def get(url, timeout):
            captured["health_url"] = url
            return _FakeHealthResp()

    # Replace the dynamic import inside _wait_for_health by injecting into
    # sys.modules — _wait_for_health does `import httpx as _httpx`.
    import sys

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)
    return captured


def test_ensure_started_full_happy_path(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc(pid=4242)
    captured = _patch_for_successful_start(monkeypatch, proc=proc, port=5556)

    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    fake_supervisor.ensure_started()

    # Subprocess command shape
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/llama-swap"
    assert "-config" in cmd
    assert "-watch-config" in cmd
    assert "-listen" in cmd
    assert "127.0.0.1:5556" in cmd

    # YAML and PID files written
    assert fake_supervisor.yaml_path.exists()
    assert fake_supervisor.pid_file.exists()
    pid_line = fake_supervisor.pid_file.read_text().strip()
    pid_str, port_str, _uuid = pid_line.split("|")
    assert int(pid_str) == 4242
    assert int(port_str) == 5556

    # Base URL points at the bound port
    assert fake_supervisor.base_url == "http://127.0.0.1:5556"
    assert fake_supervisor.is_started

    # Health check was issued at the bound port
    assert captured["health_url"] == "http://127.0.0.1:5556/health"


def test_ensure_started_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc()
    _patch_for_successful_start(monkeypatch, proc=proc)

    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    fake_supervisor.ensure_started()
    initial_pid = fake_supervisor._proc.pid  # type: ignore[union-attr]
    fake_supervisor.ensure_started()
    assert fake_supervisor._proc.pid == initial_pid  # type: ignore[union-attr]


def test_ensure_started_health_timeout_raises_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc(pid=4242)
    monkeypatch.setattr(ls.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(ls, "_pick_free_localhost_port", lambda: 5557)
    monkeypatch.setattr(ls, "_spawn_with_pdeathsig", lambda cmd, **kw: (proc, None))

    # Force health to never become ready by raising on every probe.
    class _FakeHttpx:
        @staticmethod
        def get(url, timeout):
            raise RuntimeError("connection refused")

    import sys

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)
    monkeypatch.setattr(fake_supervisor, "HEALTH_TIMEOUT_SECONDS", 0.5)

    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    with pytest.raises(ProviderError, match="did not become healthy"):
        fake_supervisor.ensure_started()
    assert proc.terminate_called  # cleanup ran
    assert not fake_supervisor.is_started
    assert not fake_supervisor.pid_file.exists()


def test_ensure_started_proc_dies_during_startup(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc(alive=False)  # already dead
    monkeypatch.setattr(ls.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(ls, "_pick_free_localhost_port", lambda: 5558)
    monkeypatch.setattr(ls, "_spawn_with_pdeathsig", lambda cmd, **kw: (proc, None))

    import sys

    class _NoHttpx:
        @staticmethod
        def get(url, timeout):
            raise RuntimeError("would never get here")

    monkeypatch.setitem(sys.modules, "httpx", _NoHttpx)

    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    with pytest.raises(ProviderError, match="exited before becoming healthy"):
        fake_supervisor.ensure_started()


# ---- shutdown -------------------------------------------------------------


def test_shutdown_terminates_running_proc(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc()
    _patch_for_successful_start(monkeypatch, proc=proc)
    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    fake_supervisor.ensure_started()

    fake_supervisor.shutdown()
    assert proc.terminate_called
    assert not fake_supervisor.is_started
    assert not fake_supervisor.pid_file.exists()


def test_shutdown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc()
    _patch_for_successful_start(monkeypatch, proc=proc)
    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    fake_supervisor.ensure_started()

    fake_supervisor.shutdown()
    fake_supervisor.shutdown()  # should not raise
    fake_supervisor.shutdown()


def test_shutdown_no_op_when_never_started(
    fake_supervisor: ls._LlamaSwapSupervisor,
) -> None:
    fake_supervisor.shutdown()  # must not raise


# ---- PID-file sweep -------------------------------------------------------


def test_pid_file_sweep_removes_stale_file(tmp_path: Path) -> None:
    pid_file = tmp_path / "swap.pid"
    pid_file.write_text("999999|1234|abc\n")  # PID won't be alive

    # Force "alive" check to return False so the sweep just removes the file.
    import llmfacade.providers._swap_lifecycle as mod

    original = mod._pid_alive_and_named
    mod._pid_alive_and_named = lambda pid, name: False  # type: ignore[assignment]
    try:
        ls._pid_file_sweep(pid_file)
    finally:
        mod._pid_alive_and_named = original  # type: ignore[assignment]
    assert not pid_file.exists()


def test_pid_file_sweep_kills_live_orphan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pid_file = tmp_path / "swap.pid"
    pid_file.write_text("4242|1234|abc\n")

    kills: list[int] = []

    # Process is "alive" until something appends its PID to `kills`.
    monkeypatch.setattr(ls, "_pid_alive_and_named", lambda pid, name: pid not in kills)
    monkeypatch.setattr(ls.os, "kill", lambda pid, sig: kills.append(pid))

    # Stub Windows taskkill: append PID to `kills` and return a CompletedProcess
    # with returncode 0 so the caller treats the kill as confirmed.
    def fake_run(args, **kwargs):
        if "/PID" in args:
            kills.append(int(args[args.index("/PID") + 1]))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(ls.subprocess, "run", fake_run)

    ls._pid_file_sweep(pid_file, expected_name="llama-swap")
    assert 4242 in kills
    assert not pid_file.exists()


def test_pid_file_sweep_keeps_pidfile_when_kill_unconfirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the kill returns failure (e.g. taskkill missing, ACL denied), the
    PID file must NOT be unlinked — otherwise the next sweep can't track the
    orphan."""
    pid_file = tmp_path / "swap.pid"
    pid_file.write_text("9999|1234|abc\n")

    monkeypatch.setattr(ls, "_pid_alive_and_named", lambda pid, name: True)
    monkeypatch.setattr(ls.os, "kill", lambda pid, sig: None)  # SIGTERM "succeeds"
    monkeypatch.setattr(ls.time, "sleep", lambda _: None)  # don't wait 1s on POSIX

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=1)  # failed

    monkeypatch.setattr(ls.subprocess, "run", fake_run)

    ls._pid_file_sweep(pid_file, expected_name="llama-swap")
    # POSIX path: SIGTERM appears to succeed but _pid_alive_and_named returns
    # True for all polls; final check still True → return False → file stays.
    # Win32 path: subprocess.run returncode=1 → False.
    assert pid_file.exists()


def test_pid_file_sweep_handles_unparseable_file(tmp_path: Path) -> None:
    pid_file = tmp_path / "swap.pid"
    pid_file.write_text("not-a-pid\n")
    ls._pid_file_sweep(pid_file)
    assert not pid_file.exists()


def test_pid_file_sweep_no_file_is_noop(tmp_path: Path) -> None:
    ls._pid_file_sweep(tmp_path / "missing.pid")  # must not raise


# ---- signal handler installation -----------------------------------------


def test_signal_handlers_installed_and_restored(
    monkeypatch: pytest.MonkeyPatch, fake_supervisor: ls._LlamaSwapSupervisor
) -> None:
    proc = _FakeProc()
    _patch_for_successful_start(monkeypatch, proc=proc)

    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    fake_supervisor.register(_LaunchEntry(model_id="m", gguf="x.gguf"))
    try:
        fake_supervisor.ensure_started()
        new_int = signal.getsignal(signal.SIGINT)
        assert new_int is not prior[signal.SIGINT]
        # The supervisor remembers the prior handler so it can restore it.
        assert signal.SIGINT in fake_supervisor._prior_signal_handlers
    finally:
        # Restore both SIGINT and SIGTERM so test isolation isn't broken even
        # if an assertion fails. signal.signal raises on non-main threads — use
        # contextlib.suppress to keep the teardown idempotent.
        import contextlib as _cl

        for sig, h in prior.items():
            with _cl.suppress(ValueError, OSError):
                signal.signal(sig, h)
        fake_supervisor.shutdown()


# ---- _pid_alive_and_named guards -----------------------------------------


def test_pid_alive_and_named_zero_pid_false() -> None:
    assert ls._pid_alive_and_named(0, "anything") is False


def test_pid_alive_and_named_negative_pid_false() -> None:
    assert ls._pid_alive_and_named(-1, "anything") is False
