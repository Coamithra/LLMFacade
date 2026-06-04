"""``_SdServerSupervisor`` tests against a stubbed subprocess + filesystem.

Covers registration, lazy spawn + readiness polling, swap-on-demand (a different
model stops the running process and spawns the new one), no-op when the right
model is already loaded, idempotent shutdown, the PID-file shape, and the
binary-missing / readiness-timeout error paths. Uses monkeypatch to stub
``shutil.which`` / ``_pick_free_localhost_port`` / ``_spawn_with_pdeathsig`` /
``_pid_file_sweep`` and an injected fake ``httpx`` so no real ``sd-server`` runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llmfacade.exceptions import ProviderError, ProviderNotInstalledError
from llmfacade.providers import _sd_lifecycle as sl
from llmfacade.providers._sd_launch import _SdLaunchEntry


class _FakeProc:
    def __init__(self, pid: int = 4242, *, alive: bool = True):
        self.pid = pid
        self._alive = alive
        self.terminate_called = False
        self.kill_called = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminate_called = True
        self._alive = False

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0


class _FakeHttpx:
    """Minimal httpx stand-in: ``get`` returns a response whose status_code is
    drawn from ``statuses`` (last value repeats), or raises if a value is an
    Exception. Lets a test sequence "not ready yet → ready"."""

    def __init__(self, statuses: list[int | Exception] | None = None):
        self._statuses = list(statuses or [200])
        self.calls = 0

    def get(self, url: str, timeout: float):
        idx = min(self.calls, len(self._statuses) - 1)
        self.calls += 1
        value = self._statuses[idx]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(status_code=value, text="")


@pytest.fixture
def supervisor(tmp_path: Path) -> sl._SdServerSupervisor:
    return sl._SdServerSupervisor(llmfacade_dir=tmp_path / "session")


def _entry(model_id: str = "flux", **kw) -> _SdLaunchEntry:
    kw.setdefault("diffusion_model", f"/m/{model_id}.safetensors")
    return _SdLaunchEntry(model_id=model_id, **kw)


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    procs: list[_FakeProc],
    httpx: _FakeHttpx,
    port: int = 7777,
) -> list[list[str]]:
    """Stub the spawn path. Returns a list that captures each spawned argv."""
    import sys

    spawned_argvs: list[list[str]] = []
    proc_iter = iter(procs)

    def fake_spawn(cmd, **kw):
        spawned_argvs.append(list(cmd))
        return next(proc_iter), None

    monkeypatch.setattr(sl.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(sl, "_pick_free_localhost_port", lambda: port)
    monkeypatch.setattr(sl, "_spawn_with_pdeathsig", fake_spawn)
    monkeypatch.setattr(sl, "_pid_file_sweep", lambda *a, **k: None)
    monkeypatch.setattr(sl.time, "sleep", lambda _t: None)
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    return spawned_argvs


# ---- registration ---------------------------------------------------------


def test_register_then_entries(supervisor: sl._SdServerSupervisor) -> None:
    e = _entry("flux")
    supervisor.register(e)
    assert supervisor.entries == [e]


def test_register_same_id_same_params_is_noop(supervisor: sl._SdServerSupervisor) -> None:
    e = _entry("flux", diffusion_fa=True)
    supervisor.register(e)
    supervisor.register(e)
    assert len(supervisor.entries) == 1


def test_register_same_id_different_params_raises(supervisor: sl._SdServerSupervisor) -> None:
    supervisor.register(_entry("flux", diffusion_fa=True))
    with pytest.raises(ValueError, match="already registered with different launch params"):
        supervisor.register(_entry("flux", diffusion_fa=False))


# ---- ensure_model: spawn / no-op / swap -----------------------------------


def test_ensure_model_spawns_and_returns_base_url(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc()
    argvs = _patch_spawn(monkeypatch, procs=[proc], httpx=_FakeHttpx([200]), port=7777)
    supervisor.register(_entry("flux"))

    base = supervisor.ensure_model("flux")

    assert base == "http://127.0.0.1:7777"
    assert supervisor.is_started
    assert supervisor.current_model_id == "flux"
    assert argvs[0][0].endswith("sd-server")
    assert "--diffusion-model" in argvs[0]


def test_ensure_model_unregistered_raises(supervisor: sl._SdServerSupervisor) -> None:
    with pytest.raises(ValueError, match="is not registered"):
        supervisor.ensure_model("nope")


def test_ensure_model_same_model_is_noop(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc()
    _patch_spawn(monkeypatch, procs=[proc], httpx=_FakeHttpx([200]))
    supervisor.register(_entry("flux"))

    supervisor.ensure_model("flux")
    supervisor.ensure_model("flux")  # second call must not respawn

    assert proc.terminate_called is False
    assert supervisor._proc is proc


def test_ensure_model_swaps_stops_old_spawns_new(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc_a = _FakeProc(pid=1)
    proc_b = _FakeProc(pid=2)
    argvs = _patch_spawn(monkeypatch, procs=[proc_a, proc_b], httpx=_FakeHttpx([200]))
    supervisor.register(_entry("flux"))
    supervisor.register(_entry("sdxl"))

    supervisor.ensure_model("flux")
    supervisor.ensure_model("sdxl")

    assert proc_a.terminate_called is True  # old process was stopped on swap
    assert supervisor._proc is proc_b
    assert supervisor.current_model_id == "sdxl"
    assert len(argvs) == 2


def test_ensure_model_waits_until_ready(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc()
    # First poll 503 (warming up), then 200.
    httpx = _FakeHttpx([503, 200])
    _patch_spawn(monkeypatch, procs=[proc], httpx=httpx)
    supervisor.register(_entry("flux"))

    supervisor.ensure_model("flux")

    assert httpx.calls >= 2


# ---- error paths ----------------------------------------------------------


def test_missing_binary_raises(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    monkeypatch.setattr(sl.shutil, "which", lambda b: None)
    supervisor.register(_entry("flux"))
    with pytest.raises(ProviderNotInstalledError, match="sd-server"):
        supervisor.ensure_model("flux")


def test_ready_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc()
    _patch_spawn(monkeypatch, procs=[proc], httpx=_FakeHttpx([ConnectionError("refused")]))
    supervisor.startup_timeout = 0.0  # deadline already passed → straight to raise
    supervisor.register(_entry("flux"))
    with pytest.raises(ProviderError, match="did not become ready"):
        supervisor.ensure_model("flux")


def test_proc_dies_before_ready_raises(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    dead = _FakeProc(alive=False)
    _patch_spawn(monkeypatch, procs=[dead], httpx=_FakeHttpx([200]))
    supervisor.register(_entry("flux"))
    with pytest.raises(ProviderError, match="exited before becoming ready"):
        supervisor.ensure_model("flux")


# ---- pidfile + shutdown ---------------------------------------------------


def test_pid_file_written_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc(pid=9988)
    _patch_spawn(monkeypatch, procs=[proc], httpx=_FakeHttpx([200]), port=7001)
    supervisor.register(_entry("flux"))
    supervisor.ensure_model("flux")

    raw = supervisor.pid_file.read_text(encoding="utf-8").strip()
    pid, port, model, _uuid = raw.split("|")
    assert pid == "9988"
    assert port == "7001"
    assert model == "flux"


def test_shutdown_is_idempotent_and_unlinks_pidfile(
    monkeypatch: pytest.MonkeyPatch, supervisor: sl._SdServerSupervisor
) -> None:
    proc = _FakeProc()
    _patch_spawn(monkeypatch, procs=[proc], httpx=_FakeHttpx([200]))
    supervisor.register(_entry("flux"))
    supervisor.ensure_model("flux")
    assert supervisor.pid_file.exists()

    supervisor.shutdown()
    supervisor.shutdown()  # second call must be a no-op, not raise

    assert proc.terminate_called is True
    assert not supervisor.pid_file.exists()
    assert supervisor.is_started is False
