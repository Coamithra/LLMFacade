"""sd-server subprocess supervisor for the localimage provider's managed mode.

The image analog of ``_swap_lifecycle.py``, but simpler: stable-diffusion.cpp's
``sd-server`` is strictly single-model-per-process (the model loads once at
startup, before the server listens) and there is no ``llama-swap`` equivalent for
images. So ``_SdServerSupervisor`` owns **at most one** ``sd-server`` process and
swaps on demand — when a request targets a different registered model than the one
currently loaded, it tears the running process down and spawns the new one.

The OS-level process-hardening primitives (kill-on-parent-death, free-port pick,
PID-file sweep, hard kill-tree) are generic and shared with the llama-swap
supervisor; we import them from ``_swap_lifecycle`` rather than duplicate them.
Importing them as module globals here also lets this supervisor's tests
monkeypatch them on this module, the same way the llamacpp supervisor tests do.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from llmfacade.exceptions import ProviderError, ProviderNotInstalledError
from llmfacade.providers._sd_launch import _SdLaunchEntry, build_sd_server_argv

# Generic process primitives shared with the llama-swap supervisor. Bound as
# module globals so this module's tests can monkeypatch them here.
from llmfacade.providers._swap_lifecycle import (
    _hard_kill_tree,
    _pick_free_localhost_port,
    _pid_file_sweep,
    _spawn_with_pdeathsig,
)

__all__ = ["_SdServerSupervisor"]


class _SdServerSupervisor:
    """Owns at most one ``sd-server`` subprocess, its PID file, and its log.

    Lifecycle:

    1. Construct: records ``llmfacade_dir`` and an empty entry registry. No
       filesystem or process side-effects.
    2. ``register(entry)`` adds a launch entry (in-memory only).
    3. ``ensure_model(model_id)`` does the lazy/swap spawn: if the requested model
       is already loaded and alive, return its base URL; otherwise stop whatever
       is running, sweep any prior-run orphan, allocate a port, spawn sd-server
       with kill-on-parent-death, write the PID file, and poll ``/v1/models``
       until ready (or raise ``ProviderError``).
    4. ``shutdown()`` is idempotent — atexit + signal handlers + explicit calls
       all funnel here. SIGTERM with a timeout, then SIGKILL.
    """

    # sd-server loads the whole model into VRAM before it begins listening, which
    # for large Flux/SDXL checkpoints can take well over a minute. Generous by
    # default; overridable via the provider's ``startup_timeout=``.
    STARTUP_TIMEOUT_SECONDS: float = 300.0

    def __init__(
        self,
        *,
        llmfacade_dir: Path,
        binary: str = "sd-server",
        startup_timeout: float | None = None,
    ) -> None:
        self.llmfacade_dir = Path(llmfacade_dir)
        self.binary = binary
        if startup_timeout is not None:
            self.startup_timeout = float(startup_timeout)
        else:
            self.startup_timeout = self.STARTUP_TIMEOUT_SECONDS

        self._entries: dict[str, _SdLaunchEntry] = {}
        # Reentrant: shutdown() can be invoked from a signal handler while the
        # main thread holds the same lock mid-ensure_model; without reentrancy
        # that would deadlock.
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._anchor: Any = None  # Win32 job handle, kept alive for the run
        self._port: int | None = None
        self._current_model_id: str | None = None
        self._log_file: Any = None
        self._session_uuid: str = uuid.uuid4().hex
        self._prior_signal_handlers: dict[int, Any] = {}
        self._atexit_registered = False
        self._signal_handlers_installed = False
        self._shutdown_done = False

    # --- accessors ---------------------------------------------------------

    @property
    def is_started(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def base_url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}"

    @property
    def current_model_id(self) -> str | None:
        return self._current_model_id

    @property
    def pid_file(self) -> Path:
        return self.llmfacade_dir / "sd-server.pid"

    @property
    def log_path(self) -> Path:
        return self.llmfacade_dir / "logs" / "sd-server.log"

    @property
    def entries(self) -> list[_SdLaunchEntry]:
        return list(self._entries.values())

    # --- registration ------------------------------------------------------

    def register(self, entry: _SdLaunchEntry) -> None:
        """Add a launch entry. Re-registering the same ``model_id`` with identical
        params is a no-op; a clash with different params raises (name aliasing
        would silently route to the wrong backend)."""
        with self._lock:
            existing = self._entries.get(entry.model_id)
            if existing is not None:
                if existing != entry:
                    raise ValueError(
                        f"image model name {entry.model_id!r} already registered with "
                        f"different launch params: existing={existing!r} new={entry!r}"
                    )
                return
            self._entries[entry.model_id] = entry

    # --- lazy / swap startup ----------------------------------------------

    def ensure_model(self, model_id: str) -> str:
        """Ensure ``model_id``'s ``sd-server`` is the one running, and return its
        base URL (``http://127.0.0.1:<port>``). Spawns lazily; swaps (stop old →
        spawn new) when a different model is currently loaded; no-ops when the
        right model is already alive."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                names = sorted(self._entries)
                raise ValueError(
                    f"image model {model_id!r} is not registered; "
                    f"registered: {names!r}. Call provider.new_image_model(...) first."
                )
            if self.is_started and self._current_model_id == model_id:
                base = self.base_url
                assert base is not None
                return base
            # Wrong model loaded (or nothing running): swap.
            self._stop_current_locked(graceful=True)
            self._spawn_locked(entry)
            base = self.base_url
            if base is None:
                raise ProviderError("sd-server supervisor started but has no base URL")
            return base

    def _spawn_locked(self, entry: _SdLaunchEntry) -> None:
        binary_path = shutil.which(self.binary)
        if binary_path is None:
            raise ProviderNotInstalledError(
                f"{self.binary!r} binary not found on PATH. Build it from "
                "https://github.com/leejet/stable-diffusion.cpp (the "
                "examples/server target builds 'sd-server') and ensure it's on "
                "PATH, or use external mode by passing base_url= to point at an "
                "already-running image server."
            )

        self.llmfacade_dir.mkdir(parents=True, exist_ok=True)
        (self.llmfacade_dir / "logs").mkdir(parents=True, exist_ok=True)
        _pid_file_sweep(self.pid_file, expected_name=self.binary)

        port = _pick_free_localhost_port()
        argv = build_sd_server_argv(binary_path, entry, port=port)

        self._log_file = self.log_path.open("wb")
        try:
            proc, anchor = _spawn_with_pdeathsig(
                argv,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            self._close_log()
            raise ProviderNotInstalledError(f"Failed to spawn {self.binary!r}: {e}") from e

        self._proc = proc
        self._anchor = anchor
        self._port = port
        self._current_model_id = entry.model_id
        # A respawn after an explicit shutdown() must be cleaned up again, so
        # re-arm the idempotency latch (atexit/shutdown become effective once more).
        self._shutdown_done = False
        self._write_pid_file()
        self._install_exit_hooks()

        try:
            self._wait_for_ready(port)
        except Exception:
            self._cleanup_after_failure()
            raise

    def _wait_for_ready(self, port: int) -> None:
        """Poll ``GET /v1/models`` until it answers ``<400``. sd-server has no
        ``/health`` endpoint, but it only begins listening *after* the model is
        loaded, so a ``/v1/models`` response means it's ready to generate."""
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        try:
            import httpx as _httpx
        except ImportError as e:
            raise ProviderNotInstalledError(
                "httpx not installed (required for the localimage managed mode). "
                "Run: pip install llmfacade[localimage]"
            ) from e

        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                tail = self._tail_log()
                raise ProviderError(f"sd-server exited before becoming ready. log tail:\n{tail}")
            try:
                resp = _httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
                if resp.status_code < 400:
                    return
                last_error = ProviderError(f"/v1/models returned {resp.status_code}")
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(0.25)
        tail = self._tail_log()
        raise ProviderError(
            f"sd-server did not become ready within {self.startup_timeout}s "
            f"(last error: {last_error}). log tail:\n{tail}"
        )

    # --- teardown ----------------------------------------------------------

    def _stop_current_locked(self, *, graceful: bool = True) -> None:
        """Stop the running sd-server (if any) and reset proc/port/current state.
        Graceful: terminate, wait, then kill on timeout. Non-graceful: hard
        kill-tree immediately (used by shutdown's force path)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            if graceful:
                with contextlib.suppress(Exception):
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(Exception):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=5)
            else:
                _hard_kill_tree(proc.pid, proc)
        self._proc = None
        self._anchor = None
        self._port = None
        self._current_model_id = None
        self._close_log()

    def _cleanup_after_failure(self) -> None:
        self._stop_current_locked(graceful=True)
        with contextlib.suppress(OSError):
            self.pid_file.unlink()

    def shutdown(self) -> None:
        """Tear down the managed subprocess. Idempotent — atexit, signal handlers,
        and explicit calls all funnel here."""
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            self._stop_current_locked(graceful=True)
            with contextlib.suppress(OSError):
                self.pid_file.unlink()

    # --- pidfile + log -----------------------------------------------------

    def _write_pid_file(self) -> None:
        if self._proc is None or self._port is None:
            return
        # owner= records *our* (Python) PID so a sibling process sharing this
        # llmfacade_dir can tell a live session's sd-server from a true orphan
        # (see _swap_lifecycle._pid_file_sweep).
        line = (
            f"{self._proc.pid}|{self._port}|{self._current_model_id}|"
            f"{self._session_uuid}|owner={os.getpid()}\n"
        )
        self.pid_file.write_text(line, encoding="utf-8")

    def _close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None

    def _tail_log(self, *, max_bytes: int = 4096) -> str:
        if not self.log_path.exists():
            return "(no log)"
        try:
            with self.log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return "(log unreadable)"

    # --- exit hooks --------------------------------------------------------

    def _install_exit_hooks(self) -> None:
        # Install once per supervisor, not on every swap: re-running this would
        # re-read our own handler as the "prior" one and chain to itself on the
        # next signal (infinite recursion). `_spawn_locked` calls it each spawn.
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
        if self._signal_handlers_installed:
            return
        # signal.signal() only works on the main thread. If driven from a worker
        # (asyncio.to_thread, a background thread), skip — atexit still covers
        # normal exit.
        if threading.current_thread() is not threading.main_thread():
            return
        self._signal_handlers_installed = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError, AttributeError):
                prior = signal.getsignal(sig)
                self._prior_signal_handlers[sig] = prior
                signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.shutdown()
        prior = self._prior_signal_handlers.get(signum)
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signum, prior if prior is not None else signal.SIG_DFL)
        if callable(prior) and prior not in (signal.SIG_DFL, signal.SIG_IGN):
            prior(signum, frame)
            return
        if prior == signal.SIG_DFL:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            if sys.platform == "win32":
                raise SystemExit(128 + signum)
            os.kill(os.getpid(), signum)

    def __del__(self) -> None:
        # Best-effort safety net; atexit + signal handlers are the primary path.
        if self._shutdown_done:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            with contextlib.suppress(Exception):
                self.shutdown()
        finally:
            with contextlib.suppress(RuntimeError):
                self._lock.release()
