"""llama-swap subprocess supervisor for the llamacpp provider's managed mode.

Three concerns live here:

* **YAML rendering** (`_render_swap_yaml`) — turns a list of `_LaunchEntry`
  values into the YAML llama-swap consumes. Pure function; trivially testable.
* **Subprocess lifecycle** (`_LlamaSwapSupervisor`) — lazy spawn on first
  request, idempotent shutdown on exit. Owns the PID file and keeps `swap.yaml`
  on disk so llama-swap's `-watch-config` can pick up later edits without us
  needing to API-call it.
* **Shutdown defense in depth** (`_spawn_with_pdeathsig` + `_pid_file_sweep`) —
  OS-level kernel guarantees on Windows (Job Object with kill-on-job-close)
  and Linux (`prctl(PR_SET_PDEATHSIG)`), plus a PID-file sweep on the next
  start that reaps any orphan that survived a hard kill or macOS exit.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llmfacade.exceptions import ProviderError, ProviderNotInstalledError
from llmfacade.providers._launch import _LaunchEntry

__all__ = [
    "_LlamaSwapSupervisor",
    "_spawn_with_pdeathsig",
    "_pid_file_sweep",
    "_render_swap_yaml",
    "_pid_alive",
    "_pid_alive_and_named",
    "_signal_reload",
]


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def _build_llama_server_cmd(entry: _LaunchEntry) -> str:
    """Build the llama-server CLI string for one entry. ``${PORT}`` is left
    literal so llama-swap substitutes its allocated port at spawn time."""
    parts: list[str] = ["llama-server", "--model", entry.gguf, "--port", "${PORT}"]
    if entry.jinja:
        # Render the GGUF's embedded chat template instead of built-in format
        # detection — prerequisite for the enable_thinking template kwarg and
        # correct tool-calling on newer Gemma 4 / Qwen3 quants. Boolean flag.
        parts.append("--jinja")
    if entry.context_size is not None:
        parts += ["--ctx-size", str(entry.context_size)]
    if entry.cache_type_k is not None:
        parts += ["--cache-type-k", entry.cache_type_k]
    if entry.cache_type_v is not None:
        parts += ["--cache-type-v", entry.cache_type_v]
    if entry.n_gpu_layers is not None:
        parts += ["--n-gpu-layers", str(entry.n_gpu_layers)]
    if entry.n_cpu_moe is not None:
        parts += ["--n-cpu-moe", str(entry.n_cpu_moe)]
    if entry.parallel is not None:
        parts += ["--parallel", str(entry.parallel)]
    if entry.flash_attn is not None:
        parts += ["--flash-attn", entry.flash_attn]
    if entry.mmproj_path is not None:
        parts += ["--mmproj", entry.mmproj_path]
    if entry.slot_save_path is not None:
        parts += ["--slot-save-path", entry.slot_save_path]
    if entry.fit is True:
        parts += ["--fit", "on"]
    elif entry.fit is False:
        parts += ["--fit", "off"]
    if entry.fit_target is not None:
        parts += ["--fit-target", ",".join(str(v) for v in entry.fit_target)]
    if entry.fit_ctx is not None:
        parts += ["--fit-ctx", str(entry.fit_ctx)]
    if entry.no_mmap:
        # Read the whole model into RAM up front instead of mmap'ing it, so a
        # low-VRAM MoE with experts in system RAM doesn't demand-page them from
        # disk mid-token. Boolean flag; requires the model to fit in RAM.
        parts.append("--no-mmap")
    if entry.mlock:
        # Pin RAM-resident pages so the kernel can't page them back to disk on
        # a long-running server (the managed-mode "day-3 slowdown" fix). Boolean
        # flag; silently no-ops under Docker without IPC_LOCK + memlock ulimit.
        parts.append("--mlock")
    parts.extend(entry.extra_args)

    quoted: list[str] = []
    for p in parts:
        # Leave the literal ${PORT} placeholder unquoted so llama-swap sees
        # it as a bare token. Otherwise quote only when there's whitespace or
        # a shell metacharacter — shlex.quote() wraps anything containing a
        # backslash in POSIX single quotes, which on Windows means every
        # absolute path. llama-swap then forwards the quoted token to
        # llama-server, which sees the literal single quotes as part of the
        # path and exits 1 on the first arg that's a path.
        if p == "${PORT}" or not _needs_quoting(p):
            quoted.append(p)
        else:
            quoted.append(shlex.quote(p))
    return " ".join(quoted)


_SHELL_METACHARS = frozenset(" \t\n\"'`$&|;<>()*?[]{}#!~")


def _needs_quoting(token: str) -> bool:
    if not token:
        return True
    return any(c in _SHELL_METACHARS for c in token)


def _render_swap_yaml(
    entries: Iterable[_LaunchEntry],
    *,
    global_ttl: int = 0,
    health_check_timeout: int = 60,
) -> str:
    """Render llama-swap config YAML for ``entries``. Deterministic: entries
    are emitted in the order given, keys within each model in a fixed order,
    and pyyaml is configured with ``sort_keys=False`` and ``default_flow_style=False``
    so output is line-stable for snapshot tests.

    ``global_ttl`` is the fallback TTL applied to any entry without its own
    ``ttl``; ``0`` means "never unload" in llama-swap's vocabulary."""
    try:
        import yaml as _yaml
    except ImportError as e:
        raise ProviderNotInstalledError(
            "pyyaml not installed (required for llamacpp managed mode). "
            "Run: pip install llmfacade[llamacpp]"
        ) from e

    models: dict[str, dict[str, Any]] = {}
    for entry in entries:
        cmd = _build_llama_server_cmd(entry)
        block: dict[str, Any] = {"cmd": cmd}
        ttl_value = entry.ttl if entry.ttl is not None else global_ttl
        block["ttl"] = ttl_value
        models[entry.model_id] = block

    doc: dict[str, Any] = {
        "healthCheckTimeout": health_check_timeout,
        "models": models,
    }
    return _yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Process spawning with kill-on-parent-death where the OS supports it
# ---------------------------------------------------------------------------


def _spawn_with_pdeathsig(cmd: list[str], **popen_kwargs: Any) -> tuple[subprocess.Popen, Any]:
    """Spawn ``cmd`` so the OS kernel kills it when our process exits.

    Returns ``(popen, anchor)`` where ``anchor`` is whatever object must stay
    referenced for the kill-on-death guarantee to hold (the Win32 Job Object
    handle on Windows; ``None`` on Linux/macOS). The caller stores it on the
    supervisor instance so it isn't garbage-collected mid-run."""
    if sys.platform == "win32":
        return _spawn_win32_jobobject(cmd, **popen_kwargs)
    if sys.platform.startswith("linux"):
        return _spawn_linux_pdeathsig(cmd, **popen_kwargs)
    # macOS and other POSIX without prctl: best-effort plain Popen. Layers 2
    # (signal handlers) and 3 (PID-file sweep) catch the orphan path.
    return subprocess.Popen(cmd, **popen_kwargs), None


def _spawn_linux_pdeathsig(cmd: list[str], **popen_kwargs: Any) -> tuple[subprocess.Popen, None]:
    import ctypes

    PR_SET_PDEATHSIG = 1  # noqa: N806

    def _preexec() -> None:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)

    popen_kwargs.setdefault("preexec_fn", _preexec)
    return subprocess.Popen(cmd, **popen_kwargs), None


def _spawn_win32_jobobject(cmd: list[str], **popen_kwargs: Any) -> tuple[subprocess.Popen, Any]:
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000  # noqa: N806
    JobObjectExtendedLimitInformation = 9  # noqa: N806

    class IO_COUNTERS(ctypes.Structure):  # noqa: N801
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        # Couldn't create the job object — fall back to plain Popen and rely on
        # layers 2 + 3.
        return subprocess.Popen(cmd, **popen_kwargs), None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(job)
        return subprocess.Popen(cmd, **popen_kwargs), None

    flags = popen_kwargs.pop("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, creationflags=flags, **popen_kwargs)

    PROCESS_SET_QUOTA = 0x0100  # noqa: N806
    PROCESS_TERMINATE = 0x0001  # noqa: N806
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, proc.pid)
    if not handle or not kernel32.AssignProcessToJobObject(job, handle):
        # Couldn't assign — close the job (no-op since nothing's in it) and
        # carry on; layers 2 + 3 will catch this.
        if handle:
            kernel32.CloseHandle(handle)
        kernel32.CloseHandle(job)
        return proc, None
    kernel32.CloseHandle(handle)
    # Returning the job handle as the anchor: the caller must keep it
    # referenced for the lifetime of the process. When this last handle to the
    # job closes (any process exit including SIGKILL), the OS kills llama-swap.
    return proc, job


# ---------------------------------------------------------------------------
# PID-file sweep — orphan recovery from prior runs
# ---------------------------------------------------------------------------


def _pid_alive_and_named(pid: int, expected_name: str) -> bool:
    """True iff PID ``pid`` is alive AND its image name matches ``expected_name``.

    We require the image-name match so a recycled PID belonging to some unrelated
    process doesn't get SIGTERM'd."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
        text = output.decode("utf-8", errors="replace").lower()
        return expected_name.lower() in text
    if sys.platform.startswith("linux"):
        path = Path(f"/proc/{pid}/comm")
        if not path.exists():
            return False
        try:
            comm = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return False
        return expected_name in comm
    # macOS / other POSIX
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "comm="],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
    return expected_name in output.decode("utf-8", errors="replace").strip()


def _pid_alive(pid: int) -> bool:
    """True iff a process with PID ``pid`` is alive. No image-name check — this
    probes the *owner* (Python) process recorded in a PID file, whose image name
    we can't predict (python.exe, pythonw.exe, an embedding app). PID reuse can
    therefore false-positive "alive", but that only makes the sweep *skip* a
    kill — it can never kill the wrong process — so it's the safe direction.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Never os.kill(pid, 0) on Windows: Python implements non-CTRL signals
        # there via TerminateProcess, which would kill the probed process.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # noqa: N806
        STILL_ACTIVE = 259  # noqa: N806
        ERROR_ACCESS_DENIED = 5  # noqa: N806
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the process exists but isn't ours to open.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 probes existence without delivering
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists but owned by someone else
    return True


def _parse_owner_pid(fields: list[str]) -> int | None:
    """Extract the ``owner=<pid>`` field written by ``_write_pid_file``.
    Returns None for legacy/corrupt files (no such field, or unparseable),
    which the sweep treats as an orphan — preserving pre-owner-field cleanup
    behaviour for stale files."""
    for field in fields:
        if field.startswith("owner="):
            try:
                return int(field[len("owner=") :])
            except ValueError:
                return None
    return None


def _pid_file_sweep(pid_file: Path, *, expected_name: str = "llama-swap") -> None:
    """If ``pid_file`` exists and its PID is alive (and matches the expected
    image name), SIGTERM it and remove the file. If the PID is dead/unknown,
    just remove the file.

    Exception: when the file records an ``owner=<pid>`` whose Python process is
    still alive (another live session sharing this llmfacade_dir), the recorded
    server is *not* an orphan — skip the kill, warn, and leave the sibling's
    file untouched. A legacy file without the owner field is treated as an
    orphan, matching the old behaviour."""
    if not pid_file.exists():
        return
    try:
        raw = pid_file.read_text(encoding="utf-8")
    except OSError:
        return
    fields = [f.strip() for f in raw.split("|")]
    try:
        pid = int(fields[0])
    except (TypeError, ValueError):
        with contextlib.suppress(OSError):
            pid_file.unlink()
        return
    owner_pid = _parse_owner_pid(fields[1:])
    if owner_pid is not None and owner_pid != os.getpid() and _pid_alive(owner_pid):
        # A live sibling session owns this directory. Killing its server would
        # abort its in-flight generations, so leave both the process and its
        # PID file alone. (Our own caller will then overwrite the file with our
        # record — last-writer-wins is the least-harm option for a config that
        # is already degraded: the sibling's primary cleanup is its own
        # atexit/Job-Object/pdeathsig layers, not this file.) An owner equal to
        # our own PID means a prior spawn in this process leaked its file —
        # that IS our orphan, so fall through and sweep it.
        warnings.warn(
            f"PID file {pid_file} is owned by another live process "
            f"(PID {owner_pid}); skipping the orphan sweep. Two sessions appear "
            "to share the same llmfacade_dir — give each its own directory.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if _pid_alive_and_named(pid, expected_name):
        killed = _try_kill(pid, expected_name)
        if not killed:
            # Couldn't confirm the kill (no taskkill on PATH, ACL denied, exotic
            # signal masking). Leave the PID file in place so the next sweep can
            # retry — removing it would silently lose track of the orphan.
            return
    with contextlib.suppress(OSError):
        pid_file.unlink()


def _try_kill(pid: int, expected_name: str) -> bool:
    """Best-effort kill of ``pid``. Returns True iff the process is gone
    afterward (either we killed it or it was dying anyway), False if we
    couldn't confirm. Caller decides what to do on False."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
        if result.returncode != 0:
            return False
        return not _pid_alive_and_named(pid, expected_name)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return not _pid_alive_and_named(pid, expected_name)
    for _ in range(10):
        time.sleep(0.1)
        if not _pid_alive_and_named(pid, expected_name):
            return True
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGKILL)
    return not _pid_alive_and_named(pid, expected_name)


def _hard_kill_tree(pid: int, proc: subprocess.Popen) -> None:
    """Force-kill ``pid`` and its child process tree *immediately* — no SIGTERM
    grace period, no waiting for an in-flight request to drain. This is the
    instant-abort path behind ``_LlamaSwapSupervisor.interrupt()``; contrast
    ``shutdown()`` / ``_try_kill`` which terminate politely first.

    Windows: ``taskkill /PID <pid> /T /F`` force-kills the whole tree (llama-swap
    plus the llama-server children it spawned), then ``proc.kill()`` as a
    belt-and-braces TerminateProcess in case taskkill isn't on PATH. POSIX:
    SIGKILL the supervised pid directly via ``proc.kill()``. We deliberately do
    NOT ``killpg`` — the supervisor spawns llama-swap in the parent's process
    group, so killing the group would take the host process down with it. (On
    POSIX, orphaned llama-server children are a known limitation of the instant
    path; the supported managed-mode target is Windows, where the tree kill is
    clean.) Best-effort; never raises."""
    if sys.platform == "win32":
        with contextlib.suppress(subprocess.SubprocessError, FileNotFoundError, OSError):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        with contextlib.suppress(Exception):
            proc.kill()
        return
    with contextlib.suppress(Exception):
        proc.kill()


# ---------------------------------------------------------------------------
# Reload signalling
# ---------------------------------------------------------------------------


def _signal_reload(pid: int) -> bool:
    """SIGHUP ``pid`` to ask llama-swap to reload its config now, instead of
    waiting up to 2s for the next ``-watch-config`` poll. Returns True if the
    signal was sent, False on Windows (no SIGHUP) or if the kill failed."""
    if sys.platform == "win32" or not hasattr(signal, "SIGHUP"):
        return False
    try:
        os.kill(pid, signal.SIGHUP)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# Supervisor — owns one llama-swap subprocess
# ---------------------------------------------------------------------------


def _pick_free_localhost_port() -> int:
    """Bind to ``127.0.0.1:0``, read the assigned port, release. Race-prone in
    principle (another process could grab the same port between release and
    spawn) but the windows is a few microseconds and llama-swap will fail
    loudly if it can't bind, so we'd notice."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class _LlamaSwapSupervisor:
    """Owns one llama-swap subprocess, its YAML config, and its PID file.

    Lifecycle:

    1. Construct: records `llmfacade_dir` and the entries list (in-memory only).
       No filesystem or process side-effects.
    2. `register(entry)` / `unregister(model_id)` mutate the entries list and,
       once the supervisor is started, rewrite the YAML so llama-swap's
       `-watch-config` picks it up.
    3. `ensure_started()` does the lazy spawn: mkdir the session dir, sweep any
       orphan from a prior run, generate the YAML, allocate a port, spawn
       llama-swap with kill-on-parent-death, write the PID file, poll
       `/health` until ready (or raise `ProviderError`).
    4. `shutdown()` is idempotent — atexit + signal handlers + explicit calls
       all funnel here. SIGTERM with a 10s timeout, then SIGKILL.
    """

    HEALTH_TIMEOUT_SECONDS: float = 60.0

    def __init__(
        self,
        *,
        llmfacade_dir: Path,
        global_ttl: int = 0,
        binary: str = "llama-swap",
    ) -> None:
        self.llmfacade_dir = Path(llmfacade_dir)
        self.global_ttl = int(global_ttl)
        self.binary = binary

        self._entries: list[_LaunchEntry] = []
        # Reentrant: shutdown() can be invoked from a signal handler while the
        # main thread is mid-`ensure_started()` and holds the same lock; without
        # reentrancy that would deadlock.
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._anchor: Any = None  # Win32 job handle, kept alive for the run
        self._port: int | None = None
        self._session_uuid: str = uuid.uuid4().hex
        self._log_file: Any = None
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
    def yaml_path(self) -> Path:
        return self.llmfacade_dir / "swap.yaml"

    @property
    def pid_file(self) -> Path:
        return self.llmfacade_dir / "swap.pid"

    @property
    def log_path(self) -> Path:
        return self.llmfacade_dir / "logs" / "llamacpp-swap.log"

    @property
    def entries(self) -> list[_LaunchEntry]:
        return list(self._entries)

    # --- registration ------------------------------------------------------

    def register(self, entry: _LaunchEntry) -> None:
        """Append an entry. If two registrations have the same `model_id` but
        different launch knobs, raise — name aliasing would silently route to
        the wrong backend.

        When the supervisor is already running, rewrite the YAML, nudge
        llama-swap with SIGHUP (POSIX) and then block on `/v1/models` until the
        new entry is visible. Without the wait, an immediate ``send()`` races
        the 2s ``-watch-config`` poll and gets a 400 'could not find suitable
        inference handler'."""
        port_for_wait: int | None = None
        pid_for_signal: int | None = None
        with self._lock:
            for existing in self._entries:
                if existing.model_id == entry.model_id:
                    if existing != entry:
                        raise ValueError(
                            f"model name {entry.model_id!r} already registered with "
                            f"different launch params: existing={existing!r} new={entry!r}"
                        )
                    return
            self._entries.append(entry)
            if self.is_started:
                self._write_yaml()
                port_for_wait = self._port
                pid_for_signal = self._proc.pid if self._proc is not None else None
        if port_for_wait is not None:
            if pid_for_signal is not None:
                _signal_reload(pid_for_signal)
            self._wait_for_model_visible(port_for_wait, entry.model_id)

    # --- lazy startup ------------------------------------------------------

    def ensure_started(self) -> None:
        with self._lock:
            if self.is_started:
                return
            self._start_locked()

    def _start_locked(self) -> None:
        if not self._entries:
            raise ProviderError(
                "managed-mode llamacpp provider has no models registered; "
                "call provider.new_model(gguf=...) first."
            )
        binary_path = shutil.which(self.binary)
        if binary_path is None:
            raise ProviderNotInstalledError(
                f"{self.binary!r} binary not found on PATH. Install from "
                "https://github.com/mostlygeek/llama-swap (e.g. `go install "
                "github.com/mostlygeek/llama-swap@latest`) and ensure it's on "
                "PATH, or use external mode by passing base_url= to point at "
                "an existing llama-server."
            )

        self.llmfacade_dir.mkdir(parents=True, exist_ok=True)
        (self.llmfacade_dir / "logs").mkdir(parents=True, exist_ok=True)
        # llama-server validates --slot-save-path as an existing directory at
        # parse time and exits 1 if it's missing. Pre-create every referenced
        # slot dir before spawning so that doesn't kill the upstream.
        for entry in self._entries:
            if entry.slot_save_path:
                Path(entry.slot_save_path).mkdir(parents=True, exist_ok=True)
        _pid_file_sweep(self.pid_file, expected_name=self.binary)

        self._write_yaml()
        port = _pick_free_localhost_port()

        self._log_file = self.log_path.open("wb")
        cmd = [
            binary_path,
            "-config",
            str(self.yaml_path),
            "-watch-config",
            "-listen",
            f"127.0.0.1:{port}",
        ]
        try:
            proc, anchor = _spawn_with_pdeathsig(
                cmd,
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
        # A respawn after an explicit shutdown() must be cleaned up again, so
        # re-arm the idempotency latch (atexit/shutdown become effective once
        # more). interrupt() never sets the latch, so the interrupt-then-respawn
        # path is unaffected.
        self._shutdown_done = False
        self._write_pid_file()
        self._install_exit_hooks()

        try:
            self._wait_for_health(port)
        except Exception:
            self._cleanup_after_failure()
            raise

    def _wait_for_health(self, port: int) -> None:
        deadline = time.monotonic() + self.HEALTH_TIMEOUT_SECONDS
        last_error: Exception | None = None
        # Lazy import — httpx is the optional dep, and importing it here keeps
        # the supervisor unit-testable without installing it.
        try:
            import httpx as _httpx
        except ImportError as e:
            raise ProviderNotInstalledError(
                "httpx not installed (required for the llamacpp provider). "
                "Run: pip install llmfacade[llamacpp]"
            ) from e

        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                tail = self._tail_log()
                raise ProviderError(
                    f"llama-swap exited before becoming healthy. log tail:\n{tail}"
                )
            try:
                resp = _httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                # llama-swap returns 200 once a backend is loaded; while no
                # backend is active it answers 200 too. 4xx/5xx mean something
                # is wrong (wrong listen port, malformed config). Treat <400 as
                # healthy so 503-during-warmup doesn't get false-positive'd.
                if resp.status_code < 400:
                    return
                last_error = ProviderError(f"/health returned {resp.status_code}")
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(0.25)
        tail = self._tail_log()
        raise ProviderError(
            f"llama-swap did not become healthy within {self.HEALTH_TIMEOUT_SECONDS}s "
            f"(last error: {last_error}). log tail:\n{tail}"
        )

    MODEL_VISIBLE_TIMEOUT_SECONDS: float = 10.0

    def _wait_for_model_visible(self, port: int, model_id: str) -> None:
        """Block until ``model_id`` appears in ``GET /v1/models``. Used after
        rewriting the YAML on a running supervisor: llama-swap's watcher polls
        every 2s, so a request fired immediately after ``register()`` would
        otherwise race the watcher and return 400 'could not find suitable
        inference handler'."""
        deadline = time.monotonic() + self.MODEL_VISIBLE_TIMEOUT_SECONDS
        last_error: Exception | None = None
        last_body: str | None = None
        try:
            import httpx as _httpx
        except ImportError as e:
            raise ProviderNotInstalledError(
                "httpx not installed (required for the llamacpp provider). "
                "Run: pip install llmfacade[llamacpp]"
            ) from e

        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                tail = self._tail_log()
                raise ProviderError(
                    f"llama-swap exited before model {model_id!r} became visible. "
                    f"log tail:\n{tail}"
                )
            try:
                resp = _httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
                if resp.status_code < 400:
                    last_body = resp.text
                    try:
                        data = resp.json()
                    except ValueError as e:
                        last_error = e
                    else:
                        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                        if model_id in ids:
                            return
                else:
                    last_error = ProviderError(f"/v1/models returned {resp.status_code}")
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(0.1)
        tail = self._tail_log()
        raise ProviderError(
            f"llama-swap did not register model {model_id!r} within "
            f"{self.MODEL_VISIBLE_TIMEOUT_SECONDS}s (last error: {last_error}, "
            f"last body: {last_body!r}). log tail:\n{tail}"
        )

    def _cleanup_after_failure(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.terminate()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)
            self._proc = None
        self._anchor = None
        self._port = None
        self._close_log()
        with contextlib.suppress(OSError):
            self.pid_file.unlink()

    # --- yaml + pidfile ----------------------------------------------------

    def _write_yaml(self) -> None:
        rendered = _render_swap_yaml(
            self._entries,
            global_ttl=self.global_ttl,
            health_check_timeout=int(self.HEALTH_TIMEOUT_SECONDS),
        )
        self.yaml_path.write_text(rendered, encoding="utf-8")

    def _write_pid_file(self) -> None:
        if self._proc is None or self._port is None:
            return
        # owner= records *our* (Python) PID so a sibling process sharing this
        # llmfacade_dir can tell a live session's swap from a true orphan
        # (see _pid_file_sweep).
        line = f"{self._proc.pid}|{self._port}|{self._session_uuid}|owner={os.getpid()}\n"
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

    # --- shutdown ----------------------------------------------------------

    def _install_exit_hooks(self) -> None:
        # Install once per supervisor, not on every (re)spawn: re-running the
        # signal.signal loop after an interrupt()-triggered respawn would read
        # our own handler back as the "prior" one and chain to itself on the
        # next signal (infinite recursion, KeyboardInterrupt never raised).
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
        if self._signal_handlers_installed:
            return
        # signal.signal() only works on the main thread. If we're being driven
        # from a worker (asyncio.to_thread, a background thread), skip — atexit
        # still covers normal exit. The user's Ctrl+C still goes wherever the
        # main thread put it.
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
        # Chain to the prior handler (or what the OS would have done) so we
        # don't swallow the user's Ctrl+C.
        if callable(prior) and prior not in (signal.SIG_DFL, signal.SIG_IGN):
            prior(signum, frame)
            return
        if prior == signal.SIG_DFL:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            if sys.platform == "win32":
                # On Windows os.kill(pid, SIGTERM) calls TerminateProcess —
                # immediate, no atexit, no cleanup. SystemExit is the polite
                # equivalent that still lets the rest of the interpreter unwind.
                raise SystemExit(128 + signum)
            os.kill(os.getpid(), signum)

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            proc = self._proc
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(Exception):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=5)
            self._proc = None
            self._anchor = None
            self._port = None
            self._close_log()
            with contextlib.suppress(OSError):
                self.pid_file.unlink()

    def interrupt(self) -> bool:
        """Hard-kill the running llama-swap subprocess (and the llama-server
        backends it spawned) *immediately*, then reset to the not-started state
        so the next ``ensure_started()`` lazily respawns. The instant-abort
        primitive behind the provider's ``interrupt()``.

        Unlike ``shutdown()`` this does NOT send a graceful SIGTERM first and
        does NOT wait for the current request to drain — it force-terminates the
        process tree (see ``_hard_kill_tree``) so a thread parked inside an HTTP
        call to the backend gets an immediate transport error and unblocks.
        Thread-safe (takes the supervisor's RLock, which a blocked worker does
        not hold) and idempotent.

        Returns True iff a live process was actually killed; False if nothing
        was running (idle, already dead, never started). Never raises. Leaves
        ``_shutdown_done`` untouched so a later explicit ``shutdown()`` / atexit
        still cleans up the respawned process."""
        with self._lock:
            proc = self._proc
            live = proc is not None and proc.poll() is None
            if live:
                assert proc is not None
                _hard_kill_tree(proc.pid, proc)
            # Reset to the not-started state even when nothing was killed, so a
            # stale (already-dead) proc doesn't leave a leaked log handle, a
            # dead-port `base_url`, or an orphan pidfile lying around before the
            # next respawn.
            self._proc = None
            self._anchor = None
            self._port = None
            self._close_log()
            with contextlib.suppress(OSError):
                self.pid_file.unlink()
            return live

    def __del__(self) -> None:
        # Best-effort safety net; atexit + signal handlers are the primary path.
        # Use a non-blocking acquire so we don't deadlock against a thread
        # that's mid-shutdown and holds the RLock from a different OS thread.
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
