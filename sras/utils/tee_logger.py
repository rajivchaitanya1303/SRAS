"""
sras.utils.tee_logger
=====================
Lightweight tee-style logging: mirrors everything printed to the terminal
(stdout + stderr) into a timestamped log file simultaneously.

Two usage modes
---------------
1. **Script-level tee** (use in run_all.py and standalone run_*.py scripts)::

       from sras.utils.tee_logger import start_run_log, finish_run_log

       log_path = start_run_log("run_all")   # opens logs/runs/<timestamp>_run_all.log
       try:
           ...main work...
       finally:
           finish_run_log(log_path)

2. **Subprocess tee** (used internally by run_all.py's _run() helper) –
   streams every line from a child process to both the terminal and the
   open log file::

       from sras.utils.tee_logger import tee_subprocess

       ok = tee_subprocess(cmd, stage_name, log_file)

3. **Context-manager style** (compact alternative for individual scripts)::

       from sras.utils.tee_logger import RunLogger

       with RunLogger("run_contrastive") as rl:
           main()

Implementation notes
--------------------
* Uses ``sys.stdout`` / ``sys.stderr`` replacement via ``TeeStream`` so that
  *all* Python ``print()`` calls, including those in imported libraries, are
  captured without any code changes to those modules.
* The log file is UTF-8 encoded; ANSI colour codes are stripped before
  writing so the log stays readable in plain-text editors.
* The log directory (``logs/runs/``) is created automatically.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from io import TextIOBase
from typing import IO, List, Optional, TextIO, Tuple


# ── ANSI escape stripping ────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJn]|\x1b\].*?(?:\x07|\x1b\\)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ── tqdm progress-bar filter ─────────────────────────────────────────────────
# tqdm writes to stderr; when captured through a pipe (no TTY) it falls back
# to printing every update on its own line, which floods the log file.
# We detect these lines and skip them when writing to the *secondary* (log)
# sink, while still letting them through to the terminal so the user gets
# live progress feedback.
#
# A tqdm line looks like one of:
#   "  5%|▌         | 41/747 [00:00<00:11, 61.77it/s]"
#   "Supervised epoch 144/150:   5%|▌   | 41/747 [00:00<00:11, 61.77it/s]"
#   "Loading weights: 100%|██████████| 257/257 [00:00<00:00, 4086.30it/s]"
#
# We match the characteristic  "NN%|...| n/N ["  fragment.
_TQDM_RE = re.compile(r"\d+%\|.*?\|\s*\d+/\d+\s*\[")


def _is_tqdm_line(text: str) -> bool:
    """Return True if *text* looks like a tqdm progress-bar update."""
    # Also catch carriage-return overwrite style (rare when piped, but safe)
    if "\r" in text and _TQDM_RE.search(text):
        return True
    return bool(_TQDM_RE.search(text))


# ── TeeStream: write to two sinks at once ────────────────────────────────────

class TeeStream(TextIOBase):
    """A file-like object that writes to *primary* and *secondary* simultaneously.

    The *secondary* (the log file) receives ANSI-stripped output.
    """

    def __init__(self, primary: TextIO, secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary
        self._lock = threading.Lock()

    # Delegate encoding info so libraries that inspect sys.stdout.encoding work
    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._primary, "encoding", "utf-8")

    @property
    def errors(self) -> Optional[str]:  # type: ignore[override]
        return getattr(self._primary, "errors", "replace")

    def write(self, text: str) -> int:
        with self._lock:
            # Write to the terminal.  On Windows the console may be cp1252 and
            # refuse Unicode characters like Greek letters.  We fall back to
            # writing a safe ASCII representation rather than crashing.
            try:
                self._primary.write(text)
                self._primary.flush()
            except UnicodeEncodeError:
                enc = getattr(self._primary, "encoding", "ascii") or "ascii"
                safe = text.encode(enc, errors="backslashreplace").decode(enc)
                try:
                    self._primary.write(safe)
                    self._primary.flush()
                except Exception:
                    pass
            except Exception:
                pass
            # Write to the log file (always UTF-8, no encoding issues).
            # Skip tqdm progress-bar lines; they would add thousands of
            # near-identical lines per training run.  The epoch-level
            # logger.info summaries are written once per epoch and are
            # sufficient for the permanent record.
            if not _is_tqdm_line(text):
                try:
                    self._secondary.write(_strip_ansi(text))
                    self._secondary.flush()
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        with self._lock:
            try:
                self._primary.flush()
            except Exception:
                pass
            try:
                self._secondary.flush()
            except Exception:
                pass

    def fileno(self) -> int:
        """Delegate fileno to primary so subprocess detection works."""
        return self._primary.fileno()

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()


# ── Log-file helpers ─────────────────────────────────────────────────────────

_LOG_DIR = os.path.join("logs", "runs")


def _make_log_path(script_name: str) -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{script_name}.log"
    return os.path.join(_LOG_DIR, filename)


def _header(script_name: str, log_path: str) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    sep = "=" * 72
    return (
        f"{sep}\n"
        f"  SRAS Experiment Log\n"
        f"  Script  : {script_name}\n"
        f"  Started : {ts}\n"
        f"  Log file: {log_path}\n"
        f"{sep}\n"
    )


def _footer(script_name: str, elapsed: float) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    sep = "=" * 72
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    dur = f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"
    return (
        f"\n{sep}\n"
        f"  Finished : {ts}\n"
        f"  Elapsed  : {dur} ({elapsed:.1f}s)\n"
        f"  Script   : {script_name}\n"
        f"{sep}\n"
    )


# ── Public API: imperative style ────────────────────────────────────────────

_active_log_file: Optional[IO[str]] = None
_original_stdout: Optional[TextIO] = None
_original_stderr: Optional[TextIO] = None
_start_time: float = 0.0


def _reconfigure_utf8(stream: TextIO) -> TextIO:
    """Try to switch *stream* to UTF-8 encoding in-place (Python ≥ 3.7).

    Returns the (possibly reconfigured) stream.  On Windows the default
    console encoding is cp1252; this upgrades it so Unicode characters (Greek
    letters, arrows, etc.) survive the trip to the terminal without errors.
    """
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
    return stream


def start_run_log(script_name: str = "sras") -> str:
    """Replace sys.stdout / sys.stderr with tee streams and open a log file.

    Returns the path to the log file so the caller can print it.
    Must be paired with ``finish_run_log()``.
    """
    global _active_log_file, _original_stdout, _original_stderr, _start_time

    log_path = _make_log_path(script_name)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    _active_log_file = log_file
    _start_time = time.time()

    # On Windows, upgrade the console to UTF-8 before wrapping so that
    # Unicode characters (e.g. Greek τ in log messages) reach the terminal
    # instead of raising UnicodeEncodeError.
    _reconfigure_utf8(sys.stdout)
    _reconfigure_utf8(sys.stderr)

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    hdr = _header(script_name, log_path)
    log_file.write(hdr)
    log_file.flush()
    sys.stdout.write(hdr)
    sys.stdout.flush()

    sys.stdout = TeeStream(_original_stdout, log_file)  # type: ignore[assignment]
    sys.stderr = TeeStream(_original_stderr, log_file)  # type: ignore[assignment]

    return log_path


def finish_run_log(log_path: str, script_name: str = "sras") -> None:
    """Restore original streams and write a footer to the log file."""
    global _active_log_file, _original_stdout, _original_stderr

    elapsed = time.time() - _start_time
    footer = _footer(script_name, elapsed)

    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr

    if _active_log_file is not None:
        try:
            _active_log_file.write(footer)
            _active_log_file.flush()
            _active_log_file.close()
        except Exception:
            pass
        _active_log_file = None

    print(footer, end="")
    print(f"[tee_logger] Full log saved to: {log_path}")


# ── Public API: context-manager style ───────────────────────────────────────

class RunLogger:
    """Context manager that tees stdout/stderr to a log file for its duration.

    Usage::

        with RunLogger("run_contrastive") as rl:
            main()
        # rl.log_path is the path written
    """

    def __init__(self, script_name: str = "sras") -> None:
        self.script_name = script_name
        self.log_path: str = ""

    def __enter__(self) -> "RunLogger":
        self.log_path = start_run_log(self.script_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        finish_run_log(self.log_path, self.script_name)
        return False  # do not suppress exceptions


# ── Subprocess tee ───────────────────────────────────────────────────────────

def _safe_write(stream: "TextIO", text: str) -> None:
    """Write *text* to *stream*, falling back to backslashreplace on cp1252."""
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", "ascii") or "ascii"
        safe = text.encode(enc, errors="backslashreplace").decode(enc)
        try:
            stream.write(safe)
            stream.flush()
        except Exception:
            pass
    except Exception:
        pass


def tee_subprocess(
    cmd: List[str],
    stage: str,
    log_file: Optional[IO[str]] = None,
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> Tuple[bool, int]:
    """Run *cmd* as a subprocess, streaming lines to stdout AND *log_file*.

    Terminal behaviour
    ------------------
    * **tqdm progress bars** are shown as a single *dynamically updating*
      line, each update overwrites the previous bar in place using ``\\r``,
      exactly as you would see when running the script directly in a TTY.
      When the epoch finishes and an INFO summary line arrives, the bar is
      erased cleanly and the summary is printed on its own line.

    * **All other output** (logger.info epoch summaries, warnings, banners)
      is printed normally with a trailing newline.

    Log-file behaviour
    ------------------
    * tqdm lines are **never** written to the log file; only the INFO
      epoch-summary lines are recorded, keeping the log compact
      (one line per epoch instead of ~50–80 per-batch updates).

    Returns ``(success: bool, returncode: int)``.
    """
    import copy
    import shutil

    sink = log_file or _active_log_file

    # Measure terminal width so tqdm (running in the subprocess) can size
    # the bar correctly even though its stdout is a pipe not a TTY.
    term_cols = shutil.get_terminal_size((120, 24)).columns

    child_env = copy.copy(env) if env is not None else os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    # Tell tqdm the terminal width so it fills the line properly.
    child_env["COLUMNS"] = str(term_cols)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=cwd,
        env=child_env,
    )

    assert proc.stdout is not None

    # Track whether the last thing printed to the terminal was a tqdm bar so
    # we know when to erase it before printing the next INFO line.
    _bar_active = False

    for line in proc.stdout:
        is_tqdm = _is_tqdm_line(line)

        if is_tqdm:
            # Overwrite the current terminal line in place.
            # Strip the trailing newline; the \r brings the cursor back to
            # column 0 so the next update overwrites this one.
            bar_text = line.rstrip("\n\r")
            # Pad to terminal width so any shorter previous bar is fully erased.
            bar_text = bar_text.ljust(term_cols)
            _safe_write(sys.stdout, "\r" + bar_text)
            _bar_active = True

        else:
            # Non-tqdm line (INFO summary, warning, banner, etc.)
            if _bar_active:
                # Move to a fresh line so the info text doesn't overwrite the bar.
                _safe_write(sys.stdout, "\r" + " " * term_cols + "\r")
                _bar_active = False
            _safe_write(sys.stdout, line)

            # Record in the log file (tqdm lines are never written here).
            if sink is not None:
                try:
                    sink.write(_strip_ansi(line))
                    sink.flush()
                except Exception:
                    pass

    # If the subprocess ended while a bar was still on screen, finish the line.
    if _bar_active:
        _safe_write(sys.stdout, "\n")

    proc.wait()
    success = proc.returncode == 0
    return success, proc.returncode
