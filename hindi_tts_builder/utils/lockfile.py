"""Single-writer lock for a project directory.

Two pipeline processes writing the same project is not a theoretical hazard: it
happened here. An earlier launch survived a command that appeared to fail, a
second was started, and both ran ``segment`` against the same clip paths. ffmpeg
processes interleaved on the same files and produced WAVs that soundfile could
not open ("No 'data' chunk marker", "Format not recognised").

The damage compounds because ``skip_existing`` treats a corrupt file as done, so
a later "resume" silently keeps the corruption instead of repairing it. The only
safe recovery was deleting every clip and recutting.

Usage::

    with ProjectLock(paths.root, "prepare"):
        ...

A lock whose owning PID is gone is stale and gets taken over, so a crashed run
does not wedge the project permanently.
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = ".pipeline.lock"


class ProjectBusy(RuntimeError):
    """Raised when another live process already holds the project lock."""


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)          # signal 0 = existence check, no signal sent
    except ProcessLookupError:
        return False
    except PermissionError:
        return True              # exists, owned by someone else
    except OSError:
        return False
    return True


@dataclass
class LockInfo:
    pid: int
    operation: str
    host: str
    started_at: str

    @classmethod
    def read(cls, path: Path) -> "LockInfo | None":
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(int(d["pid"]), d.get("operation", "?"), d.get("host", "?"),
                       d.get("started_at", "?"))
        except Exception:
            return None           # unreadable/corrupt lock is treated as stale


class ProjectLock:
    """Context manager giving one process exclusive write access to a project."""

    def __init__(self, project_root: Path, operation: str = "pipeline", *, logger=None):
        self.path = Path(project_root) / LOCK_NAME
        self.operation = operation
        self.log = logger
        self._held = False

    def _acquire(self) -> None:
        existing = LockInfo.read(self.path) if self.path.exists() else None
        if existing and _pid_alive(existing.pid) and existing.pid != os.getpid():
            raise ProjectBusy(
                f"project is already locked by PID {existing.pid} running "
                f"'{existing.operation}' since {existing.started_at} on {existing.host}.\n"
                f"  Two writers corrupt clips mid-write and the corruption survives a resume.\n"
                f"  Wait for it, or stop that process and delete {self.path} if it is dead."
            )
        if existing and self.log:
            self.log.warning(
                f"[lock] taking over stale lock from dead PID {existing.pid} "
                f"('{existing.operation}', started {existing.started_at})"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({
                "pid": os.getpid(),
                "operation": self.operation,
                "host": socket.gethostname(),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2),
            encoding="utf-8",
        )
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        info = LockInfo.read(self.path)
        # Only remove our own lock — never another process's.
        if info and info.pid == os.getpid():
            self.path.unlink(missing_ok=True)
        self._held = False

    def __enter__(self) -> "ProjectLock":
        self._acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
