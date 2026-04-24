"""Per-project background job manager.

Wraps `subprocess.Popen` so the web UI can launch `hindi-tts-builder
prepare` followed by `hindi-tts-builder train` and stream stdout to a
log file that an SSE endpoint tails.

Only one job runs per project at a time. Jobs are tracked in-process —
restarting the studio drops the registry, but a job already running as
a subprocess keeps going (and its log file remains on disk for
post-hoc inspection).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import subprocess
import sys
import threading
import time


@dataclass
class JobState:
    project: str
    stage: str
    pid: int
    started_at: float
    log_path: Path
    returncode: Optional[int] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.returncode is None and self.finished_at is None

    def to_dict(self) -> dict:
        return {
            "project": self.project,
            "stage": self.stage,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "running": self.running,
            "log_path": str(self.log_path),
            "error": self.error,
        }


class JobRegistry:
    """In-process registry of one job per project."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def get(self, project: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(project)

    def list(self) -> list[JobState]:
        with self._lock:
            return list(self._jobs.values())

    def is_running(self, project: str) -> bool:
        st = self.get(project)
        return st is not None and st.running

    def reattach_orphans(self, projects_root: Path) -> int:
        """Scan for `hindi-tts-builder (prepare|train) <project>` processes
        whose parent studio is gone (we just started, registry is empty)
        and re-register them. Lets a studio restart pick up an in-flight
        pipeline without showing it as "idle" in the UI.

        Linux/WSL only — uses /proc. Returns the count of re-attached jobs.
        """
        proc_root = Path("/proc")
        if not proc_root.exists():
            return 0
        attached = 0
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            cmd_file = entry / "cmdline"
            try:
                cmdline = cmd_file.read_bytes().split(b"\x00")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            cmd_str = " ".join(c.decode("utf-8", errors="replace") for c in cmdline if c)
            if "hindi_tts_builder.cli.main" not in cmd_str:
                continue
            stage = None
            project = None
            parts = [c.decode("utf-8", errors="replace") for c in cmdline if c]
            for i, p in enumerate(parts):
                if p in ("prepare", "train") and i + 1 < len(parts):
                    stage = p
                    project = parts[i + 1]
                    break
            if not project:
                continue
            with self._lock:
                if project in self._jobs and self._jobs[project].running:
                    continue
                pid = int(entry.name)
                log_path = projects_root / project / "logs" / "studio_run.log"
                try:
                    started_at = entry.stat().st_mtime
                except OSError:
                    import time as _t
                    started_at = _t.time()
                state = JobState(
                    project=project,
                    stage=stage or "unknown",
                    pid=pid,
                    started_at=started_at,
                    log_path=log_path,
                )
                self._jobs[project] = state
                attached += 1
                # Spawn a thin watcher: poll /proc/<pid> until it disappears,
                # then mark finished. We don't have the original Popen object
                # so we can't know returncode for sure; use 0 if /proc is gone.
                threading.Thread(
                    target=self._watch_orphan, args=(project, pid),
                    daemon=True,
                ).start()
        return attached

    def _watch_orphan(self, project: str, pid: int) -> None:
        import time as _t
        proc_dir = Path(f"/proc/{pid}")
        while proc_dir.exists():
            _t.sleep(2)
        with self._lock:
            st = self._jobs.get(project)
            if st is not None and st.running:
                st.returncode = 0  # we lost the real exit code
                st.finished_at = _t.time()

    def start_pipeline(
        self,
        project: str,
        projects_root: Path,
        skip_train: bool = False,
        no_whisperx: bool = False,
        no_whisper_qc: bool = False,
        skip_qc: bool = False,
    ) -> JobState:
        """Launch `prepare` then `train` (unless skip_train) as a single
        background subprocess chain. stdout+stderr → log file.
        """
        with self._lock:
            existing = self._jobs.get(project)
            if existing and existing.running:
                raise RuntimeError(
                    f"Project '{project}' already has a running job (pid={existing.pid})"
                )

        log_dir = projects_root / project / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "studio_run.log"

        cli = [sys.executable, "-m", "hindi_tts_builder.cli.main"]
        prepare_cmd = cli + ["prepare", project]
        if no_whisperx:
            prepare_cmd.append("--no-whisperx")
        if no_whisper_qc:
            prepare_cmd.append("--no-whisper-qc")
        if skip_qc:
            prepare_cmd.append("--skip-qc")
        train_cmd = cli + ["train", project]

        # Chain: run prepare; if it succeeds and skip_train is False, run train.
        # We use a small shell wrapper so the chain runs in one subprocess.
        if os.name == "nt":
            chain = " && ".join(_quote(c) for c in [prepare_cmd] + ([train_cmd] if not skip_train else []))
            popen_kwargs = dict(shell=True)
        else:
            chain = " && ".join(_quote(c) for c in [prepare_cmd] + ([train_cmd] if not skip_train else []))
            popen_kwargs = dict(shell=True, executable="/bin/bash")

        log_fp = open(log_path, "ab", buffering=0)
        log_fp.write(f"[studio] $ {chain}\n".encode("utf-8"))
        proc = subprocess.Popen(
            chain,
            cwd=str(projects_root.parent),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            **popen_kwargs,
        )

        state = JobState(
            project=project,
            stage="prepare+train" if not skip_train else "prepare",
            pid=proc.pid,
            started_at=time.time(),
            log_path=log_path,
        )

        with self._lock:
            self._jobs[project] = state
            self._procs[project] = proc

        threading.Thread(target=self._watch, args=(project, proc, log_fp), daemon=True).start()
        return state

    def _watch(self, project: str, proc: subprocess.Popen, log_fp) -> None:
        proc.wait()
        try:
            log_fp.close()
        except Exception:
            pass
        with self._lock:
            st = self._jobs.get(project)
            if st is not None:
                st.returncode = proc.returncode
                st.finished_at = time.time()
                if proc.returncode != 0:
                    st.error = f"exit code {proc.returncode}"

    def stop(self, project: str) -> bool:
        """Best-effort termination. Returns True if a process was signalled."""
        with self._lock:
            proc = self._procs.get(project)
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.terminate()
            return True
        except Exception:
            return False


def _quote(cmd: list[str]) -> str:
    """Minimal shell-quoting that works for both cmd.exe (Windows) and bash."""
    out = []
    for part in cmd:
        if any(c in part for c in ' "\t&|<>^'):
            out.append('"' + part.replace('"', '\\"') + '"')
        else:
            out.append(part)
    return " ".join(out)
