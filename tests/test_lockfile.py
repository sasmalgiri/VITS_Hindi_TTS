"""Tests for hindi_tts_builder.utils.lockfile."""
import json
import os
from pathlib import Path

import pytest

from hindi_tts_builder.utils.lockfile import LOCK_NAME, ProjectBusy, ProjectLock, _pid_alive


class TestPidAlive:
    def test_own_pid_is_alive(self):
        assert _pid_alive(os.getpid())

    @pytest.mark.parametrize("pid", [0, -1])
    def test_invalid_pids(self, pid):
        assert not _pid_alive(pid)

    def test_almost_certainly_dead_pid(self):
        assert not _pid_alive(999_999_999)


class TestAcquireRelease:
    def test_lock_file_created_and_removed(self, tmp_path: Path):
        lock = tmp_path / LOCK_NAME
        with ProjectLock(tmp_path, "prepare"):
            assert lock.exists()
            d = json.loads(lock.read_text(encoding="utf-8"))
            assert d["pid"] == os.getpid()
            assert d["operation"] == "prepare"
        assert not lock.exists()

    def test_released_on_exception(self, tmp_path: Path):
        with pytest.raises(ValueError):
            with ProjectLock(tmp_path):
                raise ValueError("boom")
        assert not (tmp_path / LOCK_NAME).exists()

    def test_second_lock_by_live_process_is_refused(self, tmp_path: Path):
        """The concurrent-writer case that corrupted clips."""
        (tmp_path / LOCK_NAME).write_text(
            json.dumps({"pid": os.getpid() if False else 1, "operation": "prepare",
                        "host": "h", "started_at": "t"}),
            encoding="utf-8",
        )
        # PID 1 always exists on POSIX; on Windows use our own pid instead.
        live = 1 if os.name != "nt" else os.getpid()
        (tmp_path / LOCK_NAME).write_text(
            json.dumps({"pid": live, "operation": "prepare", "host": "h", "started_at": "t"}),
            encoding="utf-8",
        )
        if live == os.getpid():
            pytest.skip("cannot simulate a foreign live PID on this platform")
        with pytest.raises(ProjectBusy, match="already locked"):
            with ProjectLock(tmp_path):
                pass

    def test_stale_lock_is_taken_over(self, tmp_path: Path):
        (tmp_path / LOCK_NAME).write_text(
            json.dumps({"pid": 999_999_999, "operation": "prepare",
                        "host": "h", "started_at": "t"}),
            encoding="utf-8",
        )
        with ProjectLock(tmp_path, "resegment"):
            d = json.loads((tmp_path / LOCK_NAME).read_text(encoding="utf-8"))
            assert d["pid"] == os.getpid()
            assert d["operation"] == "resegment"

    def test_corrupt_lock_is_treated_as_stale(self, tmp_path: Path):
        (tmp_path / LOCK_NAME).write_text("not json at all", encoding="utf-8")
        with ProjectLock(tmp_path):
            assert (tmp_path / LOCK_NAME).exists()

    def test_release_does_not_remove_another_process_lock(self, tmp_path: Path):
        lock = ProjectLock(tmp_path)
        lock._acquire()
        # Someone else overwrites it while we hold our handle.
        (tmp_path / LOCK_NAME).write_text(
            json.dumps({"pid": 424242, "operation": "other", "host": "h", "started_at": "t"}),
            encoding="utf-8",
        )
        lock.release()
        assert (tmp_path / LOCK_NAME).exists(), "must not delete a lock we no longer own"

    def test_reentrant_same_pid(self, tmp_path: Path):
        """A nested call in the same process should not deadlock itself."""
        with ProjectLock(tmp_path, "outer"):
            with ProjectLock(tmp_path, "inner"):
                assert (tmp_path / LOCK_NAME).exists()
