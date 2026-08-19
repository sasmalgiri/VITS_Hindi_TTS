"""Training health analyzer.

Read-only parser over the studio_run.log file. Returns a structured
health snapshot the studio UI can render as a status pill + sparklines.

Detects:
  - HEALTHY        — losses finite, step counter advancing
  - STARTING       — log exists but no GLOBAL_STEP yet (waiting for first step)
  - STALLED        — no new GLOBAL_STEP in > stall_threshold_sec
  - NAN_COLLAPSE   — > 5 consecutive NaN steps (matches our NaN-guard threshold)
  - NAN_WARN       — at least one NaN step in the last 100 (training survived
                     but flag for inspection — could indicate AMP instability)
  - DONE           — process is no longer running but file is complete
  - NO_RUN         — no log or no step lines at all

The analyzer is best-effort over a tail window; never reads the entire
multi-MB log on every poll. `tail_lines` controls window size.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import re


_STEP_RE = re.compile(
    r"TIME:\s+(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    r".*?GLOBAL_STEP:\s+(?P<step>\d+)"
)
_LOSS_RE = re.compile(
    r"loss_(?P<name>kl|duration|gen|disc|mel|feat)\s*:\s*(?P<value>nan|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
_AVG_RE = re.compile(
    r"avg_loss_(?P<name>kl|duration|gen|disc|mel|feat)\s*:\s*(?P<value>nan|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
_NAN_GUARD_RE = re.compile(r"\[nan-guard\] step has NaN/Inf in")
# Only match Coqui's *write* banner ("> BEST MODEL : .../best_model_<n>.pth"),
# not 'Restoring from ...' or our trainer's '[resume] weights init from ...'.
# Otherwise the panel would show 'last BEST 7350' on a fresh resume even
# though 7350 is the load source, not an improvement.
_BEST_RE = re.compile(r"BEST MODEL\s*:\s*\S*?best_model_(\d+)\.pth")

# Target loss floors (matches the studio's loss-convergence grid).
LOSS_FLOORS = {
    "mel": 12.0,
    "kl": 1.5,
    "gen": 1.8,
    "disc": 2.4,
    "feat": 2.0,
    "duration": 1.0,
}
# Anything this much above the floor counts as "diverging" (red mini-bar).
LOSS_DIVERGE_MULT = 5.0


@dataclass
class StepSample:
    step: int
    timestamp: str  # ISO-ish "YYYY-MM-DD HH:MM:SS"
    losses: dict   # {"kl": 1.99, "duration": 139.2, ...}; values may be None or "nan"


@dataclass
class HealthSnapshot:
    status: str                 # see status enum above
    status_reason: str          # one short human line for the pill subtitle
    last_step: int = 0
    last_step_at: str | None = None
    seconds_since_last_step: float | None = None
    steps_per_min: float | None = None
    samples_used: int = 0       # how many step rows we parsed in this window
    # NaN counts are scoped to the *current run* only (we slice the log to the
    # most recent run-restart marker before counting). Field name kept as
    # nan_steps_total for stable JSON contract; semantically it's
    # "NaN steps in current run within the tail window".
    nan_steps_total: int = 0
    nan_steps_recent_100: int = 0
    nan_guard_warnings: int = 0
    best_steps: list = field(default_factory=list)   # all best_model_<step> seen
    last_best_step: int | None = None
    last_eval_avg: dict = field(default_factory=dict)  # {"kl": 2.28, ...} from latest EVAL block
    loss_history: list = field(default_factory=list)   # [{"step": ..., "kl": ..., ...}]
    alarms: list = field(default_factory=list)         # human-readable strings


_RUN_START_MARKERS = (
    "[nan-guard] installed on model.train_step",  # our trainer's first log line
    "=== Training run starting ===",              # our Trainer.train() banner
    "[studio] $",                                 # studio's chain-spawn banner
)


def _tail_text(path: Path, max_bytes: int) -> str:
    """Cheap tail: read up to max_bytes from the end of the file."""
    sz = path.stat().st_size
    with path.open("rb") as f:
        if sz > max_bytes:
            f.seek(sz - max_bytes)
            f.readline()  # skip partial leading line
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _slice_to_current_run(text: str) -> str:
    """Discard everything before the most recent run-restart marker.

    h_tts_1's log is append-only across runs, so a 1 MB tail can mix v3's
    NaN-collapse era with v4's healthy steps and produce nonsense like
    'negative steps/min'. Anchor on the latest restart marker; if none is
    in the window, return the whole tail (best-effort).
    """
    best_idx = -1
    for marker in _RUN_START_MARKERS:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
    if best_idx < 0:
        return text
    # Back up to start-of-line so we don't begin mid-message
    line_start = text.rfind("\n", 0, best_idx)
    return text[line_start + 1 :] if line_start >= 0 else text[best_idx:]


def _parse_step_blocks(text: str) -> list[StepSample]:
    """A 'step block' is a TIME...GLOBAL_STEP header followed by a few lines
    of `| > loss_X: ...`. We walk the text linearly, attaching the loss
    lines that appear *after* a step header but before the next one.
    """
    samples: list[StepSample] = []
    current: StepSample | None = None
    for line in text.splitlines():
        m = _STEP_RE.search(line)
        if m:
            if current is not None:
                samples.append(current)
            current = StepSample(
                step=int(m.group("step")),
                timestamp=m.group("ts"),
                losses={},
            )
            continue
        if current is None:
            continue
        # Skip eval-section averages so we don't double-count them as step losses
        if "EVAL PERFORMANCE" in line or "avg_loss_" in line:
            continue
        lm = _LOSS_RE.search(line)
        if lm:
            name = lm.group("name").lower()
            raw = lm.group("value").lower()
            if raw == "nan":
                current.losses[name] = "nan"
            else:
                try:
                    current.losses[name] = float(raw)
                except ValueError:
                    pass
    if current is not None:
        samples.append(current)
    return samples


def _parse_last_eval(text: str) -> dict:
    """Return {kl: ..., duration: ..., ...} from the most recent COMPLETE
    EVAL block. Walks backwards because the trainer prints the
    'EVAL PERFORMANCE' header *before* it has written any avg_loss_ values
    — catching an in-progress eval mid-flight gives an empty dict and we'd
    rather show the previous eval's snapshot than nothing.
    """
    blocks = text.split("EVAL PERFORMANCE")
    if len(blocks) < 2:
        return {}
    # Each block i (i>=1) is the content AFTER the i-th 'EVAL PERFORMANCE'
    # header. Walk from most-recent to oldest until one has values.
    for block in reversed(blocks[1:]):
        out: dict = {}
        for m in _AVG_RE.finditer(block):
            name = m.group("name").lower()
            raw = m.group("value").lower()
            if raw == "nan":
                out[name] = None
            else:
                try:
                    out[name] = float(raw)
                except ValueError:
                    pass
            if len(out) >= len(LOSS_FLOORS):
                break
        if out:
            return out
    return {}


def _step_is_nan(sample: StepSample) -> bool:
    if not sample.losses:
        return False
    return any(v == "nan" for v in sample.losses.values())


def _fmt_seconds_ago(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


def analyze(
    log_path: Path,
    *,
    is_running: bool,
    tail_bytes: int = 5_000_000,    # last ~5 MB; big enough to include >=1 EVAL block
    history_keep_n: int = 60,        # number of step samples for sparklines
    stall_threshold_sec: float = 180.0,  # 3 min with no new step = stalled
) -> HealthSnapshot:
    """Compute a HealthSnapshot from the project's studio_run.log.

    The log is append-only across runs (v3 NaN-collapse era is in the same
    file as the v4 healthy steps), so we slice the tail to start at the
    most recent run-restart marker before parsing — otherwise step counters
    are non-monotonic and throughput goes negative.
    """
    snap = HealthSnapshot(status="no_run", status_reason="no log file")
    if not log_path.exists():
        return snap

    raw_text = _tail_text(log_path, tail_bytes)
    text = _slice_to_current_run(raw_text)
    samples = _parse_step_blocks(text)
    snap.samples_used = len(samples)

    # NaN-guard warnings (visible alarm even if losses are still finite)
    snap.nan_guard_warnings = sum(1 for _ in _NAN_GUARD_RE.finditer(text))

    # BEST checkpoints written this run. Take chronological order (text-order),
    # not numeric-max — when a run resumes from BEST 7350, the new v4 BESTs
    # start at lower step numbers (490, 1960, 4900, ...). User wants the
    # *latest* one written, which is the last text-order occurrence.
    best_matches = list(_BEST_RE.finditer(text))
    snap.best_steps = [int(m.group(1)) for m in best_matches]
    snap.last_best_step = int(best_matches[-1].group(1)) if best_matches else None

    # Last eval averages (the trustworthy signal — single-step values
    # bounce around, eval averages are smoothed)
    snap.last_eval_avg = _parse_last_eval(text)

    if not samples:
        if is_running:
            snap.status = "starting"
            snap.status_reason = "process running, no GLOBAL_STEP logged yet"
        else:
            snap.status = "no_run"
            snap.status_reason = "log empty / no step rows in tail window"
        return snap

    # Step velocity & last-step timing
    last = samples[-1]
    snap.last_step = last.step
    snap.last_step_at = last.timestamp
    try:
        last_dt = datetime.strptime(last.timestamp, "%Y-%m-%d %H:%M:%S")
        # Log timestamps are wall-clock local (Python logging default), so
        # compare with datetime.now() — not utcnow() — otherwise the value
        # is wrong by the system tz offset on non-UTC hosts. Clamp at 0
        # for the case where WSL clock has drifted slightly ahead of host.
        snap.seconds_since_last_step = max(0.0, (datetime.now() - last_dt).total_seconds())
    except Exception:
        snap.seconds_since_last_step = None

    if len(samples) >= 2:
        first = samples[0]
        try:
            t0 = datetime.strptime(first.timestamp, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(last.timestamp, "%Y-%m-%d %H:%M:%S")
            elapsed = (t1 - t0).total_seconds()
            if elapsed > 0:
                snap.steps_per_min = (last.step - first.step) * 60.0 / elapsed
        except Exception:
            pass

    # NaN counts
    nan_samples = [s for s in samples if _step_is_nan(s)]
    snap.nan_steps_total = len(nan_samples)
    last_100 = samples[-100:]
    snap.nan_steps_recent_100 = sum(1 for s in last_100 if _step_is_nan(s))

    # Build the per-step history (capped) for sparklines.
    # Use the *raw* per-step values; UI will smooth visually if needed.
    history_window = samples[-history_keep_n:]
    snap.loss_history = []
    for s in history_window:
        row = {"step": s.step, "ts": s.timestamp}
        for k in LOSS_FLOORS:
            v = s.losses.get(k)
            row[k] = (None if v is None or v == "nan" else float(v))
        snap.loss_history.append(row)

    # ------ Status decision ------
    alarms: list = []

    # Decisive: 5+ consecutive NaN at the tail = collapse
    consec_nan = 0
    for s in reversed(samples):
        if _step_is_nan(s):
            consec_nan += 1
        else:
            break
    if consec_nan >= 5:
        snap.status = "nan_collapse"
        snap.status_reason = f"{consec_nan} consecutive NaN steps at tail; nan-guard should have aborted"
        alarms.append(f"{consec_nan} consecutive NaN training steps")
        # Skip the rest of the status logic — collapse trumps everything
        snap.alarms = alarms
        return snap

    # Soft-warn on any NaN in the last 100
    if snap.nan_steps_recent_100 > 0:
        alarms.append(
            f"{snap.nan_steps_recent_100} NaN step(s) in last 100 — AMP may be unstable"
        )

    # NaN-guard fired but didn't abort yet
    if snap.nan_guard_warnings > 0:
        alarms.append(
            f"NaN-guard fired {snap.nan_guard_warnings} time(s); next 5 in a row will abort"
        )

    # Stall detection
    if (
        is_running
        and snap.seconds_since_last_step is not None
        and snap.seconds_since_last_step > stall_threshold_sec
    ):
        snap.status = "stalled"
        snap.status_reason = (
            f"no new step in {_fmt_seconds_ago(snap.seconds_since_last_step)} "
            f"(threshold {int(stall_threshold_sec)}s)"
        )
        alarms.append(snap.status_reason)
        snap.alarms = alarms
        return snap

    # Loss-divergence soft alarms (compare eval averages vs floors)
    for k, floor in LOSS_FLOORS.items():
        v = snap.last_eval_avg.get(k)
        if v is None:
            continue
        if v > floor * LOSS_DIVERGE_MULT:
            alarms.append(f"avg_loss_{k}={v:.2f} is > {LOSS_DIVERGE_MULT:g}× target floor {floor}")

    # Healthy / done
    if not is_running:
        snap.status = "done"
        snap.status_reason = f"process not running; last step {snap.last_step}"
    elif snap.steps_per_min is not None and snap.steps_per_min < 1.0 and snap.last_step > 100:
        # Very slow but not stalled — surface it as a warning
        snap.status = "healthy"
        snap.status_reason = (
            f"step {snap.last_step} @ {snap.steps_per_min:.1f} steps/min "
            f"(slow — check GPU contention)"
        )
        alarms.append(f"throughput {snap.steps_per_min:.1f} steps/min is unusually low")
    else:
        snap.status = "healthy"
        sub = f"step {snap.last_step}"
        if snap.steps_per_min:
            sub += f" @ {snap.steps_per_min:.1f} steps/min"
        if snap.last_best_step:
            sub += f"; last BEST {snap.last_best_step}"
        snap.status_reason = sub

    snap.alarms = alarms
    return snap


def to_dict(snap: HealthSnapshot) -> dict:
    return asdict(snap)
