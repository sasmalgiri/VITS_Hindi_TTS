#!/usr/bin/env bash
# smoke_test.sh — end-to-end pipeline exercise on real hardware.
#
# Runs the full hindi-tts-builder pipeline against a tiny dataset so you
# know the stack works on YOUR machine before committing 5+ days to a
# real training run. Exercises: yt-dlp download, WhisperX alignment,
# segmentation, QC, tokenizer fit, ~1000 training steps on GPU, engine
# export, speak, and a FastAPI /health probe.
#
# Expected wall time on RTX 3060 12GB: ~15-30 min (most of it is YouTube
# download + Whisper alignment; actual training is ~3-5 min for 1000 steps
# on a ~10-clip dataset).
#
# Usage:
#   ./scripts/smoke_test.sh --urls URLS.TXT --srts SRT_DIR/ [options]
#
# Options:
#   --urls FILE         file with 1 YouTube URL per line (required)
#   --srts DIR          directory with matching .srt files, one per URL
#                       in sorted order (required)
#   --project NAME      project name (default: smoke_test)
#   --steps N           training steps cap (default: 1000)
#   --skip-train        stop after `prepare`; skip train/export/speak
#   --skip-whisperx     use SRT timestamps verbatim (faster; less accurate)
#   --port PORT         port for the serve probe (default: 8780)
#   --keep              keep the project dir on failure (default: keep on
#                       failure, delete on success for repeatability)
#   -h, --help          show this help
#
# Exit codes:
#   0  every stage passed
#   1  argument / setup error
#   2  a pipeline stage failed (project dir preserved for inspection)
#
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT="smoke_test"
STEPS=1000
SKIP_TRAIN=0
SKIP_WX=0
PORT=8780
KEEP=0
URLS_FILE=""
SRTS_DIR=""

# --------------------------------------------------------- pretty-printing ---
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; RESET=$'\033[0m'
step() { printf "\n${BOLD}=== [%s] %s ===${RESET}\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
fail() {
  printf "${RED}✗ %s${RESET}\n" "$*"
  [[ -d "projects/${PROJECT}" ]] && \
    printf "${RED}project preserved at projects/${PROJECT} for debugging${RESET}\n"
  exit 2
}
die()  { echo "error: $*" >&2; exit 1; }
time_stage() {
  local label="$1"; shift
  local t0=$SECONDS
  "$@" || fail "$label failed"
  ok "$label done in $((SECONDS - t0))s"
}

usage() { sed -n '2,30p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --urls)          URLS_FILE="$2"; shift 2 ;;
    --srts)          SRTS_DIR="$2"; shift 2 ;;
    --project)       PROJECT="$2"; shift 2 ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --skip-train)    SKIP_TRAIN=1; shift ;;
    --skip-whisperx) SKIP_WX=1; shift ;;
    --port)          PORT="$2"; shift 2 ;;
    --keep)          KEEP=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# --------------------------------------------------------------- checks ---
# Use explicit `if` blocks rather than the `|| { die ...; }` compound —
# bash swallows the exit code of `exit N` called from inside `||` under
# `set -e`, so `die` there silently becomes exit 0.
if [[ -z "$URLS_FILE" || -z "$SRTS_DIR" ]]; then
  usage >&2
  die "--urls and --srts are required"
fi
if [[ ! -f "$URLS_FILE" ]]; then die "URLS file not found: $URLS_FILE"; fi
if [[ ! -d "$SRTS_DIR"  ]]; then die "SRTS dir not found: $SRTS_DIR"; fi
if ! command -v hindi-tts-builder >/dev/null; then
  die "hindi-tts-builder not on PATH. Activate the venv first:  source venv/bin/activate"
fi

URL_COUNT=$(grep -cvE '^\s*(#|$)' "$URLS_FILE" || true)
SRT_COUNT=$(find "$SRTS_DIR" -maxdepth 1 -name '*.srt' | wc -l)
if (( URL_COUNT != SRT_COUNT )); then
  die "URL count ($URL_COUNT) != SRT count ($SRT_COUNT). They must match 1:1."
fi

# =============================================================== stages ===

step "1/8 doctor"
hindi-tts-builder doctor || fail "doctor reported missing deps"

step "2/8 clean + new"
rm -rf "projects/${PROJECT}"
hindi-tts-builder new "$PROJECT"

step "3/8 add-sources ($URL_COUNT URL/SRT pair(s))"
hindi-tts-builder add-sources "$PROJECT" --urls "$URLS_FILE" --transcripts "$SRTS_DIR"

step "4/8 prepare (download → align → segment → QC → training set)"
if (( SKIP_WX == 1 )); then
  time_stage "prepare" hindi-tts-builder prepare "$PROJECT" --no-whisperx
else
  time_stage "prepare" hindi-tts-builder prepare "$PROJECT"
fi

# Surface the dataset stats so the user sees what got through QC.
READY="projects/${PROJECT}/training_set/ready.json"
if [[ -f "$READY" ]]; then cat "$READY"; fi

if (( SKIP_TRAIN == 1 )); then
  step "skipping train/export/speak/serve per --skip-train"
  ok "smoke test (prepare-only) complete"
  exit 0
fi

step "5/8 cap max_steps=$STEPS in training_config.yaml"
CFG="projects/${PROJECT}/training_config.yaml"
# First run creates the config; make sure it exists before editing.
hindi-tts-builder train "$PROJECT" --prepare-only
python - "$CFG" "$STEPS" <<'PY'
import sys, yaml, pathlib
cfg_path, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
# Honor whichever key the training config uses.
for k in ("max_steps", "total_steps", "epochs"):
    if k in cfg:
        cfg[k] = steps if k != "epochs" else max(1, steps // 100)
cfg["max_steps"] = steps          # always force this one
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"  wrote max_steps={steps} to {cfg_path}")
PY
ok "training config edited"

step "6/8 train ($STEPS steps on GPU)"
time_stage "train" hindi-tts-builder train "$PROJECT"

step "7/8 export engine"
time_stage "export" hindi-tts-builder export "$PROJECT"

step "8/8 speak + serve probe"
OUT="projects/${PROJECT}/smoke_test.wav"
hindi-tts-builder speak "$PROJECT" --text "यह एक परीक्षण है।" --out "$OUT" --no-validate
[[ -s "$OUT" ]] || fail "output wav missing or empty"
ok "generated $OUT ($(stat -c%s "$OUT") bytes)"

# Start serve in background, probe /health, kill it.
hindi-tts-builder serve "$PROJECT" --host 127.0.0.1 --port "$PORT" > /tmp/smoke_serve.log 2>&1 &
SERVE_PID=$!
sleep 4
HTTP=$(curl -sS -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:${PORT}/health" || echo "000")
kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true
[[ "$HTTP" == "200" ]] || { echo "--- /tmp/smoke_serve.log ---"; tail -20 /tmp/smoke_serve.log; fail "/health returned $HTTP (expected 200)"; }
ok "/health returned 200"

echo
printf "${GREEN}${BOLD}SMOKE TEST PASSED${RESET} — ${DIM}the full pipeline works on this machine.${RESET}\n"

# Clean up successful run (unless --keep).
if (( KEEP == 0 )); then
  rm -rf "projects/${PROJECT}"
  ok "cleaned up projects/${PROJECT}"
fi
exit 0
