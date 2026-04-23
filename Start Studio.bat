@echo off
setlocal
title Hindi TTS Builder - Training Studio

echo.
echo  ============================================
echo   Hindi TTS Builder - Training Studio
echo  ============================================
echo.
echo  Starting the studio inside WSL Ubuntu-22.04...
echo  URL:  http://localhost:8770
echo.
echo  The browser will open automatically in a few seconds.
echo  Close this window (or Ctrl+C) to stop the server.
echo.

REM Open the browser after a short delay so the server is ready.
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8770"

REM Run the studio inside WSL. Steps:
REM   1. free port 8770 if a stale process is still bound to it
REM   2. git pull --ff-only to keep the WSL clone in sync with GitHub
REM   3. ensure deps the studio needs are present (idempotent, fast)
REM   4. exec the studio bound to 0.0.0.0 so Windows localhost reaches it
wsl -d Ubuntu-22.04 -u root -- bash -lc "fuser -k -TERM 8770/tcp 2>/dev/null; sleep 1; cd /root/hindi-tts && git pull --ff-only --quiet origin master 2>/dev/null; source venv/bin/activate && pip install --quiet python-multipart 'transformers>=4.57,<5' >/dev/null 2>&1; exec hindi-tts-builder studio --host 0.0.0.0 --port 8770"

echo.
echo  Studio stopped.
pause
endlocal
