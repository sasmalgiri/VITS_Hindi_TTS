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

REM First, probe whether the studio is already serving on localhost:8770.
REM If yes, this bat run should be a no-op intervention - just open the
REM browser and exit. We must NOT kill the WSL distro or stale-Windows
REM processes, because doing so would also kill any in-progress training
REM pipeline running inside the existing studio.
echo  Checking if studio is already up...
curl --silent --max-time 2 --output NUL --fail http://127.0.0.1:8770/ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo  Studio already responding on http://localhost:8770 - opening browser only.
    start "" http://localhost:8770
    echo.
    echo  Leaving this window open is optional; the studio is running independently.
    pause
    exit /b 0
)

REM Studio is not responding. Clean up any stale Windows-side listener on
REM 8770 (a dead python process left from a crashed prior studio run can
REM hold the port and block new bind attempts).
echo  Studio not responding. Clearing any stale Windows process on port 8770...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8770 .*LISTENING"') do (
    echo    killing Windows PID %%p
    taskkill /F /PID %%p >nul 2>&1
)

REM If WSL has stuck localhost port forwarding from a previous session, only
REM bouncing WSL fixes it. Safe here because we already confirmed nothing
REM is responding on 8770 - so no live training to disturb.
echo  Bouncing WSL to refresh localhost forwarding...
wsl --shutdown
ping -n 4 127.0.0.1 >nul


REM Open the browser after a short delay so the server is ready.
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8770"

REM Run the studio inside WSL. Steps:
REM   1. free port 8770 in WSL if a stale process is still bound to it
REM   2. git pull --ff-only to keep the WSL clone in sync with GitHub
REM   3. ensure deps the studio needs are present (idempotent, fast):
REM        - python-multipart (FastAPI file uploads)
REM        - transformers pin (coqui-tts compat)
REM        - cuDNN 8 staged to /opt/cudnn8 so ctranslate2 (faster-whisper)
REM          can find it while torch keeps its bundled cuDNN 9
REM   4. export LD_LIBRARY_PATH so both cuDNN 8 and 9 are visible
REM   5. exec the studio bound to 0.0.0.0 so Windows localhost reaches it
wsl -d Ubuntu-22.04 -u root -- bash -lc "fuser -k -TERM 8770/tcp 2>/dev/null; sleep 1; cd /root/hindi-tts && git pull --ff-only --quiet origin master 2>/dev/null; source venv/bin/activate && pip install --quiet python-multipart 'transformers>=4.57,<5' >/dev/null 2>&1; mkdir -p /opt/cudnn8 && [ -f /opt/cudnn8/nvidia/cudnn/lib/libcudnn_ops_infer.so.8 ] || pip install --quiet --target /opt/cudnn8 nvidia-cudnn-cu12==8.9.7.29 >/dev/null 2>&1; export LD_LIBRARY_PATH=/opt/cudnn8/nvidia/cudnn/lib:/root/hindi-tts/venv/lib/python3.10/site-packages/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; exec hindi-tts-builder studio --host 0.0.0.0 --port 8770"

echo.
echo  Studio stopped.
pause
endlocal
