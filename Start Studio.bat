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

REM Kill any Windows-side process still listening on 8770. A stale Windows
REM listener shadows the WSL studio because localhost resolves to Windows
REM first - the WSL process is fine but unreachable from the browser.
REM Finds all PIDs in LISTENING state on 8770 and force-kills them.
echo  Clearing any stale Windows process on port 8770...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8770 .*LISTENING"') do (
    echo    killing Windows PID %%p
    taskkill /F /PID %%p >nul 2>&1
)

REM Bounce WSL to reset the localhost-to-WSL port-forwarding registration.
REM Without this, after a previous studio session the Windows->WSL bridge for
REM 8770 sometimes stays half-broken and the browser sees ERR_CONNECTION_REFUSED
REM even though the studio is running fine inside WSL. Cheap (~3 s) and
REM idempotent.
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
