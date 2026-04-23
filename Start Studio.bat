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
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8770"

REM Run the studio inside WSL. Bind to 0.0.0.0 so Windows localhost works.
wsl -d Ubuntu-22.04 -u root -- bash -lc "cd /root/hindi-tts && source venv/bin/activate && exec hindi-tts-builder studio --host 0.0.0.0 --port 8770"

echo.
echo  Studio stopped.
pause
endlocal
