@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ab-rtx2060s.ps1" %*
exit /b %ERRORLEVEL%
