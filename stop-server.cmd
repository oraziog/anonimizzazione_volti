@echo off
rem Doppio click per fermare il server di anonimizzazione.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-server.ps1"
pause
