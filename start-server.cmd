@echo off
rem Doppio click per avviare il server di anonimizzazione.
rem Per fermarlo: doppio click su stop-server.cmd
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-server.ps1"
pause
