@echo off
rem Trascina qui sopra una cartella di foto: viene zippata, inviata al server
rem e l'archivio anonimizzato viene salvato accanto alle foto originali.
rem Camera usata di default: CAM_010 (modificabile dentro lo script .ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0anonimizza-cartella.ps1" -Cartella "%~1"
pause
