@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\stop_spectra.ps1" %*
