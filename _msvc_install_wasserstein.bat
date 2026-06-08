@echo off
setlocal enabledelayedexpansion

set "VCVARS="D:\VISUAL STUDIO MICROSOFT\VC\Auxiliary\Build\vcvars64.bat""

echo Using vcvars64: %VCVARS%

call %VCVARS%
if errorlevel 1 (
  echo Failed to call vcvars64.bat
  exit /b 1
)

"D:\PISS\.venv\Scripts\python.exe" -m pip install wasserstein --no-cache-dir
exit /b %errorlevel%

