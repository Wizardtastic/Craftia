@echo off
REM Re-run the rustc 1.96 unicode-escape matrix on Windows.
REM Usage: tools\repro\u2014\run.bat   (requires rustc 1.96.0+ on PATH)
REM
REM Compiles each case_panic_*.rs in this directory via rustc and reports rc
REM + the first 3 lines of stderr on failure. Cleans up the binaries so
REM the next run starts fresh.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

where rustc >nul 2>&1
if errorlevel 1 (
    echo rustc not found on PATH.
    exit /b 2
)

rustc --version
echo.

set PASS=0
set FAIL=0

for %%f in (case_panic_legacy.rs case_panic_curly.rs case_panic_raw.rs) do (
    echo ----- %%f -----
    rustc %%f -o %%~nf.exe 1>nul 2> %%~nf.err
    set RC=!errorlevel!
    if "!RC!" == "0" (
        set /a PASS=PASS+1 >nul
        echo [PASS] %%f   rc=0
    ) else (
        set /a FAIL=FAIL+1 >nul
        echo [FAIL] %%f   rc=!RC!
        echo stderr head (first 3 lines):
        powershell -NoProfile -Command "Get-Content '%%~nf.err' -TotalCount 3" 2>nul
    )
    del /q "%%~nf.exe" "%%~nf.err" 2>nul
)

echo.
echo PASS: !PASS!
echo FAIL: !FAIL!

REM Documented contract on rustc 1.96.0: legacy 4-digit form rejects,
REM curly and raw-bytes accept. Exits with FAIL count so callers can gate.
endlocal & exit /b !FAIL!
