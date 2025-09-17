@echo off
REM Quick Git Push Wrapper for Windows
REM Usage: push.bat "TODO description"

if "%1"=="" (
    echo Usage: push.bat "TODO description"
    echo Example: push.bat "Implement data ingestion pipeline"
    exit /b 1
)

python scripts\git_push.py "%1"
