@echo off
REM Git Push Script for Predictive Maintenance Project (Windows Batch)
REM This script automatically commits and pushes changes to the remote repository

setlocal enabledelayedexpansion

REM Configuration
set REPO_NAME=Predictive Maintenance for Sensor Time-Series (AWS + MLOps)
set BRANCH_NAME=main
set REMOTE_NAME=origin

REM Colors (Windows doesn't support colors in batch, so we'll use text formatting)
set INFO=[INFO]
set SUCCESS=[SUCCESS]
set WARNING=[WARNING]
set ERROR=[ERROR]

REM Function to print status messages
:print_status
echo %INFO% %~1
goto :eof

:print_success
echo %SUCCESS% %~1
goto :eof

:print_warning
echo %WARNING% %~1
goto :eof

:print_error
echo %ERROR% %~1
goto :eof

REM Function to check if we're in a git repository
:check_git_repo
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    call :print_error "Not in a git repository. Please run 'git init' first."
    exit /b 1
)
goto :eof

REM Function to check if there are any changes to commit
:check_changes
git diff --quiet
if errorlevel 1 goto :has_changes
git diff --cached --quiet
if errorlevel 1 goto :has_changes
call :print_warning "No changes to commit."
exit /b 1
:has_changes
exit /b 0

REM Function to create commit message
:create_commit_message
set "todo_item=%~1"
for /f "tokens=1-6 delims=: " %%a in ("%time%") do set "timestamp=%%a:%%b:%%c"
set "date_str=%date%"
set "commit_msg=✅ Completed: %todo_item% - %date_str% %timestamp%"
goto :eof

REM Function to add all changes
:add_changes
call :print_status "Adding all changes..."
git add .
if errorlevel 1 (
    call :print_error "Failed to add changes."
    exit /b 1
)
call :print_success "Changes added to staging area"
goto :eof

REM Function to commit changes
:commit_changes
set "commit_message=%~1"
call :print_status "Committing changes..."
git commit -m "%commit_message%"
if errorlevel 1 (
    call :print_error "Failed to commit changes."
    exit /b 1
)
call :print_success "Changes committed successfully"
goto :eof

REM Function to push to remote
:push_to_remote
call :print_status "Pushing to remote repository..."

REM Check if remote exists
git remote get-url %REMOTE_NAME% >nul 2>&1
if errorlevel 1 (
    call :print_warning "Remote '%REMOTE_NAME%' not found. Please add it first:"
    call :print_warning "git remote add %REMOTE_NAME% <repository-url>"
    exit /b 1
)

REM Push to remote
git push %REMOTE_NAME% %BRANCH_NAME%
if errorlevel 1 (
    call :print_error "Failed to push to remote. Please check your connection and permissions."
    exit /b 1
)
call :print_success "Successfully pushed to %REMOTE_NAME%/%BRANCH_NAME%"
goto :eof

REM Function to show git status
:show_status
call :print_status "Current git status:"
git status --short
echo.
goto :eof

REM Function to show recent commits
:show_recent_commits
call :print_status "Recent commits:"
git log --oneline -5
echo.
goto :eof

REM Function to show help
:show_help
echo Usage: %0 [TODO_ITEM]
echo.
echo Options:
echo   TODO_ITEM    Description of the completed TODO item
echo   -h, --help   Show this help message
echo.
echo Examples:
echo   %0 "Implement data ingestion pipeline"
echo   %0 "Add anomaly detection models"
echo   %0 "Set up AWS integration"
echo.
echo This script will:
echo   1. Check if you're in a git repository
echo   2. Add all changes to staging
echo   3. Create a commit with the TODO item description
echo   4. Push changes to the remote repository
goto :eof

REM Main function
:main
set "todo_item=%~1"

echo ==========================================
echo 🚀 Git Push Script for %REPO_NAME%
echo ==========================================
echo.

REM Check if we're in a git repository
call :check_git_repo
if errorlevel 1 exit /b 1

REM Show current status
call :show_status

REM Check if there are changes to commit
call :check_changes
if errorlevel 1 (
    call :print_warning "No changes detected. Nothing to commit and push."
    exit /b 0
)

REM Add all changes
call :add_changes
if errorlevel 1 exit /b 1

REM Create commit message
call :create_commit_message "%todo_item%"
call :print_status "Commit message: %commit_msg%"

REM Commit changes
call :commit_changes "%commit_msg%"
if errorlevel 1 exit /b 1

REM Push to remote
call :push_to_remote
if errorlevel 1 exit /b 1

call :print_success "✅ All changes have been successfully pushed to remote!"
call :show_recent_commits

echo.
echo ==========================================
call :print_success "Git push completed successfully! 🎉"
echo ==========================================
goto :eof

REM Parse command line arguments
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="" goto :main
goto :main
