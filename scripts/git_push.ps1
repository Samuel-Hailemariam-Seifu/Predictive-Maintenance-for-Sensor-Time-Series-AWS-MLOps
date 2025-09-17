# Git Push Script for Predictive Maintenance Project (PowerShell)
# This script automatically commits and pushes changes to the remote repository

param(
    [Parameter(Position=0)]
    [string]$TodoItem = "",
    
    [switch]$AutoTodo,
    [switch]$Help
)

# Configuration
$RepoName = "Predictive Maintenance for Sensor Time-Series (AWS + MLOps)"
$BranchName = "main"
$RemoteName = "origin"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

function Write-Status {
    param(
        [string]$Message,
        [string]$Status = "INFO"
    )
    
    $color = switch ($Status.ToUpper()) {
        "SUCCESS" { $Colors.Green }
        "WARNING" { $Colors.Yellow }
        "ERROR" { $Colors.Red }
        "INFO" { $Colors.Blue }
        default { $Colors.White }
    }
    
    Write-Host "[$Status] $Message" -ForegroundColor $color
}

function Test-GitRepository {
    try {
        $null = git rev-parse --git-dir 2>$null
        return $true
    }
    catch {
        Write-Status "Not in a git repository. Please run 'git init' first." "ERROR"
        return $false
    }
}

function Test-GitChanges {
    $unstagedChanges = $false
    $stagedChanges = $false
    
    try {
        git diff --quiet 2>$null
        if ($LASTEXITCODE -ne 0) { $unstagedChanges = $true }
        
        git diff --cached --quiet 2>$null
        if ($LASTEXITCODE -ne 0) { $stagedChanges = $true }
        
        return $unstagedChanges -or $stagedChanges
    }
    catch {
        return $false
    }
}

function New-CommitMessage {
    param([string]$TodoItem)
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if ($TodoItem) {
        return "✅ Completed: $TodoItem - $timestamp"
    }
    else {
        return "🔄 Update: $timestamp"
    }
}

function Add-GitChanges {
    Write-Status "Adding all changes..."
    
    try {
        git add .
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Changes added to staging area" "SUCCESS"
            return $true
        }
        else {
            Write-Status "Failed to add changes." "ERROR"
            return $false
        }
    }
    catch {
        Write-Status "Failed to add changes: $_" "ERROR"
        return $false
    }
}

function Commit-GitChanges {
    param([string]$CommitMessage)
    
    Write-Status "Committing changes..."
    
    try {
        git commit -m $CommitMessage
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Changes committed successfully" "SUCCESS"
            return $true
        }
        else {
            Write-Status "Failed to commit changes." "ERROR"
            return $false
        }
    }
    catch {
        Write-Status "Failed to commit changes: $_" "ERROR"
        return $false
    }
}

function Push-ToRemote {
    Write-Status "Pushing to remote repository..."
    
    # Check if remote exists
    try {
        $null = git remote get-url $RemoteName 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Remote '$RemoteName' not found. Please add it first:" "WARNING"
            Write-Status "git remote add $RemoteName <repository-url>" "WARNING"
            return $false
        }
    }
    catch {
        Write-Status "Failed to check remote: $_" "ERROR"
        return $false
    }
    
    # Push to remote
    try {
        git push $RemoteName $BranchName
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Successfully pushed to $RemoteName/$BranchName" "SUCCESS"
            return $true
        }
        else {
            Write-Status "Failed to push to remote. Please check your connection and permissions." "ERROR"
            return $false
        }
    }
    catch {
        Write-Status "Failed to push to remote: $_" "ERROR"
        return $false
    }
}

function Show-GitStatus {
    Write-Status "Current git status:"
    try {
        git status --short
    }
    catch {
        Write-Status "Failed to get git status: $_" "ERROR"
    }
    Write-Host ""
}

function Show-RecentCommits {
    Write-Status "Recent commits:"
    try {
        git log --oneline -5
    }
    catch {
        Write-Status "Failed to get recent commits: $_" "ERROR"
    }
    Write-Host ""
}

function Get-TodoStatus {
    $todoFiles = @("TODO.md", "TODO.txt", "todos.md")
    
    foreach ($file in $todoFiles) {
        if (Test-Path $file) {
            try {
                $content = Get-Content $file -Raw
                $lines = $content -split "`n"
                foreach ($line in $lines) {
                    if ($line -match "✅|completed") {
                        return $line.Trim()
                    }
                }
            }
            catch {
                continue
            }
        }
    }
    
    return $null
}

function Show-Help {
    Write-Host "Usage: .\git_push.ps1 [TODO_ITEM] [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  TODO_ITEM    Description of the completed TODO item"
    Write-Host "  -AutoTodo    Automatically detect TODO status from TODO files"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\git_push.ps1 'Implement data ingestion pipeline'"
    Write-Host "  .\git_push.ps1 'Add anomaly detection models'"
    Write-Host "  .\git_push.ps1 -AutoTodo"
    Write-Host ""
    Write-Host "This script will:"
    Write-Host "  1. Check if you're in a git repository"
    Write-Host "  2. Add all changes to staging"
    Write-Host "  3. Create a commit with the TODO item description"
    Write-Host "  4. Push changes to the remote repository"
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "=========================================="
Write-Host "🚀 Git Push Script for $RepoName"
Write-Host "=========================================="
Write-Host ""

# Check if we're in a git repository
if (-not (Test-GitRepository)) {
    exit 1
}

# Show current status
Show-GitStatus

# Check if there are changes to commit
if (-not (Test-GitChanges)) {
    Write-Status "No changes detected. Nothing to commit and push." "WARNING"
    exit 0
}

# Get TODO item if auto mode is enabled
if ($AutoTodo -and -not $TodoItem) {
    $TodoItem = Get-TodoStatus
}

# Add all changes
if (-not (Add-GitChanges)) {
    exit 1
}

# Create commit message
$commitMessage = New-CommitMessage $TodoItem
Write-Status "Commit message: $commitMessage"

# Commit changes
if (-not (Commit-GitChanges $commitMessage)) {
    exit 1
}

# Push to remote
if (-not (Push-ToRemote)) {
    exit 1
}

Write-Status "✅ All changes have been successfully pushed to remote!" "SUCCESS"
Show-RecentCommits

Write-Host ""
Write-Host "=========================================="
Write-Status "Git push completed successfully! 🎉" "SUCCESS"
Write-Host "=========================================="
