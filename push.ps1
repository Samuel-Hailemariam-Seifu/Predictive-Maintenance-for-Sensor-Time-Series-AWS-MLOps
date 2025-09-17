# Quick Git Push Wrapper for PowerShell
# Usage: .\push.ps1 "TODO description"

param(
    [Parameter(Mandatory=$true)]
    [string]$TodoDescription
)

python scripts/git_push.py $TodoDescription
