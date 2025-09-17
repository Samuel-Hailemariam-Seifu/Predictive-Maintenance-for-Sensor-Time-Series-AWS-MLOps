#!/bin/bash
# Quick Git Push Wrapper for Linux/macOS
# Usage: ./push.sh "TODO description"

if [ $# -eq 0 ]; then
    echo "Usage: ./push.sh \"TODO description\""
    echo "Example: ./push.sh \"Implement data ingestion pipeline\""
    exit 1
fi

python3 scripts/git_push.py "$1"
