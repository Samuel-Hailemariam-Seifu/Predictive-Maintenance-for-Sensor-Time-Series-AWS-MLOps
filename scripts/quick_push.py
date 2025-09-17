#!/usr/bin/env python3
"""
Quick Git Push Script
A simplified version for quick commits and pushes
"""

import subprocess
import sys
from datetime import datetime

def run_command(command):
    """Run a command and return success status."""
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_push.py 'TODO description'")
        print("Example: python quick_push.py 'Implement data ingestion pipeline'")
        sys.exit(1)
    
    todo_description = sys.argv[1]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"✅ {todo_description} - {timestamp}"
    
    print(f"🚀 Quick Git Push: {todo_description}")
    print("=" * 50)
    
    # Check git status
    if not run_command("git status --porcelain"):
        print("❌ Not a git repository or no changes detected")
        sys.exit(1)
    
    # Add all changes
    print("📁 Adding changes...")
    if not run_command("git add ."):
        print("❌ Failed to add changes")
        sys.exit(1)
    
    # Commit changes
    print("💾 Committing changes...")
    if not run_command(f'git commit -m "{commit_message}"'):
        print("❌ Failed to commit changes")
        sys.exit(1)
    
    # Push to remote
    print("🚀 Pushing to remote...")
    if not run_command("git push"):
        print("❌ Failed to push to remote")
        sys.exit(1)
    
    print("✅ Successfully pushed to remote!")
    print(f"📝 Commit: {commit_message}")

if __name__ == "__main__":
    main()
