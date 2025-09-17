#!/usr/bin/env python3
"""
Git Push Script for Predictive Maintenance Project
This script automatically commits and pushes changes to the remote repository
"""

import subprocess
import sys
import os
from datetime import datetime
import argparse
from typing import Optional, List, Tuple


class GitPushScript:
    """Git push automation script."""
    
    def __init__(self, repo_name: str = "Predictive Maintenance for Sensor Time-Series (AWS + MLOps)",
                 branch_name: str = "main", remote_name: str = "origin"):
        self.repo_name = repo_name
        self.branch_name = branch_name
        self.remote_name = remote_name
        
        # Colors for output
        self.colors = {
            'RED': '\033[0;31m',
            'GREEN': '\033[0;32m',
            'YELLOW': '\033[1;33m',
            'BLUE': '\033[0;34m',
            'NC': '\033[0m'  # No Color
        }
    
    def print_status(self, message: str, status: str = "INFO"):
        """Print colored status message."""
        color = self.colors.get(status.upper(), self.colors['NC'])
        print(f"{color}[{status}]{self.colors['NC']} {message}")
    
    def run_command(self, command: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run a git command and return success status and output."""
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=check
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
    
    def check_git_repo(self) -> bool:
        """Check if we're in a git repository."""
        success, _ = self.run_command(["git", "rev-parse", "--git-dir"], check=False)
        if not success:
            self.print_status("Not in a git repository. Please run 'git init' first.", "ERROR")
            return False
        return True
    
    def check_changes(self) -> bool:
        """Check if there are any changes to commit."""
        # Check for unstaged changes
        success, _ = self.run_command(["git", "diff", "--quiet"], check=False)
        if not success:
            return True
        
        # Check for staged changes
        success, _ = self.run_command(["git", "diff", "--cached", "--quiet"], check=False)
        if not success:
            return True
        
        return False
    
    def create_commit_message(self, todo_item: Optional[str] = None) -> str:
        """Create a commit message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if todo_item:
            return f"✅ Completed: {todo_item} - {timestamp}"
        else:
            return f"🔄 Update: {timestamp}"
    
    def add_changes(self) -> bool:
        """Add all changes to staging."""
        self.print_status("Adding all changes...")
        success, output = self.run_command(["git", "add", "."])
        
        if success:
            self.print_status("Changes added to staging area", "SUCCESS")
        else:
            self.print_status(f"Failed to add changes: {output}", "ERROR")
        
        return success
    
    def commit_changes(self, commit_message: str) -> bool:
        """Commit changes with the given message."""
        self.print_status("Committing changes...")
        success, output = self.run_command(["git", "commit", "-m", commit_message])
        
        if success:
            self.print_status("Changes committed successfully", "SUCCESS")
        else:
            self.print_status(f"Failed to commit changes: {output}", "ERROR")
        
        return success
    
    def push_to_remote(self) -> bool:
        """Push changes to remote repository."""
        self.print_status("Pushing to remote repository...")
        
        # Check if remote exists
        success, _ = self.run_command(["git", "remote", "get-url", self.remote_name], check=False)
        if not success:
            self.print_status(f"Remote '{self.remote_name}' not found. Please add it first:", "WARNING")
            self.print_status(f"git remote add {self.remote_name} <repository-url>", "WARNING")
            return False
        
        # Push to remote
        success, output = self.run_command(["git", "push", self.remote_name, self.branch_name])
        
        if success:
            self.print_status(f"Successfully pushed to {self.remote_name}/{self.branch_name}", "SUCCESS")
        else:
            self.print_status(f"Failed to push to remote: {output}", "ERROR")
        
        return success
    
    def show_status(self):
        """Show current git status."""
        self.print_status("Current git status:")
        success, output = self.run_command(["git", "status", "--short"], check=False)
        if success:
            print(output)
        print()
    
    def show_recent_commits(self):
        """Show recent commits."""
        self.print_status("Recent commits:")
        success, output = self.run_command(["git", "log", "--oneline", "-5"], check=False)
        if success:
            print(output)
        print()
    
    def get_todo_status(self) -> Optional[str]:
        """Get current TODO status from TODO file if it exists."""
        todo_files = ["TODO.md", "TODO.txt", "todos.md"]
        
        for todo_file in todo_files:
            if os.path.exists(todo_file):
                try:
                    with open(todo_file, 'r') as f:
                        content = f.read()
                        # Look for completed items (simple heuristic)
                        lines = content.split('\n')
                        for line in lines:
                            if '✅' in line or 'completed' in line.lower():
                                return line.strip()
                except Exception:
                    continue
        
        return None
    
    def run(self, todo_item: Optional[str] = None, auto_todo: bool = False):
        """Main function to run the git push process."""
        print("=" * 50)
        print(f"🚀 Git Push Script for {self.repo_name}")
        print("=" * 50)
        print()
        
        # Check if we're in a git repository
        if not self.check_git_repo():
            return False
        
        # Show current status
        self.show_status()
        
        # Check if there are changes to commit
        if not self.check_changes():
            self.print_status("No changes detected. Nothing to commit and push.", "WARNING")
            return True
        
        # Get TODO item if auto mode is enabled
        if auto_todo and not todo_item:
            todo_item = self.get_todo_status()
        
        # Add all changes
        if not self.add_changes():
            return False
        
        # Create commit message
        commit_message = self.create_commit_message(todo_item)
        self.print_status(f"Commit message: {commit_message}")
        
        # Commit changes
        if not self.commit_changes(commit_message):
            return False
        
        # Push to remote
        if not self.push_to_remote():
            return False
        
        self.print_status("✅ All changes have been successfully pushed to remote!", "SUCCESS")
        self.show_recent_commits()
        
        print()
        print("=" * 50)
        self.print_status("Git push completed successfully! 🎉", "SUCCESS")
        print("=" * 50)
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Git Push Script for Predictive Maintenance Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python git_push.py "Implement data ingestion pipeline"
  python git_push.py "Add anomaly detection models"
  python git_push.py "Set up AWS integration"
  python git_push.py --auto-todo
  python git_push.py --help

This script will:
  1. Check if you're in a git repository
  2. Add all changes to staging
  3. Create a commit with the TODO item description
  4. Push changes to the remote repository
        """
    )
    
    parser.add_argument(
        "todo_item",
        nargs="?",
        help="Description of the completed TODO item"
    )
    
    parser.add_argument(
        "--auto-todo",
        action="store_true",
        help="Automatically detect TODO status from TODO files"
    )
    
    parser.add_argument(
        "--repo-name",
        default="Predictive Maintenance for Sensor Time-Series (AWS + MLOps)",
        help="Repository name for display purposes"
    )
    
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name to push to (default: main)"
    )
    
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote name to push to (default: origin)"
    )
    
    args = parser.parse_args()
    
    # Create and run the script
    script = GitPushScript(
        repo_name=args.repo_name,
        branch_name=args.branch,
        remote_name=args.remote
    )
    
    success = script.run(
        todo_item=args.todo_item,
        auto_todo=args.auto_todo
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
