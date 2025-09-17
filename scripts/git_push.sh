#!/bin/bash

# Git Push Script for Predictive Maintenance Project
# This script automatically commits and pushes changes to the remote repository

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="Predictive Maintenance for Sensor Time-Series (AWS + MLOps)"
BRANCH_NAME="main"
REMOTE_NAME="origin"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository. Please run 'git init' first."
        exit 1
    fi
}

# Function to check if there are any changes to commit
check_changes() {
    if git diff --quiet && git diff --cached --quiet; then
        print_warning "No changes to commit."
        return 1
    fi
    return 0
}

# Function to get current TODO status
get_todo_status() {
    # This function can be customized to read from a TODO file or use a specific format
    # For now, we'll use a simple approach
    if [ -f "TODO.md" ]; then
        echo "TODO status from TODO.md"
    else
        echo "Manual commit"
    fi
}

# Function to create commit message
create_commit_message() {
    local todo_item="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ -n "$todo_item" ]; then
        echo "✅ Completed: $todo_item - $timestamp"
    else
        echo "🔄 Update: $timestamp"
    fi
}

# Function to add all changes
add_changes() {
    print_status "Adding all changes..."
    git add .
    print_success "Changes added to staging area"
}

# Function to commit changes
commit_changes() {
    local commit_message="$1"
    
    print_status "Committing changes..."
    git commit -m "$commit_message"
    print_success "Changes committed successfully"
}

# Function to push to remote
push_to_remote() {
    print_status "Pushing to remote repository..."
    
    # Check if remote exists
    if ! git remote get-url $REMOTE_NAME > /dev/null 2>&1; then
        print_warning "Remote '$REMOTE_NAME' not found. Please add it first:"
        print_warning "git remote add $REMOTE_NAME <repository-url>"
        return 1
    fi
    
    # Push to remote
    if git push $REMOTE_NAME $BRANCH_NAME; then
        print_success "Successfully pushed to $REMOTE_NAME/$BRANCH_NAME"
    else
        print_error "Failed to push to remote. Please check your connection and permissions."
        return 1
    fi
}

# Function to show git status
show_status() {
    print_status "Current git status:"
    git status --short
    echo ""
}

# Function to show recent commits
show_recent_commits() {
    print_status "Recent commits:"
    git log --oneline -5
    echo ""
}

# Main function
main() {
    local todo_item="$1"
    
    echo "=========================================="
    echo "🚀 Git Push Script for $REPO_NAME"
    echo "=========================================="
    echo ""
    
    # Check if we're in a git repository
    check_git_repo
    
    # Show current status
    show_status
    
    # Check if there are changes to commit
    if ! check_changes; then
        print_warning "No changes detected. Nothing to commit and push."
        exit 0
    fi
    
    # Add all changes
    add_changes
    
    # Create commit message
    local commit_message=$(create_commit_message "$todo_item")
    print_status "Commit message: $commit_message"
    
    # Commit changes
    commit_changes "$commit_message"
    
    # Push to remote
    if push_to_remote; then
        print_success "✅ All changes have been successfully pushed to remote!"
        show_recent_commits
    else
        print_error "❌ Failed to push changes to remote."
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    print_success "Git push completed successfully! 🎉"
    echo "=========================================="
}

# Function to show help
show_help() {
    echo "Usage: $0 [TODO_ITEM]"
    echo ""
    echo "Options:"
    echo "  TODO_ITEM    Description of the completed TODO item"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 \"Implement data ingestion pipeline\""
    echo "  $0 \"Add anomaly detection models\""
    echo "  $0 \"Set up AWS integration\""
    echo ""
    echo "This script will:"
    echo "  1. Check if you're in a git repository"
    echo "  2. Add all changes to staging"
    echo "  3. Create a commit with the TODO item description"
    echo "  4. Push changes to the remote repository"
}

# Parse command line arguments
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
