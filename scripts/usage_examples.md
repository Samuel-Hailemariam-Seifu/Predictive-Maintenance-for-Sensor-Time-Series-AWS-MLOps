# Git Push Scripts - Usage Examples

This document provides practical examples of how to use the git push scripts for the Predictive Maintenance project.

## Quick Start

### 1. Basic Usage (All Platforms)

**After completing a TODO item:**

```bash
# Linux/macOS
./scripts/git_push.sh "Implement data ingestion pipeline"

# Windows (Command Prompt)
scripts\git_push.bat "Add anomaly detection models"

# Windows (PowerShell)
.\scripts\git_push.ps1 "Set up AWS integration"

# Cross-platform (Python)
python scripts/git_push.py "Create monitoring dashboard"
```

### 2. Quick Push (Simplified)

```bash
python scripts/quick_push.py "Fix bug in data preprocessing"
```

## Real-World Examples

### Example 1: Data Pipeline Development

```bash
# After implementing data ingestion
./scripts/git_push.sh "Implement data ingestion pipeline with Kinesis integration"

# After adding preprocessing
python scripts/git_push.py "Add data preprocessing with feature engineering"

# After completing stream processing
.\scripts\git_push.ps1 "Complete real-time stream processing implementation"
```

### Example 2: ML Models Development

```bash
# After anomaly detection models
./scripts/git_push.sh "Implement anomaly detection models (Isolation Forest, LSTM)"

# After failure prediction
python scripts/git_push.py "Add failure prediction models with ensemble methods"

# After health scoring
.\scripts\git_push.ps1 "Complete health scoring system with multiple algorithms"
```

### Example 3: AWS Integration

```bash
# After S3 integration
./scripts/git_push.sh "Set up S3 data lake with automated data ingestion"

# After SageMaker setup
python scripts/git_push.py "Configure SageMaker for model training and deployment"

# After Lambda functions
.\scripts\git_push.ps1 "Implement Lambda functions for real-time processing"
```

### Example 4: MLOps Pipeline

```bash
# After CI/CD setup
./scripts/git_push.sh "Set up GitHub Actions for automated testing and deployment"

# After monitoring
python scripts/git_push.py "Implement comprehensive monitoring and alerting system"

# After documentation
.\scripts\git_push.ps1 "Complete project documentation and API reference"
```

## Advanced Usage

### 1. Auto TODO Detection

```bash
# Automatically detect completed TODO from files
python scripts/git_push.py --auto-todo
```

### 2. Custom Configuration

```bash
# Use different branch or remote
python scripts/git_push.py "Update models" --branch develop --remote upstream
```

### 3. Batch Operations

```bash
# Multiple commits in sequence
./scripts/git_push.sh "Add data validation"
./scripts/git_push.sh "Implement error handling"
./scripts/git_push.sh "Add unit tests"
```

## Integration with Development Workflow

### 1. Feature Branch Workflow

```bash
# Create feature branch
git checkout -b feature/data-ingestion

# Work on feature...
# ... implement data ingestion ...

# Commit and push feature
./scripts/git_push.sh "Complete data ingestion feature"

# Merge to main
git checkout main
git merge feature/data-ingestion
./scripts/git_push.sh "Merge data ingestion feature to main"
```

### 2. Bug Fix Workflow

```bash
# Create bug fix branch
git checkout -b bugfix/anomaly-detection-threshold

# Fix the bug...
# ... fix threshold calculation ...

# Commit and push fix
python scripts/git_push.py "Fix anomaly detection threshold calculation"

# Merge fix
git checkout main
git merge bugfix/anomaly-detection-threshold
python scripts/git_push.py "Merge bug fix for anomaly detection"
```

### 3. Documentation Updates

```bash
# Update README
python scripts/quick_push.py "Update README with new features"

# Add API documentation
./scripts/git_push.sh "Add comprehensive API documentation"

# Update examples
.\scripts\git_push.ps1 "Add usage examples and tutorials"
```

## Error Handling Examples

### 1. No Changes to Commit

```bash
$ ./scripts/git_push.sh "Test commit"
[WARNING] No changes to commit.
[INFO] No changes detected. Nothing to commit and push.
```

### 2. Not in Git Repository

```bash
$ ./scripts/git_push.sh "Test commit"
[ERROR] Not in a git repository. Please run 'git init' first.
```

### 3. Remote Not Configured

```bash
$ python scripts/git_push.py "Test commit"
[WARNING] Remote 'origin' not found. Please add it first:
[WARNING] git remote add origin <repository-url>
```

## Best Practices

### 1. Descriptive Commit Messages

```bash
# Good
./scripts/git_push.sh "Implement LSTM autoencoder for anomaly detection with 95% accuracy"

# Better
./scripts/git_push.sh "Add LSTM autoencoder anomaly detection model with sequence length 60 and encoding dim 32"

# Best
./scripts/git_push.sh "Implement LSTM autoencoder for real-time anomaly detection with configurable thresholds and ensemble voting"
```

### 2. Atomic Commits

```bash
# Good - one feature per commit
./scripts/git_push.sh "Add data validation for sensor readings"
./scripts/git_push.sh "Implement error handling for invalid data"
./scripts/git_push.sh "Add logging for data validation errors"

# Avoid - multiple features in one commit
./scripts/git_push.sh "Add data validation, error handling, and logging"
```

### 3. Regular Commits

```bash
# Work in progress
python scripts/quick_push.py "WIP: Add data preprocessing pipeline"

# Complete feature
python scripts/git_push.py "Complete data preprocessing pipeline with feature engineering"

# Bug fix
python scripts/quick_push.py "Fix memory leak in data preprocessing"
```

## Troubleshooting

### 1. Permission Issues (Linux/macOS)

```bash
chmod +x scripts/git_push.sh
./scripts/git_push.sh "Test commit"
```

### 2. PowerShell Execution Policy (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\git_push.ps1 "Test commit"
```

### 3. Python Path Issues

```bash
# Use full path
python3 scripts/git_push.py "Test commit"

# Or make executable
chmod +x scripts/git_push.py
./scripts/git_push.py "Test commit"
```

### 4. Git Authentication

```bash
# Configure Git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use SSH key
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
# Add public key to GitHub/GitLab

# Or use personal access token
git remote set-url origin https://username:token@github.com/user/repo.git
```

## Integration with IDEs

### 1. VS Code

Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Git Push with TODO",
      "type": "shell",
      "command": "python",
      "args": ["scripts/git_push.py", "${input:todoDescription}"],
      "group": "build"
    }
  ],
  "inputs": [
    {
      "id": "todoDescription",
      "description": "TODO description",
      "type": "promptString"
    }
  ]
}
```

### 2. PyCharm

Create external tool:

- Name: Git Push with TODO
- Program: python
- Arguments: scripts/git_push.py $Prompt$
- Working directory: $ProjectFileDir$

## Automation Examples

### 1. Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running pre-commit checks..."
python -m pytest tests/
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
echo "Pre-commit checks passed."
```

### 2. Post-commit Hook

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
echo "Post-commit: Pushing to remote..."
python scripts/git_push.py --auto-todo
```

### 3. Scheduled Push

Create `scripts/scheduled_push.py`:

```python
import schedule
import time
from git_push import GitPushScript

def auto_push():
    script = GitPushScript()
    script.run(auto_todo=True)

schedule.every(1).hours.do(auto_push)

while True:
    schedule.run_pending()
    time.sleep(60)
```

This comprehensive set of examples should help you integrate the git push scripts into your development workflow effectively!
