# Git Push Scripts

This directory contains automated scripts to commit and push changes to the remote repository after completing TODO items.

## Available Scripts

### 1. `git_push.sh` (Linux/macOS)

Bash script for Unix-like systems.

**Usage:**

```bash
# Make executable
chmod +x scripts/git_push.sh

# Use with TODO item description
./scripts/git_push.sh "Implement data ingestion pipeline"

# Use without description
./scripts/git_push.sh

# Show help
./scripts/git_push.sh --help
```

### 2. `git_push.bat` (Windows)

Batch script for Windows systems.

**Usage:**

```cmd
# Use with TODO item description
scripts\git_push.bat "Add anomaly detection models"

# Use without description
scripts\git_push.bat

# Show help
scripts\git_push.bat -h
```

### 3. `git_push.py` (Cross-platform)

Python script that works on all platforms.

**Usage:**

```bash
# Use with TODO item description
python scripts/git_push.py "Set up AWS integration"

# Use without description
python scripts/git_push.py

# Auto-detect TODO status
python scripts/git_push.py --auto-todo

# Show help
python scripts/git_push.py --help
```

## Features

All scripts provide the following functionality:

- ✅ **Automatic Git Operations**: Add, commit, and push changes
- 🎯 **TODO Integration**: Include completed TODO items in commit messages
- 🔍 **Status Checking**: Verify git repository and changes before proceeding
- 🚨 **Error Handling**: Comprehensive error checking and user feedback
- 📊 **Status Display**: Show current git status and recent commits
- 🎨 **Colored Output**: Easy-to-read status messages
- 📝 **Commit Messages**: Automatic timestamped commit messages

## Configuration

### Default Settings

- **Repository Name**: "Predictive Maintenance for Sensor Time-Series (AWS + MLOps)"
- **Branch**: `main`
- **Remote**: `origin`

### Customization

#### Bash Script (`git_push.sh`)

Edit the configuration section at the top of the file:

```bash
REPO_NAME="Your Repository Name"
BRANCH_NAME="main"
REMOTE_NAME="origin"
```

#### Batch Script (`git_push.bat`)

Edit the configuration section at the top of the file:

```cmd
set REPO_NAME=Your Repository Name
set BRANCH_NAME=main
set REMOTE_NAME=origin
```

#### Python Script (`git_push.py`)

Use command-line arguments or modify the default values:

```python
script = GitPushScript(
    repo_name="Your Repository Name",
    branch_name="main",
    remote_name="origin"
)
```

## Workflow Integration

### For TODO Completion

1. Complete a TODO item
2. Run the appropriate script with the TODO description
3. Script automatically commits and pushes changes

### Example Workflow

```bash
# Complete a TODO item
# ... work on data ingestion pipeline ...

# Commit and push with description
./scripts/git_push.sh "Implement data ingestion pipeline"

# Or use Python script
python scripts/git_push.py "Implement data ingestion pipeline"
```

### Auto TODO Detection

The Python script can automatically detect completed TODO items from TODO files:

```bash
python scripts/git_push.py --auto-todo
```

## Error Handling

The scripts handle common error scenarios:

- **Not in Git Repository**: Prompts to run `git init`
- **No Changes**: Warns if there are no changes to commit
- **Remote Not Found**: Provides instructions to add remote
- **Push Failed**: Reports connection or permission issues

## Requirements

### Bash Script

- Git installed and configured
- Bash shell (Linux/macOS/WSL)

### Batch Script

- Git installed and configured
- Windows Command Prompt or PowerShell

### Python Script

- Python 3.6+
- Git installed and configured

## Troubleshooting

### Common Issues

1. **Permission Denied (Bash)**

   ```bash
   chmod +x scripts/git_push.sh
   ```

2. **Git Not Found**

   - Ensure Git is installed and in PATH
   - On Windows, may need to restart terminal after Git installation

3. **Remote Not Configured**

   ```bash
   git remote add origin <repository-url>
   ```

4. **Authentication Issues**
   - Configure Git credentials
   - Use SSH keys or personal access tokens

### Debug Mode

For the Python script, you can add debug output:

```python
# Add this to the script for more verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Auto-commit and push
  run: python scripts/git_push.py "Automated update from CI"
```

## Best Practices

1. **Always Review Changes**: Check `git status` before running scripts
2. **Use Descriptive Messages**: Provide clear TODO item descriptions
3. **Test First**: Run scripts on a test branch before main
4. **Backup Important Work**: Ensure important changes are backed up
5. **Regular Pushes**: Use scripts frequently to avoid large commits

## Contributing

To improve these scripts:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

These scripts are part of the Predictive Maintenance project and follow the same license terms.
