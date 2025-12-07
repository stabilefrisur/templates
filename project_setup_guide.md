# Project Setup Guide

## Prerequisites
- PowerShell terminal
- Git installed
- GitLab access
- Python 3.12 installed

## Setup Steps

### 1. Clone or Initialize a Git Repo
Clone an existing repo and **complete steps 2-4**.
```powershell
git clone <gitlab-existing-repo-url> <project-name>
cd <project-name>
```

Alternatively, start a new project and **complete all steps**.
```powershell
git init <project-name>
cd <project-name>
```

### 2. Create Virtual Environment
```powershell
& "C:\Program Files\Python312\python.exe" -m venv .\.venv
```

### 3. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

Adjust your IDE settings to avoid having to activate the environment manually

**VS Code** Add `"python.terminal.activateEnvironment": true` to your settings.json  
**PyCharm** Go to File → Settings → Tools → Terminal and check the box `Activate virtualenv`

### 4. Install uv in Virtual Environment
```powershell
pip install uv
```

### 5. Initialize the project with uv
```powershell
uv init --lib --python 3.12 --build-backend setuptools
```

### 6. Configure pyproject.toml
```toml
[tool.uv]
index-url = "https://pypi.site.gs.com/simple"
allow-insecure-host = ["pypi.site.gs.com"]
```

### 7. Add dependencies
```powershell
uv add pandas plotly
```

### 8. Add developer dependencies
```powershell
uv add --dev pytest ruff mypy
```

### 9. Create .gitignore
```powershell
@"
.venv/
dist/
build/
*.pyc
__pycache__/
.env
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

### 10. Git commit and push to Gitlab
```
git add .
git commit -m "chore: Initial commit"
git remote add origin <gitlab-new-repo-url>
git push -u origin main
```

## Development Workflow

### Format Code
```powershell
uv run ruff format .
```

### Lint Code
```powershell
uv run ruff check .
```

### Type Check
```powershell
uv run mypy .
```

## Notes
- uv is installed only within the virtual environment, not system-wide
- Always activate `.venv` before running commands
- Use `uv pip` for dependency management within the venv
- Use `uv run` to execute tools like ruff and mypy
