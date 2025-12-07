# Python project — concise uv + PowerShell + GitLab workflow

This is a compact, modern reference for creating, developing and publishing a Python project using uv, git and GitLab. Assumes a PowerShell shell on Windows and that you use `uv` for venv, dependency and project management.

Core ideas (short):
- Use a `pyproject.toml`-based layout with source in `src/`.
- Keep `.venv\` out of source control. Use `uv venv` to create/manage the venv.
- Treat `pyproject.toml` as the single source of truth; run `uv sync` after manual edits.
- Default branch name: `master` (or `main` if you prefer the modern default).

Prerequisites:
- uv (CLI), git, a GitLab account (or self-hosted GitLab instance).

## Quick start

Initialize a new project and create a venv (example using Python 3.13):

```powershell
uv init my_project --python 3.13
Set-Location my_project
# create the venv (uses .python-version if present)
uv venv
```

Activate the venv (PowerShell):

```powershell
# If your execution policy blocks scripts you may run this first to allow activation for the session:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
# then activate
. .venv\Scripts\Activate.ps1
```

Run the project or a script with uv (no install required):

```powershell
uv run src\my_project\main.py    # run a script directly
uv run my_project                 # run an installed entry point (see below)
```

## Source layout and build system

Prefer the src layout:

```
my_project/
├─ pyproject.toml
└─ src/
	└─ my_project/
		└─ main.py
```

Use uv's native build backend (`uv_build`) for a fast, zero-config build by default. Add this to `pyproject.toml`:

```toml
[build-system]
requires = ["uv_build>=0.9.5,<0.10.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
# optional overrides; defaults expect a single module under src/<name>
module-root = "src"
```

After editing `pyproject.toml` manually run:

```powershell
uv sync   # ensures venv metadata and pyproject stay in sync
```

## Dependencies (best practices)

- Prefer pinned versions for reproducibility: `uv add requests==2.31.0`.
- Add dev dependencies with `--dev` and commit them to `pyproject.toml`.

Examples:

```powershell
uv add pandas==2.2.0 scikit-learn plotly
uv add --dev pytest black ruff
uv pip list    # list packages in the current uv-managed venv
```

If you need to remove a package:

```powershell
uv remove matplotlib
```

## Editable install & entry points

Register a CLI entry point in `pyproject.toml`:

```toml
[project.scripts]
my_project = "my_project.main:main"
```

Install in editable mode for local development:

```powershell
uv pip install -e .
uv run my_project    # now available via uv run my_project
```

## Git & GitLab (concise)

Initialize repo, commit, and push to GitLab (use `master` or `main` as your default):

```powershell
git init
# optionally set default branch name
git branch -M master
git add .
git commit -m "chore: initial commit"
# create repo on GitLab (web UI or use the `glab` CLI), then add remote
git remote add origin https://gitlab.com/<your-namespace>/my_project.git
git push -u origin master
```

Notes on creating a GitLab repo:
- Use the GitLab web UI to create a new project under your namespace or group.
- Optional: install and use the `glab` CLI to create a project from the command line (see https://github.com/profclems/glab).

`.gitignore` essentials (example):

```
.venv\
dist/
build/
*.pyc
__pycache__/
.env
```

Stop tracking a file already committed:

```powershell
git rm --cached path\to\file
git commit -m "chore: remove sensitive file from VCS"
```

Branch workflow (feature branch → merge request → merge to master):

```powershell
git checkout -b feat/awesome
# work, stage, commit
git push -u origin feat/awesome
# open a Merge Request on GitLab and merge to master
git checkout master
git pull origin master
git branch -d feat/awesome
git push origin --delete feat/awesome
```

## Testing with pytest

Create `tests/` and configure pytest via `pyproject.toml` if desired:

```toml
[tool.pytest.ini_options]
pythonpath = "src"
testpaths = ["tests"]
```

Run tests:

```powershell
uv run -m pytest
# or, if pytest is in the venv
pytest
```

## Build & publish

Build sdist and wheel:

```powershell
uv build
# outputs will be in dist/
```

Synchronize and publish (follow prompts for credentials):

```powershell
uv sync
uv publish
```

Notes: always commit and push tags/changes before publishing. Prefer using test.pypi.org for initial tests.

Tip: when running builds in CI or when you want to ensure the build works without any custom `tool.uv.sources` settings, run:

```powershell
uv build --no-sources
```
This forces the build to use only `build-system.requires` and matches other frontends' behaviour.

## GitLab CI (quick example)

Add a basic `.gitlab-ci.yml` to run lint, tests and build on GitLab CI/CD. This is a minimal example that installs Python, uv and runs tests. Adjust the Python image and steps to your needs.

```yaml
image: python:3.11

stages:
  - test
  - build

before_script:
  - python -m pip install --upgrade pip
  - pip install uv
  - uv venv
  - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force || true
  - . .venv/Scripts/Activate.ps1 || true

test:
  stage: test
  script:
    - uv run -m pytest

build:
  stage: build
  script:
    - uv build
  artifacts:
    paths:
      - dist/
```
```yaml
# Linux runner example (recommended for shared GitLab runners)
image: python:3.11

stages:
  - test
  - build

before_script:
  - python -m pip install --upgrade pip
  - pip install uv
  # prefer leaving venv creation to the job or use uv directly; many images have Python available

test:
  stage: test
  script:
    - uv venv || true
    - uv run -m pytest

build:
  stage: build
  script:
    - uv build
  artifacts:
    paths:
      - dist/

# Optional: Windows runner example (requires a Windows runner tagged `windows`)
windows-test:
  tags:
    - windows
  stage: test
  script:
    - powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force"
    - powershell -Command "python -m pip install --upgrade pip; pip install uv; uv venv"
    - powershell -Command ". .venv\\Scripts\\Activate.ps1; uv run -m pytest"
```

Notes:
- GitLab CI runners typically run non-interactive shells. The PowerShell activation lines above are provided for completeness; many CI images will instead install and run with the system Python or a virtual environment created by the runner.
- If you use a Windows-based GitLab runner, the PowerShell activation commands will apply; for Linux runners use the bash-style activation or run directly with `uv run` without activating.

## Short tips & best practices

- Keep `pyproject.toml` authoritative. Run `uv sync` after manual edits.
- Pin runtime deps for libraries; use ranges or caret for apps where appropriate.
- Keep secrets out of the repo. Use GitLab CI variables, or a secrets manager.
- Add CI (GitLab CI/CD) to run lint, tests and build on Merge Requests.
- Prefer `master` (or `main`) as the default branch name.

## Quick checklist

- [ ] uv init + uv venv
- [ ] add dependencies and run `uv sync`
- [ ] register entry points and `uv pip install -e .`
- [ ] add tests and CI (add `.gitlab-ci.yml`)
- [ ] build (`uv build`) and publish (`uv publish`)

---

If you want, I can also create a minimal `pyproject.toml` template, a `.gitignore`, and a `.gitlab-ci.yml` CI file pre-filled for this project. Which would you like next?
