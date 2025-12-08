# Copilot Instructions for [project name]

This file provides comprehensive guidance for AI coding assistants working on [project description].

---

## Template Placeholders

Replace the following placeholders throughout this file:

- `[project name]` - Your project's package name in lowercase_with_underscores format
- `[project type]` - Type of project (e.g., "data analysis tool", "web scraper", "machine learning pipeline")
- `[project description]` - What your project does and its main features/capabilities
- `[Date]` - Last updated date
- `[Version]` - Document version number
- `[Maintainer]` - Project maintainer name/contact

---

## Quick Start for AI Agents

**Essential Commands:**
```bash
# Environment setup
uv sync                    # Install all dependencies

# Testing & Quality
uv run pytest              # Run all tests
uv run mypy src/           # Type checking
uv run ruff check src/     # Linting
```

**Critical Paths:**
- **Project root**: `config.PROJECT_ROOT = Path(__file__).parent.parent.parent`
- **Tests**: `tests/{layer}/` (mirrors `src/` structure)

**Essential Patterns:**
1. **Modern Python**: Use `str | None` not `Optional[str]`, `dict[str, Any]` not `Dict[str, Any]`
2. **Frozen configs**: Use `@dataclass(frozen=True)` with `__post_init__` validation
3. **Logging**: Use `logger = logging.getLogger(__name__)` at module level, never `basicConfig()`
4. **Functions over classes**: Pure functions for transformations, dataclasses for containers
5. **Return figures, never display**: Visualizations return `go.Figure`, caller controls rendering
6. **Vectorized operations**: Prefer NumPy/Pandas vectorized operations over loops

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [File Organization](#file-organization)
4. [Integration Rules](#integration-rules)
5. [Code Patterns & Conventions](#code-patterns--conventions)

---

## Project Overview

### Purpose

[project name] is a **Python 3.12 [project type]** for [project description].

### Design Philosophy

1. **Modularity**: Clean layer separation (data → models → processing → evaluation)
2. **Reproducibility**: Fixed seeds, metadata logging, deterministic outputs
3. **Type Safety**: Modern Python syntax (`str | None`, `dict[str, Any]`)
4. **Functions Over Classes**: Pure functions for transformations, `@dataclass` for containers
5. **Performance**: Vectorized operations over loops for numerical computations
6. **Transparency**: Clear conventions, explicit calculations
7. **No Legacy Support**: Breaking changes without deprecation warnings

### Workflow for Multi-Step Tasks

1. **Understand**: Read relevant files and documentation to grasp context
2. **Plan**: Break complex tasks into discrete, testable steps
3. **Implement**: Execute changes with proper type hints and docstrings
4. **Validate**: Run tests or check for errors after modifications
5. **Document**: Update only if creating new public APIs or changing behavior

**For simple tasks:** Skip planning and implement directly.

**Communication style:** Do NOT use decorative emojis in responses, code, or documentation unless explicitly requested.

---

## File Organization

[Add your project's file/folder structure and organization conventions here]

---

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.12 | Strict requirement, modern syntax only |
| **Data Analysis** | Pandas | 2.0+ | Time series manipulation |
| **Numerical Computing** | NumPy | 1.24+ | Array operations |
| **Visualization** | Plotly | 5.24+ | Interactive charts |
| **Statistics** | Statsmodels | 0.14+ | Statistical tests and regression |
| **Testing** | Pytest | 8.0+ | Unit and integration tests |
| **Type Checking** | Mypy | 1.11+ | Static type validation |

### Build & Package Management

- **uv**: Package installer, environment manager, and task runner
- **pyproject.toml**: Modern project metadata (PEP 621)
- **setuptools**: Build backend for distribution

### Code Quality Tools

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **pytest**: Test runner

### Modern Python Features

```python
# Union types (PEP 604)
def fetch_data(source: FileSource | ApiSource) -> pd.DataFrame | None:
    ...

# Built-in generics (PEP 585)
def process_items(items: dict[str, pd.Series]) -> list[Result]:
    ...

# Frozen dataclasses
@dataclass(frozen=True)
class Config:
    lookback: int = 20
    min_periods: int = 10
```

---

## Integration Rules

### 1. Workflow Context Sharing

**Required:** Steps communicate via context dict:

```python
# Step N produces output
def execute(self, context: dict[str, Any]) -> dict[str, Any]:
    return {"my_output": result}

# Step N+1 consumes it
def execute(self, context: dict[str, Any]) -> dict[str, Any]:
    prev_output = context["step_name"]["my_output"]
```

### 2. Frozen Dataclass Configs

**Required:** All configs use frozen dataclasses:

```python
@dataclass(frozen=True)
class DomainConfig:
    param1: float
    param2: int = 10
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.param1 < 0:
            raise ValueError("param1 must be non-negative")
```

### 3. Return Figures, Never Display

**Required:** Visualization functions return figures:

```python
def plot_chart(data: pd.Series) -> go.Figure:
    """Generate chart - caller controls rendering."""
    fig = px.line(x=data.index, y=data.values)
    return fig  # Do NOT call .show()

# Caller decides
fig = plot_chart(data)
fig.show()  # OR st.plotly_chart(fig)  OR fig.write_html("chart.html")
```

### 4. Module-Level Loggers

**Required:** Use module-level loggers, never `basicConfig()`:

```python
import logging
logger = logging.getLogger(__name__)

# INFO for user-facing operations
logger.info("Loaded %d items from catalog", len(items))

# DEBUG for implementation details
logger.debug("Cache hit: key=%s", cache_key)
```

**DO NOT**:
```python
# ❌ Never in library code
logging.basicConfig(level=logging.INFO)
```

### 5. Modern Type Hints

**Required:** Use PEP 604 union syntax and built-in generics:

```python
# ✅ Correct
def process(data: dict[str, Any], filters: list[str] | None = None) -> pd.DataFrame | None:
    ...

# ❌ Do NOT use
from typing import Optional, Union, List, Dict
def process(data: Dict[str, Any], filters: Optional[List[str]] = None) -> Union[pd.DataFrame, None]:
    ...
```

### 6. Advanced Type Patterns

**Use these patterns for better type safety:**

```python
from typing import Literal, TypedDict
from collections.abc import Callable
from pathlib import Path

# Accept both str and Path
def load_file(path: str | Path) -> pd.DataFrame:
    ...

# Restrict to specific values
def set_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    ...

# Callable type hints
def apply_transform(func: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    ...

# Structured dictionaries
class ResultMetadata(TypedDict):
    timestamp: str
    params: dict[str, Any]
```

---

## Code Patterns & Conventions

### 1. Function-Based Transformations

```python
# ✅ Prefer pure functions
def transform_data(data: pd.Series, window: int) -> pd.Series:
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / std

# ❌ Avoid classes for simple transformations
class DataTransformer:
    def __init__(self, window: int):
        self.window = window
    def transform(self, data: pd.Series) -> pd.Series:
        ...
```

### 2. Vectorized Operations

**Prefer:** NumPy/Pandas vectorized operations for performance:

```python
# ✅ Vectorized (fast)
def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute returns using vectorized operations."""
    return prices.pct_change()

def apply_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold using NumPy vectorization."""
    return np.where(values > threshold, values, 0.0)

def compute_zscore(data: pd.Series) -> pd.Series:
    """Compute z-score using vectorized operations."""
    return (data - data.mean()) / data.std()

# ❌ Avoid loops (slow)
def compute_returns_slow(prices: pd.Series) -> pd.Series:
    returns = []
    for i in range(1, len(prices)):
        ret = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
        returns.append(ret)
    return pd.Series(returns)
```

**When loops are acceptable:**
- Complex logic that cannot be vectorized
- Small datasets where performance is negligible
- Readability is significantly improved with explicit iteration

### 3. Rolling Window Computations

```python
def compute_rolling_metric(data: pd.Series, window: int = 252) -> pd.Series:
    """Compute metric over rolling windows using vectorized operations."""
    # Use pandas rolling with apply for custom aggregations
    result = data.rolling(window=window, min_periods=window).apply(
        lambda x: compute_metric(x), raw=False
    )
    return result

# For simple operations, use built-in methods directly
def compute_rolling_zscore(data: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling z-score using vectorized operations."""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    return (data - rolling_mean) / rolling_std
```

### 4. Timestamped Output Directories

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dirname = f"{self.config.label}_{timestamp}"  # Label-based naming
output_dir = DATA_DIR / "workflows" / dirname
output_dir.mkdir(parents=True, exist_ok=True)
```

### 5. Cache Checking Pattern

```python
def execute(self, context: dict[str, Any]) -> dict[str, Any]:
    # Check cache
    if not self.config.force_rerun and self.output_exists():
        logger.info("Loading cached output for %s", self.name)
        return self.load_cached_output()
    
    # Execute logic
    result = self._compute_result(context)
    
    # Save for next run
    self._save_output(result)
    
    return result
```

### 6. Automatic Directory Creation

```python
def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=True)
    return path
```

### 7. Error Handling and Validation

**Be specific with exceptions and chain errors:**

```python
import json

def load_config(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Config file not found at %s", path)
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted config: {e}") from e

def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if required columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
```

### 8. NumPy-Style Docstrings

```python
def process_data(
    data: pd.DataFrame,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Process data with rolling window computation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with required columns.
    window : int, default 20
        Rolling window size.
    min_periods : int or None, optional
        Minimum observations in window.
    
    Returns
    -------
    pd.Series
        Processed series with computed metric.
    
    Raises
    ------
    ValueError
        If window size is invalid.
    
    Examples
    --------
    >>> df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    >>> result = process_data(df, window=3)
    >>> result.head()
    """
    # Implementation...
```

### 9. Testing Patterns

**Use fixtures for deterministic test data:**

```python
import pytest
import numpy as np

@pytest.fixture
def sample_data() -> pd.Series:
    """Create deterministic test data."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100))

def test_compute_metric(sample_data):
    result = compute_metric(sample_data, window=20)
    assert isinstance(result, pd.Series)
    assert result.iloc[:19].isna().all()  # Check NaN in first window
    
def test_invalid_input():
    with pytest.raises(ValueError, match="Window must be >= 2"):
        compute_metric(pd.Series([1, 2, 3]), window=1)
```

### 10. Advanced Dataclass Features

**Use these patterns for data containers:**

```python
from dataclasses import dataclass, field, asdict
from typing import ClassVar

@dataclass(frozen=True)
class Config:
    VERSION: ClassVar[str] = "1.0"  # Class variable
    name: str  # Required
    options: dict[str, float] = field(default_factory=dict)  # Mutable default
    _cache: dict = field(default_factory=dict, repr=False)  # Hidden from repr
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 11. Import Organization Rules

**Use relative imports within package, absolute from outside:**

```python
# Within package: myproject/models/processor.py
from ..data.loader import load_data
from .base import BaseProcessor

# From tests/ or scripts/
from myproject.models.processor import DataProcessor
```

---

## Git Commit Standards

Follow **Conventional Commits** format for consistency and automated changelog generation.

### Format
```
<type>: <description>

[optional body]
```

### Rules
- **Type**: Lowercase, from the list below
- **Description**: Capitalize first letter, no period at end, imperative mood ("Add" not "Added")
- **Length**: Keep first line under 72 characters
- **Body**: Optional, explain *why* not *what*, wrap at 72 characters

### Types
- `feat`: New feature or capability
- `fix`: Bug fix or correction
- `docs`: Documentation changes only
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `perf`: Performance improvement
- `chore`: Maintenance (dependencies, tooling, config)
- `style`: Formatting, missing semicolons (not CSS)

### Examples

**Good:**
```
feat: Add custom metric computation

Implements normalized difference calculation to identify
divergence patterns in time series data.
```

```
refactor: Extract data loading to separate module

Improves modularity and testability by separating I/O from computation.
```

```
docs: Update persistence layer documentation
```

**Bad:**
```
Added new feature
```

```
Fix: bug in processing
```

```
update docs.
```

### Multi-file Commits
When changing multiple files, use the most significant type and describe the overall change:
```
refactor: Modernize type hints to modern Python syntax

- Update copilot instructions with new standards
- Add comprehensive Python guidelines document
- Remove legacy Optional/Union usage examples
```

---

## Prohibited Patterns

**1. Old-Style Type Hints:**
```python
from typing import Optional, Union, List, Dict
def func(x: Optional[int]) -> Union[str, None]:  # Wrong
    ...
```

**2. Classes for Simple Transformations:**
```python
class DataProcessor:  # Wrong - use pure function
    def __init__(self, window: int):
        self.window = window
    def process(self, data: pd.Series) -> pd.Series:
        return data.rolling(self.window).mean()
```

**3. Auto-Display in Visualization:**
```python
def plot_chart(data: pd.Series) -> None:  # Wrong
    fig = px.line(data)
    fig.show()  # Don't auto-display
```

**4. Mutable Config Objects:**
```python
@dataclass  # Wrong - missing frozen=True
class MyConfig:
    param: int
```

**5. `logging.basicConfig()` in Library Code:**
```python
import logging
logging.basicConfig(level=logging.INFO)  # Wrong - library shouldn't configure logging
```

**6. Loops Instead of Vectorized Operations:**
```python
# Wrong - slow loop-based computation
def compute_metric(data: pd.Series) -> pd.Series:
    result = []
    for value in data:
        result.append(value * 2 + 1)
    return pd.Series(result)
```

---

## Quick Reference

### File Naming Conventions
- Python modules: lowercase with underscores (`my_module.py`)
- Test files: `test_` prefix (`test_my_module.py`)
- Config dataclasses: `MyConfig` in `config.py`

### Import Organization
```python
# Standard library
import logging
from pathlib import Path
from typing import Any

# Third-party
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Local
from [project name].config import DATA_DIR
from [project name].models import compute_metric
```

### Logging Levels
- **INFO**: User-facing operations (`Loaded 10 items`, `Workflow complete`)
- **DEBUG**: Implementation details (`Cache hit`)
- **WARNING**: Recoverable issues (`Missing optional column`)
- **ERROR**: Failures (`Step failed`, `Validation error`)

### Key Constants
- `PROJECT_ROOT`: Project root directory (from `config/__init__.py`)

---

**Last Updated**: [Date]  
**Version**: [Version]  
**Maintainer**: [Maintainer]
