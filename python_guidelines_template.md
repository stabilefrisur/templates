# Python Guidelines Template

## Overview

This document defines Python coding standards and best practices for modern Python 3.12+ projects. These guidelines ensure code quality, maintainability, and reproducibility across all modules.

**Target Audience:** Developers and AI assistants working on Python projects.

---

## Type Annotations

### Modern Python Syntax
Use **built-in generics** (PEP 585) and **union syntax** (PEP 604):

✅ **CORRECT:**
```python
def process_data(
    data: dict[str, Any],
    filters: list[str] | None = None,
    threshold: int | float = 0.0,
) -> pd.DataFrame | None:
    """Process data with optional filters."""
    ...
```

❌ **AVOID (legacy syntax):**
```python
from typing import Optional, Union, List, Dict

def process_data(
    data: Dict[str, Any],
    filters: Optional[List[str]] = None,
    threshold: Union[int, float] = 0.0,
) -> Union[pd.DataFrame, None]:
    ...
```

### Type Hint Guidelines
1. **All public functions must include type hints**
2. **Use `Any` sparingly** — prefer specific types
3. **Use `TypedDict` for structured dictionaries**
4. **Mark libraries as typed** with `py.typed` marker file

### Common Type Patterns
```python
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

# Path handling
def load_file(path: str | Path) -> pd.DataFrame:
    """Accept both string and Path objects."""
    ...

# Literal types for restricted values
def set_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    """Restrict to specific string values."""
    ...

# Callable types
def apply_transform(
    data: pd.DataFrame,
    func: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Accept functions as parameters."""
    ...

# TypedDict for structured data
from typing import TypedDict

class Metadata(TypedDict):
    timestamp: str
    version: str
    params: dict[str, Any]
```

---

## Documentation Standards

### Docstring Format
Use **NumPy-style docstrings** for all public functions and classes:

```python
def compute_metric(
    data: pd.Series,
    window: int = 20,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute a rolling metric over time series data.

    This function calculates a rolling metric and optionally normalizes
    the result to make it comparable across different data ranges.

    Parameters
    ----------
    data : pd.Series
        Input time series indexed by date.
    window : int, default 20
        Rolling window size in periods.
    normalize : bool, default True
        Whether to apply normalization.

    Returns
    -------
    pd.Series
        Computed metric with same index as input.
        Returns NaN for periods with insufficient data.

    Raises
    ------
    ValueError
        If window < 2 or data is empty.

    Examples
    --------
    >>> data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2024-01-01', periods=5))
    >>> result = compute_metric(data, window=3)
    >>> print(result)

    Notes
    -----
    Requires at least `window` non-null observations to produce output.

    See Also
    --------
    compute_other_metric : Related metric calculation
    """
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    logger.debug("Computing metric: window=%d, normalize=%s", window, normalize)
    
    # Implementation...
    ...
```

### Documentation Requirements
| Component | Required Documentation |
|-----------|------------------------|
| **Public functions** | Full NumPy docstring with Parameters, Returns, Examples |
| **Private functions** | Brief docstring describing purpose |
| **Classes** | Class-level docstring + method docstrings |
| **Modules** | Module-level docstring at top of file |
| **Complex logic** | Inline comments explaining *why*, not *what* |

### Module-Level Docstrings
```python
"""
Module for data processing and transformation.

This module contains utilities for loading, cleaning, and transforming
time series data for analysis.

Key Components
--------------
- load_data: Load data from various sources
- clean_data: Apply cleaning and validation
- transform_data: Apply transformations

Examples
--------
>>> from myproject.data import load_data, clean_data
>>> df = load_data('input.csv')
>>> cleaned = clean_data(df)
"""
```

---

## Logging Standards

### Logger Initialization
**Always use module-level loggers:**

```python
import logging

logger = logging.getLogger(__name__)
```

### Logging Levels

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Implementation details, low-level operations | Iteration counts, cache hits, internal state |
| **INFO** | User-facing operations, high-level events | File loaded, processing started, task completed |
| **WARNING** | Recoverable issues, unexpected conditions | Missing optional data, using fallback values |
| **ERROR** | Operation failures requiring attention | File not found, invalid data, processing failed |

### Logging Best Practices

✅ **CORRECT:**
```python
# Use %-formatting for lazy evaluation
logger.info("Loaded %d rows from %s", len(df), filepath)
logger.debug("Applied filter: column=%s, threshold=%.2f", col_name, threshold)

# Include context in messages
logger.warning("Missing optional column '%s', using default value %s", col, default)

# Log at appropriate levels
logger.info("Starting processing: params=%s", params)  # User operation
logger.debug("Iteration %d: result=%.2f", i, result)  # Implementation detail
```

❌ **AVOID:**
```python
# Don't use f-strings (eager evaluation, no structured logging)
logger.info(f"Loaded {len(df)} rows from {filepath}")

# Don't call basicConfig in library code
logging.basicConfig(level=logging.INFO)  # Caller's responsibility

# Don't log sensitive information
logger.info(f"API key: {api_key}")  # Security risk

# Don't use print() statements in libraries
print("Processing data...")  # Use logger.info() instead
```

### Logging in Tests
```python
# pytest automatically captures logs
# Run tests with logging output:
pytest -v --log-cli-level=INFO

# In test code, logging works normally:
def test_data_loading(tmp_path):
    logger.info("Testing data load with path: %s", tmp_path)
    # Test implementation...
```

---

## Classes vs Functions

### Prefer Functions Over Classes

**Default to functions.** Only introduce classes when they provide clear value.

✅ **Use functions for:**
```python
# Simple transformations
def compute_zscore(data: pd.Series, window: int = 20) -> pd.Series:
    """Pure function - no state needed."""
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    return (data - rolling_mean) / rolling_std

# Data processing pipelines
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless transformation."""
    return df.dropna().sort_index()

# Computations
def calculate_metric(x: pd.Series, y: pd.Series, lookback: int) -> pd.Series:
    """Pure computation - easier to test and reason about."""
    x_norm = compute_zscore(x, lookback)
    y_norm = compute_zscore(y, lookback)
    return x_norm - y_norm
```

❌ **Avoid classes when functions suffice:**
```python
# BAD: Unnecessary class wrapping a single function
class Calculator:
    def __init__(self, window: int):
        self.window = window
    
    def calculate(self, data: pd.Series) -> pd.Series:
        return compute_zscore(data, self.window)

# GOOD: Simple function
def calculate(data: pd.Series, window: int) -> pd.Series:
    return compute_zscore(data, window)
```

### When to Use Classes

**Only use classes when you need:**

1. **State management** (caches, connection pools, registries)
2. **Multiple related methods** operating on shared state
3. **Lifecycle management** (setup/teardown, context managers)
4. **Plugin/interface patterns** (base classes for extensions)

---

## Classes vs Dataclasses

### When to Use Each

✅ **Use `@dataclass` for data containers:**
```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    """Immutable configuration with validation."""
    window: int = 20
    threshold: float = 1.5
    
    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")

@dataclass
class Result:
    """Mutable container for results."""
    value: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)  # Mutable default
```

❌ **Use regular classes for behavior-heavy components:**
```python
class DataCache:
    """Complex state and many methods."""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, Any] = {}
    
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...
    def clear(self) -> None: ...
```

### Decision Guide

| Use Case | Choose | Why |
|----------|--------|-----|
| Config/parameters | `@dataclass(frozen=True)` | Immutable, type-safe |
| Results/outputs | `@dataclass` | Structured data |
| Manager/cache | Regular class | Complex behavior |
| Engine/orchestrator | Regular class | Primarily methods |

### Key Dataclass Features

```python
from dataclasses import dataclass, field, asdict
from typing import ClassVar

@dataclass
class Config:
    VERSION: ClassVar[str] = "1.0"  # Class variable
    name: str  # Required field
    max_value: float = 100.0  # Optional with default
    options: dict[str, float] = field(default_factory=dict)  # Mutable default
    _internal: dict = field(default_factory=dict, repr=False)  # Hidden from repr
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
```

---

## Testing Standards

### Test Organization
```
tests/
  unit/              # Unit tests for individual functions
  integration/       # Integration tests for workflows
  conftest.py        # Shared fixtures
```

### Test Requirements
1. **All public functions must have unit tests**
2. **Tests must be deterministic** (use fixed random seeds)
3. **Use fixtures for shared test data**
4. **Test edge cases and error conditions**
5. **Aim for >80% code coverage**

### Example Test Structure
```python
"""Tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
from myproject.processing import compute_metric


@pytest.fixture
def sample_data() -> pd.Series:
    """Create deterministic test data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # Deterministic
    values = np.cumsum(np.random.randn(100))
    return pd.Series(values, index=dates)


def test_compute_metric_basic(sample_data):
    """Test basic metric calculation."""
    result = compute_metric(sample_data, window=20)
    
    # Check shape and type
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    
    # Check for NaN in first window
    assert result.iloc[:19].isna().all()
    
    # Check values are finite after window
    assert result.iloc[20:].notna().all()


def test_compute_metric_invalid_window(sample_data):
    """Test error handling for invalid window."""
    with pytest.raises(ValueError, match="Window must be >= 2"):
        compute_metric(sample_data, window=1)


def test_compute_metric_empty_series():
    """Test handling of empty input."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="empty"):
        compute_metric(empty, window=20)
```

---

## Import Organization

### Import Order (Ruff enforces this)
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# 1. Standard library
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

# 2. Third-party
import pandas as pd
import numpy as np
from plotly import graph_objects as go

# 3. Local application
from myproject.config import DATA_DIR
from myproject.utils import load_data, save_data
from myproject.models import BaseModel
```

### Relative vs Absolute Imports

✅ **Use relative imports within package:**
```python
# In myproject/models/processor.py
from ..data.loader import load_data
from ..utils.validation import validate_input
from .base import BaseProcessor
```

✅ **Use absolute imports from outside package:**
```python
# In tests/ or scripts/
from myproject.models.processor import DataProcessor
from myproject.utils import save_data
```

---

## Error Handling

### Exception Best Practices

✅ **Be specific about exceptions:**
```python
def load_config(path: Path) -> dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Config file not found at %s", path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config: %s", e)
        raise ValueError(f"Corrupted config file: {e}") from e
    
    return config
```

### Validation Functions
```python
def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Validate that DataFrame contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : list[str]
        Required column names.
        
    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.debug("Validated required columns: %s", required)
```

---

## Reproducibility

### Random Seeds
**All stochastic operations must use fixed seeds:**

```python
import numpy as np
import random

# Set seeds for reproducibility
RANDOM_SEED = 42

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic data for testing."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    values = np.cumsum(np.random.randn(n_samples))
    
    return pd.DataFrame({'date': dates, 'value': values})
```

### Versioning and Metadata
**Important outputs should include metadata:**

```python
from datetime import datetime
import sys

def process_with_metadata(data: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Process data and return results with metadata."""
    logger.info("Starting processing: params=%s", params)
    
    # Process data...
    results = perform_processing(data, params)
    
    # Add metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'random_seed': RANDOM_SEED,
        'python_version': sys.version,
        'rows_processed': len(data),
    }
    
    logger.info("Processing complete: rows=%d", len(results))
    
    return {'results': results, 'metadata': metadata}
```

---

## Performance Guidelines

### Vectorized Operations

✅ **Use vectorized operations:**
```python
# GOOD: Vectorized with NumPy/Pandas
def compute_zscore(data: pd.Series, window: int) -> pd.Series:
    """Fast vectorized computation."""
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    return (data - rolling_mean) / rolling_std

def apply_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorized filtering."""
    return np.where(values > threshold, values, 0.0)
```

❌ **Avoid loops:**
```python
# BAD: Slow row-by-row iteration
def compute_zscore_slow(data: pd.Series, window: int) -> pd.Series:
    result = []
    for i in range(len(data)):
        window_data = data.iloc[max(0, i-window):i+1]
        zscore = (data.iloc[i] - window_data.mean()) / window_data.std()
        result.append(zscore)
    return pd.Series(result)
```

### Memory Management
```python
# Use appropriate dtypes
df['date'] = pd.to_datetime(df['date'])
df['value'] = df['value'].astype('float32')  # If precision allows
df['category'] = df['category'].astype('category')

# Load large files in chunks if needed
for chunk in pd.read_csv(large_file, chunksize=10_000):
    process_chunk(chunk)
```

---

## Git Commit Standards

Follow **Conventional Commits** format:

### Format
```
<type>: <description>

[optional body]
```

### Types
- `feat`: New feature or capability
- `fix`: Bug fix or correction
- `docs`: Documentation changes only
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `perf`: Performance improvement
- `chore`: Maintenance (dependencies, tooling, config)

### Examples

**Good:**
```
feat: Add data validation function

Implements validation for required columns and data types
to catch errors early in the pipeline.
```

```
refactor: Extract data loading to separate module

Improves modularity and testability by separating I/O concerns.
```

**Bad:**
```
Added new feature
```

```
Fix: bug in processing
```

---

## Prohibited Patterns

**1. Legacy Type Hints:**
```python
from typing import Optional, Union, List, Dict
def func(x: Optional[int]) -> Union[str, None]:  # ❌ Wrong
    ...
```

**2. Classes for Simple Transformations:**
```python
class Calculator:  # ❌ Wrong - use pure function
    def __init__(self, window: int):
        self.window = window
    def calculate(self, data: pd.Series) -> pd.Series:
        return data.rolling(self.window).mean()
```

**3. Mutable Config Objects:**
```python
@dataclass  # ❌ Wrong - missing frozen=True
class Config:
    param: int
```

**4. `logging.basicConfig()` in Library Code:**
```python
import logging
logging.basicConfig(level=logging.INFO)  # ❌ Wrong - caller's responsibility
```

**5. Loops Instead of Vectorized Operations:**
```python
# ❌ Wrong - slow loop-based computation
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

### Logging Quick Reference
- **INFO**: User-facing operations (`Loaded data`, `Processing complete`)
- **DEBUG**: Implementation details (`Cache hit`, `Iteration 5`)
- **WARNING**: Recoverable issues (`Missing optional field`)
- **ERROR**: Failures (`File not found`, `Validation failed`)

---

**Last Updated:** [Date]  
**Version:** [Version]  
**Maintainer:** [Maintainer]
