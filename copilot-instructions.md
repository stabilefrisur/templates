# 🧠 Copilot Instructions — Systematic Macro Credit Project (VS Code Agent Mode)

## 🎯 Purpose

These instructions optimize **Copilot Chat and inline completions** in **VS Code Agent mode** (Claude Sonnet 4.5 / GPT‑5) for the *Systematic Macro Credit* project.  
The goal is to ensure the AI assistant generates **clean, modular, and reproducible** Python code aligned with project architecture and investment research standards.

---

## 🧩 Project Overview

You are working within a **systematic fixed‑income research environment**.  
The framework supports:
- Development and testing of **pilot investment strategies** (e.g., CDX overlay).
- Modular architecture for **data**, **model**, **backtest**, **visualization**, and **persistence** layers.
- Reproducible, documented, and version‑controlled research.

### Repository Layout

```
fia_project/
  data/               # Loaders, cleaning, transformation
  models/             # Signal & strategy logic
  backtest/           # Backtesting engine and risk tracking
  visualization/      # Plotly & Streamlit dashboards
  persistence/        # Parquet & JSON I/O utilities
  config/             # Paths, constants, environment
  tests/              # Unit tests for reproducibility
  pyproject.toml
  README.md
```

---

## ⚙️ Environment Standards

- **Python:** 3.13  
- **Environment:** managed with [`uv`](https://docs.astral.sh/uv/)  
- **Linting / Formatting:** `ruff`, `black`, `mypy`
- **Testing:** `pytest`
- **Docs:** `mkdocs` or `sphinx`
- **Visualization:** `plotly`, `streamlit`

All dependencies live in `pyproject.toml`.  
Use relative imports and avoid global state.

---

## 💡 Agent Behavior Guidelines

### General
1. **Always prioritize modular, PEP 8‑compliant, type‑annotated code.**
2. **Never mix strategy logic with infrastructure code.**
3. **Always document functions and classes using NumPy‑style docstrings.**
4. **Propose structured changes — not isolated code fragments.**
5. **Assume collaboration**: other developers must read and extend your code easily.

### Style
- Follow **PEP 8** and **type hints** strictly.
- Use **black** and **ruff** formatting conventions.
- Add **docstrings** and comments explaining rationale and data flow.

### Documentation Example
```python
def compute_spread_momentum(spread: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute short‑term momentum in CDX spreads using z‑score normalization.

    Parameters
    ----------
    spread : pd.Series
        Daily CDX spread levels.
    window : int, default 5
        Rolling lookback period in days.

    Returns
    -------
    pd.Series
        Normalized momentum signal.
    """
```

---

## 🧠 Agent Context Hints for Claude Sonnet / GPT‑5

| Context | Preferred Behavior |
|----------|--------------------|
| Editing `/data/` | Focus on reproducible data loading, cleaning, and transformation. Avoid strategy logic. |
| Editing `/models/` | Focus on signal functions and strategy modules. Use clean abstractions (`BaseModel`). |
| Editing `/backtest/` | Implement transparent, deterministic backtest logic. Include metadata logging. |
| Editing `/visualization/` | Generate reusable Plotly/Streamlit components. Separate plotting from computation. |
| Editing `/persistence/` | Handle Parquet/JSON I/O and registry management. No database dependencies. |
| Editing `/tests/` | Write unit tests for determinism, type safety, and reproducibility. |

When generating code, the assistant should **infer module context from file path** and **adhere to functional boundaries** automatically.

---

## ✅ Completion Optimization Rules

- Always suggest **imports relative to project structure**, not absolute paths.  
- Provide **runnable examples** with synthetic or minimal data.  
- Use **clear function and variable naming** (snake_case, descriptive).  
- When generating new modules, **include header comments** describing purpose and dependencies.  
- Include **type hints** and **unit test templates** by default.

### Example Ideal Output (inline completion)

```python
# models/cdx_overlay_model.py

from ..data.loader import load_market_data
from ..persistence.io import save_json
import pandas as pd

def compute_vix_cdx_gap(vix_df: pd.DataFrame, cdx_df: pd.DataFrame) -> pd.Series:
    """
    Compute the relative stress signal between equity vol (VIX) and CDX spreads.

    The signal identifies divergence between cross‑asset risk sentiment.
    Positive values indicate VIX outpacing credit widening.

    Parameters
    ----------
    vix_df : pd.DataFrame
        VIX index levels.
    cdx_df : pd.DataFrame
        CDX index spreads.

    Returns
    -------
    pd.Series
        Normalized VIX‑CDX gap signal.
    """
    gap = (vix_df['VIX'] - vix_df['VIX'].rolling(20).mean()) - (cdx_df['spread'] - cdx_df['spread'].rolling(20).mean())
    return gap / gap.rolling(20).std()
```

---

## 🚫 The Agent Should Never

- Use external databases or APIs (use Parquet/JSON only).  
- Hardcode file paths or credentials.  
- Generate non‑deterministic results without a fixed random seed.  
- Mix backtest logic with data ingestion.  
- Produce undocumented or untyped code.  
- Add notebook cells or magic commands inside modules.  

---

## 🧪 Testing & Logging Expectations

- All stochastic components must be **seeded deterministically**.  
- Every backtest or signal computation must log:  
  - Timestamp  
  - Parameters  
  - Version hash  
- Include lightweight tests for data I/O, signal correctness, and regression.  

Example:
```python
metadata = {
    "timestamp": datetime.now().isoformat(),
    "params": params_dict,
    "version": __version__,
}
save_json(metadata, "run_metadata.json")
```

---

## 🔗 Recommended Agent Prompts

When using Copilot Chat or VS Code inline completions, prefer prompts like:

- “Add a deterministic backtest class that tracks daily P&L and logs parameters.”  
- “Refactor this data loader to follow project persistence standards.”  
- “Write unit tests for this model to ensure reproducible signal outputs.”  
- “Add Streamlit components to visualize signal performance.”  

Avoid generic prompts like *“optimize this code”* — always specify layer and intent.

---

## 📘 Summary

Copilot should behave like a **quantitative developer assistant**, not a strategy designer.  
It should:
- Maintain modularity, transparency, and reproducibility.
- Focus on infrastructure excellence and analytical clarity.
- Produce code ready for production research pipelines.

> Maintained by **Systematic Macro Credit**.  
> Version 1.0 — Optimized for VS Code Agent Mode (Claude Sonnet 4.5 / GPT‑5)  
> October 2025
