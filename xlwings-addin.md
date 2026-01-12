# xlwings Shared Drive Setup Manual

## Overview

This guide walks you through setting up an xlwings custom add-in with UDFs (User Defined Functions) that runs entirely from a shared network drive. Once configured, colleagues only need to enable the add-in in Excel—no Python installation or additional configuration required on their end.

## Architecture

```
\\shared-drive\xlwings-project\
├── addin\
│   └── myproject.xlam        ← Custom add-in (contains config + VBA)
├── src\
│   └── udfs.py               ← Python UDF code
└── venv\
    └── Scripts\
        └── python.exe        ← Virtual environment
```

-----

## Prerequisites

**On your development machine:**

- Python 3.8+ installed locally
- Microsoft Excel (Windows only for UDFs)
- xlwings installed: `pip install xlwings`
- Access to the shared drive with write permissions

**On colleague machines:**

- Microsoft Excel (Windows)
- Read access to the shared drive
- Trust Center settings configured (covered in deployment section)

-----

## Part 1: Create the Shared Virtual Environment

### 1.1 Create the folder structure

```cmd
mkdir "\\server\share\xlwings-project"
mkdir "\\server\share\xlwings-project\addin"
mkdir "\\server\share\xlwings-project\src"
```

### 1.2 Create the virtual environment on the shared drive

```cmd
python -m venv "\\server\share\xlwings-project\venv"
```

### 1.3 Activate and install xlwings

```cmd
"\\server\share\xlwings-project\venv\Scripts\activate"
pip install xlwings
```

> **Note:** The xlwings Python package version must match the add-in version. Document the version in a requirements.txt for reference.

-----

## Part 2: Create Your Python UDFs

### 2.1 Create the Python module

Create `\\server\share\xlwings-project\src\udfs.py`:

```python
import xlwings as xw

@xw.func
def hello(name):
    """Returns a greeting for the given name."""
    return f"Hello, {name}!"

@xw.func
def add_numbers(x, y):
    """Returns the sum of two numbers."""
    return x + y

@xw.func
@xw.arg('data', expand='table')
def sum_table(data):
    """Sums all values in a range."""
    total = 0
    for row in data:
        for cell in row:
            if isinstance(cell, (int, float)):
                total += cell
    return total
```

### 2.2 UDF decorator reference

|Decorator                        |Purpose                          |
|---------------------------------|---------------------------------|
|`@xw.func`                       |Marks function as Excel UDF      |
|`@xw.arg('x', doc='Description')`|Adds argument documentation      |
|`@xw.arg('x', expand='table')`   |Expands range to full data region|
|`@xw.ret(expand='table')`        |Returns array formula result     |

-----

## Part 3: Create the Custom Add-in

### 3.1 Generate the add-in template

On your development machine with xlwings installed, run:

```cmd
xlwings quickstart myproject --addin
```

This creates a folder with `myproject.xlam` and `myproject.py`. You only need the .xlam file.

### 3.2 Configure the add-in

1. Open the generated `myproject.xlam` in Excel
1. Press `Alt+F11` to open the VBA editor
1. In the Project Explorer, select **ThisWorkbook**
1. In the Properties window (press F4 if not visible), change `IsAddin` from `True` to `False`
1. This reveals the hidden configuration sheet `_myaddin.conf`
1. Rename the sheet from `_myaddin.conf` to `myaddin.conf` (remove the underscore) to activate it
1. Edit the configuration values as shown below
1. Change `IsAddin` back to `True`
1. Save the add-in

### 3.3 Configuration settings

In the `myaddin.conf` sheet, set these values in two columns (A = setting name, B = value):

|Setting (Column A)|Value (Column B)                                        |
|------------------|--------------------------------------------------------|
|`INTERPRETER_WIN` |`\\server\share\xlwings-project\venv\Scripts\python.exe`|
|`PYTHONPATH`      |`\\server\share\xlwings-project\src`                    |
|`UDF MODULES`     |`udfs`                                                  |
|`USE UDF SERVER`  |`True`                                                  |
|`SHOW CONSOLE`    |`False`                                                 |

- **INTERPRETER_WIN**: Full UNC path to the Python executable in your shared venv
- **PYTHONPATH**: Folder containing your .py files (not the file itself)
- **UDF MODULES**: Module name without the .py extension
- **USE UDF SERVER**: Keeps Python running between calls for better performance
- **SHOW CONSOLE**: Set to `True` for debugging, `False` for production

### 3.4 Import the UDFs into the add-in

With the add-in open and configured:

1. Press `Alt+F11` to open the VBA editor
1. In the xlwings module, scroll to find the `ImportPythonUDFsToAddin` Sub (near the end)
1. Click inside the Sub and press `F5` to run it

This generates VBA wrapper functions for your Python UDFs. **You only need to repeat this step when you:**

- Add or remove functions
- Change function names
- Change function arguments or decorators

You do **not** need to re-import when you change the function implementation—those changes are picked up automatically.

### 3.5 Move the add-in to the shared drive

Save and close the add-in, then copy it to:

```
\\server\share\xlwings-project\addin\myproject.xlam
```

-----

## Part 4: Deployment to Colleagues

### 4.1 One-time Excel Trust Center setup (each user)

Each user needs to configure these settings once:

1. **Enable Trust Access to VBA Project Object Model** (only needed if they will import UDFs themselves):
- File → Options → Trust Center → Trust Center Settings
- Macro Settings → Check “Trust access to the VBA project object model”
1. **Add the shared drive as a Trusted Location** (recommended):
- File → Options → Trust Center → Trust Center Settings
- Trusted Locations → Add new location
- Add `\\server\share\xlwings-project\`

### 4.2 Install the add-in (each user)

**Option A: Manual installation**

1. Open Excel
1. File → Options → Add-ins
1. At the bottom, select “Excel Add-ins” and click “Go…”
1. Click “Browse…”
1. Navigate to `\\server\share\xlwings-project\addin\myproject.xlam`
1. Click OK to add it to the list
1. Ensure the checkbox next to the add-in is checked
1. Click OK

**Option B: Command line installation**

```cmd
xlwings addin install --file "\\server\share\xlwings-project\addin\myproject.xlam"
```

> **Note:** This copies the add-in to the user’s XLSTART folder. If you want all users to reference the same file on the shared drive, use Option A instead.

### 4.3 Verify installation

In any Excel workbook, test a UDF:

```
=hello("World")
```

Should return: `Hello, World!`

-----

## Part 5: Maintenance and Updates

### Updating Python code

Simply edit the .py file on the shared drive. Changes to function implementations are picked up on the next calculation (Ctrl+Alt+F9 forces recalculation).

### Adding new functions

1. Add the new function to your .py file
1. Open the .xlam file
1. Run `ImportPythonUDFsToAddin` in the VBA editor
1. Save and close the add-in

Users will have access to the new function immediately (they may need to restart Excel or click “Restart UDF Server” in the xlwings ribbon if using the standard add-in).

### Updating packages in the venv

```cmd
"\\server\share\xlwings-project\venv\Scripts\activate"
pip install --upgrade xlwings pandas numpy  # etc.
```

-----

## Troubleshooting

### “Could not activate Python COM server”

- Verify the `INTERPRETER_WIN` path is correct and accessible
- Ensure xlwings is installed in the venv
- Check that the user has read access to the shared drive

### UDFs return #VALUE! or don’t appear

- Verify `PYTHONPATH` points to the folder containing your .py file
- Verify `UDF MODULES` matches your module name (without .py)
- Set `SHOW CONSOLE` to `True` to see Python error messages
- Re-run `ImportPythonUDFsToAddin` if you changed function signatures

### Slow first call

This is normal. The first UDF call starts the Python interpreter. With `USE UDF SERVER = True`, subsequent calls are faster. If performance is still an issue, consider having users click “Restart UDF Server” after opening Excel to pre-warm the interpreter.

### Network drive performance

Running Python from a network drive is slower than local. For complex UDFs, consider:

- Using `USE UDF SERVER = True` (keeps Python running)
- Batching operations into array formulas rather than many single-cell calls
- Caching expensive computations within your Python code

-----

## Configuration Hierarchy Reference

xlwings checks configuration in this order (first found wins):

1. Sheet named `xlwings.conf` in the active workbook
1. File named `xlwings.conf` in the workbook’s directory
1. Sheet named `myaddin.conf` in the add-in (custom add-ins only)
1. User’s global config: `%USERPROFILE%\.xlwings\xlwings.conf`
1. xlwings ribbon settings

For your shared drive setup, the add-in’s `myaddin.conf` sheet (option 3) is the recommended approach since it travels with the add-in and requires no per-user configuration.