# Google Colab Tips — CAP4767 Data Mining

---

## Essential Keyboard Shortcuts

| Action | Shortcut |
|--------|---------|
| Run current cell | Shift + Enter |
| Run cell, insert new below | Alt + Enter |
| Add cell above | Ctrl + M + A |
| Add cell below | Ctrl + M + B |
| Delete cell | Ctrl + M + D |
| Toggle comment | Ctrl + / |
| Interrupt execution | Ctrl + M + I |

---

## The Three Rules

**1. Save a copy to Drive immediately.**
Every time you open a notebook: File → Save a copy in Drive. Do this before you run a single cell.

**2. If your session disconnects — do not panic.**
Runtime → Reconnect, then re-run the setup cell at the top. Your Drive copy is safe.

**3. Name your files correctly before submitting.**
`LastName_FirstName_AssignmentName.ipynb`
Example: `Marquez_Carlos_Lab01_TimeSeries.ipynb`
Download via: File → Download → Download .ipynb

---

## If a Library Is Missing

```python
!pip install library-name
```
Then: Runtime → Restart runtime → Re-run the setup cell.

---

## Run All Cells from Top

Runtime → Run all — use this when reopening a saved notebook to restore all outputs.

---

## Common Errors and Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError` | Library not installed | `!pip install library-name`, restart runtime |
| `FileNotFoundError` | Wrong file path | Use the GitHub raw URL from the setup cell |
| `KeyError` | Column name typo | Run `df.columns` to check exact names |
| `ValueError: could not convert string to float` | Mixed types in column | Check `df.dtypes` and `df.isnull().sum()` |
| Session crashed | Memory exceeded | Runtime → Factory reset runtime |
