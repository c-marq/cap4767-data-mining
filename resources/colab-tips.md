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

## Using Markdown in Text Cells

Text cells in Colab use Markdown — a simple formatting language that makes your notebooks readable and professional. Double-click any text cell to edit it. Press Shift + Enter to render it.

### Headings
```markdown
# Notebook Title (H1 — use once, at the top)
## Section Title (H2 — major sections)
### Subsection Title (H3 — steps within a section)
```

### Text Formatting
```markdown
**bold text**
*italic text*
`inline code`
> This is a callout block — good for notes and warnings
```

### Lists
```markdown
- Bullet item
- Another item
  - Indented sub-item

1. Numbered step
2. Next step
3. Final step
```

### Links and Dividers
```markdown
[Link text](https://url.com)

---
```

---

## Building a Table of Contents

A ToC at the top of your notebook makes it easy to navigate — especially for long lab submissions. Colab supports anchor links using heading IDs.

**Step 1 — Add a ToC text cell at the very top of your notebook:**
```markdown
## Table of Contents
1. [Setup & Data Loading](#setup)
2. [Exploratory Data Analysis](#eda)
3. [Model Building](#model)
4. [Results & Evaluation](#results)
5. [Reflection](#reflection)
```

**Step 2 — Add a matching anchor tag to each section heading:**
```markdown
## 1. Setup & Data Loading <a id="setup"></a>
## 2. Exploratory Data Analysis <a id="eda"></a>
## 3. Model Building <a id="model"></a>
## 4. Results & Evaluation <a id="results"></a>
## 5. Reflection <a id="reflection"></a>
```

The `<a id="section-name"></a>` tag creates the anchor the ToC link jumps to. Clicking a ToC entry scrolls directly to that section.

> **Pro tip:** Use a consistent ToC structure across all your lab submissions. It signals professionalism and makes your work significantly easier for the instructor to review and grade.

---

## Common Errors and Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError` | Library not installed | `!pip install library-name`, restart runtime |
| `FileNotFoundError` | Wrong file path | Use the GitHub raw URL from the setup cell |
| `KeyError` | Column name typo | Run `df.columns` to check exact names |
| `ValueError: could not convert string to float` | Mixed types in column | Check `df.dtypes` and `df.isnull().sum()` |
| Session crashed | Memory exceeded | Runtime → Factory reset runtime |
