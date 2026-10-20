# Getting Started — CAP4767 Data Mining

Welcome to the course. This guide walks you through everything you need to get set up before the first class session. Setup takes about 10 minutes.

---

## Step 1 — Get a Google Account

All course work runs in **Google Colab**, which requires a Google account.

* If you already have a Gmail or Google account, you are ready
* If not, create a free account at [accounts.google.com](https://accounts.google.com)

> **Recommendation:** Use your MDC student email or a personal Gmail. Either works. Just use the same account every time.

---

## Step 2 — Test Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **New Notebook**
4. In the first cell, type the following and press **Shift + Enter** to run it:

```python
print("Data Mining is ready to go!")
```

If you see the output `Data Mining is ready to go!` — you are set up correctly.

---

## Step 3 — Open a Notebook from This Repo

Every notebook in this repository has an **Open in Colab** badge at the top that looks like this:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Click the badge and the notebook opens directly in Colab — no downloading, no installing, no setup required.

> **Important:** When you open a notebook from GitHub in Colab, you are working on a temporary copy. To save your work, go to **File → Save a copy in Drive**. This saves it to your own Google Drive.

---

## Step 4 — Save Your Work to Google Drive

Every time you open a notebook to do graded work:

1. Open the notebook via the Colab badge
2. Immediately go to **File → Save a copy in Drive**
3. Rename it using this format: `LastName_FirstName_AssignmentName.ipynb`
   - Example: `Marquez_Carlos_Lab01_TimeSeries.ipynb`
4. Work in your saved copy — not the original

This is how you avoid losing work. Colab does not automatically save to your Drive.

---

## Step 5 — Verify Your Library Setup

Run this cell in any Colab notebook to confirm all course libraries are available:

```python
# Run this cell to verify your environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

print("✅ pandas version:", pd.__version__)
print("✅ numpy version:", np.__version__)
print("✅ sklearn version:", __import__('sklearn').__version__)
print("✅ tensorflow version:", tf.__version__)
print("\n🎉 All core libraries loaded successfully!")
```

If any library fails to import, run this in a new cell:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow prophet mlxtend statsmodels pmdarima
```

Then restart the runtime: **Runtime → Restart runtime**, and run the verification cell again.

---

## Step 6 — Bookmark This Repository

Return to the main repo page and bookmark it:

**[https://github.com/c-marq/cap4767-data-mining](https://github.com/c-marq/cap4767-data-mining)**

You will come back here every week to access readings, demos, and lab starter notebooks.

---

## How Notebooks Are Organized

Each notebook in this course follows the same structure:

| Section | Description |
|---------|-------------|
| **Colab Badge** | Click to open in Colab |
| **Overview** | What this notebook covers and what you will build |
| **Setup Cell** | Loads data and libraries — run this first, do not modify |
| **Content Cells** | Annotated code and explanation — follow along |
| **YOUR CODE HERE cells** | Sections you complete independently |
| **Checkpoint cells** | Confirm your output looks correct before continuing |
| **Reflection cell** | Write your response here — counts toward your grade |

> **Color-coded instruction boxes** appear throughout notebooks and readings:
> - 🟢 **DO THIS** — Action steps
> - 💡 **WHY ARE WE DOING THIS?** — Context and rationale
> - ⚠️ **COMMON MISTAKE** — Warnings about frequent errors
> - 🛑 **STOP AND CHECK** — Confirm your output before continuing

---

## Submitting Your Work

All submissions go through **Canvas** — never through GitHub.

**For group exercises:**
- Your group works together in one shared notebook during class
- Each student submits their own copy individually via Canvas
- Name your file: `LastName_FirstName_GroupEx_M##.ipynb`

**For individual labs:**
- Complete independently after class
- Submit via Canvas by the date shown there
- Name your file: `LastName_FirstName_Lab##_Topic.ipynb`

**For midterm and final:**
- Submit three items via Canvas: notebook (.ipynb), business memo (.pdf or .docx), and presentation slides (.pdf or .pptx)
- Be prepared to present in class on presentation day

---

## Frequently Asked Questions

**Q: Do I need to install anything on my computer?**
No. Everything runs in Google Colab in your browser. A laptop or Chromebook with internet access is all you need.

**Q: Can I use my phone or tablet?**
Colab works on mobile but is not practical for coding. Use a laptop or desktop for all lab work.

**Q: What if my Colab session times out?**
Colab disconnects after periods of inactivity. Your code is not lost if you saved a copy to Drive. Reconnect and rerun the setup cell.

**Q: Can I work ahead?**
Yes — all materials are posted as they become available. Reading ahead is encouraged. Do not submit assignments before they are open in Canvas.

**Q: What if the dataset URL in the setup cell does not work?**
Check Canvas for an announcement. Dataset links are occasionally updated. Never modify the setup cell on your own — post in Canvas and the professor will push a fix.

---

*You are ready. See you in class.*
