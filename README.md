# CAP4767: Data Mining with Python

**Miami Dade College | School of Engineering and Technology**

---

## 👋 Welcome

This repository contains all course materials for CAP4767 Data Mining. Everything you need — readings, demo notebooks, exercises, labs, and project files — lives here.

**Your Canvas course remains the official source for:**

* Due dates and deadlines
* Assignment submissions
* Grades and feedback
* Announcements

Think of this GitHub repo as your **textbook and workbench**. Think of Canvas as your **gradebook and calendar**.

---

## 🚀 Getting Started

### First Time Here?

1. **Bookmark this page** — You'll return here weekly
2. **Set up Google Colab** — See [Getting Started Guide](GETTING-STARTED.md)
3. **Test your setup** — Open any notebook from the `demos/` folder in Colab

### New to GitHub?

You don't need to know Git commands for this course. Simply:

* Click on folders to navigate
* Click on `.md` files to read them
* Click on `.ipynb` files, then click "Open in Colab" badge to work with notebooks

---

## 📁 Repository Structure

| Folder | What's Inside | When to Use It |
|--------|--------------|----------------|
| [`readings/`](readings/) | Chapter readings aligned to learning objectives | Before class — complete assigned reading |
| [`demos/`](demos/) | Notebooks from in-class demonstrations | During class — follow along with the professor |
| [`exercises/`](exercises/) | Group exercise starter notebooks | In class — collaborative breakout practice |
| [`labs/`](labs/) | Individual lab starter notebooks | After class — independent assignments |
| [`case-studies/`](case-studies/) | Midterm and final project materials | Major assessments |
| [`data/`](data/) | Datasets used across multiple chapters | Reference as needed |
| [`resources/`](resources/) | Slides, cheat sheets, helpful links | Reference as needed |

---

## 📅 Course Flow

Each week follows this pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│  BEFORE CLASS                                                   │
│  ✓ Complete the chapter reading                                 │
│  ✓ Review the demo notebook                                     │
│  ✓ Watch the video companion (when available)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SESSION 1                                                      │
│  → Presentation: concepts, analogies, real-world context        │
│  → Live demo: professor codes, you follow along                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SESSION 2                                                      │
│  → Group exercise: collaborative breakout with your team        │
│  → Share-out: one group member presents findings to class       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  AFTER CLASS                                                    │
│  ✓ Complete individual lab assignment (weeks assigned)          │
│  ✓ Submit in Canvas                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Chapter Overview

| Week | Chapter | Topic | Key Techniques |
|------|---------|-------|---------------|
| 1 | 1 | Time Series Foundations | Rolling windows, resampling, seasonal decomposition |
| 2 | 2 | Time Series Forecasting | SARIMAX, Prophet, RMSE, R² |
| 3 | 3 | Regression | Linear, multiple, logistic regression |
| 4 | 4 | Customer Churn: EDA + Logistic Regression | Cramér's V, Cohen's d, classification report |
| 4 | 5 | Neural Networks | Keras ANN, confusion matrix, ROC curve |
| 5 | 6 | RFM + CLTV + K-Means | Customer segmentation, Elbow method, cluster profiles |
| 6 | 7 | Market Basket Analysis | Apriori, support, confidence, lift, mlxtend |
| 7 | 8 | RFM Across Domains | Universal framework, sequence analysis, anomaly detection |
| — | — | **Midterm** — Assigned Week 4 \| Due Week 5 | RFM + CLTV + K-Means pipeline |
| — | — | **Final** — Assigned Week 6 \| Due Week 8 | Full data mining capstone |

---

## 🛠️ Tools We Use

| Tool | Purpose | Access |
|------|---------|--------|
| **Google Colab** | Write and run Python code | [colab.research.google.com](https://colab.research.google.com) |
| **GitHub** | Access course materials | You're here! |
| **Canvas** | Submissions, grades, communication | [MDC Canvas](https://mdc.instructure.com) |

No software installation required. Everything runs in your browser.

---

## 📦 Key Libraries

```python
# Data
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Time Series
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Neural Networks
import tensorflow as tf
from tensorflow import keras

# Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules
```

All libraries are pre-installed in Google Colab. No setup required.

---

## ❓ Getting Help

1. **Check the reading** — Most questions are answered in the chapter materials
2. **Review the demo notebook** — Annotated code examples with explanations
3. **Post in Canvas** — Classmates and professor can help
4. **Office Hours** — See Canvas for schedule

When asking for help with code:
* Describe what you're trying to do
* Share the error message (screenshot or copy/paste)
* Tell us what you've already tried

---

## 📋 Quick Links

* [Getting Started Guide](GETTING-STARTED.md)
* [Chapter Readings](readings/)
* [Demo Notebooks](demos/)
* [Group Exercises](exercises/)
* [Individual Labs](labs/)
* [Midterm + Final Materials](case-studies/)
* [Datasets](data/)
* [Python Quick Reference](resources/python-quick-reference.md)

---

## ⚠️ Important Notes

* **Do not fork this repository** — Simply access it directly
* **Submissions go to Canvas** — Never submit work via GitHub
* **Materials may be updated** — Refresh your browser to see the latest versions
* **Group exercise notebooks** — Work together in class, each student submits individually via Canvas

---

*Questions about this repository? Ask in Canvas or bring them to class.*
