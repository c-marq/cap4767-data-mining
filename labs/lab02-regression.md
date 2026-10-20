# Lab 2 — Regression Pipeline on a Dataset of Your Choice
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points (+3 bonus)**

---

## Overview
In this lab you run the complete regression pipeline on a dataset of your choice: exploratory analysis, simple linear regression, multiple regression with dummy variables, and a written interpretation. The lab is completed individually. It rewards process over perfection: if your code fails but you explain what you tried, you earn most of the points. An optional logistic regression extension earns bonus points.

---

## Your Dataset
Choose **one** of the following. Run the setup cell, then uncomment your chosen dataset in the next cell.

| Option | Dataset | Target | How the notebook loads it |
|--------|---------|--------|---------------------------|
| 1 | California Housing | `MedHouseVal` | `fetch_california_housing(as_frame=True)` from scikit-learn |
| 2 | Auto MPG | `mpg` | `sns.load_dataset("mpg").dropna()` |
| 3 | Medical Insurance | `charges` | `https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv` |

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab02_regression.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Part | Tasks | Points |
|------|-------|--------|
| A: Exploratory Data Analysis | Task 1 Dataset Description (2) · Task 2 Correlation Heatmap (1.5) · Task 3 Scatterplot of Strongest Predictor (1.5) | 5 |
| B: Simple Linear Regression | Task 4 Simple Regression (2) · Task 5 RMSE Interpretation (2) | 4 |
| C: Multiple Regression | Task 6 Multiple Regression with Dummies (2) · Task 7 Residual Plot (1.5) · Task 8 Coefficient Table (1.5) | 5 |
| D: Written Interpretation | Task 9 Analysis, minimum 150 words (model comparison, feature insight, business application, limitations) | 6 |
| **Total** | | **20** |
| Bonus: Logistic Regression Extension | Create a binary target, build a `LogisticRegression`, confusion matrix and classification report, one-paragraph interpretation of precision and recall (no scaffolding) | +3 |

The notebook also ends with a required Reflection (minimum 3 sentences) and lists no separate points for it.

---

## What to Document
After every task that produces output, add a markdown cell explaining what you see. Document what you tried, what did not work, and what you adjusted — that process is worth points.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab02.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.
