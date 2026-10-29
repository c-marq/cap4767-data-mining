# Lab 3 — Churn Prediction: Full Pipeline
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points (+3 bonus)**

---

## Overview
In this lab you run a full churn prediction pipeline on one of two datasets: statistical EDA, a logistic regression baseline, a Keras neural network, a model comparison, and a written business recommendation. The lab is completed individually. It rewards process over perfection: if your ANN performs worse than logistic regression, that is a valid result, and your written analysis should explain why.

---

## Your Dataset
Choose **one** of the following. Option A is already switched on in the dataset cell; to use Option B, remove the `#` from each Option B line. Then run the cell; it handles all preprocessing and gives you clean train/test splits.

| Option | Dataset | Target | Domain |
|--------|---------|--------|--------|
| A | Bank Customer Churn (about 10,000 rows) | `Exited` | Banking |
| B | Credit Card Customer Attrition (about 10,000 rows) | `Attrition_Flag` | Credit Card Services |

Load URLs:
- Option A: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/Churn_Modelling.csv`
- Option B: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/BankChurners.csv`

**Do NOT use the Telco dataset from class.** You must use one of the two options above. Using the Telco dataset is a **-5 point deduction**.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab03_churn_neural_networks.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Part | Tasks | Points |
|------|-------|--------|
| A: Exploratory Data Analysis | Task 1 Data Inspection (1) · Task 2 Cramér's V Analysis (1) · Task 3 Mann-Whitney U + Cohen's d (1) · Task 4 Business Cost Estimate (1) | 4 |
| B: Logistic Regression | Task 5 Build and Evaluate (1.5) · Task 6 Coefficient Interpretation (1.5) | 3 |
| C: Neural Network | Task 7 Build and Train a Keras ANN (2) · Task 8 Training Curves (1) · Task 9 Evaluate the ANN (1) | 4 |
| D: Model Comparison | Task 10 ROC + Metrics Table (3) | 3 |
| E: Written Analysis | Task 11 Model Recommendation, minimum 300 words | 4 |
| F: Preprocessing | Pipeline runs correctly | 2 |
| **Total** | | **20** |
| Bonus: Third Model Variant | Train a third model that changes at least two of: number of layers, neurons per layer, dropout rate, optimizer · add it to your ROC plot and comparison table | +3 |

---

## What to Document
Document what did not work. If your ANN performs worse than logistic regression, explain why in your written analysis.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab03.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.
