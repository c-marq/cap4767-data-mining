# Modules 4–5 Group Exercise — Churn: Logistic Regression and Neural Network
**CAP4767 Data Mining with Python | Modules 4–5 | 10 Points**

---

## Overview
This exercise has two parts, each with its own notebook. In Part 1, your group performs statistical EDA on a churn dataset (Cramér's V, Mann-Whitney U, and Cohen's d), estimates the business cost of churn, builds a logistic regression baseline, and interprets the coefficients as business insights. In Part 2, you build and compare a logistic regression model and a Keras neural network on the same data, and present confusion matrices, ROC curves, and a model recommendation. Each part starts with a short group discussion, ends with a 2–3 minute presentation, and uses the same four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
**Telco Customer Churn**: customer records for a telecom company with a binary churn target (`Churn`), used in both parts. Each setup cell loads the data and cleans it (rows with a missing `TotalCharges` are dropped, and `customerID` is removed). Part 2's setup cell also runs the full preprocessing pipeline from the demo and creates the train/test split. Do not modify the setup cells.

Load URL: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Starter notebooks:
- Part 1: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m04-05_group_exercise_part1.ipynb`
- Part 2: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m04-05_group_exercise_part2.ipynb`

The starter notebooks have the full instructions for every task.

---

## Tasks and Points

### Part 1 — Churn EDA + Logistic Regression

| Task | What you do | Points |
|------|-------------|--------|
| 1: Cramér's V Analysis | Cramér's V between each categorical feature and `Churn` · sorted horizontal bar chart · interpretation | 1 |
| 2: Mann-Whitney U + Cohen's d | For each numerical feature, compare churners and non-churners · display as a table · interpretation | 1 |
| 3: Business Cost of Churn | Annual revenue at risk, replacement cost, and total impact | 1 |
| 4: Logistic Regression Baseline | Build a logistic regression on the pre-filled preprocessing · classification report | 1 |
| 5: Coefficient Interpretation + Confusion Matrix | Top 5 positive and top 5 negative coefficients · confusion matrix · dollar cost of the false negatives · explain the top 3 churn drivers to a non-technical executive | 1 |
| **Total** | | **5** |

### Part 2 — Churn: Logistic Regression vs Neural Network

| Task | What you do | Points |
|------|-------------|--------|
| 1: Build the Logistic Regression Model | Train on the scaled training data · store predictions and probabilities | 0.5 |
| 2: Classification Report + Interpretation | Classification report · what precision and recall mean for the "Churned" class | 0.5 |
| 3: Build the Keras Neural Network | Sequential model with two hidden layers, dropout, and a sigmoid output · compile and print the summary | 1 |
| 4: Train with Early Stopping | Train with early stopping on validation loss | 0.5 |
| 5: Plot Training Curves | Side-by-side training vs validation loss and accuracy · overfitting interpretation | 0.5 |
| 6: Evaluate the ANN | Predictions and probabilities · classification report | 0.5 |
| 7: ROC Curve Comparison | Both ROC curves on a single figure with AUC in the legend | 1 |
| 8: Model Recommendation | Written comparison of the two models and a recommendation | 0.5 |
| **Total** | | **5** |

Each part is worth **5 points**, so the whole exercise is **10 points**.

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise04.ipynb`

**What to submit:** Completed notebooks for both parts with all cells run and all markdown discussion cells filled in.
