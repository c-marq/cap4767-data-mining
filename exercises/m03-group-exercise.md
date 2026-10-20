# Module 3 Group Exercise — Multiple Regression: Predict and Interpret
**CAP4767 Data Mining with Python | Module 3 | 10 Points**

---

## Overview
Your group builds a multiple regression model on Washington State (WA) housing data, evaluates it, interprets the coefficients, and presents the findings to the class in a 2-minute share-out. Before coding, the group imagines it has been hired by a Washington State real estate agency to predict home sale prices and discusses which features will be the strongest predictors, which may be redundant, and what feature missing from the dataset would improve predictions. Groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
**WA Housing**: home sale records for Washington State, with `price` as the target. The setup cell loads it as `housing` and keeps only homes with `sqft_living` under 8,000 and `price` above 0 and under $1,000,000; do not modify it.

Load URL: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/housingData.csv`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m03_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Task | What you do | Points |
|------|-------------|--------|
| 1: Explore the Data | `.info()` and `.describe()` | 1.25 |
| 2: Correlation Analysis | Correlations with `price`, sorted bar chart, identify the top 3 features | 1.25 |
| 3: Simple Regression Baseline | Simple linear regression on your strongest predictor (`test_size=0.33`, `random_state=42`) · report R² | 1.25 |
| 4: Multiple Regression | 3–4 numeric features, same split · report R² and compare to Task 3 | 1.25 |
| 5: Add Dummy Variables | Create `has_basement` (1 if `sqft_basement` > 0, else 0) · add it to the model · report R² | 1.25 |
| 6: Calculate RMSE | RMSE on your best model's test predictions · interpret it in dollars | 1.25 |
| 7: Residual Plot | Residuals vs predicted values with a red zero line · written interpretation | 1.25 |
| 8: Coefficient Interpretation | Coefficients with feature names in a sorted DataFrame · which feature has the largest positive effect, and whether `has_basement` makes business sense | 1.25 |
| **Total** | | **10** |

The group discussion and share-out have no point value listed in the notebook.

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise03.ipynb`

**What to submit:** Completed notebook with all cells run and all markdown discussion cells filled in.
