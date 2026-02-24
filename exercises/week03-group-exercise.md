# Group Exercise 3 — Regression Pipeline
**CAP4767 Data Mining with Python | Week 3 | 10 Points**

---

## Objective
Build a complete regression pipeline from raw data to model evaluation. Your group will work through linear regression, then logistic regression on the same underlying dataset — and develop the judgment to know when each is the right tool.

---

## Your Dataset
**IBM Telco Customer Churn** — 7,032 customers with usage, contract, and demographic features. This week we use it for regression. Next week we will use it to predict churn.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week03-group-exercise.ipynb`

---

## Group Roles
Rotate from last week.

| Role | Responsibility |
|------|---------------|
| **Lead Coder** | Drives the notebook, shares screen, types the code |
| **Data Interpreter** | Narrates what each output means in plain English |
| **QA Reviewer** | Catches errors, checks outputs match checkpoints |
| **Presenter** | Owns the share-out at the end of class |

---

## Discussion First (5 minutes before opening Colab)
As a group, answer this before writing a single line of code:

> *"A colleague says: 'I used linear regression to predict whether a customer will churn — yes or no — and got 78% accuracy. That's pretty good, right?' What would you tell them?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell.

---

## Tasks

**Step 1 — Load and explore**
Run the setup cell. Check the shape, data types, and missing values. Identify which columns are numeric, which are categorical, and which is the target.

**Step 2 — Prepare features**
Encode categorical columns using one-hot encoding. Drop any columns that would cause data leakage. Split into train and test sets (80/20). Scale numeric features.

> 🛑 **Checkpoint:** After encoding, print `X_train.shape`. The number of columns should be larger than the original dataset. If it is the same, encoding did not run.

**Step 3 — Build a linear regression model**
Use `MonthlyCharges` as your target. Fit a LinearRegression model. Calculate RMSE and R² on the test set.

**Step 4 — Interpret the coefficients**
Print the top 5 features by absolute coefficient value. In a markdown cell: what does the coefficient of the largest feature mean in plain English?

> 🛑 **Checkpoint:** R² should be between 0 and 1. A negative R² means your model is worse than predicting the mean — check your train/test split and scaling.

**Step 5 — Build a logistic regression model**
Switch the target to `Churn` (binary). Fit a LogisticRegression model. Print the classification report.

**Step 6 — Compare precision and recall**
In a markdown cell, answer as a group:
- Which class has higher recall?
- For a telecom company trying to retain customers, is precision or recall more important for the churn class? Why?

**Step 7 — Revisit your discussion answer**
Was your colleague right to use linear regression for a yes/no prediction? Use what you found in Step 5 to support your answer.

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. Your linear regression R² — is the model useful?
2. Your classification report — which metric matters most for churn and why
3. Your revised answer to the discussion question

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise03.ipynb`

**What to submit:** Completed notebook with all cells run and all markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| Both models built and evaluated | Linear and logistic complete with correct metrics | One model complete, other has minor issues | Only one model or no evaluation |
| Interpretation cells | Specific, evidence-based, uses metric names correctly | Present and relevant, somewhat general | Missing or does not reference the data |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
